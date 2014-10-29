# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2014, Imperial College London.
# Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import defaultdict
from copy import deepcopy as dcopy

from base import *
from loop_scheduler import SSALoopMerger
from utils import visit, is_perfect_loop, count_occurrences, ast_c_sum
from utils import ast_replace, loops_as_dict, od_find_next
import plan


class ExpressionRewriter(object):
    """Provide operations to re-write an expression:

    * Loop-invariant code motion: find and hoist sub-expressions which are
      invariant with respect to a loop
    * Expansion: transform an expression ``(a + b)*c`` into ``(a*c + b*c)``
    * Factorization: transform an expression ``a*b + a*c`` into ``a*(b+c)``"""

    def __init__(self, stmt_info, decls, kernel_info, hoisted, expr_graph):
        """Initialize the ExpressionRewriter.

        :arg stmt_info:   an AST node statement containing an expression and meta
                          information (MetaExpr) related to the expression itself.
                          including the iteration space it depends on.
        :arg decls:       list of AST declarations of the various symbols in ``syms``.
        :arg kernel_info: contains information about the AST nodes sorrounding the
                          expression.
        :arg hoisted:     dictionary that tracks hoisted expressions
        :arg expr_graph:  expression graph that tracks symbol dependencies
        """
        self.stmt, self.expr_info = stmt_info
        self.decls = decls
        self.header, self.kernel_decls = kernel_info
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        # Expression Manipulators used by the Expression Rewriter
        typ = self.kernel_decls[self.stmt.children[0].symbol][0].typ
        self.expr_hoister = ExpressionHoister(self.stmt,
                                              self.expr_info,
                                              self.header,
                                              typ,
                                              self.decls,
                                              self.hoisted,
                                              self.expr_graph)

        # Properties of the transformed expression
        self._expanded = False

    def licm(self, merge_and_simplify=False, compact_tmps=False):
        """Perform generalized loop-invariant code motion.

        :arg merge_and_simpliy: True if should try to merge the loops in which
                                invariant expressions are evaluated, because they
                                might be characterized by the same iteration space.
                                In this process, computation which is redundant
                                because performed in at least two merged loops, is
                                eliminated.
        :arg compact_tmps: True if temporaries accessed only once should be inlined.
        """
        self.expr_hoister.licm()

        # Try to merge the hoisted loops, because they might have the same
        # iteration space (call to merge()), and also possibly share some
        # redundant computation (call to simplify())
        if merge_and_simplify:
            lm = SSALoopMerger(self.header, self.expr_graph)
            merged_loops = lm.merge()
            for merged, merged_in in merged_loops:
                [self.hoisted.update_loop(l, merged_in) for l in merged]
            lm.simplify()

        # Remove temporaries created yet accessed only once
        if compact_tmps:
            stmt_occs = count_occurrences(self.stmt, key=1, read_only=True)
            for l in self.hoisted.all_loops:
                l_occs = count_occurrences(l, key=0, read_only=True)
                to_replace, to_delete = {}, []
                for sym_rank, sym_occs in l_occs.items():
                    # If the symbol appears once is a potential candidate for
                    # being removed. It is actually removed if it does't appear
                    # in the expression from which was extracted. Symbols appearing
                    # more than once are removed if they host an expression made
                    # of just one symbol
                    sym, rank = sym_rank
                    if sym not in self.hoisted or sym in stmt_occs:
                        continue
                    expr = self.hoisted[sym].expr
                    if sym_occs > 1 and not isinstance(expr.children[0], Symbol):
                        continue
                    if not self.hoisted[sym].loop:
                        continue

                    to_replace[str(Symbol(sym, rank))] = expr
                    to_delete.append(sym)

                for stmt in l.children[0].children:
                    symbol, expr = stmt.children
                    sym = symbol.symbol
                    ast_replace(expr, to_replace, copy=True)
                for sym in to_delete:
                    self.hoisted.delete_hoisted(sym)

    def expand(self):
        """Expand expressions such that: ::

            Y[j] = f(...)
            (X[i]*Y[j])*F + ...

        becomes: ::

            Y[j] = f(...)*F
            (X[i]*Y[j]) + ...

        This may be useful for several purposes:

        * Relieve register pressure; when, for example, ``(X[i]*Y[j])`` is
          computed in a loop L' different than the loop L'' in which ``Y[j]``
          is evaluated, and ``cost(L') > cost(L'')``
        * It is also a step towards exposing well-known linear algebra
          operations, like matrix-matrix multiplies."""

        # Select the iteration variable along which the expansion should be performed.
        # The heuristics here is that the expansion occurs along the iteration
        # variable which appears in more unique arrays. This will allow factorization
        # to be more effective.
        asm_out, asm_in = self.expr_info.unit_stride_itvars
        it_var_occs = {asm_out: 0, asm_in: 0}
        for s in count_occurrences(self.stmt.children[1]).keys():
            if s[1] and s[1][0] in it_var_occs:
                it_var_occs[s[1][0]] += 1

        exp_var = asm_out if it_var_occs[asm_out] < it_var_occs[asm_in] else asm_in
        ee = ExpressionExpander(self.hoisted, self.expr_graph)
        ee.expand(self.stmt.children[1], self.stmt, it_var_occs, exp_var)
        self.decls.update(ee.expanded_decls)
        self._expanded = True

    def distribute(self):
        """Factorize terms in the expression.
        E.g. ::

            A[i]*B[j] + A[i]*C[j]

        becomes ::

            A[i]*(B[j] + C[j])."""

        def find_prod(node, occs, to_distr):
            if isinstance(node, Par):
                find_prod(node.children[0], occs, to_distr)
            elif isinstance(node, Sum):
                find_prod(node.children[0], occs, to_distr)
                find_prod(node.children[1], occs, to_distr)
            elif isinstance(node, Prod):
                left, right = (node.children[0], node.children[1])
                l_str, r_str = (str(left), str(right))
                if occs[l_str] > 1 and occs[r_str] > 1:
                    if occs[l_str] > occs[r_str]:
                        dist = l_str
                        target = (left, right)
                        occs[r_str] -= 1
                    else:
                        dist = r_str
                        target = (right, left)
                        occs[l_str] -= 1
                elif occs[l_str] > 1 and occs[r_str] == 1:
                    dist = l_str
                    target = (left, right)
                elif occs[r_str] > 1 and occs[l_str] == 1:
                    dist = r_str
                    target = (right, left)
                elif occs[l_str] == 1 and occs[r_str] == 1:
                    dist = l_str
                    target = (left, right)
                else:
                    raise RuntimeError("Distribute error: symbol not found")
                to_distr[dist].append(target)

        # Expansion ensures the expression to be in a form like:
        # tensor[i][j] += A[i]*B[j] + C[i]*D[j] + A[i]*E[j] + ...
        if not self._expanded:
            raise RuntimeError("Distribute error: expansion required first.")

        to_distr = defaultdict(list)
        occurrences = count_occurrences(self.stmt.children[1], key=2)
        find_prod(self.stmt.children[1], occurrences, to_distr)

        # Create the new expression
        new_prods = []
        for d in to_distr.values():
            dist, target = zip(*d)
            target = Par(ast_c_sum(target)) if len(target) > 1 else ast_c_sum(target)
            new_prods.append(Par(Prod(dist[0], target)))
        self.stmt.children[1] = Par(ast_c_sum(new_prods))


class ExpressionHoister(object):
    """Perform loop-invariant code motion (licm).

    Invariant expressions found in the loop nest are moved "after" the
    outermost independent loop and "after" the fastest varying dimension
    loop. Here, "after" means that if the loop nest has two loops ``i``
    and ``j``, and ``j`` is in the body of ``i``, then ``i`` comes after
    ``j`` (i.e. the loop nest has to be read from right to left).

    For example, if a sub-expression ``E`` depends on ``[i, j]`` and the
    loop nest has three loops ``[i, j, k]``, then ``E`` is hoisted out from
    the body of ``k`` to the body of ``i``). All hoisted expressions are
    then wrapped within a suitable loop in order to exploit compiler
    autovectorization. Note that this applies to constant sub-expressions
    as well, in which case hoisting after the outermost loop takes place."""

    # Global counting the total number of expressions for which licm was licm
    # was performed
    GLOBAL_LICM_COUNTER = 0

    def __init__(self, stmt, expr_info, header, typ, decls, hoisted, expr_graph):
        """Initialize the ExpressionHoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.typ = typ
        self.decls = decls
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        # Count how many iterations of hoisting were performed. This is used to
        # create sensible variable names
        self.glb_counter = ExpressionHoister.GLOBAL_LICM_COUNTER
        self.counter = 0

        # Constants used by the extract method to charaterize sub-expressions
        self.INV = 0  # Invariant term, hoistable if part of an invariant expression
        self.KSE = 1  # Invariant expression, potentially part of larger invariant
        self.HOI = 2  # Variant expression, hoisted, can't hoist anymore

        # Variables used for communication between the extract and licm methods
        self.extracted = False  # True if managed to hoist at least one sub-expr
        self.symbols = {}
        self.real_deps = []

    def _extract_exprs(self, node, expr_dep, length=0):
        """Extract invariant sub-expressions from the original expression.
        Hoistable sub-expressions are stored in expr_dep."""

        def hoist(node, dep, expr_dep, _extract=True):
            if _extract:
                node = Par(node) if isinstance(node, Symbol) else node
                expr_dep[dep].append(node)
            self.extracted = self.extracted or _extract

        if isinstance(node, Symbol):
            return (self.symbols[node], self.INV, 1)
        if isinstance(node, Par):
            return (self._extract_exprs(node.children[0], expr_dep, length))

        # Traverse the expression tree
        left, right = node.children
        dep_l, info_l, len_l = self._extract_exprs(left, expr_dep, length)
        dep_r, info_r, len_r = self._extract_exprs(right, expr_dep, length)
        node_len = len_l + len_r

        # Filter out false dependencies
        dep_l = tuple(d for d in dep_l if d in self.real_deps)
        dep_r = tuple(d for d in dep_r if d in self.real_deps)

        if info_l == self.KSE and info_r == self.KSE:
            if dep_l != dep_r:
                # E.g. (A[i]*alpha + D[i])*(B[j]*beta + C[j])
                hoist(left, dep_l, expr_dep)
                hoist(right, dep_r, expr_dep)
                return ((), self.HOI, node_len)
            else:
                # E.g. (A[i]*alpha)+(B[i]*beta)
                return (dep_l, self.KSE, node_len)
        elif info_l == self.KSE and info_r == self.INV:
            # E.g. (A[i] + B[i])*C[j]
            hoist(left, dep_l, expr_dep)
            hoist(right, dep_r, expr_dep, not self.counter or len_r > 1)
            return ((), self.HOI, node_len)
        elif info_l == self.INV and info_r == self.KSE:
            # E.g. A[i]*(B[j]) + C[j])
            hoist(right, dep_r, expr_dep)
            hoist(left, dep_l, expr_dep, not self.counter or len_l > 1)
            return ((), self.HOI, node_len)
        elif info_l == self.INV and info_r == self.INV:
            if not dep_l and not dep_r:
                # E.g. alpha*beta
                return ((), self.INV, node_len)
            elif dep_l and dep_r and dep_l != dep_r:
                if set(dep_l).issubset(set(dep_r)):
                    # E.g. A[i]*B[i,j]
                    return (dep_r, self.KSE, node_len)
                elif set(dep_r).issubset(set(dep_l)):
                    # E.g. A[i,j]*B[i]
                    return (dep_l, self.KSE, node_len)
                else:
                    # dep_l != dep_r:
                    # E.g. A[i]*B[j]
                    hoist(left, dep_l, expr_dep, not self.counter or len_l > 1)
                    hoist(right, dep_r, expr_dep, not self.counter or len_r > 1)
                    return ((), self.HOI, node_len)
            elif dep_l and dep_r and dep_l == dep_r:
                # E.g. A[i] + B[i]
                return (dep_l, self.INV, node_len)
            elif dep_l and not dep_r:
                # E.g. A[i]*alpha
                hoist(right, dep_r, expr_dep, len_r > 1)
                return (dep_l, self.KSE, node_len)
            elif dep_r and not dep_l:
                # E.g. alpha*A[i]
                hoist(left, dep_l, expr_dep, len_l > 1)
                return (dep_r, self.KSE, node_len)
            else:
                raise RuntimeError("Error while hoisting invariant terms")
        elif info_l == self.HOI and info_r == self.KSE:
            hoist(right, dep_r, expr_dep, len_r > 2)
            return ((), self.HOI, node_len)
        elif info_l == self.KSE and info_r == self.HOI:
            hoist(left, dep_l, expr_dep, len_l > 2)
            return ((), self.HOI, node_len)
        elif info_l == self.HOI or info_r == self.HOI:
            return ((), self.HOI, node_len)
        else:
            raise RuntimeError("Fatal error while finding hoistable terms")

    def licm(self):
        """Perform loop-invariant code motion for the expression passed in at
        object construction time."""

        expr_loops = self.expr_info.loops
        dict_expr_loops = loops_as_dict(expr_loops)
        real_deps = dict_expr_loops.keys()

        # (Re)set global parameters of the extract recursive function
        self.symbols = visit(self.header, None)['symbols']
        self.real_deps = real_deps
        self.extracted = False

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        inv_dep = {}
        while True:
            expr_dep = defaultdict(list)
            self._extract_exprs(self.stmt.children[1], expr_dep)

            # While end condition
            if self.counter and not self.extracted:
                break
            self.extracted = False
            self.counter += 1

            for all_deps, expr in expr_dep.items():
                # -1) Filter dependencies that do not pertain to the expression
                dep = tuple(d for d in all_deps if d in real_deps)

                # 0) The invariant statements go in the closest outer loop to
                # dep[-1] which they depend on, and are wrapped by a loop wl
                # iterating along the same iteration space as dep[-1].
                # If there's no such an outer loop, they fall in the header,
                # provided they are within a perfect loop nest (otherwise,
                # dependencies may be broken)
                outermost_loop = expr_loops[0]
                is_outermost_perfect = is_perfect_loop(outermost_loop)
                if len(dep) == 0:
                    place, wl = self.header, None
                    next_loop = outermost_loop
                elif len(dep) == 1 and is_outermost_perfect:
                    place, wl = self.header, dict_expr_loops[dep[0]]
                    next_loop = outermost_loop
                elif len(dep) == 1 and not is_outermost_perfect:
                    place, wl = dict_expr_loops[dep[0]].children[0], None
                    next_loop = od_find_next(dict_expr_loops, dep[0])
                else:
                    dep_block = dict_expr_loops[dep[-2]].children[0]
                    place, wl = dep_block, dict_expr_loops[dep[-1]]
                    next_loop = od_find_next(dict_expr_loops, dep[-2])

                # 1) Remove identical sub-expressions
                expr = dict([(str(e), e) for e in expr]).values()

                # 2) Create the new invariant sub-expressions and temporaries
                sym_rank, for_dep = (tuple([wl.size()]), tuple([wl.it_var()])) \
                    if wl else ((), ())
                syms = [Symbol("LI_%s_%d_%s" % ("".join(dep).upper() if dep else "C",
                        self.counter, i), sym_rank) for i in range(len(expr))]
                var_decl = [Decl(self.typ, _s) for _s in syms]
                for_sym = [Symbol(_s.sym.symbol, for_dep) for _s in var_decl]

                # 3) Create the new for loop containing invariant terms
                _expr = [Par(dcopy(e)) if not isinstance(e, Par)
                         else dcopy(e) for e in expr]
                inv_for = [Assign(_s, e) for _s, e in zip(dcopy(for_sym), _expr)]

                # 4) Update the lists of decls
                self.decls.update(dict(zip([d.sym.symbol for d in var_decl],
                                           [(v, plan.LOCAL_VAR) for v in var_decl])))

                # 5) Replace invariant sub-trees with the proper tmp variable
                n_replaced = dict(zip([str(s) for s in for_sym], [0]*len(for_sym)))
                ast_replace(self.stmt.children[1], dict(zip([str(i) for i in expr],
                                                        for_sym)), n_replaced)

                # 6) Track hoisted symbols and symbols dependencies
                sym_info = [(i, j, inv_for) for i, j in zip(_expr, var_decl)]
                self.hoisted.update(zip([s.symbol for s in for_sym], sym_info))
                for s, e in zip(for_sym, expr):
                    self.expr_graph.add_dependency(s, e, n_replaced[str(s)] > 1)
                    self.symbols[s] = dep

                # 7a) Update expressions hoisted along a known dimension (same dep)
                inv_info = (for_dep, place, next_loop, wl)
                if inv_info in inv_dep:
                    _var_decl, _inv_for = inv_dep[inv_info]
                    _var_decl.extend(var_decl)
                    _inv_for.extend(inv_for)
                    continue

                # 7b) Keep track of hoisted stuff
                inv_dep[inv_info] = (var_decl, inv_for)

        for inv_info, dep_info in sorted(inv_dep.items()):
            var_decl, inv_for = dep_info
            _, place, next_loop, wl = inv_info
            # Create the hoisted code
            if wl:
                new_for = [dcopy(wl)]
                new_for[0].children[0] = Block(inv_for, open_scope=True)
                inv_for = inv_code = new_for
            else:
                inv_code = [None]
            # Append the new node at the right level in the loop nest
            ofs = place.children.index(next_loop)
            new_block = var_decl + inv_for + [FlatBlock("\n")] + place.children[ofs:]
            place.children = place.children[:ofs] + new_block
            # Update information about hoisted symbols
            for i in var_decl:
                self.hoisted.update_stmt(i.sym.symbol, **{'loop': inv_code[0],
                                                          'place': place})

        # Increase the global counter for subsequent calls to licm
        ExpressionHoister.GLOBAL_LICM_COUNTER += 1


class ExpressionExpander(object):
    """Expand expressions such that: ::

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

    becomes: ::

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    def __init__(self, hoisted, expr_graph):
        self.hoisted = hoisted
        self.expr_graph = expr_graph
        self.expanded_decls = {}
        self.found_consts = {}
        self.expanded_syms = []

    def _do_expand(self, sym, const):
        """Perform the actual expansion. If there are no dependencies, then
        the already hoisted expression is expanded. Otherwise, if the symbol to
        be expanded occurs multiple times in the expression, or it depends on
        other hoisted symbols that will also be expanded, create a new symbol."""

        old_expr = self.hoisted[sym.symbol].expr
        var_decl = self.hoisted[sym.symbol].decl
        loop = self.hoisted[sym.symbol].loop
        place = self.hoisted[sym.symbol].place

        # The expanding expression is first assigned to a temporary value in order
        # to minimize code size and, possibly, work around compiler's inefficiencies
        # when doing loop-invariant code motion
        const_str = str(const)
        if const_str in self.found_consts:
            const = dcopy(self.found_consts[const_str])
        elif not isinstance(const, Symbol):
            const_sym = Symbol("const%d" % len(self.found_consts), ())
            new_const_decl = Decl("double", dcopy(const_sym), const)
            # Keep track of the expansion
            self.expanded_decls[new_const_decl.sym.symbol] = (new_const_decl,
                                                              plan.LOCAL_VAR)
            self.expanded_syms.append(new_const_decl.sym)
            self.found_consts[const_str] = const_sym
            self.expr_graph.add_dependency(const_sym, const, False)
            # Update the AST
            place.children.insert(place.children.index(loop), new_const_decl)
            const = const_sym

        # No dependencies, just perform the expansion
        if not self.expr_graph.has_dep(sym):
            old_expr.children[0] = Prod(Par(old_expr.children[0]), dcopy(const))
            self.expr_graph.add_dependency(sym, const, False)
            return

        # Create a new symbol, expression, and declaration
        new_expr = Par(Prod(dcopy(sym), const))
        sym = dcopy(sym)
        sym.symbol += "_EXP%d" % len(self.expanded_syms)
        new_node = Assign(sym, new_expr)
        new_var_decl = dcopy(var_decl)
        new_var_decl.sym.symbol = sym.symbol
        # Append new expression and declaration
        loop.children[0].children.append(new_node)
        place.children.insert(place.children.index(var_decl), new_var_decl)
        self.expanded_decls[new_var_decl.sym.symbol] = (new_var_decl, plan.LOCAL_VAR)
        self.expanded_syms.append(new_var_decl.sym)
        # Update tracked information
        self.hoisted[sym.symbol] = (new_expr, new_var_decl, loop, place)
        self.expr_graph.add_dependency(sym, new_expr, 0)
        return sym

    def expand(self, node, parent, it_vars, exp_var):
        """Perform the expansion of the expression rooted in ``node``. Terms are
        expanded along the iteration variable ``exp_var``."""

        if isinstance(node, Symbol):
            if not node.rank:
                return ([node], self.CONST)
            elif node.rank[-1] not in it_vars.keys():
                return ([node], self.CONST)
            else:
                return ([node], self.ITVAR)
        elif isinstance(node, Par):
            return self.expand(node.children[0], node, it_vars, exp_var)
        elif isinstance(node, Prod):
            l_node, l_type = self.expand(node.children[0], node, it_vars, exp_var)
            r_node, r_type = self.expand(node.children[1], node, it_vars, exp_var)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                # Found an expandable product
                to_exp = l_node if l_node[0].rank[-1] == exp_var else r_node
                return (to_exp, self.ITVAR)
            elif l_type == self.CONST and r_type == self.CONST:
                # Product of constants; they are both used for expansion (if any)
                return ([node], self.CONST)
            else:
                # Do the expansion
                const = l_node[0] if l_type == self.CONST else r_node[0]
                expandable, exp_node = (l_node, node.children[0]) \
                    if l_type == self.ITVAR else (r_node, node.children[1])
                to_replace = {}
                for sym in expandable:
                    # Perform the expansion
                    if sym.symbol not in self.hoisted:
                        raise RuntimeError("Expansion error: no symbol: %s" % sym.symbol)
                    replacing = self._do_expand(sym, const)
                    if replacing:
                        to_replace[str(sym)] = replacing
                ast_replace(node, to_replace, copy=True)
                # Update the parent node, since an expression has been expanded
                if parent.children[0] == node:
                    parent.children[0] = exp_node
                elif parent.children[1] == node:
                    parent.children[1] = exp_node
                else:
                    raise RuntimeError("Expansion error: wrong parent-child association")
                return (expandable, self.ITVAR)
        elif isinstance(node, Sum):
            l_node, l_type = self.expand(node.children[0], node, it_vars, exp_var)
            r_node, r_type = self.expand(node.children[1], node, it_vars, exp_var)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                return (l_node + r_node, self.ITVAR)
            elif l_type == self.CONST and r_type == self.CONST:
                return ([node], self.CONST)
            else:
                return (None, self.CONST)
        else:
            raise RuntimeError("Expansion error: unknown node: %s" % str(node))
