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
import operator

from base import *
from utils import *
from loop_scheduler import SSALoopMerger


class ExpressionRewriter(object):
    """Provide operations to re-write an expression:

    * Loop-invariant code motion: find and hoist sub-expressions which are
      invariant with respect to a loop
    * Expansion: transform an expression ``(a + b)*c`` into ``(a*c + b*c)``
    * Factorization: transform an expression ``a*b + a*c`` into ``a*(b+c)``"""

    def __init__(self, stmt, expr_info, decls, header, hoisted, expr_graph):
        """Initialize the ExpressionRewriter.

        :param stmt: AST statement containing the expression to be rewritten
        :param expr_info: ``MetaExpr`` object describing the expression in ``stmt``
        :param decls: declarations for the various symbols in ``stmt``.
        :param header: the parent Block of the loop in which ``stmt`` was found.
        :param hoisted: dictionary that tracks hoisted expressions
        :param expr_graph: expression graph that tracks symbol dependencies
        """
        self.stmt = stmt
        self.expr_info = expr_info
        self.decls = decls
        self.header = header
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        # Expression Manipulators used by the Expression Rewriter
        self.expr_hoister = ExpressionHoister(self.stmt,
                                              self.expr_info,
                                              self.header,
                                              self.decls,
                                              self.hoisted,
                                              self.expr_graph)

    def licm(self, merge_and_simplify=False, compact_tmps=False):
        """Perform generalized loop-invariant code motion.

        :param merge_and_simpliy: True if should try to merge the loops in which
                                  invariant expressions are evaluated, because they
                                  might be characterized by the same iteration space.
                                  In this process, computation which is redundant
                                  because performed in at least two merged loops, is
                                  eliminated.
        :param compact_tmps: True if temporaries accessed only once should be inlined.
        """
        self.expr_hoister.licm()
        stmt_hoisted = self.expr_hoister._expr_handled

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
            stmt_occs = dict((k, v)
                             for d in [count_occurrences(stmt, key=1, read_only=True)
                                       for stmt in stmt_hoisted]
                             for k, v in d.items())
            for l in self.hoisted.all_loops:
                l_occs = count_occurrences(l, key=0, read_only=True)
                to_replace, to_delete = {}, []
                for sym_rank, sym_occs in l_occs.items():
                    # If the symbol appears once, then it is a potential candidate
                    # for removal. It is actually removed if it does't appear in
                    # the expression from which was extracted. Symbols appearing
                    # more than once are removed if they host an expression made
                    # of just one symbol
                    sym, rank = sym_rank
                    if sym not in self.hoisted or sym in stmt_occs:
                        continue
                    loop = self.hoisted[sym].loop
                    if loop is not l:
                        continue
                    expr = self.hoisted[sym].expr
                    if sym_occs > 1 and not isinstance(expr.children[0], Symbol):
                        continue
                    if not self.hoisted[sym].loop:
                        continue

                    to_replace[str(Symbol(sym, rank))] = expr
                    to_delete.append(sym)
                for stmt in l.body:
                    _, expr = stmt.children
                    ast_replace(expr, to_replace, copy=True)
                for sym in to_delete:
                    self.hoisted.delete_hoisted(sym)
                    self.decls.pop(sym)

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
        dim_occs = dict.fromkeys(self.expr_info.domain, 0)
        for _, dim in count_occurrences(self.stmt.children[1]).keys():
            if dim and dim[0] in dim_occs:
                dim_occs[dim[0]] += 1
        dim_exp = max(dim_occs.iteritems(), key=operator.itemgetter(1))[0]

        # Perform expansion
        ee = ExpressionExpander(self.stmt, self.expr_info, self.hoisted,
                                self.expr_graph, dim_occs, dim_exp)
        ee.expand()

        # Update known declarations
        self.decls.update(ee.expanded_decls)

    def factorize(self, mode='default'):
        """Factorize terms in the expression. For example: ::

            A[i]*B[j] + A[i]*C[j]

        becomes ::

            A[i]*(B[j] + C[j]).

        :param mode: different factorization strategies are possible, each exposing
                     distinct, "hidden" opportunities for code motion.

                     * mode == 'standard': this simple heuristics consists of \
                                           grouping on symbols that appear the \
                                           most in the expression.
                     * mode == 'immutable': if many static constant objects are \
                                            expected, with this strategy they are \
                                            grouped together, within the obvious \
                                            limits imposed by the expression itself.
        """
        if mode not in ['standard', 'immutable']:
            warning('Unknown factorization strategy. Skipping.')
        ef = ExpressionFactorizer(mode, self.stmt)
        ef.factorize()


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

    # Track all expressions to which LICM has been applied
    _expr_handled = []
    # Temporary variables template
    _hoisted_sym = "%(loop_dep)s_%(expr_id)d_%(round)d_%(i)d"

    # Constants used by the extract method to charaterize sub-expressions
    INV = 0  # Invariant term, hoistable if part of an invariant expression
    KSE = 1  # Invariant expression, potentially part of larger invariant
    HOI = 2  # Variant expression, hoisted, can't hoist anymore

    def __init__(self, stmt, expr_info, header, decls, hoisted, expr_graph):
        """Initialize the ExpressionHoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.decls = decls
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        # Set counters to create meaningful and unique (temporary) variable names
        try:
            self.expr_id = self._expr_handled.index(stmt)
        except ValueError:
            self._expr_handled.append(stmt)
            self.expr_id = self._expr_handled.index(stmt)
        self.counter = 0

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
        if isinstance(node, FunCall):
            arg_deps = [self._extract_exprs(n, expr_dep, length) for n in node.children]
            dep = tuple(set(flatten([dep for dep, _, _ in arg_deps])))
            info = self.INV if all([i == self.INV for _, i, _ in arg_deps]) else self.HOI
            return (dep, info, length)
        if isinstance(node, Par):
            return (self._extract_exprs(node.children[0], expr_dep, length))

        # Traverse the expression tree
        left, right = node.children
        dep_l, info_l, len_l = self._extract_exprs(left, expr_dep, length)
        dep_r, info_r, len_r = self._extract_exprs(right, expr_dep, length)
        node_len = len_l + len_r

        # Filter out false dependencies
        dep_l = tuple(d for d in dep_l if d in self.expr_deps)
        dep_r = tuple(d for d in dep_r if d in self.expr_deps)

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
            # E.g. A[i]*(B[j] + C[j])
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
        elif info_l == self.HOI:
            if info_r == self.INV:
                hoist(right, dep_r, expr_dep, not self.counter)
            elif info_r == self.KSE:
                hoist(right, dep_r, expr_dep, len_r > 2)
            return ((), self.HOI, node_len)
        elif info_r == self.HOI:
            if info_l == self.INV:
                hoist(left, dep_l, expr_dep, not self.counter)
            elif info_l == self.KSE:
                hoist(left, dep_l, expr_dep, len_l > 2)
            return ((), self.HOI, node_len)
        else:
            raise RuntimeError("Fatal error while finding hoistable terms")

    def _check_loops(self, loops):
        """Ensures hoisting is legal. As long as all inner loops are perfect,
        hoisting at the bottom of the possibly non-perfect outermost loop
        always is a legal transformation."""
        return all([is_perfect_loop(l) for l in loops[1:]])

    def licm(self):
        """Perform generalized loop-invariant code motion."""
        # Aliases
        expr_type = self.expr_info.type
        expr_loops = self.expr_info.loops
        expr_outermost_loop = expr_loops[0]
        is_expr_outermost_perfect = is_perfect_loop(expr_outermost_loop)

        if not self._check_loops(expr_loops):
            warning("Loop nest unsuitable for generalized licm. Skipping.")
            return

        # (Re)set global parameters for the /extract/ function
        self.symbols = visit(self.header, None)['symbols_dep']
        self.symbols = dict((s, [l.dim for l in dep]) for s, dep in self.symbols.items())
        self.extracted = False

        expr_dims_loops = For.fromloops(expr_loops)
        self.expr_deps = expr_dims_loops.keys()

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
                dep = tuple(d for d in all_deps if d in self.expr_deps)

                # 0) The invariant statements go in the closest outer loop to
                # dep[-1] which they depend on, and are wrapped by a loop wl
                # iterating along the same iteration space as dep[-1].
                # If there's no such an outer loop, they fall in the header,
                # provided they are within a perfect loop nest (otherwise,
                # dependencies may be broken)
                if len(dep) == 0:
                    place, wl = self.header, None
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and is_expr_outermost_perfect:
                    place, wl = self.header, expr_dims_loops[dep[0]]
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and len(expr_dims_loops) > 1:
                    place, wl = expr_dims_loops[dep[0]].children[0], None
                    next_loop = od_find_next(expr_dims_loops, dep[0])
                elif len(dep) == 1:
                    place, wl = expr_dims_loops[dep[0]].children[0], None
                    next_loop = place.children[-1]
                else:
                    dep_block = expr_dims_loops[dep[-2]].children[0]
                    place, wl = dep_block, expr_dims_loops[dep[-1]]
                    next_loop = od_find_next(expr_dims_loops, dep[-2])

                # 1) Remove identical sub-expressions
                expr = dict([(str(e), e) for e in expr]).values()

                # 2) Create the new invariant sub-expressions and temporaries
                sym_rank, for_dep = (tuple([wl.size]), tuple([wl.dim])) \
                    if wl else ((), ())
                syms = [Symbol(self._hoisted_sym % {
                    'loop_dep': '_'.join(dep).upper() if dep else 'CONST',
                    'expr_id': self.expr_id,
                    'round': self.counter,
                    'i': i
                }, sym_rank) for i in range(len(expr))]
                var_decl = [Decl(expr_type, s) for s in syms]
                for_sym = [Symbol(d.sym.symbol, for_dep) for d in var_decl]

                # 3) Create the new for loop containing invariant terms
                _expr = [Par(dcopy(e)) if not isinstance(e, Par)
                         else dcopy(e) for e in expr]
                inv_for = [Assign(s, e) for s, e in zip(dcopy(for_sym), _expr)]

                # 4) Update the dictionary of known declarations
                for d in var_decl:
                    d.scope = LOCAL
                    self.decls[d.sym.symbol] = d

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
            # Insert the new nodes at the right level in the loop nest
            ofs = place.children.index(next_loop)
            place.children[ofs:ofs] = var_decl + inv_for + [FlatBlock("\n")]
            # Update hoisted symbols metadata
            for i in var_decl:
                self.hoisted.update_stmt(i.sym.symbol, loop=inv_code[0], place=place)


class ExpressionExpander(object):
    """Expand expressions such that: ::

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

    becomes: ::

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    # Track all expanded expressions
    _expr_handled = []
    # Temporary variables template
    _expanded_sym = "CONST_EXP_%(expr_id)d_%(i)d"

    def __init__(self, stmt, expr_info, hoisted, expr_graph, dim_occs, dim_exp):
        self.stmt = stmt
        self.expr_info = expr_info
        self.hoisted = hoisted
        self.expr_graph = expr_graph
        self.dim_occs = dim_occs
        self.dim_exp = dim_exp

        # Set counters to create meaningful and unique (temporary) variable names
        try:
            self.expr_id = self._expr_handled.index(stmt)
        except ValueError:
            self._expr_handled.append(stmt)
            self.expr_id = self._expr_handled.index(stmt)

        self.expanded_decls = {}
        self.found_consts = {}
        self.expanded_syms = []

    def _make_expansion(self, sym, const, op):
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
            const_sym = Symbol(self._expanded_sym % {'expr_id': self.expr_id,
                                                     'i': len(self.found_consts)})
            new_const_decl = Decl(self.expr_info.type, dcopy(const_sym), const)
            # Keep track of the expansion
            new_const_decl.scope = LOCAL
            self.expanded_decls[new_const_decl.sym.symbol] = new_const_decl
            self.expanded_syms.append(new_const_decl.sym)
            self.found_consts[const_str] = const_sym
            self.expr_graph.add_dependency(const_sym, const, False)
            # Update the AST
            place.children.insert(place.children.index(loop), new_const_decl)
            const = const_sym

        # No dependencies, just perform the expansion
        if not self.expr_graph.has_dep(sym):
            old_expr.children[0] = op(Par(old_expr.children[0]), dcopy(const))
            self.expr_graph.add_dependency(sym, const, False)
            return

        # Create a new symbol, expression, and declaration
        new_expr = Par(op(dcopy(sym), const))
        sym = dcopy(sym)
        sym.symbol += "_EXP%d" % len(self.expanded_syms)
        new_node = Assign(sym, new_expr)
        new_var_decl = dcopy(var_decl)
        new_var_decl.sym.symbol = sym.symbol
        # Append new expression and declaration
        loop.body.append(new_node)
        place.children.insert(place.children.index(var_decl), new_var_decl)
        self.expanded_decls[new_var_decl.sym.symbol] = new_var_decl
        self.expanded_syms.append(new_var_decl.sym)
        # Update tracked information
        self.hoisted[sym.symbol] = (new_expr, new_var_decl, loop, place)
        self.expr_graph.add_dependency(sym, new_expr, 0)
        return sym

    def _expand(self, node, parent):
        if isinstance(node, Symbol):
            if not node.rank:
                return ([node], self.CONST)
            elif node.rank[-1] not in self.dim_occs.keys():
                return ([node], self.CONST)
            else:
                return ([node], self.ITVAR)
        elif isinstance(node, Par):
            return self._expand(node.children[0], node)
        elif isinstance(node, FunCall):
            # Functions are considered potentially expandable
            return ([node], self.CONST)
        elif isinstance(node, (Prod, Div)):
            l_node, l_type = self._expand(node.children[0], node)
            r_node, r_type = self._expand(node.children[1], node)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                # Found an expandable product
                to_exp = l_node if l_node[0].rank[-1] == self.dim_exp else r_node
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
                    replacing = self._make_expansion(sym, const, node.__class__)
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
                # Replace expanded symbols with the newly used symbols
                expandable = list(set(e if str(e) not in to_replace
                                      else to_replace[str(e)] for e in expandable))
                return (expandable, self.ITVAR)
        elif isinstance(node, (Sum, Sub)):
            l_node, l_type = self._expand(node.children[0], node)
            r_node, r_type = self._expand(node.children[1], node)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                return (l_node + r_node, self.ITVAR)
            elif l_type == self.CONST and r_type == self.CONST:
                return ([node], self.CONST)
            else:
                return (None, self.CONST)
        else:
            raise RuntimeError("Expansion error: unknown node: %s" % str(node))

    def expand(self):
        """Perform the expansion of the expression rooted in ``self.stmt``.
        Terms are expanded along the iteration variable ``self.dim_exp``."""
        self._expand(self.stmt.children[1], self.stmt)


class ExpressionFactorizer(object):

    def __init__(self, mode, stmt):
        self.mode = mode
        self.stmt = stmt

    def _find_prod(self, node, occs, factorizable):
        if isinstance(node, Symbol):
            return
        elif isinstance(node, Par):
            self._find_prod(node.children[0], occs, factorizable)
        elif isinstance(node, Sum):
            left, right = (node.children[0], node.children[1])
            self._find_prod(left, occs, factorizable)
            self._find_prod(right, occs, factorizable)
        elif isinstance(node, Prod):
            left, right = (node.children[0], node.children[1])
            self._find_prod(left, occs, factorizable)
            self._find_prod(right, occs, factorizable)
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
                return
            factorizable[dist].append(target)

    def factorize(self):
        if self.mode == 'immutable':
            raise NotImplementedError("Strategy yet not implemented")

        factorizable = defaultdict(list)
        occurrences = count_occurrences(self.stmt.children[1], key=2)
        self._find_prod(self.stmt.children[1], occurrences, factorizable)

        if not factorizable:
            return

        new_prods = []
        for d in factorizable.values():
            dist, target = zip(*d)
            target = Par(ast_make_sum(target)) if len(target) > 1 else ast_make_sum(target)
            new_prods.append(Par(Prod(dist[0], target)))
        self.stmt.children[1] = Par(ast_make_sum(new_prods))
