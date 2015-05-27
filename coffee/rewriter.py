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

from collections import defaultdict, Counter
from copy import deepcopy as dcopy
import itertools

from base import *
from utils import *
from coffee.visitors import *


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

        # Expression manipulators used by the Expression Rewriter
        self.expr_hoister = ExpressionHoister(self.stmt,
                                              self.expr_info,
                                              self.header,
                                              self.decls,
                                              self.hoisted,
                                              self.expr_graph)
        self.expr_expander = ExpressionExpander(self.stmt,
                                                self.expr_info,
                                                self.hoisted,
                                                self.expr_graph)
        self.expr_factorizer = ExpressionFactorizer(self.stmt,
                                                    self.expr_info)

    def licm(self, **kwargs):
        """Perform generalized loop-invariant code motion.

        Loop-invariant expressions found in the nest are moved "after" the
        outermost independent loop and "after" the fastest varying dimension
        loop. Here, "after" means that if the loop nest has two loops ``i``
        and ``j``, and ``j`` is in the body of ``i``, then ``i`` comes after
        ``j`` (i.e. the loop nest has to be read from right to left).

        For example, if a sub-expression ``E`` depends on ``[i, j]`` and the
        loop nest has three loops ``[i, j, k]``, then ``E`` is hoisted out from
        the body of ``k`` to the body of ``i``). All hoisted expressions are
        then wrapped and evaluated in a new loop in order to promote compiler
        autovectorization.

        :param kwargs:
            * hoist_domain_const: True if ``n``-dimensional arrays are allowed
                for hoisting expressions crossing ``n`` loops in the nest.
            * hoist_out_domain: True if only outer-loop invariant terms should
                be hoisted
        """
        self.expr_hoister.licm(**kwargs)

    def expand(self, mode='standard'):
        """Expand expressions over other expressions based on different heuristics.
        In the simplest example one can have: ::

            (X[i] + Y[j])*F + ...

        which could be transformed into: ::

            (X[i]*F + Y[j]*F) + ...

        When creating the expanded object, if the expanding term had already been
        hoisted, then the expansion itself is also lifted. For example, if: ::

            Y[j] = f(...)
            (X[i]*Y[j])*F + ...

        and we assume it has been decided (see below) the expansion should occur
        along the loop dimension ``j``, the transformation generates: ::

            Y[j] = f(...)*F
            (X[i]*Y[j]) + ...

        One may want to expand expressions for several reasons, which include

        * Exposing factorization opportunities;
        * Exposing high-level (linear algebra) operations (e.g., matrix multiplies)
        * Relieving register pressure; when, for example, ``(X[i]*Y[j])`` is
          computed in a loop L' different than the loop L'' in which ``Y[j]``
          is evaluated, and ``cost(L') > cost(L'')``;

        :param mode: multiple expansion strategies are possible, each exposing
                     different, "hidden" opportunities for later code motion.

                     * mode == 'standard': this heuristics consists of expanding \
                                           along the loop dimension appearing \
                                           the most in different (i.e., unique) \
                                           arrays. This has the goal of making \
                                           factorization more effective.
                     * mode == 'full': expansion is performed aggressively without \
                                       any specific restrictions.
        """
        symbols = FindInstances(Symbol).visit(self.stmt.children[1])[Symbol]

        # Select the expansion strategy
        if mode == 'standard':
            # Get the ranks...
            occurrences = [s.rank for s in symbols if s.rank]
            # ...filter out irrelevant dimensions...
            occurrences = [tuple(r for r in rank if r in self.expr_info.domain_dims)
                           for rank in occurrences]
            # ...and finally establish the expansion dimension
            dimension = Counter(occurrences).most_common(1)[0][0]
            should_expand = lambda n: set(dimension).issubset(set(n.rank))
        elif mode == 'full':
            should_expand = lambda n: \
                n.symbol in self.decls and self.decls[n.symbol].is_static_const
        else:
            warning('Unknown expansion strategy. Skipping.')
            return

        # Perform the expansion
        self.expr_expander.expand(should_expand)

        # Update known declarations
        self.decls.update(self.expr_expander.expanded_decls)

    def factorize(self, mode='standard'):
        """Factorize terms in the expression. For example: ::

            A[i]*B[j] + A[i]*C[j]

        becomes ::

            A[i]*(B[j] + C[j]).

        :param mode: multiple factorization strategies are possible, each exposing
                     different, "hidden" opportunities for code motion.

                     * mode == 'standard': this simple heuristics consists of \
                                           grouping on symbols that appear the \
                                           most in the expression.
                     * mode == 'immutable': if many static constant objects are \
                                            expected, with this strategy they are \
                                            grouped together, within the obvious \
                                            limits imposed by the expression itself.
        """
        symbols = FindInstances(Symbol).visit(self.stmt.children[1])[Symbol]

        # Select the expansion strategy
        if mode == 'standard':
            # Get the ranks...
            occurrences = [s.rank for s in symbols if s.rank]
            # ...filter out irrelevant dimensions...
            occurrences = [tuple(r for r in rank if r in self.expr_info.domain_dims)
                           for rank in occurrences]
            # ...and finally establish the expansion dimension
            dimension = Counter(occurrences).most_common(1)[0][0]
            should_factorize = lambda n: set(dimension).issubset(set(n.rank))
        elif mode == 'immutable':
            should_factorize = lambda n: \
                n.symbol in self.decls and self.decls[n.symbol].is_static_const
        if mode not in ['standard', 'immutable']:
            warning('Unknown factorization strategy. Skipping.')
            return

        # Perform the factorization
        self.expr_factorizer.factorize(should_factorize)

    def reassociate(self, reorder=lambda x: x):
        """Reorder symbols in associative operations following a convention.
        By default, the convention is to order the symbols based on their rank.
        For example, the terms in the expression ::

            a*b[i]*c[i][j]*d

        are reordered as ::

            a*d*b[i]*c[i][j]

        This as achieved by reorganizing the AST of the expression.
        """

        def _reassociate(node, parent):
            if isinstance(node, (Symbol, Div)):
                return

            elif isinstance(node, Par):
                _reassociate(node.child, node)

            elif isinstance(node, (Sum, Sub, FunCall)):
                for n in node.children:
                    _reassociate(n, node)

            elif isinstance(node, Prod):
                children = explore_operator(node)
                # Reassociate symbols
                symbols = [(n.rank, n, p) for n, p in children if isinstance(n, Symbol)]
                symbols.sort(key=lambda n: (len(n[0]), n[0]))
                # Capture the other children and recur on them
                other_nodes = [(n, p) for n, p in children if not isinstance(n, Symbol)]
                for n, p in other_nodes:
                    _reassociate(n, p)
                # Create the reassociated product and modify the original AST
                children = zip(*other_nodes)[0] if other_nodes else ()
                children += tuple(reorder(zip(*symbols)[1] if symbols else ()))
                reassociated_node = ast_make_expr(Prod, children)
                parent.children[parent.children.index(node)] = reassociated_node

            else:
                warning('Unexpect node of type %s while reassociating', typ(node))

        _reassociate(self.stmt.children[1], self.stmt)

    def inject(self):
        """Unroll any loops outside of the expression iteration space inside the
        expression itself ("injection"). For example: ::

            for i
              for r
                a += B[r]*C[i][r]
              for j
                for k
                  A[j][k] += ...f(a)... // the expression being rewritten

        gets transformed into:

            for i
              for j
                for k
                  A[j][k] += ...f(B[0]*C[i][0] + B[1]*C[i][1] + ...)...
        """
        # Get all loop nests, then discard the one enclosing the expression
        nests = [n for n in visit(self.expr_info.loops_parents[0])['fors']]
        injectable_nests = [n for n in nests if zip(*n)[0] != self.expr_info.loops]

        # Full unroll any unrollable, injectable loop
        for nest in injectable_nests:
            to_unroll = [(l, p) for l, p in nest if l not in self.expr_info.loops]
            nest_writers = FindInstances(Writer).visit(to_unroll[0][0])
            for op, stmts in nest_writers.items():
                if op in [Assign, IMul, IDiv]:
                    # Unroll is unsafe, skip
                    continue
                assert op in [Incr, Decr]
                for stmt in stmts:
                    sym, expr = stmt.children
                    if stmt in [l.incr for l, p in to_unroll]:
                        # Ignore useless stmts
                        continue
                    for l, p in reversed(to_unroll):
                        inject_expr = [dcopy(expr) for i in range(l.size)]
                        # Update rank of symbols
                        for i, e in enumerate(inject_expr):
                            e_syms = FindInstances(Symbol).visit(e)[Symbol]
                            for s in e_syms:
                                s.rank = tuple([r if r != l.dim else i for r in s.rank])
                        expr = ast_make_expr(Sum, inject_expr)
                    # Inject the unrolled operation into the expression
                    ast_replace(self.stmt, {sym: expr}, copy=True)
                    # Clean up
                    p.children.remove(self.decls[sym.symbol])
                    self.decls.pop(sym.symbol)
            for l, p in to_unroll:
                p.children.remove(l)

    def simplify(self):
        """Simplify an expression by applying transformations which should enhance
        the effectiveness of later rewriting passes. The transformations applied
        are: ::

            * ``division replacement``: replace divisions by a constant with
                multiplications, which gives expansion and factorization more
                rewriting opportunities
            * ``loop reduction``: simplify loops along which a reduction is
                performed by 1) preaccumulating along the dimension of the
                reduction, 2) removing the reduction loop, 3) precomputing
                constant expressions
        """
        # Aliases
        stmt, expr_info = self.stmt, self.expr_info

        # Division replacement
        divisions = FindInstances(Div).visit(stmt.children[1])[Div]
        to_replace = {}
        for i in divisions:
            if isinstance(i.right, Symbol) and isinstance(i.right.symbol, float):
                to_replace[i] = Prod(i.left, Div(1, i.right.symbol))
        ast_replace(stmt.children[1], to_replace, copy=True, mode='symbol')

        ###

        # Loop reduction
        if not isinstance(stmt, (Incr, Decr, IMul, IDiv)):
            # Not a reduction expression, give up
            return
        expr_syms = FindInstances(Symbol).visit(stmt.children[1])[Symbol]
        reduction_loops = expr_info.out_domain_loops_info
        if any([not is_perfect_loop(l) for l, p in reduction_loops]):
            # Unsafe if not a perfect loop nest
            return
        for i, (l, p) in enumerate(reduction_loops):
            syms_dep = SymbolDependencies().visit(l, **SymbolDependencies.default_args)
            if not all([tuple(syms_dep[s]) == expr_info.loops and
                        s.dim == len(expr_info.loops) for s in expr_syms if syms_dep[s]]):
                # A sufficient (although not necessary) condition for loop reduction to
                # be safe is that all symbols in the expression are either constants or
                # tensors assuming a distinct value in each point of the iteration space.
                # So if this condition fails, we give up
                return
            # At this point, tensors can be reduced along the reducible dimensions
            reducible_syms = [s for s in expr_syms if s.rank]
            # All involved symbols must result from hoisting
            if not all([s.symbol in self.hoisted for s in reducible_syms]):
                return
            # Replace hoisted assignments with reductions
            finder = FindInstances(Assign, stop_when_found=True, with_parent=True)
            for hoisted_loop in self.hoisted.all_loops:
                for assign, parent in finder.visit(hoisted_loop)[Assign]:
                    sym, expr = assign.children
                    decl = self.hoisted[sym.symbol].decl
                    if sym.symbol in [s.symbol for s in reducible_syms]:
                        parent.children[parent.children.index(assign)] = Incr(sym, expr)
                        sym.rank = self.expr_info.domain_dims
                        decl.sym.rank = decl.sym.rank[i+1:]
            # Remove the reduction loop and update the rank of its symbols
            p.children[p.children.index(l)] = l.body[0]
            for s in reducible_syms:
                s.rank = self.expr_info.domain_dims

        ###

        # Precompute constant expressions
        for hoisted_loop in self.hoisted.all_loops:
            evals = Evaluate(self.decls).visit(hoisted_loop, **Evaluate.default_args)
            for s, values in evals.items():
                self.hoisted[s.symbol].decl.init = ArrayInit(values)
                self.hoisted[s.symbol].decl.qual = ['static', 'const']
            self.header.children.remove(hoisted_loop)

    @staticmethod
    def reset():
        ExpressionHoister._expr_handled[:] = []
        ExpressionExpander._expr_handled[:] = []


class ExpressionHoister(object):

    # Track all expressions to which LICM has been applied
    _expr_handled = []
    # Temporary variables template
    _hoisted_sym = "%(loop_dep)s_%(expr_id)d_%(round)d_%(i)d"

    def __init__(self, stmt, expr_info, header, decls, hoisted, expr_graph):
        """Initialize the ExpressionHoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.decls = decls
        self.hoisted = hoisted
        self.expr_graph = expr_graph
        self.counter = 0

        # Set counters to create meaningful and unique (temporary) variable names
        try:
            self.expr_id = self._expr_handled.index(stmt)
        except ValueError:
            self._expr_handled.append(stmt)
            self.expr_id = self._expr_handled.index(stmt)

    def _extract(self, node, symbols, **kwargs):
        """Extract invariant subexpressions from the original expression.
        Hoistable sub-expressions are stored in ``dep_subexprs``."""

        # Parameters driving the extraction pass
        hoist_domain_const = kwargs.get('hoist_domain_const')
        hoist_out_domain = kwargs.get('hoist_out_domain')

        # Constants used to charaterize sub-expressions:
        INVARIANT = 0  # expression is loop invariant
        HOISTED = 1  # expression should not be hoisted any further

        def __try_hoist(node, dep):
            should_extract = True
            if isinstance(node, Symbol):
                should_extract = False
            elif hoist_out_domain:
                if not set(dep).issubset(set(self.expr_info.out_domain_dims)):
                    should_extract = False
            if should_extract:
                dep_subexprs[dep].append(node)
            return should_extract

        def __extract(node, dep_subexprs):
            if isinstance(node, Symbol):
                return (symbols[node], INVARIANT)

            elif isinstance(node, Par):
                return (__extract(node.child, dep_subexprs))

            elif isinstance(node, FunCall):
                arg_deps = [__extract(n, dep_subexprs) for n in node.children]
                dep = tuple(set(flatten([dep for dep, _ in arg_deps])))
                info = INVARIANT if all([i == INVARIANT for _, i in arg_deps]) else HOISTED
                return (dep, info)

            else:
                # Traverse the expression tree
                left, right = node.children
                dep_l, info_l = __extract(left, dep_subexprs)
                dep_r, info_r = __extract(right, dep_subexprs)

                # Filter out false dependencies
                dep_l = tuple(d for d in dep_l if d in self.expr_info.dims)
                dep_r = tuple(d for d in dep_r if d in self.expr_info.dims)
                dep_n = dep_l + tuple(d for d in dep_r if d not in dep_l)

                if info_l == INVARIANT and info_r == INVARIANT:
                    if dep_l == dep_r:
                        # E.g. alpha*beta, A[i] + B[i]
                        return (dep_l, INVARIANT)
                    elif (not dep_l or not dep_r) and not hoist_domain_const:
                        # E.g. A[i,j]*alpha, alpha*A[i,j]
                        if __try_hoist(left, dep_l) or __try_hoist(right, dep_r):
                            return (dep_n, HOISTED)
                        return (dep_n, INVARIANT)
                    elif not dep_l and hoist_domain_const:
                        # E.g. alpha*A[i,j], not hoistable anymore
                        __try_hoist(right, dep_r)
                    elif not dep_r and hoist_domain_const:
                        # E.g. A[i,j]*alpha, not hoistable anymore
                        __try_hoist(left, dep_l)
                    elif set(dep_l).issubset(set(dep_r)):
                        # E.g. A[i]*B[i,j]
                        return (dep_r, INVARIANT)
                    elif set(dep_r).issubset(set(dep_l)):
                        # E.g. A[i,j]*B[i]
                        return (dep_l, INVARIANT)
                    elif hoist_domain_const:
                        # E.g. A[i]*B[j], hoistable in TMP[i,j]
                        return (dep_n, INVARIANT)
                    else:
                        # E.g. A[i]*B[j]
                        __try_hoist(left, dep_l)
                        __try_hoist(right, dep_r)
                elif info_r == INVARIANT:
                    __try_hoist(right, dep_r)
                elif info_l == INVARIANT:
                    __try_hoist(left, dep_l)
                return (dep_n, HOISTED)

        dep_subexprs = defaultdict(list)
        __extract(node, dep_subexprs)
        self.counter += 1
        return dep_subexprs

    def _check_loops(self, loops):
        """Ensures hoisting is legal. As long as all inner loops are perfect,
        hoisting at the bottom of the possibly non-perfect outermost loop
        always is a legal transformation."""
        return all([is_perfect_loop(l) for l in loops[1:]])

    def licm(self, **kwargs):
        """Perform generalized loop-invariant code motion."""
        if not self._check_loops(self.expr_info.loops):
            warning("Loop nest unsuitable for generalized licm. Skipping.")
            return

        symbols = visit(self.header)['symbols_dep']
        symbols = dict((s, [l.dim for l in dep]) for s, dep in symbols.items())

        extracted = True
        expr_dims_loops = self.expr_info.loops_from_dims
        expr_outermost_loop = self.expr_info.outermost_loop
        inv_dep = {}
        while extracted:
            extracted = self._extract(self.stmt.children[1], symbols, **kwargs)
            for dep, subexprs in extracted.items():
                # -1) Remove identical subexpressions
                subexprs = uniquify(subexprs)

                # 0) Determine the loop nest level where invariant expressions
                # should be hoisted. The goal is to hoist them as far as possible
                # in the loop nest, while minimising temporary storage.
                # We distinguish six hoisting cases:
                if len(dep) == 0:
                    # As scalar (/wrap_loop=None/), outside of the loop nest;
                    place = self.header
                    wrap_loop = ()
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and is_perfect_loop(expr_outermost_loop):
                    # As vector, outside of the loop nest;
                    place = self.header
                    wrap_loop = (expr_dims_loops[dep[0]],)
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and len(expr_dims_loops) > 1:
                    # As scalar, within the loop imposing the dependency
                    place = expr_dims_loops[dep[0]].children[0]
                    wrap_loop = ()
                    next_loop = od_find_next(expr_dims_loops, dep[0])
                elif len(dep) == 1:
                    # As scalar, at the bottom of the loop imposing the dependency
                    place = expr_dims_loops[dep[0]].children[0]
                    wrap_loop = ()
                    next_loop = place.children[-1]
                elif set(dep).issuperset(set(self.expr_info.domain_dims)) and \
                        not any([self.expr_graph.is_written(e) for e in subexprs]):
                    # As n-dimensional vector, where /n == len(dep)/, outside of
                    # the loop nest
                    place = self.header
                    wrap_loop = tuple(expr_dims_loops.values())
                    next_loop = expr_outermost_loop
                else:
                    # As vector, within the outermost loop imposing the dependency
                    dep_block = expr_dims_loops[dep[0]].children[0]
                    place = dep_block
                    wrap_loop = tuple(expr_dims_loops[dep[i]] for i in range(1, len(dep)))
                    next_loop = od_find_next(expr_dims_loops, dep[0])

                # 1) Create the new invariant temporary symbols
                loop_size = tuple([l.size for l in wrap_loop])
                loop_dim = tuple([l.dim for l in wrap_loop])
                inv_syms = [Symbol(self._hoisted_sym % {
                    'loop_dep': '_'.join(dep).upper() if dep else 'CONST',
                    'expr_id': self.expr_id,
                    'round': self.counter,
                    'i': i
                }, loop_size) for i in range(len(subexprs))]
                inv_decls = [Decl(self.expr_info.type, s) for s in inv_syms]
                inv_syms = [Symbol(s.symbol, loop_dim) for s in inv_syms]

                # 2) Keep track of new declarations for later easy access
                for d in inv_decls:
                    d.scope = LOCAL
                    self.decls[d.sym.symbol] = d

                # 3) Replace invariant subtrees with the proper temporary
                to_replace = dict(zip(subexprs, inv_syms))
                n_replaced = ast_replace(self.stmt.children[1], to_replace)

                # 4) Update symbol dependencies
                for s, e in zip(inv_syms, subexprs):
                    self.expr_graph.add_dependency(s, e)
                    if n_replaced[str(s)] > 1:
                        self.expr_graph.add_dependency(s, s)
                    symbols[s] = dep

                # 5) Create the body containing invariant statements
                subexprs = [Par(dcopy(e)) if not isinstance(e, Par) else
                            dcopy(e) for e in subexprs]
                inv_stmts = [Assign(s, e) for s, e in zip(dcopy(inv_syms), subexprs)]

                # 6) Track necessary information for AST construction
                inv_info = (loop_dim, place, next_loop, wrap_loop)
                if inv_info not in inv_dep:
                    inv_dep[inv_info] = (inv_decls, inv_stmts)
                else:
                    inv_dep[inv_info][0].extend(inv_decls)
                    inv_dep[inv_info][1].extend(inv_stmts)

        for inv_info, (inv_decls, inv_stmts) in sorted(inv_dep.items()):
            loop_dim, place, next_loop, wrap_loop = inv_info
            # Create the hoisted code
            if wrap_loop:
                outer_wrap_loop = ast_make_for(inv_stmts, wrap_loop[-1])
                for l in reversed(wrap_loop[:-1]):
                    outer_wrap_loop = ast_make_for([outer_wrap_loop], l)
                inv_code = [outer_wrap_loop]
                inv_loop = inv_code[0]
            else:
                inv_code = inv_stmts
                inv_loop = None
            # Insert the new nodes at the right level in the loop nest
            ofs = place.children.index(next_loop)
            place.children[ofs:ofs] = inv_decls + inv_code + [FlatBlock("\n")]
            # Track hoisted symbols
            for i, j in zip(inv_stmts, inv_decls):
                self.hoisted[j.sym.symbol] = (i, j, inv_loop, place)

        # Finally, make sure symbols are unique in the AST
        self.stmt.children[1] = dcopy(self.stmt.children[1])


class ExpressionExpander(object):

    # Constants used by the expand method to charaterize sub-expressions:
    GROUP = 0  # Expression /will/ not trigger expansion
    EXPAND = 1  # Expression /could/ be expanded

    # Track all expanded expressions
    _expr_handled = []
    # Temporary variables template
    _expanded_sym = "%(loop_dep)s_EXP_%(expr_id)d_%(i)d"

    def __init__(self, stmt, expr_info, hoisted, expr_graph):
        self.stmt = stmt
        self.expr_info = expr_info
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        # Set counters to create meaningful and unique (temporary) variable names
        try:
            self.expr_id = self._expr_handled.index(stmt)
        except ValueError:
            self._expr_handled.append(stmt)
            self.expr_id = self._expr_handled.index(stmt)

        self.expanded_decls = {}
        self.cache = {}

    def _hoist(self, expansion):
        """Try to aggregate an expanded expression E with a previously hoisted
        expression H. If there are no dependencies, H is expanded with E, so
        no new symbols need be introduced. Otherwise (e.g., the H temporary
        appears in multiple places), create a new symbol."""
        exp, grp = expansion.left, expansion.right
        cache_key = (str(exp), str(grp))

        # First, check if any of the symbols in /exp/ have been hoisted
        try:
            exp = [s for s in FindInstances(Symbol).visit(exp)[Symbol]
                   if s.symbol in self.hoisted and self.should_expand(s)][0]
        except:
            # No hoisted symbols in the expanded expression, so return
            return {}

        # Before moving on, access the cache to check whether the same expansion
        # has alredy been performed. If that's the case, we retrieve and return the
        # result of that expansion, since there is no need to add further temporaries
        if cache_key in self.cache:
            return {exp: self.cache[cache_key]}

        # Aliases
        hoisted_expr = self.hoisted[exp.symbol].stmt.children[1]
        hoisted_decl = self.hoisted[exp.symbol].decl
        hoisted_loop = self.hoisted[exp.symbol].loop
        hoisted_place = self.hoisted[exp.symbol].place
        op = expansion.__class__

        # Is the grouped symbol hoistable, or does it break some data dependency?
        grp_syms = SymbolReferences().visit(grp).keys()
        for l in reversed(self.expr_info.loops):
            for g in grp_syms:
                g_refs = self.info['symbol_refs'][g]
                g_deps = set(flatten([self.info['symbols_dep'][r[0]] for r in g_refs]))
                if any([l.dim in g.dim for g in g_deps]):
                    return {}
            if l in hoisted_place.children:
                break

        # No dependencies, just perform the expansion
        if not self.expr_graph.is_read(exp):
            hoisted_expr.children[0] = op(Par(hoisted_expr.children[0]), dcopy(grp))
            self.expr_graph.add_dependency(exp, grp)
            self.cache[cache_key] = exp
            return {exp: exp}

        # Create new symbol, expression, and declaration
        expr = Par(op(dcopy(exp), grp))
        hoisted_exp = dcopy(exp)
        hoisted_exp.symbol = self._expanded_sym % {'loop_dep': exp.symbol,
                                                   'expr_id': self.expr_id,
                                                   'i': len(self.expanded_decls)}
        decl = dcopy(hoisted_decl)
        decl.sym.symbol = hoisted_exp.symbol
        stmt = Assign(hoisted_exp, expr)
        # Update the AST
        hoisted_loop.body.append(stmt)
        insert_at_elem(hoisted_place.children, hoisted_decl, decl)
        # Update tracked information
        self.expanded_decls[decl.sym.symbol] = decl
        self.hoisted[hoisted_exp.symbol] = (stmt, decl, hoisted_loop, hoisted_place)
        self.expr_graph.add_dependency(hoisted_exp, expr)
        self.cache[cache_key] = hoisted_exp
        return {exp: hoisted_exp}

    def _expand(self, node, parent):
        if isinstance(node, Symbol):
            return ([node], self.EXPAND) if self.should_expand(node) \
                else ([node], self.GROUP)

        elif isinstance(node, Par):
            return self._expand(node.child, node)

        elif isinstance(node, (Div, FunCall)):
            # Try to expand /within/ the children, but then return saying "I'm not
            # expandable any further"
            for n in node.children:
                self._expand(n, node)
            return ([node], self.GROUP)

        elif isinstance(node, Prod):
            l_exps, l_type = self._expand(node.left, node)
            r_exps, r_type = self._expand(node.right, node)
            if l_type == self.GROUP and r_type == self.GROUP:
                return ([node], self.GROUP)
            # At least one child is expandable (marked as EXPAND), whereas the
            # other could either be expandable as well or groupable (marked
            # as GROUP): so we can perform the expansion
            groupable = l_exps if l_type == self.GROUP else r_exps
            expandable = r_exps if l_type == self.GROUP else l_exps
            to_replace = defaultdict(list)
            for exp, grp in itertools.product(expandable, groupable):
                expansion = Prod(exp, dcopy(grp))
                to_replace[exp].append(expansion)
                self.expansions.append(expansion)
            ast_replace(node, {k: ast_make_expr(Sum, v) for k, v in to_replace.items()},
                        mode='symbol')
            # Update the parent node, since an expression has just been expanded
            expanded = node.right if l_type == self.GROUP else node.left
            parent.children[parent.children.index(node)] = expanded
            return (list(flatten(to_replace.values())) or [expanded], self.EXPAND)

        elif isinstance(node, (Sum, Sub)):
            l_exps, l_type = self._expand(node.left, node)
            r_exps, r_type = self._expand(node.right, node)
            if l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sum):
                return (l_exps + r_exps, self.EXPAND)
            elif l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sub):
                return (l_exps + [Neg(r) for r in r_exps], self.EXPAND)
            else:
                return ([node], self.GROUP)

        else:
            raise RuntimeError("Expansion error: unknown node: %s" % str(node))

    def expand(self, should_expand):
        """Perform the expansion of the expression rooted in ``self.stmt``.
        Symbols for which the lambda function ``should_expand`` returns
        True are expansion candidates."""
        # Preload and set data structures for expansion
        self.expansions = []
        self.should_expand = should_expand
        self.info = visit(self.expr_info.outermost_loop)

        # Expand according to the /should_expand/ heuristic
        self._expand(self.stmt.children[1], self.stmt)

        # Try to aggregate expanded terms with hoisted expressions (if any)
        for expansion in self.expansions:
            hoisted = self._hoist(expansion)
            if hoisted:
                ast_replace(self.stmt, hoisted, copy=True, mode='symbol')
                ast_remove(self.stmt, expansion.right, mode='symbol')


class ExpressionFactorizer(object):

    class Term():

        def __init__(self, operands, factors=None, op=None):
            # Example: in the Term /a*(b+c)/, /a/ is an 'operand', /b/ and /c/
            # are 'factors', and /+/ is the 'op'
            self.operands = operands
            self.factors = factors or set()
            self.op = op

        @property
        def operands_ast(self):
            # Exploiting associativity, establish an order for the operands
            operands = sorted(list(self.operands), key=lambda o: str(o))
            return ast_make_expr(Prod, tuple(operands))

        @property
        def factors_ast(self):
            factors = sorted(list(self.factors), key=lambda f: str(f))
            return ast_make_expr(self.op, tuple(factors))

        @property
        def generate_ast(self):
            if len(self.factors) == 0:
                return self.operands_ast
            elif len(self.operands) == 0:
                return self.factors_ast
            else:
                return Prod(self.operands_ast, self.factors_ast)

        @staticmethod
        def process(symbols, should_factorize, op=None):
            operands = set(s for s in symbols if should_factorize(s))
            factors = set(s for s in symbols if not should_factorize(s))
            return ExpressionFactorizer.Term(operands, factors, op)

    def __init__(self, stmt, expr_info):
        self.stmt = stmt
        self.expr_info = expr_info

    def _simplify_sum(self, terms):
        unique_terms = {}
        for t in terms:
            unique_terms.setdefault(str(t.generate_ast), list()).append(t)

        for t_repr, t_list in unique_terms.items():
            occurrences = len(t_list)
            unique_terms[t_repr] = t_list[0]
            if occurrences > 1:
                unique_terms[t_repr].operands.add(Symbol(occurrences))

        terms[:] = unique_terms.values()

    def _factorize(self, node, parent):
        if isinstance(node, Symbol):
            return self.Term.process([node], self.should_factorize)

        elif isinstance(node, Par):
            return self._factorize(node.child, node)

        elif isinstance(node, (FunCall, Div)):
            # Try to factorize /within/ the children, but then return saying
            # "I'm not factorizable any further"
            for n in node.children:
                self._factorize(n, node)
            return self.Term(set([node]))

        elif isinstance(node, Prod):
            children = explore_operator(node)
            symbols = [n for n, _ in children if isinstance(n, Symbol)]
            other_nodes = [(n, p) for n, p in children if n not in symbols]
            term = self.Term.process(symbols, self.should_factorize, Prod)
            for n, p in other_nodes:
                term.operands |= self._factorize(n, p).operands
            return term

        # The fundamental case is when /node/ is a Sum (or Sub, equivalently).
        # Here, we try to factorize the terms composing the operation
        elif isinstance(node, (Sum, Sub)):
            children = explore_operator(node)
            # First try to factorize within /node/'s children
            terms = [self._factorize(n, p) for n, p in children]
            # Then check if it's possible to aggregate operations
            # Example: replace (a*b)+(a*b) with 2*(a*b)
            self._simplify_sum(terms)
            # Finally try to factorize some of the operands composing the operation
            factorized = {}
            for t in terms:
                operand = set([t.operands_ast]) if t.operands else set()
                factor = set([t.factors_ast]) if t.factors else set()
                factorized_term = self.Term(operand, factor, node.__class__)
                _t = factorized.setdefault(str(t.operands_ast), factorized_term)
                _t.factors |= factor
            factorized = ast_make_expr(Sum, [t.generate_ast for t in factorized.values()])
            parent.children[parent.children.index(node)] = factorized
            return self.Term(set([factorized]))

        else:
            raise RuntimeError("Factorization error: unknown node: %s" % str(node))

    def factorize(self, should_factorize):
        self.should_factorize = should_factorize
        self._factorize(self.stmt.children[1], self.stmt)
