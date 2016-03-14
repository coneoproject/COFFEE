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

from collections import Counter
from warnings import warn as warning
import pulp as ilp

from base import *
from utils import *
from coffee.visitors import *
from hoister import Hoister
from expander import Expander
from factorizer import Factorizer
from cse import CSEUnpicker


class ExpressionRewriter():
    """Provide operations to re-write an expression:

    * Loop-invariant code motion: find and hoist sub-expressions which are
      invariant with respect to a loop
    * Expansion: transform an expression ``(a + b)*c`` into ``(a*c + b*c)``
    * Factorization: transform an expression ``a*b + a*c`` into ``a*(b+c)``"""

    def __init__(self, stmt, expr_info, decls, header=None, hoisted=None, expr_graph=None):
        """Initialize the ExpressionRewriter.

        :param stmt: the node whose rvalue is the expression for rewriting
        :param expr_info: ``MetaExpr`` object describing the expression
        :param decls: all declarations for the symbols in ``stmt``.
        :param header: the kernel's top node
        :param hoisted: dictionary that tracks all hoisted expressions
        :param expr_graph: a graph for data dependence analysis
        """
        self.stmt = stmt
        self.expr_info = expr_info
        self.decls = decls
        self.header = header or Root()
        self.hoisted = hoisted if hoisted is not None else StmtTracker()
        self.expr_graph = expr_graph or ExpressionGraph(self.header)

        self.expr_hoister = Hoister(self.stmt, self.expr_info, self.header,
                                    self.decls, self.hoisted, self.expr_graph)
        self.expr_expander = Expander(self.stmt, self.expr_info, self.decls,
                                      self.hoisted, self.expr_graph)
        self.expr_factorizer = Factorizer(self.stmt)

    def licm(self, mode='normal', **kwargs):
        """Perform generalized loop-invariant code motion, a transformation
        detailed in a paper available at:

            http://dl.acm.org/citation.cfm?id=2687415

        :param mode: drive code motion by specifying what subexpressions should
            be hoisted
            * normal: (default) all subexpressions that depend on one loop at most
            * aggressive: all subexpressions, depending on any number of loops.
                This may require introducing N-dimensional temporaries.
            * only_const: only all constant subexpressions
            * only_domain: only all domain-dependent subexpressions
            * only_outdomain: only all subexpressions independent of the domain loops
        :param kwargs:
            * look_ahead: (default: False) should be set to True if only a projection
                of the hoistable subexpressions is needed (i.e., hoisting not performed)
            * max_sharing: (default: False) should be set to True if hoisting should be
                avoided in case the same set of symbols appears in different hoistable
                sub-expressions. By not hoisting, factorization opportunities are preserved
            * iterative: (default: True) should be set to False if interested in
                hoisting only the smallest subexpressions matching /mode/
            * lda: an up-to-date loop dependence analysis, as returned by a call
                to ``ldanalysis(node, 'dim'). By providing this information, loop
                dependence analysis can be avoided, thus speeding up the transformation.
            * global_cse: (default: False) search for common sub-expressions across
                all previously hoisted terms. Note that no data dependency analysis is
                performed, so this is at caller's risk.
        """

        if kwargs.get('look_ahead'):
            return self.expr_hoister.extract(mode, **kwargs)
        if mode == 'aggressive':
            # Reassociation may promote more hoisting in /aggressive/ mode
            self.reassociate()
        self.expr_hoister.licm(mode, **kwargs)
        return self

    def expand(self, mode='standard', **kwargs):
        """Expand expressions based on different rules. For example: ::

            (X[i] + Y[j])*F + ...

        can be expanded into: ::

            (X[i]*F + Y[j]*F) + ...

        The expanded term could also be lifted. For example, if we have: ::

            Y[j] = f(...)
            (X[i]*Y[j])*F + ...

        where ``Y`` was produced by code motion, expansion results in: ::

            Y[j] = f(...)*F
            (X[i]*Y[j]) + ...

        Reasons for expanding expressions include:

        * Exposing factorization opportunities
        * Exposing higher level operations (e.g., matrix multiplies)
        * Relieving register pressure

        :param mode: multiple expansion strategies are possible
            * mode == 'standard': expand along the loop dimension appearing most
                often in different symbols
            * mode == 'dimensions': expand along the loop dimensions provided in
                /kwargs['dimensions']/
            * mode == 'all': expand when symbols depend on at least one of the
                expression's dimensions
            * mode == 'domain': expand when symbols depending on the expressions's
                domain are encountered.
            * mode == 'outdomain': expand when symbols independent of the
                expression's domain are encountered.
        :param kwargs:
            * not_aggregate: True if should not try to aggregate expanded symbols
                with previously hoisted expressions.
            * subexprs: an iterator of subexpressions rooted in /self.stmt/. If
                provided, expansion will be performed only within these trees,
                rather than within the whole expression.
            * lda: an up-to-date loop dependence analysis, as returned by a call
                to ``ldanalysis(node, 'symbol', 'dim'). By providing this information,
                loop dependence analysis can be avoided, thus speeding up the
                transformation.
        """

        if mode == 'standard':
            retval = FindInstances.default_retval()
            symbols = FindInstances(Symbol).visit(self.stmt.rvalue, ret=retval)[Symbol]
            # The heuristics privileges domain dimensions
            dims = self.expr_info.out_domain_dims
            if not dims or self.expr_info.dimension >= 2:
                dims = self.expr_info.domain_dims
            # Get the dimension occurring most often
            occurrences = [tuple(r for r in s.rank if r in dims) for s in symbols]
            occurrences = [i for i in occurrences if i]
            if not occurrences:
                return self
            # Finally, establish the expansion dimension
            dimension = Counter(occurrences).most_common(1)[0][0]
            should_expand = lambda n: set(dimension).issubset(set(n.rank))
        elif mode == 'dimensions':
            dimensions = kwargs.get('dimensions', ())
            should_expand = lambda n: set(dimensions).issubset(set(n.rank))
        elif mode in ['all', 'domain', 'outdomain']:
            lda = kwargs.get('lda') or ldanalysis(self.expr_info.outermost_loop,
                                                  key='symbol', value='dim')
            if mode == 'all':
                should_expand = lambda n: lda.get(n.symbol) and \
                    any(r in self.expr_info.dims for r in lda[n.symbol])
            elif mode == 'domain':
                should_expand = lambda n: lda.get(n.symbol) and \
                    any(r in self.expr_info.domain_dims for r in lda[n.symbol])
            elif mode == 'outdomain':
                should_expand = lambda n: lda.get(n.symbol) and \
                    not lda[n.symbol].issubset(set(self.expr_info.domain_dims))
        else:
            warning('Unknown expansion strategy. Skipping.')
            return

        self.expr_expander.expand(should_expand, **kwargs)
        return self

    def factorize(self, mode='standard', **kwargs):
        """Factorize terms in the expression. For example: ::

            A[i]*B[j] + A[i]*C[j]

        becomes ::

            A[i]*(B[j] + C[j]).

        :param mode: multiple factorization strategies are possible. Note that
                     different strategies may expose different code motion opportunities

            * mode == 'standard': factorize symbols along the dimension that appears
                most often in the expression.
            * mode == 'dimensions': factorize symbols along the loop dimensions provided
                in /kwargs['dimensions']/
            * mode == 'all': factorize symbols depending on at least one of the
                expression's dimensions.
            * mode == 'domain': factorize symbols depending on the expression's domain.
            * mode == 'outdomain': factorize symbols independent of the expression's
                domain.
            * mode == 'constants': factorize symbols independent of any loops enclosing
                the expression.
            * mode == 'adhoc': factorize only symbols in /kwargs['adhoc']/ (details below)
            * mode == 'heuristic': no global factorization rule is used; rather, within
                each Sum tree, factorize the symbols appearing most often in that tree
        :param kwargs:
            * subexprs: an iterator of subexpressions rooted in /self.stmt/. If
                provided, factorization will be performed only within these trees,
                rather than within the whole expression
            * adhoc: a list of symbols that can be factorized and, for each symbol,
                a list of symbols that can be grouped. For example, if we have
                ``kwargs['adhoc'] = [(A, [B, C]), (D, [E, F, G])]``, and the
                expression is ``A*B + D*E + A*C + A*F``, the result will be
                ``A*(B+C) + A*F + D*E``. If the A's list were empty, all of the
                three symbols B, C, and F would be factorized. Recall that this
                option is ignored unless ``mode == 'adhoc'``.
            * lda: an up-to-date loop dependence analysis, as returned by a call
                to ``ldanalysis(node, 'symbol', 'dim'). By providing this information,
                loop dependence analysis can be avoided, thus speeding up the
                transformation.
        """

        if mode == 'standard':
            retval = FindInstances.default_retval()
            symbols = FindInstances(Symbol).visit(self.stmt.rvalue, ret=retval)[Symbol]
            # The heuristics privileges domain dimensions
            dims = self.expr_info.out_domain_dims
            if not dims or self.expr_info.dimension >= 2:
                dims = self.expr_info.domain_dims
            # Get the dimension occurring most often
            occurrences = [tuple(r for r in s.rank if r in dims) for s in symbols]
            occurrences = [i for i in occurrences if i]
            if not occurrences:
                return self
            # Finally, establish the factorization dimension
            dimension = Counter(occurrences).most_common(1)[0][0]
            should_factorize = lambda n: set(dimension).issubset(set(n.rank))
        elif mode == 'dimensions':
            dimensions = kwargs.get('dimensions', ())
            should_factorize = lambda n: set(dimensions).issubset(set(n.rank))
        elif mode == 'adhoc':
            adhoc = kwargs.get('adhoc')
            if not adhoc:
                return self
            should_factorize = lambda n: n.urepr in adhoc
        elif mode == 'heuristic':
            kwargs['heuristic'] = True
            should_factorize = lambda n: False
        elif mode in ['all', 'domain', 'outdomain', 'constants']:
            lda = kwargs.get('lda') or ldanalysis(self.expr_info.outermost_loop,
                                                  key='symbol', value='dim')
            if mode == 'all':
                should_factorize = lambda n: lda.get(n.symbol) and \
                    any(r in self.expr_info.dims for r in lda[n.symbol])
            elif mode == 'domain':
                should_factorize = lambda n: lda.get(n.symbol) and \
                    any(r in self.expr_info.domain_dims for r in lda[n.symbol])
            elif mode == 'outdomain':
                should_factorize = lambda n: lda.get(n.symbol) and \
                    not lda[n.symbol].issubset(set(self.expr_info.domain_dims))
            elif mode == 'constants':
                should_factorize = lambda n: not lda.get(n.symbol)
        else:
            warning('Unknown factorization strategy. Skipping.')
            return

        # Perform the factorization
        self.expr_factorizer.factorize(should_factorize, **kwargs)
        return self

    def reassociate(self, reorder=None):
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
                symbols = [n for n, p in children if isinstance(n, Symbol)]
                # Capture the other children and recur on them
                other_nodes = [(n, p) for n, p in children if not isinstance(n, Symbol)]
                for n, p in other_nodes:
                    _reassociate(n, p)
                # Create the reassociated product and modify the original AST
                children = zip(*other_nodes)[0] if other_nodes else ()
                children += tuple(sorted(symbols, key=reorder))
                reassociated_node = ast_make_expr(Prod, children, balance=False)
                parent.children[parent.children.index(node)] = reassociated_node

            else:
                warning('Unexpect node of type %s while reassociating', typ(node))

        reorder = reorder if reorder else lambda n: (n.rank, n.dim)
        _reassociate(self.stmt.rvalue, self.stmt)
        return self

    def replacediv(self):
        """Replace divisions by a constant with multiplications."""
        retval = FindInstances.default_retval()
        divisions = FindInstances(Div).visit(self.stmt.rvalue, ret=retval)[Div]
        to_replace = {}
        for i in divisions:
            if isinstance(i.right, Symbol):
                if isinstance(i.right.symbol, (int, float)):
                    to_replace[i] = Prod(i.left, 1.0 / i.right.symbol)
                elif isinstance(i.right.symbol, str) and i.right.symbol.isdigit():
                    to_replace[i] = Prod(i.left, 1.0 / float(i.right.symbol))
                else:
                    to_replace[i] = Prod(i.left, Div(1.0, i.right))
        ast_replace(self.stmt, to_replace, copy=True, mode='symbol')
        return self

    def preevaluate(self):
        """Preevaluates subexpressions which values are compile-time constants.
        In this process, reduction loops might be removed if the reduction itself
        could be pre-evaluated."""
        # Aliases
        stmt, expr_info = self.stmt, self.expr_info

        # Simplify reduction loops
        if not isinstance(stmt, (Incr, Decr, IMul, IDiv)):
            # Not a reduction expression, give up
            return
        retval = FindInstances.default_retval()
        expr_syms = FindInstances(Symbol).visit(stmt.rvalue, ret=retval)[Symbol]
        reduction_loops = expr_info.out_domain_loops_info
        if any([not is_perfect_loop(l) for l, p in reduction_loops]):
            # Unsafe if not a perfect loop nest
            return
        # The following check is because it is unsafe to simplify if non-loop or
        # non-constant dimensions are present
        hoisted_stmts = self.hoisted.all_stmts
        hoisted_syms = [FindInstances(Symbol).visit(h)[Symbol] for h in hoisted_stmts]
        hoisted_dims = [s.rank for s in flatten(hoisted_syms)]
        hoisted_dims = set([r for r in flatten(hoisted_dims) if not is_const_dim(r)])
        if any(d not in expr_info.dims for d in hoisted_dims):
            # Non-loop dimension or non-constant dimension found, e.g. A[i], with /i/
            # not being a loop iteration variable
            return
        for i, (l, p) in enumerate(reduction_loops):
            retval = SymbolDependencies.default_retval()
            syms_dep = SymbolDependencies().visit(l, ret=retval,
                                                  **SymbolDependencies.default_args)
            if not all([tuple(syms_dep[s]) == expr_info.loops and
                        s.dim == len(expr_info.loops) for s in expr_syms if syms_dep[s]]):
                # A sufficient (although not necessary) condition for loop reduction to
                # be safe is that all symbols in the expression are either constants or
                # tensors assuming a distinct value in each point of the iteration space.
                # So if this condition fails, we give up
                return
            # At this point, tensors can be reduced along the reducible dimensions
            reducible_syms = [s for s in expr_syms if not s.is_const]
            # All involved symbols must result from hoisting
            if not all([s.symbol in self.hoisted for s in reducible_syms]):
                return
            # Replace hoisted assignments with reductions
            finder = FindInstances(Assign, stop_when_found=True, with_parent=True)
            for hoisted_loop in self.hoisted.all_loops:
                retval = FindInstances.default_retval()
                for assign, parent in finder.visit(hoisted_loop, ret=retval)[Assign]:
                    sym, expr = assign.children
                    decl = self.hoisted[sym.symbol].decl
                    if sym.symbol in [s.symbol for s in reducible_syms]:
                        parent.children[parent.children.index(assign)] = Incr(sym, expr)
                        sym.rank = self.expr_info.domain_dims
                        decl.sym.rank = decl.sym.rank[i+1:]
            # Remove the reduction loop
            p.children[p.children.index(l)] = l.body[0]
            # Update symbols' ranks
            for s in reducible_syms:
                s.rank = self.expr_info.domain_dims
            # Update expression metadata
            self.expr_info._loops_info.remove((l, p))

        # Precompute constant expressions
        evaluator = Evaluate(self.decls, any(d.nonzero for s, d in self.decls.items()))
        for hoisted_loop in self.hoisted.all_loops:
            evals = evaluator.visit(hoisted_loop, **Evaluate.default_args)
            # First, find out identical tables
            mapper = defaultdict(list)
            for s, values in evals.items():
                mapper[str(values)].append(s)
            # Then, map identical tables to a single symbol
            for values, symbols in mapper.items():
                to_replace = {s: symbols[0] for s in symbols[1:]}
                ast_replace(self.stmt, to_replace, copy=True)
                # Clean up
                for s in symbols[1:]:
                    s_decl = self.hoisted[s.symbol].decl
                    self.header.children.remove(s_decl)
                    self.hoisted.pop(s.symbol)
                    evals.pop(s)
            # Finally, update the hoisted symbols
            for s, values in evals.items():
                hoisted = self.hoisted[s.symbol]
                hoisted.decl.init = values
                hoisted.decl.qual = ['static', 'const']
                self.hoisted.pop(s.symbol)
                # Move all decls at the top of the kernel
                self.header.children.remove(hoisted.decl)
                self.header.children.insert(0, hoisted.decl)
            self.header.children.insert(0, FlatBlock("// Preevaluated tables"))
            # Clean up
            self.header.children.remove(hoisted_loop)
        return self

    def SGrewrite(self):
        """Rewrite the expression based on its sharing graph. Details in the
        paper:

            On Optimality of Finite Element Integration, Luporini et. al.
        """
        lda = ldanalysis(self.expr_info.domain_loops[0], key='symbol', value='dim')
        sg_visitor = SharingGraph(self.expr_info, lda)

        # First, eliminate sharing "locally", i.e., within Sums
        sgraph, mapper = sg_visitor.visit(self.stmt.rvalue)
        handled = set()
        for n in sgraph.nodes():
            mapped = mapper.get((n,), [])
            with_sharing = [e for e in mapped if e and e not in handled]
            for e in with_sharing:
                self.expand(mode='dimensions', subexprs=[e],
                            not_aggregate=True, dimensions=n[1])
                self.factorize(mode='dimensions', subexprs=[e], dimensions=n[1])
                handled.add(e)
        self.factorize(mode='heuristic')
        self.licm(mode='only_outdomain')

        # Maximize the visibility of linear symbols
        sgraph, mapper = sg_visitor.visit(self.stmt.rvalue)
        if 'topsum' in mapper:
            self.expand(mode='domain', subexprs=[mapper['topsum']], not_aggregate=True)
            sgraph, mapper = sg_visitor.visit(self.stmt.rvalue)

        nodes, edges = sgraph.nodes(), sgraph.edges()

        if self.expr_info.dimension == 1:
            self.factorize(mode='adhoc', adhoc={n: [] for n in nodes})
            self.licm('only_outdomain')
        elif self.expr_info.dimension == 2:
            # Resort to an ILP formulation to find out the best factorization candidates
            if not all(sgraph.degree(n) > 0 for n in nodes):
                return
            # Note: need to use short variable names otherwise Pulp might complain
            nodes_vars = {i: n for i, n in enumerate(nodes)}
            vars_nodes = {n: i for i, n in nodes_vars.items()}
            edges = [(vars_nodes[i], vars_nodes[j]) for i, j in edges]

            # ... declare variables
            x = ilp.LpVariable.dicts('x', nodes_vars.keys(), 0, 1, ilp.LpBinary)
            y = ilp.LpVariable.dicts('y', [(i, j) for i, j in edges] + [(j, i) for i, j in edges],
                                     0, 1, ilp.LpBinary)
            limits = defaultdict(int)
            for i, j in edges:
                limits[i] += 1
                limits[j] += 1

            # ... define the problem
            prob = ilp.LpProblem("Factorizer", ilp.LpMinimize)

            # ... define the constraints
            for i in nodes_vars:
                prob += ilp.lpSum(y[(i, j)] for j in nodes_vars if (i, j) in y) <= limits[i]*x[i]

            for i, j in edges:
                prob += y[(i, j)] + y[(j, i)] == 1

            # ... define the objective function (min number of factorizations)
            prob += ilp.lpSum(x[i] for i in nodes_vars)

            # ... solve the problem
            prob.solve(ilp.GLPK(msg=0))

            # Finally, factorize and hoist (note: the order in which factorizations are carried
            # out is crucial)
            nodes = [nodes_vars[n] for n, v in x.items() if v.value() == 1]
            other_nodes = [nodes_vars[n] for n, v in x.items() if nodes_vars[n] not in nodes]
            for n in nodes + other_nodes:
                self.factorize(mode='adhoc', adhoc={n: []})
            self.licm()

    def unpickCSE(self):
        """Search for factorization opportunities across temporaries created by
        common sub-expression elimination. If a gain in operation count is detected,
        unpick CSE and apply factorization + code motion."""
        cse_unpicker = CSEUnpicker(self.stmt, self.expr_info, self.header, self.hoisted,
                                   self.decls, self.expr_graph)
        cse_unpicker.unpick()

    @staticmethod
    def reset():
        Hoister._handled = 0
        Expander._handled = 0
