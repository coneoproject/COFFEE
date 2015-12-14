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

import operator
import resource
from collections import OrderedDict
from itertools import combinations
from warnings import warn as warning
from math import factorial as fact

import plan
from base import *
from utils import *
from expression import MetaExpr
from loop_scheduler import ExpressionFissioner, ZeroRemover, SSALoopMerger
from linear_algebra import LinearAlgebra
from rewriter import ExpressionRewriter
from ast_analyzer import ExpressionGraph, StmtTracker
from coffee.visitors import MaxLoopDepth, FindInstances, ProjectExpansion


class LoopOptimizer(object):

    def __init__(self, loop, header, decls, exprs):
        """Initialize the LoopOptimizer.

        :param loop: root AST node of a loop nest
        :param header: parent AST node of ``loop``
        :param decls: list of Decl objects accessible in ``loop``
        :param exprs: list of expressions to be optimized
        """
        self.loop = loop
        self.header = header
        self.decls = decls
        self.exprs = exprs

        # Track nonzero regions accessed in each symbol
        self.nz_syms = {}
        # Track data dependencies
        self.expr_graph = ExpressionGraph(header)
        # Track hoisted expressions
        self.hoisted = StmtTracker()

    def rewrite(self, mode):
        """Rewrite a compute-intensive expression found in the loop nest so as to
        minimize floating point operations and to relieve register pressure.
        This involves several possible transformations:

        1. Generalized loop-invariant code motion
        2. Factorization of common loop-dependent terms
        3. Expansion of constants over loop-dependent terms

        :param mode: Any value in (0, 1, 2, 3). Each ``mode`` corresponds to a
            different expression rewriting strategy. A strategy impacts the aspects:
            amount of floating point calculations required to evaluate the expression;
            amount of temporary storage introduced; accuracy of the result.

            * mode == 0: No rewriting is performed
            * mode == 1: Apply one pass: generalized loop-invariant code motion.
                Safest: accuracy not affected
            * mode == 2: Apply four passes: generalized loop-invariant code motion;
                expansion of inner-loop dependent expressions; factorization of
                inner-loop dependent terms; generalized loop-invariant code motion.
                Factorization may affect accuracy. Improves performance while trying to
                minimize temporary storage
            * mode == 3: Apply nine passes: Expansion of inner-loops dependent terms.
                Factorization of inner-loops dependent terms; generalized loop-invariant
                code motion of outer-loop dependent terms. Factorization of constants.
                Reassociation, followed by 'aggressive' generalized loop-invariant code
                motion (in which n-rank temporary arrays can be allocated to host
                expressions independent of more than one loop). Simplification is the
                key pass: it tries to remove reduction loops and precompute constant
                expressions. Finally, two last sweeps of factorization and code motion
                are applied.
        """
        ExpressionRewriter.reset()

        # Set a rewrite mode for each expression
        for stmt, expr_info in self.exprs.items():
            expr_info.mode = mode

        # Analyze the individual expressions and try to select an optimal rewrite
        # mode for each of them. A preliminary transformation of the loop nest may
        # take place in this pass (e.g., injection)
        if mode == 'auto':
            self._dissect('greedy')
        elif mode == 'auto-aggressive':
            self._dissect('aggressive')

        # Expression rewriting, expressed as a sequence of AST transformation passes
        for stmt, expr_info in self.exprs.items():
            ew = ExpressionRewriter(stmt, expr_info, self.decls, self.header,
                                    self.hoisted, self.expr_graph)

            if expr_info.mode == 1:
                if expr_info.dimension in [0, 1]:
                    ew.licm(only_outdomain=True)
                else:
                    ew.licm()

            elif expr_info.mode == 2:
                if expr_info.dimension == 0:
                    ew.licm(only_outdomain=True)
                elif expr_info.dimension == 1:
                    ew.licm(only_outdomain=True)
                    ew.expand(mode='domain')
                    ew.factorize(mode='domain')
                    ew.licm(only_outdomain=True)
                else:
                    ew.licm()
                    ew.expand()
                    ew.factorize()
                    ew.licm()

            elif expr_info.mode == 3:
                ew.expand(mode='all')
                ew.factorize(mode='all')
                ew.licm(only_const=True)
                ew.factorize(mode='constants')
                ew.licm(only_domain=True)
                ew.preevaluate()
                ew.factorize(mode='domain')
                ew.licm(only_const=True)

            elif expr_info.mode == 4:
                ew.replacediv()
                ew.licm(only_outdomain=True)
                ew.expand(mode='domain')
                ew.factorize(mode='domain')
                ew.licm(only_outdomain=True)
                ew.licm(only_const=True)
                for i in range(1, expr_info.dimension):
                    ew.factorize()
                    ew.licm()

            elif expr_info.mode == 5:
                ew.replacediv()
                ew.expand(mode='all')
                ew.factorize(mode='domain')
                ew.licm(only_const=True)
                ew.factorize(mode='domain')
                ew.licm(only_const=True)
                for i in range(1, expr_info.dimension):
                    ew.factorize()
                    ew.licm()

        # Try merging the loops created by expression rewriting
        merged_loops = SSALoopMerger(self.expr_graph).merge(self.header)
        # Update the trackers
        for merged, merged_in in merged_loops:
            for l in merged:
                self.hoisted.update_loop(l, merged_in)
                # Was /merged/ an expression loops? If so, need to update the
                # corresponding MetaExpr
                for stmt, expr_info in self.exprs.items():
                    if expr_info.loops[-1] == l:
                        expr_info._loops_info[-1] = (merged_in, expr_info.loops_parents[-1])
                        expr_info._parent = merged_in.children[0]

        # Reduce memory pressure by rearranging operations ...
        self._rearrange()
        # ... plus, at max rewrite level, apply rewriting to hoisted subexprs too ...
        if mode == 'auto':
            self._rewrite_hoisted()
        # ... which in turn require updating the expression graph
        self.expr_graph = ExpressionGraph(self.header)

        # Handle the effects, at the C-level, of the AST transformation
        self._recoil()

    def eliminate_zeros(self):
        """Restructure the iteration spaces nested in this LoopOptimizer to
        avoid evaluation of arithmetic operations involving zero-valued blocks
        in statically initialized arrays."""

        if any([d.nonzero for d in self.decls.values()]):
            zls = ZeroRemover(self.exprs, self.decls, self.hoisted)
            self.nz_syms = zls.reschedule(self.header)

    def precompute(self, mode='perfect'):
        """Precompute statements out of ``self.loop``. This is achieved through
        scalar code hoisting.

        :arg mode: drives the precomputation. Two values are possible: ['perfect',
        'noloops']. The 'perfect' mode attempts to hoist everything, making the loop
        nest perfect. The 'noloops' mode excludes inner loops from the precomputation.

        Example: ::

        for i
          for r
            A[r] += f(i, ...)
          for j
            for k
              B[j][k] += g(A[r], ...)

        with mode='perfect', becomes: ::

        for i
          for r
            A[i][r] += f(...)
        for i
          for j
            for k
              B[j][k] += g(A[i][r], ...)
        """

        precomputed_block = []
        precomputed_syms = {}

        def _precompute(node, outer_block):

            if isinstance(node, Symbol):
                if node.symbol in precomputed_syms:
                    node.rank = precomputed_syms[node.symbol] + node.rank

            elif isinstance(node, FlatBlock):
                outer_block.append(node)

            elif isinstance(node, Expr):
                for n in node.children:
                    _precompute(n, outer_block)

            elif isinstance(node, Writer):
                sym, expr = node.children
                precomputed_syms[sym.symbol] = (self.loop.dim,)
                _precompute(sym, outer_block)
                _precompute(expr, outer_block)
                outer_block.append(node)

            elif isinstance(node, Decl):
                outer_block.append(node)
                if isinstance(node.init, Symbol):
                    node.init.symbol = "{%s}" % node.init.symbol
                elif isinstance(node.init, Expr):
                    _precompute(Assign(dcopy(node.sym), node.init), outer_block)
                    node.init = EmptyStatement()
                node.sym.rank = (self.loop.size,) + node.sym.rank

            elif isinstance(node, For):
                new_children = []
                for n in node.body:
                    _precompute(n, new_children)
                node.body = new_children
                outer_block.append(node)

            else:
                raise RuntimeError("Precompute error: unexpteced node: %s" % str(node))

        # If the outermost loop is already perfect, there is nothing to precompute
        if is_perfect_loop(self.loop):
            return

        # Get the nodes that should not be precomputed
        no_precompute = set()
        if mode == 'noloops':
            for l in self.hoisted.values():
                if l.loop:
                    no_precompute.add(l.decl)
                    no_precompute.add(l.loop)

        # Visit the AST and perform the precomputation
        to_remove = []
        for n in self.loop.body:
            if n in flatten(self.expr_domain_loops):
                break
            elif n not in no_precompute:
                _precompute(n, precomputed_block)
                to_remove.append(n)

        # Clean up
        for n in to_remove:
            self.loop.body.remove(n)

        # Wrap precomputed statements within a loop
        searching, outer_block = [], []
        for n in precomputed_block:
            if searching and not isinstance(n, Writer):
                outer_block.append(ast_make_for(searching, self.loop))
                searching = []
            if isinstance(n, For):
                outer_block.append(ast_make_for([n], self.loop))
            elif isinstance(n, Writer):
                searching.append(n)
            else:
                outer_block.append(n)
        if searching:
            outer_block.append(ast_make_for(searching, self.loop))

        # Update the AST ...
        # ... adding the newly precomputed blocks
        insert_at_elem(self.header.children, self.loop, outer_block)
        # ... scalar-expanding the precomputed symbols
        ast_update_rank(self.loop, precomputed_syms)

    def _rearrange(self):
        """Relieve the memory pressure by removing temporaries read by only
        one statement."""

        in_stmt = [count(s, mode='symbol_id', read_only=True) for s in self.exprs.keys()]
        stmt_occs = dict((k, v) for d in in_stmt for k, v in d.items())

        for l in self.hoisted.all_loops:
            info = visit(l)
            occurrences = count(l, read_only=True)
            innermost_block = FindInstances(Block).visit(l)[Block][-1]
            to_replace, to_remove = {}, []
            for (symbol, rank), sym_occurrences in occurrences.items():
                # A temporary /symbol/ is removed if any of the following
                # conditions hold:
                # - it is read once in /l/ and it is not read any longer (this
                #   is checked through /stmt_occs/)
                # - it is read one or more times in /l/, but it actually hosts a
                #   symbol (this is checked through /sym_occurrences/)
                if symbol not in self.hoisted or symbol in stmt_occs:
                    continue
                if self.hoisted[symbol].loop is not l:
                    continue
                decl = self.hoisted[symbol].decl
                place = self.hoisted[symbol].place
                expr = self.hoisted[symbol].stmt.children[1]
                if sym_occurrences > 1 and explore_operator(expr):
                    continue

                symbol_refs = info['symbol_refs'][symbol]
                syms_mode = info['symbols_mode']
                # Note: only one write is possible at this point
                write = [(s, p) for s, p in symbol_refs if syms_mode[s][0] == WRITE][0]
                to_replace[write[0]] = expr
                to_remove.append(write[1])
                place.children.remove(decl)
                # Update trackers
                self.hoisted.pop(symbol)
                self.decls.pop(symbol)

            # Perform replacement of selected symbols
            for stmt in innermost_block.children:
                ast_replace(stmt.children[1], to_replace, copy=True)

            # Clean up
            for stmt in to_remove:
                innermost_block.children.remove(stmt)

    def _dissect(self, heuristics):
        """Analyze the set of expressions in the LoopOptimizer and infer an
        optimal rewrite mode for each of them.

        If an expression is embedded in a non-perfect loop nest, then injection
        may be performed. Injection consists of unrolling any loops outside of
        the expression iteration space into the expression itself.
        For example: ::

            for i
              for r
                a += B[r]*C[i][r]
              for j
                for k
                  A[j][k] += ...f(a)... // the expression at hand

        gets transformed into:

            for i
              for j
                for k
                  A[j][k] += ...f(B[0]*C[i][0] + B[1]*C[i][1] + ...)...

        Injection could be necessary to maximize the impact of rewrite mode=3,
        which tries to pre-evaluate subexpressions whose values are known at
        code generation time. Injection is essential to factorize such subexprs.

        :arg heuristic: any value in ['greedy', 'aggressive']. With 'greedy', a greedy
            approach is used to decide which of the expressions for which injection
            looks beneficial should be dissected (e.g., injection increases the memory
            footprint, and some memory constraints must always be preserved).
            With 'aggressive', the whole space of possibilities is analyzed.
        """
        # The memory threshold. The total size of temporaries will not have to
        # be greated than this value. If we predict that injection will lead
        # to too much temporary space, we have to partially drop it
        threshold = plan.arch['cache_size'] * 1.2

        # 1) Find out and unroll injectable loops. For unrolling we create new
        # expressions; that is, for now, we do not modify the AST in place.
        analyzed, injectable = [], {}
        for stmt, expr_info in self.exprs.items():
            # Get all loop nests, then discard the one enclosing the expression
            nests = [n for n in visit(expr_info.loops_parents[0])['fors']]
            injectable_nests = [n for n in nests if zip(*n)[0] != expr_info.loops]

            for nest in injectable_nests:
                to_unroll = [(l, p) for l, p in nest if l not in expr_info.loops]
                unroll_cost = reduce(operator.mul, (l.size for l, p in to_unroll))

                nest_writers = FindInstances(Writer).visit(to_unroll[0][0])
                for op, i_stmts in nest_writers.items():
                    # Check safety of unrolling
                    if op in [Assign, IMul, IDiv]:
                        continue
                    assert op in [Incr, Decr]

                    for i_stmt in i_stmts:
                        i_sym, i_expr = i_stmt.children

                        # Avoid injecting twice the same loop
                        if i_stmt in analyzed + [l.incr for l, p in to_unroll]:
                            continue
                        analyzed.append(i_stmt)

                        # Create unrolled, injectable expressions
                        for l, p in reversed(to_unroll):
                            i_expr = [dcopy(i_expr) for i in range(l.size)]
                            for i, e in enumerate(i_expr):
                                e_syms = FindInstances(Symbol).visit(e)[Symbol]
                                for s in e_syms:
                                    s.rank = tuple([r if r != l.dim else i for r in s.rank])
                            i_expr = ast_make_expr(Sum, i_expr)

                        # Track the unrolled, injectable expressions and their cost
                        if i_sym.symbol in injectable:
                            old_i_expr, old_cost = injectable[i_sym.symbol]
                            new_i_expr = ast_make_expr(Sum, [i_expr, old_i_expr])
                            new_cost = unroll_cost + old_cost
                            injectable[i_sym.symbol] = (new_i_expr, new_cost)
                        else:
                            injectable[i_sym.symbol] = (i_expr, unroll_cost)

        # 2) Will rewrite mode=3 be cheaper than rewrite mode=2?
        def find_save(target_expr, expr_info):
            save_factor = [l.size for l in expr_info.out_domain_loops] or [1]
            save_factor = reduce(operator.mul, save_factor)
            # The save factor should be multiplied by the number of terms
            # that will /not/ be pre-evaluated. To obtain this number, we
            # can exploit the linearity of the expression in the terms
            # depending on the domain loops.
            syms = FindInstances(Symbol).visit(target_expr)[Symbol]
            inner = lambda s: any(r == expr_info.domain_dims[-1] for r in s.rank)
            nterms = len(set(s.symbol for s in syms if inner(s)))
            save = nterms * save_factor
            return save_factor, save

        should_unroll = True
        storage = 0
        i_syms, injected = injectable.keys(), defaultdict(list)
        for stmt, expr_info in self.exprs.items():
            sym, expr = stmt.children

            # Divide /expr/ into subexpressions, each subexpression affected
            # differently by injection
            if i_syms:
                dissected = find_expression(expr, Prod, expr_info.domain_dims, i_syms)
                leftover = find_expression(expr, dims=expr_info.domain_dims, out_syms=i_syms)
                leftover = {(): list(flatten(leftover.values()))}
                dissected = dict(dissected.items() + leftover.items())
            else:
                dissected = {(): [expr]}
            if any(i not in flatten(dissected.keys()) for i in i_syms):
                should_unroll = False
                continue

            # Apply the profitability model
            analysis = OrderedDict()
            for i_syms, target_exprs in dissected.items():
                for target_expr in target_exprs:

                    # *** Save ***
                    save_factor, save = find_save(target_expr, expr_info)

                    # *** Cost ***
                    # The number of operations increases by a factor which
                    # corresponds to the number of possible /combinations with
                    # repetitions/ in the injected-values set. We consider
                    # combinations and not dispositions to take into account the
                    # (future) effect of factorization.
                    retval = ProjectExpansion.default_retval()
                    projection = ProjectExpansion(i_syms).visit(target_expr, ret=retval)
                    projection = [i for i in projection if i]
                    increase_factor = 0
                    for i in projection:
                        partial = 1
                        for j in self.expr_graph.shares(i):
                            # _n=number of unique elements, _k=group size
                            _n = injectable[j[0]][1]
                            _k = len(j)
                            partial *= fact(_n + _k - 1)/(fact(_k)*fact(_n - 1))
                        increase_factor += partial
                    increase_factor = increase_factor or 1
                    if increase_factor > save_factor:
                        # We immediately give up if this holds since it ensures
                        # that /cost > save/ (but not that cost <= save)
                        should_unroll = False
                        continue
                    # The increase factor should be multiplied by the number of
                    # terms that will be pre-evaluated. To obtain this number,
                    # we need to project the output of factorization.
                    fake_stmt = stmt.__class__(stmt.children[0], dcopy(target_expr))
                    fake_parent = expr_info.parent.children
                    fake_parent[fake_parent.index(stmt)] = fake_stmt
                    ew = ExpressionRewriter(fake_stmt, expr_info, self.decls)
                    ew.expand(mode='all').factorize(mode='all').factorize(mode='domain')
                    nterms = ew.licm(look_ahead=True, only_domain=True)
                    nterms = len(uniquify(nterms[expr_info.dims])) or 1
                    fake_parent[fake_parent.index(fake_stmt)] = stmt
                    cost = nterms * increase_factor

                    # Pre-evaluation will also increase the working set size by
                    # /cost/ * /sizeof(term)/.
                    size = [l.size for l in expr_info.domain_loops]
                    size = reduce(operator.mul, size, 1)
                    storage_increase = cost * size * plan.arch[expr_info.type]

                    # Track the injectable sub-expression and its cost/save. The
                    # final decision of whether to actually perform injection or not
                    # is postponed until all dissected expressions have been analyzed
                    analysis[target_expr] = (cost, save, storage_increase)

            # So what should we inject afterall ? Time to *use* the cost model
            if heuristics == 'greedy':
                for target_expr, (cost, save, storage_increase) in analysis.items():
                    if cost > save or storage_increase + storage > threshold:
                        should_unroll = False
                    else:
                        # Update the available storage
                        storage += storage_increase
                        # At this point, we can happily inject
                        to_replace = {k: v[0] for k, v in injectable.items()}
                        ast_replace(target_expr, to_replace, copy=True)
                        injected[stmt].append(target_expr)
            elif heuristics == 'aggressive':
                # A) Remove expression that we already know should never be injected
                not_injected = []
                for target_expr, (cost, save, storage_increase) in analysis.items():
                    if cost > save:
                        should_unroll = False
                        analysis.pop(target_expr)
                        not_injected.append(target_expr)
                # B) Find all possible bipartitions: each bipartition represents
                # the set of expressions that will be pre-evaluated and the set
                # of expressions that could also be pre-evaluated, but might not
                # (e.g. because of memory constraints)
                target_exprs = analysis.keys()
                bipartitions = []
                for i in range(len(target_exprs)+1):
                    for e1 in combinations(target_exprs, i):
                        bipartitions.append((e1, tuple(e2 for e2 in target_exprs
                                                       if e2 not in e1)))
                # C) Eliminate those bipartitions that would lead to exceeding
                # the memory threshold
                bipartitions = [(e1, e2) for e1, e2 in bipartitions if
                                sum(analysis[i][2] for i in e1) <= threshold]
                # D) Find out what is best to pre-evaluate (and therefore
                # what should be injected)
                totals = OrderedDict()
                for e1, e2 in bipartitions:
                    # Is there any value in actually not pre-evaluating the
                    # expressions in /e2/ ?
                    fake_expr = ast_make_expr(Sum, list(e2) + not_injected)
                    _, save = find_save(fake_expr, expr_info) if fake_expr else (0, 0)
                    cost = sum(analysis[i][0] for i in e1)
                    totals[(e1, e2)] = save + cost
                best = min(totals, key=totals.get)
                # At this point, we can happily inject
                to_replace = {k: v[0] for k, v in injectable.items()}
                for target_expr in best[0]:
                    ast_replace(target_expr, to_replace, copy=True)
                    injected[stmt].append(target_expr)
                if best[1]:
                    # At least one non-injected expressions, let's be sure we
                    # don't unroll everything
                    should_unroll = False

        # 3) Purge the AST from now useless symbols/expressions
        if should_unroll:
            for stmt, expr_info in self.exprs.items():
                nests = [n for n in visit(expr_info.loops_parents[0])['fors']]
                injectable_nests = [n for n in nests if zip(*n)[0] != expr_info.loops]
                for nest in injectable_nests:
                    unrolled = [(l, p) for l, p in nest if l not in expr_info.loops]
                    for l, p in unrolled:
                        p.children.remove(l)
                        for i_sym in injectable.keys():
                            decl = self.decls.get(i_sym)
                            if decl and decl in p.children:
                                p.children.remove(decl)
                                self.decls.pop(i_sym)

        # 4) Split the expressions if injection has been performed
        for stmt, expr_info in self.exprs.items():
            expr_info.mode = 2
            inj_exprs = injected.get(stmt)
            if not inj_exprs:
                continue
            fissioner = ExpressionFissioner(match=inj_exprs, loops='all', perfect=True)
            new_exprs = fissioner.fission(stmt, self.exprs.pop(stmt))
            self.exprs.update(new_exprs)
            for stmt, expr_info in new_exprs.items():
                expr_info.mode = 3 if stmt in fissioner.matched else 2

    def _rewrite_hoisted(self):
        """Rewrite hoisted expressions."""

        retval = FindLoopNests.default_retval()
        nests = FindLoopNests().visit(self.header, parent=None, ret=retval)
        hoisted_exprs = OrderedDict()

        # First, need to create a /MetaExpr/ for each hoisted expression
        type = self.exprs.values()[0].type
        for s, hoisted in self.hoisted.items():
            stmt, loop, place = hoisted.stmt, hoisted.loop, hoisted.place
            if not loop and place == self.header:
                # Outside of any loops, so ignoring it
                continue
            elif not loop:
                # Need to find the loop (an outermost loop in a nest)
                # within which the statement lives
                for nest in nests:
                    loops_info = [(l, p) for l, p in nest if stmt in l.body]
                    if loops_info:
                        break
                domain = tuple(l.dim for l, p in loops_info)
            else:
                # Need to find the loop nest in which the statement lives
                for nest in nests:
                    if any(l == loop for l, p in nest):
                        break
                loops_info = nest
                domain = tuple(l.dim for l, p in loops_info
                               if l.dim in stmt.children[0].rank)
            parent = loops_info[-1][0].children[0]
            hoisted_exprs[stmt] = MetaExpr(type, parent, loops_info, domain)

        # Now we do the usual: for each expression, we create an ExpressionRewriter
        # and use it to manipulate the expression itself
        for stmt, expr_info in hoisted_exprs.items():
            ew = ExpressionRewriter(stmt, expr_info, self.decls, self.header,
                                    self.hoisted, self.expr_graph)
            ew.expand(mode='domain', not_aggregate=True)
            ew.factorize(mode='domain')
            ew.licm(only_outdomain=True)
            ew.factorize().factorize('constants')

    def _recoil(self):
        """Increase the stack size if the kernel arrays exceed the stack limit
        threshold (at the C level)."""

        # Assume the size of a C type double is 8 bytes
        c_double_size = 8
        # Assume the stack size is 1.7 MB (2 MB is usually the limit)
        stack_size = 1.7*1024*1024

        decls = [d for d in self.decls.values() if d.sym.rank]
        size = sum([reduce(operator.mul, d.sym.rank) for d in decls])

        if size * c_double_size > stack_size:
            # Increase the stack size if the kernel's stack size seems to outreach
            # the space available
            try:
                resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,
                                                           resource.RLIM_INFINITY))
            except resource.error:
                warning("Stack may blow up, and could not increase its size.")

    @property
    def expr_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        loops enclosing expressions."""
        return [expr_info.loops for expr_info in self.exprs.values()]

    @property
    def expr_domain_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where a tuple contains all
        loops representing the domain of the expressions' output tensor."""
        return [expr_info.domain_loops for expr_info in self.exprs.values()]


class CPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for CPU architectures."""

    def unroll(self, loop_uf):
        """Unroll loops enclosing expressions as specified by ``loop_uf``.

        :param loop_uf: dictionary from iteration spaces to unroll factors."""

        def update_expr(node, var, factor):
            """Add an offset ``factor`` to every iteration variable ``var`` in
            ``node``."""
            if isinstance(node, Symbol):
                new_ofs = []
                node.offset = node.offset or ((1, 0) for i in range(len(node.rank)))
                for r, ofs in zip(node.rank, node.offset):
                    new_ofs.append((ofs[0], ofs[1] + factor) if r == var else ofs)
                node.offset = tuple(new_ofs)
            else:
                for n in node.children:
                    update_expr(n, var, factor)

        unrolled_loops = set()
        for itspace, uf in loop_uf.items():
            new_exprs = OrderedDict()
            for stmt, expr_info in self.exprs.items():
                loop = [l for l in expr_info.perfect_loops if l.dim == itspace]
                if not loop:
                    # Unroll only loops in a perfect loop nest
                    continue
                loop = loop[0]  # Only one loop possibly found
                for i in range(uf-1):
                    new_stmt = dcopy(stmt)
                    update_expr(new_stmt, itspace, i+1)
                    expr_info.parent.children.append(new_stmt)
                    new_exprs.update({new_stmt: expr_info})
                if loop not in unrolled_loops:
                    loop.incr.children[1].symbol += uf-1
                    unrolled_loops.add(loop)
            self.exprs.update(new_exprs)

    def split(self, cut=1):
        """Split expressions into multiple chunks exploiting sum's associativity.
        Each chunk will have ``cut`` summands.

        For example, consider the following piece of code: ::

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j] + B[i]*X[j]

        If ``cut=1`` the expression is cut into chunks of length 1: ::

            for i
              for j
                A[i][j] += X[i]*Y[j]
            for i
              for j
                A[i][j] += Z[i]*K[j]
            for i
              for j
                A[i][j] += B[i]*X[j]

        If ``cut=2`` the expression is cut into chunks of length 2, plus a
        remainder chunk of size 1: ::

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j]
            # Remainder:
            for i
              for j
                A[i][j] += B[i]*X[j]
        """

        new_exprs = OrderedDict()
        elf = ExpressionFissioner(cut=cut, loops='expr')
        for stmt, expr_info in self.exprs.items():
            new_exprs.update(elf.fission(stmt, expr_info))
        self.exprs = new_exprs

    def blas(self, library):
        """Convert an expression into sequences of calls to external dense linear
        algebra libraries. Currently, MKL, ATLAS, and EIGEN are supported."""

        # First, check that the loop nest has depth 3, otherwise it's useless
        if MaxLoopDepth().visit(self.loop) != 3:
            return

        linear_algebra = LinearAlgebra(self.loop, self.header, self.kernel_decls)
        return linear_algebra.transform(library)


class GPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for GPU architectures."""

    def extract(self):
        """Remove the fully-parallel loops of the loop nest. No data dependency
        analysis is performed; rather, these are the loops that are marked with
        ``pragma coffee itspace``."""

        info = visit(self.loop, self.header, info_items=['symbols_dep', 'fors'])
        symbols = info['symbols_dep']

        itspace_vrs = set()
        for nest in info['fors']:
            for loop, parent in reversed(nest):
                if '#pragma coffee itspace' not in loop.pragma:
                    continue
                parent = parent.children
                for n in loop.body:
                    parent.insert(parent.index(loop), n)
                parent.remove(loop)
                itspace_vrs.add(loop.dim)

        accessed_vrs = [s for s in symbols if any_in(s.rank, itspace_vrs)]

        return (itspace_vrs, accessed_vrs)
