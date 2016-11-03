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

from __future__ import absolute_import, print_function, division

import operator
import resource
from collections import OrderedDict
from itertools import combinations
from math import factorial as fact

from . import system
from .base import *
from .utils import *
from .scheduler import ExpressionFissioner, ZeroRemover, SSALoopMerger
from .rewriter import ExpressionRewriter
from .cse import CSEUnpicker
from .logger import warn
from coffee.visitors import FindInstances, ProjectExpansion
from functools import reduce


class LoopOptimizer(object):

    def __init__(self, loop, header, decls, exprs):
        """Initialize the LoopOptimizer.

        :param loop: root AST node of a loop nest
        :param header: the kernel's top node
        :param decls: list of Decl objects accessible in ``loop``
        :param exprs: list of expressions to be optimized
        """
        self.loop = loop
        self.header = header
        self.decls = decls
        self.exprs = exprs

        # Track nonzero regions accessed in each symbol
        self.nz_syms = {}
        # Track hoisted expressions
        self.hoisted = StmtTracker()

    def rewrite(self, mode):
        """Rewrite all compute-intensive expressions detected in the loop nest to
        minimize the number of floating point operations performed.

        :param mode: Any value in (0, 1, 2, 3, 4). Each ``mode`` corresponds to a
            different expression rewriting strategy.

            * mode == 0: no rewriting is performed.
            * mode == 1: generalized loop-invariant code motion.
            * mode == 2: apply four passes: generalized loop-invariant code motion;
                expansion of inner-loop dependent expressions; factorization of
                inner-loop dependent terms; generalized loop-invariant code motion.
            * mode == 3: apply multiple passes; aims at pre-evaluating sub-expressions
                that fully depend on reduction loops.
            * mode == 4: rewrite an expression based on its sharing graph
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

        # Search for factorization opportunities across temporaries in the kernel
        if mode > 1 and self.exprs:
            self._unpick_cse()

        # Expression rewriting, expressed as a sequence of AST transformation passes
        for stmt, expr_info in self.exprs.items():
            ew = ExpressionRewriter(stmt, expr_info, self.decls, self.header,
                                    self.hoisted)

            if expr_info.mode == 1:
                if expr_info.dimension in [0, 1]:
                    ew.licm(mode='only_outlinear')
                else:
                    ew.licm()

            elif expr_info.mode == 2:
                if expr_info.dimension > 0:
                    ew.replacediv()
                    ew.SGrewrite()

            elif expr_info.mode == 3:
                ew.expand(mode='all')
                ew.factorize(mode='all')
                ew.licm(mode='only_const')
                ew.factorize(mode='constants')
                ew.licm(mode='aggressive')
                ew.preevaluate()
                ew.factorize(mode='linear')
                ew.licm(mode='only_const')

            elif expr_info.mode == 4:
                ew.replacediv()
                ew.factorize()
                ew.licm(mode='only_outlinear')
                if expr_info.dimension > 0:
                    ew.licm(mode='only_linear', iterative=False, max_sharing=True)
                    ew.SGrewrite()
                    ew.expand()

        # Try merging the loops created by expression rewriting
        merged_loops = SSALoopMerger().merge(self.header)
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

        # Reduce memory pressure by avoiding useless temporaries
        self._min_temporaries()

        # Handle the effects, at the C-level, of the AST transformation
        self._recoil()

    def eliminate_zeros(self):
        """Restructure the iteration spaces nested in this LoopOptimizer to
        avoid evaluation of arithmetic operations involving zero-valued blocks
        in statically initialized arrays."""

        zls = ZeroRemover(self.exprs, self.decls, self.hoisted)
        self.nz_syms = zls.reschedule(self.header)

    def _unpick_cse(self):
        """Search for factorization opportunities across temporaries created by
        common sub-expression elimination. If a gain in operation count is detected,
        unpick CSE and apply factorization + code motion."""
        cse_unpicker = CSEUnpicker(self.exprs, self.header, self.hoisted, self.decls)
        cse_unpicker.unpick()

    def _min_temporaries(self):
        """Remove unnecessary temporaries, thus relieving memory pressure.
        A temporary is removed iff:

            * it is written once, AND
            * it is read once OR it is read n times, but it hosts only a Symbol
        """
        occs = count(self.header, mode='symbol_id', read_only=True)

        for l in self.hoisted.all_loops:
            info = visit(l)
            l_occs = count(l, read_only=True)
            to_replace, to_remove = {}, []
            for (temporary, _, _), temporary_occs in l_occs.items():
                if temporary not in self.hoisted:
                    continue
                if self.hoisted[temporary].loop is not l:
                    continue
                if occs.get(temporary) != temporary_occs:
                    continue
                decl = self.hoisted[temporary].decl
                place = self.hoisted[temporary].place
                expr = self.hoisted[temporary].stmt.rvalue
                if temporary_occs > 1 and explore_operator(expr):
                    continue
                temporary_refs = info['symbol_refs'][temporary]
                syms_mode = info['symbols_mode']
                # Note: only one write is possible at this point
                write = [(s, p) for s, p in temporary_refs if syms_mode[s][0] == WRITE][0]
                to_replace[write[0]] = expr
                to_remove.append(write[1])
                place.children.remove(decl)
                # Update trackers
                self.hoisted.pop(temporary)
                self.decls.pop(temporary)

            # Replace temporary symbols and clean up
            l_innermost_body = inner_loops(l)[-1].body
            for stmt in l_innermost_body:
                if stmt.lvalue in to_replace:
                    continue
                while ast_replace(stmt, to_replace, copy=True):
                    pass
            for stmt in to_remove:
                l_innermost_body.remove(stmt)

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
        threshold = system.architecture['cache_size'] * 1.2

        expr_graph = ExpressionGraph(header)

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
            save_factor = [l.size for l in expr_info.out_linear_loops] or [1]
            save_factor = reduce(operator.mul, save_factor)
            # The save factor should be multiplied by the number of terms
            # that will /not/ be pre-evaluated. To obtain this number, we
            # can exploit the linearity of the expression in the terms
            # depending on the linear loops.
            syms = FindInstances(Symbol).visit(target_expr)[Symbol]
            inner = lambda s: any(r == expr_info.linear_dims[-1] for r in s.rank)
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
                dissected = find_expression(expr, Prod, expr_info.linear_dims, i_syms)
                leftover = find_expression(expr, dims=expr_info.linear_dims, out_syms=i_syms)
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
                        for j in expr_graph.shares(i):
                            # _n=number of unique elements, _k=group size
                            _n = injectable[j[0]][1]
                            _k = len(j)
                            partial *= fact(_n + _k - 1)//(fact(_k)*fact(_n - 1))
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
                    ew.expand(mode='all').factorize(mode='all').factorize(mode='linear')
                    nterms = ew.licm(mode='aggressive', look_ahead=True)
                    nterms = len(uniquify(nterms[expr_info.dims])) or 1
                    fake_parent[fake_parent.index(fake_stmt)] = stmt
                    cost = nterms * increase_factor

                    # Pre-evaluation will also increase the working set size by
                    # /cost/ * /sizeof(term)/.
                    size = [l.size for l in expr_info.linear_loops]
                    size = reduce(operator.mul, size, 1)
                    storage_increase = cost * size * system.architecture[expr_info.type]

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
            expr_info.mode = 4
            inj_exprs = injected.get(stmt)
            if not inj_exprs:
                continue
            fissioner = ExpressionFissioner(match=inj_exprs, loops='all', perfect=True)
            new_exprs = fissioner.fission(stmt, self.exprs.pop(stmt))
            self.exprs.update(new_exprs)
            for stmt, expr_info in new_exprs.items():
                expr_info.mode = 3 if stmt in fissioner.matched else 4

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
                warn("Stack may blow up, could not increase its size.")

    @property
    def expr_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        loops enclosing expressions."""
        return [expr_info.loops for expr_info in self.exprs.values()]

    @property
    def expr_linear_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        linear loops enclosing expressions."""
        return [expr_info.linear_loops for expr_info in self.exprs.values()]


class CPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for CPU architectures."""

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
