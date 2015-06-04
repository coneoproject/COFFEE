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

from base import *
from utils import *
from loop_scheduler import ExpressionFissioner, ZeroRemover, SSALoopMerger
from linear_algebra import LinearAlgebra
from rewriter import ExpressionRewriter
from ast_analyzer import ExpressionGraph, StmtTracker
from coffee.visitors import MaxLoopDepth, FindInstances


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

        # Track nonzero regions accessed in the loop nest
        self.nz_info = {}
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
            * mode == 3: Apply seven passes: injection (or 'inlining') of loops in the
                expressions. Expansion of inner-loops dependent terms. Factorization of
                inner-loops dependent terms; generalized loop-invariant code motion of
                outer-loop dependent terms. Reassociation, followed by 'aggressive'
                generalized loop-invariant code motion (in which n-rank temporary
                arrays can be allocated to host expressions independent of more than
                one loop). The last pass, 'simplify', applies a few transformations,
                including optimization of associative reductions. This sequence of
                steps mimics the 'Tensor Contraction representation' of multilinear
                forms as performed by the 'FEniCS Form Compiler', but makes profoundly
                different implementation choices (e.g., no aggressive unrolling).
        """
        ExpressionRewriter.reset()
        for stmt, expr_info in self.exprs.items():
            ew = ExpressionRewriter(stmt, expr_info, self.decls, self.header,
                                    self.hoisted, self.expr_graph)
            # 1) Transform the expression applying one or more rewriting passes
            if expr_info.dimension in [0, 1] and mode in [1, 2]:
                ew.licm(hoist_out_domain=True)
                continue

            if mode == 1:
                ew.licm()

            elif mode == 2:
                ew.licm()
                if expr_info.is_tensor:
                    ew.expand()
                    ew.factorize()
                    ew.licm()

            elif mode == 3:
                if expr_info.is_tensor:
                    ew.inject()
                    ew.expand(mode='full')
                    ew.factorize(mode='immutable')
                    ew.licm(hoist_out_domain=True)
                    ew.factorize(mode='constants')
                    ew.reassociate()
                    ew.licm(hoist_domain_const=True)
                    ew.simplify()
                    ew.factorize(mode='immutable')
                    ew.licm(hoist_out_domain=True)

            # 2) Try merging and optimizing the loops created by rewriting
            merged_loops = SSALoopMerger(ew.expr_graph).merge(self.header)
            for merged, merged_in in merged_loops:
                [self.hoisted.update_loop(l, merged_in) for l in merged]

        # 3) Reduce storage by removing temporaries read in only one place
        stmt_occs = dict((k, v)
                         for d in [count(stmt, mode='symbol_id', read_only=True)
                                   for stmt in self.exprs.keys()]
                         for k, v in d.items())
        for l in self.hoisted.all_loops:
            l_occs = count(l, read_only=True)
            info = visit(l, info_items=['symbol_refs', 'symbols_mode'])
            retval = FindInstances.default_retval()
            innermost_block = FindInstances(Block).visit(l, ret=retval)[Block][-1]
            to_replace, to_remove = {}, []
            for (symbol, rank), sym_occs in l_occs.items():
                # If the symbol appears once, then it is a potential candidate
                # for removal. It is actually removed if it does't appear in
                # the expression from which was extracted. Symbols appearing
                # more than once are removed if they host an expression made
                # of just one symbol
                if symbol not in self.hoisted or symbol in stmt_occs:
                    continue
                if self.hoisted[symbol].loop is not l:
                    continue
                decl = self.hoisted[symbol].decl
                place = self.hoisted[symbol].place
                expr = self.hoisted[symbol].stmt.children[1]
                if sym_occs > 1 and not isinstance(expr.children[0], Symbol):
                    continue
                # Delete any replaced hoisted symbol, declaration, and evaluation
                symbol_refs = info['symbol_refs'][symbol]
                syms_mode = info['symbols_mode']
                # Note: only one write is possible
                write = [(s, p) for s, p in symbol_refs if syms_mode[s][0] == WRITE][0]
                to_replace[write[0]] = expr
                to_remove.append(write[1])
                place.children.remove(decl)
                self.hoisted.pop(symbol)
                self.decls.pop(symbol)
            for stmt in innermost_block.children:
                ast_replace(stmt.children[1], to_replace, copy=True)
            for stmt in to_remove:
                innermost_block.children.remove(stmt)

    def eliminate_zeros(self):
        """Restructure the iteration spaces nested in this LoopOptimizer to
        avoid evaluation of arithmetic operations involving zero-valued blocks
        in statically initialized arrays."""

        if any([d.nonzero for d in self.decls.values()]):
            zls = ZeroRemover(self.exprs, self.decls, self.hoisted)
            self.nz_info = zls.reschedule(self.header)

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
            new_exprs = {}
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

        new_exprs = {}
        elf = ExpressionFissioner(cut)
        for stmt, expr_info in self.exprs.items():
            # Split the expression
            new_exprs.update(elf.fission(stmt, expr_info, True))
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
