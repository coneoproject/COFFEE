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
from loop_scheduler import ExpressionFissioner, ZeroLoopScheduler, SSALoopMerger
from linear_algebra import LinearAlgebra
from rewriter import ExpressionRewriter
from ast_analyzer import ExpressionGraph, StmtTracker
from coffee.visitors import MaxLoopDepth


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
        self.nonzero_info = {}
        # Track data dependencies
        self.expr_graph = ExpressionGraph(loop)
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
                Barely affects accuracy, improves performance while trying to
                minimize temporary storage
            * mode == 3: Apply four passes: generalized loop-invariant code motion
                of outer-loop dependent expressions; expansion of inner-loop dependent
                expressions; factorization of inner-loop dependent terms; 'aggressive'
                generalized loop-invariant code motion (in which n-rank temporary
                arrays can be allocated to keep expressions independent of more than
                one loop). Due to hoisting less expressions, factorization can be
                more aggressive, so accuracy can be more affected than in modes 0, 1,
                and 2. This ``mode`` is ideal if one wants to precompute as much
                expressions as possible outside the whole loop nest. Therefore, it is
                recommended to execute a ``precompute`` pass after ``rewrite(mode=3)``.
                This aggressively reduces flops, but also increases temporary storage
        """
        ExpressionRewriter.reset()
        for stmt, expr_info in self.exprs.items():
            ew = ExpressionRewriter(stmt, expr_info, self.decls, self.header,
                                    self.hoisted, self.expr_graph)
            # 1) Rewrite the expressions
            if mode == 1:
                ew.licm()

            elif mode == 2:
                ew.licm()
                if expr_info.is_tensor:
                    ew.expand()
                    ew.factorize()
                    ew.licm()

            elif mode == 3:
                ew.inject()
                if expr_info.is_tensor:
                    ew.expand(mode='full')
                    ew.factorize(mode='immutable')
                    ew.licm(out_domain=True)
                    ew.reassociate()
                    ew.licm(nrank_tmps=True)

            # 2) Try merging and optimizing the loops created by rewriting
            lm = SSALoopMerger(self.header, ew.expr_graph)
            merged_loops = lm.merge()
            for merged, merged_in in merged_loops:
                [self.hoisted.update_loop(l, merged_in) for l in merged]
            lm.simplify()

        # 3) Reduce storage by removing temporaries read in only one place
        stmt_occs = dict((k, v)
                         for d in [count(stmt, mode='symbol_id', read_only=True)
                                   for stmt in self.exprs.keys()]
                         for k, v in d.items())
        for l in self.hoisted.all_loops:
            l_occs = count(l, read_only=True)
            info = visit(l, search=Block)
            innermost_block = info['search'][Block][-1]
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
                expr = self.hoisted[symbol].expr
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
        """Avoid accessing blocks of contiguous (i.e. unit-stride) zero-valued
        columns when computing an expression."""

        # Search for zero-valued columns and restructure the iteration spaces;
        # the ZeroLoopScheduler analyzes statements "one by one", and changes
        # the iteration spaces of the enclosing loops accordingly.
        if not any([d.nonzero for d in self.decls.values()]):
            return
        zls = ZeroLoopScheduler(self.exprs, self.expr_graph, self.decls, self.hoisted)
        zls.reschedule()
        self.nonzero_info = zls.nonzero_info

    def precompute(self, mode=0):
        """Precompute statements out of ``self.loop``, which implies scalar
        expansion and code hoisting. If ``mode == 0``, all statements in the loop
        nest rooted in ``self.loop`` are precomputed, which makes it perfect. If
        ``mode == 1``, loops due to code hoisting are excluded from precomputation.

        For example: ::

        for i
          for r
            A[r] += f(i, ...)
          for j
            for k
              LT[j][k] += g(A[r], ...)

        becomes: ::

        for i
          for r
            A[i][r] += f(...)
        for i
          for j
            for k
              LT[j][k] += g(A[i][r], ...)

        """

        def precompute_stmt(node, precomputed, new_outer_block):
            """Recursively precompute, and vector-expand if already precomputed,
            all terms rooted in node."""

            if isinstance(node, Symbol):
                # Vector-expand the symbol if already pre-computed
                if node.symbol in precomputed:
                    node.rank = precomputed[node.symbol] + node.rank
            elif isinstance(node, FlatBlock):
                # Do nothing
                new_outer_block.append(node)
            elif isinstance(node, Expr):
                for n in node.children:
                    precompute_stmt(n, precomputed, new_outer_block)
            elif isinstance(node, (Assign, Incr)):
                # Precompute the LHS of the assignment
                symbol = node.children[0]
                precomputed[symbol.symbol] = (self.loop.dim,)
                new_rank = (self.loop.dim,) + symbol.rank
                symbol.rank = new_rank
                # Vector-expand the RHS
                precompute_stmt(node.children[1], precomputed, new_outer_block)
                # Finally, append the new node
                new_outer_block.append(node)
            elif isinstance(node, Decl):
                new_outer_block.append(node)
                if isinstance(node.init, Symbol):
                    node.init.symbol = "{%s}" % node.init.symbol
                elif isinstance(node.init, Expr):
                    new_assign = Assign(dcopy(node.sym), node.init)
                    precompute_stmt(new_assign, precomputed, new_outer_block)
                    node.init = EmptyStatement()
                # Vector-expand the declaration of the precomputed symbol
                node.sym.rank = (self.loop.size,) + node.sym.rank
            elif isinstance(node, For):
                # Precompute and/or Vector-expand inner statements
                new_children = []
                for n in node.body:
                    precompute_stmt(n, precomputed, new_children)
                node.body = new_children
                new_outer_block.append(node)
            else:
                raise RuntimeError("Precompute error: unexpteced node: %s" % str(node))

        # Check if the outermost loop is not perfect, in which case precomputation
        # is triggered
        if is_perfect_loop(self.loop):
            return

        # Precomputation
        do_not_precompute = set()
        if mode == 1:
            for l in self.hoisted.values():
                if l.loop:
                    do_not_precompute.add(l.decl)
                    do_not_precompute.add(l.loop)
        to_remove, precomputed_block, precomputed_syms = ([], [], {})
        for i in self.loop.body:
            if i in flatten(self.expr_domain_loops):
                break
            elif i not in do_not_precompute:
                precompute_stmt(i, precomputed_syms, precomputed_block)
                to_remove.append(i)
        # Remove precomputed statements
        for i in to_remove:
            self.loop.body.remove(i)

        # Wrap hoisted for/assignments/increments within a loop
        new_outer_block = []
        searching_stmt = []
        for i in precomputed_block:
            if searching_stmt and not isinstance(i, (Assign, Incr)):
                new_outer_block.append(ast_make_for(searching_stmt, self.loop))
                searching_stmt = []
            if isinstance(i, For):
                new_outer_block.append(ast_make_for([i], self.loop))
            elif isinstance(i, (Assign, Incr)):
                searching_stmt.append(i)
            else:
                new_outer_block.append(i)
        if searching_stmt:
            new_outer_block.append(ast_make_for(searching_stmt, self.loop))

        # Update the AST adding the newly precomputed blocks
        insert_at_elem(self.header.children, self.loop, new_outer_block)

        # Update the AST by scalar-expanding the pre-computed accessed variables
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

    def permute(self, transpose=False):
        """Permute the outermost loop with the innermost loop in the loop nest.
        This transformation is legal if ``_precompute`` was invoked. Storage layout of
        all 2-dimensional arrays involved in the element matrix computation is
        transposed."""

        def transpose_layout(node, transposed, to_transpose):
            """Transpose the storage layout of symbols in ``node``. If the symbol is
            in a declaration, then its statically-known size is transposed (e.g.
            double A[3][4] -> double A[4][3]). Otherwise, its iteration variables
            are swapped (e.g. A[i][j] -> A[j][i]).

            If ``to_transpose`` is empty, then all symbols encountered in the traversal of
            ``node`` are transposed. Otherwise, only symbols in ``to_transpose`` are
            transposed."""
            if isinstance(node, Symbol):
                if not to_transpose:
                    transposed.add(node.symbol)
                elif node.symbol in to_transpose:
                    node.rank = (node.rank[1], node.rank[0])
            elif isinstance(node, Decl):
                transpose_layout(node.sym, transposed, to_transpose)
            elif isinstance(node, FlatBlock):
                return
            else:
                for n in node.children:
                    transpose_layout(n, transposed, to_transpose)

        # Check if the outermost loop is perfect, otherwise avoid permutation
        if not is_perfect_loop(self.loop):
            return

        # Get the innermost loop and swap it with the outermost
        inner_loop = inner_loops(self.loop)[0]

        tmp = dcopy(inner_loop)
        itspace_copy(self.loop, inner_loop)
        itspace_copy(tmp, self.loop)

        to_transpose = set()
        if transpose:
            transpose_layout(inner_loop, to_transpose, set())
            transpose_layout(self.header, set(), to_transpose)

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

        info = visit(self.loop, self.header)
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
