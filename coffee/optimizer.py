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

from copy import deepcopy as dcopy

from base import *
from utils import inner_loops, visit, is_perfect_loop, flatten, ast_update_rank
from utils import set_itspace
from expression import MetaExpr
from loop_scheduler import PerfectSSALoopMerger, ExpressionFissioner, ZeroLoopScheduler
from linear_algebra import LinearAlgebra
from rewriter import ExpressionRewriter
from ast_analyzer import ExpressionGraph, StmtTracker
import plan


class LoopOptimizer(object):

    """Loop optimizer class."""

    def __init__(self, loop, header, kernel_decls):
        """Initialize the LoopOptimizer.

        :arg loop:         root loop node o a loop nest.
        :arg header:       parent of the root loop node
        :arg kernel_decls: list of declarations of the variables that are visible
                           within ``loop``."""
        self.loop = loop
        self.header = header
        self.kernel_decls = kernel_decls
        # Track nonzero regions accessed in the various loops
        self.nz_in_fors = {}
        # Integration loop (if any)
        self.int_loop = loop if "#pragma pyop2 integration" in loop.pragma else None
        # Expression graph tracking data dependencies
        self.expr_graph = ExpressionGraph()
        # Dictionary contaning various information about hoisted expressions
        self.hoisted = StmtTracker()

        # Inspect the loop nest and collect info
        info = visit(self.loop, self.header)
        self.decls, self.asm_expr = ({}, {})
        for decl_str, decl in info['decls'].items():
            self.decls[decl_str] = (decl, plan.LOCAL_VAR)
        for stmt, expr_info in info['exprs'].items():
            self.asm_expr[stmt] = MetaExpr(*expr_info)

    def rewrite(self, level):
        """Rewrite a compute-intensive expression found in the loop nest so as to
        minimize floating point operations and to relieve register pressure.
        This involves several possible transformations:

        1. Generalized loop-invariant code motion
        2. Factorization of common loop-dependent terms
        3. Expansion of constants over loop-dependent terms
        4. Zero-valued columns avoidance
        5. Precomputation of integration-dependent terms

        :arg level: The optimization level (0, 1, 2, 3, 4). The higher, the more
                    invasive is the re-writing of the expression, trying to
                    eliminate unnecessary floating point operations.

                    * level == 1: performs "basic" generalized loop-invariant \
                                  code motion
                    * level == 2: level 1 + expansion of terms, factorization of \
                                  basis functions appearing multiple times in the \
                                  same expression, and finally another run of \
                                  loop-invariant code motion to move invariant \
                                  sub-expressions exposed by factorization
                    * level == 3: level 2 + avoid computing zero-columns
                    * level == 4: level 3 + precomputation of read-only expressions \
                                  out of the loop nest
        """

        if not self.asm_expr:
            return

        # Expression rewriting
        kernel_info = (self.header, self.kernel_decls)
        for stmt_info in self.asm_expr.items():
            ew = ExpressionRewriter(stmt_info, self.decls, kernel_info,
                                    self.hoisted, self.expr_graph)
            if level > 0:
                ew.licm()
            if level > 1:
                ew.expand()
                ew.distribute()
                ew.licm()

        # Fuse loops iterating along the same iteration space and remove
        # any duplicate sub-expressions
        if level > 1:
            lm = PerfectSSALoopMerger(self.root, self.expr_graph)
            merged_loops = lm.merge()
            for merged, merged_in in merged_loops:
                [self.hoisted.update_loop(l, merged_in) for l in merged]
            lm.simplify()

    def eliminate_zeros(self):
        """Avoid accessing blocks of contiguous zero-valued columns when computing
        an expression."""

        # First, split expressions into separate loop nests, based on sum's
        # associativity. This exposes more opportunities for restructuring loops,
        # since different summands may have contiguous regions of zero-valued
        # columns in different positions
        elf = ExpressionFissioner(1)
        for expr in self.asm_expr.items():
            elf.fission(expr, False)
        # Search for zero-valued columns and restructure the iteration spaces;
        # the ZeroLoopScheduler analyzes statements "one by one", and changes
        # the iteration spaces of the enclosing loops accordingly.
        zls = ZeroLoopScheduler(self.root, self.expr_graph,
                                (self.kernel_decls, self.decls))
        self.asm_expr = zls.reschedule()[-1]
        self.nz_in_fors = zls.nz_in_fors

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
                precomputed[symbol.symbol] = (self.loop.it_var(),)
                new_rank = (self.loop.it_var(),) + symbol.rank
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
                node.sym.rank = (self.loop.size(),) + node.sym.rank
            elif isinstance(node, For):
                # Precompute and/or Vector-expand inner statements
                new_children = []
                for n in node.children[0].children:
                    precompute_stmt(n, precomputed, new_children)
                node.children[0].children = new_children
                new_outer_block.append(node)
            else:
                raise RuntimeError("Precompute error: unexpteced node: %s" % str(node))

        def create_prec_for(stmts):
            """Create a for loop having the same iteration space as  ``self.loop``
            enclosing the statements in  ``stmts``."""
            wrap = Block(stmts, open_scope=True)
            precompute_for = For(dcopy(self.loop.init), dcopy(self.loop.cond),
                                 dcopy(self.loop.incr), wrap, dcopy(self.loop.pragma))
            return precompute_for

        # Check if the outermost loop is not perfect, in which case precomputation
        # is triggered
        if is_perfect_loop(self.loop):
            return

        # Precomputation
        no_prec = set()
        if mode == 1:
            for l in self.hoisted.values():
                if l.loop:
                    no_prec.add(l.decl)
                    no_prec.add(l.loop)
        to_remove, precomputed_block, precomputed_syms = ([], [], {})
        for i in self.loop.children[0].children:
            if i in flatten(self.expr_unit_stride_loops):
                break
            elif i not in no_prec:
                precompute_stmt(i, precomputed_syms, precomputed_block)
                to_remove.append(i)
        # Remove precomputed statements
        for i in to_remove:
            self.loop.children[0].children.remove(i)

        # Wrap hoisted for/assignments/increments within a loop
        new_outer_block = []
        searching_stmt = []
        for i in precomputed_block:
            if searching_stmt and not isinstance(i, (Assign, Incr)):
                new_outer_block.append(create_prec_for(searching_stmt))
                searching_stmt = []
            if isinstance(i, For):
                new_outer_block.append(create_prec_for([i]))
            elif isinstance(i, (Assign, Incr)):
                searching_stmt.append(i)
            else:
                new_outer_block.append(i)
        if searching_stmt:
            new_outer_block.append(create_prec_for(searching_stmt))

        # Update the AST adding the newly precomputed blocks
        root = self.header.children
        ofs = root.index(self.loop)
        self.header.children = root[:ofs] + new_outer_block + root[ofs:]

        # Update the AST by scalar-expanding the pre-computed accessed variables
        ast_update_rank(self.loop, precomputed_syms)

    @property
    def root(self):
        """Return the root node of the assembly loop nest. It can be either the
        loop over quadrature points or, if absent, a generic point in the
        assembly routine."""
        return self.int_loop.children[0] if self.int_loop else self.header

    @property
    def expr_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        loops that expressions depend on."""
        return [expr_info.loops for expr_info in self.asm_expr.values()]

    @property
    def expr_unit_stride_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where a tuple contains all
        loops along which an expression performs unit-stride memory accesses."""
        return [expr_info.unit_stride_loops for expr_info in self.asm_expr.values()]


class CPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for CPU architectures."""

    def unroll(self, loop_uf):
        """Unroll loops enclosing expressions as specified by ``loop_uf``.

        :arg loop_uf: dictionary from iteration spaces to unroll factors."""

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
            new_asm_expr = {}
            for stmt, expr_info in self.asm_expr.items():
                loop = [l for l in expr_info.perfect_loops if l.it_var() == itspace]
                if not loop:
                    # Unroll only loops in a perfect loop nest
                    continue
                loop = loop[0]  # Only one loop possibly found
                for i in range(uf-1):
                    new_stmt = dcopy(stmt)
                    update_expr(new_stmt, itspace, i+1)
                    expr_info.parent.children.append(new_stmt)
                    new_asm_expr.update({new_stmt: expr_info})
                if loop not in unrolled_loops:
                    loop.incr.children[1].symbol += uf-1
                    unrolled_loops.add(loop)
            self.asm_expr.update(new_asm_expr)

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
        set_itspace(inner_loop, self.loop)
        set_itspace(self.loop, tmp)

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

        if not self.asm_expr:
            return

        new_asm_expr = {}
        elf = ExpressionFissioner(cut)
        for splittable in self.asm_expr.items():
            # Split the expression
            new_asm_expr.update(elf.fission(splittable, True))
        self.asm_expr = new_asm_expr

    def blas(self, library):
        """Convert an expression into sequences of calls to external dense linear
        algebra libraries. Currently, MKL, ATLAS, and EIGEN are supported."""

        # First, check that the loop nest has depth 3, otherwise it's useless
        if visit(self.loop, self.header)['max_depth'] != 3:
            return

        linear_algebra = LinearAlgebra(self.loop, self.header, self.kernel_decls)
        return linear_algebra.transform(library)


class GPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for GPU architectures."""

    def extract(self):
        """Remove the fully-parallel loops of the loop nest. No data dependency
        analysis is performed; rather, these are the loops that are marked with
        ``pragma pyop2 itspace``."""

        info = visit(self.loop, self.header)
        fors = info['fors']
        syms = info['symbols']

        itspace_vrs = set()
        for node, parent in reversed(fors):
            if '#pragma pyop2 itspace' not in node.pragma:
                continue
            parent = parent.children
            for n in node.children[0].children:
                parent.insert(parent.index(node), n)
            parent.remove(node)
            itspace_vrs.add(node.it_var())

        from utils import any_in
        accessed_vrs = [s for s in syms if any_in(s.rank, itspace_vrs)]

        return (itspace_vrs, accessed_vrs)
