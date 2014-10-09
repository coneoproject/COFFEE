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

try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict
from collections import defaultdict
from copy import deepcopy as dcopy

import networkx as nx

from base import *
from utils import inner_loops, visit, is_perfect_loop, flatten, ast_update_rank
from utils import ast_replace, set_itspace, loops_as_dict, od_find_next
from expression import MetaExpr
from loop_scheduler import PerfectSSALoopMerger, ExprLoopFissioner, ZeroLoopScheduler
from linear_algebra import LinearAlgebra
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
        self.hoisted = OrderedDict()

        # Inspect the loop nest and collect info
        info = visit(self.loop, self.header)
        self.decls, self.asm_expr = ({}, {})
        for decl_str, decl in info['decls'].items():
            self.decls[decl_str] = (decl, plan.LOCAL_VAR)
        for stmt, expr_info in info['exprs'].items():
            self.asm_expr[stmt] = MetaExpr(*expr_info)

    def rewrite(self, level, is_block_sparse):
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
        :arg is_block_sparse: True if the expression is characterized by the
                              presence of block-sparse arrays
        """

        if not self.asm_expr:
            return

        kernel_info = (self.header, self.kernel_decls)
        for stmt_info in self.asm_expr.items():
            ew = ExpressionRewriter(stmt_info, self.decls, kernel_info,
                                    self.hoisted, self.expr_graph)
            # Perform expression rewriting
            if level > 0:
                ew.licm()
            if level > 1:
                ew.expand()
                ew.distribute()
                ew.licm()
                # Fuse loops iterating along the same iteration space
                lm = PerfectSSALoopMerger(self.expr_graph, self.root)
                lm.merge()
                ew.simplify()

        # Eliminate zero-valued columns if the kernel operation uses block-sparse
        # arrays (contiguous zero-valued columns are present)
        if level == 3 and is_block_sparse:
            # Split the expression into separate loop nests, based on sum's
            # associativity. This exposes more opportunities for restructuring loops,
            # since different summands may have contiguous regions of zero-valued
            # columns in different positions. The ZeroLoopScheduler analyzes statements
            # "one by one", and changes the iteration spaces of the enclosing
            # loops accordingly.
            elf = ExprLoopFissioner(self.expr_graph, self.root, 1)
            new_asm_expr = {}
            for expr in self.asm_expr.items():
                new_asm_expr.update(elf.expr_fission(expr, False))
            # Search for zero-valued columns and restructure the iteration spaces
            zls = ZeroLoopScheduler(self.expr_graph, self.root,
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
            no_prec = set([l[1] for l in self.hoisted.values() if l[2]])
            no_prec = no_prec.union([l[2] for l in self.hoisted.values() if l[2]])
        to_remove, precomputed_block, precomputed_syms = ([], [], {})
        for i in self.loop.children[0].children:
            if i in flatten(self.expr_fast_loops):
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
    def expr_fast_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        loops along which expressions iterate fastest."""
        return [expr_info.fast_loops for expr_info in self.asm_expr.values()]


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
        elf = ExprLoopFissioner(self.expr_graph, self.root, cut)
        for splittable in self.asm_expr.items():
            # Split the expression
            new_asm_expr.update(elf.expr_fission(splittable, True))
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

        # Properties of the transformed expression
        self._licm = 0
        self._expanded = False

    def licm(self):
        """Perform loop-invariant code motion.

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

        def extract(node, expr_dep, length=0):
            """Extract invariant sub-expressions from the original expression.
            Hoistable sub-expressions are stored in expr_dep."""

            def hoist(node, dep, expr_dep, _extract=True):
                if _extract:
                    node = Par(node) if isinstance(node, Symbol) else node
                    expr_dep[dep].append(node)
                extract.has_extracted = extract.has_extracted or _extract

            if isinstance(node, Symbol):
                return (extract.symbols[node], extract.INV, 1)
            if isinstance(node, Par):
                return (extract(node.children[0], expr_dep, length))

            # Traverse the expression tree
            left, right = node.children
            dep_l, info_l, len_l = extract(left, expr_dep, length)
            dep_r, info_r, len_r = extract(right, expr_dep, length)
            node_len = len_l + len_r

            # Filter out false dependencies
            dep_l = tuple(d for d in dep_l if d in extract.real_deps)
            dep_r = tuple(d for d in dep_r if d in extract.real_deps)

            if info_l == extract.KSE and info_r == extract.KSE:
                if dep_l != dep_r:
                    # E.g. (A[i]*alpha + D[i])*(B[j]*beta + C[j])
                    hoist(left, dep_l, expr_dep)
                    hoist(right, dep_r, expr_dep)
                    return ((), extract.HOI, node_len)
                else:
                    # E.g. (A[i]*alpha)+(B[i]*beta)
                    return (dep_l, extract.KSE, node_len)
            elif info_l == extract.KSE and info_r == extract.INV:
                # E.g. (A[i] + B[i])*C[j]
                hoist(left, dep_l, expr_dep)
                hoist(right, dep_r, expr_dep, not self._licm or len_r > 1)
                return ((), extract.HOI, node_len)
            elif info_l == extract.INV and info_r == extract.KSE:
                # E.g. A[i]*(B[j]) + C[j])
                hoist(right, dep_r, expr_dep)
                hoist(left, dep_l, expr_dep, not self._licm or len_l > 1)
                return ((), extract.HOI, node_len)
            elif info_l == extract.INV and info_r == extract.INV:
                if not dep_l and not dep_r:
                    # E.g. alpha*beta
                    return ((), extract.INV, node_len)
                elif dep_l and dep_r and dep_l != dep_r:
                    if set(dep_l).issubset(set(dep_r)):
                        # E.g. A[i]*B[i,j]
                        return (dep_r, extract.KSE, node_len)
                    elif set(dep_r).issubset(set(dep_l)):
                        # E.g. A[i,j]*B[i]
                        return (dep_l, extract.KSE, node_len)
                    else:
                        # dep_l != dep_r:
                        # E.g. A[i]*B[j]
                        hoist(left, dep_l, expr_dep, not self._licm or len_l > 1)
                        hoist(right, dep_r, expr_dep, not self._licm or len_r > 1)
                        return ((), extract.HOI, node_len)
                elif dep_l and dep_r and dep_l == dep_r:
                    # E.g. A[i] + B[i]
                    return (dep_l, extract.INV, node_len)
                elif dep_l and not dep_r:
                    # E.g. A[i]*alpha
                    hoist(right, dep_r, expr_dep, len_r > 1)
                    return (dep_l, extract.KSE, node_len)
                elif dep_r and not dep_l:
                    # E.g. alpha*A[i]
                    hoist(left, dep_l, expr_dep, len_l > 1)
                    return (dep_r, extract.KSE, node_len)
                else:
                    raise RuntimeError("Error while hoisting invariant terms")
            elif info_l == extract.HOI and info_r == extract.KSE:
                hoist(right, dep_r, expr_dep, len_r > 2)
                return ((), extract.HOI, node_len)
            elif info_l == extract.KSE and info_r == extract.HOI:
                hoist(left, dep_l, expr_dep, len_l > 2)
                return ((), extract.HOI, node_len)
            elif info_l == extract.HOI or info_r == extract.HOI:
                return ((), extract.HOI, node_len)
            else:
                raise RuntimeError("Fatal error while finding hoistable terms")

        expr_loops = self.expr_info.loops
        dict_expr_loops = loops_as_dict(expr_loops)
        real_deps = dict_expr_loops.keys()

        # Set global parameters of the extract recursive function
        extract.symbols = visit(self.header, None)['symbols']
        extract.has_extracted = False
        extract.real_deps = real_deps
        extract.INV = 0  # Invariant term(s)
        extract.KSE = 1  # Keep searching invariant sub-expressions
        extract.HOI = 2  # Stop searching, done hoisting

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        inv_dep = {}
        typ = self.kernel_decls[self.stmt.children[0].symbol][0].typ
        while True:
            expr_dep = defaultdict(list)
            extract(self.stmt.children[1], expr_dep)

            # While end condition
            if self._licm and not extract.has_extracted:
                break
            extract.has_extracted = False
            self._licm += 1

            for all_deps, expr in sorted(expr_dep.items()):
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
                        self._licm, i), sym_rank) for i in range(len(expr))]
                var_decl = [Decl(typ, _s) for _s in syms]
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
                    extract.symbols[s] = dep

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
            _,place, next_loop, wl = inv_info
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
                old_sym_info = self.hoisted[i.sym.symbol]
                old_sym_info = old_sym_info[0:2] + (inv_code[0],) + (place.children,)
                self.hoisted[i.sym.symbol] = old_sym_info

    def count_occurrences(self, str_key=False):
        """For each variable in the expression, count how many times
        it appears as involved in some operations. For example, for the
        expression ``a*(5+c) + b*(a+4)``, return ``{a: 2, b: 1, c: 1}``."""

        def count(node, counter):
            if isinstance(node, Symbol):
                node = str(node) if str_key else (node.symbol, node.rank)
                if node in counter:
                    counter[node] += 1
                else:
                    counter[node] = 1
            else:
                for c in node.children:
                    count(c, counter)

        counter = {}
        count(self.stmt.children[1], counter)
        return counter

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
        asm_out, asm_in = self.expr_info.fast_itvars
        it_var_occs = {asm_out: 0, asm_in: 0}
        for s in self.count_occurrences().keys():
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

        def create_sum(symbols):
            if len(symbols) == 1:
                return symbols[0]
            else:
                return Sum(symbols[0], create_sum(symbols[1:]))

        # Expansion ensures the expression to be in a form like:
        # tensor[i][j] += A[i]*B[j] + C[i]*D[j] + A[i]*E[j] + ...
        if not self._expanded:
            raise RuntimeError("Distribute error: expansion required first.")

        to_distr = defaultdict(list)
        find_prod(self.stmt.children[1], self.count_occurrences(True), to_distr)

        # Create the new expression
        new_prods = []
        for d in to_distr.values():
            dist, target = zip(*d)
            target = Par(create_sum(target)) if len(target) > 1 else create_sum(target)
            new_prods.append(Par(Prod(dist[0], target)))
        self.stmt.children[1] = Par(create_sum(new_prods))

    def simplify(self):
        """Scan the hoisted terms one by one and eliminate duplicate sub-expressions.
        Remove useless assignments (e.g. a = b, and b never used later)."""

        def replace_expr(node, parent, parent_idx, it_var, hoisted_expr):
            """Recursively search for any sub-expressions rooted in node that have
            been hoisted and therefore are already kept in a temporary. Replace them
            with such temporary."""
            if isinstance(node, Symbol):
                return
            else:
                tmp_sym = hoisted_expr.get(str(node)) or hoisted_expr.get(str(parent))
                if tmp_sym:
                    # Found a temporary value already hosting the value of node
                    parent.children[parent_idx] = Symbol(dcopy(tmp_sym), (it_var,))
                else:
                    # Go ahead recursively
                    for i, n in enumerate(node.children):
                        replace_expr(n, node, i, it_var, hoisted_expr)

        # Remove duplicates
        hoisted_expr = {}
        for sym, sym_info in self.hoisted.items():
            expr, var_decl, inv_for, place = sym_info
            if not isinstance(inv_for, For):
                continue
            # Check if any sub-expressions rooted in expr is alredy stored in a temporary
            replace_expr(expr.children[0], expr, 0, inv_for.it_var(), hoisted_expr)
            # Track the (potentially modified) hoisted expression
            hoisted_expr[str(expr)] = sym


class ExpressionExpander(object):
    """Expand expressions such that: ::

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

    becomes: ::

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    def __init__(self, var_info, expr_graph):
        self.var_info = var_info
        self.expr_graph = expr_graph
        self.expanded_decls = {}
        self.found_consts = {}
        self.expanded_syms = []

    def _do_expand(self, sym, const):
        """Perform the actual expansion. If there are no dependencies, then
        the already hoisted expression is expanded. Otherwise, if the symbol to
        be expanded occurs multiple times in the expression, or it depends on
        other hoisted symbols that will also be expanded, create a new symbol."""

        old_expr, var_decl, inv_for, place = self.var_info[sym.symbol]

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
            self.expanded_decls[new_const_decl.sym.symbol] = (new_const_decl, plan.LOCAL_VAR)
            self.expanded_syms.append(new_const_decl.sym)
            self.found_consts[const_str] = const_sym
            self.expr_graph.add_dependency(const_sym, const, False)
            # Update the AST
            place.insert(place.index(inv_for), new_const_decl)
            const = const_sym

        # No dependencies, just perform the expansion
        if not self.expr_graph.has_dep(sym):
            old_expr.children[0] = Prod(Par(old_expr.children[0]), dcopy(const))
            self.expr_graph.add_dependency(sym, const, False)
            return

        # Create a new symbol, expression, and declaration
        new_expr = Par(Prod(dcopy(sym), const))
        sym.symbol += "_EXP%d" % len(self.expanded_syms)
        new_node = Assign(dcopy(sym), new_expr)
        new_var_decl = dcopy(var_decl)
        new_var_decl.sym.symbol = sym.symbol
        # Append new expression and declaration
        inv_for.children[0].children.append(new_node)
        place.insert(place.index(var_decl), new_var_decl)
        self.expanded_decls[new_var_decl.sym.symbol] = (new_var_decl, plan.LOCAL_VAR)
        self.expanded_syms.append(new_var_decl.sym)
        # Update tracked information
        self.var_info[sym.symbol] = (new_expr, new_var_decl, inv_for, place)
        self.expr_graph.add_dependency(sym, new_expr, 0)

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
                for sym in expandable:
                    # Perform the expansion
                    if sym.symbol not in self.var_info:
                        raise RuntimeError("Expansion error: no symbol: %s" % sym.symbol)
                    old_expr, var_decl, inv_for, place = self.var_info[sym.symbol]
                    self._do_expand(sym, const)
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


class ExpressionGraph(object):

    """Track read-after-write dependencies between symbols."""

    def __init__(self):
        self.deps = nx.DiGraph()

    def add_dependency(self, sym, expr, self_loop):
        """Extract symbols from ``expr`` and create a read-after-write dependency
        with ``sym``. If ``sym`` already has a dependency, then ``sym`` has a
        self dependency on itself."""

        def extract_syms(sym, node, deps):
            if isinstance(node, Symbol):
                deps.add_edge(sym, node.symbol)
            else:
                for n in node.children:
                    extract_syms(sym, n, deps)

        sym = sym.symbol
        # Add self-dependency
        if self_loop:
            self.deps.add_edge(sym, sym)
        extract_syms(sym, expr, self.deps)

    def has_dep(self, sym, target_sym=None):
        """If ``target_sym`` is not provided, return True if ``sym`` has a
        read-after-write dependency with some other symbols. This is the case if
        ``sym`` has either a self dependency or at least one input edge, meaning
        that other symbols depend on it.
        Otherwise, if ``target_sym`` is not None, return True if ``sym`` has a
        read-after-write dependency on it, i.e. if there is an edge from
        ``target_sym`` to ``sym``."""

        sym = sym.symbol
        if not target_sym:
            return sym in self.deps and zip(*self.deps.in_edges(sym))
        else:
            target_sym = target_sym.symbol
            return sym in self.deps and self.deps.has_edge(sym, target_sym)
