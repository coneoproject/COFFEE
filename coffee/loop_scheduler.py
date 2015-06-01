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
from itertools import product
from copy import deepcopy as dcopy

from base import *
from utils import *
from expression import MetaExpr, copy_metaexpr
from coffee.visitors import FindLoopNests


class LoopScheduler(object):

    """Base class for classes that handle loop scheduling; that is, loop fusion,
    loop distribution, etc."""


class SSALoopMerger(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then merge fusible loops.
    Statements must be in "soft" SSA form: they can be declared and initialized
    at declaration time, then they can be assigned a value in only one place."""

    def __init__(self, expr_graph):
        """Initialize the SSALoopMerger.

        :param expr_graph: the ExpressionGraph tracking all data dependencies
            involving identifiers that appear in ``header``.
        """
        self.expr_graph = expr_graph
        self.merged_loops = []

    def _merge_loops(self, root, loop_a, loop_b):
        """Merge the body of ``loop_a`` in ``loop_b`` and eliminate ``loop_a``
        from the tree rooted in ``root``. Return a reference to the block
        containing the merged loop as well as the iteration variables used
        in the respective iteration spaces."""
        # Find the first statement in the perfect loop nest loop_b
        dims_a, dims_b = [], []
        while isinstance(loop_b.children[0], (Block, For)):
            if isinstance(loop_b, For):
                dims_b.append(loop_b.dim)
            loop_b = loop_b.children[0]
        # Find the first statement in the perfect loop nest loop_a
        root_loop_a = loop_a
        while isinstance(loop_a.children[0], (Block, For)):
            if isinstance(loop_a, For):
                dims_a.append(loop_a.dim)
            loop_a = loop_a.children[0]
        # Merge body of loop_a in loop_b
        loop_b.children[0:0] = loop_a.children
        # Remove loop_a from root
        root.children.remove(root_loop_a)
        return (loop_b, tuple(dims_a), tuple(dims_b))

    def _simplify(self, merged_loops):
        """Scan the list of merged loops and eliminate sub-expressions that became
        duplicate as now iterating along the same iteration space. For example: ::

            for i = 0 to N
              A[i] = B[i] + C[i]
            for j = 0 to N
              D[j] = B[j] + C[j]

        After merging this becomes: ::

            for i = 0 to N
              A[i] = B[i] + C[i]
              D[i] = B[i] + C[i]

        And finally, after simplification (i.e. after ``simplify`` is applied): ::

            for i = 0 to N
              A[i] = B[i] + C[i]
              D[i] = A[i]
        """

        def replace_expr(node, parent, parent_idx, dim, hoisted_expr):
            """Recursively search for any sub-expressions rooted in node that have
            been hoisted and therefore are already kept in a temporary. Replace them
            with such temporary."""
            if isinstance(node, Symbol):
                return
            else:
                tmp_sym = hoisted_expr.get(str(node)) or hoisted_expr.get(str(parent))
                if tmp_sym:
                    # Found a temporary value already hosting the value of node
                    parent.children[parent_idx] = dcopy(tmp_sym)
                else:
                    # Go ahead recursively
                    for i, n in enumerate(node.children):
                        replace_expr(n, node, i, dim, hoisted_expr)

        hoisted_expr = {}
        for loop in merged_loops:
            block = loop.body
            for stmt in block:
                sym, expr = stmt.children
                replace_expr(expr.children[0], expr, 0, loop.dim, hoisted_expr)
                hoisted_expr[str(expr)] = sym

    def merge(self, root):
        """Merge perfect loop nests in ``root``."""
        found_nests = defaultdict(list)
        # Collect some info visiting the tree rooted in node
        for n in root.children:
            if isinstance(n, For):
                # Track structure of iteration spaces
                loops_infos = FindLoopNests().visit(n, {'node_parent': root})
                for li in loops_infos:
                    loops, loops_parents = zip(*li)
                    # Note that only inner loops can be fused, and that they share
                    # the same parent
                    key = (tuple(l.header for l in loops), loops_parents[-1])
                    found_nests[key].append(loops[-1])

        all_merged, merged_loops = [], []
        # A perfect loop nest L1 is mergeable in a loop nest L2 if
        # 1 - their iteration space is identical; implicitly true because the keys,
        #     in the dictionary, are iteration spaces.
        # 2 - between the two nests, there are no statements that read from values
        #     computed in L1. This is checked next.
        # 3 - there are no read-after-write dependencies between variables written
        #     in L1 and read in L2. This is checked next.
        # Here, to simplify the data flow analysis, the last loop in the tree
        # rooted in node is selected as L2
        for (itspace, parent), loop_nests in found_nests.items():
            if len(loop_nests) == 1:
                # At least two loops are necessary for merging to be meaningful
                continue
            mergeable = []
            merging_in = loop_nests[-1]
            merging_in_reads = SymbolModes().visit(merging_in.body,
                                                   env=SymbolModes.default_env)
            merging_in_reads = [s for s, m in merging_in_reads.items() if m[0] == READ]
            for l in loop_nests[:-1]:
                is_mergeable = True
                # Get the symbols written in /l/
                l_writes = SymbolModes().visit(l.body, env=SymbolModes.default_env)
                l_writes = [s for s, m in l_writes.items() if m[0] == WRITE]
                # Get the symbols written between loop /l/ (excluded) and loop
                # merging_in (included)
                bound_left = parent.children.index(l)+1
                bound_right = parent.children.index(merging_in)
                for n in parent.children[bound_left:bound_right]:
                    in_writes = SymbolModes().visit(n, env=SymbolModes.default_env)
                    in_writes = [s for s, m in in_writes.items() if m[0] == WRITE]
                    # Check condition 2
                    for iw, lw in product(in_writes, l_writes):
                        if self.expr_graph.is_written(iw, lw):
                            is_mergeable = False
                            break
                    # Check condition 3
                    for lw, mir in product(l_writes, merging_in_reads):
                        if lw.symbol == mir.symbol and not lw.rank and not mir.rank:
                            is_mergeable = False
                            break
                    if not is_mergeable:
                        break
                # Track mergeable loops
                if is_mergeable:
                    mergeable.append(l)
            # If there is at least one mergeable loops, do the merging
            for l in reversed(mergeable):
                merged, l_dims, m_dims = self._merge_loops(parent, l, merging_in)
                ast_update_rank(merged, dict(zip(l_dims, m_dims)))
            # Update the lists of merged loops
            all_merged.append((mergeable, merging_in))
            merged_loops.append(merging_in)

        # Reuse temporaries in the merged loops, where possible
        self._simplify(merged_loops)

        # Return the list of merged loops and the resulting loop
        return all_merged


class ExpressionFissioner(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then fission associative
    operations in expressions.
    Fissioned expressions are placed in a separate loop nest."""

    def __init__(self, cut):
        """Initialize the ExpressionFissioner.

        :param cut: number of operands requested to fission expressions."""
        self.cut = cut

    def _split_sum(self, node, parent, is_left, found, sum_count):
        """Exploit sum's associativity to cut node when a sum is found.
        Return ``True`` if a potentially splittable node is found, ``False``
        otherwise."""
        if isinstance(node, Symbol):
            return False
        elif isinstance(node, Par):
            return self._split_sum(node.child, (node, 0), is_left, found, sum_count)
        elif isinstance(node, Prod) and found:
            return False
        elif isinstance(node, Prod) and not found:
            if not self._split_sum(node.left, (node, 0), is_left, found, sum_count):
                return self._split_sum(node.right, (node, 1), is_left, found, sum_count)
            return True
        elif isinstance(node, Sum):
            sum_count += 1
            if not found:
                # Track the first Sum we found while cutting
                found = parent
            if sum_count == self.cut:
                # Perform the cut
                if is_left:
                    parent, parent_leaf = parent
                    parent.children[parent_leaf] = node.left
                else:
                    found, found_leaf = found
                    found.children[found_leaf] = node.right
                return True
            else:
                if not self._split_sum(node.left, (node, 0), is_left, found, sum_count):
                    return self._split_sum(node.right, (node, 1), is_left, found, sum_count)
                return True
        else:
            raise RuntimeError("Split error: found unknown node: %s" % str(node))

    def _sum_fission(self, stmt_info, copy_loops):
        """Split an expression after ``cut`` operands. This results in two
        sub-expressions that are placed in different, although identical
        loop nests if ``copy_loops`` is true; they are placed in the same
        original loop nest otherwise. Return the two split expressions as a
        2-tuple, in which the second element is potentially further splittable."""

        stmt, expr_info = stmt_info
        expr_parent = expr_info.parent
        domain_outerloop, domain_outerparent = expr_info.domain_loops_info[0]

        # Copy the original expression twice, and then split the two copies, that
        # we refer to as ``left`` and ``right``, meaning that the left copy will
        # be transformed in the sub-expression from the origin up to the cut point,
        # and analoguously for right.
        # For example, consider the expression a*b + c*d; the cut point is the sum
        # operator. Therefore, the left part is a*b, whereas the right part is c*d
        stmt_left = dcopy(stmt)
        stmt_right = dcopy(stmt)
        expr_left = Par(stmt_left.children[1])
        expr_right = Par(stmt_right.children[1])
        sleft = self._split_sum(expr_left.children[0], (expr_left, 0), True, None, 0)
        sright = self._split_sum(expr_right.children[0], (expr_right, 0), False, None, 0)

        if sleft and sright:
            index = expr_parent.children.index(stmt)

            # Append the left-split expression, reusing existing loop nest
            expr_parent.children[index] = stmt_left
            split = (stmt_left, MetaExpr(expr_info.type,
                                         expr_parent,
                                         expr_info.loops_info,
                                         expr_info.domain_dims))

            # Append the right-split (remainder) expression
            if copy_loops:
                # Create a new loop nest
                new_domain_outerloop = dcopy(domain_outerloop)
                new_domain_innerloop = new_domain_outerloop.body[0]
                new_domain_innerloop_block = new_domain_innerloop.children[0]
                new_domain_innerloop_block.children[0] = stmt_right
                new_domain_outerloop_info = (new_domain_outerloop,
                                             domain_outerparent)
                new_domain_innerloop_info = (new_domain_innerloop,
                                             new_domain_innerloop_block)
                new_loops_info = expr_info.out_domain_loops_info + \
                    (new_domain_outerloop_info,) + (new_domain_innerloop_info,)
                domain_outerparent.children.append(new_domain_outerloop)
            else:
                # Reuse loop nest created in the previous function call
                expr_parent.children.insert(index, stmt_right)
                new_domain_innerloop_block = expr_parent
                new_loops_info = expr_info.loops_info
            splittable = (stmt_right, MetaExpr(expr_info.type,
                                               new_domain_innerloop_block,
                                               new_loops_info,
                                               expr_info.domain_dims))
            return (split, splittable)
        return ((stmt, expr_info), ())

    def fission(self, stmt, expr_info, copy_loops):
        """Split an expression containing ``x`` summands into ``x/cut`` chunks.
        Each chunk is placed in a separate loop nest if ``copy_loops`` is true,
        in the same loop nest otherwise. In the former case, the split occurs
        in the largest perfect loop nest wrapping the expression in ``stmt``.
        Return a dictionary of all of the split chunks, which associates the split
        statements to meta data (in terms of ``MetaExpr`` objects)

        :param stmt: AST statement containing the expression to be fissioned
        :param expr_info: ``MetaExpr`` object describing the expression in ``stmt``
        :param copy_loops: true if the split expressions should be placed in two
                           separate, adjacent loop nests (iterating, of course,
                           along the same iteration space); false, otherwise."""

        split_stmts = {}
        splittable_stmt = (stmt, expr_info)
        while splittable_stmt:
            split_stmt, splittable_stmt = self._sum_fission(splittable_stmt, copy_loops)
            split_stmts[split_stmt[0]] = split_stmt[1]
        return split_stmts


class ZeroRemover(LoopScheduler):

    """Analyze data dependencies and iteration spaces to remove arithmetic
    operations in loops that iterate over zero-valued blocks. Consequently,
    loop nests can be fissioned and/or merged. For example: ::

        for i = 0, N
          A[i] = C[i]*D[i]
          B[i] = E[i]*F[i]

    If the evaluation of A requires iterating over a block of zero (0.0) values,
    because for instance C and D are block-sparse, then A is evaluated in a
    different, smaller (i.e., with less iterations) loop nest: ::

        for i = 0 < (N-k)
          A[i+k] = C[i+k][i+k]
        for i = 0, N
          B[i] = E[i]*F[i]

    The implementation is based on symbolic execution. Control flow is not
    admitted.
    """

    def __init__(self, exprs, decls, hoisted):
        """Initialize the ZeroRemover.

        :param exprs: the expressions for which zero removal is performed.
        :param decls: lists of declarations visible to ``exprs``.
        :param hoisted: dictionary that tracks hoisted sub-expressions
        """
        self.exprs = exprs
        self.decls = decls
        self.hoisted = hoisted

    def _track_nz_expr(self, node, nz_in_syms, nest):
        """For the expression rooted in ``node``, return iteration space and
        offset required to iterate over non zero-valued blocks. For example: ::

            for i = 0 to N
              for j = 0 to N
                A[i][j] = B[i]*C[j]

        If B along `i` is non-zero in ranges [0, k1] and [k2, k3], while C along
        `j` is non-zero in range [N-k4, N], return the following dictionary: ::

            {i: [(k1, 0), (k3-k2, k2)], j: [(N-(N-k4), N-k4)]}

        That is, for each iteration space variable, return a list of 2-tuples,
        in which the first entry represents the size of the iteration space,
        and the second entry represents the offset in memory to access the
        correct values.
        """

        if isinstance(node, Symbol):
            itspace = OrderedDict([(l.dim, [(l.size, 0)]) for l, p in nest])
            nz_bounds = nz_in_syms.get(node.symbol, ())
            for r, o, nz_bs in zip(node.rank, node.offset, nz_bounds):
                if o[0] != 1:
                    # Cannot handle jumps in memory accesses
                    continue
                try:
                    loop = [l for l, p in nest if l.dim == r][0]
                except:
                    # No loop means constant access along /r/
                    continue
                offset = o[1]
                r_size_ofs = []
                for nz_b in nz_bs:
                    nz_b_size, nz_b_offset = nz_b
                    end = nz_b_size + nz_b_offset
                    start = max(offset, nz_b_offset)
                    r_offset = start - offset
                    r_size = max(min(offset + loop.size, end) - start, 0)
                    r_size_ofs.append((r_size, r_offset))
                itspace[r] = r_size_ofs
            return itspace

        elif isinstance(node, (Par, FunCall)):
            return self._track_nz_expr(node.children[0], nz_in_syms, nest)

        else:
            itspace_l = self._track_nz_expr(node.left, nz_in_syms, nest)
            itspace_r = self._track_nz_expr(node.right, nz_in_syms, nest)
            # Take the intersection of the iteration spaces
            itspace = OrderedDict()
            itspace.update(itspace_l)
            for r_r, r_size_ofs in itspace_r.items():
                if not r_r in itspace:
                    itspace[r_r] = r_size_ofs
                l_size_ofs = itspace[r_r]
                if isinstance(node, (Prod, Div)):
                    to_intersect = product(l_size_ofs, r_size_ofs)
                    size_ofs = [ItSpace(mode=1).intersect(b) for b in to_intersect]
                elif isinstance(node, (Sum, Sub)):
                    size_ofs = ItSpace(mode=1).merge(l_size_ofs + r_size_ofs)
                else:
                    raise RuntimeError("Zero-avoidance: unexpected op %s", str(node))
                itspace[r_r] = size_ofs
            return itspace

    def _track_nz_blocks(self, node, nz_in_syms, nz_info, nest=None, parent=None):
        """Track the propagation of zero-valued blocks in the AST rooted in ``node``

        ``nz_in_syms`` contains, for each known identifier, the ranges of
        its non zero-valued blocks. For example, assuming identifier A is an
        array and has non-zero values in positions [0, k] and [N-k, N], then
        ``nz_in_syms`` will contain an entry {"A": ((0, k), (N-k, N))}.
        If A is modified by some statements rooted in ``node``, then
        ``nz_in_syms["A"]`` will be modified accordingly.

        This method also populates ``nz_info``, which maps loop nests to the
        enclosed symbols' non-zero blocks. For example, given the following
        code: ::

            { // root
              ...
              for i
                for j
                  A = ...
                  B = ...
            }

        After the traversal of the AST, the ``nz_info`` dictionary will look like: ::

            ((<for i>, <for j>), root) -> {A: (i, (nz_along_i)), (j, (nz_along_j))}

        """
        if isinstance(node, (Assign, Incr, Decr)):
            sym, expr = node.children
            symbol, rank = sym.symbol, sym.rank

            # Note: outer, non-perfect loops are discarded for transformation
            # safety. In fact, splitting non-perfect loop nests inherently
            # breaks the code
            nest = tuple([(l, p) for l, p in (nest or []) if is_perfect_loop(l)])
            if not nest:
                return

            # Track the propagation of non zero-valued blocks. If it is not the
            # first time that /symbol/ is encountered, info get merged.
            itspace = self._track_nz_expr(expr, nz_in_syms, nest)
            if not all([r in itspace for r in rank]):
                return
            nz_in_expr = tuple(itspace[r] for r in rank)
            if symbol in nz_in_syms:
                nz_in_expr = tuple([ItSpace(mode=1).merge(flatten(i)) for i in
                                    zip(nz_in_expr, nz_in_syms[symbol])])
            nz_in_syms[symbol] = nz_in_expr

            # Record loop nest bounds and memory offsets for /node/
            nz_info.setdefault(nest, []).append((node, itspace))

        elif isinstance(node, For):
            new_nest = (nest or []) + [(node, parent)]
            self._track_nz_blocks(node.children[0], nz_in_syms, nz_info, new_nest, node)

        elif isinstance(node, (Root, Block)):
            for n in node.children:
                self._track_nz_blocks(n, nz_in_syms, nz_info, nest, node)

        elif isinstance(node, (If, Switch)):
            raise RuntimeError("Unexpected control flow while tracking zero blocks")

    def _reschedule_itspace(self, root):
        """Consider two statements A and B, and their iteration space. If the
        two iteration spaces have

        * Same size and same bounds, then put A and B in the same loop nest: ::

           for i, for j
             W1[i][j] = W2[i][j]
             Z1[i][j] = Z2[i][j]

        * Same size but different bounds, then put A and B in the same loop
          nest, but add suitable offsets to all of the involved iteration
          variables: ::

           for i, for j
             W1[i][j] = W2[i][j]
             Z1[i+k][j+k] = Z2[i+k][j+k]

        * Different size, then put A and B in two different loop nests: ::

           for i, for j
             W1[i][j] = W2[i][j]
           for i, for j  // Different loop bounds
             Z1[i][j] = Z2[i][j]

        A dictionary describing the structure of the new iteration spaces is
        returned.
        """

        # 1) Identify the initial sparsity pattern of the symbols in /root/
        nz_in_syms = {s: d.nonzero for s, d in self.decls.items() if d.nonzero}
        nz_info = OrderedDict()

        # 2) Track propagation of non-zero blocks by symbolic execution. This
        # has the effect of populating /nz_info/
        self._track_nz_blocks(root, nz_in_syms, nz_info)

        # 3) At this point we know where non-zero blocks are located, so we have
        # to create proper loop nests to access them
        track_exprs, new_nz_info = {}, {}
        for nest, stmt_itspaces in nz_info.items():
            loops, loops_parents = zip(*nest)
            fissioned_nests = defaultdict(list)
            # Fission the nest to get rid of computation over zero-valued blocks
            for stmt, itspaces in stmt_itspaces:
                sym, expr = stmt.children
                # For each non zero-valued region iterated over...
                for dim_size_ofs in [zip(itspaces, x) for x in product(*itspaces.values())]:
                    dim_offset = dict([(d, o) for d, (sz, o) in dim_size_ofs])
                    dim_size = tuple([((0, sz), d) for d, (sz, o) in dim_size_ofs])
                    # ...add an offset to /stmt/ to access the correct values
                    new_stmt = ast_update_ofs(dcopy(stmt), dim_offset, increase=True)
                    # ...add /stmt/ to a new, shorter loop nest
                    fissioned_nests[dim_size].append((new_stmt, dim_offset))
                    # ...initialize arrays to 0.0 for correctness
                    if sym.symbol in self.hoisted:
                        self.hoisted[sym.symbol].decl.init = ArrayInit("{0.0}")
                    # ...track fissioned expressions
                    if stmt in self.exprs:
                        track_exprs[new_stmt] = self.exprs[stmt]
            # Generate the fissioned loop nests
            # Note: the dictionary is sorted because smaller loop nests should
            # be executed first, since larger ones depend on them
            for dim_size, stmt_dim_offsets in sorted(fissioned_nests.items()):
                if all([sz == (0, 0) for sz, dim in dim_size]):
                    # Discard empty loop nests
                    continue
                # Create the new loop nest ...
                new_loops = ItSpace(mode=0).to_for(*zip(*dim_size))
                for stmt, _ in stmt_dim_offsets:
                    # ... populate it
                    new_loops[-1].body.append(stmt)
                    # ... and update tracked data
                    if stmt in track_exprs:
                        new_nest = zip(new_loops, loops_parents)
                        self.exprs[stmt] = copy_metaexpr(track_exprs[stmt],
                                                         parent=new_loops[-1].body,
                                                         loops_info=new_nest)
                    self.hoisted.update_stmt(stmt.children[0].symbol,
                                             loop=new_loops[0],
                                             place=loops_parents[0])
                new_nz_info[new_loops[-1]] = stmt_dim_offsets
                # Append the new loops to the root
                insert_at_elem(loops_parents[0].children, loops[0], new_loops[0])
            loops_parents[0].children.remove(loops[0])

        return new_nz_info

    def reschedule(self, root):
        """Restructure the loop nests in ``root`` to avoid computation over
        zero-valued data spaces. This is achieved through symbolic execution
        starting from ``root``. Control flow, in the form of If, Switch, etc.,
        is forbidden."""

        # First, split expressions to maximize the impact of the transformation.
        # This is because different summands may have zero-valued blocks at
        # different offsets
        elf = ExpressionFissioner(cut=1)
        for stmt, expr_info in self.exprs.items():
            if expr_info.is_scalar:
                continue
            elif expr_info.dimension > 1:
                self.exprs.pop(stmt)
                self.exprs.update(elf.fission(stmt, expr_info, False))

        # Perform symbolic execution, track the propagation of non zero-valued
        # blocks, restructure the iteration spaces to avoid useless arithmetic ops
        return self._reschedule_itspace(root)
