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
from expression import copy_metaexpr
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
        to_replace = {}
        for loop in merged_loops:
            for stmt in loop.body:
                ast_replace(stmt, to_replace, copy=True)
                to_replace[stmt.rvalue] = stmt.lvalue

    def merge(self, root):
        """Merge perfect loop nests in ``root``."""
        found_nests = defaultdict(list)
        # Collect iteration spaces visiting the tree rooted in /root/
        for n in root.children:
            if isinstance(n, For):
                retval = FindLoopNests.default_retval()
                loops_infos = FindLoopNests().visit(n, parent=root, ret=retval)
                for li in loops_infos:
                    loops, loops_parents = zip(*li)
                    # Note that only inner loops can be fused, and that they must
                    # share the same parent node
                    key = (tuple(l.header for l in loops), loops_parents[-1])
                    found_nests[key].append(loops[-1])

        all_merged, merged_loops = [], []
        # A perfect loop nest L1 is mergeable in a loop nest L2 if
        # 1 - their iteration space is identical; implicitly true because the keys,
        #     in the dictionary, are iteration spaces.
        # 2 - between the two nests, there are no statements that read/write values
        #     computed in L1. This is checked later
        # 3 - there are no read-after-write dependencies between variables written
        #     in L1 and read in L2. This is checked later
        # In the following, convention is that L2 = /merging_in/, L1 = /l/
        for (itspace, parent), loop_nests in found_nests.items():
            if len(loop_nests) == 1:
                # At least two loops are necessary for merging to be meaningful
                continue
            mergeable = []
            merging_in = loop_nests[-1]
            retval = SymbolModes.default_retval()
            merging_in_reads = SymbolModes().visit(merging_in.body, ret=retval)
            merging_in_reads = [s for s, m in merging_in_reads.items() if m[0] == READ]
            for l in loop_nests[:-1]:
                is_mergeable = True
                # Get the symbols written in /l/
                l_writes = SymbolModes().visit(l.body, ret=SymbolModes.default_retval())
                l_writes = [s for s, m in l_writes.items() if m[0] == WRITE]

                # Check condition 2
                # Get the symbols written between loop /l/ (excluded) and loop
                # merging_in (excluded)
                bound_left = parent.children.index(l)+1
                bound_right = parent.children.index(merging_in)
                for n in parent.children[bound_left:bound_right]:
                    in_writes = SymbolModes().visit(n, ret=SymbolModes.default_retval())
                    in_writes = [s for s, m in in_writes.items()]
                    for iw, lw in product(in_writes, l_writes):
                        if self.expr_graph.is_written(iw, lw):
                            is_mergeable = False
                            break

                # Check condition 3
                for lw, mir in product(l_writes, merging_in_reads):
                    if lw.symbol == mir.symbol and not lw.rank and not mir.rank:
                        is_mergeable = False
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

        # Reuse temporaries in merged loops
        self._simplify(merged_loops)

        return all_merged


class ExpressionFissioner(LoopScheduler):

    """Split expressions embedded in a loop nest."""

    def __init__(self, **kwargs):
        """Initialize the ExpressionFissioner.

        :arg kwargs:
            * cut: the number of operands an expression should be fissioned into
            * match: a list of subexpressions that should be cut from the input
                expression. ``cut`` is ignored if ``match`` is provided.
            * loops: a value in ['all', 'expr', 'none']. 'all' means that an
                expression is split, and its chunks are placed in separate loop
                nests. 'expr' implies that the chunks share the outer loops of the
                expression, particularly those not belonging to its domain. 'none'
                means that all chunks are placed within the original loop nest
            * perfect: if True, create perfect loop nests. This means that any
                new loop nest in which a chunk is placed is purged from any extra
                statement (apart, obviously, from the chunk itself)
        """
        self.cut = kwargs.get('cut', -1)
        self.match = [str(i) for i in kwargs.get('match', [])]
        self.loops = kwargs.get('loops', 'expr')
        self.perfect = kwargs.get('perfect', False)

        if 'match' in kwargs:
            self.cutter = self.CutterMatch(self)
        elif self.cut > 0:
            self.cutter = self.CutterSum(self)
        else:
            raise RuntimeError("Must specify a `cut` or a `match`.")

    class Cutter():

        def __init__(self, expr_fissioner):
            self.expr_fissioner = expr_fissioner

        def cut(self, node):
            """
            Split ``node`` into /two halves/, called /split/ and /remainder/

            For example, consider the expression a*b + c*d; if the expression is cut
            into chunks containing only one operand (i.e., self.cut=1), then we have
            precisely two chunks, /split/ = a*b, /remainder/ = c*d

            If the input expression is a*b + c*d + e*f, and still self.cut=1, then we
            have two chunks, /split/ = a*b, /remainder/ = c*d + e*f; that is,
            /remainder/ always contains the subexpression after the fission point
            """
            self._success = False
            left = dcopy(node)
            self._cut(left.children[1], left, 'split')

            self._success = False
            right = dcopy(node)
            self._cut(right.children[1], right, 'remainder')

            return left, right

    class CutterSum(Cutter):

        def _cut(self, node, parent, side, topsum=None):
            if isinstance(node, (Symbol, FunCall)):
                return 0

            elif isinstance(node, (Par, Div)):
                return self._cut(node.children[0], node, side, topsum)

            elif isinstance(node, Prod):
                if topsum:
                    return 0
                if self._cut(node.left, node, side, topsum) == 0:
                    return self._cut(node.right, node, side, topsum)
                # Prods zero the sum/sub counter
                return 0

            elif isinstance(node, (Sum, Sub)):
                topsum = topsum or (parent, parent.children.index(node))
                counter = 1
                counter += self._cut(node.left, node, side, topsum)
                counter += self._cut(node.right, node, side, topsum)
                if not self._success and counter >= self.expr_fissioner.cut:
                    # We now are on the topleft sum of this sub-expression such
                    # that enough sum/sub have been encountered
                    if not parent:
                        return 0
                    self._success = True
                    if side == 'split':
                        topsum[0].children[topsum[1]] = node.left
                    else:
                        right = Neg(node.right) if isinstance(node, Sub) else node.right
                        parent.children[parent.children.index(node)] = right
                    return counter
                else:
                    return counter

            else:
                raise RuntimeError("Fission error: found unknown node: %s" % str(node))

        def cut(self, node, expr_info):
            left, right = ExpressionFissioner.Cutter.cut(self, node)
            if self._success:
                index = expr_info.parent.children.index(node)

                # Append /left/ to the original loop nest
                expr_info.parent.children[index] = left
                split = (left, copy_metaexpr(expr_info))

                # Append /right/ ...
                if self.expr_fissioner.loops in ['expr', 'all']:
                    # ... in a new loop nest ...
                    right_info = self.expr_fissioner._embedexpr(right, expr_info)
                else:
                    # ... to the original loop nest
                    expr_info.parent.children.insert(index, right)
                    right_info = copy_metaexpr(expr_info)
                splittable = (right, right_info)

                return (split, splittable)
            return ((node, expr_info), ())

    class CutterMatch(Cutter):

        def __init__(self, expr_fissioner):
            ExpressionFissioner.Cutter.__init__(self, expr_fissioner)
            self.matched = []

        def _cut(self, node, parent, side, topsum=None):
            if not self._success and str(node) in self.expr_fissioner.match:
                # We initially assume that the found 'match' corresponds
                # to the entire node provided as input to the /CutterMatch/.
                # Recurring back, we might switch /_success/ to 'match_and_cut',
                # if /node/ actually was a summand of a Sum/Sub
                self._success = 'match'
                return node

            elif isinstance(node, (Symbol, FunCall)):
                return None

            elif isinstance(node, Par):
                return self._cut(node.child, node, side, topsum)

            elif isinstance(node, Div):
                return self._cut(node.left, node, side)

            elif isinstance(node, Prod):
                cutting = self._cut(node.left, node, side)
                if cutting:
                    # Found a match /within/ /node.left/; for correctness, we
                    # need to be sure we will be cutting the whole Prod, so we
                    # return /node/ instead of /cutting/.
                    return node
                cutting = self._cut(node.right, node, side)
                if cutting:
                    # Same as above
                    return node
                return None

            elif isinstance(node, (Sum, Sub)):
                topsum = topsum or (parent, parent.children.index(node))
                # Find out if one of the two children is cuttable
                cutting = self._cut(node.left, node, side, topsum)
                if cutting and side == 'remainder':
                    # Need to swap
                    cutting = node.right
                elif not cutting:
                    cutting = self._cut(node.right, node, side, topsum)
                    if cutting and side == 'remainder':
                        # Need to swap
                        cutting = node.left
                if not cutting:
                    return None
                # Adjust if a Sub
                if isinstance(node, Sub) and cutting == node.right:
                    cutting = Neg(cutting)
                self._success = 'match_and_cut'
                if side == 'split':
                    # In a tree of Sum/Subs, only the /top/ Sum/Sub performs the
                    # actual cut, while the others just propagate upwards the
                    # notification "a cut point was found"
                    if parent == topsum[0]:
                        topsum[0].children[topsum[1]] = cutting
                        return parent
                    else:
                        return cutting
                else:
                    parent.children[parent.children.index(node)] = cutting
                    return None

            else:
                raise RuntimeError("Fission error: found unknown node: %s" % str(node))

        def cut(self, node, expr_info):
            left, right = ExpressionFissioner.Cutter.cut(self, node)

            if self._success == 'match_and_cut':
                # Append /left/ to a new loop nest
                split = (left, self.expr_fissioner._embedexpr(left, expr_info))
                self.matched.append(left)

                # Append /right/ to the original loop nest
                index = expr_info.parent.children.index(node)
                expr_info.parent.children[index] = right
                splittable = (right, copy_metaexpr(expr_info))
                return (split, splittable)

            elif self._success == 'match':
                # A match was actualy found, but there's just nothing to cut
                # (i.e., the /match/ is a direct child of /node/)
                self.matched.append(node)

            return ((node, expr_info), ())

    def _embedexpr(self, stmt, expr_info):
        """Build a loop nest for ``stmt`` and return its :class:`MetaExpr` object."""
        if self.loops == 'none':
            return copy_metaexpr(expr_info)

        # Handle the domain loops
        domain_loops = ItSpace(mode=2).to_for(expr_info.domain_loops, stmts=[stmt])
        domain_outerloop = domain_loops[0]

        # Handle the out-domain loops
        if self.loops == 'all' and expr_info.out_domain_loops_info:
            out_domain_loop, out_domain_loop_parent = expr_info.out_domain_loops_info[0]
            index = out_domain_loop.body.index(expr_info.domain_loops[0])
            out_domain_loop = dcopy(out_domain_loop)
            if self.perfect:
                out_domain_loop.body[:] = [domain_outerloop]
            else:
                out_domain_loop.body[index] = domain_outerloop
            out_domain_loops_info = ((out_domain_loop, out_domain_loop_parent),)
            domain_outerloop_parent = out_domain_loop.children[0]
        else:
            out_domain_loops_info = expr_info.out_domain_loops_info
            domain_outerloop_parent = expr_info.domain_loops_parents[0]

        # Build new loops info
        finder, env = FindLoopNests(), {'node_parent': domain_outerloop_parent}
        loops_info = out_domain_loops_info
        loops_info += tuple(finder.visit(domain_outerloop, env=env)[0])

        # Append the newly created loop nest
        if self.loops == 'all' and expr_info.out_domain_loops_info:
            expr_info.outermost_parent.children.append(out_domain_loop)
        else:
            domain_outerloop_parent.children.append(domain_outerloop)

        # Finally, create and return the MetaExpr object
        parent = loops_info[-1][0].children[0]
        return copy_metaexpr(expr_info, parent=parent, loops_info=loops_info)

    @property
    def matched(self):
        return self.cutter.matched if self.match else []

    def fission(self, stmt, expr_info):
        """Split, or fission, an expression ``stmt``, whose metadata are provided
        through ``expr_info``.

        Return a dictionary mapping expression chunks to :class:`MetaExpr` objects.

        :arg stmt: the expression to be fissioned
        :arg expr_info: ``MetaExpr`` object describing ``stmt``
        """
        exprs = OrderedDict()
        splittable = (stmt, expr_info)
        while splittable:
            split, splittable = self.cutter.cut(*splittable)
            exprs[split[0]] = split[1]
        return exprs


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

    def _track_nz_expr(self, node, nz_syms, nest):
        """For the expression rooted in ``node``, return iteration space and
        offset required to iterate over non zero-valued blocks. For example: ::

            for i = 0 to N
              for j = 0 to N
                A[i][j] = B[i]*C[j]

        If B along `i` is non-zero in ranges [0, k1] and [k2, k3], while C along
        `j` is non-zero in range [N-k4, N], return the intersection of the non-zero
        regions as: ::

            [(('i', k1, 0), ('j', N-(N-k4), N-k4))),
             (('i', k3-k2, k2), ('j', N-(N-k4), N-k4))]

        That is, for each iteration space variable, return a list of 2-tuples,
        in which the first entry represents the size of the iteration space,
        and the second entry represents the offset in memory to access the
        correct values.
        """

        if isinstance(node, Symbol):
            itspace = []
            def_itspace = [tuple((l.dim, (l.size, 0)) for l, p in nest)]
            nz_bounds = zip(*nz_syms.get(node.symbol, []))
            for i, (r, o, nz_bs) in enumerate(zip(node.rank, node.offset, nz_bounds)):
                if o[0] != 1 or isinstance(o[1], str):
                    # Cannot handle jumps nor non-integer offsets
                    continue
                try:
                    # Let's see if I know the loop which defines the iteration
                    # space of dimension /r/ ...
                    loop = [l for l, p in nest if l.dim == r][0]
                except:
                    # ... whops! No, so I just assume it covers the entire
                    # non zero-valued region
                    itspace.append([(r, nz_b) for nz_b in nz_bs])
                    continue
                # Now I can intersect the loop's iteration space with the non
                # zero-valued regions
                offset = o[1]
                r_size_ofs = []
                for nz_b in nz_bs:
                    nz_b_size, nz_b_offset = nz_b
                    end = nz_b_size + nz_b_offset
                    start = max(offset, nz_b_offset)
                    r_offset = start - offset
                    r_size = max(min(offset + loop.size, end) - start, 0)
                    r_size_ofs.append((r, (r_size, r_offset)))
                itspace.append(r_size_ofs)
            itspace = zip(*itspace) or def_itspace
            return itspace

        elif isinstance(node, (Par, FunCall)):
            return self._track_nz_expr(node.children[0], nz_syms, nest)

        else:
            itspace_l = self._track_nz_expr(node.left, nz_syms, nest)
            itspace_r = self._track_nz_expr(node.right, nz_syms, nest)
            itspace = OrderedDict()
            for l in itspace_l:
                for i, size_ofs in l:
                    itspace.setdefault(i, []).append(size_ofs)
            asdict = OrderedDict()
            for r in itspace_r:
                for i, size_ofs in r:
                    asdict.setdefault(i, []).append(size_ofs)
            itspace_r = asdict
            for i, size_ofs in itspace_r.items():
                if i not in itspace:
                    itspace[i] = size_ofs
                elif isinstance(node, (Prod, Div)):
                    result = []
                    for j in product(itspace[i], size_ofs):
                        # Products over zero-valued regions are ininfluent
                        result += [ItSpace(mode=1).intersect(j)]
                    itspace[i] = result
                elif isinstance(node, (Sum, Sub)):
                    # Sums over zeros remove the zero-valued region (in other words,
                    # the non zero-valued regions get /merged/)
                    itspace[i] = ItSpace(mode=1).merge(itspace[i] + size_ofs)
                else:
                    raise RuntimeError("Zero-avoidance: unexpected op %s", str(node))
            itspace = [zip(itspace, i) for i in product(*itspace.values())]
            itspace = list(set([tuple(i) for i in itspace]))
            return itspace

    def _track_nz_blocks(self, node, nz_syms, nz_info, nest=None, parent=None):
        """Track the propagation of zero-valued blocks in the AST rooted in ``node``

        ``nz_syms`` contains, for each known identifier, the ranges of
        its non zero-valued blocks. For example, assuming identifier A is an
        array and has non-zero values in positions [0, k] and [N-k, N], then
        ``nz_syms`` will contain an entry {"A": ((0, k), (N-k, N))}.
        If A is modified by some statements rooted in ``node``, then
        ``nz_syms["A"]`` will be modified accordingly.

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

            # Note: outer, non-perfect loops are discarded for transformation
            # safety, because splitting (which is a consequence of zero-removal)
            # non-perfect loop nests is unsafe
            nest = tuple([(l, p) for l, p in (nest or []) if is_perfect_loop(l)])
            if not nest:
                return

            # Track the propagation of non zero-valued blocks: ...
            # ... within the rvalue
            itspaces = self._track_nz_expr(expr, nz_syms, nest)
            for i in itspaces:
                # ... and then through the lvalue (merging overlaps)
                nz_expr = tuple(dict(i).get(r) for r in sym.rank if not isinstance(r, int))
                if any(j is None for j in nz_expr):
                    return
                nz_node = list(nz_syms.setdefault(sym.symbol, [nz_expr]))
                if not nz_expr:
                    continue
                merged = False
                for e, j in enumerate(nz_node):
                    # Merging condition: complete overlap in all dimensions but
                    # the innermost one, for which partial overlap is accepted
                    inner_merge = ItSpace(mode=1).merge([nz_expr[-1], j[-1]])
                    if len(inner_merge) == 1 and \
                            all(ItSpace(mode=1).intersect([m, n]) == m for m, n in
                                zip(nz_expr[:-1], j[:-1])):
                        nz_syms[sym.symbol][e] = j[:-1] + tuple(inner_merge)
                        merged = True
                        break
                if not merged:
                    nz_syms[sym.symbol].append(nz_expr)

            # Record loop nest bounds and memory offsets for /node/
            dims = [l.dim for l, p in nest]
            itspaces = [tuple(j for j in i if j[0] in dims) for i in itspaces]
            nz_info.setdefault(nest, []).append((node, itspaces))

        elif isinstance(node, For):
            new_nest = (nest or []) + [(node, parent)]
            self._track_nz_blocks(node.children[0], nz_syms, nz_info, new_nest, node)

        elif isinstance(node, (Root, Block)):
            for n in node.children:
                self._track_nz_blocks(n, nz_syms, nz_info, nest, node)

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
        nz_info = OrderedDict()

        # 1) Elaborate the initial sparsity pattern of the symbols in /root/
        nz_syms = defaultdict(list)
        for s, d in self.decls.items():
            if not d.nonzero:
                continue
            for nz_b in product(*d.nonzero):
                entries = [range(i[1], i[1] + i[0]) for i in nz_b]
                if not np.all(d.init.values[np.ix_(*entries)] == 0.0):
                    nz_syms[s].append(nz_b)

        # 2) Track the propagation of non zero-valued blocks through symbolic
        # execution. This populates /nz_info/ and updates /nz_syms/
        self._track_nz_blocks(root, nz_syms, nz_info)

        # 3) At this point we know where non-zero blocks are located, so we have
        # to create proper loop nests to access them
        new_exprs, new_nz_info = OrderedDict(), OrderedDict()
        for nest, stmt_itspaces in nz_info.items():
            loops, loops_parents = zip(*nest)
            fissioned_nests = defaultdict(list)
            # Fission the nest to get rid of computation over zero-valued blocks
            for stmt, itspaces in stmt_itspaces:
                sym, expr = stmt.children
                # For each non zero-valued region iterated over...
                for i in itspaces:
                    dim_offset = {d: o for d, (sz, o) in i}
                    dim_size = tuple(((0, dict(i)[l.dim][0]), l.dim) for l in loops)
                    # ...add an offset to /stmt/ to access the correct values
                    new_stmt = ast_update_ofs(dcopy(stmt), dim_offset, increase=True)
                    # ...add /stmt/ to a new, shorter loop nest
                    fissioned_nests[dim_size].append((new_stmt, dim_offset))
                    # ...initialize arrays to 0.0 for correctness
                    if sym.symbol in self.hoisted:
                        self.hoisted[sym.symbol].decl.init = ArrayInit(np.array([0.0]))
                    # ...track fissioned expressions
                    if stmt in self.exprs:
                        new_exprs[new_stmt] = self.exprs[stmt]
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
                    if stmt in new_exprs:
                        new_nest = zip(new_loops, loops_parents)
                        new_exprs[stmt] = copy_metaexpr(new_exprs[stmt],
                                                        parent=new_loops[-1].body,
                                                        loops_info=new_nest)
                    self.hoisted.update_stmt(stmt.children[0].symbol,
                                             loop=new_loops[0],
                                             place=loops_parents[0])
                new_nz_info[tuple(new_loops)] = stmt_dim_offsets
                # Append the new loops to the root
                insert_at_elem(loops_parents[0].children, loops[0], new_loops[0])
            loops_parents[0].children.remove(loops[0])

        self.exprs.clear()
        self.exprs.update(new_exprs)
        return nz_syms, new_nz_info

    def _shrink(self, root, nz_syms, nz_info):
        references = SymbolReferences().visit(root, ret=SymbolReferences.default_retval())
        dataspaces = defaultdict(list)

        # Calculate the dataspaces
        for nest, stmt_dim_offsets in nz_info.items():
            nest = {l.dim: l for l in nest}
            for stmt, dim_offset in stmt_dim_offsets:
                symbols = FindInstances(Symbol).visit(stmt)[Symbol]
                symbols = [s for s in symbols if s.symbol in self.decls and
                           isinstance(self.decls[s.symbol].init, SparseArrayInit)]
                for s in symbols:
                    decl = self.decls[s.symbol]
                    # Each dataspace is stored as a 2-tuple (size, offset)
                    dataspace = []
                    for r, o, d_size in zip(s.rank, s.offset, decl.sym.rank):
                        size = nest[r].size if r in nest else d_size
                        dataspace.append((size, o[1]))
                    dataspaces[s.symbol].append(tuple(dataspace))

        # For each symbol, and for each of its dimension, aggregate the dataspaces
        # if they are contiguous
        for symbol, dataspace in dataspaces.items():
            decl = self.decls[symbol]
            merged_dataspace = []
            for regions, size in zip(zip(*dataspace), decl.sym.rank):
                merged_dataspace.append(ItSpace(mode=1).merge(regions, within=size)[0])
            dataspaces[symbol] = tuple(merged_dataspace)

        # Now that we know the dataspaces, we should see if they are smaller than
        # the actual array size
        for symbol, dataspace in dataspaces.items():
            decl = self.decls[symbol]
            sections = [range(d[1], d[1] + d[0]) for d in dataspace]
            values = decl.init.values[np.ix_(*sections)]
            if values.shape != decl.init.values.shape:
                # Set the shrunk values ...
                decl.init.values = values
                decl.sym.rank = values.shape
                # ... changes any relevant offsets
                s_references = [s[0] for s in references[symbol] if s[0] is not decl.sym]
                for s in s_references:
                    offset = []
                    for i, section in enumerate(sections):
                        # Is the shrunk array still starting from the old base?
                        # If not, update the offset of each symbol occurence
                        if s.offset[i][1] > 0 and section[0] > 0:
                            offset.append((s.offset[i][0], s.offset[i][1] - section[0]))
                        else:
                            offset.append(s.offset[i])
                    s.offset = tuple(offset)
                # ... and update the /nz_syms/ dictionary
                nz_syms[symbol] = tuple([(i, 0)] for i in values.shape)

        # Finally, we can merge tables that became identical after shrinking
        mapper = defaultdict(list)
        for symbol, dataspace in dataspaces.items():
            values = self.decls[symbol].init.values
            mapper[values.tostring()].append(symbol)
        for values, symbols in mapper.items():
            to_replace = {s: symbols[0] for s in symbols[1:]}
            for replacing, target in to_replace.items():
                for s, p in references[replacing]:
                    s.symbol = target
                    if isinstance(p, Decl):
                        root.children.remove(p)
                # Clean up
                self.decls.pop(replacing)
                if replacing in self.hoisted:
                    self.hoisted.pop(replacing)

    def _recombine(self, nz_info):
        """Recombine expressions writing to the same lvalue."""
        new_exprs = OrderedDict()
        ops = {Incr: Sum, Decr: Sub, IMul: Prod}

        for nest, stmt_dim_offsets in nz_info.items():
            mapper = OrderedDict()
            for stmt, dim_offsets in stmt_dim_offsets:
                sym, expr = stmt.children
                if type(stmt) in ops:
                    # The /key/ means: I'm in the same loop nest, I'm writing to
                    # the same symbol, and in particular to the same symbol
                    # locations, and I'm doing an associative AugmentedAssignment.
                    key = (str(sym), type(stmt))
                    mapper.setdefault(key, []).append(stmt)

            for (_, op), stmts in mapper.items():
                exprs = [i.children[1] for i in stmts]
                for i in stmts:
                    nest[-1].body.remove(i)
                stmt = op(i.children[0], ast_make_expr(ops[op], exprs))
                nest[-1].body.append(stmt)
                # Update the tracked expressions, if necessary
                if all(i in self.exprs for i in stmts):
                    new_exprs[stmt] = self.exprs[i]

        self.exprs.clear()
        self.exprs.update(new_exprs)

    def reschedule(self, root):
        """Restructure the loop nests in ``root`` to avoid computation over
        zero-valued data spaces. This is achieved through symbolic execution
        starting from ``root``. Control flow, in the form of If, Switch, etc.,
        is forbidden."""

        # First, split expressions to maximize the impact of the transformation.
        # This is because different summands may have zero-valued blocks at
        # different offsets
        elf = ExpressionFissioner(cut=1, loops='none')
        for stmt, expr_info in self.exprs.items():
            if expr_info.is_scalar:
                continue
            self.exprs.pop(stmt)
            self.exprs.update(elf.fission(stmt, expr_info))

        # Perform symbolic execution, track the propagation of non zero-valued
        # blocks, restructure the iteration spaces to avoid useless arithmetic ops
        nz_syms, nz_info = self._reschedule_itspace(root)

        # Shrink sparse arrays by removing zero-valued regions
        self._shrink(root, nz_syms, nz_info)

        # Finally, "inline" the expressions that were originally split
        self._recombine(nz_info)

        return nz_syms
