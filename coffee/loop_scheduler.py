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
import itertools
from copy import deepcopy as dcopy
from warnings import warn as warning

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

    def __init__(self, root, expr_graph):
        """Initialize the SSALoopMerger.

        :param expr_graph: the ExpressionGraph tracking all data dependencies
                           involving identifiers that appear in ``root``.
        :param root: the node where loop scheduling takes place."""
        self.root = root
        self.expr_graph = expr_graph
        self.merged_loops = []

    def _accessed_syms(self, node, mode):
        """Return a list of symbols that are being accessed in the tree
        rooted in ``node``. If ``mode == 0``, looks for written to symbols;
        if ``mode==1`` looks for read symbols."""
        if isinstance(node, Symbol):
            return [node]
        elif isinstance(node, FlatBlock):
            return []
        elif isinstance(node, (Assign, Incr, Decr)):
            if mode == 0:
                return self._accessed_syms(node.children[0], mode)
            elif mode == 1:
                return self._accessed_syms(node.children[1], mode)
        elif isinstance(node, Decl):
            if mode == 0 and node.init and not isinstance(node.init, EmptyStatement):
                return self._accessed_syms(node.sym, mode)
            else:
                return []
        else:
            accessed_syms = []
            for n in node.children:
                accessed_syms.extend(self._accessed_syms(n, mode))
            return accessed_syms

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

    def _update_dims(self, node, dims):
        """Change the iteration variables in the nodes rooted in ``node``
        according to the map defined in ``dims``, which is a dictionary
        from old_iteration_variable to new_iteration_variable. For example,
        given dims = {'i': 'j'} and a node "A[i] = B[i]", change the node
        into "A[j] = B[j]"."""
        if isinstance(node, Symbol):
            new_rank = []
            for r in node.rank:
                new_rank.append(r if r not in dims else dims[r])
            node.rank = tuple(new_rank)
        elif not isinstance(node, FlatBlock):
            for n in node.children:
                self._update_dims(n, dims)

    def merge(self):
        """Merge perfect loop nests rooted in ``self.root``."""
        found_nests = defaultdict(list)
        # Collect some info visiting the tree rooted in node
        for n in self.root.children:
            if isinstance(n, For):
                # Track structure of iteration spaces
                retval = FindLoopNests.default_retval()
                loops_infos = FindLoopNests().visit(n, parent=self.root, ret=retval)
                for li in loops_infos:
                    loops, loops_parents = zip(*li)
                    # Note that only inner loops can be fused, and that they share
                    # the same parent
                    key = (itspace_from_for(loops, mode=0), loops_parents[-1])
                    found_nests[key].append(loops[-1])

        all_merged = []
        # A perfect loop nest L1 is mergeable in a loop nest L2 if
        # 1 - their iteration space is identical; implicitly true because the keys,
        #     in the dictionary, are iteration spaces.
        # 2 - between the two nests, there are no statements that read from values
        #     computed in L1. This is checked next.
        # 3 - there are no read-after-write dependencies between variables written
        #     in L1 and read in L2. This is checked next.
        # Here, to simplify the data flow analysis, the last loop in the tree
        # rooted in node is selected as L2
        for itspace_parent, loop_nests in found_nests.items():
            if len(loop_nests) == 1:
                # At least two loops are necessary for merging to be meaningful
                continue
            itspace, parent = itspace_parent
            mergeable = []
            merging_in = loop_nests[-1]
            merging_in_read_syms = self._accessed_syms(merging_in, 1)
            for ln in loop_nests[:-1]:
                is_mergeable = True
                # Get the symbols written to in the loop nest ln
                ln_written_syms = self._accessed_syms(ln, 0)
                # Get the symbols written to between loop ln (excluded) and
                # loop merging_in (included)
                bound_left = parent.children.index(ln)+1
                bound_right = parent.children.index(merging_in)
                _written_syms = flatten([self._accessed_syms(l, 0) for l in
                                         parent.children[bound_left:bound_right]])
                # Check condition 2
                for ws, lws in itertools.product(_written_syms, ln_written_syms):
                    if self.expr_graph.is_written(ws, lws):
                        is_mergeable = False
                        break
                # Check condition 3
                for lws, mirs in itertools.product(ln_written_syms,
                                                   merging_in_read_syms):
                    if lws.symbol == mirs.symbol and not lws.rank and not mirs.rank:
                        is_mergeable = False
                        break
                # Track mergeable loops
                if is_mergeable:
                    mergeable.append(ln)
            # If there is at least one mergeable loops, do the merging
            for l in reversed(mergeable):
                merged, l_dims, m_dims = self._merge_loops(parent, l, merging_in)
                self._update_dims(merged, dict(zip(l_dims, m_dims)))
            # Update the lists of merged loops
            all_merged.append((mergeable, merging_in))
            self.merged_loops.append(merging_in)

        # Return the list of merged loops and the resulting loop
        return all_merged

    def simplify(self):
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
        for loop in self.merged_loops:
            block = loop.body
            for stmt in block:
                sym, expr = stmt.children
                replace_expr(expr.children[0], expr, 0, loop.dim, hoisted_expr)
                hoisted_expr[str(expr)] = sym


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


class ZeroLoopScheduler(LoopScheduler):

    """Analyze data dependencies, iteration spaces, and domain-specific
    information to perform symbolic execution of the code so as to
    determine how to restructure the loop nests to skip iteration over
    zero-valued columns.
    This implies that loops can be fissioned or merged. For example: ::

        for i = 0, N
          A[i] = C[i]*D[i]
          B[i] = E[i]*F[i]

    If the evaluation of A requires iterating over a region of contiguous
    zero-valued columns in C and D, then A is computed in a separate (smaller)
    loop nest: ::

        for i = 0 < (N-k)
          A[i+k] = C[i+k][i+k]
        for i = 0, N
          B[i] = E[i]*F[i]
    """

    def __init__(self, exprs, expr_graph, decls, hoisted):
        """Initialize the ZeroLoopScheduler.

        :param exprs: the expressions for which the zero-elimination is performed.
        :param expr_graph: the ExpressionGraph tracking all data dependencies involving
                           identifiers that appear in ``root``.
        :param decls: lists of declarations visible to ``exprs``.
        :param hoisted: dictionary that tracks hoisted sub-expressions
        """
        self.exprs = exprs
        self.expr_graph = expr_graph
        self.decls = decls
        self.hoisted = hoisted

    def _merge_dims_nz_bounds(self, dim_nz_bounds_l, dim_nz_bounds_r):
        """Given two dictionaries associating iteration variables to ranges
        of non-zero columns, merge the two dictionaries by combining ranges
        along the same iteration variables and return the merged dictionary.
        For example: ::

            dict1 = {'j': [(1,3), (5,6)], 'k': [(5,7)]}
            dict2 = {'j': [(3,4)], 'k': [(1,4)]}
            dict1 + dict2 -> {'j': [(1,6)], 'k': [(1,7)]}
        """
        new_dim_nz_bounds = {}
        for dim, nz_bounds in dim_nz_bounds_l.items():
            if dim.isdigit():
                # Skip constant dimensions
                continue
            # Compute the union of non-zero bounds along the same
            # iteration variable. Unify contiguous regions (for example,
            # [(1,3), (4,6)] -> [(1,6)]
            new_nz_bounds = nz_bounds + dim_nz_bounds_r.get(dim, ())
            merged_nz_bounds = list(itspace_merge(new_nz_bounds))
            new_dim_nz_bounds[dim] = merged_nz_bounds
        return new_dim_nz_bounds

    def _init_decl_to_zero(self, node, nz_in_syms, ofs, itspace):
        """Scan each variable ``v`` in ``node``: if non-initialized elements in ``v``
        are touched as iterating along ``itspace``, initialize ``v`` to 0.0."""

        # Determine the symbols accessed in node and their non-zero regions
        symbols_nz_regions = []
        syms = FindInstances(Symbol).visit(node)[Symbol]
        syms = [s for s in syms if nz_in_syms.get(s.symbol)]
        for s in syms:
            nz_regions = dict(zip(s.rank, nz_in_syms[s.symbol]))
            symbols_nz_regions.append((s.symbol, nz_regions))

        # If iteration space along which they are accessed is bigger than the
        # non-zero region, hoisted symbols must be initialized to zero
        for symbol, nz_regions in symbols_nz_regions:
            if not self.hoisted.get(symbol):
                continue
            for dim, size in itspace:
                dim_nz_regions = nz_regions.get(dim)
                dim_ofs = ofs.get(dim)
                if not dim_nz_regions or dim_ofs is None:
                    # Sym does not iterate along this iteration variable, so skip
                    # the check
                    continue
                iteration_ok = False
                # Check that the iteration space actually corresponds to one of the
                # non-zero regions in the symbol currently analyzed
                for dim_nz_region in dim_nz_regions:
                    init_nz_reg, end_nz_reg = dim_nz_region
                    if dim_ofs == init_nz_reg and size == end_nz_reg + 1 - init_nz_reg:
                        iteration_ok = True
                        break
                if not iteration_ok:
                    # Iterating over a non-initialized region, need to zero it
                    self.hoisted[symbol].decl.init = ArrayInit("{0.0}")

    def _track_nz_expr(self, node, nz_in_syms):
        """Return the first and last indices assumed by the iteration variables
        appearing in ``node`` over regions of non-zero columns. For example,
        consider the following node, particularly its right-hand side: ::

            A[i][j] = B[i]*C[j]

        If B over i is non-zero in the ranges [0, k1] and [k2, k3], while C over
        j is non-zero in the range [N-k4, N], then return a dictionary: ::

            {i: ((0, k1), (k2, k3)), j: ((N-k4, N),)}

        If there are no zero-columns, return {}."""
        if isinstance(node, Symbol):
            if any([o != (1, 0) for o in node.offset]):
                raise RuntimeError("Zeros error: offsets not supported: %s" % str(node))
            nz_bounds = nz_in_syms.get(node.symbol)
            return dict(zip(node.rank, nz_bounds)) if nz_bounds else {}
        elif isinstance(node, (Par, FunCall)):
            return self._track_nz_expr(node.children[0], nz_in_syms)
        else:
            dim_nz_bounds_l = self._track_nz_expr(node.children[0], nz_in_syms)
            dim_nz_bounds_r = self._track_nz_expr(node.children[1], nz_in_syms)
            if isinstance(node, (Prod, Div)):
                # Merge the non-zero bounds of different iteration variables
                # within the same dictionary
                return dict(dim_nz_bounds_l.items() + dim_nz_bounds_r.items())
            elif isinstance(node, Sum):
                return self._merge_dims_nz_bounds(dim_nz_bounds_l, dim_nz_bounds_r)
            else:
                raise RuntimeError("Zeros error: unsupported operation: %s" % str(node))

    def _track_nz_blocks(self, node, nz_in_syms, nz_info, loop_nest=()):
        """Track the propagation of zero blocks along the computation which is
        rooted in ``node``.

        ``nz_in_syms`` contains, for each known identifier, the ranges of
        its zero blocks. For example, assuming identifier A is an array and has
        non-zero values in positions from 0 to k and from N-k to N, then
        ``nz_in_syms`` will contain an entry {"A": ((0, k), (N-k, N))}.
        If A is modified by some statements rooted in ``node``, then
        ``nz_in_syms["A"]`` will be modified accordingly.

        This method also updates ``nz_info``, which maps loop nests to the
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
            dim_nz_bounds = self._track_nz_expr(expr, nz_in_syms)
            if not all([r in dim_nz_bounds for r in rank]):
                return
            # Reflect the propagation of non-zero blocks in /symbol/. If /symbol/ had
            # already been encountered, non-zero bounds get merged together.
            nz_in_expr = tuple(dim_nz_bounds[r] for r in rank)
            if symbol in nz_in_syms:
                nz_in_expr = tuple([itspace_merge(flatten(i)) for i in
                                    zip(nz_in_expr, nz_in_syms[symbol])])
            nz_in_syms[symbol] = nz_in_expr
            if loop_nest:
                # Track the propagation of non-zero blocks in this specific
                # loop nest. Outer loops, i.e. loops that have non been
                # encountered as visiting from the root, are discarded.
                dim_nz_bounds = dict([(k, v) for k, v in dim_nz_bounds.items()
                                      if k in [l.dim for l in loop_nest]])
                nz_info.setdefault(loop_nest, []).append((node, dim_nz_bounds))

        elif isinstance(node, For):
            new_loop_nest = loop_nest + (node,)
            self._track_nz_blocks(node.children[0], nz_in_syms, nz_info, new_loop_nest)

        elif isinstance(node, (Root, Block)):
            for n in node.children:
                self._track_nz_blocks(n, nz_in_syms, nz_info, loop_nest)

        elif isinstance(node, (If, Switch)):
            raise RuntimeError("Unexpected control flow while tracking zero blocks")

    def _track_nz(self, root):
        """Track the propagation of non-zero valued blocks in the AST rooted in
        ``root``. Control flow, in terms of constructs like if-then-else, switch,
        etc, is forbidden."""

        # The starting point of this pass consists of identifying the location
        # of non-zero valued blocks in the symbols appearing in /root/. For this,
        # the declarations of such symbols are examined.
        nz_in_syms = {s: d.nonzero for s, d in self.decls.items() if d.nonzero}

        # Track propagation of zero blocks by symbolically executing the code
        nz_info = OrderedDict()
        self._track_nz_blocks(root, nz_in_syms, nz_info)

        return (nz_in_syms, nz_info)

    def _reschedule_itspace(self, root, nz_in_syms, nz_info):
        """Consider two statements A and B, and their iteration spaces.
        If the two iteration spaces have

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

        Return the dictionary of the updated expressions.
        """

        track_exprs, new_nz_info = {}, {}
        for loop, stmt_itspaces in nz_info.items():
            fissioned_loops = defaultdict(list)
            # Fission the loops on an intermediate representation
            for stmt, stmt_itspace in stmt_itspaces:
                nz_bounds_list = [i for i in itertools.product(*stmt_itspace.values())]
                for nz_bounds in nz_bounds_list:
                    dim_nz_bounds = tuple(zip(stmt_itspace.keys(), nz_bounds))
                    if not dim_nz_bounds:
                        # If no non_zero bounds, then just reuse the existing loops
                        dim_nz_bounds = itspace_from_for(loop, mode=1)
                    itspace, stmt_ofs = itspace_size_ofs(dim_nz_bounds)
                    copy_stmt = dcopy(stmt)
                    fissioned_loops[itspace].append((copy_stmt, stmt_ofs))
                    if stmt in self.exprs:
                        track_exprs[copy_stmt] = self.exprs[stmt]
            # Generate the actual code.
            # The dictionary is sorted because we must first execute smaller
            # loop nests, since larger ones may depend on them
            for itspace, stmt_ofs in sorted(fissioned_loops.items()):
                loops_info, inner_block = itspace_to_for(itspace, root)
                for stmt, ofs in stmt_ofs:
                    dict_ofs = dict(ofs)
                    ast_update_ofs(stmt, dict_ofs)
                    self._init_decl_to_zero(stmt, nz_in_syms, dict_ofs, itspace)
                    inner_block.children.append(stmt)
                    # Update expressions and hoisting-related information
                    if stmt in track_exprs:
                        self.exprs[stmt] = copy_metaexpr(track_exprs[stmt],
                                                         parent=inner_block,
                                                         loops_info=loops_info)
                    self.hoisted.update_stmt(stmt.children[0].symbol,
                                             loop=loops_info[0][0], place=root)
                new_nz_info[loops_info[-1][0]] = stmt_ofs
                # Append the created loops to the root
                index = root.children.index(loop[0])
                root.children.insert(index, loops_info[0][0])
            root.children.remove(loop[0])

        nz_info.clear()
        nz_info.update(new_nz_info)

    def reschedule(self):
        """Restructure the loop nests embedding ``self.exprs`` based on the
        propagation of zero-valued columns along the computation. This, therefore,
        involves fissing and fusing loops so as to remove iterations spent
        performing arithmetic operations over zero-valued entries."""

        roots = set()
        elf = ExpressionFissioner(1)
        for stmt, expr_info in self.exprs.items():
            if expr_info.is_scalar:
                continue
            elif expr_info.dimension > 1:
                # Split expressions based on sum's associativity. This exposes more
                # opportunities for rescheduling loops, since different summands
                # may have zero-valued blocks at different offsets
                self.exprs.pop(stmt)
                self.exprs.update(elf.fission(stmt, expr_info, False))
            roots.add(expr_info.domain_loops_parents[0])

        if len(roots) > 1:
            warning("Found multiple roots while performing zero-elimination")
            warning("The code generation is undefined")
        root = roots.pop()

        # Symbolically execute the code starting from root to track the
        # propagation of zero-valued blocks in the various arrays
        nz_in_syms, nz_info = self._track_nz(root)

        # At this point, we know the final location of non-zero values, so we can
        # restructure the iteration spaces to avoid useless computation
        self._reschedule_itspace(root, nz_in_syms, nz_info)

        return nz_info
