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

"""COFFEE's loop scheduler."""

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
from expression import MetaExpr, copy_metaexpr
from utils import ast_update_ofs, itspace_size_ofs, itspace_merge, itspace_to_for
from utils import itspace_from_for, visit, flatten


class LoopScheduler(object):

    """Base class for classes that handle loop scheduling; that is, loop fusion,
    loop distribution, etc."""


class SSALoopMerger(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then merge fusable
    loops.
    Statements must be in "soft" SSA form: they can be declared and initialized
    at declaration time, then they can be assigned a value in only one place."""

    def __init__(self, root, expr_graph):
        """Initialize the SSALoopMerger.

        :arg expr_graph: the ExpressionGraph tracking all data dependencies
                         involving identifiers that appear in ``root``.
        :arg root: the node where loop scheduling takes place."""
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
        itvars_a, itvars_b = [], []
        while isinstance(loop_b.children[0], (Block, For)):
            if isinstance(loop_b, For):
                itvars_b.append(loop_b.itvar)
            loop_b = loop_b.children[0]
        # Find the first statement in the perfect loop nest loop_a
        root_loop_a = loop_a
        while isinstance(loop_a.children[0], (Block, For)):
            if isinstance(loop_a, For):
                itvars_a.append(loop_a.itvar)
            loop_a = loop_a.children[0]
        # Merge body of loop_a in loop_b
        loop_b.children[0:0] = loop_a.children
        # Remove loop_a from root
        root.children.remove(root_loop_a)
        return (loop_b, tuple(itvars_a), tuple(itvars_b))

    def _update_itvars(self, node, itvars):
        """Change the iteration variables in the nodes rooted in ``node``
        according to the map defined in ``itvars``, which is a dictionary
        from old_iteration_variable to new_iteration_variable. For example,
        given itvars = {'i': 'j'} and a node "A[i] = B[i]", change the node
        into "A[j] = B[j]"."""
        if isinstance(node, Symbol):
            new_rank = []
            for r in node.rank:
                new_rank.append(r if r not in itvars else itvars[r])
            node.rank = tuple(new_rank)
        elif not isinstance(node, FlatBlock):
            for n in node.children:
                self._update_itvars(n, itvars)

    def merge(self):
        """Merge perfect loop nests rooted in ``self.root``."""
        found_nests = defaultdict(list)
        # Collect some info visiting the tree rooted in node
        for n in self.root.children:
            if isinstance(n, For):
                # Track structure of iteration spaces
                loops_infos = visit(n, self.root)['fors']
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
                    if self.expr_graph.has_dep(ws, lws):
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
                merged, l_itvars, m_itvars = self._merge_loops(parent, l, merging_in)
                self._update_itvars(merged, dict(zip(l_itvars, m_itvars)))
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

        Note this last step is not done by compilers like intel's (version 14).
        """

        def replace_expr(node, parent, parent_idx, itvar, hoisted_expr):
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
                        replace_expr(n, node, i, itvar, hoisted_expr)

        hoisted_expr = {}
        for loop in self.merged_loops:
            block = loop.children[0].children
            for stmt in block:
                sym, expr = stmt.children
                replace_expr(expr.children[0], expr, 0, loop.itvar, hoisted_expr)
                hoisted_expr[str(expr)] = sym


class ExpressionFissioner(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then fission associative
    operations in expressions.
    Fissioned expressions are placed in a separate loop nest."""

    def __init__(self, cut):
        """Initialize the ExpressionFissioner.

        :arg cut: number of operands requested to fission expressions."""
        self.cut = cut

    def _split_sum(self, node, parent, is_left, found, sum_count):
        """Exploit sum's associativity to cut node when a sum is found.
        Return ``True`` if a potentially splittable node is found, ``False``
        otherwise."""
        if isinstance(node, Symbol):
            return False
        elif isinstance(node, Par):
            return self._split_sum(node.children[0], (node, 0), is_left, found,
                                   sum_count)
        elif isinstance(node, Prod) and found:
            return False
        elif isinstance(node, Prod) and not found:
            if not self._split_sum(node.children[0], (node, 0), is_left, found,
                                   sum_count):
                return self._split_sum(node.children[1], (node, 1), is_left, found,
                                       sum_count)
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
                    parent.children[parent_leaf] = node.children[0]
                else:
                    found, found_leaf = found
                    found.children[found_leaf] = node.children[1]
                return True
            else:
                if not self._split_sum(node.children[0], (node, 0), is_left,
                                       found, sum_count):
                    return self._split_sum(node.children[1], (node, 1), is_left,
                                           found, sum_count)
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
        unit_stride_outerloop_info = expr_info.unit_stride_loops_info[0]
        unit_stride_outerloop, unit_stride_outerparent = unit_stride_outerloop_info

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
                                         expr_info.unit_stride_itvars))

            # Append the right-split (remainder) expression
            if copy_loops:
                # Create a new loop nest
                new_unit_stride_outerloop = dcopy(unit_stride_outerloop)
                new_unit_stride_innerloop = new_unit_stride_outerloop.children[0].children[0]
                new_unit_stride_innerloop_block = new_unit_stride_innerloop.children[0]
                new_unit_stride_innerloop_block.children[0] = stmt_right
                new_unit_stride_outerloop_info = (new_unit_stride_outerloop,
                                                  unit_stride_outerparent)
                new_unit_stride_innerloop_info = (new_unit_stride_innerloop,
                                                  new_unit_stride_innerloop_block)
                new_loops_info = expr_info.slow_loops_info + \
                    (new_unit_stride_outerloop_info,) + (new_unit_stride_innerloop_info,)
                unit_stride_outerparent.children.append(new_unit_stride_outerloop)
            else:
                # Reuse loop nest created in the previous function call
                expr_parent.children.insert(index, stmt_right)
                new_unit_stride_innerloop_block = expr_parent
                new_loops_info = expr_info.loops_info
            splittable = (stmt_right, MetaExpr(expr_info.type,
                                               new_unit_stride_innerloop_block,
                                               new_loops_info,
                                               expr_info.unit_stride_itvars))
            return (split, splittable)
        return ((stmt, expr_info), ())

    def fission(self, stmt_info, copy_loops):
        """Split an expression containing ``x`` summands into ``x/cut`` chunks.
        Each chunk is placed in a separate loop nest if ``copy_loops`` is true,
        in the same loop nest otherwise. In the former case, the split occurs
        in the largest perfect loop nest wrapping the expression in ``stmt_info``.
        Return a dictionary of all of the split chunks, in which each entry has
        the same format of ``stmt_info``.

        :arg stmt_info:  the expression that needs to be split. This is given as
                         a tuple of two elements: the former is the expression
                         root node; the latter includes info about the expression,
                         particularly iteration variables of the enclosing loops,
                         the enclosing loops themselves, and the parent block.
        :arg copy_loops: true if the split expressions should be placed in two
                         separate, adjacent loop nests (iterating, of course,
                         along the same iteration space); false, otherwise."""

        split_stmts = {}
        splittable_stmt = stmt_info
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

    def __init__(self, exprs, expr_graph, decls):
        """Initialize the ZeroLoopScheduler.

        :arg exprs: the expressions for which the zero-elimination is performed.
        :arg expr_graph: the ExpressionGraph tracking all data dependencies involving
                         identifiers that appear in ``root``.
        :arg decls: lists of array declarations. A 2-tuple is expected: the first
                    element is the list of kernel declarations; the second element
                    is the list of hoisted temporaries declarations.
        """
        self.exprs = exprs
        self.expr_graph = expr_graph
        self.kernel_decls, self.hoisted = decls
        # Track zero blocks in each symbol accessed in the computation rooted in root
        self.nz_in_syms = {}
        # Track blocks accessed for evaluating symbols in the various for loops
        # rooted in root
        self.nz_in_fors = OrderedDict()

    def _get_nz_bounds(self, node):
        if isinstance(node, Symbol):
            return (node.rank[-1], self.nz_in_syms[node.symbol])
        elif isinstance(node, Par):
            return self._get_nz_bounds(node.children[0])
        elif isinstance(node, Prod):
            return tuple([self._get_nz_bounds(n) for n in node.children])
        else:
            raise RuntimeError("Group iter space error: unknown node: %s" % str(node))

    def _merge_itvars_nz_bounds(self, itvar_nz_bounds_l, itvar_nz_bounds_r):
        """Given two dictionaries associating iteration variables to ranges
        of non-zero columns, merge the two dictionaries by combining ranges
        along the same iteration variables and return the merged dictionary.
        For example: ::

            dict1 = {'j': [(1,3), (5,6)], 'k': [(5,7)]}
            dict2 = {'j': [(3,4)], 'k': [(1,4)]}
            dict1 + dict2 -> {'j': [(1,6)], 'k': [(1,7)]}
        """
        new_itvar_nz_bounds = {}
        for itvar, nz_bounds in itvar_nz_bounds_l.items():
            if itvar.isdigit():
                # Skip constant dimensions
                continue
            # Compute the union of nonzero bounds along the same
            # iteration variable. Unify contiguous regions (for example,
            # [(1,3), (4,6)] -> [(1,6)]
            new_nz_bounds = nz_bounds + itvar_nz_bounds_r.get(itvar, ())
            merged_nz_bounds = itspace_merge(new_nz_bounds)
            new_itvar_nz_bounds[itvar] = merged_nz_bounds
        return new_itvar_nz_bounds

    def _set_var_to_zero(self, node, ofs, itspace):
        """Scan each variable ``v`` in ``node``: if non-initialized elements in ``v``
        are touched as iterating along ``itspace``, initialize ``v`` to 0.0."""

        def get_accessed_syms(node, nz_in_syms, found_syms):
            if isinstance(node, Symbol):
                nz_in_node = nz_in_syms.get(node.symbol)
                if nz_in_node:
                    nz_regions = dict(zip([r for r in node.rank], nz_in_node))
                    found_syms.append((node.symbol, nz_regions))
            else:
                for n in node.children:
                    get_accessed_syms(n, nz_in_syms, found_syms)

        # Determine the symbols accessed in node and their non-zero regions
        found_syms = []
        get_accessed_syms(node.children[1], self.nz_in_syms, found_syms)

        # If iteration space along which they are accessed is bigger than the
        # non-zero region, hoisted symbols must be initialized to zero
        for sym, nz_regions in found_syms:
            sym_decl = self.hoisted.get(sym)
            if not sym_decl:
                continue
            for itvar, size in itspace:
                itvar_nz_regions = nz_regions.get(itvar)
                itvar_ofs = ofs.get(itvar)
                if not itvar_nz_regions or itvar_ofs is None:
                    # Sym does not iterate along this iteration variable, so skip
                    # the check
                    continue
                iteration_ok = False
                # Check that the iteration space actually corresponds to one of the
                # non-zero regions in the symbol currently analyzed
                for itvar_nz_region in itvar_nz_regions:
                    init_nz_reg, end_nz_reg = itvar_nz_region
                    if itvar_ofs == init_nz_reg and size == end_nz_reg + 1 - init_nz_reg:
                        iteration_ok = True
                        break
                if not iteration_ok:
                    # Iterating over a non-initialized region, need to zero it
                    sym_decl.decl.init = FlatBlock("{0.0}")

    def _track_expr_nz_columns(self, node):
        """Return the first and last indices assumed by the iteration variables
        appearing in ``node`` over regions of non-zero columns. For example,
        consider the following node, particularly its right-hand side: ::

        A[i][j] = B[i]*C[j]

        If B over i is non-zero in the ranges [0, k1] and [k2, k3], while C over
        j is non-zero in the range [N-k4, N], then return a dictionary: ::

        {i: ((0, k1), (k2, k3)), j: ((N-k4, N),)}

        If there are no zero-columns, return {}."""
        if isinstance(node, Symbol):
            if node.offset:
                raise RuntimeError("Zeros error: offsets not supported: %s" % str(node))
            nz_bounds = self.nz_in_syms.get(node.symbol)
            if nz_bounds:
                itvars = [r for r in node.rank]
                return dict(zip(itvars, nz_bounds))
            else:
                return {}
        elif isinstance(node, Par):
            return self._track_expr_nz_columns(node.children[0])
        else:
            itvar_nz_bounds_l = self._track_expr_nz_columns(node.children[0])
            itvar_nz_bounds_r = self._track_expr_nz_columns(node.children[1])
            if isinstance(node, (Prod, Div)):
                # Merge the nonzero bounds of different iteration variables
                # within the same dictionary
                return dict(itvar_nz_bounds_l.items() +
                            itvar_nz_bounds_r.items())
            elif isinstance(node, Sum):
                return self._merge_itvars_nz_bounds(itvar_nz_bounds_l,
                                                    itvar_nz_bounds_r)
            else:
                raise RuntimeError("Zeros error: unsupported operation: %s" % str(node))

    def _track_nz_blocks(self, node, parent=None, loop_nest=()):
        """Track the propagation of zero blocks along the computation which is
        rooted in ``self.root``.

        Before start tracking zero blocks in the nodes rooted in ``node``,
        ``self.nz_in_syms`` contains, for each known identifier, the ranges of
        its zero blocks. For example, assuming identifier A is an array and has
        zero-valued entries in positions from 0 to k and from N-k to N,
        ``self.nz_in_syms`` will contain an entry "A": ((0, k), (N-k, N)).
        If A is modified by some statements rooted in ``node``, then
        ``self.nz_in_syms["A"]`` will be modified accordingly.

        This method also updates ``self.nz_in_fors``, which maps loop nests to
        the enclosed symbols' non-zero blocks. For example, given the following
        code: ::

        { // root
          ...
          for i
            for j
              A = ...
              B = ...
        }

        Once traversed the AST, ``self.nz_in_fors`` will contain a (key, value)
        such that:
        ((<for i>, <for j>), root) -> {A: (i, (nz_along_i)), (j, (nz_along_j))}

        :arg node:      the node being currently inspected for tracking zero
                        blocks
        :arg parent:    the parent node of ``node``
        :arg loop_nest: tuple of for loops enclosing ``node``
        """
        if isinstance(node, (Assign, Incr, Decr)):
            symbol = node.children[0].symbol
            rank = node.children[0].rank
            itvar_nz_bounds = self._track_expr_nz_columns(node.children[1])
            if not itvar_nz_bounds:
                return
            # Reflect the propagation of non-zero blocks in the node's
            # target symbol. Note that by scanning loop_nest, the nonzero
            # bounds are stored in order. For example, if the symbol is
            # A[][], that is, it has two dimensions, then the first element
            # of the tuple stored in nz_in_syms[symbol] represents the nonzero
            # bounds for the first dimension, the second element the same for
            # the second dimension, and so on if it had had more dimensions.
            # Also, since nz_in_syms represents the propagation of non-zero
            # columns "up to this point of the computation", we have to merge
            # the non-zero columns produced by this node with those that we
            # had already found.
            nz_in_sym = tuple(itvar_nz_bounds[l.itvar] for l in loop_nest
                              if l.itvar in rank)
            if symbol in self.nz_in_syms:
                merged_nz_in_sym = []
                for i in zip(nz_in_sym, self.nz_in_syms[symbol]):
                    flat_nz_bounds = [nzb for nzb_sym in i for nzb in nzb_sym]
                    merged_nz_in_sym.append(itspace_merge(flat_nz_bounds))
                nz_in_sym = tuple(merged_nz_in_sym)
            self.nz_in_syms[symbol] = nz_in_sym
            if loop_nest:
                # Track the propagation of non-zero blocks in this specific
                # loop nest. Outer loops, i.e. loops that have non been
                # encountered as visiting from the root, are discarded.
                key = loop_nest
                itvar_nz_bounds = dict([(k, v) for k, v in itvar_nz_bounds.items()
                                        if k in [l.itvar for l in loop_nest]])
                if key not in self.nz_in_fors:
                    self.nz_in_fors[key] = []
                self.nz_in_fors[key].append((node, itvar_nz_bounds))
        if isinstance(node, For):
            self._track_nz_blocks(node.children[0], node, loop_nest + (node,))
        if isinstance(node, (Root, Block)):
            for n in node.children:
                self._track_nz_blocks(n, node, loop_nest)

    def _track_nz(self, root):
        """Track the propagation of zero columns along the computation which is
        rooted in ``root``."""

        # Initialize a dict mapping symbols to their zero columns with the info
        # already available in the kernel's declarations
        for i, j in self.kernel_decls.items():
            nz_col_bounds = j[0].get_nonzero_columns()
            if nz_col_bounds:
                # Note that nz_bounds are stored as second element of a 2-tuple,
                # because the declared array is two-dimensional, in which the
                # second dimension represents the columns
                self.nz_in_syms[i] = (((0, j[0].sym.rank[0] - 1),),
                                      (nz_col_bounds,))
            else:
                self.nz_in_syms[i] = tuple(((0, r-1),) for r in j[0].size)

        # If zeros were not found, then just give up
        if not self.nz_in_syms:
            return {}

        # Track propagation of zero blocks by symbolically executing the code
        self._track_nz_blocks(root)

    def _reschedule_itspace(self, root, exprs):
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

        new_exprs = {}
        new_nz_in_fors = {}
        track_exprs = {}
        for loop, stmt_itspaces in self.nz_in_fors.items():
            fissioned_loops = defaultdict(list)
            # Fission the loops on an intermediate representation
            for stmt, stmt_itspace in stmt_itspaces:
                nz_bounds_list = [i for i in itertools.product(*stmt_itspace.values())]
                for nz_bounds in nz_bounds_list:
                    itvar_nz_bounds = tuple(zip(stmt_itspace.keys(), nz_bounds))
                    if not itvar_nz_bounds:
                        # If no non_zero bounds, then just reuse the existing loops
                        itvar_nz_bounds = itspace_from_for(loop, mode=1)
                    itspace, stmt_ofs = itspace_size_ofs(itvar_nz_bounds)
                    copy_stmt = dcopy(stmt)
                    fissioned_loops[itspace].append((copy_stmt, stmt_ofs))
                    if stmt in exprs:
                        track_exprs[copy_stmt] = exprs[stmt]
            # Generate the actual code.
            # The dictionary is sorted because we must first execute smaller
            # loop nests, since larger ones may depend on them
            for itspace, stmt_ofs in sorted(fissioned_loops.items()):
                loops_info, inner_block = itspace_to_for(itspace, root)
                for stmt, ofs in stmt_ofs:
                    dict_ofs = dict(ofs)
                    ast_update_ofs(stmt, dict_ofs)
                    self._set_var_to_zero(stmt, dict_ofs, itspace)
                    inner_block.children.append(stmt)
                    # Update expressions and hoisting-related information
                    if stmt in track_exprs:
                        new_exprs[stmt] = copy_metaexpr(track_exprs[stmt],
                                                        **{'parent': inner_block,
                                                           'loops_info': loops_info})
                    self.hoisted.update_stmt(stmt.children[0].symbol,
                                             **{'loop': loops_info[0][0], 'place': root})
                new_nz_in_fors[loops_info[-1][0]] = stmt_ofs
                # Append the created loops to the root
                index = root.children.index(loop[0])
                root.children.insert(index, loops_info[0][0])
            root.children.remove(loop[0])

        self.nz_in_fors = new_nz_in_fors
        return new_exprs

    def reschedule(self):
        """Restructure the loop nests embedding ``self.exprs`` based on the
        propagation of zero-valued columns along the computation. This, therefore,
        involves fissing and fusing loops so as to remove iterations spent
        performing arithmetic operations over zero-valued entries."""

        if not self.exprs:
            return

        roots, new_exprs = set(), {}
        elf = ExpressionFissioner(1)
        for expr in self.exprs.items():
            # First, split expressions into separate loop nests, based on sum's
            # associativity. This exposes more opportunities for restructuring loops,
            # since different summands may have contiguous regions of zero-valued
            # columns in different positions
            new_exprs.update(elf.fission(expr, False))
            roots.add(expr[1].unit_stride_loops_parents[0])
            self.exprs.pop(expr[0])

            if len(roots) > 1:
                warning("Found multiple roots while performing zero-elimination")
                warning("The code generation is undefined")

        root = roots.pop()
        # Symbolically execute the code starting from root to track the
        # propagation of zeros
        self._track_nz(root)

        # Finally, restructure the iteration spaces
        self.exprs.update(self._reschedule_itspace(root, new_exprs))
