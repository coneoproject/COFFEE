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

"""COFFEE's SIMD vectorizer"""

from math import ceil
from copy import deepcopy as dcopy
from collections import defaultdict

from base import *
from utils import *
import plan as ap


class LoopVectorizer(object):

    """ Expression vectorizer class."""

    def __init__(self, loop_opt, intrinsics, compiler):
        self.loop_opt = loop_opt
        self.intr = intrinsics
        self.comp = compiler

    def alignment(self, decl_scope):
        """Align all data structures accessed in the loop nest to the size in
        bytes of the vector length."""
        for decl, scope in decl_scope.values():
            if decl.sym.rank and scope != ap.PARAM_VAR:
                decl.attr.append(self.comp["align"](self.intr["alignment"]))

    def padding(self, decl_scope):
        """Pad all data structures accessed in the loop nest to the nearest
        multiple of the vector length. Adjust trip counts and bounds of all
        innermost loops where padded arrays are written to. Since padding
        enforces data alignment of multi-dimensional arrays, add suitable
        pragmas to inner loops to inform the backend compiler about this
        property."""
        info = visit(self.loop_opt.header, None, search=LinAlg)
        symbols_dep = info['symbols_dep']
        symbols_mode = info['symbols_mode']
        symbol_refs = info['symbol_refs']
        used_syms = [s.symbol for s in symbols_mode.keys()]
        decl_scope = {s: ds for s, ds in decl_scope.items() if s in used_syms}

        # 0) Under some circumstances, do not apply padding
        # A- Loop increments must be equal to 1, because at the moment the
        #    machinery for ensuring the correctness of the transformation for
        #    non-uniform and non-unitary increments is missing
        for nest in info['fors']:
            if any(l.increment != 1 for l, _ in nest):
                return

        # Heuristically assuming that the fastest varying dimension is the
        # innermost (e.g., in A[x][y][z], z is the innermost dimension), padding
        # occurs along such dimension
        p_dim = -1
      
        # 1) Pad arrays by extending the innermost dimension. For example, if we
        # assume a vector length of 4 and the following input code:
        #
        # void foo(int A[10][10]):
        #   int B[10] = ...
        #   for i = 0 to 10:
        #     for j = 0 to 10:
        #       A[i][j] = B[i][j]
        #
        # Then after padding we get:
        #
        # void foo(int A[10][10]):
        #   int _A[10][12] = {{0.0}};
        #   int B[10][12] = ...
        #   for i = 0 to 10:
        #     for j = 0 to 12:
        #       _A[i][j] = B[i][j]
        #
        #   for i = 0 to 10:
        #     for j = 0 to 10:
        #       A[i][j] = _A[i][j]
        #
        # Also, extra care is taken in presence of offsets (e.g. A[i+3][j+3] ...)
        for decl, scope in decl_scope.values():
            if not decl.sym.rank:
                continue
            if scope == ap.LOCAL_VAR:
                p_rank = decl.sym.rank[:p_dim] + (vect_roundup(decl.sym.rank[p_dim]),)
                if p_rank == decl.sym.rank:
                    continue
                decl.sym.rank = p_rank
                continue
            # Examined symbol is a FunDecl argument
            # With padding, the computation runs on a /temporary/ padded array
            # ('_A' in the example above), in the following referred to as 'buffer'.
            #
            # A- analyze /how/ the examined FunDecl argument is accessed; that is,
            #    if it's read, written, or incremented; the iteration space; and the
            #    actual data space (i.e., which entries are eventually accessed)
            itspace, p_dataspace = [], []
            s_refs, b_refs = [], []
            for s_ref, _ in symbol_refs[decl.sym.symbol]:
                if s_ref is decl.sym or not s_ref.rank:
                    continue
                s_refs.append(dcopy(s_ref))
                # Track the iteration space of /s_ref/
                s_itspace = itspace_from_for([l for l in symbols_dep[s_ref]
                                              if l.itvar in s_ref.rank])
                s_itspace = [(start, end) for start, end, _ in s_itspace]
                itspace.append(s_itspace)
                # Track the data space of /s_ref/ and analyze offsets to ensure
                # the safeness of padding. Offsets can be adjusted to exploit
                # the padded region
                s_p_dataspace = s_itspace[p_dim]
                if s_ref.offset:
                    ofs = s_ref.offset[p_dim][-1]
                    if not type(ofs) == int:
                        raise RuntimeError("Illegal padding with non-static offsets")
                    if ofs > 0:
                        s_p_dataspace = tuple(i+ofs for i in s_p_dataspace)
                        new_ofs = (s_ref.offset[p_dim][0], vect_roundup(ofs))
                        s_ref.offset = s_ref.offset[:p_dim] + (new_ofs,)
                p_dataspace.append(s_p_dataspace)
                b_refs.append(s_ref)
            itspace = [itspace_merge(r)[0] for r in zip(*itspace)]
            if len(itspace_merge(p_dataspace)) > 1:
                raise RuntimeError("Illegal padding with non-contiguous data space")
            p_dataspace = [list(s_p_dataspace) for s_p_dataspace in set(p_dataspace)]
            p_dataspace = sorted(p_dataspace)
            # B- Determine the rank of the temporary buffer
            shift = vect_roundup(p_dataspace[0][0]) - p_dataspace[0][0]
            for s_p_dataspace in p_dataspace:
                s_p_dataspace[0] += shift
                s_p_dataspace[1] += shift
                p_ofs = vect_roundup(s_p_dataspace[1])
                shift += p_ofs - s_p_dataspace[1]
                s_p_dataspace[1] = p_ofs
            p_rank = decl.sym.rank[:p_dim] + (p_dataspace[-1][p_dim],)
            if p_rank == decl.sym.rank:
                continue
            # C- Create the temporary buffer and insert it at the top of the AST
            buf_decl = dcopy(decl)
            buf_sym = buf_decl.sym
            buf_sym.symbol = "_%s" % buf_sym.symbol
            buf_sym.rank = p_rank
            buf_decl.init = ArrayInit('%s0.0%s' % ('{'*len(p_rank), '}'*len(p_rank)))
            self.loop_opt.header.children.insert(0, buf_decl)
            # D- Replace any symbol occurrences with the temporary buffer
            s_access_modes = []
            for s_ref, _ in symbol_refs[decl.sym.symbol]:
                if s_ref is decl.sym:
                    continue
                s_ref.symbol = buf_sym.symbol
                s_access_modes.append(symbols_mode[s_ref])
            # E- Create and append a loop nest(s) for copying data into/from
            # the temporary buffer. Depending on how the symbol is accessed
            # (read only, read and write, incremented, etc.), different sort
            # of copies are made
            first, last = s_access_modes[0], s_access_modes[-1]
            if first[0] == READ:
                copy, init = ast_c_make_copy(b_refs, s_refs, itspace, Assign)
                self.loop_opt.header.children.insert(0, copy.children[0])
            if last[0] == WRITE:
                # If extra information (i.e., a pragma) is present telling that
                # the argument does not need to be incremented because it does
                # not contain any meaningful values, then we can safely write
                # to it. This is an optimization to avoid increments when not
                # necessarily required
                d_access_mode = [p for p in decl.pragma if isinstance(p, Access)]
                op = Assign if d_access_mode and d_access_mode[0] == WRITE else last[1]
                copy, init = ast_c_make_copy(s_refs, b_refs, itspace, op)
                self.loop_opt.header.children.append(copy.children[0])
            # Update global data structures
            decl_scope[buf_sym.symbol] = (buf_decl, ap.LOCAL_VAR)

        # 2) Padding, handle special nodes
        linalg_nodes = info['search'][LinAlg]
        for n in linalg_nodes:
            if isinstance(n, Invert):
                _, _, lda = n.children
                lda.symbol = vect_roundup(lda.symbol)

        # 3) Bounds adjustment
        # Bounds adjustment consists of modifying the start point and the
        # end point of an innermost loop (i.e. its bounds) and the offsets
        # of all of its statements such that the memory accesses are aligned
        # to the vector length.
        # Bounds adjustment of a loop is safe iff:
        # A- all statements's lhs in the loop body have as fastest varying
        #    dimension the iteration variable of the innermost loop
        # B- the extra iterations fall either in a padded region or in a
        #    zero-valued region; this must be checked for every statement
        #    in the loop.
        iloops = inner_loops(self.loop_opt.header)
        adjusted_loops = []
        for l in iloops:
            adjust = True
            # Condition A
            for stmt in l.children[0].children:
                sym = stmt.children[0]
                if not (sym.rank and sym.rank[-1] == l.itvar):
                    adjust = False
                    break
            # Condition B
            alignable_stmts = []
            nz_in_l = self.loop_opt.nz_in_fors.get(l, [])
            # Note that if nz_in_l is None, the full iteration space is traversed,
            # from the beginning to the end, so no offsets are used and it's ok
            # to adjust the top bound of the loop over the region that is going
            # to be padded, at least for this statememt
            read_regions = defaultdict(list)
            for stmt, ofs in nz_in_l:
                expr = dcopy(stmt.children[1])
                ast_update_ofs(expr, dict([(l.itvar, 0)]))
                l_ofs = dict(ofs)[l.itvar]
                # The statement can be aligned only if the new start and end
                # points cover the whole iteration space. Also, the padded
                # region cannot be exceeded.
                start_point = vect_rounddown(l_ofs)
                end_point = start_point + vect_roundup(l.end)  # == tot iters
                if end_point >= l_ofs + l.end:
                    alignable_stmts.append((stmt, dict([(l.itvar, start_point)])))
                read_regions[str(expr)].append((start_point, end_point))
            for rr in read_regions.values():
                if len(itspace_merge(rr)) < len(rr):
                    # Bound adjustment cause overlapping, so give up
                    adjust = False
                    break
            # Conditions checked, if both passed then adjust loop and offsets
            if adjust:
                # Adjust end point
                l.cond.children[1] = c_sym(vect_roundup(l.end))
                # Adjust start points
                for stmt, ofs in alignable_stmts:
                    ast_update_ofs(stmt, ofs)
                # If all statements were successfully aligned, then put a
                # suitable pragma to tell the compiler
                if len(alignable_stmts) == len(nz_in_l):
                    adjusted_loops.append(l)
                # Successful bound adjustment allows forcing simdization
                if self.comp.get('force_simdization'):
                    l.pragma.add(self.comp['force_simdization'])

        # 4) Adding pragma alignment is safe iff
        # A- the start point of the loop is a multiple of the vector length
        # B- the size of the loop is a multiple of the vector length (note that
        #    at this point, we have already checked the loop increment is 1)
        for l in adjusted_loops:
            if not (l.start % self.intr["dp_reg"] and l.size % self.intr["dp_reg"]):
                l.pragma.add(self.comp["decl_aligned_for"])

    def specialize(self, opts, factor=1):
        """Generate code for specialized expression vectorization. Check for peculiar
        memory access patterns in an expression and replace scalar code with highly
        optimized vector code. Currently, the following patterns are supported:

        * Outer products - e.g. A[i]*B[j]

        Also, code generation is supported for the following instruction sets:

        * AVX

        The parameter ``opts`` can be used to drive the transformation process:

        * ``opts = V_OP_PADONLY`` : no peeling, just use padding
        * ``opts = V_OP_PEEL`` : peeling for autovectorisation
        * ``opts = V_OP_UAJ`` : set unroll_and_jam as specified by ``factor``
        * ``opts = V_OP_UAJ_EXTRA`` : as above, but extra iters avoid remainder
          loop factor is an additional parameter to specify things like
          unroll-and-jam factor. Note that factor is just a suggestion to the
          compiler, which can freely decide to use a higher or lower value.
        """
        layout = None
        for stmt, expr_info in self.loop_opt.exprs.items():
            parent = expr_info.parent
            unit_stride_loops, unit_stride_loops_parents = \
                zip(*expr_info.unit_stride_loops_info)

            # Check if outer-product vectorization is actually doable
            vect_len = self.intr["dp_reg"]
            rows = unit_stride_loops[0].size
            if rows < vect_len:
                continue
            if len(unit_stride_loops) != 2:
                # There must be exactly two unit-strided dimensions
                continue

            op = OuterProduct(stmt, unit_stride_loops, self.intr, self.loop_opt)

            # Vectorisation
            unroll_factor = factor if opts in [ap.V_OP_UAJ, ap.V_OP_UAJ_EXTRA] else 1
            rows_per_it = vect_len*unroll_factor
            if opts == ap.V_OP_UAJ:
                if rows_per_it <= rows:
                    body, layout = op.generate(rows_per_it)
                else:
                    # Unroll factor too big
                    body, layout = op.generate(vect_len)
            elif opts == ap.V_OP_UAJ_EXTRA:
                if rows <= rows_per_it or vect_roundup(rows) % rows_per_it > 0:
                    # Cannot unroll too much
                    body, layout = op.generate(vect_len)
                else:
                    body, layout = op.generate(rows_per_it)
            elif opts in [ap.V_OP_PADONLY, ap.V_OP_PEEL]:
                body, layout = op.generate(vect_len)
            else:
                raise RuntimeError("Don't know how to vectorize option %s" % opts)

            # Construct the remainder loop
            if opts != ap.V_OP_UAJ_EXTRA and rows > rows_per_it and rows % rows_per_it > 0:
                # peel out
                loop_peel = dcopy(unit_stride_loops)
                # Adjust main, layout and remainder loops bound and trip
                bound = unit_stride_loops[0].cond.children[1].symbol
                bound -= bound % rows_per_it
                unit_stride_loops[0].cond.children[1] = c_sym(bound)
                layout.cond.children[1] = c_sym(bound)
                loop_peel[0].init.init = c_sym(bound)
                loop_peel[0].incr.children[1] = c_sym(1)
                loop_peel[1].incr.children[1] = c_sym(1)
                # Append peeling loop after the main loop
                unit_stride_outerparent = unit_stride_loops_parents[0]
                ofs = unit_stride_outerparent.children.index(unit_stride_loops[0])
                unit_stride_outerparent.children.insert(ofs+1, loop_peel[0])

            # Insert the vectorized code at the right point in the loop nest
            blk = parent.children
            ofs = blk.index(stmt)
            parent.children = blk[:ofs] + body + blk[ofs + 1:]

        # Append the layout code after the whole loop nest
        if layout:
            parent = self.loop_opt.header.children.append(layout)


class OuterProduct():

    """Generate outer product vectorisation of a statement. """

    OP_STORE_IN_MEM = 0
    OP_REGISTER_INC = 1

    def __init__(self, stmt, loops, intr, nest):
        self.stmt = stmt
        self.intr = intr
        # Outer product loops
        self.loops = loops
        # The whole loop nest in which outer product loops live
        self.nest = nest

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, intr, tensor_size):
            nres = max(intr["dp_reg"], tensor_size)
            self.ntot = intr["avail_reg"]
            self.res = [intr["reg"](v) for v in range(nres)]
            self.var = [intr["reg"](v) for v in range(nres, self.ntot)]
            self.i = intr

        def get_reg(self):
            if len(self.var) == 0:
                l = self.ntot * 2
                self.var += [self.i["reg"](v) for v in range(self.ntot, l)]
                self.ntot = l
            return self.var.pop(0)

        def free_regs(self, regs):
            for r in reversed(regs):
                self.var.insert(0, r)

        def get_tensor(self):
            return self.res

    def _swap_reg(self, step, vrs):
        """Swap values in a vector register. """

        # Find inner variables
        regs = [reg for node, reg in vrs.items()
                if node.rank and node.rank[-1] == self.loops[1].itvar]

        if step in [0, 2]:
            return [Assign(r, self.intr["l_perm"](r, "5")) for r in regs]
        elif step == 1:
            return [Assign(r, self.intr["g_perm"](r, r, "1")) for r in regs]
        elif step == 3:
            return []

    def _vect_mem(self, vrs, decls):
        """Return a list of vector variable declarations representing
        loads, sets, broadcasts.

        :arg vrs:   Dictionary that associates scalar variables to vector.
                    variables, for which it will be generated a corresponding
                    intrinsics load/set/broadcast.
        :arg decls: List of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated. Used to avoid
                    regenerating the same line. Can be updated.
        """
        stmt = []
        for node, reg in vrs.items():
            if node.rank and node.rank[-1] in [i.itvar for i in self.loops]:
                exp = self.intr["symbol_load"](node.symbol, node.rank, node.offset)
            else:
                exp = self.intr["symbol_set"](node.symbol, node.rank, node.offset)
            if not decls.get(node.gencode()):
                decls[node.gencode()] = reg
                stmt.append(Decl(self.intr["decl_var"], reg, exp))
        return stmt

    def _vect_expr(self, node, ofs, regs, decls, vrs):
        """Turn a scalar expression into its intrinsics equivalent.
        Also return dicts of allocated vector variables.

        :arg node:  AST Expression which is inspected to generate an equivalent
                    intrinsics-based representation.
        :arg ofs:   Contains the offset of the entry in the left hand side that
                    is being computed.
        :arg regs:  Register allocator.
        :arg decls: List of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated. Used to determine
                    which vector variable contains a certain scalar, if any.
        :arg vrs:   Dictionary that associates scalar variables to vector
                    variables. Updated every time a new scalar variable is
                    encountered.
        """
        if isinstance(node, Symbol):
            if node.rank and self.loops[0].itvar == node.rank[-1]:
                # The symbol depends on the outer loop dimension, so add offset
                n_ofs = tuple([(1, 0) for i in range(len(node.rank)-1)]) + ((1, ofs),)
                node = Symbol(node.symbol, dcopy(node.rank), n_ofs)
            node_ide = node.gencode()
            if node_ide not in decls:
                reg = [k for k in vrs.keys() if k.gencode() == node_ide]
                if not reg:
                    vrs[node] = c_sym(regs.get_reg())
                    return vrs[node]
                else:
                    return vrs[reg[0]]
            else:
                return decls[node_ide]
        elif isinstance(node, Par):
            return self._vect_expr(node.children[0], ofs, regs, decls, vrs)
        else:
            left = self._vect_expr(node.children[0], ofs, regs, decls, vrs)
            right = self._vect_expr(node.children[1], ofs, regs, decls, vrs)
            if isinstance(node, Sum):
                return self.intr["add"](left, right)
            elif isinstance(node, Sub):
                return self.intr["sub"](left, right)
            elif isinstance(node, Prod):
                return self.intr["mul"](left, right)
            elif isinstance(node, Div):
                return self.intr["div"](left, right)

    def _incr_tensor(self, tensor, ofs, regs, out_reg, mode):
        """Add the right hand side contained in out_reg to tensor.

        :arg tensor:  The left hand side of the expression being vectorized.
        :arg ofs:     Contains the offset of the entry in the left hand side that
                      is being computed.
        :arg regs:    Register allocator.
        :arg out_reg: Register variable containing the left hand side.
        :arg mode:    It can be either `OP_STORE_IN_MEM`, for which stores in
                      memory are performed, or `OP_REGISTER_INC`, by means of
                      which left hand side's values are accumulated in a register.
                      Usually, `OP_REGISTER_INC` is not recommended unless the
                      loop sizes are extremely small.
        """
        if mode == self.OP_STORE_IN_MEM:
            # Store in memory
            sym = tensor.symbol
            rank = tensor.rank
            ofs = ((1, ofs), (1, 0))
            load = self.intr["symbol_load"](sym, rank, ofs)
            return self.intr["store"](Symbol(sym, rank, ofs),
                                      self.intr["add"](load, out_reg))
        elif mode == self.OP_REGISTER_INC:
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, self.intr["add"](reg, out_reg))

    def _restore_layout(self, regs, tensor, mode):
        """Restore the storage layout of the tensor.

        :arg regs:    Register allocator.
        :arg tensor:  The left hand side of the expression being vectorized.
        :arg mode:    It can be either `OP_STORE_IN_MEM`, for which load/stores in
                      memory are performed, or `OP_REGISTER_INC`, by means of
                      which left hand side's values are read from registers.
        """
        code = []
        t_regs = [Symbol(r, ()) for r in regs.get_tensor()]
        n_regs = len(t_regs)

        # Determine tensor symbols
        tensor_syms = []
        for i in range(n_regs):
            rank = (tensor.rank[0] + "+" + str(i), tensor.rank[1])
            tensor_syms.append(Symbol(tensor.symbol, rank))

        # Load LHS values from memory
        if mode == self.OP_STORE_IN_MEM:
            for i, j in zip(tensor_syms, t_regs):
                load_sym = self.intr["symbol_load"](i.symbol, i.rank)
                code.append(Decl(self.intr["decl_var"], j, load_sym))

        # In-register restoration of the tensor
        # TODO: AVX only at the present moment
        # TODO: here some __m256 vars could not be declared if rows < 4
        perm = self.intr["g_perm"]
        uphi = self.intr["unpck_hi"]
        uplo = self.intr["unpck_lo"]
        typ = self.intr["decl_var"]
        vect_len = self.intr["dp_reg"]
        # Do as many times as the unroll factor
        spins = int(ceil(n_regs / float(vect_len)))
        for i in range(spins):
            # In-register permutations
            tmp = [Symbol(regs.get_reg(), ()) for r in range(vect_len)]
            code.append(Decl(typ, tmp[0], uphi(t_regs[1], t_regs[0])))
            code.append(Decl(typ, tmp[1], uplo(t_regs[0], t_regs[1])))
            code.append(Decl(typ, tmp[2], uphi(t_regs[2], t_regs[3])))
            code.append(Decl(typ, tmp[3], uplo(t_regs[3], t_regs[2])))
            code.append(Assign(t_regs[0], perm(tmp[1], tmp[3], 32)))
            code.append(Assign(t_regs[1], perm(tmp[0], tmp[2], 32)))
            code.append(Assign(t_regs[2], perm(tmp[3], tmp[1], 49)))
            code.append(Assign(t_regs[3], perm(tmp[2], tmp[0], 49)))
            regs.free_regs([s.symbol for s in tmp])

            # Store LHS values in memory
            for j in range(min(vect_len, n_regs - i * vect_len)):
                ofs = i * vect_len + j
                code.append(self.intr["store"](tensor_syms[ofs], t_regs[ofs]))

        return code

    def generate(self, rows):
        """Generate the outer-product intrinsics-based vectorisation code. """
        cols = self.intr["dp_reg"]

        # Determine order of loops w.r.t. the local tensor entries.
        # If j-k are the inner loops and A[j][k], then increments of
        # A are performed within the k loop, otherwise we would lose too many
        # vector registers for keeping tmp values. On the other hand, if i is
        # the innermost loop (i.e. loop nest is j-k-i), stores in memory are
        # done outside of ip, i.e. immediately before the outer product's
        # inner loop terminates.
        if self.loops[1] in inner_loops(self.loops[0]):
            mode = self.OP_STORE_IN_MEM
            tensor_size = cols
        else:
            mode = self.OP_REGISTER_INC
            tensor_size = rows

        tensor = self.stmt.children[0]
        expr = self.stmt.children[1]

        # Get source-level variables
        regs = self.Alloc(self.intr, tensor_size)

        # Adjust loops' increment
        self.loops[0].incr.children[1] = c_sym(rows)
        self.loops[1].incr.children[1] = c_sym(cols)

        stmt = []
        decls = {}
        vrs = {}
        rows_per_col = rows / cols
        rows_to_peel = rows % cols
        peeling = 0
        for i in range(cols):
            # Handle extra rows
            if peeling < rows_to_peel:
                nrows = rows_per_col + 1
                peeling += 1
            else:
                nrows = rows_per_col
            for j in range(nrows):
                # Vectorize, declare allocated variables, increment tensor
                ofs = j * cols
                v_expr = self._vect_expr(expr, ofs, regs, decls, vrs)
                stmt.extend(self._vect_mem(vrs, decls))
                incr = self._incr_tensor(tensor, i + ofs, regs, v_expr, mode)
                stmt.append(incr)
            # Register shuffles
            if rows_per_col + (rows_to_peel - peeling) > 0:
                stmt.extend(self._swap_reg(i, vrs))

        # Set initialising and tensor layout code
        layout = self._restore_layout(regs, tensor, mode)
        if mode == self.OP_STORE_IN_MEM:
            # Tensor layout
            layout_loops = dcopy(self.loops)
            layout_loops[0].incr.children[1] = c_sym(cols)
            layout_loops[0].children = [Block([layout_loops[1]], open_scope=True)]
            layout_loops[1].children = [Block(layout, open_scope=True)]
            layout = layout_loops[0]
        elif mode == self.OP_REGISTER_INC:
            # Initialiser
            for r in regs.get_tensor():
                decl = Decl(self.intr["decl_var"], Symbol(r, ()), self.intr["setzero"])
                self.loops[1].children[0].children.insert(0, decl)
            # Tensor layout
            self.loops[1].children[0].children.extend(layout)
            layout = None

        return (stmt, layout)


# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = ap.intrinsics.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len


def vect_rounddown(x):
    """Return x rounded down to the vector length. """
    word_len = ap.intrinsics.get("dp_reg") or 1
    return x - (x % word_len)
