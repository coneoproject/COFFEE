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
        to_invert = info['search'][Invert][0] if info['search'][Invert] else None
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
            p_rank = decl.sym.rank[:p_dim] + (vect_roundup(decl.sym.rank[p_dim]),)
            if scope == ap.LOCAL_VAR:
                if p_rank == decl.sym.rank:
                    continue
                decl.sym.rank = p_rank
                continue
            # Examined symbol is a FunDecl argument
            # With padding, the computation runs on a /temporary/ padded array
            # ('_A' in the example above), in the following referred to as 'buffer'.
            #
            # A- Analyze a FunDecl argument: ...
            acc_modes = []
            dataspace_syms = defaultdict(list)
            buf_rank = [0] + list(p_rank)
            for s, _ in symbol_refs[decl.sym.symbol]:
                if s is decl.sym or not s.rank:
                    continue
                # ... the access mode (READ, WRITE, ...)
                acc_modes.append(symbols_mode[s])
                # ... the presence of offsets
                ofs = s.offset[-1][1] if s.offset else 0
                # ... the iteration space
                s_itspace = [l for l in symbols_dep[s] if l.itvar in s.rank]
                s_itspace = tuple((s, e) for s, e, _ in itspace_from_for(s_itspace))
                s_p_itspace = s_itspace[p_dim]
                # ... combining the last two, the actual dataspace
                if decl.sym.rank[p_dim] != buf_rank[p_dim] or isinstance(ofs, Symbol) \
                        or (vect_roundup(ofs) > ofs):
                    dataspace_syms[(s_itspace, ofs)].append(s)
            if not dataspace_syms:
                continue
            # B- At this point we are sure we need a temporary buffer for efficient
            #    vectorization, so we create it and insert it at the top of the AST
            buffer = Decl(decl.typ, Symbol('_%s' % decl.sym.symbol, buf_rank),
                          ArrayInit('%s0.0%s' % ('{'*len(buf_rank), '}'*len(buf_rank))))
            self.loop_opt.header.children.insert(0, buffer)
            # C- Replace all FunDecl argument occurrences in the body with the buffer
            itspace_binding = defaultdict(list)
            for (s_itspace, offset), syms in dataspace_syms.items():
                for s in syms:
                    itspace_binding[s_itspace].append((dcopy(s), s))
                    s.symbol = buffer.sym.symbol
                    s.rank = (buf_rank[0],) + tuple(s.rank)
                    s.offset = ((1, 0),) + s.offset[:p_dim] + ((1, 0),)
                buf_rank[0] += 1
            buffer.sym.rank = tuple(buffer.sym.rank)
            # D- Create and append a loop nest(s) for copying data into/from
            # the temporary buffer. Depending on how the symbol is accessed
            # (read only, read and write, incremented, etc.), different sort
            # of copies are made
            first, last = acc_modes[0], acc_modes[-1]
            for s_p_itspace, binding in itspace_binding.items():
                s_refs, b_refs = zip(*binding)
                if first[0] == READ:
                    copy, init = ast_c_make_copy(b_refs, s_refs, s_p_itspace, Assign)
                    self.loop_opt.header.children.insert(0, copy.children[0])
                if last[0] == WRITE:
                    # If extra information (i.e., a pragma) is present telling that
                    # the argument does not need to be incremented because it does
                    # not contain any meaningful values, then we can safely write
                    # to it. This is an optimization to avoid increments when not
                    # necessarily required
                    op = last[1]
                    ext_acc_mode = [p for p in decl.pragma if isinstance(p, Access)]
                    if ext_acc_mode and ext_acc_mode[0] == WRITE and \
                            len(itspace_binding) == 1:
                        op = Assign
                    copy, init = ast_c_make_copy(s_refs, b_refs, s_p_itspace, op)
                    if to_invert:
                        to_invert = self.loop_opt.header.children.index(to_invert)
                        self.loop_opt.header.children.insert(to_invert, copy.children[0])
                    else:
                        self.loop_opt.header.children.append(copy.children[0])
            # Update the global data structure
            decl_scope[buffer.sym.symbol] = (buffer, ap.LOCAL_VAR)

        # 2) Try adjusting the bounds (i.e. /start/ and /end/ points) of innermost
        # loops such that memory accesses get aligned to the vector length
        iloops = inner_loops(self.loop_opt.header)
        adjusted_loops = []
        for l in iloops:
            adjust = True
            for stmt in l.body:
                sym = stmt.children[0]
                # Cond A- all lvalues must have as fastest varying dimension the
                # one dictated by the innermost loop
                if not (sym.rank and sym.rank[-1] == l.itvar):
                    adjust = False
                    break
                # Cond B- all lvalues must be paddable; that is, they cannot be
                # kernel parameters
                if sym.symbol in decl_scope and decl_scope[sym.symbol][1] == ap.PARAM_VAR:
                    adjust = False
                    break
            # Cond C- the extra iterations induced by bounds adjustment must not
            # alter the result. This is the case if they fall either in a padded
            # region or in a zero-valued region
            alignable_stmts = []
            read_regions = defaultdict(list)
            nonzero_info_l = self.loop_opt.nonzero_info.get(l, [])
            for stmt, ofs in nonzero_info_l:
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
                    # Bound adjustment causes overlapping, so give up
                    adjust = False
                    break
            # Conditions checked, if both passed then adjust loop and offsets
            if adjust:
                # Adjust end point
                l.cond.children[1] = Symbol(vect_roundup(l.end))
                # Adjust start points
                for stmt, ofs in alignable_stmts:
                    ast_update_ofs(stmt, ofs)
                # If all statements were successfully aligned, then put a
                # suitable pragma to tell the compiler
                if len(alignable_stmts) == len(nonzero_info_l):
                    adjusted_loops.append(l)
                # Successful bound adjustment allows forcing simdization
                if self.comp.get('force_simdization'):
                    l.pragma.add(self.comp['force_simdization'])

        # 3) Adding pragma alignment is safe iff:
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

            op = OuterProduct(stmt, unit_stride_loops, self.intr, 'STORE')

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

    """Generate an intrinsics-based outer product vectorisation of a statement."""

    def __init__(self, stmt, loops, intr, mode):
        self.stmt = stmt
        self.intr = intr
        self.loops = loops
        self.mode = mode

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

        :arg vrs: dictionary that associates scalar variables to vector.
                  variables, for which it will be generated a corresponding
                  intrinsics load/set/broadcast.
        :arg decls: list of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated, possibly updated
                    by this method.
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

        :arg node: AST expression to be vectorized.
        :arg ofs: contains the offset of the entry in the left hand side that
                  is being vectorized.
        :arg regs: register allocator.
        :arg decls: list of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated.
        :arg vrs: dictionary that associates scalar variables to vector variables.
                  Updated every time a new scalar variable is encountered.
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

    def _incr_tensor(self, tensor, ofs, regs, out_reg):
        """Add the right hand side contained in out_reg to tensor.

        :arg tensor: the left hand side of the expression being vectorized.
        :arg ofs: contains the offset of the entry in the left hand side that
                  is being computed.
        :arg regs: register allocator.
        :arg out_reg: register variable containing the left hand side.
        """
        if self.mode == 'STORE':
            # Store in memory
            sym = tensor.symbol
            rank = tensor.rank
            ofs = ((1, 0), (1, ofs), (1, 0))
            load = self.intr["symbol_load"](sym, rank, ofs)
            return self.intr["store"](Symbol(sym, rank, ofs),
                                      self.intr["add"](load, out_reg))
        elif self.mode == 'MOVE':
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, self.intr["add"](reg, out_reg))

    def _restore_layout(self, regs, tensor):
        """Restore the storage layout of the tensor.

        :arg regs: register allocator.
        :arg tensor: the left hand side of the expression being vectorized.
        """
        code = []
        t_regs = [Symbol(r, ()) for r in regs.get_tensor()]
        n_regs = len(t_regs)

        # Create tensor symbols
        tensor_syms = []
        for i in range(n_regs):
            ofs = ((1, 0), (1, i), (1, 0))
            tensor_syms.append(Symbol(tensor.symbol, tensor.rank, ofs))

        # Load LHS values from memory
        if self.mode == 'STORE':
            for i, j in zip(tensor_syms, t_regs):
                load_sym = self.intr["symbol_load"](i.symbol, i.rank)
                code.append(Decl(self.intr["decl_var"], j, load_sym))

        # In-register restoration of the tensor layout
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
        """Generate the outer-product intrinsics-based vectorisation code.

        By default, the tensor computed by the outer product vectorization is
        kept in memory, so the layout is restored by means of explicit load and
        store instructions. The resulting code will therefore look like: ::

        for ...
          for j
            for k
              for ...
                A[j][k] = ...intrinsics-based outer product along ``j-k``...
        for j
          for k
            A[j][k] = ...intrinsics-based code for layout restoration...

        The other possibility would be to keep the computed values in temporaries
        after a suitable permutation of the loops in the nest; this variant can be
        activated by passing ``mode='MOVE'``, but it is not recommended unless
        loops are very small *and* a suitable permutation of the nest has been
        chosen to minimize register spilling.
        """
        cols = self.intr["dp_reg"]
        tensor, expr = self.stmt.children
        tensor_size = cols

        # Get source-level variables
        regs = self.Alloc(self.intr, tensor_size)

        # Adjust loops' increment
        self.loops[0].incr.children[1] = c_sym(rows)
        self.loops[1].incr.children[1] = c_sym(cols)

        stmts, decls, vrs = [], {}, {}
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
                stmts.extend(self._vect_mem(vrs, decls))
                incr = self._incr_tensor(tensor, i + ofs, regs, v_expr)
                stmts.append(incr)
            # Register shuffles
            if rows_per_col + (rows_to_peel - peeling) > 0:
                stmts.extend(self._swap_reg(i, vrs))

        # Set initialising and tensor layout code
        layout = self._restore_layout(regs, tensor)
        if self.mode == 'STORE':
            # Tensor layout
            layout_loops = dcopy(self.loops)
            layout_loops[0].incr.children[1] = c_sym(cols)
            layout_loops[0].children = [Block([layout_loops[1]], open_scope=True)]
            layout_loops[1].children = [Block(layout, open_scope=True)]
            layout = layout_loops[0]
        elif self.mode == 'MOVE':
            # Initialiser
            for r in regs.get_tensor():
                decl = Decl(self.intr["decl_var"], Symbol(r, ()), self.intr["setzero"])
                self.loops[1].body.insert(0, decl)
            # Tensor layout
            self.loops[1].body.extend(layout)
            layout = None

        return (stmts, layout)


# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = ap.intrinsics.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len


def vect_rounddown(x):
    """Return x rounded down to the vector length. """
    word_len = ap.intrinsics.get("dp_reg") or 1
    return x - (x % word_len)
