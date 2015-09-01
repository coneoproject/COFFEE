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

from math import ceil
from copy import deepcopy as dcopy
from collections import OrderedDict
from itertools import product

from base import *
from utils import *
import plan
from coffee.visitors import FindInstances


class VectStrategy():

    """Supported vectorization modes."""

    """Generate scalar code suitable to compiler auto-vectorization"""
    AUTO = 1

    """Specialized (intrinsics-based) vectorization using padding"""
    SPEC_PADD = 2

    """Specialized (intrinsics-based) vectorization using peel loop"""
    SPEC_PEEL = 3

    """Specialized (intrinsics-based) vectorization composed with unroll-and-jam
    of outer loops, padding (to enforce data alignment), and peeling of padded
    iterations"""
    SPEC_UAJ_PADD = 4

    """Specialized (intrinsics-based) vectorization composed with unroll-and-jam
    of outer loops and padding (to enforce data alignment)"""
    SPEC_UAJ_PADD_FULL = 5


class LoopVectorizer(object):

    def __init__(self, loop_opt):
        self.loop_opt = loop_opt

    def pad_and_align(self):
        """Padding consists of three major steps:

            * Pad the innermost dimension of all n-dimensional arrays to the nearest
                multiple of the vector length.
            * Round up, to the nearest multiple of the vector length, bounds of all
                innermost loops in which padded arrays are accessed.
            * Since padding may induce data alignment of multi-dimensional arrays
            (in practice, this depends on the presence of offsets as well), add
            suitable '#pragma' to innermost loops to tell the backend compiler
            about this property.

        Padding works as follows. Assume a vector length of size 4, and consider
        the following piece of code: ::

            void foo(int A[10][10]):
              int B[10] = ...
              for i = 0 to 10:
                for j = 0 to 10:
                  A[i][j] = B[i][j]

        Once padding is applied, the code will look like: ::

            void foo(int A[10][10]):
              int _A[10][12] = {{0.0}};
              int B[10][12] = ...
              for i = 0 to 10:
                for j = 0 to 12:
                  _A[i][j] = B[i][j]

              for i = 0 to 10:
                for j = 0 to 10:
                  A[i][j] = _A[i][j]

        Extra care is taken if offsets (e.g. A[i+3][j+3] ...) are used. In such
        a case, the buffer array '_A' in the example above can be vector-expanded: ::

            int _A[x][10][12];
            ...

        Where 'x' corresponds to the number of different offsets used in a given
        iteration space along the innermost dimension.

        Finally, all arrays are decorated with suitable attributes to enforce
        alignment to (the size in bytes of) the vector length.
        """
        # Aliases
        decls = self.loop_opt.decls
        header = self.loop_opt.header
        info = visit(header, info_items=['symbols_dep', 'symbols_mode', 'symbol_refs',
                                         'fors'])
        nz_syms = self.loop_opt.nz_syms
        symbols_dep = info['symbols_dep']
        symbols_mode = info['symbols_mode']
        symbol_refs = info['symbol_refs']
        retval = FindInstances.default_retval()
        to_invert = FindInstances(Invert).visit(header, ret=retval)[Invert]
        # Vectorization aliases
        vector_length = plan.isa["dp_reg"]
        align_attr = plan.compiler['align'](plan.isa['alignment'])

        # 0) Under some circumstances, do not pad
        # A- Loop increments must be equal to 1, because at the moment the
        #    machinery for ensuring the correctness of the transformation for
        #    non-uniform and non-unitary increments is missing
        if any([l.increment != 1 for l, _ in flatten(info['fors'])]):
            return

        # Padding occurs along the innermost dimension
        p_dim = -1

        # 1) Pad arrays by extending the innermost dimension
        buffers = []
        for decl_name, decl in decls.items():
            if not decl.sym.rank:
                continue
            p_rank = decl.sym.rank[:p_dim] + (vect_roundup(decl.sym.rank[p_dim]),)
            if decl.scope == LOCAL:
                if p_rank != decl.sym.rank:
                    # Padding
                    decl.sym.rank = p_rank
                # Alignment
                decl.attr.append(align_attr)
                continue
            # Examined symbol is a FunDecl argument, so a buffer might be required
            access_modes, dataspaces = [], []
            p_info = OrderedDict()
            # A- Analyze occurrences of the FunDecl argument in the AST
            for s, _ in symbol_refs[decl_name]:
                if s is decl.sym or not s.rank:
                    continue
                # ... the access mode (READ, WRITE, ...)
                access_modes.append(symbols_mode[s])
                # ... the offset along the innermost dimension
                p_offset = s.offset[p_dim][1]
                # ... and the iteration and the data spaces
                loops = tuple(l for l in symbols_dep[s] if l.dim in s.rank)
                itspace = tuple((l.start, l.end) for l in loops)
                dataspace = [None for r in s.rank]
                for l in loops:
                    # Assume, initially, the dataspace spans the whole dimension,
                    # then try to limit it based on available information
                    index = s.rank.index(l.dim)
                    l_dataspace = (0, decl.sym.rank[index])
                    offset = s.offset[index][1]
                    if not isinstance(offset, str):
                        l_dataspace = (l.start + offset, l.end + offset)
                    dataspace[index] = l_dataspace
                dataspaces.append(tuple(dataspace))
                p_info.setdefault((itspace, p_offset), (loops, []))[1].append(s)
            # B- Check dataspace overlap. Dataspaces ...
            # ... should either completely overlap (will be mapped to the same buffer)
            # ... or be disjoint
            will_break = False
            for ds1, ds2 in product(dataspaces, dataspaces):
                for d1, d2 in zip(ds1, ds2):
                    if ItSpace(mode=0).intersect([d1, d2]) not in [(0, 0), d1]:
                        will_break = True
            if will_break:
                continue
            # C- Create a padded temporary buffer for efficient vectorization
            buf_name, buf_rank = '_%s' % decl_name, 0
            itspace_mapper = OrderedDict()
            for (itspace, p_offset), (loops, syms) in p_info.items():
                if not (p_rank != decl.sym.rank or
                        isinstance(p_offset, str) or
                        vect_roundup(p_offset) > p_offset):
                    # Useless to pad in this case
                    continue
                mapped = set()
                for s in syms:
                    original_s = dcopy(s)
                    s.symbol = buf_name
                    s.rank = (buf_rank,) + s.rank
                    s.offset = ((1, 0),) + s.offset[:p_dim] + ((1, 0),)
                    if s.offset not in mapped:
                        # Map buffer symbol to each FunDecl symbol occurrence
                        # avoiding duplicate, useless copies
                        mapping = (original_s, dcopy(s))
                        itspace_mapper.setdefault(itspace, (loops, []))[1].append(mapping)
                        mapped.add(s.offset)
                buf_rank += 1
            if buf_rank == 0:
                continue
            buf_rank = (buf_rank,) + p_rank
            init = ArrayInit(np.ndarray(shape=(1,)*len(buf_rank), buffer=np.array(0.0)))
            buffer = Decl(decl.typ, Symbol(buf_name, buf_rank), init, attributes=[align_attr])
            buffer.scope = LOCAL
            buffer.sym.rank = tuple(buffer.sym.rank)
            header.children.insert(0, buffer)
            # D- Create and append a loop nest(s) for copying data into/from
            # the temporary buffer. Depending on how the symbol is accessed
            # (read only, read and write, incremented, etc.), different sort
            # of copies are made
            first, last = access_modes[0], access_modes[-1]
            for itspace, (loops, mapper) in itspace_mapper.items():
                if first[0] == READ:
                    stmts = [Assign(b, s) for s, b in mapper]
                    copy_back = ItSpace(mode=2).to_for(loops, stmts=stmts)
                    header.children.insert(0, copy_back[0])
                if last[0] == WRITE:
                    # If extra information (a pragma) is present, telling that
                    # the argument does not need to be incremented because it does
                    # not contain any meaningful values, then we can safely write
                    # to it. This is an optimization to avoid increments when not
                    # necessarily required
                    could_incr = WRITE in decl.pragma and len(itspace_mapper) == 1
                    op = Assign if could_incr else last[1]
                    stmts = [op(s, b) for s, b in mapper]
                    copy_back = ItSpace(mode=2).to_for(loops, stmts=stmts)
                    if to_invert:
                        insert_at_elem(header.children, to_invert[0], copy_back[0])
                    else:
                        header.children.append(copy_back[0])
            # E) Update the global data structures
            decls[buffer.sym.symbol] = buffer
            nz_syms[buf_name] = tuple([(r, 0)] for r in buf_rank)
            buffers.append(buffer)

        # 2) Round up the bounds (i.e. /start/ and /end/ points) of innermost
        # loops such that memory accesses get aligned to the vector length
        for l in inner_loops(header):
            should_round, should_vectorize = True, True
            for stmt in l.body:
                sym, expr = stmt.children
                # Condition A: all lvalues must have the innermost loop as fastest
                # varying dimension
                if not (sym.rank and sym.rank[p_dim] == l.dim):
                    should_round = False
                    should_vectorize = False
                    break
                # Condition B: all lvalues must be paddable; that is, they cannot be
                # kernel parameters
                if sym.symbol in decls and decls[sym.symbol].scope == EXTERNAL:
                    should_round = False
                    break
            lvalues = {}
            aligned_l = dcopy(l)
            for stmt in aligned_l.body:
                sym, expr = stmt.children
                # Condition C: statements using offsets to write buffers should
                # not be aligned
                if decls.get(sym.symbol) in buffers and sym.offset[p_dim][1] > 0:
                    should_round = False
                    break
                # Condition D: extra iterations induced by bounds and offset rounding
                # should /not/ alter the result.
                symbols = FindInstances(Symbol).visit(stmt)[Symbol]
                symbols = [s for s in symbols if any(r == l.dim for r in s.rank)]
                for s in symbols:
                    # First of all, we need to be sure we can inspect the symbol
                    # declaration
                    if s.symbol not in decls or \
                            not isinstance(decls[s.symbol].init, ArrayInit):
                        should_round = False
                        break
                    values = decls[s.symbol].init.values
                    # Now we check if lowering the start point would be unsafe
                    # because it would result in /not/ executing iterations
                    # that should be executed
                    offset = s.offset[p_dim][1]
                    start = vect_rounddown(offset)
                    end = start + vect_roundup(l.end)
                    if end < offset + l.end:
                        should_round = False
                    # It remains to check if the extra iterations would alter the
                    # result because they would access non-zero entries
                    extra = range(start, offset) + range(offset + l.end + 1, end + 1)
                    for i in extra:
                        if i >= values.shape[p_dim]:
                            # In the padded region, safe
                            continue
                        nz_s = nz_syms.get(s.symbol, ([(0, 0)],))[p_dim]
                        if any(i in range(j[1], j[0] + j[1]) for j in nz_s):
                            # The i-th extra iteration does not fall in a zero-valued
                            # region, so we should not round
                            should_round = False
                    # Round down the start point
                    ast_update_ofs(s, {l.dim: start})
                    # Track the modified lvalues
                    if s is sym:
                        lvalues[s] = (start, offset)
            if should_round:
                l.body = aligned_l.body
                # Round up the end point
                l.end = vect_roundup(l.end)
                # Note: it was safe to round an lvalue S, but now all subsequent
                # accesses to the same symbol S might also have to be rounded.
                # This is the case when the offset used by S' falls in the rounded
                # region. Note such an S' /cannot/ be an lvalue since the rounding
                # happens over a zero-valued region.
                for lvalue, (start, orig_ofs) in lvalues.items():
                    references = SymbolReferences().visit(header)[lvalue.symbol]
                    references = [r for r, p in references]
                    for r in references[references.index(lvalue)+1:]:
                        r_rank, r_ofs = r.rank[p_dim], r.offset[p_dim][1]
                        if r_ofs in range(orig_ofs, orig_ofs + l.end):
                            ast_update_ofs(r, {r_rank: start - r_ofs}, increase=True)
                    # The corresponding /nz_syms/ info should also be updated
                    nz_lvalue = list(nz_syms[r.symbol])
                    for i, (size, offset) in enumerate(nz_syms[r.symbol][p_dim]):
                        if orig_ofs in range(offset, offset + size):
                            nz_lvalue[p_dim][i] = (size, offset - (orig_ofs - start))
                    nz_syms[r.symbol] = tuple(nz_lvalue)
                if l.start % vector_length == 0 and l.size % vector_length == 0:
                    l.pragma.add(plan.compiler["align_forloop"])
            # Enforce vectorization if loop size is a multiple of the vector length
            if should_vectorize and l.size % vector_length == 0:
                l.pragma.add(plan.compiler['force_simdization'])

    def specialize(self, opts, factor=1):
        """Generate code for specialized expression vectorization. Check for peculiar
        memory access patterns in an expression and replace scalar code with highly
        optimized vector code. Currently, the following patterns are supported:

        * Outer products - e.g. A[i]*B[j]

        Also, code generation is supported for the following instruction sets:

        * AVX

        The parameter ``opts`` can be used to drive the transformation process by
        specifying one of the vectorization strategies in :class:`VectStrategy`.
        """
        layout = None
        for stmt, expr_info in self.loop_opt.exprs.items():
            if expr_info.dimension != 2:
                continue
            parent = expr_info.parent
            domain_loops = expr_info.domain_loops
            domain_loops_parents = expr_info.domain_loops_parents

            # Check if outer-product vectorization is actually doable
            vect_len = plan.isa["dp_reg"]
            rows = domain_loops[0].size
            if rows < vect_len:
                continue

            op = OuterProduct(stmt, domain_loops, 'STORE')

            # Vectorisation
            vs = VectStrategy
            unroll_factor = factor if opts in [vs.SPEC_UAJ_PADD, vs.SPEC_UAJ_PADD_FULL] else 1
            rows_per_it = vect_len*unroll_factor
            if opts == vs.SPEC_UAJ_PADD:
                if rows_per_it <= rows:
                    body, layout = op.generate(rows_per_it)
                else:
                    # Unroll factor too big
                    body, layout = op.generate(vect_len)
            elif opts == SPEC_UAJ_PADD_FULL:
                if rows <= rows_per_it or vect_roundup(rows) % rows_per_it > 0:
                    # Cannot unroll too much
                    body, layout = op.generate(vect_len)
                else:
                    body, layout = op.generate(rows_per_it)
            elif opts in [vs.SPEC_PADD, vs.SPEC_PEEL]:
                body, layout = op.generate(vect_len)
            else:
                raise RuntimeError("Don't know how to vectorize option %s" % opts)

            # Construct the remainder loop
            if opts != vs.SPEC_UAJ_PADD_FULL and rows > rows_per_it and rows % rows_per_it > 0:
                # Adjust bounds and increments of the main, layout and remainder loops
                domain_outerloop = domain_loops[0]
                peel_loop = dcopy(domain_loops)
                bound = domain_outerloop.end
                bound -= bound % rows_per_it
                domain_outerloop.end, layout.end = bound, bound
                peel_loop[0].init.init = Symbol(bound)
                peel_loop[0].increment, peel_loop[1].increment = 1, 1
                # Append peeling loop after the main loop
                domain_outerparent = domain_loops_parents[0].children
                insert_at_elem(domain_outerparent, domain_outerloop, peel_loop[0], 1)

            # Replace scalar with vector code
            ofs = parent.children.index(stmt)
            parent.children[ofs:ofs] = body
            parent.children.remove(stmt)

        # Insert the layout code right after the loop nest enclosing the expression
        if layout:
            insert_at_elem(self.loop_opt.header.children, expr_info.loops[0], layout, 1)


class OuterProduct():

    """Generate an intrinsics-based outer product vectorisation of a statement."""

    def __init__(self, stmt, loops, mode):
        self.stmt = stmt
        self.loops = loops
        self.mode = mode

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, tensor_size):
            nres = max(plan.isa["dp_reg"], tensor_size)
            self.ntot = plan.isa["avail_reg"]
            self.res = [plan.isa["reg"](v) for v in range(nres)]
            self.var = [plan.isa["reg"](v) for v in range(nres, self.ntot)]
            self.i = plan.isa

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
                if node.rank and node.rank[-1] == self.loops[1].dim]

        if step in [0, 2]:
            return [Assign(r, plan.isa["l_perm"](r, "5")) for r in regs]
        elif step == 1:
            return [Assign(r, plan.isa["g_perm"](r, r, "1")) for r in regs]
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
            if node.rank and node.rank[-1] in [l.dim for l in self.loops]:
                exp = plan.isa["symbol_load"](node.symbol, node.rank, node.offset)
            else:
                exp = plan.isa["symbol_set"](node.symbol, node.rank, node.offset)
            if not decls.get(node.gencode()):
                decls[node.gencode()] = reg
                stmt.append(Decl(plan.isa["decl_var"], reg, exp))
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
            if node.rank and self.loops[0].dim == node.rank[-1]:
                # The symbol depends on the outer loop dimension, so add offset
                n_ofs = tuple([(1, 0) for i in range(len(node.rank)-1)]) + ((1, ofs),)
                node = Symbol(node.symbol, dcopy(node.rank), n_ofs)
            node_ide = node.gencode()
            if node_ide not in decls:
                reg = [k for k in vrs.keys() if k.gencode() == node_ide]
                if not reg:
                    vrs[node] = Symbol(regs.get_reg())
                    return vrs[node]
                else:
                    return vrs[reg[0]]
            else:
                return decls[node_ide]
        elif isinstance(node, Par):
            return self._vect_expr(node.child, ofs, regs, decls, vrs)
        else:
            left = self._vect_expr(node.left, ofs, regs, decls, vrs)
            right = self._vect_expr(node.right, ofs, regs, decls, vrs)
            if isinstance(node, Sum):
                return plan.isa["add"](left, right)
            elif isinstance(node, Sub):
                return plan.isa["sub"](left, right)
            elif isinstance(node, Prod):
                return plan.isa["mul"](left, right)
            elif isinstance(node, Div):
                return plan.isa["div"](left, right)

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
            ofs = tensor.offset[:-2] + ((1, ofs),) + tensor.offset[-1:]
            load = plan.isa["symbol_load"](sym, rank, ofs)
            return plan.isa["store"](Symbol(sym, rank, ofs),
                                     plan.isa["add"](load, out_reg))
        elif self.mode == 'MOVE':
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, plan.isa["add"](reg, out_reg))

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
            ofs = tensor.offset[:-2] + ((1, i),) + tensor.offset[-1:]
            tensor_syms.append(Symbol(tensor.symbol, tensor.rank, ofs))

        # Load LHS values from memory
        if self.mode == 'STORE':
            for i, j in zip(tensor_syms, t_regs):
                load_sym = plan.isa["symbol_load"](i.symbol, i.rank, i.offset)
                code.append(Decl(plan.isa["decl_var"], j, load_sym))

        # In-register restoration of the tensor layout
        perm = plan.isa["g_perm"]
        uphi = plan.isa["unpck_hi"]
        uplo = plan.isa["unpck_lo"]
        typ = plan.isa["decl_var"]
        vect_len = plan.isa["dp_reg"]
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
                code.append(plan.isa["store"](tensor_syms[ofs], t_regs[ofs]))

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
        cols = plan.isa["dp_reg"]
        tensor, expr = self.stmt.children
        tensor_size = cols

        # Get source-level variables
        regs = self.Alloc(tensor_size)

        # Adjust loops' increment
        self.loops[0].incr.children[1] = Symbol(rows)
        self.loops[1].incr.children[1] = Symbol(cols)

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
            layout_loops[0].incr.children[1] = Symbol(cols)
            layout_loops[0].children = [Block([layout_loops[1]], open_scope=True)]
            layout_loops[1].children = [Block(layout, open_scope=True)]
            layout = layout_loops[0]
        elif self.mode == 'MOVE':
            # Initialiser
            for r in regs.get_tensor():
                decl = Decl(plan.isa["decl_var"], Symbol(r, ()), plan.isa["setzero"])
                self.loops[1].body.insert(0, decl)
            # Tensor layout
            self.loops[1].body.extend(layout)
            layout = None

        return (stmts, layout)


# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = plan.isa.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len


def vect_rounddown(x):
    """Return x rounded down to the vector length. """
    word_len = plan.isa.get("dp_reg") or 1
    return x - (x % word_len)
