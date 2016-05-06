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
import system
from logger import warn
from coffee.visitors import FindInstances


class VectStrategy(object):

    """Supported vectorization modes."""

    """Generate scalar code suitable for compiler auto-vectorization"""
    AUTO = 1

    """Specialized (intrinsics-based) vectorization using padding"""
    SPEC_PADD = 2

    """Specialized (intrinsics-based) vectorization using a peeling loop"""
    SPEC_PEEL = 3

    """Specialized (intrinsics-based) vectorization composed with unroll-and-jam
    of outer loops, padding (to enforce data alignment), and peeling of padded
    iterations"""
    SPEC_UAJ_PADD = 4

    """Specialized (intrinsics-based) vectorization composed with unroll-and-jam
    of outer loops and padding (to enforce data alignment)"""
    SPEC_UAJ_PADD_FULL = 5


class LoopVectorizer(object):

    def __init__(self, loop_opt, kernel=None):
        self.kernel = kernel or loop_opt.header
        self.header = loop_opt.header
        self.loop = loop_opt.loop
        self.decls = loop_opt.decls
        self.exprs = loop_opt.exprs
        self.expr_graph = loop_opt.expr_graph
        self.nz_syms = loop_opt.nz_syms

    def autovectorize(self, p_dim=-1):
        """Generate code suitable for compiler auto-vectorization.

        Three code transformations may be applied here:

            * Padding
            * Data alignment

        OR, if the outermost loop has an interation space much larger than that
        of the inner loops,

            * Data layout transposition

        Padding consists of three major steps:

            * Pad the innermost dimension of all n-dimensional arrays to the nearest
                multiple of the vector length.
            * Round up, to the nearest multiple of the vector length, the bounds of all
                innermost loops in which padded arrays are accessed.
            * Since padding may induce data alignment of multi-dimensional arrays
            (in practice, this depends on the presence of offsets as well), add
            suitable '#pragma' to innermost loops to tell the backend compiler
            if this property holds.

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

        :arg p_dim: the array dimension that should be padded (defaults to the
            innermost, or -1)
        """
        buffer = self._pad(p_dim)
        if buffer:
            self._align_data(buffer, p_dim)

    def _pad(self, p_dim):
        """Apply padding."""
        info = visit(self.header, info_items=['fors', 'symbols_dep',
                                              'symbols_mode', 'symbol_refs'])
        symbols_dep = info['symbols_dep']
        symbols_mode = info['symbols_mode']
        symbol_refs = info['symbol_refs']
        retval = FindInstances.default_retval()
        to_invert = FindInstances(Invert).visit(self.header, ret=retval)[Invert]

        # Loop increments different than 1 are unsupported
        if any([l.increment != 1 for l, _ in flatten(info['fors'])]):
            return None

        buf_decl = None
        for decl_name, decl in self.decls.items():
            if not decl.size or decl.is_pointer_type:
                continue

            p_rank = decl.size[:p_dim] + (vect_roundup(decl.size[p_dim]),)
            if decl.size[p_dim] == 1 or p_rank == decl.size:
                continue

            if decl.scope == LOCAL:
                decl.pad(p_rank)
                continue

            # At this point we are sure /decl/ is a FunDecl argument

            # A) Can a buffer actually be allocated ?
            symbols = [s for s, _ in symbol_refs[decl_name] if s is not decl.sym]
            if not all(s.dim == decl.sym.dim and s.is_const_offset for s in symbols):
                continue
            periods = flatten([s.periods for s in symbols])
            if not all(p == 1 for p in periods):
                continue

            # ... must be either READ or WRITE mode
            modes = [symbols_mode[s][0] for s in symbols]
            if not modes or any(m != modes[0] for m in modes):
                continue
            mode = modes[0]
            if mode not in [READ, WRITE]:
                continue

            # ... accesses to entries in /decl/ must be explicit in all loop nests
            nests = OrderedDict((s, [l for l in symbols_dep[s] if l.dim in s.rank])
                                for s in symbols)
            if not all(s.dim == len(n) for s, n in nests.items()):
                continue
            for s, n in nests.items():
                n.sort(key=lambda l: s.rank.index(l.dim))

            # ... is there any overlap in the memory accesses? Memory accesses must:
            # - either completely overlap (they will be mapped to the same buffer)
            # - OR be disjoint
            dspace = []
            for s, n in nests.items():
                dspace.append(tuple((l.start + o, l.end + o) for l, o in zip(n, s.strides)))
            will_break = False
            for ds1, ds2 in product(dspace, dspace):
                for d1, d2 in zip(ds1, ds2):
                    if ItSpace(mode=0).intersect([d1, d2]) not in [(0, 0), d1]:
                        will_break = True
            if will_break:
                continue

            # B) Create a buffer of suitable size
            # ... collect iteration space info
            p_info = OrderedDict()
            for s, n in nests.items():
                itspace = tuple((l.start, l.end) for l in n)
                n, symbols = p_info.setdefault(itspace, (n, []))
                symbols.append(s)

            # ... initialize buffer-related data
            buf_name = '_' + decl_name
            buf_nz = self.nz_syms.setdefault(buf_name, [])

            # ... determine the non zero-valued region in the buffer
            for n, itspace in enumerate(p_info.keys()):
                for s in p_info[itspace][1]:
                    new_nz = [(1, n)]
                    for i, j in zip(s.strides[:p_dim], itspace[:p_dim]):
                        new_nz.append((j[1] - j[0], i))
                    new_nz.append((itspace[p_dim][1] - itspace[p_dim][0], 0))
                    buf_nz.append(tuple(new_nz))

            # ... replace symbols in the AST with proper buffer instances
            itspace_mapper = OrderedDict()
            for n, itspace in enumerate(p_info.keys()):
                nest, symbols = p_info[itspace]
                mapper = OrderedDict()
                itspace_mapper[itspace] = (nest, mapper)
                for s in symbols:
                    original = Symbol(s.symbol, s.rank, s.offset)
                    s.symbol = buf_name
                    s.rank = (n,) + s.rank
                    s.offset = ((1, 0),) + s.offset[:p_dim] + ((1, 0),)
                    if s.urepr not in [i.urepr for i in mapper.values()]:
                        mapper[original] = Symbol(s.symbol, s.rank, s.offset)

            # ... insert the buffer into the AST
            buf_rank = (n,) + decl.size
            init = ArrayInit(np.ndarray(shape=(1,)*len(buf_rank), buffer=np.array(0.0)))
            buf_decl = Decl(decl.typ, Symbol(buf_name, buf_rank), init, scope=BUFFER)
            buf_decl.pad((n,) + p_rank)
            self.header.children.insert(0, buf_decl)

            # C) Create a loop nest for copying data into/from the buffer
            for itspace, (nest, mapper) in itspace_mapper.items():

                if mode == READ:
                    stmts = [Assign(b, s) for s, b in mapper.items()]
                    copy_back = ItSpace(mode=2).to_for(nest, stmts=stmts)
                    insert_at_elem(self.header.children, buf_decl, copy_back[0], ofs=1)

                elif mode == WRITE:
                    # If extra information (a pragma) is present, telling that
                    # the argument does not need to be incremented because it does
                    # not contain any meaningful values, then we can safely write
                    # to it. This optimization may avoid useless increments
                    can_write = WRITE in decl.pragma and len(itspace_mapper) == 1
                    op = Assign if can_write else Incr
                    stmts = [op(s, b) for s, b in mapper.items()]
                    copy_back = ItSpace(mode=2).to_for(nest, stmts=stmts)
                    if to_invert:
                        insert_at_elem(self.header.children, to_invert[0], copy_back[0])
                    else:
                        self.header.children.append(copy_back[0])

            # D) Update the global data structures
            self.decls[buf_name] = buf_decl

        return buf_decl

    def _align_data(self, buffer, p_dim):
        """Apply data alignment. This boils down to:

            * Decorate declarations with qualifiers for data alignment
            * Round up the bounds (i.e. /start/ and /end/ points) of loops such
            that all memory accesses get aligned to the vector length. Several
            checks ensure the correctness of the transformation.
        """
        vector_length = system.isa["dp_reg"]
        align = system.compiler['align'](system.isa['alignment'])

        # Array alignment
        for decl in self.decls.values():
            if decl.sym.rank and decl.scope == LOCAL:
                decl.attr.append(align)

        # Loop bounds adjustment
        for l in inner_loops(self.header):
            should_round = True

            for stmt in l.body:
                sym, expr = stmt.lvalue, stmt.rvalue
                decl = self.decls[sym.symbol]

                # Condition A: the fastest varying dimension of the lvalue must be /l/
                if not sym.rank or not sym.rank[p_dim] == l.dim:
                    should_round = False
                    break

                # Condition B: the lvalue must have been padded
                if decl.size[p_dim] != vect_roundup(decl.size[p_dim]):
                    should_round = False
                    break

                # Condition C: extra iterations induced by bounds and offset rounding
                # must not alter the computation
                symbols = [sym] + FindInstances(Symbol).visit(expr)[Symbol]
                symbols = [s for s in symbols if s.rank and any(r == l.dim for r in s.rank)]
                if any(not s.is_unit_period for s in symbols):
                    # Cannot infer the access pattern so must break
                    should_round = False
                    break
                for s in symbols:
                    stride = s.strides[p_dim]
                    extra = range(stride + l.size, stride + vect_roundup(l.size))
                    # Do any of the extra iterations alter the computation ?
                    if any(i > decl.size[p_dim] for i in extra):
                        # ... outside of the legal region, abort
                        should_round = False
                        break
                    if all(i >= decl.core[p_dim for i in extra]):
                        # ... in the padded region, pass
                        continue
                    nz = list(self.nz_syms.get(s.symbol))
                    if not nz:
                        # ... lacks the non zero-valued entries mapping, abort
                        should_round = False
                        break
                    if s.symbol == buffer.lvalue:
                        # ... the buffer requires special handling
                        nz = [i for i in nz if s.rank[0] in range(i[0][1], ...)]
                        # WIP TODO use namedtuple
                        for i in list(nz):
                            dim = i[0]
                            if s.rank[0] not in range(dim[1], dim[0] + dim[1]):
                                nz.remove(i)


                    # It remains to check if the extra iterations would alter the
                    # result by accessing non zero-valued entries
                    extra = range(start, offset) + range(offset + l.end + 1, end + 1)
                    for i in extra:
                        if i > decl.size[p_dim]:
                            should_round = False
                            break
                        if i >= decl.core[p_dim]:
                            # In the padded region, safe
                            continue
                        # If /s/ is the buffer, we need to access the actual dimension
                        # written by /stmt/
                        nz = list(self.nz_syms.get(s.symbol,
                                  [tuple((r, 0) for r in decl.core)]))
                        if buffer and s.symbol == buffer.sym.symbol:
                            for j in list(nz):
                                dim = j[0]
                                if s.rank[0] not in range(dim[1], dim[0]+dim[1]):
                                    nz.remove(j)
                        # Now we can finally check if the i-th extra iteration falls in a
                        # zero-valued region (in which case we are happy) or not
                        if any(i in range(k, j + k) for j, k in [j[p_dim] for j in nz]):
                            should_round = False
                            break
                    # Round down the start point
                    ast_update_ofs(s, {l.dim: start})

                    # Track the rounding in the lvalue and infer if the zero-valued
                    # region in /sym/ has now become a /non/ zero-valued region.
                    # Note: /sym/ is the first element in the /symbols/ list
                    lvalues.setdefault(sym, [start, offset, False])
                    if start < offset or \
                            any(v[2] for k, v in lvalues.items() if k.urepr == s.urepr):
                        lvalues[sym][2] = True
                    #if s is sym:
                        # Assume there's no need to remove the non zero-valued region ...
                    #    lvalues[sym] = (start, offset, False)
                    #else:
                    #    from IPython import embed; embed()
                    #    # ... and then potentially change it as the rvalue is examined: ...
                    #    if start == offset and lvalues[sym][0] < lvalues[sym][1] or \
                    #            any(v[2] for k, v in lvalues.items() if k.urepr == s.urepr):
                            # ... this is the case if at least one symbol in the rvalue
                            # was not rounded while the lvalue was, /or/
                            # if at least one symbol in the rvalue appeared as an lvalue
                            # in a previoud /stmt/ and such lvalue was rounded down
                    #        lvalues[sym] = lvalues[sym][:-1] + (True,)

            if should_round:
                l.body = aligned.body
                l.end = vect_roundup(l.end)

                # It was safe to round an lvalue S, but now all subsequent
                # accesses to the same symbol S /might/ have to be rounded too.
                # This is the case /iff/:
                #
                # 1) in rounding, we are writing to the rounded region (e.g.,
                # A[i+2] += B[i] ===> A[i] += B[i]: since B[i] wasn't rounded,
                # it means that B[i] is != 0, so we are now writing to A[i],
                # which was previously 0, instead of A[i+2]), /and/
                #
                # 2) the offset used by S', which is a later reference to S, falls
                # in the rounded region (e.g., starting from the example above,
                # if there's another line D[i] = A[i+2], then it must be D[i] = A[i],
                # otherwise if it were D[i] = A[i+5] we should not round). Note
                # how such an S' /cannot/ be an lvalue, since we have already
                # enforced that rounding only happens over zero-valued regions.
                # The first property is captured by /remove_nz/

                for lvalue, (start, orig_ofs, remove_nz) in lvalues.items():
                    if not remove_nz:
                        # Property 1) above does not hold
                        continue
                    references = SymbolReferences().visit(self.header)[lvalue.symbol]
                    references = [r for r, p in references]
                    for r in references[references.index(lvalue)+1:]:
                        r_rank, r_ofs = r.rank[p_dim], r.offset[p_dim][1]
                        if r_ofs in range(orig_ofs, orig_ofs + l.end):
                            ast_update_ofs(r, {r_rank: start - r_ofs}, increase=True)
                    # The corresponding /nz_syms/ info should also be updated
                    nz_lvalue = zip(*self.nz_syms.get(r.symbol, [((0, 0),)]))
                    for i, (size, offset) in enumerate(nz_lvalue[p_dim]):
                        if orig_ofs in range(offset, offset + size):
                            self.nz_syms[r.symbol][i] = \
                                self.nz_syms[r.symbol][i][:p_dim] + \
                                ((size, offset-(orig_ofs-start)),)
                if l.start % vector_length == 0 and l.size % vector_length == 0:
                    l.pragma.add(system.compiler["align_forloop"])

            # Enforce vectorization if loop size is a multiple of the vector length
            if should_round and l.size % vector_length == 0:
                l.pragma.add(system.compiler['force_simdization'])

    def _transpose_layout(self):
        dim = self.loop.dim
        retval = FindInstances.default_retval()
        symbols = FindInstances(Symbol).visit(self.loop, ret=retval)[Symbol]
        symbols = [s for s in symbols if any(r == dim for r in s.rank) and s.dim > 1]

        # Cannot handle arrays with more than 2 dimensions
        if any(s.dim > 2 for s in symbols):
            return

        mapper = OrderedDict()
        for s in symbols:
            mapper.setdefault(self.decls[s.symbol], list()).append(s)

        for decl, syms in mapper.items():
            # Adjust the declaration
            transposed_values = decl.init.values.transpose()
            decl.init.values = transposed_values
            decl.sym.rank = transposed_values.shape

            # Adjust the instances
            for s in syms:
                s.rank = tuple(reversed(s.rank))

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
        vs = VectStrategy
        if opts not in [vs.SPEC_UAJ_PADD, vs.SPEC_UAJ_PADD_FULL,
                        vs.SPEC_PADD, vs.SPEC_PEEL]:
            warn("Don't know how to specialize vectorization for %s" % opts)
        if system.isa['inst_set'] == 'SSE':
            warn("Don't know how to specialize vectorization for SSE")

        layout = None
        for stmt, expr_info in self.exprs.items():
            if expr_info.dimension != 2:
                continue
            parent = expr_info.parent
            linear_loops = expr_info.linear_loops
            linear_loops_parents = expr_info.linear_loops_parents

            # Check if outer-product vectorization is actually doable
            vect_len = system.isa["dp_reg"]
            rows = linear_loops[0].size
            if rows < vect_len:
                continue

            op = OuterProduct(stmt, linear_loops, 'STORE')

            # Vectorisation
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

            # Construct the remainder loop
            if opts != vs.SPEC_UAJ_PADD_FULL and rows > rows_per_it and rows % rows_per_it > 0:
                # Adjust bounds and increments of the main, layout and remainder loops
                linear_outerloop = linear_loops[0]
                peel_loop = dcopy(linear_loops)
                bound = linear_outerloop.end
                bound -= bound % rows_per_it
                linear_outerloop.end, layout.end = bound, bound
                peel_loop[0].init.init = Symbol(bound)
                peel_loop[0].increment, peel_loop[1].increment = 1, 1
                # Append peeling loop after the main loop
                linear_outerparent = linear_loops_parents[0].children
                insert_at_elem(linear_outerparent, linear_outerloop, peel_loop[0], 1)

            # Replace scalar with vector code
            ofs = parent.children.index(stmt)
            parent.children[ofs:ofs] = body
            parent.children.remove(stmt)

        # Insert the layout code right after the loop nest enclosing the expression
        if layout:
            insert_at_elem(self.header.children, expr_info.loops[0], layout, 1)


class OuterProduct(object):

    """Generate an intrinsics-based outer product vectorisation of a statement."""

    def __init__(self, stmt, loops, mode):
        self.stmt = stmt
        self.loops = loops
        self.mode = mode

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, tensor_size):
            nres = max(system.isa["dp_reg"], tensor_size)
            self.ntot = system.isa["avail_reg"]
            self.res = [system.isa["reg"](v) for v in range(nres)]
            self.var = [system.isa["reg"](v) for v in range(nres, self.ntot)]
            self.i = system.isa

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
            return [Assign(r, system.isa["l_perm"](r, "5")) for r in regs]
        elif step == 1:
            return [Assign(r, system.isa["g_perm"](r, r, "1")) for r in regs]
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
                exp = system.isa["symbol_load"](node.symbol, node.rank, node.offset)
            else:
                exp = system.isa["symbol_set"](node.symbol, node.rank, node.offset)
            if not decls.get(node.gencode()):
                decls[node.gencode()] = reg
                stmt.append(Decl(system.isa["decl_var"], reg, exp))
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
        else:
            left = self._vect_expr(node.left, ofs, regs, decls, vrs)
            right = self._vect_expr(node.right, ofs, regs, decls, vrs)
            if isinstance(node, Sum):
                return system.isa["add"](left, right)
            elif isinstance(node, Sub):
                return system.isa["sub"](left, right)
            elif isinstance(node, Prod):
                return system.isa["mul"](left, right)
            elif isinstance(node, Div):
                return system.isa["div"](left, right)

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
            load = system.isa["symbol_load"](sym, rank, ofs)
            return system.isa["store"](Symbol(sym, rank, ofs),
                                       system.isa["add"](load, out_reg))
        elif self.mode == 'MOVE':
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, system.isa["add"](reg, out_reg))

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
                load_sym = system.isa["symbol_load"](i.symbol, i.rank, i.offset)
                code.append(Decl(system.isa["decl_var"], j, load_sym))

        # In-register restoration of the tensor layout
        perm = system.isa["g_perm"]
        uphi = system.isa["unpck_hi"]
        uplo = system.isa["unpck_lo"]
        typ = system.isa["decl_var"]
        vect_len = system.isa["dp_reg"]
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
                code.append(system.isa["store"](tensor_syms[ofs], t_regs[ofs]))

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
        cols = system.isa["dp_reg"]
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
                decl = Decl(system.isa["decl_var"], Symbol(r, ()), system.isa["setzero"])
                self.loops[1].body.insert(0, decl)
            # Tensor layout
            self.loops[1].body.extend(layout)
            layout = None

        return (stmts, layout)


# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = system.isa.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len


def vect_rounddown(x):
    """Return x rounded down to the vector length. """
    word_len = system.isa.get("dp_reg") or 1
    return x - (x % word_len)
