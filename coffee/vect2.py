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

from __future__ import absolute_import, print_function, division
from six.moves import range

from math import ceil
from itertools import product

from coffee.utils import *
from coffee import system
from coffee.logger import warn
from coffee.visitors import Find


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
        self.exprs = loop_opt.exprs
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


        Finally, all arrays are decorated with suitable attributes to enforce
        alignment to (the size in bytes of) the vector length.

        :arg p_dim: the array dimension that should be padded (defaults to the
            innermost, or -1)
        """
        info = visit(self.kernel, info_items=['decls', 'fors', 'symbols_dep',
                                              'symbols_mode', 'symbol_refs'])

        self._pad(p_dim, info['decls'], info['fors'], info['symbols_dep'],
                  info['symbols_mode'], info['symbol_refs'])
        self._align_data(p_dim, info['decls'])

    def _pad(self, p_dim, decls, fors, symbols_dep, symbols_mode, symbol_refs):
        """Apply padding."""
        to_invert = Find(Invert).visit(self.header)[Invert]

        # Loop increments different than 1 are unsupported
        if any([l.increment != 1 for l, _ in flatten(fors)]):
            return None

        DSpace = namedtuple('DSpace', ['region', 'nest', 'symbols'])
        ISpace = namedtuple('ISpace', ['region', 'nest', 'bag'])

        for decl_name, decl in decls.items():
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
            deps = OrderedDict((s, [l for l in symbols_dep[s] if l.dim in s.rank])
                               for s in symbols)
            if not all(s.dim == len(n) for s, n in deps.items()):
                continue

            # ... organize symbols based on their dataspace
            dspace_mapper = OrderedDict()
            for s, n in deps.items():
                n.sort(key=lambda l: s.rank.index(l.dim))
                region = tuple(Region(l.size, l.start + i) for i, l in zip(s.strides, n))
                dspace = DSpace(region=region, nest=n, symbols=[])
                dspace_mapper.setdefault(dspace.region, dspace)
                dspace.symbols.append(s)

            # ... is there any overlap in the memory accesses? Memory accesses must:
            # - either completely overlap (they will be mapped to the same buffer)
            # - OR be disjoint
            will_break = False
            for regions1, regions2 in product(dspace_mapper.keys(), dspace_mapper.keys()):
                for r1, r2 in zip(regions1, regions2):
                    if ItSpace(mode=1).intersect([r1, r2]) not in [(0, 0), r1]:
                        will_break = True
            if will_break:
                continue

            # ... initialize buffer-related data
            buf_name = '_' + decl_name
            buf_nz = self.nz_syms.setdefault(buf_name, [])

            # ... determine the non zero-valued region in the buffer
            for n, region in enumerate(dspace_mapper.keys()):
                p_region = (Region(region[p_dim].size, 0),)
                buf_nz.append((Region(1, n),) + region[:p_dim] + p_region)

            # ... replace symbols in the AST with proper buffer instances
            itspace_mapper = OrderedDict()
            for n, dspace in enumerate(dspace_mapper.values()):
                itspace = ISpace(region=tuple((l.size, l.start) for l in dspace.nest),
                                 nest=dspace.nest, bag=OrderedDict())
                itspace = itspace_mapper.setdefault(itspace.region, itspace)
                for s in dspace.symbols:
                    original = Symbol(s.symbol, s.rank, s.offset)
                    s.symbol = buf_name
                    s.rank = (n,) + s.rank
                    s.offset = ((1, 0),) + s.offset[:p_dim] + ((1, 0),)
                    if s.urepr not in [i.urepr for i in itspace.bag.values()]:
                        itspace.bag[original] = Symbol(s.symbol, s.rank, s.offset)

            # ... insert the buffer into the AST
            buf_dim = n + 1
            buf_rank = (buf_dim,) + decl.size
            init = ArrayInit(np.ndarray(shape=(1,)*len(buf_rank), buffer=np.array(0.0)))
            buf_decl = Decl(decl.typ, Symbol(buf_name, buf_rank), init, scope=BUFFER)
            buf_decl.pad((buf_dim,) + p_rank)
            self.header.children.insert(0, buf_decl)

            # C) Create a loop nest for copying data into/from the buffer
            for itspace in itspace_mapper.values():

                if mode == READ:
                    stmts = [Assign(b, s) for s, b in itspace.bag.items()]
                    copy_back = ItSpace(mode=2).to_for(itspace.nest, stmts=stmts)
                    insert_at_elem(self.header.children, buf_decl, copy_back[0], ofs=1)

                elif mode == WRITE:
                    # If extra information (a pragma) is present, telling that
                    # the argument does not need to be incremented because it does
                    # not contain any meaningful values, then we can safely write
                    # to it. This optimization may avoid useless increments
                    can_write = WRITE in decl.pragma and len(itspace_mapper) == 1
                    op = Assign if can_write else Incr
                    stmts = [op(s, b) for s, b in itspace.bag.items()]
                    copy_back = ItSpace(mode=2).to_for(itspace.nest, stmts=stmts)
                    if to_invert:
                        insert_at_elem(self.header.children, to_invert[0], copy_back[0])
                    else:
                        self.header.children.append(copy_back[0])

            # D) Update the global data structures
            decls[buf_name] = buf_decl

    def _align_data(self, p_dim, decls):
        """Apply data alignment. This boils down to:

            * Decorate declarations with qualifiers for data alignment
            * Round up the bounds (i.e. /start/ and /end/ points) of loops such
            that all memory accesses get aligned to the vector length. Several
            checks ensure the correctness of the transformation.
        """
        vector_length = system.isa["dp_reg"]
        align = system.compiler['align'](system.isa['alignment'])

        # Array alignment
        for decl in decls.values():
            if decl.sym.rank and decl.scope == LOCAL:
                decl.attr.append(align)

        # Loop bounds adjustment
        for l in inner_loops(self.header):
            should_round = True

            for stmt in l.body:
                sym, expr = stmt.lvalue, stmt.rvalue
                decl = decls[sym.symbol]

                # Condition A: the lvalue can be a scalar only if /stmt/ is not an
                # augmented assignment, otherwise the extra iterations would alter
                # its value
                if not sym.rank and isinstance(stmt, AugmentedAssign):
                    should_round = False
                    break

                # Condition B: the fastest varying dimension of the lvalue must be /l/
                if sym.rank and not sym.rank[p_dim] == l.dim:
                    should_round = False
                    break

                # Condition C: the lvalue must have been padded
                if sym.rank and decl.size[p_dim] != vect_roundup(decl.size[p_dim]):
                    should_round = False
                    break

                symbols = [sym] + Find(Symbol).visit(expr)[Symbol]
                symbols = [s for s in symbols if s.rank and any(r == l.dim for r in s.rank)]

                # Condition D: the access pattern must be accessible
                if any(not s.is_unit_period for s in symbols):
                    # Cannot infer the access pattern so must break
                    should_round = False
                    break

                # Condition E: extra iterations induced by bounds and offset rounding
                # must not alter the computation
                for s in symbols:
                    decl = decls[s.symbol]
                    index = s.rank.index(l.dim)
                    stride = s.strides[index]
                    extra = list(range(stride + l.size, stride + vect_roundup(l.size)))
                    # Do any of the extra iterations alter the computation ?
                    if any(i > decl.size[index] for i in extra):
                        # ... outside of the legal region, abort
                        should_round = False
                        break
                    if all(i >= decl.core[index] for i in extra):
                        # ... in the padded region, pass
                        continue
                    nz = list(self.nz_syms.get(s.symbol))
                    if not nz:
                        # ... lacks the non zero-valued entries mapping, abort
                        should_round = False
                        break
                    # ... get the non zero-valued entries in the right dimension
                    nz_index = []
                    for i in nz:
                        can_skip = False
                        for j, r in enumerate(s.rank[:index]):
                            if not is_const_dim(r):
                                continue
                            if not (i[j].ofs <= r < i[j].ofs + i[j].size):
                                # ... actually on a different outer dimension, safe
                                # to avoid this check
                                can_skip = True
                        if not can_skip:
                            nz_index.append(i[index])
                    if any(ofs <= i < ofs + size for size, ofs in nz_index):
                        # ... writing to a non-zero region, abort
                        should_round = False
                        break

            if should_round:
                l.end = vect_roundup(l.end)
                if all(i % vector_length == 0 for i in [l.start, l.size]):
                    l.pragma.add(system.compiler["align_forloop"])
                    l.pragma.add(system.compiler['force_simdization'])



# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = system.isa.get("dp_reg") or 1
    return int(ceil(x / word_len)) * word_len


def vect_rounddown(x):
    """Return x rounded down to the vector length. """
    word_len = system.isa.get("dp_reg") or 1
    return x - (x % word_len)
