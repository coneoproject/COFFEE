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

from base import *
from utils import *


class Extractor():

    EXT = 0  # expression marker: extract
    STOP = 1  # expression marker: do not extract

    def __init__(self, stmt, expr_info):
        self.stmt = stmt
        self.expr_info = expr_info
        self.counter = 0

    def _try(self, node, dep):
        if isinstance(node, Symbol):
            # Never extract individual symbols
            return False
        should_extract = True
        if self.mode == 'aggressive':
            # Do extract everything
            should_extract = True
        elif self.mode == 'only_const':
            # Do not extract unless constant in all loops
            if dep and dep.issubset(set(self.expr_info.dims)):
                should_extract = False
        elif self.mode == 'only_domain':
            # Do not extract unless dependent on domain loops
            if dep.issubset(set(self.expr_info.out_domain_dims)):
                should_extract = False
        elif self.mode == 'only_outdomain':
            # Do not extract unless independent of the domain loops
            if not dep.issubset(set(self.expr_info.out_domain_dims)):
                should_extract = False
        if should_extract or self.look_ahead:
            dep = sorted(dep, key=lambda i: self.expr_info.dims.index(i))
            self.extracted.setdefault(tuple(dep), []).append(node)
        return should_extract

    def _soft(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
        if info_l == self.EXT and info_r == self.EXT:
            if dep_l == dep_r:
                # E.g. alpha*beta, A[i] + B[i]
                return (dep_l, self.EXT)
            elif dep_l.issubset(dep_r):
                # E.g. A[i]*B[i,j]
                if not self._try(right, dep_r):
                    return (dep_n, self.EXT)
            elif dep_r.issubset(dep_l):
                # E.g. A[i,j]*B[i]
                if not self._try(left, dep_l):
                    return (dep_n, self.EXT)
            else:
                # E.g. A[i]*B[j]
                self._try(left, dep_l)
                self._try(right, dep_r)
        elif info_r == self.EXT:
            self._try(right, dep_r)
        elif info_l == self.EXT:
            self._try(left, dep_l)
        return (dep_n, self.STOP)

    def _normal(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
        if info_l == self.EXT and info_r == self.EXT:
            if dep_l == dep_r:
                # E.g. alpha*beta, A[i] + B[i]
                return (dep_l, self.EXT)
            elif not dep_l:
                # E.g. alpha*A[i,j]
                if not set(self.expr_info.domain_dims) & dep_r or \
                        not (self._try(left, dep_l) or self._try(right, dep_r)):
                    return (dep_r, self.EXT)
            elif not dep_r:
                # E.g. A[i,j]*alpha
                if not set(self.expr_info.domain_dims) & dep_l or \
                        not (self._try(right, dep_r) or self._try(left, dep_l)):
                    return (dep_l, self.EXT)
            elif dep_l.issubset(dep_r):
                # E.g. A[i]*B[i,j]
                if not self._try(left, dep_l):
                    return (dep_n, self.EXT)
            elif dep_r.issubset(dep_l):
                # E.g. A[i,j]*B[i]
                if not self._try(right, dep_r):
                    return (dep_n, self.EXT)
            else:
                # E.g. A[i]*B[j]
                self._try(left, dep_l)
                self._try(right, dep_r)
        elif info_r == self.EXT:
            self._try(right, dep_r)
        elif info_l == self.EXT:
            self._try(left, dep_l)
        return (dep_n, self.STOP)

    def _aggressive(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
        if info_l == self.EXT and info_r == self.EXT:
            if dep_l == dep_r:
                # E.g. alpha*beta, A[i] + B[i]
                return (dep_l, self.EXT)
            elif not dep_l:
                # E.g. alpha*A[i,j], not hoistable anymore
                self._try(right, dep_r)
            elif not dep_r:
                # E.g. A[i,j]*alpha, not hoistable anymore
                self._try(left, dep_l)
            elif dep_l.issubset(dep_r):
                # E.g. A[i]*B[i,j]
                if not self._try(left, dep_l):
                    return (dep_n, self.EXT)
            elif dep_r.issubset(dep_l):
                # E.g. A[i,j]*B[i]
                if not self._try(right, dep_r):
                    return (dep_n, self.EXT)
            else:
                # E.g. A[i]*B[j], hoistable in TMP[i,j]
                return (dep_n, self.EXT)
        elif info_r == self.EXT:
            self._try(right, dep_r)
        elif info_l == self.EXT:
            self._try(left, dep_l)
        return (dep_n, self.STOP)

    def _extract(self, node):
        if isinstance(node, Symbol):
            return (self.lda[node], self.EXT)

        elif isinstance(node, Par):
            return self._extract(node.child)

        elif isinstance(node, (FunCall, Ternary)):
            arg_deps = [self._extract(n) for n in node.children]
            dep = tuple(set(flatten([dep for dep, _ in arg_deps])))
            info = self.EXT if all(i == self.EXT for _, i in arg_deps) else self.STOP
            return (dep, info)

        else:
            # Traverse the expression tree
            left, right = node.children
            dep_l, info_l = self._extract(left)
            dep_r, info_r = self._extract(right)

            # Filter out false dependencies
            dep_l = {d for d in dep_l if d in self.expr_info.dims}
            dep_r = {d for d in dep_r if d in self.expr_info.dims}
            dep_n = dep_l | dep_r

            args = left, right, dep_l, dep_r, dep_n, info_l, info_r

            if self.mode in ['normal', 'only_const', 'only_outdomain']:
                return self._normal(*args)
            elif self.mode == 'only_domain':
                return self._soft(*args)
            elif self.mode == 'aggressive':
                return self._aggressive(*args)
            else:
                raise RuntimeError("licm: unexpected hoisting mode (%s)" % self.mode)

    def __call__(self, mode, look_ahead, lda):
        """Extract invariant subexpressions from /self.expr/."""

        self.mode = mode
        self.look_ahead = look_ahead
        self.lda = lda
        self.extracted = OrderedDict()
        self._extract(self.stmt.rvalue)

        self.counter += 1
        return self.extracted


class Hoister():

    # How many times the hoister was invoked
    _handled = 0
    # Temporary variables template
    _hoisted_sym = "%(loop_dep)s_%(expr_id)d_%(round)d_%(i)d"

    def __init__(self, stmt, expr_info, header, decls, hoisted, expr_graph):
        """Initialize the Hoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.decls = decls
        self.hoisted = hoisted
        self.expr_graph = expr_graph
        self.extractor = Extractor(self.stmt, self.expr_info)

        # Increment counters for unique variable names
        self.expr_id = Hoister._handled
        Hoister._handled += 1

    def _filter(self, dep, subexprs, make_unique=True, sharing=None):
        """Filter hoistable subexpressions."""
        if make_unique:
            # Uniquify expressions
            subexprs = uniquify(subexprs)

        if sharing:
            # Partition expressions such that expressions sharing the same
            # set of symbols are in the same partition
            if dep == self.expr_info.dims:
                return []
            sharing = [str(s) for s in sharing]
            finder = FindInstances(Symbol)
            partitions = defaultdict(list)
            for e in subexprs:
                retval = FindInstances.default_retval()
                symbols = tuple(set(str(s) for s in finder.visit(e, ret=retval)[Symbol]
                                    if str(s) in sharing))
                partitions[symbols].append(e)
            for shared, partition in partitions.items():
                if len(partition) > len(shared):
                    subexprs = [e for e in subexprs if e not in partition]

        return subexprs

    def extract(self, mode, **kwargs):
        """Return a dictionary of hoistable subexpressions."""
        lda = kwargs.get('lda') or ldanalysis(self.header, value='dim')
        return self.extractor(mode, True, lda)

    def licm(self, mode, **kwargs):
        """Perform generalized loop-invariant code motion."""
        max_sharing = kwargs.get('max_sharing', False)
        iterative = kwargs.get('iterative', True)
        lda = kwargs.get('lda') or ldanalysis(self.header, value='dim')
        global_cse = kwargs.get('global_cse', False)

        expr_dims_loops = self.expr_info.loops_from_dims
        expr_outermost_loop = self.expr_info.outermost_loop

        mapper = {}
        extracted = True
        while extracted:
            extracted = self.extractor(mode, False, lda)
            for dep, subexprs in extracted.items():
                # 1) Filter subexpressions that will be hoisted
                sharing = []
                if max_sharing:
                    sharing = uniquify([s for s, d in lda.items() if d == dep])
                subexprs = self._filter(dep, subexprs, sharing=sharing)
                if not subexprs:
                    continue

                # 2) Determine the loop nest level where invariant expressions
                # should be hoisted. The goal is to hoist them as far as possible
                # in the loop nest, while minimising temporary storage.
                # We distinguish six hoisting cases:
                if len(dep) == 0:
                    # As scalar (/wrap_loop=None/), outside of the loop nest;
                    place = self.header
                    wrap_loop = ()
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and is_perfect_loop(expr_outermost_loop):
                    # As scalar, outside of the loop nest;
                    place = self.header
                    wrap_loop = (expr_dims_loops[dep[0]],)
                    next_loop = expr_outermost_loop
                elif len(dep) == 1 and len(expr_dims_loops) > 1:
                    # As scalar, within the loop imposing the dependency
                    place = expr_dims_loops[dep[0]].children[0]
                    wrap_loop = ()
                    next_loop = od_find_next(expr_dims_loops, dep[0])
                elif len(dep) == 1:
                    # As scalar, right before the expression (which is enclosed
                    # in just a single loop, we can claim at this point)
                    place = expr_dims_loops[dep[0]].children[0]
                    wrap_loop = ()
                    next_loop = place.children[place.children.index(self.stmt)]
                elif mode == 'aggressive' and set(dep) == set(self.expr_info.dims) and \
                        not any([self.expr_graph.is_written(e) for e in subexprs]):
                    # As n-dimensional vector, where /n == len(dep)/, outside of
                    # the loop nest
                    place = self.header
                    wrap_loop = tuple(expr_dims_loops.values())
                    next_loop = expr_outermost_loop
                elif not is_perfect_loop(expr_dims_loops[dep[-1]]):
                    # As scalar, within the closest loop imporsing the dependency
                    place = expr_dims_loops[dep[-1]].children[0]
                    wrap_loop = ()
                    next_loop = od_find_next(expr_dims_loops, dep[-1])
                else:
                    # As vector, within the outermost loop imposing the dependency
                    place = expr_dims_loops[dep[0]].children[0]
                    wrap_loop = tuple(expr_dims_loops[dep[i]] for i in range(1, len(dep)))
                    next_loop = od_find_next(expr_dims_loops, dep[0])

                loop_size = tuple([l.size for l in wrap_loop])
                loop_dim = tuple([l.dim for l in wrap_loop])

                # 3) Create the required new AST nodes
                symbols, decls, stmts = [], [], []
                for i, e in enumerate(subexprs):
                    if global_cse and self.hoisted.get_symbol(e):
                        name = self.hoisted.get_symbol(e)
                    else:
                        name = self._hoisted_sym % {
                            'loop_dep': '_'.join(dep) if dep else 'c',
                            'expr_id': self.expr_id,
                            'round': self.extractor.counter,
                            'i': i
                        }
                        stmts.append(Assign(Symbol(name, loop_dim), dcopy(e)))
                        decl = Decl(self.expr_info.type, Symbol(name, loop_size))
                        decl.scope = LOCAL
                        decls.append(decl)
                        self.decls[name] = decl
                    symbols.append(Symbol(name, loop_dim))

                # 4) Replace invariant sub-expressions with temporaries
                to_replace = dict(zip(subexprs, symbols))
                n_replaced = ast_replace(self.stmt.rvalue, to_replace)

                # 5) Update data dependencies
                for s, e in zip(symbols, subexprs):
                    self.expr_graph.add_dependency(s, e)
                    if n_replaced[str(s)] > 1:
                        self.expr_graph.add_dependency(s, s)
                    lda[s] = dep

                # 6) Track necessary information for AST construction
                info = (loop_dim, place, next_loop, wrap_loop)
                if info not in mapper:
                    mapper[info] = (decls, stmts)
                else:
                    mapper[info][0].extend(decls)
                    mapper[info][1].extend(stmts)

            if not iterative:
                break

        for info, (decls, stmts) in sorted(mapper.items()):
            loop_dim, place, next_loop, wrap_loop = info
            # Create the hoisted code
            if wrap_loop:
                outer_wrap_loop = ast_make_for(stmts, wrap_loop[-1])
                for l in reversed(wrap_loop[:-1]):
                    outer_wrap_loop = ast_make_for([outer_wrap_loop], l)
                code = decls + [outer_wrap_loop]
                wrap_loop = outer_wrap_loop
            else:
                code = decls + stmts
                wrap_loop = None
            # Insert the new nodes at the right level in the loop nest
            ofs = place.children.index(next_loop)
            place.children[ofs:ofs] = code + [FlatBlock("\n")]
            # Track hoisted symbols
            for i, j in zip(stmts, decls):
                self.hoisted[j.sym.symbol] = (i, j, wrap_loop, place)

        # Finally, make sure symbols are unique in the AST
        self.stmt.rvalue = dcopy(self.stmt.rvalue)
