# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2016, Imperial College London.
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

from .base import *
from .utils import *
from .logger import warn


class Extractor(object):

    EXT = 0  # expression marker: extract
    STOP = 1  # expression marker: do not extract

    @staticmethod
    def factory(mode, stmt, expr_info):
        if mode == 'normal':
            should_extract = lambda d: True
            return MainExtractor(stmt, expr_info, should_extract)
        elif mode == 'only_const':
            # Do not extract unless constant in all loops
            should_extract = lambda d: not (d and d.issubset(set(expr_info.dims)))
            return MainExtractor(stmt, expr_info, should_extract)
        elif mode == 'only_outlinear':
            should_extract = lambda d: d.issubset(set(expr_info.out_linear_dims))
            return MainExtractor(stmt, expr_info, should_extract)
        elif mode == 'only_linear':
            should_extract = lambda d: not (d.issubset(set(expr_info.out_linear_dims)))
            return SoftExtractor(stmt, expr_info, should_extract)
        elif mode == 'aggressive':
            should_extract = lambda d: True
            return AggressiveExtractor(stmt, expr_info, should_extract)
        else:
            raise RuntimeError("Requested an invalid Extractor (%s)" % mode)

    def __init__(self, stmt, expr_info, should_extract):
        self.stmt = stmt
        self.expr_info = expr_info
        self.should_extract = should_extract

    def _handle_expr(*args):
        raise NotImplementedError("Extractor is an abstract class")

    def _apply_cse(self):
        # Find common sub-expressions heuristically looking at binary terminal
        # operations (i.e., a terminal has two Symbols as children). This may
        # induce more sweeps of extraction to find all common sub-expressions,
        # but at least it keeps the algorithm simple and probably more effective
        finder = FindInstances(Symbol, with_parent=True)
        for dep, subexprs in self.extracted.items():
            cs = OrderedDict()
            retval = FindInstances.default_retval()
            values = [finder.visit(e, retval=retval)[Symbol] for e in subexprs]
            binexprs = zip(*flatten(values))[1]
            binexprs = [b for b in binexprs if binexprs.count(b) > 1]
            for b in binexprs:
                t = cs.setdefault(b.urepr, [])
                if b not in t:
                    t.append(b)
            cs = [v for k, v in cs.items() if len(v) > 1]
            if cs:
                self.extracted[dep] = list(flatten(cs))

    def _try(self, node, dep):
        if isinstance(node, Symbol):
            return False
        should_extract = self.should_extract(dep)
        if should_extract or self._look_ahead:
            dep = sorted(dep, key=lambda i: self.expr_info.dims.index(i))
            self.extracted.setdefault(tuple(dep), []).append(node)
        return should_extract

    def _visit(self, node):
        if isinstance(node, Symbol):
            return (self._lda[node], self.EXT)

        elif isinstance(node, (FunCall, Ternary)):
            arg_deps = [self._visit(n) for n in node.children]
            dep = tuple(set(flatten([dep for dep, _ in arg_deps])))
            info = self.EXT if all(i == self.EXT for _, i in arg_deps) else self.STOP
            return (dep, info)

        else:
            left, right = node.children
            dep_l, info_l = self._visit(left)
            dep_r, info_r = self._visit(right)

            dep_l = {d for d in dep_l if d in self.expr_info.dims}
            dep_r = {d for d in dep_r if d in self.expr_info.dims}
            dep_n = dep_l | dep_r

            return self._handle_expr(left, right, dep_l, dep_r, dep_n, info_l, info_r)

    def extract(self, look_ahead, lda, with_cse=False):
        """Extract invariant subexpressions from /self.expr/."""
        self._lda = lda
        self._look_ahead = look_ahead
        self.extracted = OrderedDict()

        self._visit(self.stmt.rvalue)
        if with_cse:
            self._apply_cse()

        del self._lda
        del self._look_ahead

        return self.extracted


class MainExtractor(Extractor):

    def _handle_expr(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
        if info_l == self.EXT and info_r == self.EXT:
            if dep_l == dep_r:
                # E.g. alpha*beta, A[i] + B[i]
                return (dep_l, self.EXT)
            elif not dep_l:
                # E.g. alpha*A[i,j]
                self._try(left, dep_l)
                if not (set(self.expr_info.linear_dims) & dep_r and self._try(left, dep_l)):
                    return (dep_r, self.EXT)
            elif not dep_r:
                # E.g. A[i,j]*alpha
                self._try(right, dep_r)
                if not (set(self.expr_info.linear_dims) & dep_l and self._try(left, dep_l)):
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


class SoftExtractor(Extractor):

    def _handle_expr(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
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


class AggressiveExtractor(Extractor):

    def _handle_expr(self, left, right, dep_l, dep_r, dep_n, info_l, info_r):
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


class Hoister(object):

    # How many times the hoister was invoked
    _handled = 0
    # Temporary variables template
    _hoisted_sym = "%(loop_dep)s_%(expr_id)d_%(round)d_%(i)d"

    def __init__(self, stmt, expr_info, header, decls, hoisted):
        """Initialize the Hoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.decls = decls
        self.hoisted = hoisted

        # Increment counters for unique variable names
        self.nextracted = 0
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

    def _is_hoistable(self, subexprs, loop):
        """Return True if the sub-expressions provided in ``subexprs`` are
        hoistable outside of ``loop``, False otherwise."""
        written = in_written(loop, 'symbol')
        finder, reads = FindInstances(Symbol), FindInstances.default_retval()
        for e in subexprs:
            finder.visit(e, ret=reads)
        reads = [s.symbol for s in reads[Symbol]]
        return set.isdisjoint(set(reads), set(written))

    def _locate(self, dep, subexprs, mode):
        # TODO add check that exprs can only live in innermost loops
        # TODO add lookup method to Block ?
        # TODO apply `in_written` to all loops in mapper ONCE, in `licm`,
        #      and then update it as exprs are hoisted
        # TODO replace od_find_next with a `next_loop` method in expr ?
        outer = self.expr_info.outermost_loop
        inner = self.expr_info.innermost_loop

        loops = list(reversed(self.expr_info.loops))
        candidates = [l.block for l in loops[1:]] + [self.header]

        # Start assuming no real hoisting can take place -- subexprs only "moved"
        # right before the main expression
        place, offset = inner.block, self.stmt

        # Then, determine how far in the loop nest can /subexprs/ be computed
        for loop, candidate in zip(loops, candidates):
            if not self._is_hoistable(subexprs, loop):
                break
            if loop.dim not in dep:
                place, offset = candidate, loop

        # Finally, determine how much extra memory and clone loops are needed
        jumped = loops[:candidates.index(place)]
        clone = tuple(l for l in reversed(jumped) if l.dim in dep)

        return place, offset, clone

    def extract(self, mode, **kwargs):
        """Return a dictionary of hoistable subexpressions."""
        lda = kwargs.get('lda') or loops_analysis(self.header, value='dim')
        extractor = Extractor.factory(mode, self.stmt, self.expr_info)
        return extractor.extract(True, lda)

    def licm(self, mode, **kwargs):
        """Perform generalized loop-invariant code motion."""
        max_sharing = kwargs.get('max_sharing', False)
        iterative = kwargs.get('iterative', True)
        lda = kwargs.get('lda') or loops_analysis(self.header, value='dim')
        global_cse = kwargs.get('global_cse', False)

        expr_dims_loops = self.expr_info.loops_from_dims
        expr_outermost_loop = self.expr_info.outermost_loop
        expr_outermost_linear_loop = self.expr_info.outermost_linear_loop
        is_bilinear = self.expr_info.is_bilinear

        extractor = Extractor.factory(mode, self.stmt, self.expr_info)
        extracted = True

        while extracted:
            extracted = extractor.extract(False, lda, global_cse)
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
                # We distinguish several cases:
                depth = len(dep)
                if depth == 0:
                    # As scalar, outside of the loop nest;
                    place = self.header
                    wrap_loop = ()
                    offset = expr_outermost_loop
                elif depth == 1 and len(expr_dims_loops) == 1:
                    # As scalar, within the only loop present
                    place = expr_outermost_loop.children[0]
                    wrap_loop = ()
                    offset = place.children[place.children.index(self.stmt)]
                elif depth == 1 and len(expr_dims_loops) > 1:
                    if expr_dims_loops[dep[0]] == expr_outermost_loop:
                        # As scalar, within the outermost loop
                        place = expr_outermost_loop.children[0]
                        wrap_loop = ()
                        offset = od_find_next(expr_dims_loops, dep[0])
                    else:
                        # As vector, outside of the loop nest;
                        place = self.header
                        wrap_loop = (expr_dims_loops[dep[0]],)
                        offset = expr_outermost_loop
                elif mode == 'aggressive' and set(dep) == set(self.expr_info.dims) and \
                        not any([self.expr_graph.is_written(e) for e in subexprs]):
                    # As n-dimensional vector (n == depth), outside of the loop nest
                    place = self.header
                    wrap_loop = tuple(expr_dims_loops.values())
                    offset = expr_outermost_loop
                elif depth == 2:
                    if self._is_hoistable(subexprs, expr_outermost_linear_loop):
                        # As vector, within the outermost loop imposing the dependency
                        place = expr_dims_loops[dep[0]].children[0]
                        wrap_loop = tuple(expr_dims_loops[dep[i]] for i in range(1, depth))
                        offset = od_find_next(expr_dims_loops, dep[0])
                    elif expr_outermost_linear_loop.dim == dep[-1] and is_bilinear:
                        # As scalar, within the closest loop imposing the dependency
                        place = expr_dims_loops[dep[-1]].children[0]
                        wrap_loop = ()
                        offset = od_find_next(expr_dims_loops, dep[-1])
                    else:
                        # As scalar, within the closest loop imposing the dependency
                        place = expr_dims_loops[dep[-1]].children[0]
                        wrap_loop = ()
                        offset = place.children[place.children.index(self.stmt)]
                else:
                    warn("Skipping unexpected code motion case.")
                    return

                loop_size = tuple([l.size for l in wrap_loop])
                loop_dim = tuple([l.dim for l in wrap_loop])

                # 3) Create the required new AST nodes
                symbols, decls, stmts = [], [], []
                for i, e in enumerate(subexprs):
                    already_hoisted = False
                    if global_cse and self.hoisted.get_symbol(e):
                        name = self.hoisted.get_symbol(e)
                        decl = self.hoisted[name].decl
                        if decl in place.children and \
                                place.children.index(decl) < place.children.index(offset):
                            already_hoisted = True
                    if not already_hoisted:
                        name = self._hoisted_sym % {
                            'loop_dep': '_'.join(dep) if dep else 'c',
                            'expr_id': self.expr_id,
                            'round': self.nextracted,
                            'i': i
                        }
                        stmts.append(Assign(Symbol(name, loop_dim), dcopy(e)))
                        decl = Decl(self.expr_info.type, Symbol(name, loop_size),
                                    scope=LOCAL)
                        decls.append(decl)
                        self.decls[name] = decl
                    symbols.append(Symbol(name, loop_dim))

                # 4) Replace invariant sub-expressions with temporaries
                to_replace = dict(zip(subexprs, symbols))
                n_replaced = ast_replace(self.stmt.rvalue, to_replace)

                # 5) Update data dependencies
                for s, e in zip(symbols, subexprs):
                    lda[s] = dep

                # 6) Modify the AST adding the hoisted expressions
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
                offset = place.children.index(offset)
                place.children[offset:offset] = code
                # Track hoisted symbols
                for i, j in zip(stmts, decls):
                    self.hoisted[j.sym.symbol] = (i, j, wrap_loop, place)

            self.nextracted += 1
            if not iterative:
                break

        # Finally, make sure symbols are unique in the AST
        self.stmt.rvalue = dcopy(self.stmt.rvalue)
