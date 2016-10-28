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


class Extractor(object):

    EXT = 0  # expression marker: extract
    STOP = 1  # expression marker: do not extract

    @staticmethod
    def factory(mode, stmt, expr_info):
        if mode == 'normal':
            should_extract = lambda d: d != set(expr_info.dims)
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
            elif dep_l | dep_r == set(self.expr_info.dims):
                # E.g. A[i]*B[j] within two loops i and j
                self._try(left, dep_l)
                self._try(right, dep_r)
                return (dep_n, self.STOP)
            else:
                # E.g. A[i]*B[j] within at least three loops
                return (dep_n, self.EXT)
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

    # Temporary variables template
    _template = "ct%d"

    def __init__(self, stmt, expr_info, header, decls, hoisted):
        """Initialize the Hoister."""
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.decls = decls
        self.hoisted = hoisted

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
        # TODO apply `in_written` to all loops in mapper ONCE, in `licm`,
        #      and then update it as exprs are hoisted
        outer = self.expr_info.outermost_loop
        inner = self.expr_info.innermost_loop

        # Start assuming no "real" hoisting can take place
        # E.g.: for i {a[i]*(t1 + t2);} --> for i {t3 = t1 + t2; a[i]*t3;}
        place, offset = inner.block, self.stmt

        if mode != 'aggressive':
            # "Standard" code motion case, i.e. moving /subexprs/ as far as
            # possible in the loop nest such that dependencies are honored
            loops = list(reversed(self.expr_info.loops))
            candidates = [l.block for l in loops[1:]] + [self.header]

            for loop, candidate in zip(loops, candidates):
                if not self._is_hoistable(subexprs, loop):
                    break
                if loop.dim not in dep:
                    place, offset = candidate, loop

            # Determine how much extra memory and whether clone loops are needed
            jumped = loops[:candidates.index(place)]
            clone = tuple(l for l in reversed(jumped) if l.dim in dep)
        else:
            # Hoist outside of the loop nest, even though this doesn't
            # result in any operation count reduction
            if all(self._is_hoistable(subexpr, outer)):
                place, offset = self.header, outer
                clone = tuple(self.expr_info.loops)

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

                # 2) Determine the outermost loop where invariant expressions
                # can be hoisted without breaking data dependencies.
                place, offset, clone = self._locate(dep, subexprs, mode)

                loop_size = tuple(l.size for l in clone)
                loop_dim = tuple(l.dim for l in clone)

                # 3) Create the required new AST nodes
                symbols, decls, stmts = [], [], []
                for e in subexprs:
                    already_hoisted = False
                    if global_cse and self.hoisted.get_symbol(e):
                        name = self.hoisted.get_symbol(e)
                        decl = self.hoisted[name].decl
                        if decl in place.children and \
                                place.children.index(decl) < place.children.index(offset):
                            already_hoisted = True
                    if not already_hoisted:
                        name = self._template % (len(self.hoisted) + len(stmts))
                        stmts.append(Assign(Symbol(name, loop_dim), dcopy(e)))
                        decl = Decl(self.expr_info.type, Symbol(name, loop_size),
                                    scope=LOCAL)
                        decls.append(decl)
                        self.decls[name] = decl
                    symbols.append(Symbol(name, loop_dim))

                # 4) Replace invariant sub-expressions with temporaries
                ast_replace(self.stmt.rvalue, dict(zip(subexprs, symbols)))

                # 5) Update data dependencies
                lda.update({s: set(dep) for s in symbols})

                # 6) Modify the AST adding the hoisted expressions
                if clone:
                    outer_clone = ast_make_for(stmts, clone[-1])
                    for l in reversed(clone[:-1]):
                        outer_clone = ast_make_for([outer_clone], l)
                    code = decls + [outer_clone]
                    clone = outer_clone
                else:
                    code = decls + stmts
                    clone = None
                # Insert the new nodes at the right level in the loop nest
                offset = place.children.index(offset)
                place.children[offset:offset] = code
                # Track hoisted symbols
                for i, j in zip(stmts, decls):
                    self.hoisted[j.sym.symbol] = (i, j, clone, place)

            if not iterative:
                break

        # Finally, make sure symbols are unique in the AST
        self.stmt.rvalue = dcopy(self.stmt.rvalue)
