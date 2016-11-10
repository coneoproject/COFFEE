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

    def __init__(self, stmt, expr_info, should_extract):
        self.stmt = stmt
        self.expr_info = expr_info
        self.should_extract = should_extract

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
            retval = [(n,) + self._visit(n) for n in node.children]
            dep = set.union(*[d for _, d, _ in retval])
            dep = {d for d in dep if d in self.expr_info.dims}
            if self.should_extract(dep) or self._look_ahead:
                # Still a chance of finding a bigger expression
                return (dep, self.EXT)
            else:
                for n, n_dep, n_info in retval:
                    if n_info == self.EXT and not isinstance(n, Symbol):
                        k = sorted(n_dep, key=lambda i: self.expr_info.dims.index(i))
                        self.extracted.setdefault(tuple(k), []).append(n)
                return (dep, self.STOP)

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

    def _locate(self, dep, subexprs, with_promotion=False):
        # Start assuming no "real" hoisting can take place
        # E.g.: for i {a[i]*(t1 + t2);} --> for i {t3 = t1 + t2; a[i]*t3;}
        place, offset = self.expr_info.innermost_loop.block, self.stmt

        if with_promotion:
            # Hoist outside a loop even though this doesn't result in any
            # operation count reduction
            should_jump = lambda l: True
        else:
            # "Standard" code motion case, i.e. moving /subexprs/ as far as
            # possible in the loop nest such that dependencies are honored
            should_jump = lambda l: l.dim not in dep

        loops = list(reversed(self.expr_info.loops))
        candidates = [l.block for l in loops[1:]] + [self.header]

        for loop, candidate in zip(loops, candidates):
            if not self._is_hoistable(subexprs, loop):
                break
            if should_jump(loop):
                place, offset = candidate, loop

        # Determine how much extra memory and whether clone loops are needed
        jumped = loops[:candidates.index(place) + 1]
        clone = tuple(l for l in reversed(jumped) if l.dim in dep)

        return place, offset, clone

    def extract(self, should_extract, **kwargs):
        """Return a dictionary of hoistable subexpressions."""
        lda = kwargs.get('lda', loops_analysis(self.header, value='dim'))
        extractor = Extractor(self.stmt, self.expr_info, should_extract)
        return extractor.extract(True, lda)

    def licm(self, should_extract, **kwargs):
        """Perform generalized loop-invariant code motion."""
        max_sharing = kwargs.get('max_sharing', False)
        with_promotion = kwargs.get('with_promotion', False)
        iterative = kwargs.get('iterative', True)
        lda = kwargs.get('lda', loops_analysis(self.header, value='dim'))
        global_cse = kwargs.get('global_cse', False)

        extractor = Extractor(self.stmt, self.expr_info, should_extract)

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
                place, offset, clone = self._locate(dep, subexprs, with_promotion)

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

    def trim(self, candidate, **kwargs):
        """
        Remove unnecessary reduction loops from the expression loop nest.
        Sometimes, reduction loops can be factored out in outer loops, thus
        reducing the operation count, without breaking data dependencies.
        """
        # Rule out unsafe cases
        if not is_perfect_loop(self.expr_info.innermost_loop):
            return

        # Find out all reducible symbols
        lda = kwargs.get('lda', loops_analysis(self.header))
        reducible, other = [], []
        for i in summands(self.stmt.rvalue):
            symbols = FindInstances(Symbol).visit(i)[Symbol]
            unavoidable = set.intersection(*[set(lda[s]) for s in symbols])
            if candidate in unavoidable:
                return
            reducible.extend([s.symbol for s in symbols if candidate in lda[s]])
            other.extend([s.symbol for s in symbols if candidate not in lda[s]])

        # Make sure we do not break data dependencies
        make_reduce = []
        writes = FindInstances(Writer).visit(candidate)
        for w in flatten(writes.values()):
            if isinstance(w.rvalue, EmptyStatement):
                continue
            if any(s == w.lvalue.symbol for s in other):
                return
            if any(s == w.lvalue.symbol for s in reducible):
                loop = lda[w.lvalue][-1]
                make_reduce.append((w, loop))

        assignments = [(w, p) for w, p in make_reduce if isinstance(w, Assign)]
        loops, parents = zip(*self.expr_info.loops_info)
        index = loops.index(candidate)

        # Perform a number of checks to ensure lifting reductions is safe
        if not all(s in [w.lvalue.symbol for w, _ in make_reduce] for s in reducible):
            return
        if any(p != candidate and not is_perfect_loop(p) for w, p in make_reduce):
            return
        if any(candidate.dim in w.lvalue.rank for w, _ in assignments):
            return
        if any(set(loops[index + 1:]) & set(lda[w.lvalue]) for w, _ in make_reduce):
            return

        # Inject the reductions into the AST
        for w, p in make_reduce:
            name = self._template % len(self.hoisted)
            reduction = Incr(Symbol(name, w.lvalue.rank, w.lvalue.offset),
                             ast_reconstruct(w.rvalue))
            insert_at_elem(p.body, w, reduction)
            handle = self.decls[w.lvalue.symbol]
            declaration = Decl(handle.typ, Symbol(name, handle.lvalue.rank),
                               ArrayInit(np.array([0.0])), handle.qual, handle.attr)
            insert_at_elem(parents[index].children, candidate, declaration)
            ast_replace(self.stmt, {w.lvalue: reduction.lvalue}, copy=True)
            self.hoisted[name] = (reduction, declaration, p, p.body)

        # Pull out the candidate reduction loop
        pulling = loops[index + 1:]
        pulling = zip(*[((l.start, l.end), l.dim) for l in pulling])
        pulling = ItSpace().to_for(*pulling, stmts=[self.stmt])
        parents[index].children.append(pulling[0])
        if len(self.expr_info.parent.children) == 1:
            loops[index].body.remove(loops[index + 1])
        else:
            self.expr_info.parent.children.remove(self.stmt)

        # Clean up removing any now unnecessary symbols
        reads = in_read(candidate, key='symbol')
        declarations = FindInstances(Decl, with_parent=True).visit(self.header)[Decl]
        declarations = dict(declarations)
        for w, p in make_reduce:
            if w.lvalue.symbol not in reads:
                p.body.remove(w)
                if not isinstance(w, Decl):
                    key = self.decls.pop(w.lvalue.symbol)
                    declarations[key].children.remove(key)
