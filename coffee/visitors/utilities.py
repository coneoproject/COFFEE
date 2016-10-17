from __future__ import absolute_import, print_function, division

import itertools
import operator
from copy import deepcopy
from collections import OrderedDict, defaultdict
import numpy as np
import networkx as nx

from coffee.visitor import Visitor
from coffee.base import Sum, Sub, Prod, Div, ArrayInit, SparseArrayInit
import coffee.utils


__all__ = ["ReplaceSymbols", "CheckUniqueness", "Uniquify", "Evaluate",
           "EstimateFlops", "ProjectExpansion", "SharingGraph"]


class ReplaceSymbols(Visitor):

    """Replace named symbols in a tree, returning a new tree.

    :arg syms: A dict mapping symbol names to new Symbol objects.
    :arg key: a callable to generate a key from a Symbol, defaults to
         the string representation.
    :arg copy_result: optionally copy the new Symbol whenever it is
         used (guaranteeing that it will be unique)"""
    def __init__(self, syms, key=lambda x: str(x),
                 copy_result=False):
        self.syms = syms
        self.key = key
        self.copy_result = copy_result
        super(ReplaceSymbols, self).__init__()

    def visit_Symbol(self, o):
        try:
            ret = self.syms[self.key(o)]
            if self.copy_result:
                ops, okwargs = ret.operands()
                ret = ret.reconstruct(ops, **okwargs)
            return ret
        except KeyError:
            return o

    def visit_object(self, o):
        return o

    visit_Node = Visitor.maybe_reconstruct


class CheckUniqueness(Visitor):

    """
    Check if all nodes in a tree are unique instances.
    """
    def visit_object(self, o, seen=None):
        return seen

    # Some lists appear in operands()
    def visit_list(self, o, seen=None):
        # Walk list entrys
        for entry in o:
            seen = self.visit(entry, seen=seen)
        return seen

    def visit_Node(self, o, seen=None):
        if seen is None:
            seen = set()
        ops, _ = o.operands()
        for op in ops:
            seen = self.visit(op, seen=seen)
        if o in seen:
            raise RuntimeError("Tree does not contain unique nodes")
        seen.add(o)
        return seen


class Uniquify(Visitor):
    """
    Uniquify all nodes in a tree by recursively calling reconstruct
    """

    visit_Node = Visitor.always_reconstruct

    def visit_object(self, o):
        return deepcopy(o)

    def visit_list(self, o):
        return [self.visit(e) for e in o]


class Evaluate(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """
    Symbolically evaluate an expression enclosed in a loop nest, provided that
    all of the symbols involved are constants and their value is known.

    Return a dictionary mapping symbol names to (newly created) Decl nodes, each
    declaration being initialized with a proper (newly computed and created)
    ArrayInit object.

    :arg decls: dictionary mapping symbol names to known Decl nodes.
    :arg track_zeros: True if the evaluated arrays are expected to be block-sparse
        and the pattern of zeros should be tracked.
    """

    default_args = dict(loop_nest=[])

    def __init__(self, decls, track_zeros):
        self.decls = decls
        self.track_zeros = track_zeros
        self.mapper = {
            Sum: np.add,
            Sub: np.subtract,
            Prod: np.multiply,
            Div: np.divide
        }

        import coffee.vectorizer
        self.up = coffee.vectorizer.vect_roundup
        self.down = coffee.vectorizer.vect_rounddown
        self.make_itspace = coffee.utils.ItSpace
        super(Evaluate, self).__init__()

    def visit_object(self, o, *args, **kwargs):
        return self.default_retval()

    def visit_list(self, o, *args, **kwargs):
        ret = self.default_retval()
        for entry in o:
            ret.update(self.visit(entry, *args, **kwargs))
        return ret

    def visit_Node(self, o, *args, **kwargs):
        ret = self.default_retval()
        for n in o.children:
            ret.update(self.visit(n, *args, **kwargs))
        return ret

    def visit_For(self, o, *args, **kwargs):
        nest = kwargs.pop("loop_nest")
        kwargs["loop_nest"] = nest + [o]
        return self.visit(o.body, *args, **kwargs)

    def visit_Writer(self, o, *args, **kwargs):
        lvalue = o.children[0]
        writes = [l for l in kwargs["loop_nest"] if l.dim in lvalue.rank]

        # Evaluate the expression for each point in in the n-dimensional space
        # represented by /writes/
        dims = tuple(l.dim for l in writes)
        shape = tuple(l.size for l in writes)
        values, precision = np.zeros(shape), None
        for i in itertools.product(*[range(j) for j in shape]):
            point = {d: v for d, v in zip(dims, i)}
            expr_values, precision = self.visit(o.children[1], point=point, *args, **kwargs)
            # The sum takes into account reductions
            values[i] = np.sum(expr_values)

        # If values is not expected to be block-sparse, just return
        if not self.track_zeros:
            return {lvalue: ArrayInit(values)}

        # Sniff the values to check for the presence of zero-valued blocks: ...
        # ... set default nonzero patten
        nonzero = [[(i, 0)] for i in shape]
        # ... track nonzeros in each dimension
        nonzeros_bydim = values.nonzero()
        mapper = []
        for nz_dim in nonzeros_bydim:
            mapper_dim = defaultdict(set)
            for i, nz in enumerate(nz_dim):
                point = []
                # ... handle outer dimensions
                for j in nonzeros_bydim[:-1]:
                    if j is not nz_dim:
                        point.append((j[i],))
                # ... handle the innermost dimension, which is treated "specially"
                # to retain data alignment
                for j in nonzeros_bydim[-1:]:
                    if j is not nz_dim:
                        point.append(tuple(range(self.down(j[i]), self.up(j[i]+1))))
                mapper_dim[nz].add(tuple(point))
            mapper.append(mapper_dim)
        for i, dim in enumerate(mapper[:-1]):
            # Group indices iff contiguous /and/ same codomain
            def grouper(arg):
                m, n = arg
                return m-n, dim[n]
            ranges = []
            for k, g in itertools.groupby(enumerate(sorted(dim.keys())), grouper):
                group = map(operator.itemgetter(1), g)
                ranges.append((group[-1]-group[0]+1, group[0]))
            nonzero[i] = ranges or nonzero[i]
        # Group indices in the innermost dimension iff within vector length size
        ranges, grouper = [], lambda n: self.down(n)
        for k, g in itertools.groupby(sorted(mapper[-1].keys()), grouper):
            group = list(g)
            ranges.append((group[-1]-group[0]+1, group[0]))
        nonzero[-1] = self.make_itspace(mode=1).merge(ranges or nonzero[-1], within=-1)

        return {lvalue: SparseArrayInit(values, precision, tuple(nonzero))}

    def visit_BinExpr(self, o, *args, **kwargs):
        ops, _ = o.operands()
        transformed = [self.visit(op, *args, **kwargs) for op in ops]
        if any([a is None for a in transformed]):
            return
        values, precisions = zip(*transformed)
        # Precisions must match
        assert precisions.count(precisions[0]) == len(precisions)
        # Return the result of the binary operation plus forward the precision
        return self.mapper[o.__class__](*values), precisions[0]

    def visit_Symbol(self, o, *args, **kwargs):
        try:
            # Any time a symbol is encountered, we expect to know the /point/ of
            # the iteration space which is being evaluated. In particular,
            # /point/ is pushed (and then popped) on the environment by a Writer
            # node. If /point/ is missing, that means the root of the visit does
            # not enclose the whole iteration space, which in turn indicates an
            # error in the use of the visitor.
            point = kwargs["point"]
        except KeyError:
            raise RuntimeError("Unknown iteration space point.")
        try:
            decl = self.decls[o.symbol]
        except KeyError:
            raise RuntimeError("Couldn't find a declaration for symbol %s" % o)
        try:
            values = decl.init.values
            precision = decl.init.precision
            shape = values.shape
        except AttributeError:
            raise RuntimeError("%s not initialized with a numpy array" % decl)
        sliced = 0
        for i, (r, s) in enumerate(zip(o.rank, shape)):
            dim = i - sliced
            # Three possible cases...
            if isinstance(r, int):
                # ...the index is used to access a specific dimension (e.g. A[5][..])
                values = values.take(r, dim)
                sliced += 1
            elif r in point:
                # ...a value is being evaluated along dimension /r/ (e.g. A[r] = B[..][r])
                values = values.take(point[r], dim)
                sliced += 1
            else:
                # .../r/ is a reduction dimension
                values = values.take(range(s), dim)
        return values, precision


class ProjectExpansion(Visitor):

    @classmethod
    def default_retval(cls):
        return list()

    """
    Project the output of expression expansion.
    The caller should provid a collection of symbols C. The expression tree (nodes
    that are not of type :class:`~.Expr` are not allowed) is visited and a set of
    tuples returned, one tuple for each symbol in C. Each tuple represents the subset
    of symbols in C that will appear in at least one term after expansion.

    For example, be C = [a, b], and consider the following input expression: ::

        (a*c + d*e)*(b*c + b*f)

    After expansion, the expression becomes: ::

        a*c*b*c + a*c*b*f + d*e*b*c + d*e*b*f

    In which there are four product terms. In these terms, there are two in which
    both 'a' and 'b' appear, and there are two in which only 'b' appears. So the
    visit will return [(a, b), (b,)].

    :arg symbols: the collection of symbols searched for
    """

    def __init__(self, symbols):
        self.symbols = symbols
        super(ProjectExpansion, self).__init__()

    def visit_object(self, o, *args, **kwargs):
        return self.default_retval()

    def visit_Expr(self, o, parent=None, *args, **kwargs):
        projection = self.default_retval()
        for n in o.children:
            projection.extend(self.visit(n, parent=o, *args, **kwargs))
        ret = []
        for n in projection:
            if n not in ret:
                ret.append(n)
        return ret

    def visit_Prod(self, o, parent=None, *args, **kwargs):
        if isinstance(parent, Prod):
            projection = self.default_retval()
            for n in o.children:
                projection.extend(self.visit(n, parent=o, *args, **kwargs))
            return [list(coffee.utils.flatten(projection))]
        else:
            # Only the top level Prod, in a chain of Prods, should do the
            # tensor product
            projection = [self.visit(n, parent=o, *args, **kwargs) for n in o.children]
            product = itertools.product(*projection)
            ret = [list(coffee.utils.flatten(i)) for i in product] or projection
        return ret

    def visit_Symbol(self, o, *args, **kwargs):
        return [[o.symbol]] if o.symbol in self.symbols else [[]]


class EstimateFlops(Visitor):
    """
    Estimate the number of floating point operations a tree performs.

    Does not look inside flat blocks, and all function calls are
    assumed flop free, so this probably underestimates the number of
    flops performed.

    Also, these are "effective" flops, since the compiler may do fancy
    things.
    """

    def visit_object(self, o, *args, **kwargs):
        return 0

    def visit_list(self, o, *args, **kwargs):
        return sum(self.visit(e) for e in o)

    def visit_Node(self, o, *args, **kwargs):
        ops, _ = o.operands()
        return sum(self.visit(op) for op in ops)

    def visit_BinExpr(self, o, *args, **kwargs):
        ops, _ = o.operands()
        return 1 + sum(self.visit(op) for op in ops)

    def visit_AVXBinOp(self, o, *args, **kwargs):
        ops, _ = o.operands()
        return 4 + sum(self.visit(op) for op in ops)

    def visit_Assign(self, o, *args, **kwargs):
        ops, _ = o.operands()
        return sum(self.visit(op) for op in ops[1:])

    def visit_AugmentedAssign(self, o, *args, **kwargs):
        ops, _ = o.operands()
        return 1 + sum(self.visit(op) for op in ops[1:])

    def visit_For(self, o, *args, **kwargs):
        body_flops = sum(self.visit(b) for b in o.body)
        return (o.size // o.increment) * body_flops

    def visit_Invert(self, o, *args, **kwargs):
        ops, _ = o.operands()
        n = ops[1].symbol
        return n**3

    def visit_Determinant1x1(self, o, *args, **kwargs):
        return 1

    def visit_Determinant2x2(self, o, *args, **kwargs):
        return 3

    def visit_Determinant3x3(self, o, *args, **kwargs):
        return 14


class SharingGraph(Visitor):

    @classmethod
    def default_retval(cls):
        return (nx.Graph(), OrderedDict())

    """
    A sharing graph is a particular graph in which vertices represent symbols
    iterating along the expression's linear loops, while an edge between /v1/
    and /v2/ indicates that both /v1/ and /v2/ appear in the same sub-expression,
    or would appear in the same sub-expression if expansion were performed.

    Simultaneously, build a mapper from symbols to nodes in the expression.
    A symbol /s/ (a vertex in the sharing graph) is mapped to a list of nodes
    /[n]/, with /n/ in /[n]/ being the root of a Sum in which /s/ appears in
    both children (i.e., the Sum induces sharing).

    :arg expr_info: A :class:`~.MetaExpr` object describing the expression for
        which the sharing graph is built.
    """

    def __init__(self, expr_info, lda):
        self.expr_info = expr_info
        self.lda = lda
        super(SharingGraph, self).__init__()

    def _update_mapper(self, mapper, loc_syms, pointer=None):
        if pointer:
            old_pointer = None
            for s in set.intersection(*loc_syms):
                v = mapper.setdefault(s, [None])
                old_pointer = v[-1]
                v[-1] = pointer
            for s in set.union(*loc_syms):
                if s in mapper and mapper[s][-1] == old_pointer:
                    mapper[s][-1] = pointer
        else:
            for s in set.union(*loc_syms):
                if s in mapper:
                    mapper[s].append(None)

    def visit_object(self, o, ret=None, *args, **kwargs):
        return self.default_retval()

    def visit_Node(self, o, ret=None, parent=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o, *args, **kwargs)
        return ret

    def visit_Prod(self, o, ret=None, syms=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = self.default_retval()
        if syms is None:
            syms = set()
        G, mapper = ret
        ops, _ = o.operands()
        loc_syms = [set() for i in ops]
        for i, op in enumerate(ops):
            ret = self.visit(op, ret=ret, syms=loc_syms[i], parent=o)
        if all(i for i in loc_syms):
            self._update_mapper(mapper, loc_syms)
            loc_syms = itertools.product(*loc_syms)
            loc_syms = [tuple(coffee.utils.flatten(i)) for i in loc_syms]
            syms |= set(loc_syms)
            G.add_edges_from(loc_syms)
        else:
            for i in loc_syms:
                syms |= i
        return ret

    def visit_Sum(self, o, ret=None, syms=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = self.default_retval()
        if syms is None:
            syms = set()
        pointer = (o, parent)
        _, mapper = ret
        ops, _ = o.operands()
        loc_syms = [set() for i in ops]
        for i, op in enumerate(ops):
            ret = self.visit(op, ret=ret, syms=loc_syms[i], parent=o)
            syms |= loc_syms[i]
        self._update_mapper(mapper, loc_syms, pointer)
        mapper['topsum'] = pointer
        return ret

    visit_Sub = visit_Sum

    def visit_Symbol(self, o, ret=None, syms=None, *args, **kwargs):
        if ret is None:
            ret = self.default_retval()
        G, _ = ret
        deps = [d for d in self.lda[o.symbol]]
        if syms is not None and any(i in self.expr_info.linear_dims for i in deps):
            syms.add((o.urepr,))
            try:
                G.node[o.urepr]['occs'] += 1
            except:
                G.add_node(o.urepr, occs=1)
        return ret
