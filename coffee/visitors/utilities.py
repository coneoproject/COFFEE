from __future__ import absolute_import

import itertools
import operator
from copy import deepcopy
import numpy as np

from coffee.visitor import Visitor, Environment
from coffee.base import Sum, Sub, Prod, Div, SparseArrayInit
from coffee.utils import ItSpace


__all__ = ["ReplaceSymbols", "CheckUniqueness", "Uniquify", "Evaluate",
           "EstimateFlops"]


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
    """
    Symbolically evaluate an expression enclosed in a loop nest, provided that
    all of the symbols involved are constants and their value is known.

    Return a dictionary mapping symbol names to (newly created) Decl nodes, each
    declaration being initialized with a proper (newly computed and created)
    ArrayInit object.

    :arg decls: dictionary mapping symbol names to known Decl nodes.
    """

    default_args = dict(loop_nest=[])

    def __init__(self, decls):
        self.decls = decls
        self.mapper = {
            Sum: np.add,
            Sub: np.subtract,
            Prod: np.multiply,
            Div: np.divide
        }
        from coffee.plan import isa
        self.min_nzblock = isa['dp_reg']
        super(Evaluate, self).__init__()

    def visit_object(self, o, *args, **kwargs):
        return {}

    def visit_list(self, o, *args, **kwargs):
        ret = {}
        for entry in o:
            ret.update(self.visit(entry, *args, **kwargs))
        return ret

    def visit_Node(self, o, *args, **kwargs):
        ret = {}
        for n in o.children:
            ret.update(self.visit(n, *args, **kwargs))
        return ret

    def visit_For(self, o, *args, **kwargs):
        nest = kwargs.pop("loop_nest")
        kwargs["loop_nest"] = nest + [o]
        return self.visit(o.body, *args, **kwargs)

    def visit_Writer(self, o, *args, **kwargs):
        lvalue = o.children[0]
        writes = [l for l in env["loop_nest"] if l.dim in lvalue.rank]

        # Evaluate the expression for each point in in the n-dimensional space
        # represented by /writes/
        dims = tuple(l.dim for l in writes)
        shape = tuple(l.size for l in writes)
        values, precision = np.zeros(shape), None
        for i in itertools.product(*[range(j) for j in shape]):
            point = {d: v for d, v in zip(dims, i)}
            new_env = Environment(env, point=point)
            expr_values, precision = self.visit(o.children[1], new_env)
            # The sum takes into account reductions
            values[i] = np.sum(expr_values)

        # Sniff the values to check for the presence of zero-valued blocks
        nonzero = [[(i, 0)] for i in shape]
        for i, nz_per_dim in enumerate(values.nonzero()):
            if not nz_per_dim.size:
                continue
            unique_nz_per_dim = np.unique(nz_per_dim)
            ranges = []
            for k, g in itertools.groupby(enumerate(unique_nz_per_dim), lambda (i, x): i-x):
                group = map(operator.itemgetter(1), g)
                # Stored as (size, offset), as expected by SparseArrayInit
                ranges.append((group[-1]-group[0]+1, group[0]))
            nonzero[i] = ranges
        # The minimum size of a non zero-valued block along the innermost dimension
        # is given by /self.min_nzblock/. This avoids breaking alignment and
        # vectorization
        nonzero[-1] = ItSpace(mode=1).merge(nonzero[-1], within=self.min_nzblock)
        return {lvalue: SparseArrayInit(values, precision, tuple(nonzero))}

    def visit_BinExpr(self, o, *args, **kwargs):
        ops, _ = o.operands()
        transformed = [self.visit(op, *args, **kwargs) for op in ops]
        if any([a is None for a in transformed]):
            return
        values, precisions = zip(*args)
        # Precisions must match
        assert precisions.count(precisions[0]) == len(precisions)
        # Return the result of the binary operation plus forward the precision
        return self.mapper[o.__class__](*values), precisions[0]

    def visit_Par(self, o, *args, **kwargs):
        return self.visit(o.child, *args, **kwargs)

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
        return (o.size / o.increment) * body_flops

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
