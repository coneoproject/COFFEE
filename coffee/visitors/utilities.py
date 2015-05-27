from __future__ import absolute_import
from coffee.visitor import Visitor, Environment
from coffee.base import Sum, Sub, Prod, Div
from copy import deepcopy
import numpy as np
import itertools


__all__ = ["ReplaceSymbols", "CheckUniqueness", "Uniquify", "Evaluate"]


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
    def visit_object(self, o, env):
        return set()

    # Some lists appear in operands()
    def visit_list(self, o, env):
        ret = set()
        # Walk list entrys
        for entry in o:
            a = self.visit(entry, env=env)
            if len(ret.intersection(a)) != 0:
                raise RuntimeError("Tree does not contain unique nodes")
            ret.update(a)
        return ret

    def visit_Node(self, o, env, *args, **kwargs):
        ret = set([o])
        for a in args:
            if len(ret.intersection(a)) != 0:
                raise RuntimeError("Tree does not contain unique nodes")
            ret.update(a)
        return ret


class Uniquify(Visitor):
    """
    Uniquify all nodes in a tree by recursively calling reconstruct
    """

    visit_Node = Visitor.always_reconstruct

    def visit_object(self, o, env):
        return deepcopy(o)

    def visit_list(self, o, env):
        return [self.visit(e, env=env) for e in o]


class Evaluate(Visitor):
    """
    Symbolically evaluate an expression enclosed in a loop nest, provided that
    all of the symbols involved are constants and their value is known.

    Return a dictionary mapping symbol names to (newly created) Decl nodes, each
    declaration being initialized with a proper (newly computed and created)
    ArrayInit object.

    :arg decls: dictionary mapping symbol names to known Decl nodes.
    """

    default_env = dict(loop_nest=[])

    def __init__(self, decls):
        self.decls = decls
        self.mapper = {
            Sum: np.add,
            Sub: np.subtract,
            Prod: np.multiply,
            Div: np.divide
        }
        super(Evaluate, self).__init__()

    def visit_object(self, o, env):
        return {}

    def visit_list(self, o, env):
        ret = {}
        for entry in o:
            ret.update(self.visit(entry, env=env))
        return ret

    def visit_Node(self, o, env):
        ret = {}
        for n in o.children:
            ret.update(self.visit(n, env=env))
        return ret

    def visit_For(self, o, env):
        new_env = Environment(env, loop_nest=env["loop_nest"] + [o])
        return self.visit(o.body, env=new_env)

    def visit_Writer(self, o, env):
        lvalue = o.children[0]
        writes = [l for l in env["loop_nest"] if l.dim in lvalue.rank]
        # Evaluate the expression for each point in in the n-dimensional space
        # represented by /writes/
        dims = tuple(l.dim for l in writes)
        shape = tuple(l.size for l in writes)
        values = np.zeros(shape)
        for i in itertools.product(*[range(j) for j in shape]):
            point = {d: v for d, v in zip(dims, i)}
            new_env = Environment(env, point=point)
            # The sum takes into account reductions
            values[i] = np.sum(self.visit(o.children[1], new_env))
        return {lvalue: values}

    def visit_BinExpr(self, o, env, *args, **kwargs):
        if any([a is None for a in args]):
            return
        return self.mapper[o.__class__](*args)

    def visit_Par(self, o, env):
        return self.visit(o.child, env)

    def visit_Symbol(self, o, env):
        try:
            # Any time a symbol is encountered, we expect to know the /point/ of
            # the iteration space which is being evaluated. In particular,
            # /point/ is pushed (and then popped) on the environment by a Writer
            # node. If /point/ is missing, that means the root of the visit does
            # not enclose the whole iteration space, which in turn indicates an
            # error in the use of the visitor.
            point = env["point"]
        except KeyError:
            raise RuntimeError("Unknown iteration space point.")
        try:
            decl = self.decls[o.symbol]
        except KeyError:
            raise RuntimeError("Couldn't find a declaration for symbol %s" % o)
        try:
            values = decl.init.values
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
        return values
