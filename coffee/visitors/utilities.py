from __future__ import absolute_import
from coffee.visitor import Visitor
from copy import deepcopy


__all__ = ["ReplaceSymbols", "CheckUniqueness", "Uniquify"]


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

    def visit_Symbol(self, o, env):
        try:
            ret = self.syms[self.key(o)]
            if self.copy_result:
                ops, kwargs = ret.operands()
                ret = ret.reconstruct(ops, **kwargs)
            return ret
        except KeyError:
            return o

    def visit_object(self, o, env):
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
