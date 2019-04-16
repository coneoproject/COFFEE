import numpy as np

from coffee.visitor import Visitor


__all__ = ["EstimateFlops"]


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

    def visit_ArrayInit(self, o, *args, **kwargs):
        vals = o.values
        if isinstance(vals, np.ndarray):
            return sum(self.visit(vals[i]) for i in np.ndindex(vals.shape))
        else:
            return self.visit(vals)

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
