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

"""This file contains the hierarchy of classes that implement a kernel's
Abstract Syntax Tree (AST)."""
from __future__ import absolute_import, print_function, division

from copy import deepcopy as dcopy
from cmath import isnan
import numbers
import numpy as np

# Utilities for simple exprs and commands
point = lambda p: "[%s]" % p
point_ofs = lambda p, o: "[%s*%s+%s]" % (p, o[0], o[1])
point_ofs_stride = lambda p, o: "[%s+%s]" % (p, o)
assign = lambda s, e: "%s = %s" % (s, e)
incr = lambda s, e: "%s += %s" % (s, e)
incr_by_1 = lambda s: "++%s" % s
decr = lambda s, e: "%s -= %s" % (s, e)
decr_by_1 = lambda s: "--%s" % s
idiv = lambda s, e: "%s /= %s" % (s, e)
imul = lambda s, e: "%s *= %s" % (s, e)
wrap = lambda e: "(%s)" % e
bracket = lambda s: "{%s}" % s
decl = lambda q, t, s, a: "%s%s %s %s" % (q, t, s, a)
decl_init = lambda q, t, s, a, e: "%s%s %s %s = %s" % (q, t, s, a, e)
for_loop = lambda s1, e, s2, s3: "for (%s; %s; %s)\n%s" % (s1, e, s2, s3)
ternary = lambda e, s1, s2: wrap("%s ? %s : %s" % (e, s1, s2))
init_array = lambda v, f: '{%s}' % ', '.join([f(i) for i in v])

as_symbol = lambda s: s if isinstance(s, Node) else Symbol(s)


# Base classes of the AST ###


class Node(object):

    """The base class of the AST."""

    def __init__(self, children=None, pragma=None):
        self.children = list(map(as_symbol, children)) if children else []
        # Pragmas are used to attach semantical information to nodes
        self._pragma = self._format_pragma(pragma)

    def reconstruct(self, *args, **kwargs):
        """Return a new instance of this :class:`Node`.

        :arg args: positional arguments to the constructor.
        :arg kwargs: keyword arguments to the constructor.
        """
        return type(self)(*args, **kwargs)

    def operands(self):
        """Return the operands of this :class:`Node` as an iterable,
        along with a :class:`dict` of keyword arguments (the pragma decorator)."""
        return self.children, {'pragma': self.pragma}

    def gencode(self):
        return "\n".join([n.gencode() for n in self.children])

    def __str__(self):
        return self.gencode()

    def _format_pragma(self, pragma):
        if pragma is None:
            return set()
        elif isinstance(pragma, (str, Access)):
            return set([pragma])
        elif isinstance(pragma, tuple):
            return set(pragma)
        elif isinstance(pragma, set):
            return pragma
        else:
            raise TypeError("Type '%s' cannot be used as Node pragma" % type(pragma))

    @property
    def urepr(self):
        """A unique representation for this node."""
        return self.gencode()

    @property
    def pragma(self):
        return self._pragma

    @pragma.setter
    def pragma(self, _pragma):
        self._pragma = self._format_pragma(_pragma)


class Root(Node):

    """Root of the AST."""

    def gencode(self):
        header = '// This code is generated visiting a COFFEE AST\n\n'
        return header + Node.gencode(self)


# Meta classes for semantic decoration of AST nodes ##


class Writer(Node):
    """Dummy mixin class used to decorate classes which represent write
    operations (e.g., assignments, since lvalues get modified)."""
    pass


class LinAlg(Node):
    """Dummy mixin class used to decorate classes which represent linear
    algebra operations."""
    pass


# Expressions ###

class Expr(Node):

    """Generic expression."""

    pass


class BinExpr(Expr):

    """Generic binary expression."""

    def __init__(self, expr1, expr2):
        super(BinExpr, self).__init__([expr1, expr2])

    def reconstruct(self, expr1, expr2, **kwargs):
        return type(self)(expr1, expr2)

    def __deepcopy__(self, memo):
        """Binary expressions always need to be copied as plain new objects,
        ignoring whether they have been copied before; that is, the ``memo``
        dictionary tracking the objects copied up to ``self``, which is used
        by the classic ``deepcopy`` method, is ignored."""
        return self.__class__(dcopy(self.children[0]), dcopy(self.children[1]))

    def gencode(self, not_scope=True, parent=None):
        children = [n.gencode(not_scope, self) for n in self.children]
        subtree = (" "+type(self).op+" ").join(children)
        if parent:
            return wrap(subtree)
        return subtree

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


class UnaryExpr(Expr):

    """Generic unary expression."""

    def __init__(self, expr):
        super(UnaryExpr, self).__init__([expr])

    def reconstruct(self, expr, **kwargs):
        return type(self)(expr)

    def gencode(self, not_scope=True, parent=None):
        return "%s(%s)" % (type(self).op, self.children[0].gencode(not_scope)) + semicolon(not_scope)

    def __deepcopy__(self, memo):
        """Unary expressions always need to be copied as plain new objects,
        ignoring whether they have been copied before; that is, the ``memo``
        dictionary tracking the objects copied up to ``self``, which is used
        by the classic ``deepcopy`` method, is ignored."""
        return self.__class__(dcopy(self.children[0]))

    @property
    def child(self):
        return self.children[0]


class Neg(UnaryExpr):
    """Unary negation of an expression"""
    op = "-"


class Not(UnaryExpr):
    """Compare an expression to ``NULL`` using the operator ``!``."""
    op = "!"


class ArrayInit(Expr):

    """Array Initilizer. A n-dimensional array A can be statically initialized
    to some values. For example ::

        A[3][3] = {{0.0}} or A[3] = {1, 1, 1}.
    """

    _default_precision = 12

    def __init__(self, values, precision=None):
        """Initialize an ArrayInit object.

        :arg values: a representation of the values the array is initialized to
        :type values: a string or a numpy ndarray.
        :arg precision: the number of decimal digits that should be used when
            converting a float (in a numpy array) to a string.
        :type precision: integer (defaults to 12)
        """
        self.values = values
        self.precision = precision or ArrayInit._default_precision

    def reconstruct(self, values, precision, **kwargs):
        return type(self)(values, precision, **kwargs)

    def operands(self):
        return [self.values, self.precision], {}

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val):
        if not isinstance(val, (np.ndarray, str)):
            raise TypeError
        self._values = val

    def _formatter(self, v):
        """Format a float into a string, showing up to ``precision`` decimal digits.
        This function is partly extracted from the open_source "FFC: the FEniCS Form
        Compiler", freely accessible at https://bitbucket.org/fenics-project/ffc."""
        f = "%%.%dg" % self.precision
        f_int = "%%.%df" % 1
        eps = 10.0**(-self.precision)
        if not isinstance(v, numbers.Number):
            return v.gencode(not_scope=True)
        elif isnan(v):
            return "NAN"
        elif abs(v - round(v, 1)) < eps:
            return f_int % v
        else:
            return f % v

    def _tabulate_values(self, arr):
        if len(arr.shape) == 1:
            # 1-dimensional case
            return init_array(arr, lambda v: self._formatter(v))
        else:
            # n-dimensional case
            return init_array([self._tabulate_values(arr[0])] +
                              ["\n%s" % self._tabulate_values(arr[i])
                               for i in range(1, arr.shape[0])], str)

    def gencode(self, not_scope=True, parent=None):
        if isinstance(self.values, np.ndarray):
            if parent and not parent.sym.rank:
                return self._formatter(self.values[0])
            return self._tabulate_values(self.values)
        return self.values


class SparseArrayInit(ArrayInit):

    """Array initializer in which non-zero blocks are explictly tracked."""

    def __init__(self, values, precision, nonzero):
        """Initialize a SparseArrayInit object.

        :arg values: a representation of the values the array is initialized to
        :type values: a string or a numpy ndarray
        :arg precision: the number of decimal digits that should be used when
            converting a float (in a numpy array) to a string
        :type precision: integer (defaults to 12)
        :arg nonzero: track a non-zero valued block in the initializer
        :type nonzero: an n-tuple, where n is the rank of the tensor initialized.
            Each entry is a list of 2-tuple. A 2-tuple represents a "panel" of
            non zero-values in the array by indicating 1) size of the region and
            2) offset from the start. For example, consider the following: ::

                A[4][3] = {{0, 0, 0},
                           {2, 1, 0},
                           {2, 1, 0},
                           {0, 0, 0},
                           {3, 3, 0}}

            then, ``nonzero`` takes the following form: ::

                nonzero = ([(2, 1), (1, 4)], [(2, 0)])

            since there are two non-contiguous groups of non zero-valued rows
            (1-2 and 3) and one group of non zero-valued columns
        """
        super(SparseArrayInit, self).__init__(values, precision)

        from coffee.utils import Region
        nonzero = [[Region(size, ofs) for size, ofs in dim] for dim in nonzero]
        self.nonzero = tuple(nonzero)

    def reconstruct(self, values, precision, nonzero, **kwargs):
        return type(self)(values, precision, nonzero, **kwargs)

    def operands(self):
        return [self.values, self.precision, self.nonzero], {}


class Sum(BinExpr):
    """Binary sum."""
    op = "+"


class Sub(BinExpr):
    """Binary subtraction."""
    op = "-"


class Prod(BinExpr):
    """Binary product."""
    op = "*"


class Div(BinExpr):
    """Binary division."""
    op = "/"


class Eq(BinExpr):
    """Compare two expressions using the operand ``==``."""
    op = "=="


class NEq(BinExpr):
    """Compare two expressions using the operand ``!=``."""
    op = "!="


class Less(BinExpr):
    """Compare two expressions using the operand ``<``."""
    op = "<"


class LessEq(BinExpr):
    """Compare two expressions using the operand ``<=``."""
    op = "<="


class Greater(BinExpr):
    """Compare two expressions using the operand ``>``."""
    op = ">"


class GreaterEq(BinExpr):
    """Compare two expressions using the operand ``>=``."""
    op = ">="


class And(BinExpr):
    op = "&&"


class Or(BinExpr):
    op = "||"


class FunCall(Expr):

    """Function call. """

    def __init__(self, function_name, *args):
        super(Expr, self).__init__(args)
        self.funcall = as_symbol(function_name)

    def reconstruct(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def operands(self):
        return [self.funcall] + self.children, {}

    def gencode(self, not_scope=False, parent=None):
        return self.funcall.gencode() + \
            wrap(", ".join([n.gencode(True) for n in self.children])) + \
            semicolon(not_scope)


class Ternary(Expr):

    """Ternary operator: ``expr ? true_stmt : false_stmt``."""
    def __init__(self, expr, true_stmt, false_stmt):
        super(Ternary, self).__init__([expr, true_stmt, false_stmt])

    def reconstruct(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def gencode(self, not_scope=True, parent=None):
        return ternary(*[c.gencode(True) for c in self.children]) + \
            semicolon(not_scope)


class Symbol(Expr):

    """A generic symbol. The length of ``rank`` is the tensor rank:

    * 0: scalar
    * 1: array
    * 2: matrix, etc.

    :param symbol: the Symbol name.
    :param rank: entries represent the iteration variables the symbol
        depends on, or explicit numbers representing the entry of a tensor the
        symbol is accessing, or the size of the tensor itself.
    :param offset: an iterator of 2-tuple (period, stride), one 2-tuple for
        each entry in rank. The period is the multiplier of the rank, while
        stride is a quantity summed to the rank. E.g., if rank=(i, j) and
        offset=((1, 0), (3, 2)), printing a symbol 'a' returns the string
        a[i][3*j + 2].
    """

    def __init__(self, symbol, rank=None, offset=None):
        super(Symbol, self).__init__([])
        self.symbol = symbol
        self.rank = Rank(rank or ())
        self.offset = offset or tuple([(1, 0) for r in self.rank])

    def operands(self):
        return [self.symbol, self.rank, self.offset], {}

    @property
    def dim(self):
        return len(self.rank)

    @property
    def is_const(self):
        from .utils import is_const_dim
        return not self.rank or all(is_const_dim(r) for r in self.rank)

    @property
    def is_number(self):
        try:
            complex(self.symbol)
            return True
        except ValueError:
            return False

    @property
    def is_const_offset(self):
        from .utils import is_const_dim, flatten
        return not self.offset or all(is_const_dim(o) for o in flatten(self.offset))

    @property
    def periods(self):
        return tuple(o[0] for o in self.offset)

    @property
    def strides(self):
        return tuple(o[1] for o in self.offset)

    @property
    def is_unit_period(self):
        return self.is_const_offset and all(i == 1 for i in self.periods)

    @property
    def is_unit_stride(self):
        return self.is_const_offset and all(i == 1 for i in self.strides)

    @property
    def urepr(self):
        """Provide a unique representation of Symbols having same name,
        iteration space, and offset."""
        return (self.symbol, self.rank, self.offset)

    def _genpoints(self):
        points = ""
        if not self.offset:
            for p in self.rank:
                points += point(p)
        else:
            for p, ofs in zip(self.rank, self.offset):
                if ofs == (1, 0):
                    points += point(p)
                elif ofs[0] == 1:
                    points += point_ofs_stride(p, ofs[1])
                else:
                    points += point_ofs(p, ofs)
        return points

    def gencode(self, not_scope=True, parent=None):
        return str(self.symbol) + self._genpoints()


class SymbolIndirection(Symbol):

    def gencode(self, not_scope=True, parent=None):
        return "(*%s)%s" % (str(self.symbol), self._genpoints())


# Vector expression classes ###

class AVXBinOp(BinExpr):

    def gencode(self, not_scope=True):
        op1 = self.children[0]
        op2 = self.children[1]
        return "%s(%s, %s)" % (type(self).op, op1.gencode(), op2.gencode())


class AVXSum(AVXBinOp, Sum):
    """Sum of two vector registers using AVX intrinsics."""
    op = "_mm256_add_pd"


class AVXSub(AVXBinOp, Sub):
    """Subtraction of two vector registers using AVX intrinsics."""
    op = "mm256_sub_pd"


class AVXProd(AVXBinOp, Prod):
    """Product of two vector registers using AVX intrinsics."""
    op = "_mm256_mul_pd"


class AVXDiv(AVXBinOp, Div):
    """Division of two vector registers using AVX intrinsics."""
    op = "_mm256_div_pd"


class AVXLoad(Symbol):

    """Load of values in a vector register using AVX intrinsics."""

    def gencode(self, not_scope=True):
        points = ""
        if not self.offset:
            for p in self.rank:
                points += point(p)
        else:
            for p, ofs in zip(self.rank, self.offset):
                points += point_ofs(p, ofs) if ofs != (1, 0) else point(p)
        symbol = str(self.symbol) + points
        return "_mm256_load_pd (&%s)" % symbol


class AVXSet(Symbol):

    """Replicate the symbol's value in all slots of a vector register
    using AVX intrinsics."""

    def gencode(self, not_scope=True):
        points = ""
        for p in self.rank:
            points += point(p)
        symbol = str(self.symbol) + points
        return "_mm256_set1_pd (%s)" % symbol


# Statements ###


class Statement(Node):

    """Base class for commands productions."""

    def __init__(self, children=None, pragma=None):
        super(Statement, self).__init__(children, pragma)

    @property
    def lvalue(self):
        return None

    @property
    def rvalue(self):
        return None


class EmptyStatement(Statement):

    """Empty statement."""

    def gencode(self, not_scope=False, parent=None):
        return ""


class FlatBlock(Statement):
    """Treat a chunk of code as a single statement, i.e. a C string"""

    def __init__(self, code, pragma=None):
        self.pragma = pragma
        self.children = [code]

    def gencode(self, not_scope=False):
        return self.children[0]


class Assign(Statement, Writer):

    """Assign an expression to a symbol."""

    def __init__(self, sym, exp, pragma=None):
        super(Assign, self).__init__([sym, exp], pragma)

    @property
    def lvalue(self):
        return self.children[0]

    @property
    def rvalue(self):
        return self.children[1]

    @rvalue.setter
    def rvalue(self, val):
        self.children[1] = val

    def gencode(self, not_scope=False):
        prefix = ""
        if self.pragma:
            prefix = "\n".join(p for p in self.pragma) + "\n"
        return prefix + \
            assign(self.children[0].gencode(True),
                   self.children[1].gencode(True)) + \
            semicolon(not_scope)


class AugmentedAssign(Statement, Writer):

    def __init__(self, sym, exp, pragma=None):
        super(AugmentedAssign, self).__init__([sym, exp], pragma)

    @property
    def lvalue(self):
        return self.children[0]

    @property
    def rvalue(self):
        return self.children[1]

    @rvalue.setter
    def rvalue(self, val):
        self.children[1] = val

    def gencode(self, not_scope=False):
        sym, exp = self.children
        prefix = ""
        if self.pragma:
            prefix = "\n".join(p for p in self.pragma) + "\n"
        return "%s%s %s %s%s" % (prefix, sym.gencode(True), type(self).op, exp.gencode(True), semicolon(not_scope))


class Incr(AugmentedAssign):
    """Increment a symbol by an expression."""
    op = "+="


class Decr(AugmentedAssign):
    """Decrement a symbol by an expression."""
    op = "-="


class IMul(AugmentedAssign):
    """In-place multiplication of a symbol by an expression."""
    op = "*="


class IDiv(AugmentedAssign):
    """In-place division of a symbol by an expression."""
    op = "/="


class Decl(Writer):

    """Declaration of a symbol.

    Syntax: ::

        [qualifiers] typ sym [attributes] [= init];

    E.g.: ::

        static const double FE0[3][3] __attribute__(align(32)) = {{...}};"""

    def __init__(self, typ, sym, init=None, qualifiers=None, attributes=None,
                 pointers=None, pragma=None, scope=None):
        super(Decl, self).__init__(pragma=pragma)
        self.typ = typ
        sym = as_symbol(sym)
        self.pointers = pointers or []
        self.qual = qualifiers or []
        self.attr = attributes or []
        init = as_symbol(init) if init is not None else EmptyStatement()
        self.children = [sym, init]

        self._core = self.sym.rank
        self._scope = scope or UNKNOWN

    def operands(self):
        return [self.typ, self.sym, self.init, self.qual, self.attr,
                self.pointers], {}

    def pad(self, new_rank):
        self.sym.rank = new_rank

    @property
    def sym(self):
        return self.children[0]

    @property
    def init(self):
        return self.children[1]

    @init.setter
    def init(self, val):
        self.children[1] = val

    @property
    def lvalue(self):
        return self.children[0]

    @property
    def rvalue(self):
        return self.children[1]

    @rvalue.setter
    def rvalue(self, val):
        self.children[1] = val

    @property
    def size(self):
        """Return the size of the pointed region. In particular, return

        * ``()``, if it is a scalar or it is unknown
        * a tuple, if it is a N-dimensional array, such that each entry
          represents the size of an array dimension (e.g. ``double A[20][10]``
          -> ``(20, 10)``)
        """
        return self.sym.rank or ()

    @property
    def dimension(self):
        """Return the dimension of the declared variable (0 if it's a scalar,
        1 for 1-dimensional arrays, etc.)."""
        return len(self.pointers) + len(self.sym.rank)

    @property
    def core(self):
        """Return the size of the declaraed variable without including padding."""
        return self._core

    @core.setter
    def core(self, val):
        self._core = val

    @property
    def is_const(self):
        """Return True if the declared symbol is constant"""
        return 'const' in self.qual

    @property
    def is_static(self):
        """Return True if the declared symbol is static"""
        return 'static' in self.qual

    @property
    def is_static_const(self):
        """Return True if the declared symbol is static and constant"""
        return self.is_static and self.is_const

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, val):
        self._scope = val

    @property
    def is_pointer_type(self):
        return len(self.pointers) > 0

    @property
    def nonzero(self):
        """Return the location of non-zero valued blocks, if any."""
        return self.init.nonzero if isinstance(self.init, SparseArrayInit) else ()

    def gencode(self, not_scope=False):
        pointers = " " + " ".join(['*' + ' '.join(i) for i in self.pointers])
        if isinstance(self.init, EmptyStatement):
            return decl(spacer(self.qual), self.typ + pointers, self.sym.gencode(),
                        spacer(self.attr)) + semicolon(not_scope)
        else:
            return decl_init(spacer(self.qual), self.typ + pointers, self.sym.gencode(True),
                             spacer(self.attr), self.init.gencode(True, parent=self)) + \
                semicolon(not_scope)


class Block(Statement):

    """Block of statements."""

    def __init__(self, stmts, pragma=None, open_scope=None):
        if len(stmts) == 1 and isinstance(stmts[0], Block):
            # Avoid nesting of blocks
            super(Block, self).__init__(stmts[0].children, pragma)
        else:
            super(Block, self).__init__(stmts, pragma)
        self.open_scope = open_scope

    def reconstruct(self, *stmts, **kwargs):
        return type(self)(stmts, **kwargs)

    def operands(self):
        return self.children, {'pragma': self.pragma, 'open_scope': self.open_scope}

    def gencode(self, not_scope=False):
        code = "".join([n.gencode(not_scope) for n in self.children])
        if self.open_scope:
            code = "{\n%s\n}\n" % indent(code)
        return code


class For(Statement):

    """Represent the classic for loop of an imperative language, although
    some restrictions must be considered: only a single iteration variable
    can be declared and modified (i.e. it is not supported something like ::

        for (int i = 0, j = 0; ...)"""

    def __init__(self, init, cond, incr, body, pragma=None):
        super(For, self).__init__([enforce_block(body)], pragma)
        self.init = init
        self.cond = cond
        self.incr = incr

    def operands(self):
        return [self.init, self.cond, self.incr, self.children[0]], {'pragma': self.pragma}

    def reconstruct(self, init, cond, incr, body, **kwargs):
        return type(self)(init, cond, incr, body, **kwargs)

    @property
    def dim(self):
        if isinstance(self.init, Decl):
            return self.init.sym.symbol
        elif isinstance(self.init, Assign):
            return self.init.children[0]

    @property
    def start(self):
        return self.init.rvalue.symbol

    @property
    def end(self):
        return self.cond.children[1].symbol

    @end.setter
    def end(self, value):
        self.cond.children[1] = as_symbol(value)

    @property
    def size(self):
        return int(self.end) - int(self.start)

    @property
    def increment(self):
        return int(self.incr.children[1].symbol)

    @increment.setter
    def increment(self, value):
        self.incr.children[1] = as_symbol(value)

    @property
    def header(self):
        return (self.start, self.size, self.increment)

    @property
    def block(self):
        return self.children[0]

    @property
    def body(self):
        return self.children[0].children

    @property
    def is_linear(self):
        return '#pragma coffee linear loop' in self.pragma

    @body.setter
    def body(self, new_body):
        self.children[0].children = new_body

    def gencode(self, not_scope=False):
        pragma = [i for i in self.pragma if 'coffee' not in i]
        return "\n".join(pragma) + "\n" + for_loop(self.init.gencode(True),
                                                   self.cond.gencode(),
                                                   self.incr.gencode(True),
                                                   self.children[0].gencode())


class Switch(Statement):
    """Switch construct.

    :param switch_expr: The expression over which to switch.
    :param cases: A tuple of pairs ((case, statement),...)
    """

    def __init__(self, switch_expr, cases):
        super(Switch, self).__init__([s for i, s in cases])

        self.switch_expr = switch_expr
        self.cases = cases

    def operands(self):
        return [self.switch_expr, self.cases], {}

    def reconstruct(self, expr, cases, **kwargs):
        return type(self)(expr, cases, **kwargs)

    def gencode(self):
        return "switch (" + str(self.switch_expr) + ")\n{\n" \
            + indent("\n".join("case %s: \n{\n%s\n}" % (str(i), indent(str(s)))
                               for i, s in self.cases)) + "}"


class If(Statement):
    """If-else construct.

    :param if_expr: The expression driving the jump
    :param branches: A 2-tuple of AST nodes, respectively the 'if' and the 'else'
                     branches
    """

    def __init__(self, if_expr, branches):
        super(If, self).__init__(branches)
        self.if_expr = if_expr

    def operands(self):
        return [self.if_expr, self.children], {}

    def reconstruct(self, expr, branches, **kwargs):
        return type(self)(expr, branches, **kwargs)

    def gencode(self, not_scope=False):
        else_branch = ""
        if len(self.children) == 2:
            else_branch = "else %s" % str(self.children[1])
        return "if (%s) %s %s" % (self.if_expr, str(self.children[0]), else_branch)


class FunDecl(Statement):

    """Function declaration.

    Syntax: ::

        [template]
        [pred] ret name ([args]) {body};

    E.g.(Not templated): ::

        static inline void foo(int a, int b) {return;};

    E.g.(Template C++ Function): ::

        template <typename bar>
        static inline void foo(BaseClass<bar>& a, BaseClass<bar>& b) {return};"""

    def __init__(self, ret, name, args, body, pred=None, headers=None, template=None):
        super(FunDecl, self).__init__([enforce_block(body)])
        self.pred = pred or []
        self.ret = ret
        self.name = name
        self.args = args
        self.headers = headers or []
        self.template = template or ""

    def operands(self):
        return [self.ret, self.name, self.args, self.children[0], self.pred, self.headers, self.template], {}

    def reconstruct(self, ret, name, args, body, pred, headers, template, **kwargs):
        return type(self)(ret, name, args, body, pred, headers, template, **kwargs)

    @property
    def body(self):
        return self.children[0].children

    @body.setter
    def body(self, new_body):
        self.children[0].children = new_body

    def gencode(self):
        headers = "" if not self.headers else \
                  "\n".join(["#include <%s>" % h for h in self.headers])
        sign_list = ["" if not self.template else (self.template + "\n")] + self.pred + \
                    [self.ret, self.name, wrap(", ".join([arg.gencode(True) for arg in self.args]))]
        return headers + "\n" + " ".join(sign_list) + \
            "\n{\n%s\n}" % indent(self.children[0].gencode())


# Vector statements classes


class AVXStore(Assign):

    """Store of values in a vector register using AVX intrinsics."""

    def gencode(self, not_scope=False):
        op1 = self.children[0].gencode()
        op2 = self.children[1].gencode()
        return "_mm256_store_pd (&%s, %s)" % (op1, op2) + semicolon(not_scope)


class AVXLocalPermute(Statement):

    """Permutation of values in a vector register using AVX intrinsics.
    The intrinsic function used is ``_mm256_permute_pd``."""

    def __init__(self, r, mask):
        self.r = r
        self.mask = mask

    def reconstruct(self, r, mask, **kwargs):
        return type(self)(r, mask, **kwargs)

    def operands(self):
        return [self.r, self.mask], {}

    def gencode(self, not_scope=True):
        op = self.r.gencode()
        return "_mm256_permute_pd (%s, %s)" \
            % (op, self.mask) + semicolon(not_scope)


class AVXGlobalPermute(Statement):

    """Permutation of values in two vector registers using AVX intrinsics.
    The intrinsic function used is ``_mm256_permute2f128_pd``."""

    def __init__(self, r1, r2, mask):
        self.r1 = r1
        self.r2 = r2
        self.mask = mask

    def reconstruct(self, r1, r2, mask, **kwargs):
        return type(self)(r1, r2, mask, **kwargs)

    def operands(self):
        return [self.r1, self.r2, self.mask], {}

    def gencode(self, not_scope=True):
        op1 = self.r1.gencode()
        op2 = self.r2.gencode()
        return "_mm256_permute2f128_pd (%s, %s, %s)" \
            % (op1, op2, self.mask) + semicolon(not_scope)


class AVXUnpack(Statement):
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def reconstruct(self, r1, r2, **kwargs):
        return type(self)(r1, r2, **kwargs)

    def operands(self):
        return [self.r1, self.r2], {}

    def gencode(self, not_scope=True):
        op1 = self.r1.gencode()
        op2 = self.r2.gencode()
        return "%s(%s, %s)" % (type(self).op, op1, op2) + semicolon(not_scope)


class AVXUnpackHi(AVXUnpack):

    """Unpack of values in a vector register using AVX intrinsics.
    The intrinsic function used is ``_mm256_unpackhi_pd``."""
    op = "_mm256_unpackhi_pd"


class AVXUnpackLo(AVXUnpack):

    """Unpack of values in a vector register using AVX intrinsics.
    The intrinsic function used is ``_mm256_unpacklo_pd``."""
    op = "_mm256_unpacklo_pd"


class AVXSetZero(Statement):

    """Set to 0 the entries of a vector register using AVX intrinsics."""

    def gencode(self, not_scope=True):
        # mm256_setzero_pd takes no arguments and returns zeroed vector register
        return "_mm256_setzero_pd ()" + semicolon(not_scope)


# Linear Algebra classes


class Invert(Statement, LinAlg):
    """In-place inversion of a square array."""
    def __init__(self, sym, dim, pragma=None):
        super(Invert, self).__init__([sym, dim, dim], pragma)

    def reconstruct(self, sym, dim, **kwargs):
        return type(self)(sym, dim, **kwargs)

    def operands(self):
        return [self.children[0], self.children[1]], {'pragma': self.pragma}

    def gencode(self, not_scope=True):
        sym, dim, lda = self.children
        return """{
  int n = %s;
  int lda = %s;
  int ipiv[n];
  int lwork = n*n;
  double work[lwork];
  int info;

  dgetrf_(&n,&n,%s,&lda,ipiv,&info);
  dgetri_(&n,%s,&lda,ipiv,work,&lwork,&info);
}
""" % (str(dim), str(lda), str(sym), str(sym))


class Determinant(Expr, LinAlg):
    """Generic determinant"""
    def __init__(self, sym, pragma=None):
        super(Determinant, self).__init__([sym, type(self).dim, type(self).lda], pragma=pragma)

    def reconstruct(self, sym, **kwargs):
        return type(self)(sym, **kwargs)

    def operands(self):
        return self.children[0], {}

    def gencode(self):
        raise NotImplementedError("Not implemented")


class Determinant1x1(Determinant):

    """Determinant of a 1x1 square array."""
    dim = 2
    lda = 2

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        return Symbol(sym.gencode(), (0, 0))


class Determinant2x2(Determinant):

    """Determinant of a 2x2 square array."""
    dim = 2
    lda = 2

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        v = sym.gencode()
        return Sub(Prod(Symbol(v, (0, 0)), Symbol(v, (1, 1))),
                   Prod(Symbol(v, (0, 1)), Symbol(v, (1, 0))))


class Determinant3x3(Determinant):

    """Determinant of a 3x3 square array."""
    dim = 2
    lda = 2

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        v = sym.gencode()
        a0 = Sub(Prod(Symbol(v, (1, 1)), Symbol(v, (2, 2))),
                 Prod(Symbol(v, (1, 2)), Symbol(v, (2, 1))))
        a1 = Sub(Prod(Symbol(v, (1, 0)), Symbol(v, (2, 2))),
                 Prod(Symbol(v, (1, 2)), Symbol(v, (2, 0))))
        a2 = Sub(Prod(Symbol(v, (1, 0)), Symbol(v, (2, 1))),
                 Prod(Symbol(v, (1, 1)), Symbol(v, (2, 0))))
        return Sum(Sub(Prod(Symbol(v, (0, 0)), a0),
                       Prod(Symbol(v, (0, 1)), a1)),
                   Prod(Symbol(v, (0, 2)), a2))


# Extra ###


class PreprocessNode(Node):

    """Represent directives which are handled by the C's preprocessor. """

    def __init__(self, prep):
        super(PreprocessNode, self).__init__([prep])

    def reconstruct(self, prep, **kwargs):
        return type(self)(prep, **kwargs)

    def gencode(self, not_scope=False):
        return self.children[0].gencode()


class Rank(tuple):

    def __contains__(self, val):
        from coffee.visitors import Find
        if isinstance(val, Node):
            val, search = str(val), type(Node)
        elif isinstance(val, str):
            val, search = val, Symbol
        else:
            return False
        for i in self:
            if isinstance(i, Node):
                items = Find(search).visit(i)
                if any(val == str(i) for i in items[search]):
                    return True
            elif isinstance(i, str) and val == i:
                return True
        return False


# Utility functions ###


def indent(block):
    """Indent each row of the given string block with ``n*2`` spaces."""
    indentation = " " * 2
    return "\n".join([indentation + s for s in block.split('\n')])


def semicolon(not_scope):
    return "" if not_scope else ";\n"


def spacer(v):
    return " ".join(v) + " " if v else ""


def c_sym(const):
    return Symbol(const, ())


def c_for(var, to, code, pragma="#pragma coffee itspace", init=None):
    i = Symbol(var)
    init = init or Symbol(0)
    end = Symbol(to)
    if type(code) == str:
        code = FlatBlock(code)
    elif type(code) == list:
        code = Block(code, open_scope=True)
    elif type(code) is not Block:
        code = Block([code], open_scope=True)
    return Block(
        [For(Decl("int", i, init), Less(i, end), Incr(i, Symbol(1)),
             code, pragma)], open_scope=True)


def c_flat_for(code, parent):
    new_block = Block([], open_scope=True)
    parent.children.append(FlatBlock(code))
    parent.children.append(new_block)
    return new_block


def enforce_block(body, open_scope=True):
    """Wrap ``body`` in a Block if not already a Block."""
    if not isinstance(body, Block):
        if not isinstance(body, list):
            body = [body]
        body = Block(body, open_scope=open_scope)
    return body


# Access modes for a symbol ##


class Access(object):

    _modes = ["READ", "WRITE", "RW", "INC", "DEC", "IMUL", "IDIV"]

    def __init__(self, mode):
        if mode not in Access._modes:
            raise TypeError
        self._mode = mode


READ = Access("READ")
WRITE = Access("WRITE")
RW = Access("RW")
INC = Access("INC")
DEC = Access("DEC")
IMUL = Access("IMUL")
IDIV = Access("IDIV")


# Scope of a declaration ##

class Scope(object):

    """Four /Scope/s are possible:

        * ``UNKNOWN``: unknown (default)
        * ``EXTERNAL``: a kernel argument
        * ``LOCAL``: within the kernel body
        * ``BUFFER``: within the kernel body, but it is actually a ``shadow copy``
            of a kernel argument.
    """

    _scopes = ["UNKNOWN", "LOCAL", "EXTERNAL", "BUFFER"]

    def __init__(self, scope):
        if scope not in Scope._scopes:
            raise TypeError
        self._scope = scope

    def __str__(self):
        return self._scope


LOCAL = Scope("LOCAL")
EXTERNAL = Scope("EXTERNAL")
BUFFER = Scope("BUFFER")
UNKNOWN = Scope("UNKNOWN")
