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


from copy import deepcopy as dcopy

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

as_symbol = lambda s: s if isinstance(s, Node) else Symbol(s)


# Meta classes for semantic decoration of AST nodes ##


class Perfect(object):
    """Dummy mixin class used to decorate classes which can form part
    of a perfect loop nest."""
    pass


class LinAlg(object):
    """Dummy mixin class used to decorate classes which represent linear
    algebra operations."""
    pass


# Base classes of the AST ###


class Node(object):

    """The base class of the AST."""

    def __init__(self, children=None, pragma=None):
        self.children = map(as_symbol, children) if children else []
        # Pragmas are used to attach semantical information to nodes
        self._pragma = self._format_pragma(pragma)

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


# Expressions ###

class Expr(Node):

    """Generic expression."""

    pass


class BinExpr(Expr):

    """Generic binary expression."""

    def __init__(self, expr1, expr2, op):
        super(BinExpr, self).__init__([expr1, expr2])
        self.op = op

    def __deepcopy__(self, memo):
        """Binary expressions always need to be copied as plain new objects,
        ignoring whether they have been copied before; that is, the ``memo``
        dictionary tracking the objects copied up to ``self``, which is used
        by the classic ``deepcopy`` method, is ignored."""
        return self.__class__(dcopy(self.children[0]), dcopy(self.children[1]))

    def gencode(self, not_scope=True):
        return (" "+self.op+" ").join([n.gencode(not_scope) for n in self.children])


class UnaryExpr(Expr):

    """Generic unary expression."""

    def __init__(self, expr):
        super(UnaryExpr, self).__init__([expr])

    def __deepcopy__(self, memo):
        """Unary expressions always need to be copied as plain new objects,
        ignoring whether they have been copied before; that is, the ``memo``
        dictionary tracking the objects copied up to ``self``, which is used
        by the classic ``deepcopy`` method, is ignored."""
        return self.__class__(dcopy(self.children[0]))


class Neg(UnaryExpr):

    "Unary negation of an expression"
    def gencode(self, not_scope=False):
        return "-%s" % wrap(self.children[0].gencode()) + semicolon(not_scope)


class ArrayInit(Expr):

    """Array Initilizer. A n-dimensional array A can be statically initialized
    to some values. For example ::

        A[3][3] = {{0.0}} or A[3] = {1, 1, 1}.

    At the moment, initial values like ``{{0.0}}`` and ``{1, 1, 1}`` are passed
    in as simple strings."""

    def __init__(self, values):
        self.values = values

    def gencode(self):
        return self.values


class ColSparseArrayInit(ArrayInit):

    """Array initilizer in which zero-columns, i.e. columns full of zeros, are
    explictly tracked. Only bi-dimensional arrays are allowed."""

    def __init__(self, values, nonzero_bounds, numpy_values):
        """Zero columns are tracked once the object is instantiated.

        :arg values: string representation of the values the array is initialized to
        :arg zerobounds: a tuple of two integers indicating the indices of the first
                         and last nonzero columns
        """
        super(ColSparseArrayInit, self).__init__(values)
        self.nonzero_bounds = nonzero_bounds
        self.numpy_values = numpy_values

    def gencode(self):
        return self.values


class Par(UnaryExpr):

    """Parenthesis object."""

    def gencode(self, not_scope=True):
        return wrap(self.children[0].gencode(not_scope))


class Sum(BinExpr):

    """Binary sum."""

    def __init__(self, expr1, expr2):
        super(Sum, self).__init__(expr1, expr2, "+")


class Sub(BinExpr):

    """Binary subtraction."""

    def __init__(self, expr1, expr2):
        super(Sub, self).__init__(expr1, expr2, "-")


class Prod(BinExpr):

    """Binary product."""

    def __init__(self, expr1, expr2):
        super(Prod, self).__init__(expr1, expr2, "*")


class Div(BinExpr):

    """Binary division."""

    def __init__(self, expr1, expr2):
        super(Div, self).__init__(expr1, expr2, "/")


class Eq(BinExpr):

    """Compare two expressions using the operand ``==``."""

    def __init__(self, expr1, expr2):
        super(Eq, self).__init__(expr1, expr2, "==")


class NEq(BinExpr):

    """Compare two expressions using the operand ``!=``."""

    def __init__(self, expr1, expr2):
        super(NEq, self).__init__(expr1, expr2, "!=")


class Less(BinExpr):

    """Compare two expressions using the operand ``<``."""

    def __init__(self, expr1, expr2):
        super(Less, self).__init__(expr1, expr2, "<")


class LessEq(BinExpr):

    """Compare two expressions using the operand ``<=``."""

    def __init__(self, expr1, expr2):
        super(LessEq, self).__init__(expr1, expr2, "<=")


class Greater(BinExpr):

    """Compare two expressions using the operand ``>``."""

    def __init__(self, expr1, expr2):
        super(Greater, self).__init__(expr1, expr2, ">")


class GreaterEq(BinExpr):

    """Compare two expressions using the operand ``>=``."""

    def __init__(self, expr1, expr2):
        super(GreaterEq, self).__init__(expr1, expr2, ">=")


class Not(UnaryExpr):

    """Compare an expression to ``NULL`` using the operator ``!``."""

    def __init__(self, expr):
        super(Not, self).__init__(expr)

    def gencode(self, not_scope=True):
        return "!%s" % self.children[0].gencode(not_scope)


class FunCall(Expr, Perfect):

    """Function call. """

    def __init__(self, function_name, *args):
        super(Expr, self).__init__(args)
        self.funcall = as_symbol(function_name)

    def gencode(self, not_scope=False):
        return self.funcall.gencode() + \
            wrap(", ".join([n.gencode(not_scope) for n in self.children])) + \
            semicolon(not_scope)


class Ternary(Expr):

    """Ternary operator: ``expr ? true_stmt : false_stmt``."""
    def __init__(self, expr, true_stmt, false_stmt):
        super(Ternary, self).__init__([expr, true_stmt, false_stmt])

    def gencode(self, not_scope=True):
        return ternary(*[c.gencode(True) for c in self.children]) + \
            semicolon(not_scope)


class Symbol(Expr):

    """A generic symbol. The length of ``rank`` is the tensor rank:

    * 0: scalar
    * 1: array
    * 2: matrix, etc.

    :param tuple rank: entries represent the iteration variables the symbol
        depends on, or explicit numbers representing the entry of a tensor the
        symbol is accessing, or the size of the tensor itself. """

    def __init__(self, symbol, rank=(), offset=()):
        super(Symbol, self).__init__([])
        self.symbol = symbol
        self.rank = rank
        self.offset = offset

    def gencode(self, not_scope=True):
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
        return str(self.symbol) + points


# Vector expression classes ###


class AVXSum(Sum):

    """Sum of two vector registers using AVX intrinsics."""

    def gencode(self, not_scope=True):
        op1, op2 = self.children
        return "_mm256_add_pd (%s, %s)" % (op1.gencode(), op2.gencode())


class AVXSub(Sub):

    """Subtraction of two vector registers using AVX intrinsics."""

    def gencode(self, not_scope=True):
        op1, op2 = self.children
        return "_mm256_add_pd (%s, %s)" % (op1.gencode(), op2.gencode())


class AVXProd(Prod):

    """Product of two vector registers using AVX intrinsics."""

    def gencode(self, not_scope=True):
        op1, op2 = self.children
        return "_mm256_mul_pd (%s, %s)" % (op1.gencode(), op2.gencode())


class AVXDiv(Div):

    """Division of two vector registers using AVX intrinsics."""

    def gencode(self, not_scope=True):
        op1, op2 = self.children
        return "_mm256_div_pd (%s, %s)" % (op1.gencode(), op2.gencode())


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


class EmptyStatement(Statement, Perfect):

    """Empty statement."""

    def gencode(self):
        return ""


class FlatBlock(Statement, Perfect):
    """Treat a chunk of code as a single statement, i.e. a C string"""

    def __init__(self, code, pragma=None):
        Statement.__init__(self, pragma)
        self.children.append(code)

    def gencode(self, not_scope=False):
        return self.children[0]


class Assign(Statement, Perfect):

    """Assign an expression to a symbol."""

    def __init__(self, sym, exp, pragma=None):
        super(Assign, self).__init__([sym, exp], pragma)

    def gencode(self, not_scope=False):
        return assign(self.children[0].gencode(),
                      self.children[1].gencode()) + semicolon(not_scope)


class Incr(Statement, Perfect):

    """Increment a symbol by an expression."""

    def __init__(self, sym, exp, pragma=None):
        super(Incr, self).__init__([sym, exp], pragma)

    def gencode(self, not_scope=False):
        sym, exp = self.children
        if isinstance(exp, Symbol) and exp.symbol == 1:
            return incr_by_1(sym.gencode()) + semicolon(not_scope)
        else:
            return incr(sym.gencode(), exp.gencode()) + semicolon(not_scope)


class Decr(Statement, Perfect):

    """Decrement a symbol by an expression."""
    def __init__(self, sym, exp, pragma=None):
        super(Decr, self).__init__([sym, exp], pragma)

    def gencode(self, not_scope=False):
        sym, exp = self.children
        if isinstance(exp, Symbol) and exp.symbol == 1:
            return decr_by_1(sym.gencode()) + semicolon(not_scope)
        else:
            return decr(sym.gencode(), exp.gencode()) + semicolon(not_scope)


class IMul(Statement, Perfect):

    """In-place multiplication of a symbol by an expression."""
    def __init__(self, sym, exp, pragma=None):
        super(IMul, self).__init__([sym, exp], pragma)

    def gencode(self, not_scope=False):
        sym, exp = self.children
        return imul(sym.gencode(), exp.gencode()) + semicolon(not_scope)


class IDiv(Statement, Perfect):

    """In-place division of a symbol by an expression."""
    def __init__(self, sym, exp, pragma=None):
        super(IDiv, self).__init__([sym, exp], pragma)

    def gencode(self, not_scope=False):
        sym, exp = self.children
        return idiv(sym.gencode(), exp.gencode()) + semicolon(not_scope)


class Decl(Statement, Perfect):

    """Declaration of a symbol.

    Syntax: ::

        [qualifiers] typ sym [attributes] [= init];

    E.g.: ::

        static const double FE0[3][3] __attribute__(align(32)) = {{...}};"""

    def __init__(self, typ, sym, init=None, qualifiers=None, attributes=None, pragma=None):
        super(Decl, self).__init__(pragma=pragma)
        self.typ = typ
        self.sym = as_symbol(sym)
        self.qual = qualifiers or []
        self.attr = attributes or []
        self.init = as_symbol(init) if init is not None else EmptyStatement()

    @property
    def size(self):
        """Return the size of the declared variable. In particular, return

        * ``(0,)``, if it is a scalar
        * a tuple, if it is a N-dimensional array, such that each entry
          represents the size of an array dimension (e.g. ``double A[20][10]``
          -> ``(20, 10)``)
        """
        return self.sym.rank or (0,)

    @property
    def is_const(self):
        """Return True if the declaration is a constant."""
        return 'const' in self.qual

    @property
    def scope(self):
        if not hasattr(self, '_scope'):
            raise RuntimeError("Declaration scope available only after a tree visit")
        return self._scope

    @scope.setter
    def scope(self, val):
        if val not in Scope._scopes:
            raise RuntimeError("Only %s are valid scopes" % Scope._scopes)
        self._scope = val

    def gencode(self, not_scope=False):
        if isinstance(self.init, EmptyStatement):
            return decl(spacer(self.qual), self.typ, self.sym.gencode(),
                        spacer(self.attr)) + semicolon(not_scope)
        else:
            return decl_init(spacer(self.qual), self.typ, self.sym.gencode(),
                             spacer(self.attr), self.init.gencode()) + semicolon(not_scope)

    def get_nonzero_columns(self):
        """If the declared array:

        * is a bi-dimensional array,
        * is initialized to some values,
        * the initialized values are of type ColSparseArrayInit

        Then return a tuple of the first and last non-zero columns in the array.
        Else, return an empty tuple."""
        if len(self.sym.rank) == 2 and isinstance(self.init, ColSparseArrayInit):
            return self.init.nonzero_bounds
        else:
            return ()


class Block(Statement):

    """Block of statements."""

    def __init__(self, stmts, pragma=None, open_scope=False):
        if len(stmts) == 1 and isinstance(stmts[0], Block):
            # Avoid nesting of blocks
            super(Block, self).__init__(stmts[0].children, pragma)
        else:
            super(Block, self).__init__(stmts, pragma)
        self.open_scope = open_scope

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
        # If the body is a plain list, cast it to a Block.
        if not isinstance(body, Block):
            if not isinstance(body, list):
                body = [body]
            body = Block(body, open_scope=True)

        super(For, self).__init__([body], pragma)
        self.init = init
        self.cond = cond
        self.incr = incr

    @property
    def itvar(self):
        if isinstance(self.init, Decl):
            return self.init.sym.symbol
        elif isinstance(self.init, Assign):
            return self.init.children[0]

    @property
    def start(self):
        return self.init.init.symbol

    @property
    def end(self):
        return self.cond.children[1].symbol

    @property
    def size(self):
        return self.cond.children[1].symbol - self.init.init.symbol

    @property
    def increment(self):
        return self.incr.children[1].symbol

    @property
    def body(self):
        return self.children[0].children

    @body.setter
    def body(self, new_body):
        self.children[0].children = new_body

    def gencode(self, not_scope=False):
        return "\n".join(self.pragma) + "\n" + for_loop(self.init.gencode(True),
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

    def gencode(self, not_scope=False):
        else_branch = ""
        if len(self.children) == 2:
            else_branch = "else %s" % str(self.children[1])
        return "if (%s) %s %s" % (self.if_expr, str(self.children[0]), else_branch)


class FunDecl(Statement):

    """Function declaration.

    Syntax: ::

        [pred] ret name ([args]) {body};

    E.g.: ::

        static inline void foo(int a, int b) {return;};"""

    def __init__(self, ret, name, args, body, pred=[], headers=None):
        super(FunDecl, self).__init__([body])
        self.pred = pred
        self.ret = ret
        self.name = name
        self.args = args
        self.headers = headers or []

    @property
    def body(self):
        return self.children[0].children

    @body.setter
    def body(self, new_body):
        self.children[0].children = new_body

    def gencode(self):
        headers = "" if not self.headers else \
                  "\n".join(["#include <%s>" % h for h in self.headers])
        sign_list = self.pred + [self.ret, self.name,
                                 wrap(", ".join([arg.gencode(True) for arg in self.args]))]
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

    def gencode(self, not_scope=True):
        op1 = self.r1.gencode()
        op2 = self.r2.gencode()
        return "_mm256_permute2f128_pd (%s, %s, %s)" \
            % (op1, op2, self.mask) + semicolon(not_scope)


class AVXUnpackHi(Statement):

    """Unpack of values in a vector register using AVX intrinsics.
    The intrinsic function used is ``_mm256_unpackhi_pd``."""

    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def gencode(self, not_scope=True):
        op1 = self.r1.gencode()
        op2 = self.r2.gencode()
        return "_mm256_unpackhi_pd (%s, %s)" % (op1, op2) + semicolon(not_scope)


class AVXUnpackLo(Statement):

    """Unpack of values in a vector register using AVX intrinsics.
    The intrinsic function used is ``_mm256_unpacklo_pd``."""

    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def gencode(self, not_scope=True):
        op1 = self.r1.gencode()
        op2 = self.r2.gencode()
        return "_mm256_unpacklo_pd (%s, %s)" % (op1, op2) + semicolon(not_scope)


class AVXSetZero(Statement):

    """Set to 0 the entries of a vector register using AVX intrinsics."""

    def gencode(self, not_scope=True):
        return "_mm256_setzero_pd ()" + semicolon(not_scope)


# Linear Algebra classes


class Invert(Statement, Perfect, LinAlg):
    """In-place inversion of a square array."""
    def __init__(self, sym, dim, pragma=None):
        super(Invert, self).__init__([sym, dim, dim], pragma)

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


class Determinant1x1(Expr, Perfect, LinAlg):

    """Determinant of a 1x1 square array."""
    def __init__(self, sym, pragma=None):
        super(Determinant1x1, self).__init__([sym, 2, 2])

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        return Symbol(sym.gencode(), (0, 0))


class Determinant2x2(Expr, Perfect, LinAlg):

    """Determinant of a 2x2 square array."""
    def __init__(self, sym, pragma=None):
        super(Determinant2x2, self).__init__([sym, 2, 2])

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        v = sym.gencode()
        return Sub(Prod(Symbol(v, (0, 0)), Symbol(v, (1, 1))),
                   Prod(Symbol(v, (0, 1)), Symbol(v, (1, 0))))


class Determinant3x3(Expr, Perfect, LinAlg):

    """Determinant of a 3x3 square array."""
    def __init__(self, sym, pragma=None):
        super(Determinant3x3, self).__init__([sym, 2, 2])

    def gencode(self, scope=False):
        sym, dim, lda = self.children
        v = sym.gencode()
        a0 = Par(Sub(Prod(Symbol(v, (1, 1)), Symbol(v, (2, 2))),
                     Prod(Symbol(v, (1, 2)), Symbol(v, (2, 1)))))
        a1 = Par(Sub(Prod(Symbol(v, (1, 0)), Symbol(v, (2, 2))),
                     Prod(Symbol(v, (1, 2)), Symbol(v, (2, 0)))))
        a2 = Par(Sub(Prod(Symbol(v, (1, 0)), Symbol(v, (2, 1))),
                     Prod(Symbol(v, (1, 1)), Symbol(v, (2, 0)))))
        return Sum(Sub(Prod(Symbol(v, (0, 0)), a0),
                       Prod(Symbol(v, (0, 1)), a1)),
                   Prod(Symbol(v, (0, 2)), a2))


# Extra ###


class PreprocessNode(Node):

    """Represent directives which are handled by the C's preprocessor. """

    def __init__(self, prep):
        super(PreprocessNode, self).__init__([prep])

    def gencode(self, not_scope=False):
        return self.children[0].gencode()


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
    i = c_sym(var)
    init = init or c_sym(0)
    end = c_sym(to)
    if type(code) == str:
        code = FlatBlock(code)
    elif type(code) == list:
        code = Block(code, open_scope=True)
    elif type(code) is not Block:
        code = Block([code], open_scope=True)
    return Block(
        [For(Decl("int", i, init), Less(i, end), Incr(i, c_sym(1)),
             code, pragma)], open_scope=True)


def c_flat_for(code, parent):
    new_block = Block([], open_scope=True)
    parent.children.append(FlatBlock(code))
    parent.children.append(new_block)
    return new_block


# Access modes for a symbol ##


class Access(object):

    def __init__(self, mode):
        self._mode = mode

    def __eq__(self, other):
        return self._mode == other._mode


READ = Access("READ")
WRITE = Access("WRITE")
RW = Access("RW")
INC = Access("INC")
DEC = Access("DEC")
IMUL = Access("IMUL")
IDIV = Access("IDIV")
Access._modes = [READ, WRITE, RW, INC, DEC, IMUL, IDIV]


# Scope of a declaration ##

class Scope(object):

    """An ``EXTERNAL`` scope means the /Decl/ is an argument of a kernel (i.e.,
    when it appears in the list of declarations of a /FunDecl/ object). Otherwise,
    a ``LOCAL`` scope indicates the /Decl/ is within the body of a kernel."""

    def __init__(self, scope):
        self._scope = scope

    def __eq__(self, other):
        return self._scope == other._scope

    def __str__(self):
        return self._scope

LOCAL = Scope("LOCAL")
EXTERNAL = Scope("EXTERNAL")
Scope._scopes = [LOCAL, EXTERNAL]
