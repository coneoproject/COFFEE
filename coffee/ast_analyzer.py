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

try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict

import networkx as nx

from base import *
from coffee.visitors import FindInstances


class StmtInfo():
    """Simple container class defining ``StmtTracker`` values."""

    INFO = ['stmt', 'decl', 'loop', 'place']

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            assert(k in self.__class__.INFO)
            setattr(self, k, v)


class StmtTracker(OrderedDict):

    """Track the location of generic statements in an abstract syntax tree.

    Each key in the dictionary is a string representing a symbol. As such,
    StmtTracker can be used only in SSA scopes. Each entry in the dictionary
    is a 4-tuple containing information about the symbol: ::

        (statement, declaration, closest_for, place)

    whose semantics is, respectively, as follows:

        * The AST node whose ``str(lvalue)`` is used as dictionary key
        * The AST node of the symbol declaration
        * The AST node of the closest loop enclosing the statement
        * The parent of the closest loop
    """

    def __setitem__(self, key, value):
        if not isinstance(value, StmtInfo):
            if not isinstance(value, tuple):
                raise RuntimeError("StmtTracker accepts tuple or StmtInfo objects")
            value = StmtInfo(**dict(zip(StmtInfo.INFO, value)))
        return OrderedDict.__setitem__(self, key, value)

    def update_stmt(self, sym, **kwargs):
        """Given the symbol ``sym``, it updates information related to it as
        specified in ``kwargs``. If ``sym`` is not present, return ``None``.
        ``kwargs`` is based on the following special keys:

            * "stmt": change the statement
            * "decl": change the declaration
            * "loop": change the closest loop
            * "place": change the parent the closest loop
        """
        if sym not in self:
            return None
        for k, v in kwargs.iteritems():
            assert(k in StmtInfo.INFO)
            setattr(self[sym], k, v)

    def update_loop(self, loop_a, loop_b):
        """Replace all occurrences of ``loop_a`` with ``loop_b`` in all entries."""

        for sym, sym_info in self.items():
            if sym_info.loop == loop_a:
                self.update_stmt(sym, **{'loop': loop_b})

    @property
    def stmt(self, sym):
        return self[sym].stmt if self.get(sym) else None

    @property
    def decl(self, sym):
        return self[sym].decl if self.get(sym) else None

    @property
    def loop(self, sym):
        return self[sym].loop if self.get(sym) else None

    @property
    def place(self, sym):
        return self[sym].place if self.get(sym) else None

    @property
    def all_stmts(self):
        return set((stmt_info.stmt for stmt_info in self.values() if stmt_info.stmt))

    @property
    def all_places(self):
        return set((stmt_info.place for stmt_info in self.values() if stmt_info.place))

    @property
    def all_loops(self):
        return set((stmt_info.loop for stmt_info in self.values() if stmt_info.loop))


class ExpressionGraph(object):

    """Track read-after-write dependencies between symbols."""

    def __init__(self, node):
        """Initialize the ExpressionGraph.

        :param node: root of the AST visited to initialize the ExpressionGraph.
        """
        self.deps = nx.DiGraph()
        writes = FindInstances(Writer).visit(node, ret=FindInstances.default_retval())
        for type, nodes in writes.items():
            for n in nodes:
                lvalue = n.lvalue
                rvalue = n.rvalue
                if isinstance(rvalue, EmptyStatement):
                    continue
                self.add_dependency(lvalue, rvalue)

    def add_dependency(self, sym, expr):
        """Add dependency between ``sym`` and symbols appearing in ``expr``."""
        retval = FindInstances.default_retval()
        expr_symbols = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for es in expr_symbols:
            self.deps.add_edge(sym.symbol, es.symbol)

    def is_read(self, expr, target_sym=None):
        """Return True if any symbols in ``expr`` is read by ``target_sym``,
        False otherwise. If ``target_sym`` is None, Return True if any symbols
        in ``expr`` are read by at least one symbol, False otherwise."""
        retval = FindInstances.default_retval()
        input_syms = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for s in input_syms:
            if s.symbol not in self.deps:
                continue
            elif not target_sym:
                if zip(*self.deps.in_edges(s.symbol)):
                    return True
            elif nx.has_path(self.deps, target_sym.symbol, s.symbol):
                return True
        return False

    def is_written(self, expr, target_sym=None):
        """Return True if any symbols in ``expr`` is written by ``target_sym``,
        False otherwise. If ``target_sym`` is None, Return True if any symbols
        in ``expr`` are written by at least one symbol, False otherwise."""
        retval = FindInstances.default_retval()
        input_syms = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for s in input_syms:
            if s.symbol not in self.deps:
                continue
            elif not target_sym:
                if zip(*self.deps.out_edges(s.symbol)):
                    return True
            elif nx.has_path(self.deps, s.symbol, target_sym.symbol):
                return True
        return False

    def shares(self, symbols):
        """Return an iterator of tuples, each tuple being a group of symbols
        identifiers sharing the same reads."""
        groups = set()
        for i in [set(self.reads(s)) for s in symbols]:
            group = tuple(j for j in symbols if i.intersection(set(self.reads(j))))
            groups.add(group)
        return list(groups)

    def readers(self, sym):
        """Return the list of symbol identifiers that read from ``sym``."""
        return [i for i, j in self.deps.in_edges(sym)]

    def reads(self, sym):
        """Return the list of symbol identifiers that ``sym`` reads from."""
        return [j for i, j in self.deps.out_edges(sym)]
