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


class StmtInfo():
    """Simple container class defining ``StmtTracker`` values."""

    INFO = ['expr', 'decl', 'loop', 'place']

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            assert(k in self.__class__.INFO)
            setattr(self, k, v)


class StmtTracker(OrderedDict):

    """Track the location of generic statements in an abstract syntax tree.

    Each key in the dictionary is a string representing a symbol. As such,
    StmtTracker can be used only in SSA scopes. Each entry in the dictionary
    is tuple containing information about the symbol: ::

        (expression, declaration, closest_for, place)

    whose semantics is, respectively, as follows:

        * The AST root node of the right-hand side of the statement whose
          left-hand side is ``sym``
        * The AST node of the symbol declaration
        * The AST node of the closest loop enclosing the statement
        * The parent block of the loop
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

            * "expr": change the expression
            * "decl": change the declaration
            * "loop": change the closest loop
            * "place": change the parent block of the loop
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
    def expr(self, sym):
        return self[sym].expr if self.get(sym) else None

    @property
    def decl(self, sym):
        return self[sym].decl if self.get(sym) else None

    @property
    def loop(self, sym):
        return self[sym].loop if self.get(sym) else None

    @property
    def place(self, sym):
        return self[sym].place if self.get(sym) else None


class ExpressionGraph(object):

    """Track read-after-write dependencies between symbols."""

    def __init__(self):
        self.deps = nx.DiGraph()

    def add_dependency(self, sym, expr, self_loop):
        """Extract symbols from ``expr`` and create a read-after-write dependency
        with ``sym``. If ``sym`` already has a dependency, then ``sym`` has a
        self dependency on itself."""

        def extract_syms(sym, node, deps):
            if isinstance(node, Symbol):
                deps.add_edge(sym, node.symbol)
            else:
                for n in node.children:
                    extract_syms(sym, n, deps)

        sym = sym.symbol
        # Add self-dependency
        if self_loop:
            self.deps.add_edge(sym, sym)
        extract_syms(sym, expr, self.deps)

    def has_dep(self, sym, target_sym=None):
        """If ``target_sym`` is not provided, return True if ``sym`` has a
        read-after-write dependency with some other symbols. This is the case if
        ``sym`` has either a self dependency or at least one input edge, meaning
        that other symbols depend on it.
        Otherwise, if ``target_sym`` is not None, return True if ``sym`` has a
        read-after-write dependency on it, i.e. if there is an edge from
        ``target_sym`` to ``sym``."""

        sym = sym.symbol
        if not target_sym:
            return sym in self.deps and zip(*self.deps.in_edges(sym))
        else:
            target_sym = target_sym.symbol
            return sym in self.deps and self.deps.has_edge(sym, target_sym)
