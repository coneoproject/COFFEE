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

import networkx as nx

from base import *


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
