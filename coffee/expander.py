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

import itertools

from base import *
from utils import *


class Cache():
    """A cache for expanded expressions."""

    def __init__(self):
        self._map = {}
        self._hits = defaultdict(int)

    def make_key(self, exp, grp):
        return (str(exp), str(grp))

    def retrieve(self, key):
        exp = self._map.get(key)
        if exp:
            self._hits[key] += 1
        return exp

    def invalidate(self, exp):
        was_hit = False
        for i, j in self._map.items():
            if str(j) == str(exp):
                self._map.pop(i)
                if self._hits[i] > 0:
                    was_hit = True
        return was_hit

    def add(self, key, exp):
        self._map[key] = exp


class Expander():

    # Constants used by the expand method to charaterize sub-expressions:
    GROUP = 0  # Expression /will/ not trigger expansion
    EXPAND = 1  # Expression /could/ be expanded

    # How many times the expander was invoked
    _handled = 0
    # Temporary variables template
    _expanded_sym = "%(loop_dep)s_EXP_%(expr_id)d_%(i)d"

    def __init__(self, stmt, expr_info=None, decls=None, hoisted=None, expr_graph=None):
        self.stmt = stmt
        self.expr_info = expr_info
        self.decls = decls
        self.hoisted = hoisted
        self.expr_graph = expr_graph

        self.cache = Cache()
        self.local_decls = {}

        # Increment counters for unique variable names
        self.expr_id = Expander._handled
        Expander._handled += 1

    def _hoist(self, expansion, info):
        """Try to aggregate an expanded expression E with a previously hoisted
        expression H. If there are no dependencies, H is expanded with E, so
        no new symbols need be introduced. Otherwise (e.g., the H temporary
        appears in multiple places), create a new symbol."""
        exp, grp = expansion.left, expansion.right

        # First, check if any of the symbols in /exp/ have been hoisted
        try:
            retval = FindInstances.default_retval()
            exp = [s for s in FindInstances(Symbol).visit(exp, ret=retval)[Symbol]
                   if s.symbol in self.hoisted and self.should_expand(s)][0]
        except:
            # No hoisted symbols in the expanded expression, so return
            return {}

        # Before moving on, access the cache to check whether the same expansion
        # has alredy been performed. If that's the case, we retrieve and return the
        # result of that expansion, since there is no need to add further temporaries
        cache_key = self.cache.make_key(exp, grp)
        cached = self.cache.retrieve(cache_key)
        if cached:
            return {exp: cached}

        # Aliases
        hoisted_stmt = self.hoisted[exp.symbol].stmt
        hoisted_decl = self.hoisted[exp.symbol].decl
        hoisted_loop = self.hoisted[exp.symbol].loop
        hoisted_place = self.hoisted[exp.symbol].place
        op = expansion.__class__

        # Is the grouped symbol hoistable, or does it break some data dependency?
        retval = SymbolReferences.default_retval()
        grp_syms = SymbolReferences().visit(grp, ret=retval).keys()
        for l in reversed(self.expr_info.loops):
            for g in grp_syms:
                g_refs = info['symbol_refs'][g]
                g_deps = set(flatten([info['symbols_dep'].get(r[0], []) for r in g_refs]))
                if any([l.dim in g.dim for g in g_deps]):
                    return {}
            if l in hoisted_place.children:
                break

        # Perform the expansion in place unless cache conflicts are detected
        if not self.expr_graph.is_read(exp) and not self.cache.invalidate(exp):
            hoisted_stmt.rvalue = op(hoisted_stmt.rvalue, dcopy(grp))
            self.expr_graph.add_dependency(exp, grp)
            return {exp: exp}

        # Create the necessary new AST objects
        expr = op(dcopy(exp), grp)
        hoisted_exp = dcopy(exp)
        hoisted_exp.symbol = self._expanded_sym % {'loop_dep': exp.symbol,
                                                   'expr_id': self.expr_id,
                                                   'i': len(self.local_decls)}
        decl = dcopy(hoisted_decl)
        decl.sym.symbol = hoisted_exp.symbol
        decl.scope = LOCAL
        stmt = Assign(hoisted_exp, expr)

        # Update the AST
        hoisted_loop.body.append(stmt)
        insert_at_elem(hoisted_place.children, hoisted_decl, decl)

        # Update tracked information
        self.local_decls[decl.sym.symbol] = decl
        self.hoisted[hoisted_exp.symbol] = (stmt, decl, hoisted_loop, hoisted_place)
        self.expr_graph.add_dependency(hoisted_exp, expr)
        self.cache.add(cache_key, hoisted_exp)

        return {exp: hoisted_exp}

    def _build(self, exp, grp):
        """Create a node for the expansion and keep track of it."""
        expansion = Prod(exp, dcopy(grp))
        # Track the new expansion
        self.expansions.append(expansion)
        # Untrack any expansions occured in children nodes
        if grp in self.expansions:
            self.expansions.remove(grp)
        return expansion

    def _expand(self, node, parent):
        if isinstance(node, Symbol):
            return ([node], self.EXPAND) if self.should_expand(node) \
                else ([node], self.GROUP)

        elif isinstance(node, Par):
            return self._expand(node.child, node)

        elif isinstance(node, (Div, FunCall)):
            # Try to expand /within/ the children, but then return saying "I'm not
            # expandable any further"
            for n in node.children:
                self._expand(n, node)
            return ([node], self.GROUP)

        elif isinstance(node, Prod):
            l_exps, l_type = self._expand(node.left, node)
            r_exps, r_type = self._expand(node.right, node)
            if l_type == self.GROUP and r_type == self.GROUP:
                return ([node], self.GROUP)
            # At least one child is expandable (marked as EXPAND), whereas the
            # other could either be expandable as well or groupable (marked
            # as GROUP): so we can perform the expansion
            groupable = l_exps if l_type == self.GROUP else r_exps
            expandable = r_exps if l_type == self.GROUP else l_exps
            to_replace = OrderedDict()
            for exp, grp in itertools.product(expandable, groupable):
                expansion = self._build(exp, grp)
                to_replace.setdefault(exp, []).append(expansion)
            ast_replace(node, {k: ast_make_expr(Sum, v) for k, v in to_replace.items()},
                        mode='symbol')
            # Update the parent node, since an expression has just been expanded
            expanded = node.right if l_type == self.GROUP else node.left
            parent.children[parent.children.index(node)] = expanded
            return (list(flatten(to_replace.values())) or [expanded], self.EXPAND)

        elif isinstance(node, (Sum, Sub)):
            l_exps, l_type = self._expand(node.left, node)
            r_exps, r_type = self._expand(node.right, node)
            if l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sum):
                return (l_exps + r_exps, self.EXPAND)
            elif l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sub):
                return (l_exps + [Neg(r) for r in r_exps], self.EXPAND)
            else:
                return ([node], self.GROUP)

        else:
            raise RuntimeError("Expansion error: unknown node: %s" % str(node))

    def expand(self, should_expand, **kwargs):
        not_aggregate = kwargs.get('not_aggregate')
        expressions = kwargs.get('subexprs', [(self.stmt.rvalue, self.stmt)])

        self.should_expand = should_expand

        for node, parent in expressions:
            self.expansions = []
            self._expand(node, parent)

            if not_aggregate:
                continue
            info = visit(self.expr_info.outermost_loop) if self.expr_info else visit(parent)
            for expansion in self.expansions:
                hoisted = self._hoist(expansion, info)
                if hoisted:
                    ast_replace(parent, hoisted, copy=True, mode='symbol')
                    ast_remove(parent, expansion.right, mode='symbol')

        self.decls.update(self.local_decls)
