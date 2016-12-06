# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2016, Imperial College London.
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

from __future__ import absolute_import, print_function, division

import itertools

from .base import *
from .utils import *
from .exceptions import UnexpectedNode


class Expander(object):

    """Expand the products in an expression according to a set of rules. For a
    comprehensive list of possible rules, refer to the documentation of the
    corresponding wrapper function ``expand`` in ``ExpressionRewriter``."""

    # Constants used by the /expand/ method to charaterize sub-expressions:
    GROUP = 0  # Expression /will/ not trigger expansion
    EXPAND = 1  # Expression /could/ be expanded

    def __init__(self, stmt):
        self.stmt = stmt

    def _build(self, exp, grp, expansions):
        """Create a node for the expansion and keep track of it."""
        expansion = Prod(exp, dcopy(grp))
        # Track the new expansion
        expansions.append(expansion)
        # Untrack any expansions occured in children nodes
        if grp in expansions:
            expansions.remove(grp)
        return expansion

    def _expand(self, node, parent, expansions):
        if isinstance(node, Symbol):
            return ([node], self.EXPAND) if self.should_expand(node) \
                else ([node], self.GROUP)

        elif isinstance(node, (Div, Ternary, FunCall)):
            # Try to expand /within/ the children, but then return saying "I'm not
            # expandable any further"
            for n in node.children:
                self._expand(n, node, expansions)
            return ([node], self.GROUP)

        elif isinstance(node, Prod):
            l_exps, l_type = self._expand(node.left, node, expansions)
            r_exps, r_type = self._expand(node.right, node, expansions)
            if l_type == self.GROUP and r_type == self.GROUP:
                return ([node], self.GROUP)
            # At least one child is expandable (marked as EXPAND), whereas the
            # other could either be expandable as well or groupable (marked
            # as GROUP): so we can perform the expansion
            groupable = l_exps if l_type == self.GROUP else r_exps
            expandable = r_exps if l_type == self.GROUP else l_exps
            to_replace = OrderedDict()
            for exp, grp in itertools.product(expandable, groupable):
                expansion = self._build(exp, grp, expansions)
                to_replace.setdefault(exp, []).append(expansion)
            ast_replace(node, {k: ast_make_expr(Sum, v) for k, v in to_replace.items()},
                        copy=False, mode='symbol')
            # Update the parent node, since an expression has just been expanded
            expanded = node.right if l_type == self.GROUP else node.left
            parent.children[parent.children.index(node)] = expanded
            return (list(flatten(to_replace.values())) or [expanded], self.EXPAND)

        elif isinstance(node, (Sum, Sub)):
            l_exps, l_type = self._expand(node.left, node, expansions)
            r_exps, r_type = self._expand(node.right, node, expansions)
            if l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sum):
                return (l_exps + r_exps, self.EXPAND)
            elif l_type == self.EXPAND and r_type == self.EXPAND and isinstance(node, Sub):
                return (l_exps + [Neg(r) for r in r_exps], self.EXPAND)
            else:
                return ([node], self.GROUP)

        else:
            raise UnexpectedNode("Expansion: %s" % str(node))

    def expand(self, should_expand, **kwargs):
        expressions = kwargs.get('subexprs', [(self.stmt.rvalue, self.stmt)])

        self.should_expand = should_expand

        for node, parent in expressions:
            self._expand(node, parent, [])
