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

from __future__ import absolute_import, print_function, division

from .utils import *
from collections import OrderedDict


class MetaExpr(object):

    """Metadata container for a compute-intensive expression."""

    def __init__(self, type, parent, loops_info, mode=0):
        """Initialize the MetaExpr.

        :param type: the C type of the expression.
        :param parent: the node in which the expression is embedded.
        :param loops_info: an iterator of 2-tuples; each tuple represents a loop
            enclosing the expression (first entry) and its parent (second entry).
        :param mode: the suggested rewrite mode.
        """
        self._type = type
        self._parent = parent
        self._loops_info = list(loops_info)
        self._mode = mode

    @property
    def type(self):
        return self._type

    @property
    def dims(self):
        return tuple(l.dim for l in self.loops)

    @property
    def linear_dims(self):
        return tuple(l.dim for l in self.linear_loops)

    @property
    def out_linear_dims(self):
        return tuple(d for d in self.dims if d not in self.linear_dims)

    @property
    def reduction_dims(self):
        return tuple(l.dim for l in self.reduction_loops)

    @property
    def loops(self):
        return zip(*self._loops_info)[0]

    @property
    def loops_from_dims(self):
        return OrderedDict(zip(self.dims, self.loops))

    @property
    def loops_parents(self):
        return zip(*self._loops_info)[1]

    @property
    def loops_info(self):
        return self._loops_info

    @property
    def linear_loops(self):
        return tuple([l for l in self.loops if l.is_linear])

    @property
    def linear_loops_parents(self):
        return tuple([p for l, p in self._loops_info if l.is_linear])

    @property
    def linear_loops_info(self):
        return tuple([(l, p) for l, p in self._loops_info if l.is_linear])

    @property
    def out_linear_loops(self):
        return tuple([l for l in self.loops if l not in self.linear_loops])

    @property
    def out_linear_loops_parents(self):
        return tuple([p for p in self.loops_parents if p not in self.linear_loops_parents])

    @property
    def out_linear_loops_info(self):
        return tuple([i for i in self.loops_info if i not in self.linear_loops_info])

    @property
    def reduction_loops(self):
        stmts = FindInstances((Writer, Incr)).visit(self.outermost_loop)
        if stmts[Incr]:
            writers = flatten(stmts.values())
            return tuple(l for l in self.loops
                         if all(l.dim not in i.lvalue.rank for i in writers))
        else:
            return ()

    @property
    def reduction_loops_parents(self):
        retval = self.reduction_loops_info
        return zip(*retval)[1] if retval else ()

    @property
    def reduction_loops_info(self):
        return tuple((l, p) for l, p in self.loops_info if l in self.reduction_loops)

    @property
    def perfect_loops(self):
        """Return the loops in a perfect loop nest for the expression."""
        return [l for l in self.loops if is_perfect_loop(l)]

    @property
    def parent(self):
        return self._parent

    @property
    def outermost_loop(self):
        return self.loops[0] if len(self.loops) > 0 else None

    @property
    def outermost_parent(self):
        return self.loops_parents[0] if len(self.loops_parents) > 0 else None

    @property
    def outermost_linear_loop(self):
        return self.linear_loops[0] if len(self.linear_loops) > 0 else None

    @property
    def outermost_linear_loop_parent(self):
        return self.linear_loops_parents[0] if len(self.linear_loops_parents) > 0 else None

    @property
    def innermost_loop(self):
        return self.loops[-1] if len(self.loops) > 0 else None

    @property
    def innermost_parent(self):
        return self.loops_parents[-1] if len(self.loops_parents) > 0 else None

    @property
    def innermost_linear_loop(self):
        return self.linear_loops[-1] if len(self.linear_loops) > 0 else None

    @property
    def innermost_linear_loop_parent(self):
        return self.linear_loops_parents[-1] if len(self.linear_loops_parents) > 0 else None

    @property
    def dimension(self):
        return len(self.linear_dims) if not self.is_scalar else 0

    @property
    def is_scalar(self):
        return all([isinstance(i, int) for i in self.linear_dims])

    @property
    def is_tensor(self):
        return not self.is_scalar

    @property
    def is_linear(self):
        return self.dimension == 1

    @property
    def is_bilinear(self):
        return self.dimension == 2

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value


def copy_metaexpr(expr_info, **kwargs):
    """Given a ``MetaExpr``, return a plain new ``MetaExpr`` starting from a
    copy of ``expr_info``, and replaces some attributes as specified in
    ``kwargs``. ``kwargs`` accepts the following keys: parent, loops_info,
    mode."""

    parent = kwargs.get('parent', expr_info.parent)
    mode = kwargs.get('mode', expr_info.mode)

    new_loops_info, old_loops_info = [], expr_info.loops_info
    to_replace_loops_info = kwargs.get('loops_info', [])
    to_replace_loops_info = dict(zip([l.dim for l, p in to_replace_loops_info],
                                     to_replace_loops_info))
    for loop_info in old_loops_info:
        loop_dim = loop_info[0].dim
        if loop_dim in to_replace_loops_info:
            new_loops_info.append(to_replace_loops_info[loop_dim])
        else:
            new_loops_info.append(loop_info)

    return MetaExpr(expr_info.type, parent, new_loops_info, mode)
