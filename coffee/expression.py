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

from utils import is_perfect_loop


class MetaExpr(object):

    """Metadata container for a compute-intensive expression."""

    def __init__(self, type, parent, loops_info, unit_stride_itvars):
        """Initialize the MetaExpr.

        :param type: the C type of the expression.
        :param parent: the parent block node in which the expression is embedded.
        :param loops_info: the ordered tuple of (loop, parent) the expression
                           depends on.
        :param unit_stride_itvars: the unite-stride loop dimensions, as iteration
                                   variables, along which writes are performed.
        """
        self._type = type
        self._parent = parent
        self._loops_info = loops_info
        self._unit_stride_itvars = unit_stride_itvars

    @property
    def type(self):
        return self._type

    @property
    def loops(self):
        return zip(*self._loops_info)[0]

    @property
    def loops_parents(self):
        return zip(*self._loops_info)[1]

    @property
    def loops_info(self):
        return self._loops_info

    @property
    def unit_stride_loops(self):
        return tuple([l for l in self.loops if l.itvar in self._unit_stride_itvars])

    @property
    def unit_stride_loops_parents(self):
        return tuple([p for l, p in self._loops_info if l.itvar
                      in self._unit_stride_itvars])

    @property
    def unit_stride_loops_info(self):
        return tuple([(l, p) for l, p in self._loops_info if l.itvar
                      in self._unit_stride_itvars])

    @property
    def slow_loops(self):
        return tuple(set(self.loops) - set(self.unit_stride_loops))

    @property
    def slow_loops_parents(self):
        return tuple(set(self.loops_parents) - set(self.unit_stride_loops_parents))

    @property
    def slow_loops_info(self):
        return tuple(set(self.loops_info) - set(self.unit_stride_loops_info))

    @property
    def perfect_loops(self):
        """Return the loops in a perfect loop nest for the expression."""
        return [l for l in self.loops if is_perfect_loop(l)]

    @property
    def parent(self):
        return self._parent

    @property
    def unit_stride_itvars(self):
        return self._unit_stride_itvars

    @property
    def dimension(self):
        return len(self.unit_stride_loops)


def copy_metaexpr(expr_info, **kwargs):
    """Given a ``MetaExpr``, return a plain new ``MetaExpr`` starting from a
    copy of ``expr_info``, and replaces some attributes as specified in
    ``kwargs``. In particular, ``kwargs`` has the following keys:

    * ``parent``: the block node that embeds the expression.
    * ``loops_info``: an iterator of 2-tuple ``(loop, loop_parent)`` which
      substitute analogous information in the new ``MetaExpr``.
    * ``itvars``: the iteration variables along which the expression performs
      unit-stride accesses.
    """

    parent = kwargs.get('parent', expr_info.parent)
    unit_stride_itvars = kwargs.get('itvars', expr_info.unit_stride_itvars)

    new_loops_info, old_loops_info = [], expr_info.loops_info
    to_replace_loops_info = kwargs.get('loops_info', [])
    to_replace_loops_info = dict(zip([l.itvar for l, p in to_replace_loops_info],
                                     to_replace_loops_info))
    for loop_info in old_loops_info:
        loop_itvar = loop_info[0].itvar
        if loop_itvar in to_replace_loops_info:
            new_loops_info.append(to_replace_loops_info[loop_itvar])
        else:
            new_loops_info.append(loop_info)

    return MetaExpr(expr_info.type, parent, new_loops_info, unit_stride_itvars)
