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

from utils import *
from collections import OrderedDict


class MetaExpr(object):

    """Metadata container for a compute-intensive expression."""

    def __init__(self, type, parent, loops_info, domain):
        """Initialize the MetaExpr.

        :param type: the C type of the expression.
        :param parent: the parent block node in which the expression is embedded.
        :param loops_info: the ordered tuple of (loop, parent) the expression is
                           enclosed in.
        :param domain: an ``n``-tuple, where ``n`` is the rank of the tensor
                       evaluated by the expression. The i-th entry corresponds to
                       the loop dimension along which iteration occurs (For example,
                       given an output tensor ``A[i][j]``, ``domain=(i, j)``).
        """
        self._type = type
        self._parent = parent
        self._loops_info = loops_info
        self._domain = domain

    @property
    def type(self):
        return self._type

    @property
    def dims(self):
        return [l.dim for l in self.loops]

    @property
    def domain_dims(self):
        return self._domain

    @property
    def out_domain_dims(self):
        return [d for d in self.dims if d not in self.domain_dims]

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
    def domain_loops(self):
        return tuple([l for l in self.loops if l.dim in self.domain_dims])

    @property
    def domain_loops_parents(self):
        return tuple([p for l, p in self._loops_info if l.dim in self.domain_dims])

    @property
    def domain_loops_info(self):
        return tuple([(l, p) for l, p in self._loops_info if l.dim in self.domain_dims])

    @property
    def out_domain_loops(self):
        return tuple([l for l in self.loops if l not in self.domain_loops])

    @property
    def out_domain_loops_parents(self):
        return tuple([p for p in self.loops_parents if p not in self.domain_loops_parents])

    @property
    def out_domain_loops_info(self):
        return tuple([i for i in self.loops_info if i not in self.domain_loops_info])

    @property
    def perfect_loops(self):
        """Return the loops in a perfect loop nest for the expression."""
        return [l for l in self.loops if is_perfect_loop(l)]

    @property
    def parent(self):
        return self._parent

    @property
    def outermost_loop(self):
        return self.loops[0]

    @property
    def dimension(self):
        return len(self.domain_dims) if not self.is_scalar else 0

    @property
    def is_scalar(self):
        return all([isinstance(i, int) for i in self.domain_dims])

    @property
    def is_tensor(self):
        return not self.is_scalar


def copy_metaexpr(expr_info, **kwargs):
    """Given a ``MetaExpr``, return a plain new ``MetaExpr`` starting from a
    copy of ``expr_info``, and replaces some attributes as specified in
    ``kwargs``. In particular, ``kwargs`` has the following keys:

    * ``parent``: the block node that embeds the expression.
    * ``loops_info``: an iterator of 2-tuple ``(loop, loop_parent)`` which
      substitute analogous information in the new ``MetaExpr``.
    * ``domain``: the domain of the output tensor evaluated by the expression.
    """
    parent = kwargs.get('parent', expr_info.parent)
    domain = kwargs.get('domain', expr_info.domain_dims)

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

    return MetaExpr(expr_info.type, parent, new_loops_info, domain)
