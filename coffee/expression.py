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

    """Information container for a compute-intensive expression."""

    def __init__(self, parent, loops_info, unit_stride_itvars):
        """Initialize the MetaExpr.

        :arg parent: the parent block node in which the expression is embedded.
        :arg loops_info:  the ordered tuple of (loop, parent) the expression
                          depends on.
        :arg unit_stride_itvars: the iteration variables along which the
                                 expression performs unit_stride memory accesses.
        """
        self._parent = parent
        self._loops_info = loops_info
        self._unit_stride_itvars = unit_stride_itvars

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
        return tuple([l for l in self.loops if l.it_var()
                      in self._unit_stride_itvars])

    @property
    def unit_stride_loops_parents(self):
        return tuple([p for l, p in self._loops_info if l.it_var()
                      in self._unit_stride_itvars])

    @property
    def unit_stride_loops_info(self):
        return tuple([(l, p) for l, p in self._loops_info if l.it_var()
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
