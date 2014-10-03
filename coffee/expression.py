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

    def __init__(self, parent, loops):
        """Initialize the MetaExpr.

        :arg parent: the parent block node in which the expression is embedded.
        :arg loops:  a 2-tuple of tuples. The first entry is the tuple of all loops
                     whose iteration variables represent are fastest varyind dimensions
                     in the arrays in ``stmt``. The second entry is the tuple of all
                     loops whose iteration variables represent slower dimensions in
                     the arrays in ``stmt``. For example, in the expression
                     ``A[i][j]*B[i][k]``, ``loops = ((for_j, for_k), (for_i))``.
        """
        self._parent = parent
        self._fast_loops, self._slow_loops = loops

    @property
    def loops(self):
        return self._fast_loops + self._slow_loops

    @property
    def fast_itvars(self):
        return tuple([l.it_var() for l in self._fast_loops])

    @property
    def fast_loops(self):
        return self._fast_loops

    @property
    def slow_loops(self):
        return self._slow_loops

    @property
    def perfect_loops(self):
        """Return the loops in a perfect loop nest for the expression."""
        return [l for l in self.loops if is_perfect_loop(l)]

    @property
    def parent(self):
        return self._parent
