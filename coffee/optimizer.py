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

from .utils import StmtTracker


class LoopOptimizer(object):

    def __init__(self, loop, header, exprs):
        """Initialize the LoopOptimizer.

        :param loop: root AST node of a loop nest
        :param header: the kernel's top node
        :param exprs: list of expressions to be optimized
        """
        self.loop = loop
        self.header = header
        self.exprs = exprs

        # Track nonzero regions accessed in each symbol
        self.nz_syms = {}
        # Track hoisted expressions
        self.hoisted = StmtTracker()

    @property
    def expr_linear_loops(self):
        """Return ``[(loop1, loop2, ...), ...]``, where each tuple contains all
        linear loops enclosing expressions."""
        return [expr_info.linear_loops for expr_info in self.exprs.values()]


class CPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for CPU architectures."""


class GPULoopOptimizer(LoopOptimizer):

    """Loop optimizer for GPU architectures."""
