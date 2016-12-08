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

"""COFFEE's optimization plan constructor."""
from __future__ import absolute_import, print_function, division

import coffee
from .base import *
from .utils import *
from .optimizer import CPULoopOptimizer, GPULoopOptimizer
from .vectorizer import LoopVectorizer, VectStrategy
from .expression import MetaExpr
from .logger import log, warn, PERF_OK, PERF_WARN
from coffee.visitors import Find, EstimateFlops

from collections import defaultdict, OrderedDict
import time


class ASTKernel(object):

    """Manipulate the kernel's Abstract Syntax Tree."""

    def __init__(self, ast, include_dirs=None):
        self.ast = ast
        self.include_dirs = include_dirs or []

    def plan_cpu(self, opts):
        """Optimize this :class:`ASTKernel` for CPU execution.

        :param opts: a dictionary of optimizations to be applied. For a description
            of the recognized optimizations, please refer to the ``coffee.set_opt_level``
            documentation. If equal to ``None``, the default optimizations in
            ``coffee.options['optimizations']`` are applied; these are either the
            optimizations set when COFFEE was initialized or those changed through
            a call to ``set_opt_level``. In this way, a default set of optimizations
            is applied to all kernels, but users are also allowed to select
            specific transformations for individual kernels.
        """

        start_time = time.time()

        kernels = Find(FunDecl, stop_when_found=True).visit(self.ast)[FunDecl]

        if opts is None:
            opts = coffee.OptimizationLevel.retrieve(coffee.options['optimizations'])
        else:
            opts = coffee.OptimizationLevel.retrieve(opts.get('optlevel', {}))

        flops_pre = EstimateFlops().visit(self.ast)

        for kernel in kernels:
            rewrite = opts.get('rewrite')
            vectorize = opts.get('vectorize', (None, None))
            align_pad = opts.get('align_pad')
            split = opts.get('split')
            dead_ops_elimination = opts.get('dead_ops_elimination')

            info = visit(kernel, info_items=['decls', 'exprs'])
            # Collect expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, info['decls']), parent, nest)
                nests[nest[0]].update({stmt: metaexpr})
            loop_opts = [CPULoopOptimizer(loop, header, exprs)
                         for (loop, header), exprs in nests.items()]

            # Combining certain optimizations is forbidden.
            if dead_ops_elimination and split:
                warn("Split forbidden with dead-ops elimination")
                return
            if dead_ops_elimination and vectorize[0]:
                warn("Vect forbidden with dead-ops elimination")
                return
            if rewrite == 'auto' and len(info['exprs']) > 1:
                warn("Rewrite auto forbidden with multiple exprs")
                rewrite = 4

            # Main Ootimization pipeline
            for loop_opt in loop_opts:

                # 0) Expression Rewriting
                if rewrite:
                    loop_opt.rewrite(rewrite)

                # 1) Dead-operations elimination
                if dead_ops_elimination:
                    loop_opt.eliminate_zeros()

                # 2) Code specialization
                if split:
                    loop_opt.split(split)
                if coffee.initialized and flatten(loop_opt.expr_linear_loops):
                    vect = LoopVectorizer(loop_opt, kernel)
                    if align_pad:
                        # Padding and data alignment
                        vect.autovectorize()
                    if vectorize[0] and vectorize[0] != VectStrategy.AUTO:
                        # Specialize vectorization for the memory access pattern
                        # of the expression
                        vect.specialize(*vectorize)

            # Ensure kernel is always marked static inline
            # Remove either or both of static and inline (so that we get the order right)
            kernel.pred = [q for q in kernel.pred if q not in ['static', 'inline']]
            kernel.pred.insert(0, 'inline')
            kernel.pred.insert(0, 'static')

            # Post processing of the AST ensures higher-quality code
            postprocess(kernel)

        flops_post = EstimateFlops().visit(self.ast)

        tot_time = time.time() - start_time

        output = "COFFEE finished in %g seconds (flops: %d -> %d)" % \
            (tot_time, flops_pre, flops_post)
        log(output, PERF_OK if flops_post <= flops_pre else PERF_WARN)

    def plan_gpu(self):
        """Transform the kernel suitably for GPU execution.

        Loops decorated with a ``pragma coffee itspace`` are hoisted out of
        the kernel. The list of arguments in the function signature is
        enriched by adding iteration variables of hoisted loops. The size of any
        kernel's non-constant tensor is modified accordingly.

        For example, consider the following function: ::

            void foo (int A[3]) {
              int B[3] = {...};
              #pragma coffee itspace
              for (int i = 0; i < 3; i++)
                A[i] = B[i];
            }

        plan_gpu modifies its AST such that the resulting output code is ::

            void foo(int A[1], int i) {
              A[0] = B[i];
            }
        """

        # The optimization passes are performed individually (i.e., "locally") for
        # each function (or "kernel") found in the provided AST
        kernels = Find(FunDecl, stop_when_found=True).visit(self.ast)[FunDecl]

        for kernel in kernels:
            info = visit(kernel, info_items=['decls', 'exprs'])
            # Collect expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, info['decls']), parent, nest)
                nests[nest[0]].update({stmt: metaexpr})
            loop_opts = [GPULoopOptimizer(loop, header, exprs)
                         for (loop, header), exprs in nests.items()]

            for loop_opt in loop_opts:
                itspace_vrs, accessed_vrs = loop_opt.extract()

                for v in accessed_vrs:
                    # Change declaration of non-constant iteration space-dependent
                    # parameters by shrinking the size of the iteration space
                    # dimension to 1
                    decl = set(
                        [d for d in kernel.args if d.sym.symbol == v.symbol])
                    dsym = decl.pop().sym if len(decl) > 0 else None
                    if dsym and dsym.rank:
                        dsym.rank = tuple([1 if i in itspace_vrs else j
                                           for i, j in zip(v.rank, dsym.rank)])

                    # Remove indices of all iteration space-dependent and
                    # kernel-dependent variables that are accessed in an itspace
                    v.rank = tuple([0 if i in itspace_vrs and dsym else i
                                    for i in v.rank])

                # Add iteration space arguments
                kernel.args.extend([Decl("int", Symbol("%s" % i)) for i in itspace_vrs])

            # Clean up the kernel removing variable qualifiers like 'static'
            for decl in decls.values():
                d, place = decl
                d.qual = [q for q in d.qual if q not in ['static', 'const']]

            kernel.pred = [q for q in kernel.pred if q not in ['static', 'inline']]

    def gencode(self):
        """Generate a string representation of the AST."""
        return self.ast.gencode()
