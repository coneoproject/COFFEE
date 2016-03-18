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

# The following global variables capture the internal state of COFFEE
arch = {}
compiler = {}
isa = {}
blas = {}
verbose = False
initialized = False

from base import *
from utils import *
from optimizer import CPULoopOptimizer, GPULoopOptimizer
from vectorizer import LoopVectorizer, VectStrategy
from expression import MetaExpr
from coffee.visitors import FindInstances, EstimateFlops

from collections import defaultdict, OrderedDict
from warnings import warn as warning
import sys
import time


class ASTKernel(object):

    """Manipulate the kernel's Abstract Syntax Tree."""

    def __init__(self, ast, include_dirs=None):
        self.ast = ast
        self.include_dirs = include_dirs or []
        self.blas = False

    def plan_cpu(self, opts):
        """Transform and optimize the kernel suitably for CPU execution."""

        def _generate_cpu_code(self, kernel, **kwargs):
            """Generate kernel code according to the various optimization options."""

            rewrite = kwargs.get('rewrite')
            vectorize = kwargs.get('vectorize')
            v_type, v_param = vectorize if vectorize else (None, None)
            align_pad = kwargs.get('align_pad')
            split = kwargs.get('split')
            precompute = kwargs.get('precompute')
            dead_ops_elimination = kwargs.get('dead_ops_elimination')

            info = visit(kernel)
            decls = info['decls']
            # Structure up expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest, linear_dims = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, decls), parent, nest, linear_dims)
                nests[nest[0]].update({stmt: metaexpr})
            loop_opts = [CPULoopOptimizer(loop, header, decls, exprs)
                         for (loop, header), exprs in nests.items()]
            # Combining certain optimizations is meaningless/forbidden.
            if dead_ops_elimination and split:
                raise RuntimeError("Split forbidden with zero-valued blocks avoidance")
            if dead_ops_elimination and v_type and v_type != VectStrategy.AUTO:
                raise RuntimeError("SIMDization forbidden with zero-valued blocks avoidance")
            if rewrite == 'auto' and len(info['exprs']) > 1:
                warning("Rewrite mode=auto not supported with multiple expressions")
                warning("Switching to rewrite mode=4")
                rewrite = 4

            ### Optimization pipeline ###
            for loop_opt in loop_opts:
                # 0) Expression Rewriting
                if rewrite:
                    loop_opt.rewrite(rewrite)

                # 1) Dead-operations elimination
                if dead_ops_elimination:
                    loop_opt.eliminate_zeros()

                # 2) Splitting
                if split:
                    loop_opt.split(split)

                # 3) Precomputation
                if precompute:
                    loop_opt.precompute(precompute)

                # 4) Vectorization
                if initialized and flatten(loop_opt.expr_linear_loops):
                    vect = LoopVectorizer(loop_opt, kernel)
                    if align_pad:
                        # Padding and data alignment
                        vect.autovectorize()
                    if v_type and v_type != VectStrategy.AUTO:
                        if isa['inst_set'] == 'SSE':
                            raise RuntimeError("SSE vectorization not supported")
                        # Specialize vectorization for the memory access pattern
                        # of the expression
                        vect.specialize(v_type, v_param)

            # Ensure kernel is always marked static inline
            # Remove either or both of static and inline (so that we get the order right)
            kernel.pred = [q for q in kernel.pred if q not in ['static', 'inline']]
            kernel.pred.insert(0, 'inline')
            kernel.pred.insert(0, 'static')

            return loop_opts

        start_time = time.time()

        retval = FindInstances.default_retval()
        kernels = FindInstances(FunDecl, stop_when_found=True).visit(self.ast,
                                                                     ret=retval)[FunDecl]

        if opts.get('Ofast'):
            params = {
                'rewrite': 2,
                'align_pad': True,
                'vectorize': (VectStrategy.SPEC_UAJ_PADD, 2),
                'precompute': 'noloops'
            }
        elif opts.get('O4'):
            params = {
                'rewrite': 'auto',
                'align_pad': True,
                'dead_ops_elimination': True
            }
        elif opts.get('O3'):
            params = {
                'rewrite': 2,
                'align_pad': True,
                'dead_ops_elimination': True
            }
        elif opts.get('O2'):
            params = {
                'rewrite': 2
            }
        elif opts.get('O1'):
            params = {
                'rewrite': 1,
                'align_pad': True
            }
        elif opts.get('O0'):
            params = {}
        else:
            params = opts

        # The optimization passes are performed individually (i.e., "locally") for
        # each function (or "kernel") found in the provided AST
        flops_pre = EstimateFlops().visit(self.ast)
        for kernel in kernels:
            # Generate a specific code version
            _generate_cpu_code(self, kernel, **params)
            # Post processing of the AST ensures higher-quality code
            postprocess(kernel)
        flops_post = EstimateFlops().visit(self.ast)

        tot_time = time.time() - start_time

        out_string = "COFFEE finished in %g seconds (flops: %d -> %d)" % \
            (tot_time, flops_pre, flops_post)
        print (GREEN if flops_post <= flops_pre else RED) % out_string

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
        retval = FindInstances.default_retval()
        kernels = FindInstances(FunDecl, stop_when_found=True).visit(self.ast,
                                                                     ret=retval)[FunDecl]
        for kernel in kernels:
            info = visit(kernel, info_items=['decls', 'exprs'])
            decls = info['decls']
            # Structure up expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest, linear_dims = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, decls), parent, nest, linear_dims)
                nests[nest[0]].update({stmt: metaexpr})

            loop_opts = [GPULoopOptimizer(l, header, decls) for l, header in nests]
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


def init_coffee(_isa, _compiler, _blas, _arch='default'):
    """Initialize COFFEE."""

    global arch, isa, compiler, blas, initialized

    # Populate dictionaries with keywords for backend-specific (hardware,
    # compiler, ...) optimizations
    arch = _init_arch(_arch)
    isa = _init_isa(_isa)
    compiler = _init_compiler(_compiler)
    blas = _init_blas(_blas)
    if isa and compiler:
        initialized = True

    # Allow visits of ASTs that become huge due to transformation. The constant
    # /4000/ was empirically found to be an acceptable value
    sys.setrecursionlimit(4000)


def _init_arch(arch):
    """Set architecture-specific parameters."""

    # In the following, all sizes are in Bytes
    if arch == 'default':
        # The default architecture is a conventional multi-core CPU, such as
        # an Intel Haswell
        return {
            'cache_size': 256 * 10**3,  # The private, closest memory to the core
            'double': 8
        }

    else:
        return {
            'cache_size': 0,
            'double': sys.maxint
        }


def _init_isa(isa):
    """Set the instruction set architecture (isa)."""

    if isa == 'sse':
        return {
            'inst_set': 'SSE',
            'avail_reg': 16,
            'alignment': 16,
            'dp_reg': 2,  # Number of double values per register
            'reg': lambda n: 'xmm%s' % n
        }

    if isa == 'avx':
        return {
            'inst_set': 'AVX',
            'avail_reg': 16,
            'alignment': 32,
            'dp_reg': 4,  # Number of double values per register
            'reg': lambda n: 'ymm%s' % n,
            'zeroall': '_mm256_zeroall ()',
            'setzero': AVXSetZero(),
            'decl_var': '__m256d',
            'align_array': lambda p: '__attribute__((aligned(%s)))' % p,
            'symbol_load': lambda s, r, o=None: AVXLoad(s, r, o),
            'symbol_set': lambda s, r, o=None: AVXSet(s, r, o),
            'store': lambda m, r: AVXStore(m, r),
            'mul': lambda r1, r2: AVXProd(r1, r2),
            'div': lambda r1, r2: AVXDiv(r1, r2),
            'add': lambda r1, r2: AVXSum(r1, r2),
            'sub': lambda r1, r2: AVXSub(r1, r2),
            'l_perm': lambda r, f: AVXLocalPermute(r, f),
            'g_perm': lambda r1, r2, f: AVXGlobalPermute(r1, r2, f),
            'unpck_hi': lambda r1, r2: AVXUnpackHi(r1, r2),
            'unpck_lo': lambda r1, r2: AVXUnpackLo(r1, r2)
        }

    return {}


def _init_compiler(compiler):
    """Set compiler-specific keywords. """

    if compiler == 'intel':
        return {
            'name': 'intel',
            'cmd': 'icc',
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'align_forloop': '#pragma vector aligned',
            'force_simdization': '#pragma simd',
            'AVX': '-xAVX',
            'SSE': '-xSSE',
            'ipo': '-ip',
            'native_opt': '-xHost',
            'vect_header': '#include <immintrin.h>'
        }

    if compiler == 'gnu':
        return {
            'name': 'gnu',
            'cmd': 'gcc',
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'align_forloop': '',
            'force_simdization': '',
            'AVX': '-mavx',
            'SSE': '-msse',
            'ipo': '',
            'native_opt': '-mtune=native',
            'vect_header': '#include <immintrin.h>'
        }

    return {}


def _init_blas(blas):
    """Initialize a dictionary of blas-specific keywords for code generation."""

    import os

    blas_dict = {
        'dir': os.environ.get("PYOP2_BLAS_DIR", ""),
        'namespace': ''
    }

    return blas_dict
