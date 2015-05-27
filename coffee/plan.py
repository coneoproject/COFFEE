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
compiler = {}
isa = {}
blas = {}
initialized = False

from base import *
from utils import *
from optimizer import CPULoopOptimizer, GPULoopOptimizer
from vectorizer import LoopVectorizer, VectStrategy
from autotuner import Autotuner
from expression import MetaExpr
from coffee.visitors import FindInstances

from copy import deepcopy as dcopy
from collections import defaultdict, OrderedDict
import itertools
import operator


class ASTKernel(object):

    """Manipulate the kernel's Abstract Syntax Tree.

    The single functionality present at the moment is provided by the
    :meth:`plan_gpu` method, which transforms the AST for GPU execution.
    """

    def __init__(self, ast, include_dirs=[]):
        # Abstract syntax tree of the kernel
        self.ast = ast
        # Used in case of autotuning
        self.include_dirs = include_dirs
        # True if successful conversion to blas operations
        self.blas = False

    def plan_gpu(self):
        """Transform the kernel suitably for GPU execution.

        Loops decorated with a ``pragma coffee itspace`` are hoisted out of
        the kernel. The list of arguments in the function signature is
        enriched by adding iteration variables of hoisted loops. Size of
        kernel's non-constant tensors modified in hoisted loops are modified
        accordingly.

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
        kernels = FindInstances(FunDecl, stop_when_found=True).visit(self.ast)[FunDecl]
        for kernel in kernels:
            info = visit(kernel)
            decls = info['decls']
            # Structure up expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest, domain = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, decls), parent, nest, domain)
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

    def plan_cpu(self, opts):
        """Transform and optimize the kernel suitably for CPU execution."""

        # Unrolling thresholds when autotuning
        autotune_unroll_ths = {
            'default': 10,
            'minimal': 4,
            'hoisted>20': 4,
            'hoisted>40': 1
        }
        # The higher, the more precise and costly is autotuning
        autotune_resolution = 100000000
        # Kernel variant when blas transformation is selected
        blas_config = {'rewrite': 2, 'align_pad': True, 'split': 1, 'precompute': 2}
        # Kernel variants tested when autotuning is enabled
        autotune_min = [('rewrite', {'rewrite': 1, 'align_pad': True}),
                        ('split', {'rewrite': 2, 'align_pad': True, 'split': 1}),
                        ('vect', {'rewrite': 2, 'align_pad': True,
                                  'vectorize': (VectStrategy.SPEC_UAJ_PADD, 1)})]
        autotune_all = [('base', {}),
                        ('base', {'rewrite': 1, 'align_pad': True}),
                        ('rewrite', {'rewrite': 2, 'align_pad': True}),
                        ('rewrite', {'rewrite': 2, 'align_pad': True, 'precompute': 1}),
                        ('rewrite_full', {'rewrite': 2, 'align_pad': True,
                                          'dead_ops_elimination': True}),
                        ('rewrite_full', {'rewrite': 2, 'align_pad': True,
                                          'precompute': 1,
                                          'dead_ops_elimination': True}),
                        ('split', {'rewrite': 2, 'align_pad': True, 'split': 1}),
                        ('split', {'rewrite': 2, 'align_pad': True, 'split': 4}),
                        ('vect', {'rewrite': 2, 'align_pad': True,
                                  'vectorize': (VectStrategy.SPEC_UAJ_PADD, 1)}),
                        ('vect', {'rewrite': 2, 'align_pad': True,
                                  'vectorize': (VectStrategy.SPEC_UAJ_PADD, 2)}),
                        ('vect', {'rewrite': 2, 'align_pad': True,
                                  'vectorize': (VectStrategy.SPEC_UAJ_PADD, 3)})]

        def _generate_cpu_code(self, kernel, **kwargs):
            """Generate kernel code according to the various optimization options."""

            rewrite = kwargs.get('rewrite')
            vectorize = kwargs.get('vectorize')
            v_type, v_param = vectorize if vectorize else (None, None)
            align_pad = kwargs.get('align_pad')
            split = kwargs.get('split')
            toblas = kwargs.get('blas')
            unroll = kwargs.get('unroll')
            permute = kwargs.get('permute')
            precompute = kwargs.get('precompute')
            dead_ops_elimination = kwargs.get('dead_ops_elimination')

            # Combining certain optimizations is meaningless/forbidden.
            if unroll and toblas:
                raise RuntimeError("Cannot unroll and then convert to BLAS")
            if permute and toblas:
                raise RuntimeError("Cannot permute and then convert to BLAS")
            if permute and not precompute:
                raise RuntimeError("Cannot permute without precomputation")
            if rewrite == 3 and split:
                raise RuntimeError("Split forbidden when avoiding zero-columns")
            if rewrite == 3 and toblas:
                raise RuntimeError("BLAS forbidden when avoiding zero-columns")
            if rewrite == 3 and v_type and v_type != VectStrategy.AUTO:
                raise RuntimeError("Zeros removal only supports auto-vectorization")
            if unroll and v_type and v_type != VectStrategy.AUTO:
                raise RuntimeError("Outer-product vectorization needs no unroll")
            if permute and v_type and v_type != VectStrategy.AUTO:
                raise RuntimeError("Outer-product vectorization needs no permute")

            info = visit(kernel)
            decls = info['decls']
            # Structure up expressions and related metadata
            nests = defaultdict(OrderedDict)
            for stmt, expr_info in info['exprs'].items():
                parent, nest, domain = expr_info
                if not nest:
                    continue
                metaexpr = MetaExpr(check_type(stmt, decls), parent, nest, domain)
                nests[nest[0]].update({stmt: metaexpr})

            loop_opts = [CPULoopOptimizer(loop, header, decls, exprs)
                         for (loop, header), exprs in nests.items()]
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

                # 3) Permute outer loop
                if permute:
                    loop_opt.permute(True)

                # 3) Unroll/Unroll-and-jam
                if unroll:
                    loop_opt.unroll(dict(unroll))

                # 4) Vectorization
                if initialized and loop_opt.expr_domain_loops[0]:
                    vect = LoopVectorizer(loop_opt)
                    if align_pad:
                        # Data alignment
                        vect.alignment()
                        # Padding
                        if not toblas:
                            vect.padding()
                    if v_type and v_type != VectStrategy.AUTO:
                        if isa['inst_set'] == 'SSE':
                            raise RuntimeError("SSE vectorization not supported")
                        # Specialize vectorization for the memory access pattern
                        # of the expression
                        vect.specialize(v_type, v_param)

                # 5) Conversion into blas calls
                if toblas:
                    self.blas = loop_opt.blas(toblas)

            # Ensure kernel is always marked static inline
            # Remove either or both of static and inline (so that we get the order right)
            kernel.pred = [q for q in kernel.pred if q not in ['static', 'inline']]
            kernel.pred.insert(0, 'inline')
            kernel.pred.insert(0, 'static')

            return loop_opts

        kernels = FindInstances(FunDecl, stop_when_found=True).visit(self.ast)[FunDecl]
        if opts.get('autotune'):
            if not (compiler and isa):
                raise RuntimeError("Must initialize COFFEE prior to autotuning")
            if len(kernels) > 1:
                raise RuntimeError("Cannot autotune if multiple functions are present")
            # Set granularity of autotuning
            resolution = autotune_resolution
            autotune_configs = autotune_all
            if opts['autotune'] == 'minimal':
                resolution = 1
                autotune_configs = autotune_min
            elif blas:
                library = ('blas', blas['name'])
                autotune_configs.append(('blas', dict(blas_config.items() + [library])))
            variants = []
            autotune_configs_uf = []
            tunable = True
            original_ast = dcopy(self.ast)
            for opt, params in autotune_configs:
                # Generate basic kernel variants
                loop_opts = _generate_cpu_code(self, self.ast, **params)
                if not loop_opts:
                    # No expressions, nothing to tune
                    tunable = False
                    break
                # Increase the stack size, if needed
                increase_stack(loop_opts)
                # Add the base variant to the autotuning process
                variants.append((self.ast, params))
                self.ast = dcopy(original_ast)

                # Calculate variants characterized by different unroll factors,
                # determined heuristically
                loop_opt = loop_opts[0]
                if opt in ['rewrite', 'split']:
                    # Set the unroll threshold
                    if opts['autotune'] == 'minimal':
                        unroll_ths = autotune_unroll_ths['minimal']
                    elif len(loop_opt.hoisted) > 40:
                        unroll_ths = autotune_unroll_ths['hoisted>40']
                    elif len(loop_opt.hoisted) > 20:
                        unroll_ths = autotune_unroll_ths['hoisted>20']
                    else:
                        unroll_ths = autotune_unroll_ths['default']
                    expr_loops = loop_opt.expr_loops
                    if not expr_loops:
                        continue
                    loops_uf = unroll_factors(flatten(expr_loops))
                    all_uf = [bind(k, v) for k, v in loops_uf.items()]
                    all_uf = [uf for uf in list(itertools.product(*all_uf))
                              if reduce(operator.mul, zip(*uf)[1]) <= unroll_ths]
                    for uf in all_uf:
                        params_uf = dict(params.items() + [('unroll', uf)])
                        autotune_configs_uf.append((opt, params_uf))

            # On top of some of the basic kernel variants, apply unroll/unroll-and-jam
            for _, params in autotune_configs_uf:
                loop_opts = _generate_cpu_code(self, self.ast, **params)
                variants.append((self.ast, params))
                self.ast = dcopy(original_ast)

            if tunable:
                # Determine the fastest kernel implementation
                autotuner = Autotuner(variants, self.include_dirs, compiler, isa, blas)
                fastest = autotuner.tune(resolution)
                all_params = autotune_configs + autotune_configs_uf
                name, params = all_params[fastest]
                # Discard values set while autotuning
                if name != 'blas':
                    self.blas = False
            else:
                # The kernel does not get transformed since it does not contain any
                # optimizable expression
                params = {}
        elif opts.get('blas'):
            # Conversion into blas requires a specific set of transformations
            # in order to identify and extract matrix multiplies.
            if not blas:
                raise RuntimeError("Must set PYOP2_BLAS to convert into BLAS calls")
            params = dict(blas_config.items() + [('blas', blas['name'])])
        elif opts.get('Ofast'):
            params = {
                'rewrite': 2,
                'align_pad': True,
                'vectorize': (VectStrategy.SPEC_UAJ_PADD, 2),
                'precompute': 1
            }
        elif opts.get('O4'):
            params = {
                'rewrite': 2,
                'dead_ops_elimination': True,
                'align_pad': True,
                'precompute': 1
            }
        elif opts.get('O3'):
            params = {
                'rewrite': 3,
                'align_pad': True
            }
        elif opts.get('O2'):
            params = {
                'rewrite': 2,
                'align_pad': True
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
        for kernel in kernels:
            # Generate a specific code version
            loop_opts = _generate_cpu_code(self, kernel, **params)

            # Increase stack size if too much space is used on the stack
            increase_stack(loop_opts)

            # Post processing of the AST ensures higher-quality code
            postprocess(kernel)

    def gencode(self):
        """Generate a string representation of the AST."""
        return self.ast.gencode()


def init_coffee(_isa, _compiler, _blas):
    """Initialize COFFEE."""

    global isa, compiler, blas, initialized
    isa = _init_isa(_isa)
    compiler = _init_compiler(_compiler)
    blas = _init_blas(_blas)
    if isa and compiler:
        initialized = True


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
            'decl_aligned_for': '#pragma vector aligned',
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
            'decl_aligned_for': '#pragma vector aligned',
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

    if blas == 'mkl':
        blas_dict.update({
            'name': 'mkl',
            'header': '#include <mkl.h>',
            'link': ['-lmkl_rt']
        })
    elif blas == 'atlas':
        blas_dict.update({
            'name': 'atlas',
            'header': '#include "cblas.h"',
            'link': ['-lsatlas']
        })
    elif blas == 'eigen':
        blas_dict.update({
            'name': 'eigen',
            'header': '#include <Eigen/Dense>',
            'namespace': 'using namespace Eigen;',
            'link': []
        })
    else:
        return {}
    return blas_dict
