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

"""Provide mechanisms to initialize COFFEE or change its state."""

import sys

from base import *
from citations import update_citations
from vectorizer import VectStrategy
from logger import LOG_DEFAULT, set_log_level, warn


__all__ = ['architecture', 'compiler', 'isa', 'options', 'initialized',
           'O0', 'O1', 'O2', 'O3', 'Ofast']


class Options(dict):

    def __init__(self):
        self._callbacks = {}

    def __setitem__(self, key, value):
        super(Options, self).__setitem__(key, value)
        self.maybe_update_backend(key, value)

    def register(self, key, value=None, callback=None):
        self[key] = value
        if callback:
            self._callbacks[key] = callback

    def maybe_update_backend(self, key, value):
        if key in self._callbacks:
            self._callbacks[key](value)


class OptimizationLevel(dict):

    _KNOWN = {}

    @classmethod
    def retrieve(cls, optlevel):
        """Retrieve the set of transformations corresponding to ``optlevel``.

        :param optlevel: may be an :class:`OptimizationLevel` itself (in which
        case ``optlevel`` itself is returned) or the name of the level (a string).
        """
        if isinstance(optlevel, OptimizationLevel):
            return optlevel
        elif isinstance(optlevel, str) and optlevel in cls._KNOWN:
            return cls._KNOWN[optlevel]
        elif not optlevel:
            return O0
        else:
            warn("Unrecognized optimization specified.")
            return O0

    def __init__(self, name, **kwargs):
        self.name = name

        for key, value in kwargs.iteritems():
            self[key] = value

        OptimizationLevel._KNOWN[name] = self


def coffee_init(**kwargs):
    """Initialize COFFEE.

    :param compiler: Options: ``gnu``, ``intel``. By knowing the backend compiler,
        COFFEE can generate specialized code (e.g., by inserting suitable loop pragmas).
    :param isa: Options: ``sse``, ``avx``. The instruction set architecture tells
        COFFEE the vector length and the available intrinsics so that optimized
        vector code (or scalar code suitable for compiler auto-vectorization) is
        generated.
    :param architecture: Options: ``default``, ``intel``.
    :param optlevel: Options: ``O0`` (default), ``O1``, ``O2``, ``O3``, ``Ofast``.
        The higher the optimization level, the more aggresively are the transformations.
        For more details, refer to the ``set_opt_level``'s documentation.
    """

    global initialized, options, architecture, compiler, isa

    architecture_id = kwargs.get('architecture', 'default')
    compiler_id = kwargs.get('compiler')
    isa_id = kwargs.get('isa')
    optlevel = kwargs.get('optlevel', O0)

    architecture = set_architecture(architecture_id)
    compiler = set_compiler(compiler_id)
    isa = set_isa(isa_id)

    if all([architecture, compiler, isa]):
        initialized = True

    options['architecture'] = architecture_id
    options['compiler'] = compiler_id
    options['isa'] = isa_id
    options['optimizations'] = optlevel

    # Allow visits of ASTs that become huge due to transformation. The constant
    # /4000/ was empirically found to be an acceptable value
    sys.setrecursionlimit(4000)


def coffee_reconfigure(**kwargs):
    """Reconfigure the internal state of COFFEE."""

    options['optimizations'] = kwargs.get('optlevel')


def set_opt_level(optlevel):
    """Set the default optimization level.

    :param optlevel: accepted values are: ::

        ``O0``: No optimizations are applied at all (default).
        ``O1``: Apply generalized loop-invariant code motion. Refer to
            ``citations.LUPORINI2015`` for more information.
        ``O2``: Apply sharing elimination and elimination of useless floating
            point operations (e.g., a + 0 == a). Refer to ``citations.LUPORINI2016``
            for more information.
        ``O3``: Apply ``O2`` and enforce data alignment through array padding.
            This maximizes the impact of compiler auto-vectorization, as thoroughly
            discussed in ``citations.LUPORINI2015``.
        ``Ofast``: Apply ``O3``, but resort to explicit outer-product vectorization
            instead. Vector promotion is also attempted to maximize vectorization in
            the outer loops. Refer to ``citations.LUPORINI2015`` for more information.

        Alternatively, one can craft a new composite transformation by creating a
        new :class:`OptimizationLevel`.
    """

    optimizations = OptimizationLevel.retrieve(optlevel)

    update_citations(optimizations)


def set_architecture(architecture_id):
    """Set architecture-specific parameters. Supported architectures:

        * 'default'/'intel': a conventional multi-core CPU, such as an Intel Haswell
    """

    # All sizes are in Bytes
    # The /cache_size/ is the size of the private memory closest to a core

    if architecture_id in ['default', 'intel']:
        return {
            'cache_size': 256 * 10**3,
            'double': 8
        }

    return {}


def set_compiler(compiler_id):
    """Set compiler-specific keywords. Supported compilers:

        * 'gnu' (aka gcc)
        * 'intel' (aka icc)
    """

    if compiler_id == 'gnu':
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

    if compiler_id == 'intel':
        return {
            'name': 'intel',
            'cmd': 'icc',
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'align_forloop': '#pragma vector aligned',
            'force_simdization': '#pragma simd',
            'AVX': '',
            'SSE': '',
            'ipo': '-ip',
            'native_opt': '-xHost',
            'vect_header': '#include <immintrin.h>'
        }

    return {}


def set_isa(isa_id):
    """Set the instruction set architecture (ISA). Supported ISAs:

        * 'sse'
        * 'avx'
    """

    if isa_id == 'sse':
        return {
            'inst_set': 'SSE',
            'avail_reg': 16,
            'alignment': 16,
            'dp_reg': 2,  # Number of values in double precision per register
            'reg': lambda n: 'xmm%s' % n
        }

    if isa_id == 'avx':
        return {
            'inst_set': 'AVX',
            'avail_reg': 16,
            'alignment': 32,
            'dp_reg': 4,  # Number of values in double precision per register
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


O0 = OptimizationLevel('O0')
O1 = OptimizationLevel('O1', rewrite=1)
O2 = OptimizationLevel('O2', rewrite=2, dead_ops_elimination=True)
O3 = OptimizationLevel('O3', align_pad=True, **O2)
Ofast = OptimizationLevel('Ofast', vectorize=(VectStrategy.SPEC_UAJ_PADD, 2),
                          precompute='noloops', **O3)

initialized = False

options = Options()
options.register('logging', LOG_DEFAULT, set_log_level)
options.register('architecture', callback=set_architecture)
options.register('compiler', callback=set_compiler)
options.register('isa', callback=set_isa)
options.register('optimizations', O0.name, callback=set_opt_level)

architecture = {}
compiler = {}
isa = {}  # Instruction Set Architecture
