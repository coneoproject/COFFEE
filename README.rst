COFFEE
======

COFFEE is a COmpiler For Fast Expression Evaluation. Given mathematical expressions
embedded in a possibly non-perfect loop nest, it restructures the expression so as
to:

* Optimize cross-loop arithmetic intensity.

    - This makes extensive use of operator
      properties such as commutativity, associativity, and distributivity to expose and
      move loop-invariant sub-expressions at the right level in the enclosing loop nest.
      Invariant sub-expressions are hoisted such that they can be easily vectorized by
      a backend compiler, like LLVM, GCC, and ICC.
    - Expressions are also transformed with the goal of minimizing register pressure in
      the various levels of the loop nest: for example, factorization and reassociatition
      are exploited to heuristically generate source code that is less likely to suffer
      from register spilling.
    - Symbolic execution is performed to detect the presence of block-sparse arrays and
      avoid useless computation (e.g. when involving products with zero-valued scalars).

* Specialize the code for the underlying architecture. COFFEE currently supports
  standard CPU architectures, but its intermediate representation was designed to be
  easily extendible to other platforms, like accelerators (e.g. GPUs). Code
  specialization focuses on:

    - padding and data alignment, to maximize the effectiveness of vectorization;
    - expression splitting, which is essentially loop fission based on operators'
      associativity;
    - register tiling, driven by the application semantics (i.e. optimized
      implementation of tiling exploiting vector registers are provided for expressions
      characterized by "special" memory access patterns);
    - translation into blas: COFFEE inspects the expressions and tries to replace
      sub-expressions with highly-optimized BLAS-like calls (currently, both BLAS and
      Eigen and supported)

COFFEE provides a broad range of code transformations, which makes it suitable to
automatically and efficiently explore a large space of possible optimizations for
compute-bound numerical kernels. An auto-tuning system is employed to individuate the
optimal composition of code transformations for a given problem.

So far, COFFEE has been used extensively and with success for the optimization of
scientific codes. In particular, it is fully integrated with Firedrake, an automated
system for the portable solution of partial differential equations using the finite
element method.
