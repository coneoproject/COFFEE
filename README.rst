COFFEE
======

COFFEE is a COmpiler For Fast Expression Evaluation. Given mathematical expressions
embedded in a loop nest, COFFEE can be driven in the application of two classes of
transformations:

* Expression rewriting, for flops minimisation. A small set of rewrite operators,
  such as expansion, factorization, and generalized code motion, can be composed
  with the aim of reducing the operation count.

* Code specialisation, to maximise the impact of low level optimisation. For
  example, COFFEE can analyse the arrays, loops, and memory access pattern in
  a kernel, and then autonomously decide on the application of padding and data
  alignment, a transformation for enhancing the effectiveness of compiler
  auto-vectorisation.

COFFEE currently has one user, Firedrake, an automated system for the resolution of
partial differential equations using the finite element method. An optimisation
pipeline in COFFEE was designed to exploit a fundamental mathematical property
of finite element integration kernels, namely linearity of operators in test and
trial functions.
