# Multicore CPU Backend (`src/backend/cpu/`)

## Purpose
The CPU backend provides parallelized and vectorized linear algebra operations optimized for modern multicore CPU architectures.

## Planned Responsibilities
- **Thread Scheduling**: Thread pool orchestrator for parallelizing row/column operations and independent branch-and-bound subproblems.
- **SIMD Vectorization**: Vectorized dot products, vector additions, and dense/sparse kernel inner loops.
- **Cache-Optimized Traversals**: Blocking and cache-conscious memory layouts for large matrix operations.

## Architectural Boundaries
- Implements abstract interfaces defined in `src/backend/`.
- Zero CUDA dependencies.
