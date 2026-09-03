# Project Development Roadmap (SIH26119)

This document delineates the strategic development phases for building the indigenous mathematical optimization solver.

> **Status Notice**: Only **Phase 0** is currently active and established. All subsequent phases represent scheduled engineering milestones and are **NOT** yet implemented.

---

## Phase 0: Project Constitution & Engineering Foundation [CURRENT]
- [x] Establish Git repository structure and directory layer boundaries.
- [x] Create top-level CMake configuration with C++20 enforcement and strict compiler warnings.
- [x] Define engineering constitution, zero-solver constraints, and licensing.
- [x] Establish native lightweight CTest build verification harness.
- [x] Publish comprehensive architecture, requirements, and contribution guidelines.
- [x] Ensure zero external solver dependencies exist in the repository.

---

## Phase 1: Problem Representation & Interchange Formats [SCHEDULED]
- [ ] Implement `src/model/` data structures: variables, bounds, linear rows, quadratic terms, objective representation.
- [ ] Implement `src/io/` parsers: standard MPS (fixed and free format) and LP format parser.
- [ ] Implement serialization routines to export models to standard MPS and LP files.
- [ ] Create unit tests validating parser accuracy across standard Netlib model files.

---

## Phase 2: Numerical Linear Algebra & Sparse Foundations [SCHEDULED]
- [ ] Implement `src/numerics/`: Dense vector operations, Compressed Sparse Row (CSR), and Compressed Sparse Column (CSC) matrices.
- [ ] Implement Sparse Matrix-Vector Multiplication (SpMV) kernels for CPU.
- [ ] Implement matrix conditioning and scaling algorithms (Ruiz equilibration, $L_{\infty}$ scaling).
- [ ] Implement sparse LU factorization with Markowitz threshold pivoting for basis inversion and stability.

---

## Phase 3: Linear Programming (LP) Core Algorithms [SCHEDULED]
- [ ] Implement standard form transformation and slack variable generation.
- [ ] Implement Two-Phase Primal Simplex and Dual Simplex algorithms.
- [ ] Implement Steepest-Edge and Devex pricing strategies.
- [ ] Implement Primal-Dual Interior Point Method (IPM) with Mehrotra predictor-corrector.
- [ ] Implement `src/verification/` feasibility and residual audit modules ($Ax - b$ residuals, KKT conditions).

---

## Phase 4: Hardware Acceleration Backends (Multicore CPU & CUDA GPU) [SCHEDULED]
- [ ] Define abstract hardware acceleration interface in `src/backend/`.
- [ ] Implement Multicore CPU backend utilizing thread pools and SIMD vectorization.
- [ ] Implement CUDA GPU backend in `src/backend/cuda/`: asynchronous host-device memory transfers, GPU SpMV, and parallel linear system solve routines.
- [ ] Benchmark CPU vs. GPU performance crossover thresholds across sparse problem dimensions.

---

## Phase 5: Quadratic Programming (QP) & Mixed-Integer Programming (MILP) [SCHEDULED]
- [ ] Implement Convex QP algorithm (Augmented Lagrangian and Primal-Dual IPM extensions for quadratic terms).
- [ ] Implement MILP Branch-and-Bound search tree manager.
- [ ] Implement node selection strategies (Best-Bound, Depth-First) and variable branching rules (Most Infeasible, Pseudocost).
- [ ] Implement basic cutting planes (Gomory fractional cuts).

---

## Phase 6: Benchmarking, Validation & Performance Hardening [SCHEDULED]
- [ ] Ingest standard benchmark suites (Netlib LP, MIPLIB 2017, Mittelmann benchmark instances).
- [ ] Build automated benchmarking pipeline measuring execution time, memory footprint, iteration counts, and residual norms.
- [ ] Conduct comparative performance profiling against reference solvers.
- [ ] Hardening for numerical edge cases, degeneracy, and cycling prevention (Bland's rule, perturbation).
