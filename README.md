# SIH26119 — Indigenous GPU-Accelerated Optimization Solver

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](#)
[![Phase](https://img.shields.io/badge/Status-Phase_0%20(Foundation)-yellow.svg)](#)

> **Official Problem Statement SIH26119**: Development of an indigenous, high-performance mathematical optimization solver targeting Linear Programming (LP), Mixed-Integer Linear Programming (MILP), and Quadratic Programming (QP) using multicore CPU execution and measurable GPU acceleration.

---

## Current Status

> ### **PHASE 0 — Foundation only**
>
> **Important Notice on Solver Capabilities:**
> This repository is currently in **Phase 0 (Engineering Foundation & Project Constitution)**.
> - **NO** optimization algorithms (simplex, interior-point, branch-and-bound, branch-and-cut) are implemented yet.
> - **NO** LP, MILP, or QP solver engines are active.
> - **NO** CUDA solver kernels or numerical linear algebra operations are active.
> - **NO** external optimization solver libraries are linked, embedded, wrapped, or called.
>
> Phase 0 establishes the engineering constitution, architectural boundaries, modern C++20 build framework, lightweight testing harness, documentation standards, and dependency hygiene.

---

## Strict Zero-Solver Constraint

In strict adherence to the indigenous research mandate of SIH26119:

**The solver core is NOT implemented by wrapping, embedding, calling, subclassing, or delegating optimization tasks to existing commercial or open-source solver engines, including but not limited to:**
- Gurobi
- IBM ILOG CPLEX
- FICO Xpress
- HiGHS
- SCIP
- CBC / Clp
- GLPK
- OR-Tools optimization solver backends
- NVIDIA cuOpt

All mathematical optimization algorithms, numerical linear algebra routines, presolve operations, simplex/interior-point/branch-and-bound routines, and GPU kernels will be designed, implemented, and verified from mathematical first principles.

---

## Architectural Separation of Concerns

The repository is designed around strict separation of responsibilities:

1. **Model Layer (`src/model/`)**: Mathematical problem representation (variables, variable domains/types, linear/quadratic constraints, objective sense and terms, bounds).
2. **IO Layer (`src/io/`)**: Parsers and exporters for industry-standard interchange formats (MPS, LP).
3. **Numerical Layer (`src/numerics/`)**: Dense and sparse matrix data structures (CSR, CSC), vector operations, scaling routines, and linear system solvers.
4. **Algorithm Layer (`src/algorithms/`)**: Optimization algorithm state machines and drivers for LP, QP, and MILP.
5. **Backend Layer (`src/backend/`)**: Hardware acceleration interfaces abstracting Multicore CPU execution (`src/backend/cpu/`) and CUDA GPU execution (`src/backend/cuda/`).
6. **Verification Layer (`src/verification/`)**: Independent validation engines for feasibility checking, residual norm evaluation, and objective optimality bounds verification.
7. **Benchmark Layer (`benchmarks/`)**: Standardized instance suites, benchmarking orchestration scripts, and reproducible reporting pipelines.
8. **Public API (`src/api/`)**: Clean C++ programmatic interfaces exposing the solver to client applications.

For full architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Repository Structure

```
/
├── CMakeLists.txt              # Top-level CMake configuration (C++20, warnings, CTest)
├── README.md                   # Project overview, status, and constitution
├── LICENSE                     # Apache 2.0 license
├── .gitignore                  # Git ignore rules for build artifacts and IDEs
├── ARCHITECTURE.md             # Detailed layer specifications and interaction contracts
├── REQUIREMENTS.md             # Engineering requirements (Mandatory, Target, Extensibility)
├── ROADMAP.md                  # Milestone phases and execution schedule
├── CONTRIBUTING.md             # Code standards, git conventions, zero-solver rules
├── docs/                       # Mathematical foundations and technical specifications
│   ├── mathematics/            # Formulations, duality, KKT conditions, optimality theory
│   ├── algorithms/             # Simplex, IPM, Branch-and-bound algorithmic designs
│   ├── numerical/              # Numerical linear algebra, factorization, condition numbers
│   ├── gpu/                    # GPU acceleration paradigms, memory hierarchy, kernel design
│   └── benchmarks/             # Benchmarking methodology, performance metrics, suites
├── src/                        # Source code architectural boundaries
│   ├── core/                   # Basic types, status codes, logging, configuration
│   ├── model/                  # Optimization model abstractions (variables, constraints)
│   ├── io/                     # Model parsers (MPS, LP format specifications)
│   ├── numerics/               # Sparse matrices (CSR/CSC), vector operations, linear algebra
│   ├── algorithms/             # Algorithm orchestrators (LP, QP, MILP)
│   ├── backend/                # Hardware execution abstraction
│   │   ├── cpu/                # Multicore CPU backend (thread pools, vectorization)
│   │   └── cuda/               # CUDA GPU acceleration backend
│   ├── verification/           # Feasibility verification, residual checking, optimality audit
│   └── api/                    # Public solver C++ API
├── tests/                      # Testing harness
│   ├── unit/                   # Unit and infrastructure tests
│   ├── integration/            # End-to-end pipeline verification tests (future)
│   ├── numerical/              # Numerical accuracy and conditioning tests (future)
│   └── regression/             # Bug regression test suites (future)
├── benchmarks/                 # Benchmarking framework
│   ├── instances/              # Test problem instances (Mittelmann, Netlib, MIPLIB)
│   ├── scripts/                # Benchmark automation and measurement scripts
│   └── results/                # Performance comparison outputs and profiles
├── examples/                   # Usage examples and sample programs
└── scripts/                    # Developer tooling, environment setups, and CI scripts
```

---

## Building from Source

### Prerequisites
- Modern C++ compiler supporting **C++20** (GCC 11+, Clang 13+, or MSVC 2019/2022)
- **CMake** 3.25 or later
- Build system: **Ninja** (recommended) or Make

*(Note: Developer tools such as compilers and CMake installed on the system are development prerequisites, not project/runtime software dependencies).*

### Build Steps

```powershell
# 1. Configure the build directory
cmake -S . -B build -G Ninja

# 2. Compile the project
cmake --build build --config Release

# 3. Execute the infrastructure test suite
ctest --test-dir build --output-on-failure
```

---

## License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.