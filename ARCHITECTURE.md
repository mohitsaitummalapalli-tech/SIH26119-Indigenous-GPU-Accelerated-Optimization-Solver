# System Architecture Specification

## 1. Architectural Philosophy & Principles

The **SIH26119 Optimization Solver** is architected to provide an indigenous, mathematically rigorous, high-performance optimization engine. The design follows strict engineering tenets:

- **C++ as the Primary Core**: All performance-critical data structures, algorithms, and linear algebra routines are written in ISO standard C++20.
- **Strict Hardware Isolation**: Hardware-specific paradigms (such as CUDA GPU execution and multicore CPU thread pools) are strictly abstracted behind unified backend interfaces. Higher-level algorithms interact only with abstract linear algebra and vector execution contracts.
- **Python as Ancillary Tooling Only**: If Python is utilized in later phases, its role is strictly confined to high-level benchmarking scripts, test orchestration, or external bindings. Python will **never** be used for internal numerical or linear algebra bottlenecks.
- **Minimal Dependencies**: Zero third-party optimization solver dependencies. External utility libraries are forbidden unless genuinely justified and peer-reviewed.
- **No Extraneous Frameworks**: No GUIs, no web frontends, no machine learning dependencies, and no blockchain integrations.
- **Testability & Determinism**: All algorithmic components are designed for deterministic execution, state inspection, and reproducible mathematical validation.

---

## 2. Repository Layer Map

The physical directory hierarchy in `src/` directly reflects the logical separation of concerns:

```
src/
├── core/           # Foundational utilities, error codes, configuration, logging
├── model/          # Problem formulations, variables, constraints, objectives
├── io/             # File parsers and serializers (MPS, LP format specifications)
├── numerics/       # Sparse matrices, dense vectors, linear systems, scaling
├── algorithms/     # Optimization engines (LP, QP, MILP orchestration)
├── backend/        # Hardware execution abstractions
│   ├── cpu/        # Multicore CPU thread scheduling and vectorization
│   └── cuda/       # CUDA GPU memory management and kernel acceleration
├── verification/   # Feasibility, residual, and optimality certificates
└── api/            # Public-facing C++ solver API
```

---

## 3. Detailed Layer Specifications

### 3.1. Core Layer (`src/core/`)
- **Responsibility**: Provides fundamental data types, standardized solver return codes, memory allocation strategies, timing harnesses, and logging abstractions.
- **Key Concepts**:
  - Precision types (e.g., standard 64-bit IEEE 754 `Float64`, integer `Int64`).
  - Solver status enumerations (`Optimal`, `Infeasible`, `Unbounded`, `IterationLimit`, `TimeLimit`, `NumericalFailure`).
  - Diagnostic and logging channels with zero overhead when disabled.

### 3.2. Model Layer (`src/model/`)
- **Responsibility**: Houses problem formulation primitives and immutable problem descriptions independent of algorithmic solvers.
- **Key Concepts**:
  - **Variables**: Identifiers, column indices, lower/upper bounds, variable types (Continuous, Binary, Integer).
  - **Constraints**: Linear rows, lower/upper bounds, senses (Equal, Less-than-or-equal, Greater-than-or-equal, Ranged).
  - **Objective**: Linear and quadratic objective coefficients, optimization direction (Minimize, Maximize), constant offset.
  - **Model Container**: Aggregates variables, constraints, and objective into a coherent optimization problem instance.

### 3.3. IO Layer (`src/io/`)
- **Responsibility**: Ingestion and serialization of standard mathematical programming interchange formats.
- **Key Concepts**:
  - **MPS Parser**: Full support for fixed and free MPS formats (NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ/QCMATRIX).
  - **LP Parser**: Human-readable format parsing.
  - Clean separation: Parsers construct a `Model` object without coupling to solver or numerical internal structures.

### 3.4. Numerical Layer (`src/numerics/`)
- **Responsibility**: High-performance linear algebra and matrix/vector data representations.
- **Key Concepts**:
  - **Sparse Matrix Formats**: Compressed Sparse Row (CSR) for row-wise operations, Compressed Sparse Column (CSC) for column-wise operations and simplex basis representations.
  - **Dense Vectors**: Memory-aligned continuous numerical buffers.
  - **Numerical Utilities**: Matrix scaling (Ruiz equilibration, geometric scaling), condition number estimators, basis factorization stubs.

### 3.5. Algorithm Layer (`src/algorithms/`)
- **Responsibility**: Algorithmic state machines implementing pure mathematical optimization strategies.
- **Key Concepts**:
  - **LP Solvers**: Revised Simplex method (Primal and Dual) with steep-edge pricing; Primal-Dual Interior Point Method (IPM) with Mehrotra predictor-corrector.
  - **QP Solvers**: Convex Quadratic Programming via Augmented Lagrangian or IPM.
  - **MILP Solvers**: Branch-and-Bound / Branch-and-Cut tree managers, pseudocost branching, node selection, cutting plane management.
  - **Presolve Engine**: Row/column singleton removal, bound tightening, redundant constraint detection.

### 3.6. Backend Layer (`src/backend/`)
- **Responsibility**: Abstract execution boundary decoupling algorithmic math from hardware specifics.
- **Key Concepts**:
  - **Unified Compute Interface**: Defines abstract vector operations (axpy, dot, norm), matrix-vector multiplication (SpMV), and linear system solve calls.
  - **CPU Backend (`src/backend/cpu/`)**: Cache-friendly multicore implementations using thread pools, SIMD instruction sets, and CPU BLAS/LAPACK-style routines.
  - **CUDA GPU Backend (`src/backend/cuda/`)**: Asynchronous stream management, device memory pools, GPU SpMV, and parallel linear system solve operations for massive systems where data transfer overhead is amortized.

### 3.7. Verification Layer (`src/verification/`)
- **Responsibility**: Independent mathematical verification and solution certification.
- **Key Concepts**:
  - **Feasibility Checking**: Verifies primal and dual bounds against user-configured tolerances ($\epsilon_{feas}$).
  - **Residual Norm Evaluation**: Computes $\|Ax - b\|_{\infty}$, $\|A^T y + z - c\|_{\infty}$, and complementarity gap $x^T z$.
  - **Optimality Auditing**: Ensures claimed optimal solutions satisfy Karush-Kuhn-Tucker (KKT) conditions or duality gap bounds.

### 3.8. API Layer (`src/api/`)
- **Responsibility**: Exposes clean, stable C++ entry points for embedding the solver into applications.
- **Key Concepts**:
  - Context and configuration handle management.
  - Asynchronous solve interruption and progress callback handlers.
  - High-level solve invocation: `Solver::solve(const Model&, const SolverOptions&) -> SolverResult`.

---

## 4. Layer Interaction & Data Flow Boundaries

```
[ Model File (MPS/LP) ]
         │
         ▼ (IO Layer)
   [ Problem Model ] ──────────┐
         │                     │
         ▼ (Presolve/Numerics) │
[ Scaled Standard Form ]       │
         │                     │
         ▼ (Algorithm Layer)   │ (Original Model Reference)
   [ Solver State ]            │
         │                     │
         ▼ (Backend Abstraction)│
   [ CPU / CUDA Kernels ]      │
         │                     │
         ▼ (Raw Solution)      │
 [ Postsolve Expansion ]       │
         │                     │
         ▼                     ▼
   [ Verification Layer ] ◄────┘
         │ (Feasibility & KKT Check)
         ▼
    [ Final Result / API ]
```

1. **Top-Down Dependency**: The `model` layer does not know about `algorithms` or `backend`. The `numerics` layer does not know about high-level `algorithms`.
2. **Backend Agnosticism**: Algorithms communicate with the `backend` solely via abstract vector and matrix operators. The algorithm does not contain CUDA runtime calls directly.
3. **Independent Verification**: The `verification` layer takes the original unprocessed `Model` and the candidate solution, computing residuals from scratch without trusting solver internal state.
