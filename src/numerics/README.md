# Numerics Layer (`src/numerics/`)

## Purpose
The Numerics module provides fundamental numerical linear algebra structures, sparse matrix storage, matrix scaling, and basis factorization interfaces.

## Planned Responsibilities
- **Sparse Representations**:
  - Compressed Sparse Column (CSC) format for column-oriented basis updates and Simplex pricing.
  - Compressed Sparse Row (CSR) format for row-oriented constraint processing.
- **Dense Vectors**:
  - Cache-aligned contiguous buffer abstractions for primal/dual solution vectors and reduced costs.
- **Numerical Conditioning**:
  - Matrix scaling routines (Ruiz equilibration, geometric mean scaling).
  - Sparse LU factorization and basis inversion with dynamic Markowitz threshold pivoting.

## Architectural Boundaries
- Depends on `src/core/`.
- Must NOT depend on `src/algorithms/`, `src/io/`, or `src/api/`.
- Hardware-specific acceleration is implemented via `src/backend/`, not embedded in numerical containers.
