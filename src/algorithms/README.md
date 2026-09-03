# Algorithm Layer (`src/algorithms/`)

## Purpose
The Algorithm module encapsulates mathematical optimization engines for solving Linear Programming (LP), Quadratic Programming (QP), and Mixed-Integer Linear Programming (MILP) models.

## Planned Responsibilities
- **Presolve Engine**:
  - Detection of redundant constraints, empty rows/columns, singleton row bound tightening.
- **LP Algorithms**:
  - Two-Phase Revised Primal Simplex and Dual Simplex algorithms.
  - Steepest-Edge and Devex pricing strategies.
  - Primal-Dual Interior Point Method (IPM) with Mehrotra Predictor-Corrector.
- **QP Algorithms**:
  - Convex Quadratic Programming solvers via Augmented Lagrangian / IPM.
- **MILP Algorithms**:
  - Branch-and-Bound / Branch-and-Cut tree search manager.
  - Node selection, variable branching, and cutting plane management.

## Architectural Boundaries
- Consumes `src/model/` and invokes linear algebra via `src/numerics/` and abstract `src/backend/` interfaces.
- Must NOT depend on `src/api/` or specific GPU runtime implementations directly.
