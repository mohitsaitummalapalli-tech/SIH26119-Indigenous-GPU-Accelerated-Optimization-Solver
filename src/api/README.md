# Public API Layer (`src/api/`)

## Purpose
The API layer defines the public-facing C++ interfaces through which external client programs interact with the solver.

## Planned Responsibilities
- **Solver Interface**:
  - Unified entry point for model loading, option configuration, and solve execution.
  - Asynchronous interruption hooks and progress query callbacks.
- **Result Containers**:
  - Solution vectors (primal $x$, dual $y$, reduced costs $z$).
  - Termination status, solve wall-clock time, iteration counts, and optimality certificates.
- **Options Interface**:
  - Configuration of tolerances ($\epsilon_{feas}, \epsilon_{dual}, \epsilon_{opt}$), iteration limits, time limits, and backend selection (CPU vs. GPU).

## Architectural Boundaries
- The public gateway to the solver core.
- Exposes high-level abstractions while encapsulating numerical and algorithmic implementation details.
