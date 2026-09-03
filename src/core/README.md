# Core Layer (`src/core/`)

## Purpose
The Core module provides foundational primitives, standard data types, status enumerations, logging utilities, and timing/profiling harnesses utilized across all solver components.

## Planned Responsibilities
- **Standard Types**: Strict definitions for floating-point and integer types (e.g. `Float64`, `Int64`).
- **Solver Status Enums**: `Optimal`, `Infeasible`, `Unbounded`, `IterationLimit`, `TimeLimit`, `NumericalError`.
- **Diagnostic Logging**: Zero-cost logging macros with adjustable verbosity levels.
- **High-Resolution Timers**: Platform-independent wall-clock and CPU cycle measurement tools.

## Architectural Boundaries
- Depends strictly on the C++20 standard library.
- Must NOT depend on any higher-level layers (`model`, `numerics`, `algorithms`, `backend`, `verification`, `api`).
