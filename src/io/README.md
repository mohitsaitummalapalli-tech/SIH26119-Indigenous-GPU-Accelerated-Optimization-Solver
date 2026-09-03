# IO Layer (`src/io/`)

## Purpose
The IO module provides robust readers and writers for standard mathematical optimization interchange file formats.

## Planned Responsibilities
- **MPS Parser & Writer**:
  - Support for fixed and free MPS formats.
  - Sections: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, and quadratic extensions (`QUADOBJ`, `QCMATRIX`).
- **LP Parser & Writer**:
  - Human-readable algebraic CPLEX LP format parser and exporter.
- **Format Validation**: Detailed diagnostic reporting for malformed input files with line-number precision.

## Architectural Boundaries
- Reads input streams/files and instantiates problem structures in `src/model/`.
- Depends on `src/core/` and `src/model/`.
- Must NOT depend on `src/algorithms/`, `src/backend/`, or `src/verification/`.
