# Benchmarking Protocol & Methodology

This directory documents the methodology for evaluating solver performance, numerical accuracy, and scalability:

- **Benchmark Problem Sets**:
  - **Netlib LP**: Classic linear programming test problems for numerical correctness and pivot behavior.
  - **Mittelmann Benchmarks**: High-performance optimization benchmark suites for LP, QP, and MILP.
  - **MIPLIB 2017**: Standard mixed-integer linear programming collection.
- **Key Metrics**:
  - Wall-clock execution time and CPU time.
  - Simplex iteration count and IPM step count.
  - Primal and dual residual norms ($\|Ax - b\|_{\infty}$, $\|A^Ty + z - c\|_{\infty}$).
  - Relative optimality gap and Karush-Kuhn-Tucker (KKT) violation bounds.
  - Memory consumption (peak RSS).
- **Comparison & Reporting**:
  - Shifted geometric mean calculations for timing comparisons.
  - Performance profiles (Dolan-Moré profiles).
