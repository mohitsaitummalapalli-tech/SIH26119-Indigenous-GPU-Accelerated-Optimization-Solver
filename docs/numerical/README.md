# Numerical Linear Algebra & Stability

This directory documents numerical linear algebra methods, stability considerations, and sparse matrix representations:

- **Matrix Representations**:
  - Compressed Sparse Column (CSC) and Compressed Sparse Row (CSR) storage layouts.
  - Coordinate (COO) format for dynamic matrix assembly.
- **Sparse Factorization**:
  - Sparse LU factorization with threshold partial pivoting and Markowitz strategy.
  - Cholesky and $LDL^T$ factorization for symmetric positive semi-definite systems in QP and IPM normal equations ($A \Theta A^T$).
- **Equilibration & Conditioning**:
  - Ruiz equilibration and geometric mean row/column scaling.
  - Condition number monitoring and stability thresholds ($\epsilon_{piv}$, $\epsilon_{zero}$).
