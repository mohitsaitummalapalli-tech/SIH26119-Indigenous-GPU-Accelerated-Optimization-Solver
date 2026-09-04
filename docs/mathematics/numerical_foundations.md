# Mathematical Foundations: Sparse Numerical Linear Algebra Layer

**Document Version:** 1.0.0  
**Phase:** 2 (Sparse Numerical Linear Algebra + Numerical Foundations)  
**Repository:** `SIH26119-Indigenous-GPU-Accelerated-Optimization-Solver`  

---

## 1. Scope and Design Philosophy

Phase 2 establishes the native CPU numerical linear algebra primitives required for future mathematical optimization algorithms (simplex, interior-point methods, first-order methods, basis operations, presolve, and factorizations). 

The numerical layer is mathematically decoupled from:
- Problem representation and interchange formats (Phase 1 canonical model, MPS, LP)
- Future optimization algorithms (Phase 3+ simplex, interior point, QP, MILP, presolve)
- Hardware acceleration (GPU/CUDA backends)
- External third-party numerical or solver packages (Eigen, SuiteSparse, cuSPARSE, MKL, OpenBLAS, etc.)

Every operation is implemented natively in C++20 with strict mathematical semantics, deterministic behavior, checked error handling, and robust invariants.

---

## 2. Mathematical Contracts

### 2.1 Vectors

A real dense vector $x \in \mathbb{R}^n$ of dimension $n \ge 0$ is an ordered tuple of finite real numbers:
$$x = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_{n-1} \end{bmatrix}, \quad x_i \in \mathbb{R}$$

#### Vector Operations:
1. **Dot Product**:
   For vectors $x, y \in \mathbb{R}^n$:
   $$\langle x, y \rangle = x^T y = \sum_{i=0}^{n-1} x_i y_i$$
   Precondition: $\text{dim}(x) = \text{dim}(y)$. If dimensions mismatch, an explicit error is returned. For $n = 0$, $x^T y = 0.0$.

2. **AXPY Operation**:
   For vectors $x, y \in \mathbb{R}^n$ and scalar $\alpha \in \mathbb{R}$:
   $$y \leftarrow \alpha x + y \quad \iff \quad y_i \leftarrow \alpha x_i + y_i, \quad \forall i \in \{0, \dots, n-1\}$$
   Precondition: $\text{dim}(x) = \text{dim}(y)$, $\alpha, x, y$ finite.

3. **Euclidean 2-Norm**:
   $$\|x\|_2 = \sqrt{\sum_{i=0}^{n-1} x_i^2}$$
   Implemented using scaled accumulation ($\max_i |x_i|$) to prevent intermediate floating-point overflow or premature underflow on extreme finite magnitudes. For $n = 0$, $\|x\|_2 = 0.0$.

4. **Infinity Norm**:
   $$\|x\|_\infty = \max_{0 \le i < n} |x_i|$$
   For $n = 0$, $\|x\|_\infty = 0.0$.

---

### 2.2 Dense Matrices (Column-Major Storage)

A dense matrix $A \in \mathbb{R}^{m \times n}$ with $m$ rows and $n$ columns is represented in contiguous memory using **column-major order**.

#### Memory Layout:
For entry $A_{i,j} = A(i, j)$ where $0 \le i < m$ and $0 \le j < n$, the linear index in contiguous storage is:
$$\text{linear\_index}(i, j) = i + j \cdot m$$

Column-major layout is chosen to align with standard numerical linear algebra routines and basis factorizations, ensuring that matrix columns (basis vectors) reside contiguously in memory.

#### Matrix-Vector Multiplication:
For $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$:
$$y = Ax \in \mathbb{R}^m, \quad y_i = \sum_{j=0}^{n-1} A_{i,j} x_j, \quad \forall i \in \{0, \dots, m-1\}$$

- **Aliasing Contract**: In $y = Ax$, vector $x$ and output vector $y$ must not alias ($\&x \ne \&y$ and $x.\text{data}() \ne y.\text{data}()$). If aliasing is detected, an explicit error is returned.
- **Dimension Invariant**: $\text{cols}(A) = \text{dim}(x)$ and $\text{rows}(A) = \text{dim}(y)$. Dimension mismatches return an explicit `StatusCode::InvalidArgument`.

---

### 2.3 Sparse Matrices (Compressed Sparse Row - CSR)

A sparse matrix $A \in \mathbb{R}^{m \times n}$ stores only explicitly present nonzero entries in Compressed Sparse Row (CSR) format:

$$\mathcal{A} = (m, n, \text{row\_ptr}, \text{col\_idx}, \text{values})$$

- $\text{row\_ptr} \in \mathbb{N}_0^{m+1}$
- $\text{col\_idx} \in \mathbb{N}_0^{\text{nnz}}$
- $\text{values} \in \mathbb{R}^{\text{nnz}}$

#### Invariants:
1. $\text{row\_ptr}[0] = 0$
2. $\text{row\_ptr}[m] = \text{nnz}$
3. $\text{row\_ptr}[i] \le \text{row\_ptr}[i+1]$ for all $0 \le i < m$
4. Entries in row $i$ are stored in contiguous slice $k \in [\text{row\_ptr}[i], \text{row\_ptr}[i+1])$:
   $$A_{i, \text{col\_idx}[k]} = \text{values}[k]$$
5. $0 \le \text{col\_idx}[k] < n$ for all $0 \le k < \text{nnz}$
6. Within each row $i$, column indices are strictly increasing:
   $$\text{col\_idx}[k] < \text{col\_idx}[k+1], \quad \forall k \in [\text{row\_ptr}[i], \text{row\_ptr}[i+1]-1)$$

#### Triplet Construction Semantics:
When constructing a CSR matrix from a list of triplets $(i_k, j_k, v_k)$:
1. Coordinate validation: ensure $0 \le i_k < m$ and $0 \le j_k < n$ and $v_k$ is finite.
2. Deterministic sort: sort triplets primarily by row index $i$, secondarily by column index $j$.
3. Duplicate accumulation: duplicate entries at the same coordinate $(i, j)$ are accumulated by summation:
   $$A_{i,j} \leftarrow \sum_{k: (i_k, j_k) = (i, j)} v_k$$
4. Exact structural zero elimination: any accumulated entry whose value is identically zero ($v == 0.0$) is excluded from the CSR storage. Small nonzero entries (e.g., $10^{-15}$) are preserved.

#### Element Query (`get(i, j)`):
- For $i \ge m$ or $j \ge n$, returns an explicit error (`StatusCode::InvalidArgument`).
- For valid $i, j$, performs binary search on $\text{col\_idx}$ in $[\text{row\_ptr}[i], \text{row\_ptr}[i+1])$.
  - If present: returns stored $\text{values}[k]$.
  - If absent: returns exactly $0.0$.

#### Sparse Matrix-Vector Multiplication (SpMV):
$$y = Ax \in \mathbb{R}^m, \quad y_i = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot x_{\text{col\_idx}[k]}$$
Precondition: $\text{cols}(A) = \text{dim}(x)$, $\text{rows}(A) = \text{dim}(y)$, $x$ and $y$ must not alias.

#### Residual Computation:
For linear system $Ax \approx b$ with $A \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^n, b \in \mathbb{R}^m$:
$$r = b - Ax \in \mathbb{R}^m, \quad r_i = b_i - \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot x_{\text{col\_idx}[k]}$$
$$\|r\|_\infty = \max_{0 \le i < m} |r_i|$$

---

## 3. Numerical Stability & Floating-Point Standards

### 3.1 Non-Finite Value Contract
- Scalars must be finite (`std::isfinite`).
- Inputs containing `NaN`, `+Inf`, or `-Inf` return an explicit domain error (`StatusCode::InvalidArgument`).
- Norm functions return `Result<Scalar>`; if any element is non-finite, an error status is returned.
- No silent clamping, NaN-to-zero substitution, or undefined IEEE signaling propagation is allowed.

### 3.2 Semantic Tolerance Model
Floating-point comparisons use the project-wide foundational tolerance:
$$|a - b| \le \text{abs\_tol} + \text{rel\_tol} \cdot \max(1.0, |a|, |b|)$$

Foundational defaults:
$$\text{abs\_tol} = 10^{-12}, \quad \text{rel\_tol} = 10^{-12}$$

Approximate zero check:
$$\text{approx\_zero}(a) \iff |a| \le \text{abs\_tol}$$
