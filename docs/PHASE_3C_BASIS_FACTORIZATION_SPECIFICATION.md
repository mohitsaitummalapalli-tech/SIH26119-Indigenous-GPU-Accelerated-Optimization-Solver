# Phase 3C: Basis Factorization Layer — Mathematical and Numerical Specification

> **Authoritative Mathematical, Numerical, and Architectural Specification**
> **Repository:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver
> **Status:** Phase 3C Specification Gate
> **Authoritative Baseline Commit:** `4caf928997c6086f56264295d815f982f2d0022e`
> **Mandatory Scope Declaration:** *Phase 3C defines the numerical factorization and linear solve layer ($B x = r$ and $B^T y = r$) for the simplex basis matrix. It does NOT implement factorization code, simplex pricing, pivots, ratio tests, Phase I, Phase II, primal simplex, dual simplex, GPU acceleration, or external solver interfaces.*

---

## 1. Architectural Boundary & Repository Context

The linear programming solver architecture enforces strict layered isolation across Phases:

```
+-----------------------------------------------------------------------------------+
| Phase 3A: LP Standardization Layer (lp_standard_form.hpp)                         |
| min c_bar^T x_bar + c0_bar  s.t.  A_bar x_bar = b_bar,  x_bar >= 0                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3B: Basis Representation & State Layer (basis.hpp, basic_solution.hpp)     |
| Structural Invariants: m <= n, Bijective row <-> col map, Versioning              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+===================================================================================+
| Phase 3C: Basis Factorization Layer (SPECIFICATION ONLY)                          |
| Solves B x = r (FTRAN) and B^T y = r (BTRAN) via P B = L U                        |
| Singularity detection, condition estimation, zero-allocation solves               |
+===================================================================================+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3D: Simplex Optimization Engine (FUTURE)                                    |
| Revised Simplex iterations, pricing, ratio tests, Phase I / Phase II              |
+-----------------------------------------------------------------------------------+
```

### 1.1 Source Grounding
This specification builds strictly upon existing committed Phase 1, Phase 2, Phase 3A, and Phase 3B contracts:
- [`src/solver/lp/basis.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis.hpp): Authoritative structural basis state, $O(1)$ query tables, and monotonically increasing `version()`.
- [`src/solver/lp/basis_matrix_view.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis_matrix_view.hpp): Non-owning logical view over $A \in \mathbb{R}^{m \times n}$ where column $k$ is $A_{:, B(k)}$.
- [`src/numerics/sparse_matrix.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/sparse_matrix.hpp): Zero-allocation CSR SpMV and residual contracts (`multiply`, `residual`).
- [`src/numerics/dense_vector.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/dense_vector.hpp): Zero-allocation dense vector layer with `is_finite_scalar` guarantees.
- [`src/numerics/tolerances.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/tolerances.hpp): Authoritative `approx_equal` and `approx_zero` semantics.
- [`src/core/status.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/core/status.hpp): Native error and result types. Numerical failure conditions map to `StatusCode::NumericalFailure`.

---

## 2. Mathematical Definition of the Basis Matrix & State Lifecycle

### 2.1 The Basis Matrix $B$
Let the standardized constraint system be $A x = b$ with $A \in \mathbb{R}^{m \times n}$, $m \le n$. Given a structurally valid [`Basis`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis.hpp) with column mapping $B(0), B(1), \dots, B(m-1)$, the basis matrix $B$ is the square $m \times m$ matrix formed by selecting the basic columns of $A$ in row order:

$$B = \begin{bmatrix} A_{:, B(0)} & A_{:, B(1)} & \cdots & A_{:, B(m-1)} \end{bmatrix} \in \mathbb{R}^{m \times m}$$

Entry-wise:
$$B_{i, k} = A_{i, B(k)}, \quad \forall i \in \{0, \dots, m-1\}, \; k \in \{0, \dots, m-1\}$$

### 2.2 Factorization Lifecycle States
The factorization object maintains an explicit, observable lifecycle state:

```
           +------------------+
           |   UNFACTORIZED   |<--------------------+
           +------------------+                     |
                     |                              |
            factorize(view)                         |
                     |                              |
         +-----------+-----------+                  |
         |                       |                  |
         v                       v                  |
   [Success]               [Failure]                |
         |                       |                  |
         v                       v                  |
  +--------------+     +--------------------+       |
  |   FACTORED   |     | NUMERICALLY_       |       |
  +--------------+     | SINGULAR / INVALID |       |
         |             +--------------------+       |
         |                                          |
  basis.replace_basic_variable()                    |
  (basis.version() > fact.basis_version)            |
         |                                          |
         v                                          |
  +--------------+                                  |
  |    STALE     |----------------------------------+
  +--------------+         (triggers refactorization)
```

1. **`Unfactorized`**: Initial state prior to factorization, or following an explicit reset.
2. **`Factored`**: Factorization $P B = L U$ succeeded. `fact.basis_version == basis.version()`. Both forward solve ($B x = r$) and dual transpose solve ($B^T y = r$) are valid.
3. **`NumericallySingular`**: Factorization detected that $\min_k |U_{k, k}| \le \varepsilon_{\text{sing}}$, encountered non-finite arithmetic, or suffered pivot breakdown. Solves are rejected with `StatusCode::NumericalFailure`.
4. **`Invalid`**: Dimensions or internal invariants corrupted.
5. **`Stale`**: The underlying `Basis` has pivoted (`basis.version() > fact.basis_version`). Any call to `solve()` or `solve_transpose()` is strictly rejected with `StatusCode::InconsistentModel`.

### 2.3 Strict Status Code Semantics
> [!IMPORTANT]
> **Status Code Separation Rule:**
> A structurally valid LP model whose basis is numerically singular or ill-conditioned is **NOT** a modeling defect.
> - `StatusCode::NumericalFailure`: Exclusively reserved for numerically singular basis matrices, unacceptable pivots, factorization breakdown, and solve residual failures.
> - `StatusCode::InconsistentModel` / `StatusCode::InvalidArgument`: Strictly reserved for structural dimension mismatches, stale basis versions, or illegal caller parameters.

### 2.4 Strict Version Invariant
> $$\text{factorization.basis\_version} \equiv \text{basis.version()}$$
> A factorization created for basis version $v$ **MUST NOT and CANNOT** be silently executed for basis version $v+1$.

---

## 3. Factorization Method & Numerical Stability Strategy

### 3.1 Initial Implementation Convention: Row-Only Partial Pivoting ($P B = L U$)
For the initial Phase 3C implementation contract, the factorization is strictly:

$$P B = L U$$

- **Row Permutation Only:** $Q = I$. Column permutations are not supported in the initial implementation. Future sparse column-permuted variants (e.g. Markowitz $P B Q = L U$) are documented as future extension points for Phase 3D.
- **Unit Lower Triangular $L$:** $L_{ii} = 1.0$.
- **Upper Triangular $U$:** $U_{ij} = 0$ for $i > j$.

### 3.2 Removal of Overclaims: Realistic Numerical Stability
"partial pivoting is the selected baseline numerical-stability strategy; it does not provide an unconditional error guarantee for every matrix."

Phase 3C explicitly distinguishes:
1. **Practical numerical robustness:** In optimization practice, pivot growth is almost always modest ($\rho_m \ll 10^3$) for typical basis matrices.
2. **Backward-error verification:** Every solve undergoes an a posteriori normwise backward-residual test $\|B \hat{x} - \text{rhs}\|_\infty \le \tau_{\text{resid}} (\|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty)$ to guarantee reliability rather than relying on an a priori assumption.
3. **Worst-case pivot growth:** Theoretical pivot growth $\rho_m = \frac{\max_{i,j,k} |M_{i,j}^{(k)}|}{\max_{i,j} |B_{i,j}|}$ can grow as large as $2^{m-1}$ (e.g. Wilkinson matrices).
4. **Formal stability guarantees:** Row partial pivoting formally guarantees that every subdiagonal element of $L$ satisfies $|L_{ij}| \le 1.0$ ($i > j$) as a direct consequence of largest-magnitude pivot selection in column $k$. It does not guarantee small backward error if exponential pivot growth occurs.


---

## 4. Complete LU Mathematics & Solve Derivations ($P B = L U$)

### 4.1 Matrix Decomposition Structure
The factorization computes a row permutation matrix $P \in \mathbb{R}^{m \times m}$, a unit lower triangular matrix $L \in \mathbb{R}^{m \times m}$, and an upper triangular matrix $U \in \mathbb{R}^{m \times m}$ such that:

$$P B = L U$$

- $P$: Row permutation matrix ($P e_i = e_{\pi_r(i)}$ where $\pi_r(i)$ is the row swapped into pivot position $i$).
- $P^{-1} = P^T$: Unpermutation matrix.
- $L$: Unit lower triangular:
  $$L = \begin{bmatrix}
  1 & 0 & \cdots & 0 \\
  l_{1, 0} & 1 & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  l_{m-1, 0} & l_{m-1, 1} & \cdots & 1
  \end{bmatrix}, \quad |l_{i, j}| \le 1.0 \quad (\forall i > j)$$
- $U$: Upper triangular:
  $$U = \begin{bmatrix}
  u_{0, 0} & u_{0, 1} & \cdots & u_{0, m-1} \\
  0 & u_{1, 1} & \cdots & u_{1, m-1} \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & u_{m-1, m-1}
  \end{bmatrix}$$

---

### 4.2 Derivation of Primal Solve: $B x = \text{rhs}$ (FTRAN)

We wish to solve for $x \in \mathbb{R}^{m}$ given RHS $\text{rhs} \in \mathbb{R}^m$:

$$B x = \text{rhs}$$

1. Multiply by row permutation $P$:
   $$P B x = P \text{rhs}$$
2. Substitute $P B = L U$:
   $$L U x = P \text{rhs}$$
3. **Step 1: Permute RHS by Row Permutation $P$:**
   $$z = P \text{rhs} \iff z_i = \text{rhs}_{\pi_r(i)}, \quad \forall i \in \{0, \dots, m-1\}$$
4. **Step 2: Forward Substitution ($L w = z$):**
   Since $L$ is unit lower triangular ($L_{ii} = 1$):
   $$w_i = z_i - \sum_{j=0}^{i-1} L_{i, j} w_j, \quad i = 0, 1, \dots, m-1$$
5. **Step 3: Backward Substitution ($U x = w$):**
   Since $U$ is upper triangular:
   $$x_i = \frac{w_i - \sum_{j=i+1}^{m-1} U_{i, j} x_j}{U_{i, i}}, \quad i = m-1, m-2, \dots, 0$$
   Here $x$ is the final primal solution.

---

### 4.3 Derivation of Dual / Transpose Solve: $B^T y = \text{rhs}$ (BTRAN)

We wish to solve for $y \in \mathbb{R}^m$ given dual RHS $\text{rhs} \in \mathbb{R}^m$:

$$B^T y = \text{rhs}$$

1. Transpose the factorization equation $P B = L U$:
   $$(P B)^T = (L U)^T \implies B^T P^T = U^T L^T$$
2. Multiply on the right by $P$ (since $P^T P = I$):
   $$B^T = U^T L^T P$$
3. Substitute $B^T$ into the equation $B^T y = \text{rhs}$:
   $$U^T L^T P y = \text{rhs}$$
4. **Step 1: Forward Substitution on Lower Triangular $U^T$ ($U^T w = \text{rhs}$):**
   $U^T$ is lower triangular with diagonal entries $U_{ii}$.
   $$w_i = \frac{\text{rhs}_i - \sum_{j=0}^{i-1} U_{j, i} w_j}{U_{i, i}}, \quad i = 0, 1, \dots, m-1$$
5. **Step 2: Backward Substitution on Unit Upper Triangular $L^T$ ($L^T v = w$):**
   $L^T$ is unit upper triangular ($L_{ii} = 1$).
   $$v_i = w_i - \sum_{j=i+1}^{m-1} L_{j, i} v_j, \quad i = m-1, m-2, \dots, 0$$
6. **Step 3: Row Unpermutation via $P^T$ ($y = P^T v$):**
   We have $P y = v$. Multiplying by $P^T$ gives:
   $$y = P^T v \iff y_{\pi_r(i)} = v_i, \quad \forall i \in \{0, \dots, m-1\}$$
   Explicitly in vector indexing:
   $$y[\pi_r(i)] = v[i], \quad \forall i \in \{0, \dots, m-1\}$$

> [!IMPORTANT]
> **Summary of Transpose Sequencing ($P B = L U$):**
> - FTRAN ($B x = \text{rhs}$): $\text{rhs} \xrightarrow{P} z \xrightarrow{L^{-1}} w \xrightarrow{U^{-1}} x$
> - BTRAN ($B^T y = \text{rhs}$): $\text{rhs} \xrightarrow{U^{-T}} w \xrightarrow{L^{-T}} v \xrightarrow{P^T} y$
> The final unpermutation step in BTRAN is $y = P^T v$.

---

## 5. Pivoting Policy & Deterministic Tie-Breaking

### 5.1 Standard Row Partial Pivoting
At elimination step $k \in \{0, \dots, m-1\}$ on active working submatrix $M^{(k)}$:
1. Search for maximum column magnitude:
   $$p = \operatorname{argmax}_{i \ge k} |M_{i, k}^{(k)}|$$
2. **Deterministic Tie-Breaking:**
   If multiple candidate rows achieve the exact maximum magnitude:
   $$p^* = \min \left\{ p \in \{k, \dots, m-1\} : |M_{p, k}^{(k)}| = \max_{i \ge k} |M_{i, k}^{(k)}| \right\}$$
   Ties are broken strictly by **smallest row index $p$**.

### 5.2 Pivot Acceptance Policy
The pivot candidate magnitude $|M_{p^*, k}^{(k)}|$ is evaluated against separate criteria:
1. **Non-Finite Pivot:** If $\text{isnan}(M_{p^*, k}^{(k)})$ or $\text{isinf}(M_{p^*, k}^{(k)})$, abort immediately and return `StatusCode::NumericalFailure`.
2. **Zero / Sub-Threshold Pivot:** If $|M_{p^*, k}^{(k)}| \le \varepsilon_{\text{sing}}$ (where $\varepsilon_{\text{sing}} = 10^{-12}$ by default):
   The matrix is numerically rank-deficient. Abort factorization and return `StatusCode::NumericalFailure`.
3. **Multiplier Magnitude Guarantee:**
   Because row $p^*$ is swapped with row $k$, the multiplier eliminated in row $i > k$ is:
   $$L_{i, k} = \frac{M_{i, k}^{(k)}}{M_{p^*, k}^{(k)}}$$
   Since $|M_{p^*, k}^{(k)}| \ge |M_{i, k}^{(k)}|$ for all $i \ge k$, this guarantees $|L_{i, k}| \le 1.0$.

---

## 6. Condition Number Estimate as a Diagnostic Signal

### 6.1 Role of Condition Number
- The condition number estimate $\kappa^*(B) = \|B\|_1 \|B^{-1}\|_1$ (computed via the Hager-Higham 1-norm algorithm) is an **operational diagnostic metric**.
- **Crucial Policy Rule:** The solver **MUST NOT** reject a solve or declare a solve invalid based on $\kappa^*(B) > 10^{13}$ alone.
- Ill-conditioned triangular or diagonal systems can often be solved with acceptable backward residual.
- **Operational Trigger:** A high condition number estimate ($\kappa^*(B) > 10^{13}$) acts as a signal to:
  1. Emit an internal numerical warning.
  2. Schedule basis refactorization or basis repair in Phase 3D.
  3. Declare `StatusCode::NumericalFailure` **ONLY** if combined with solve residual failure.

---

## 7. Solve and Factorization Residual Criteria

### 7.1 Normwise Backward-Residual Criterion for Solves
For computed primal solution $\hat{x}$ ($B \hat{x} \approx \text{rhs}$), success is declared if and only if:

$$\|B \hat{x} - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty \right)$$

For computed dual solution $\hat{y}$ ($B^T \hat{y} \approx \text{rhs}$), success is declared if and only if:

$$\|B^T \hat{y} - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{y}\|_\infty + \|\text{rhs}\|_\infty \right)$$

- **Default Tolerance:** $\tau_{\text{resid}} = 10^{-8}$.
- **Zero-Denominator Handling:** If $(\|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty) == 0.0$, the residual test requires:
  $$\|B \hat{x} - \text{rhs}\|_\infty == 0.0$$
- If the backward residual exceeds the threshold, the solve returns `StatusCode::NumericalFailure`.

### 7.2 Scale-Aware Factorization Check
During diagnostic audits or testing, the factorization residual $\|P B - L U\|_\infty$ is validated using a scale-aware criterion:

$$\|P B - L U\|_\infty \le \tau_{\text{fact}} \left( \|P B\|_\infty + \|L U\|_\infty \right)$$

- **Default Tolerance:** $\tau_{\text{fact}} = 10^{-12}$.
- **Zero-Denominator Handling:** If $(\|P B\|_\infty + \|L U\|_\infty) == 0.0$, the test requires:
  $$\|P B - L U\|_\infty == 0.0$$
- Exact floating-point equality ($P B == L U$) is never expected or asserted.

---

## 8. Scaling Strategy

- **Phase 3C Default:** **Unscaled Factorization ($D_r = I, D_c = I$)**.
  Standardized matrices from Phase 3A already normalize RHS signs. Eliminating artificial scaling avoids introducing scaling roundoff into the initial baseline.
- **Extension Point:** If row scaling $D_r$ is enabled in future iterations:
  - $(D_r B) x = D_r \text{rhs} \implies \tilde{B} x = \tilde{\text{rhs}}$
  - $(B^T D_r) (D_r^{-1} y) = \text{rhs} \implies \tilde{B}^T \tilde{y} = \text{rhs}$, then $y = D_r \tilde{y}$.

---

## 9. Storage Model & Zero-Allocation Solve Contracts

### 9.1 Storage Structure

```cpp
struct FactorizationStorage {
    Dimension m{0};                       ///< Basis dimension m x m
    std::vector<Scalar> lu_data;         ///< Dense m x m column-major storage
    std::vector<Index> row_perm;         ///< pi_r: length m (row permutation P)
    std::vector<Index> row_perm_inv;     ///< pi_r_inv: length m (P^T mapping)

    // Diagnostic metrics
    Scalar max_growth{1.0};
    Scalar condition_estimate{1.0};
    uint64_t basis_version{0};
    bool is_factored{false};
};
```

### 9.2 Zero-Allocation Solve API

```cpp
namespace sih26119 {

struct FactorizationTolerances {
    Scalar singularity_tol{1e-12};   ///< Min acceptable diagonal pivot |U_kk|
    Scalar residual_tol{1e-8};       ///< Relative solve backward error ceiling
    Scalar max_growth_tol{1e12};     ///< Maximum permissible pivot growth
    Scalar condition_ceiling{1e13};  ///< Operational ill-conditioning signal
};

class BasisFactorization {
public:
    BasisFactorization() = default;

    /**
     * @brief Performs row partial pivoting LU factorization: P B = L U
     *
     * @param basis_view Non-owning view over SparseMatrix A and Basis.
     * @param tols Configurable numerical tolerances.
     * @return Status::ok() on success, or StatusCode::NumericalFailure upon breakdown.
     */
    [[nodiscard]] Status factorize(
        const BasisMatrixView& basis_view,
        FactorizationTolerances tols = FactorizationTolerances{});

    /**
     * @brief Solves B * x = rhs (FTRAN) with ZERO dynamic heap allocations.
     *
     * Preconditions:
     * - is_factored() == true
     * - basis.version() == basis_version()
     * - rhs.size() == m
     * - solution.size() == m
     * - scratch.size() >= m
     * - Strict pairwise distinctness (no aliasing between rhs, solution, scratch).
     */
    [[nodiscard]] Status solve(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept;

    /**
     * @brief Solves B^T * y = rhs (BTRAN) with ZERO dynamic heap allocations.
     *
     * Preconditions:
     * - Same as solve().
     */
    [[nodiscard]] Status solve_transpose(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept;

    [[nodiscard]] bool is_factored() const noexcept;
    [[nodiscard]] uint64_t basis_version() const noexcept;
    [[nodiscard]] Dimension dimension() const noexcept;
    [[nodiscard]] Scalar condition_estimate() const noexcept;
    [[nodiscard]] Scalar pivot_growth() const noexcept;
};

} // namespace sih26119
```

---

## 10. Independent Numerical Oracle Architecture

### 10.1 Primary Oracle: Gaussian Elimination with Complete Pivoting
The reference oracle for Phase 3C tests will be an independently authored struct [`CompletePivotingOracle`]:
1. **Independent Implementation:** Does NOT call `BasisFactorization` and does NOT use the same pivot selection code.
2. **Complete Pivoting:** At each step $k$, searches the entire $(m-k) \times (m-k)$ active submatrix for the global maximum:
   $$(p, q) = \operatorname{argmax}_{i \ge k, j \ge k} |M_{i, j}^{(k)}|$$
3. **Capabilities:** Solves both $B x = \text{rhs}$ and $B^T y = \text{rhs}$ independently.
4. **Independent Verification:** Computes backward residuals independently using full 80-bit or independent double calculations.
5. **Cramer's Rule:** Used strictly as an auxiliary analytical sanity check for $m \le 2$.

---

## 11. Concrete Test Matrix (24 Test Fixtures)

| Test ID | Test Category | Mathematical Invariant Verified |
| :--- | :--- | :--- |
| `TEST-FACT-01` | Degenerate $0 \times 0$ | $m=0$ basis: trivially factored, solve on empty vector succeeds |
| `TEST-FACT-02` | Scalar $1 \times 1$ | $B = [5.0]$, solves $5x = 10 \implies x=2$, $5y = 15 \implies y=3$ |
| `TEST-FACT-03` | Diagonal Matrix | $B = \operatorname{diag}(2, -4, 5)$, $L=I$, $U=B$, solves match elementwise reciprocals |
| `TEST-FACT-04` | Upper Triangular | $B = U$, $L=I$, backward substitution exact match |
| `TEST-FACT-05` | Lower Triangular | $B = L$, $U=I$, forward substitution exact match |
| `TEST-FACT-06` | Permutation Matrix | $B = P$, $L=I, U=I$, pure index shuffle on FTRAN and BTRAN |
| `TEST-FACT-07` | Dense Nonsingular $3 \times 3$ | Random well-conditioned $3 \times 3$, verified against complete pivoting oracle |
| `TEST-FACT-08` | Dense Nonsingular $5 \times 5$ | Random well-conditioned $5 \times 5$, verified against complete pivoting oracle |
| `TEST-FACT-09` | Sparse Banded $10 \times 10$ | Tridiagonal / banded matrix, verifies zero-allocation solve |
| `TEST-FACT-10` | Negative Pivots | System requiring negative pivots, verifies sign preservation |
| `TEST-FACT-11` | Mixed-Sign Entries | Matrix with large positive and negative numbers |
| `TEST-FACT-12` | Exact Singular (Zero Row) | Row of zeros $\implies$ returns `StatusCode::NumericalFailure` |
| `TEST-FACT-13` | Exact Singular (Col Dep) | Linearly dependent columns $\implies$ returns `StatusCode::NumericalFailure` |
| `TEST-FACT-14` | Numerically Singular | Pivot $|U_{kk}| \le \varepsilon_{\text{sing}}$, returns `StatusCode::NumericalFailure` |
| `TEST-FACT-15` | Ill-Conditioned Matrix | High condition number, verifies solve succeeds if residual passes |
| `TEST-FACT-16` | Badly Scaled Matrix | Entries ranging from $10^{-6}$ to $10^{6}$, verifies backward residual test |
| `TEST-FACT-17` | Multiple RHS Solves | Successive solves with distinct RHS vectors using same factorization |
| `TEST-FACT-18` | Transpose Duality | Verifies that $y^T (B x) == x^T (B^T y)$ within machine tolerance |
| `TEST-FACT-19` | NaN / Inf Input Rejection | Rejects non-finite entries in RHS with `StatusCode::InvalidArgument` |
| `TEST-FACT-20` | Workspace Aliasing | Rejects `&solution == &rhs` or `&scratch == &solution` |
| `TEST-FACT-21` | Workspace Size Violation | Rejects `scratch.size() < m` with `StatusCode::InvalidArgument` |
| `TEST-FACT-22` | Stale Version Invalidation | Pivot on `Basis` increments version; subsequent `solve()` rejected |
| `TEST-FACT-23` | Transactional Rollback | Failed solve leaves destination `solution` unchanged |
| `TEST-FACT-24` | Scale-Aware Factorization Check| Verifies $\|P B - L U\|_\infty \le \tau_{\text{fact}} (\|P B\|_\infty + \|L U\|_\infty)$ |

---

## 12. Strict Non-Goals

Phase 3C strictly excludes:
- Simplex pricing or reduced cost calculations
- Simplex ratio tests
- Primal or dual simplex iteration loops
- Phase I or Phase II optimization procedures
- GPU acceleration or CUDA kernels
- Third-party solver wrappers
