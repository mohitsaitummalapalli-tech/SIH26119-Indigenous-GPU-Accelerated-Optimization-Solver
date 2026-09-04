# Phase 3C: Basis Factorization Layer — Mathematical and Numerical Specification

> **Authoritative Mathematical, Numerical, and Architectural Specification**  
> **Repository:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver  
> **Status:** Phase 3C Specification Gate  
> **Authoritative Baseline Commit:** `68e3d8553e8b749e4e08d00730bfd131188e8c3e`  
> **Mandatory Scope Declaration:** *Phase 3C defines the numerical factorization and linear solve layer ($B x = r$ and $B^T y = r$) for the simplex basis matrix. It does NOT implement simplex pricing, pivots, ratio tests, Phase I, Phase II, primal simplex, dual simplex, GPU acceleration, or external solver interfaces.*

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
| Solves B x = r (FTRAN) and B^T y = r (BTRAN) via P B Q = L U                      |
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
This specification builds strictly upon the existing committed Phase 1, Phase 2, Phase 3A, and Phase 3B contracts:
- [`src/solver/lp/basis.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis.hpp): Authoritative structural basis state, $O(1)$ query tables, and monotonically increasing `version()`.
- [`src/solver/lp/basis_matrix_view.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis_matrix_view.hpp): Non-owning logical view over $A \in \mathbb{R}^{m \times n}$ where column $k$ is $A_{:, B(k)}$.
- [`src/numerics/sparse_matrix.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/sparse_matrix.hpp): Zero-allocation CSR SpMV and residual contracts (`multiply`, `residual`).
- [`src/numerics/dense_vector.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/dense_vector.hpp): Zero-allocation dense vector layer with `is_finite_scalar` guarantees.
- [`src/numerics/tolerances.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/tolerances.hpp): Authoritative `approx_equal` and `approx_zero` semantics.
- [`src/core/status.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/core/status.hpp): Native error and result types (`StatusCode::InvalidArgument`, `StatusCode::InconsistentModel`, `StatusCode::InvalidBounds`).

---

## 2. Mathematical Definition of the Basis Matrix & State Lifecycle

### 2.1 The Basis Matrix $B$
Let the standardized constraint system be $A x = b$ with $A \in \mathbb{R}^{m \times n}$, $m \le n$. Given a structurally valid [`Basis`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis.hpp) with column mapping $B(0), B(1), \dots, B(m-1)$, the basis matrix $B$ is the square $m \times m$ matrix formed by selecting the basic columns of $A$ in row order:

$$B = \begin{bmatrix} A_{:, B(0)} & A_{:, B(1)} & \cdots & A_{:, B(m-1)} \end{bmatrix} \in \mathbb{R}^{m \times m}$$

Entry-wise:
$$B_{i, k} = A_{i, B(k)}, \quad \forall i \in \{0, \dots, m-1\}, \; k \in \{0, \dots, m-1\}$$

### 2.2 Factorization Lifecycle States
The factorization object must maintain an explicit, observable lifecycle state:

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

1. **`Unfactorized`**: Initial state prior to any factorization attempt, or following an explicit reset.
2. **`Factored`**: Factorization $P B Q = L U$ succeeded. `fact.basis_version == basis.version()`. Both forward solve ($B x = r$) and backward/dual solve ($B^T y = r$) are valid.
3. **`NumericallySingular`**: Factorization detected that $\min_k |U_{k, k}| \le \varepsilon_{\text{sing}}$ or encountered non-finite arithmetic. Solves are rejected with `StatusCode::InconsistentModel`.
4. **`Invalid`**: Dimensions or internal invariants corrupted.
5. **`Stale`**: The underlying `Basis` has pivoted (`basis.version() > fact.basis_version`). Any call to `solve()` or `solve_transpose()` is strictly rejected.

### 2.3 Strict Version Invariant
> [!IMPORTANT]
> **Mandatory Version Binding Invariant:**  
> $$\text{factorization.basis\_version} \equiv \text{basis.version()}$$
> A factorization created for basis version $v$ **MUST NOT and CANNOT** be silently executed for basis version $v+1$. Any attempt to solve with a stale factorization immediately fails with `StatusCode::InconsistentModel`.

---

## 3. Factorization Method Evaluation & Selection

### 3.1 Candidate Architectures

| Criteria | Dense LU with Partial Pivoting ($P B = L U$) | Sparse LU with Markowitz ($P B Q = L U$) | Product Form of Inverse (PFI) |
| :--- | :--- | :--- | :--- |
| **Numerical Stability** | Maximum: Guaranteed $\vert L_{ij} \vert \le 1.0$ via row partial pivoting | High: Guaranteed by threshold parameter $u \in [0.1, 1.0]$ | Degrades rapidly; requires periodic refactorization |
| **Fill-in Control** | None ($O(m^2)$ storage always allocated) | Explicit: Markowitz count $\mu_{ij} = (r_i - 1)(c_j - 1)$ | Accumulates eta vectors linearly with pivots |
| **Algorithmic Complexity** | $O(m^3)$ factorization, $O(m^2)$ solve | $O(m + \text{fill})$ solve; $O(\text{nnz} \cdot m)$ factorization | $O(m + k \cdot m)$ solve after $k$ pivots |
| **Implementation Risk** | Low: Deterministic, no dynamic linked structures | Medium: Requires sparse matrix dynamic allocation / garbage collection | High if used without robust base factorization |
| **Simplex Integration** | Ideal reference base for Phase 3C | Primary engine for large-scale industrial simplex | Simplex update mechanism, not a base factorization |

### 3.2 Chosen Method for Phase 3C
**Primary Production Architecture for Phase 3C:**  
**Threshold Partial Pivoting Sparse/Dense Unification ($P B Q = L U$)**:
1. **Mathematical Structure:** Row permutations $P$ and column permutations $Q$ such that:
   $$P B Q = L U$$
   where $L$ is unit lower triangular ($L_{ii} = 1$) and $U$ is upper triangular.
2. **Initial Phase 3C Engine:**
   - For small to medium systems ($m \le 2000$), an explicit Dense Column-Major Block LU with Row Partial Pivoting ($Q = I$, $P B = L U$) provides an unconditional, zero-allocation, numerically bulletproof foundation.
   - For sparse large systems, the exact same API and permutation algebra $P B Q = L U$ generalizes directly to Markowitz threshold pivoting without changing a single line of solver interface code.
3. **Justification:**
   - Industrial solvers (HiGHS, CPLEX, GLPK) all separate the base factorization contract from basis update routines. Establishing a robust, deterministic, zero-allocation $P B Q = L U$ solver in Phase 3C allows Phase 3D to implement simplex pricing and ratio tests with 100% numerical confidence.

---

## 4. Complete LU Mathematics & Solve Derivations

### 4.1 Matrix Decomposition Structure
The factorization computes permutation matrices $P, Q \in \mathbb{R}^{m \times m}$, a unit lower triangular matrix $L \in \mathbb{R}^{m \times m}$, and an upper triangular matrix $U \in \mathbb{R}^{m \times m}$ satisfying:

$$P B Q = L U$$

- $P$: Row permutation matrix ($P e_i = e_{\pi_r(i)}$ where $\pi_r$ is the row pivot index vector).
- $Q$: Column permutation matrix ($Q e_j = e_{\pi_c(j)}$ where $\pi_c$ is the column pivot index vector). For row-only pivoting, $Q = I$ and $\pi_c(j) = j$.
- $L$: Unit lower triangular:
  $$L = \begin{bmatrix}
  1 & 0 & \cdots & 0 \\
  l_{1, 0} & 1 & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  l_{m-1, 0} & l_{m-1, 1} & \cdots & 1
  \end{bmatrix}, \quad |l_{i, j}| \le \frac{1}{u} \le 10.0$$
- $U$: Upper triangular:
  $$U = \begin{bmatrix}
  u_{0, 0} & u_{0, 1} & \cdots & u_{0, m-1} \\
  0 & u_{1, 1} & \cdots & u_{1, m-1} \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & u_{m-1, m-1}
  \end{bmatrix}$$

---

### 4.2 Derivation of Primal Solve: $B x = r$ (FTRAN)

We wish to solve for $x \in \mathbb{R}^m$ given RHS $r \in \mathbb{R}^m$:

$$B x = r$$

1. Multiply by row permutation $P$ and insert $Q Q^T = I$:
   $$P B (Q Q^T) x = P r \implies (P B Q) (Q^T x) = P r$$
2. Substitute $P B Q = L U$:
   $$L U (Q^T x) = P r$$
3. Let $z = P r \in \mathbb{R}^m$ (permute RHS by row permutation):
   $$z_i = r_{\pi_r(i)}, \quad i \in \{0, \dots, m-1\}$$
4. **Forward Substitution ($L w = z$):**
   Since $L$ is unit lower triangular ($L_{ii} = 1$):
   $$w_i = z_i - \sum_{j=0}^{i-1} L_{i, j} w_j, \quad i = 0, 1, \dots, m-1$$
5. **Backward Substitution ($U v = w$):**
   Since $U$ is upper triangular:
   $$v_i = \frac{w_i - \sum_{j=i+1}^{m-1} U_{i, j} v_j}{U_{i, i}}, \quad i = m-1, m-2, \dots, 0$$
6. **Column Unpermutation ($x = Q v$):**
   $$x_{\pi_c(i)} = v_i \iff x = Q v, \quad i = 0, 1, \dots, m-1$$

---

### 4.3 Derivation of Dual / Transpose Solve: $B^T y = r$ (BTRAN)

We wish to solve for $y \in \mathbb{R}^m$ given dual RHS $r \in \mathbb{R}^m$:

$$B^T y = r$$

1. Transpose the factorization equation $P B Q = L U$:
   $$(P B Q)^T = (L U)^T \implies Q^T B^T P^T = U^T L^T$$
2. Isolate $B^T$:
   $$B^T = Q U^T L^T P$$
3. Substitute into $B^T y = r$:
   $$Q U^T L^T P y = r$$
4. Multiply both sides by $Q^T$ (since $Q^T Q = I$):
   $$U^T L^T (P y) = Q^T r$$
5. Let $z = Q^T r \in \mathbb{R}^m$ (permute RHS by column permutation):
   $$z_i = r_{\pi_c(i)}, \quad i \in \{0, \dots, m-1\}$$
6. **Forward Substitution on Transpose ($U^T w = z$):**
   $U^T$ is lower triangular with non-unit diagonal $U_{ii}$:
   $$w_i = \frac{z_i - \sum_{j=0}^{i-1} U_{j, i} w_j}{U_{i, i}}, \quad i = 0, 1, \dots, m-1$$
7. **Backward Substitution on Transpose ($L^T v = w$):**
   $L^T$ is unit upper triangular ($L_{ii} = 1$):
   $$v_i = w_i - \sum_{j=i+1}^{m-1} L_{j, i} v_j, \quad i = m-1, m-2, \dots, 0$$
8. **Row Unpermutation ($y = P^T v$):**
   Since $v = P y \implies y = P^T v$:
   $$y_{\pi_r(i)} = v_i \iff y = P^T v, \quad i = 0, 1, \dots, m-1$$

> [!IMPORTANT]
> **Summary of Transpose Permutation Sequencing:**  
> - FTRAN: $r \xrightarrow{P} z \xrightarrow{L^{-1}} w \xrightarrow{U^{-1}} v \xrightarrow{Q} x$  
> - BTRAN: $r \xrightarrow{Q^T} z \xrightarrow{U^{-T}} w \xrightarrow{L^{-T}} v \xrightarrow{P^T} y$  
> Notice that the operations and permutations reverse in exact algebraic duality.

---

## 5. Pivoting Policy & Deterministic Tie-Breaking

### 5.1 Threshold Partial Pivoting (TPP)
At elimination step $k \in \{0, \dots, m-1\}$ on working submatrix $M^{(k)}$:
1. Candidate search in active column $k$:
   $$c_{\max} = \max_{i \ge k} |M_{i, k}^{(k)}|$$
2. If $c_{\max} \le \varepsilon_{\text{sing}}$ or $\text{isnan}(c_{\max})$:
   The submatrix has numerical rank deficiency. Factorization halts immediately and reports `StatusCode::InconsistentModel`.
3. **Threshold Acceptance Criterion:**
   A candidate row $p \ge k$ is acceptable if:
   $$|M_{p, k}^{(k)}| \ge u \cdot c_{\max}$$
   where $u \in (0.0, 1.0]$ is the Markowitz/TPP threshold parameter.
   - **Default:** $u = 0.1$ for sparse Markowitz, $u = 1.0$ for dense partial pivoting (standard Gaussian elimination).
4. **Deterministic Tie-Breaking Rule:**
   If multiple candidate rows satisfy $|M_{p, k}^{(k)}| \ge u \cdot c_{\max}$:
   $$p^* = \min \left\{ p \in \{k, \dots, m-1\} : |M_{p, k}^{(k)}| = \max_{i \ge k} |M_{i, k}^{(k)}| \right\}$$
   Ties on identical magnitude are broken strictly by **smallest row index $p$**. This ensures bit-level determinism across platforms.

### 5.2 Singularity Criteria vs. Condition Estimation vs. Residual Check
The specification enforces three distinct numerical metrics that must never be conflated:

| Diagnostic Metric | Mathematical Definition | Role / Decision Boundary | Action upon Violation |
| :--- | :--- | :--- | :--- |
| **Pivot Singularity ($\varepsilon_{\text{sing}}$)** | $\min_k |U_{k, k}| \le \varepsilon_{\text{sing}}$ (Default: $10^{-12}$) | Local pivot acceptance during LU elimination | Immediate factorization abort; return `InconsistentModel` |
| **Condition Estimate ($\kappa^*(B)$)** | $\kappa^*(B) = \|B\|_\infty \|B^{-1}\|_\infty \approx \|B\|_\infty \frac{\|w\|_\infty}{\|z\|_\infty}$ (via Hager-Higham 1-norm estimator) | Global conditioning diagnostic (Ceiling: $10^{13}$) | Flag ill-conditioning; schedule basis refactorization / pivot reject |
| **Solve Residual ($\tau_{\text{resid}}$)** | $\frac{\|B x - r\|_\infty}{\|B\|_\infty \|x\|_\infty + \|r\|_\infty} \le \tau_{\text{resid}}$ (Default: $10^{-8}$) | A posteriori backward error verification on every solve | Return error; trigger refactorization |

---

## 6. Numerical Stability & Error Bounds

### 6.1 Pivot Growth Factor
During elimination, the growth factor $\rho_m$ is monitored:

$$\rho_m = \frac{\max_{i, j, k} |M_{i, j}^{(k)}|}{\max_{i, j} |B_{i, j}|}$$

If $\rho_m > 10^{12}$, severe roundoff accumulation is occurring; the factorization is marked unstable.

### 6.2 Backward Error Bounds for Solves
For computed solution $\hat{x}$, the componentwise backward error $\omega$ is:

$$\omega = \max_i \frac{|(B \hat{x} - r)_i|}{(|B| |\hat{x}| + |r|)_i}$$

A solve is accepted as numerically sound if:

$$\|B \hat{x} - r\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{x}\|_\infty + \|r\|_\infty \right)$$

where $\tau_{\text{resid}} = 10^{-8}$ by default (configurable in `FactorizationTolerances`).

---

## 7. Scaling Strategy

### 7.1 Mathematical Transformation
To minimize condition numbers and equalize row magnitudes prior to factorization, row scaling $D_r$ and column scaling $D_c$ may optionally be applied:

$$\tilde{B} = D_r B D_c$$

where $D_r = \operatorname{diag}(d_{r, 0}, \dots, d_{r, m-1})$ and $D_c = \operatorname{diag}(d_{c, 0}, \dots, d_{c, m-1})$ are positive diagonal matrices.

### 7.2 Phase 3C Policy
- **Phase 3C Default:** **Unscaled Factorization ($D_r = I, D_c = I$)**.
  The standardized matrix $A$ from Phase 3A is already normalized in RHS sign. Avoiding intermediate scaling in Phase 3C ensures that verification residuals directly match the canonical matrix without compounding scaling roundoff.
- **Undoing Scaling in Solves (Architecture Ready):**
  - Primal: $B x = r \iff (D_r B D_c) (D_c^{-1} x) = D_r r \implies \tilde{B} \tilde{x} = D_r r$, then $x = D_c \tilde{x}$.
  - Dual: $B^T y = r \iff (D_c B^T D_r) (D_r^{-1} y) = D_c r \implies \tilde{B}^T \tilde{y} = D_c r$, then $y = D_r \tilde{y}$.

---

## 8. Storage Model & Zero-Allocation Memory Layout

### 8.1 Memory Structures
All storage is preallocated during initialization or resized only when basis dimension $m$ changes. Inside the simplex loop, **zero heap allocations are executed**.

```cpp
struct FactorizationStorage {
    Dimension m{0};                       ///< Basis dimension m x m
    std::vector<Scalar> lu_data;         ///< Dense m x m column-major (or sparse CSR L and U)
    std::vector<Index> row_perm;         ///< pi_r: length m (row permutation P)
    std::vector<Index> row_perm_inv;     ///< pi_r_inv: length m (P^T)
    std::vector<Index> col_perm;         ///< pi_c: length m (column permutation Q)
    std::vector<Index> col_perm_inv;     ///< pi_c_inv: length m (Q^T)
    
    // Diagnostic metrics
    Scalar max_growth{1.0};
    Scalar condition_estimate{1.0};
    uint64_t basis_version{0};
    bool is_factored{false};
};
```

### 8.2 Memory Bounds
- For dense storage: Exactly $m^2$ `Scalar` entries ($\approx 8 \times m^2$ bytes) + $4m$ `Index` entries.
- For $m = 1000$: $8 \times 10^6 \text{ bytes} \approx 8 \text{ MB}$.
- For $m = 2000$: $32 \text{ MB}$.

---

## 9. Solve API & Workspace Contracts

### 9.1 C++ Interface Specification

```cpp
namespace sih26119 {

struct FactorizationTolerances {
    Scalar singularity_tol{1e-12};   ///< Min acceptable diagonal pivot |U_kk|
    Scalar residual_tol{1e-8};       ///< Relative solve residual ceiling
    Scalar max_growth_tol{1e12};     ///< Maximum permissible pivot growth
    Scalar condition_ceiling{1e13};  ///< Ill-conditioning alarm ceiling
    Scalar tpp_threshold{1.0};       ///< Threshold partial pivoting parameter u in (0, 1]
};

class BasisFactorization {
public:
    BasisFactorization() = default;

    /**
     * @brief Performs full LU factorization of the basis matrix view:
     *
     *     P B Q = L U
     *
     * @param basis_view Non-owning view over the standard constraint matrix A and Basis.
     * @param tols Configurable numerical tolerances.
     * @return Status::ok() on success, or error status upon numerical singularity / failure.
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

    /// Observable state queries (O(1), zero allocation, noexcept)
    [[nodiscard]] bool is_factored() const noexcept;
    [[nodiscard]] uint64_t basis_version() const noexcept;
    [[nodiscard]] Dimension dimension() const noexcept;
    [[nodiscard]] Scalar condition_estimate() const noexcept;
    [[nodiscard]] Scalar pivot_growth() const noexcept;
};

} // namespace sih26119
```

### 9.2 Strict Aliasing & Workspace Requirements
Adhering to Phase 2 and Phase 3B contracts:
1. `&rhs != &solution && (rhs.size() == 0 || rhs.data() != solution.data())`.
2. `&rhs != &scratch && (rhs.size() == 0 || rhs.data() != scratch.data())`.
3. `&solution != &scratch && (solution.size() == 0 || solution.data() != scratch.data())`.
4. `scratch.size() >= m`.
5. On failure, `solution` remains strictly unmodified (transactional guarantee).

---

## 10. Refactorization Triggers & Basis Update Roadmap

### 10.1 Refactorization Triggers
A full refactorization is mandatorily triggered when:
1. `basis.version() != fact.basis_version` (basis was modified).
2. Number of cumulative rank-1 updates exceeds $K_{\text{refactor}}$ (e.g. $K_{\text{refactor}} = 50$).
3. Relative solve residual exceeds $\tau_{\text{resid}}$ ($\|B x - r\|_\infty > \tau_{\text{resid}}(\|B\| \|x\| + \|r\|)$).
4. Condition number estimate $\kappa^*(B) > 10^{13}$.
5. Explicit solver request.

### 10.2 Phase 3C Policy vs. Phase 3D Update Roadmap
- **Phase 3C:** **Full Refactorization on Every Basis Change**.
  In Phase 3C, any pivot on `Basis` increments `version()`, requiring a clean, full call to `factorize()`. This establishes the baseline oracle of exact factorization accuracy.
- **Phase 3D Roadmap:**
  Phase 3D will introduce rank-1 basis update algorithms (Forrest-Tomlin or Product Form of Inverse $B_k = B_0 E_1 E_2 \cdots E_k$), comparing them against Phase 3C's full refactorization baseline.

---

## 11. Independent Verification & Testing Strategy

### 11.1 Independent Factorization Oracle
An independent oracle [`GaussianEliminationOracle`] will be authored in `tests/unit/test_basis_factorization.cpp`:
- Operates independently from `BasisFactorization`.
- Uses naive Cramer's rule for $m \le 3$, and independent textbook Gaussian elimination with full pivoting and double-precision residual checking for $m > 3$.
- Compares solutions of $B x = r$ and $B^T y = r$ bit-for-bit or within machine epsilon $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$.

### 11.2 Comprehensive Test Matrix (24 Required Test Cases)

| Test ID | Test Category | Mathematical Contract Verified |
| :--- | :--- | :--- |
| `TEST-FACT-01` | Degenerate $0 \times 0$ | $m=0$ basis: trivially factored, solve on empty vector succeeds |
| `TEST-FACT-02` | Scalar $1 \times 1$ | $B = [5.0]$, solves $5x = 10 \implies x=2$, $5y = 15 \implies y=3$ |
| `TEST-FACT-03` | Diagonal Matrix | $B = \operatorname{diag}(2, -4, 5)$, $L=I$, $U=B$, solves match elementwise reciprocals |
| `TEST-FACT-04` | Upper Triangular | $B = U$, $L=I$, backward substitution exact match |
| `TEST-FACT-05` | Lower Triangular | $B = L$, $U=I$, forward substitution exact match |
| `TEST-FACT-06` | Permutation Matrix | $B = P$, $L=I, U=I$, pure index shuffle on FTRAN and BTRAN |
| `TEST-FACT-07` | Dense Nonsingular $3 \times 3$ | Random well-conditioned $3 \times 3$, verified against independent oracle |
| `TEST-FACT-08` | Dense Nonsingular $5 \times 5$ | Random well-conditioned $5 \times 5$, verified against independent oracle |
| `TEST-FACT-09` | Sparse Nonsingular $10 \times 10$ | Tridiagonal / banded matrix, verifies zero-allocation solve |
| `TEST-FACT-10` | Negative Pivots | System requiring negative pivots, verifies sign preservation |
| `TEST-FACT-11` | Mixed-Sign Entries | Matrix with large positive and negative numbers |
| `TEST-FACT-12` | Exact Singular (Zero Row) | Row of zeros $\implies$ detected at step $k$, returns `InconsistentModel` |
| `TEST-FACT-13` | Exact Singular (Col Dep) | Linearly dependent columns $\implies$ detected, returns `InconsistentModel` |
| `TEST-FACT-14` | Numerically Singular | Pivot $|U_{kk}| = 10^{-13} < \varepsilon_{\text{sing}}$, rejected as singular |
| `TEST-FACT-15` | Nearly Singular / Ill-Cond | Condition number $\approx 10^{14}$, condition warning triggered |
| `TEST-FACT-16` | Badly Scaled Matrix | Entries ranging from $10^{-6}$ to $10^{6}$, verifies stability |
| `TEST-FACT-17` | Multiple RHS Solves | Successive solves with distinct RHS vectors using same factorization |
| `TEST-FACT-18` | Transpose Duality ($B^T y = r$)| Verifies that $y^T (B x) == x^T (B^T y)$ within machine tolerance |
| `TEST-FACT-19` | NaN / Inf Input Rejection | Rejects non-finite entries in RHS or matrix with `InvalidArgument` |
| `TEST-FACT-20` | Workspace Aliasing | Rejects `&solution == &rhs` or `&scratch == &solution` |
| `TEST-FACT-21` | Workspace Size Violation | Rejects `scratch.size() < m` |
| `TEST-FACT-22` | Stale Version Invalidation | Pivot on `Basis` increments version; subsequent `solve()` rejected |
| `TEST-FACT-23` | Transactional Rollback | Failed solve leaves destination `solution` unchanged |
| `TEST-FACT-24` | Residual Verification Metric | Computes $\|B x - r\|_\infty$ and asserts $\le \tau_{\text{resid}} (\|B\| \|x\| + \|r\|)$ |

### 11.3 Deterministic Property Testing (`TEST-FACT-PROP-01`)
- PRNG seeded with fixed deterministic seed `0x3C3C3C`.
- Generates 50 random nonsingular matrices of dimensions $m \in [3, 20]$ with condition numbers $\kappa(B) \le 10^6$.
- Solves both $B x = r$ and $B^T y = r$ for randomized RHS vectors.
- Verifies:
  1. $\|B x - r\|_\infty \le 10^{-8} (\|B\|_\infty \|x\|_\infty + \|r\|_\infty)$
  2. $\|B^T y - r\|_\infty \le 10^{-8} (\|B\|_\infty \|y\|_\infty + \|r\|_\infty)$
  3. Solutions match the independent Gaussian elimination oracle within $10^{-9}$.

---

## 12. Performance Measurement Protocols (Non-Claimative)

Phase 3C defines instrumentation metrics for benchmarking without making unsubstantiated performance claims:
1. **Factorization Wall Time ($\mu s$):** Time to compute $P B Q = L U$.
2. **Solve Wall Time ($\mu s$):** Time to compute FTRAN ($B x = r$).
3. **Transpose Solve Wall Time ($\mu s$):** Time to compute BTRAN ($B^T y = r$).
4. **Fill Ratio:** $\frac{\operatorname{nnz}(L) + \operatorname{nnz}(U) - m}{\operatorname{nnz}(B)}$.
5. **Peak Working Memory (bytes):** Bytes allocated during initialization.
6. **Dynamic Heap Allocations per Solve:** Must be strictly **0**.

---

## 13. Acceptance Gate Checklist

Prior to authorizing Phase 3C implementation:
- [x] $P B Q = L U$ mathematical sign, diagonal, and permutation conventions explicitly specified.
- [x] Complete algebraic derivation of forward solve ($B x = r$) and backward dual solve ($B^T y = r$) documented.
- [x] Transpose solve permutation sequencing strictly derived and verified.
- [x] Deterministic tie-breaking rules for partial pivoting defined.
- [x] Clear mathematical distinction between pivot singularity threshold, condition number estimate, and solve residual.
- [x] Preallocated storage model and zero-allocation solve workspace contract specified.
- [x] Invariant binding `fact.basis_version == basis.version()` enforced.
- [x] Independent oracle and 24-case test matrix specified.
- [x] NO solver code, simplex code, pricing, ratio test, or GPU code added.
