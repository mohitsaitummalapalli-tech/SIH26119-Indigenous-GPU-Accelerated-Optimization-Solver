# Phase 3C: Basis Factorization Layer — Implementation Report

> **Project:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver
> **Authoritative Specification Commit:** `76507ef2fb42eaa6f8354ce62d4a2f3c458d5dea`
> **Component:** Basis Factorization & Hot-Path Solve Layer (`src/solver/lp/basis_factorization.hpp`, `basis_factorization.cpp`)
> **Scope Status:** Strictly limited to numerical basis factorization and solves ($B x = \text{rhs}$ and $B^T y = \text{rhs}$). Contains **zero** simplex pricing, reduced costs, ratio tests, primal/dual iterations, Phase I/II, or GPU kernels.

---

## 1. Executive Summary & Factorization Convention

Phase 3C implements the CPU basis-factorization layer adhering strictly to the contract:

$$P B = L U$$

where:
- $B \in \mathbb{R}^{m \times m}$ is the basis matrix formed by selecting $m$ basic columns from the standardized constraint matrix $A$ via a non-owning `BasisMatrixView`.
- $P \in \mathbb{R}^{m \times m}$ is an orthogonal row-permutation matrix, tracked deterministically via permutation vector $\pi_r \in \mathbb{N}^m$ and its exact inverse $\pi_r^{-1} \in \mathbb{N}^m$.
- $Q = I$ (strictly **no column permutations** in this implementation; column-permuted/sparse Markowitz variants are documented as extension points for Phase 3D).
- $L \in \mathbb{R}^{m \times m}$ is unit lower triangular ($L_{ii} = 1.0$, subdiagonal $|L_{ij}| \le 1.0$ in exact arithmetic).
- $U \in \mathbb{R}^{m \times m}$ is upper triangular ($U_{ij} = 0.0$ for $i > j$).

### Mathematical Deviations
- **Zero deviations:** The implementation adheres completely and without exception to the approved mathematical specification.

---

## 2. Mathematical Algorithms & Solve Sequences

### 2.1 Standard Row Partial Pivoting
At each elimination step $k \in \{0, \dots, m-1\}$:
1. **Pivot Selection:**
   $$p = \operatorname{argmax}_{i \ge k} |M_{i, k}^{(k)}|$$
   **Deterministic Tie-Breaking:** If multiple row candidates achieve the exact maximum magnitude, the candidate with the **smallest row index** $p$ is chosen.
2. **Pivot Validation:**
   - If $|M_{p, k}^{(k)}|$ is NaN or Inf $\implies$ returns `StatusCode::NumericalFailure`.
   - If $|M_{p, k}^{(k)}| \le \varepsilon_{\text{sing}}$ (default $10^{-12}$) $\implies$ returns `StatusCode::NumericalFailure`.
3. **Row Swaps:**
   Rows $k$ and $p$ are swapped in working matrix $M$ (swapping both previously computed multipliers for $j < k$ and active submatrix entries for $j \ge k$) and in permutation vector $\pi_r$.
4. **Multiplier Elimination:**
   $$L_{i, k} = \frac{M_{i, k}^{(k)}}{U_{k, k}}, \quad M_{i, j} \leftarrow M_{i, j} - L_{i, k} U_{k, j} \quad (i > k, j > k)$$
   Because $|U_{k, k}| = \max_{i \ge k} |M_{i, k}^{(k)}|$, we have $|L_{i, k}| \le 1.0$ in exact arithmetic.

### 2.2 Primal Solve (FTRAN): $B x = \text{rhs}$
Algebra:
$$B = P^T L U \implies P^T L U x = \text{rhs} \iff L U x = P \text{rhs}$$

Execution sequence:
1. **Permutation & Forward Substitution on $L$:**
   Compute $w = L^{-1} (P \text{rhs})$ directly in `scratch`:
   $$w_i = \text{rhs}[\pi_r(i)] - \sum_{j=0}^{i-1} L_{i, j} w_j, \quad i = 0, \dots, m-1$$
2. **Backward Substitution on $U$:**
   Compute $x = U^{-1} w$ in-place in `scratch`:
   $$x_i = \frac{w_i - \sum_{j=i+1}^{m-1} U_{i, j} x_j}{U_{i, i}}, \quad i = m-1, m-2, \dots, 0$$
3. **Validation & Independent Backward Residual Check:**
   Validate finite candidate entries, then compute:
   $$\|B x - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|x\|_\infty + \|\text{rhs}\|_\infty \right)$$
   using the original basis entries. If denominator is $0.0$, requires exact $\|B x - \text{rhs}\|_\infty == 0.0$.
4. **Transactional Commit:**
   Copy `scratch` to `solution` only after residual verification succeeds. On any failure, `solution` remains strictly unmodified.

### 2.3 Dual Transpose Solve (BTRAN): $B^T y = \text{rhs}$
Algebra:
$$B = P^T L U \implies B^T = (P^T L U)^T = U^T L^T P$$
$$B^T y = \text{rhs} \iff U^T L^T P y = \text{rhs}$$

Execution sequence:
1. **Forward Substitution on $U^T$ (Lower Triangular):**
   Solve $U^T w = \text{rhs}$ in `scratch`:
   $$w_i = \frac{\text{rhs}_i - \sum_{j=0}^{i-1} U_{j, i} w_j}{U_{i, i}}, \quad i = 0, \dots, m-1$$
2. **Backward Substitution on $L^T$ (Unit Upper Triangular):**
   Solve $L^T v = w$ in `scratch`:
   $$v_i = w_i - \sum_{j=i+1}^{m-1} L_{j, i} v_j, \quad i = m-1, m-2, \dots, 0$$
3. **Independent Backward Residual Check:**
   Using the exact unpermutation mapping $y = P^T v \iff y[\pi_r(i)] = v_i \iff y[k] = v[\pi_r^{-1}(k)]$, compute:
   $$\|B^T y - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|y\|_\infty + \|\text{rhs}\|_\infty \right)$$
   with zero-denominator handling.
4. **Transactional Commit:**
   Commit unpermuted values into destination `solution`:
   $$\text{solution}[\pi_r(i)] = \text{scratch}[i], \quad \forall i \in \{0, \dots, m-1\}$$

---

## 3. Architecture & Contracts

### 3.1 Zero-Allocation Contract
- All dynamic allocations for storing $B$, $L$, $U$, $\pi_r$, and $\pi_r^{-1}$ occur during `factorize()` setup.
- Once factorized, hot-path calls to `solve()` and `solve_transpose()` perform **strictly zero dynamic heap allocations**.
- The test harness instruments global `operator new` to verify `g_alloc_count == 0` during all hot-path solves.

### 3.2 Pairwise Distinctness & Workspace Aliasing
Preconditions enforce:
- `&rhs != &solution`
- `&rhs != &scratch`
- `&solution != &scratch`
Violations are caught and return `StatusCode::InvalidArgument`.

### 3.3 Transactional State Machine
- States: `FactorizationState::Empty`, `FactorizationState::Factored`, `FactorizationState::Failed`.
- Failed factorizations or refactorizations immediately reset internal buffers and transition to `FactorizationState::Failed`. No half-factored or corrupted factorization is ever usable.
- Failed solves leave `solution` strictly untouched (verified via sentinels in `TEST-FACT-21`).

### 3.4 Basis Version Binding
- During factorization, `factorize()` records `basis.version()`.
- Solves inspect `basis_view_->basis().version() == basis_version_`.
- If the basis has been modified (e.g. by column replacement), subsequent solves are rejected with `StatusCode::InconsistentModel` without modifying `solution`.

### 3.5 Scale-Aware Factorization Check & Reconciled Tolerances
- An independent verification method `compute_factorization_residual()` validates:
  $$\|P B - L U\|_\infty \le \tau_{\text{fact}} \left( \|P B\|_\infty + \|L U\|_\infty \right)$$
  ensuring numerical integrity of the computed factors.
- **Tolerance Reconciliation:** Default `fact_residual_tol` is explicitly set to `1e-12` (reconciled with the authoritative specification). Double precision arithmetic yields residuals on the order of $10^{-16}$, making $10^{-12}$ a rigorous quality gate that guards against any precision loss or factor inconsistency.
- **Source Independence:** The factorization residual reads $P B$ directly from the authoritative `BasisMatrixView` (querying the original CSR matrix entries with zero allocations) rather than using internal dense buffers or LU data.

### 3.6 Independent Solve Residual Verification (FTRAN & BTRAN)
- Both primal solve ($B x = \text{rhs}$) and dual transpose solve ($B^T y = \text{rhs}$) compute backward residuals directly against the authoritative `BasisMatrixView`.
- Coefficients $B_{i, j}$ are queried on-the-fly via `BasisMatrixView::get(row, col)`, executing binary searches across CSR column arrays with zero dynamic allocations on the hot path.
- The verification does not rely on `b_dense_` or reconstruct $B$ from LU factors.

### 3.7 Numerical Norm Overflow Safety
- Norm accumulations ($\|B\|_\infty$, $\|B\|_1$, residual norms, and $L U$ product accumulations) are strictly guarded against $+ \infty$, $\text{NaN}$, and non-finite intermediate values.
- Non-finite values or arithmetic overflows trigger graceful returns with `StatusCode::NumericalFailure` and transition factorization state cleanly to `FactorizationState::Failed`.

### 3.8 Pivot-Growth & Condition Number Policy
- **Pivot Growth Limit (`max_growth_tol = 1e12`):** Documented strictly as an operational safeguard against extreme growth, not a general numerical stability theorem.
- **Diagnostic Condition Estimation:** The Hager-Higham 1-norm condition estimator $\kappa^*(B) = \|B\|_1 \|B^{-1}\|_1$ is purely diagnostic and never terminates a factorization or solve prematurely on its own without residual evidence.

---

## 4. Test Matrix & Validation Results

The implementation is verified by a test suite consisting of 30 fixtures and 50 randomized property tests:

| Test ID | Description | Result |
| :--- | :--- | :--- |
| `TEST-FACT-01` | $0 \times 0$ empty basis factorization and solve | PASS |
| `TEST-FACT-02` | $1 \times 1$ scalar basis forward and transpose solve | PASS |
| `TEST-FACT-03` | Diagonal matrix solves matching reciprocals | PASS |
| `TEST-FACT-04` | Upper triangular matrix backward substitution | PASS |
| `TEST-FACT-05` | Dense nonsingular $3 \times 3$ primal and transpose solve | PASS |
| `TEST-FACT-06` | Sparse tridiagonal $5 \times 5$ zero-allocation solve | PASS |
| `TEST-FACT-07` | Permutation matrix exact index unpermutation | PASS |
| `TEST-FACT-08` | Negative pivot selection and sign preservation | PASS |
| `TEST-FACT-09` | Mixed-sign entries with positive/negative numbers | PASS |
| `TEST-FACT-10` | Singular matrices rejected with `StatusCode::NumericalFailure` | PASS |
| `TEST-FACT-11` | Nearly singular pivot ($\le 10^{-12}$) rejected with `StatusCode::NumericalFailure` | PASS |
| `TEST-FACT-12` | Repeated/equal candidate pivots deterministic tie-breaking | PASS |
| `TEST-FACT-13` | Badly scaled matrix ($10^{-6}$ to $10^{6}$) scale-aware residual test | PASS |
| `TEST-FACT-14` | FTRAN primal solve with independent residual test | PASS |
| `TEST-FACT-15` | BTRAN transpose solve with independent residual test | PASS |
| `TEST-FACT-16` | Multiple consecutive RHS solves on same factorization | PASS |
| `TEST-FACT-17` | NaN in RHS rejected with `StatusCode::InvalidArgument` | PASS |
| `TEST-FACT-18` | Inf in RHS rejected with `StatusCode::InvalidArgument` | PASS |
| `TEST-FACT-19` | Pairwise aliasing between RHS, solution, scratch rejected | PASS |
| `TEST-FACT-20` | Stale basis version rejected with `StatusCode::InconsistentModel` | PASS |
| `TEST-FACT-21` | Transactional rollback: destination unmodified on failure | PASS |
| `TEST-FACT-22` | Factorization residual $\|P B - L U\|_\infty$ verification | PASS |
| `TEST-FACT-23` | Deterministic pivoting repeatability across runs | PASS |
| `TEST-FACT-24` | Agreement and residual verification with independent complete-pivoting oracle | PASS |
| `TEST-FACT-25` | Reconciled tolerance defaults verified (`fact_residual_tol = 1e-12`) | PASS |
| `TEST-FACT-26` | Arithmetic overflow and non-finite input safety | PASS |
| `TEST-FACT-27` | Independent residual verification directly against `BasisMatrixView` | PASS |
| `TEST-FACT-28` | Transpose solve correctness under non-trivial row permutations ($P \ne I$) | PASS |
| `TEST-FACT-29` | Transactional failure preservation across all error paths | PASS |
| `TEST-FACT-30` | Badly scaled nonsingular basis solve & diagnostic condition validation | PASS |
| **Property Tests** | 50 randomized non-diagonally dominant test cases (Seed `0x3C3C3C`) | PASS |

### Complete Test Suite Summary
- **12/12 CTest targets passing (100%)**:
  1. `InfrastructureTest`
  2. `ModelTest`
  3. `MpsParserTest`
  4. `LpParserTest`
  5. `SerializationTest`
  6. `DenseVectorTest`
  7. `DenseMatrixTest`
  8. `SparseMatrixTest`
  9. `NumericsTest`
  10. `LpStandardizationTest`
  11. `BasisTest`
  12. `BasisFactorizationTest`
- **Compiler output:** 0 errors, 0 warnings.
