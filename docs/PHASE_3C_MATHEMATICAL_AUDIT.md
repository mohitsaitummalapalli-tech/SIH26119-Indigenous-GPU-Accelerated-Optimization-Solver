# Phase 3C: Mathematical Audit — Basis Factorization & Solve Duality

> **Independent Mathematical Audit & Algebraic Proofs**
> **Repository:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver
> **Authoritative Specification:** `docs/PHASE_3C_BASIS_FACTORIZATION_SPECIFICATION.md`
> **Scope:** Rigorous verification of $P B = L U$ Algebra, Transpose Permutations, Numerical Stability, and Error Bounds

---

## 1. Initial Factorization Convention: $P B = L U$

For the initial Phase 3C contract, factorization is strictly row-permuted:

$$P B = L U$$

where:
- $B \in \mathbb{R}^{m \times m}$ is the basis matrix formed by $m$ columns of standard-form constraint matrix $A$.
- $P \in \mathbb{R}^{m \times m}$ is an orthogonal row-permutation matrix ($P^T P = P P^T = I$).
- $Q = I$ (column permutations are not part of the initial contract; sparse Markowitz ordering is reserved as an extension point).
- $L \in \mathbb{R}^{m \times m}$ is unit lower triangular ($L_{ii} = 1.0, L_{ij} = 0 \text{ for } j > i$).
- $U \in \mathbb{R}^{m \times m}$ is upper triangular ($U_{ij} = 0 \text{ for } i > j$).

---

## 2. Algebraic Soundness of Permutation Transpose Duality

### 2.1 Theorem (Primal and Transpose Solve Invariance)
Let $B \in \mathbb{R}^{m \times m}$ be nonsingular. Let $P B = L U$ be its row-pivoted LU factorization. Then:
1. The unique solution to $B x = r$ is given by:
   $$x = U^{-1} L^{-1} P r$$
2. The unique solution to $B^T y = r$ is given by:
   $$y = P^T L^{-T} U^{-T} r$$

### 2.2 Proof of (1) — Primal Solve (FTRAN)
From $P B = L U$, multiply on the left by $P^T$ (since $P^T P = I$):
$$B = P^T L U$$
Substitute into $B x = r$:
$$P^T L U x = r$$
Multiply on the left by $P$:
$$L U x = P r$$
Let $z = P r$. The solve executes in two triangular phases:
1. **Forward Substitution (Unit Lower Triangular):**
   Solve $L w = z$ for $w = L^{-1} P r$. Since $L$ is unit lower triangular, diagonals are $1.0$ and division by diagonals is exact.
2. **Backward Substitution (Upper Triangular):**
   Solve $U x = w$ for $x = U^{-1} w = U^{-1} L^{-1} P r$.

Solve sequence:
$$r \xrightarrow{P} z \xrightarrow{L^{-1}} w \xrightarrow{U^{-1}} x \quad \blacksquare$$

### 2.3 Proof of (2) — Dual / Transpose Solve (BTRAN)
From $B = P^T L U$, take the matrix transpose:
$$B^T = (P^T L U)^T = U^T L^T (P^T)^T = U^T L^T P$$
Substitute $B^T$ into $B^T y = r$:
$$U^T L^T P y = r$$
Notice that:
- $U^T$ is lower triangular with non-zero diagonals $U_{ii}$.
- $L^T$ is unit upper triangular with diagonal elements $1.0$.

The solve proceeds through three distinct stages:
1. **Forward Substitution on $U^T$ (Lower Triangular):**
   Solve $U^T w = r$ for $w = U^{-T} r$.
   $$w_i = \frac{r_i - \sum_{j=0}^{i-1} U_{ji} w_j}{U_{ii}}, \quad i = 0, \dots, m-1$$
2. **Backward Substitution on $L^T$ (Unit Upper Triangular):**
   Solve $L^T v = w$ for $v = L^{-T} w = L^{-T} U^{-T} r$.
   $$v_i = w_i - \sum_{j=i+1}^{m-1} L_{ji} v_j, \quad i = m-1, m-2, \dots, 0$$
3. **Unpermutation by $P^T$:**
   From $L^T P y = w$ and $L^T v = w$, we have:
   $$P y = v \implies y = P^T v$$

### 2.4 Permutation Vector Indexing Verification
Let the row permutation be stored as vector $\pi_r \in \mathbb{N}^m$, where row $i$ of $P B$ is row $\pi_r(i)$ of $B$. That is:
$$(P v)_i = v[\pi_r(i)]$$
For the transpose unpermutation $y = P^T v$:
$$(P^T v)[\pi_r(i)] = v[i] \implies y[\pi_r(i)] = v[i] \quad \text{for each } i \in \{0, \dots, m-1\}$$
Equivalently, using the inverse permutation $\pi_r^{-1}$:
$$y[j] = v[\pi_r^{-1}(j)] \quad \text{for each } j \in \{0, \dots, m-1\}$$
This unpermutation is algebraically exact, preserves data ordering, and requires zero heap allocation.

---

## 3. Numerical Stability Strategy & $L$-Multiplier Bound

### 3.1 Partial Pivoting Contract
At step $k \in \{0, \dots, m-1\}$, the pivot row $p \ge k$ is chosen by:
$$p = \operatorname{argmax}_{i \ge k} |M_{i, k}^{(k)}|$$
with deterministic tie-breaking: the smallest row index $i$ is chosen if multiple rows attain the maximum.

### 3.2 Pivot Acceptance Policy
Acceptance of a pivot is governed strictly by numerical thresholds:
- If $|M_{p, k}^{(k)}| == 0.0$ or is non-finite ($\text{NaN}/\text{Inf}$), the matrix is singular $\implies$ `StatusCode::NumericalFailure`.
- If $|M_{p, k}^{(k)}| \le \varepsilon_{\text{sing}}$ (default $10^{-12}$), the pivot is unacceptably small $\implies$ `StatusCode::NumericalFailure`.
These checks are separate from condition number evaluation and do not indicate model structural inconsistency.

### 3.3 Proof of $|L_{ij}| \le 1.0$
- **Proposition:** For row-only partial pivoting with pivot threshold $u = 1.0$, every subdiagonal entry satisfies $|L_{ij}| \le 1.0$.
- **Proof:** At step $k = j$, row $p$ is swapped to row $k$ where $|M_{k, k}^{(k)}| = \max_{i \ge k} |M_{i, k}^{(k)}|$. The multiplier computed for row $i > k$ is:
  $$L_{ik} = \frac{M_{i, k}^{(k)}}{M_{k, k}^{(k)}}$$
  Since $|M_{i, k}^{(k)}| \le |M_{k, k}^{(k)}|$, we have $|L_{ik}| \le 1.0$.
- **Scope Note:** This property holds strictly because of the partial pivoting selection rule $u = 1.0$. It is NOT an arbitrary property of generic LU factorizations (such as unpivoted LU or threshold pivoting with $u < 1$).

### 3.4 Rigorous Stability Characterization
- Partial pivoting is the selected baseline numerical-stability strategy; it does **not** provide an unconditional error guarantee for every matrix.
- **Worst-case growth:** The theoretical pivot growth factor $\rho_m = \frac{\max_{i,j,k} |M_{i,j}^{(k)}|}{\max_{i,j} |B_{i,j}|}$ can reach $2^{m-1}$ (e.g., Wilkinson counterexamples).
- **Distinction of Concepts:**
  1. *Practical numerical robustness:* Moderate growth $\rho_m \ll 10^3$ in virtually all LP basis instances.
  2. *Backward-error verification:* A posteriori residual checks on every solve ensure computed results are trustworthy regardless of intermediate growth.
  3. *Worst-case bounds:* $\rho_m \le 2^{m-1}$ in theory.
  4. *Formal stability guarantees:* Gaussian elimination with partial pivoting is backward stable only when pivot growth $\rho_m$ remains small.

---

## 4. Condition Number Diagnostics vs. Operational Policy

### 4.1 Condition Number Estimation via Hager-Higham Algorithm
The Hager-Higham 1-norm estimator estimates $\|B^{-1}\|_1$ via 4–5 solves of $B w = x$ and $B^T z = \xi$ without materializing $B^{-1}$.
$$\kappa^*(B) = \|B\|_1 \cdot \|w\|_1$$

### 4.2 Independence of Condition Number from Pivot Magnitude
The Ostrowski lower-triangular matrix:
$$T_m = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
-1 & 1 & 0 & \cdots & 0 \\
-1 & -1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-1 & -1 & -1 & \cdots & 1
\end{bmatrix}$$
has all pivots $U_{ii} = 1.0 > \varepsilon_{\text{sing}}$, yet $\kappa(T_m) = O(2^m)$. Thus, pivot thresholds alone cannot bound condition numbers.

### 4.3 Operational Diagnostic Policy
- $\kappa^*(B)$ is maintained strictly as an **operational diagnostic signal**.
- The solver does **NOT** assert that $\kappa^*(B) > 10^{13}$ implies a mathematically invalid solve.
- Instead, high condition estimates trigger:
  1. Informational diagnostics or warnings.
  2. Basis reinversion / refactorization.
  3. Numerical failure decision **only when combined with residual/error evidence**.
- The final acceptance or rejection of a solve never relies on condition estimate alone.

---

## 5. Solve Success Criteria & Scale-Aware Residuals

### 5.1 Backward Residual Test (Solve Success)
A solve $\hat{x}$ for $B x = \text{rhs}$ is declared successful if and only if:
$$\|B \hat{x} - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty \right)$$
where $\tau_{\text{resid}} = 10^{-8}$.

Similarly, a transpose solve $\hat{y}$ for $B^T y = \text{rhs}$ succeeds if and only if:
$$\|B^T \hat{y} - \text{rhs}\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{y}\|_\infty + \|\text{rhs}\|_\infty \right)$$

### 5.2 Zero Denominator Case
If $\|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty == 0.0$ (e.g., $B = 0$ or $\hat{x} = 0, \text{rhs} = 0$), the residual check requires:
$$\|B \hat{x} - \text{rhs}\|_\infty == 0.0$$
Otherwise, division by zero is avoided and exact equality is verified.

### 5.3 Scale-Aware Factorization Residual
For verifying the quality of the factorization itself, an independent check on $P B - L U$ is defined:
$$\|P B - L U\|_\infty \le \tau_{\text{fact}} \left( \|P B\|_\infty + \|L U\|_\infty \right)$$
where $\tau_{\text{fact}} = 10^{-10}$.
- If $\|P B\|_\infty + \|L U\|_\infty == 0.0$, the check requires $\|P B - L U\|_\infty == 0.0$.
- Exact floating-point equality $P B = L U$ is never claimed or expected due to roundoff error.

---

## 6. Numerical Status Code Semantics

A valid mathematical basis that is numerically unusable must NOT be classified as an invalid or inconsistent model.

The solver enforces:
- `StatusCode::NumericalFailure`:
  - Numerically singular basis ($|U_{kk}| \le \varepsilon_{\text{sing}}$)
  - Unacceptable or non-finite pivot
  - Failed factorization
  - Solve residual failure ($\|B \hat{x} - \text{rhs}\|_\infty > \tau_{\text{resid}} (\|B\|_\infty \|\hat{x}\|_\infty + \|\text{rhs}\|_\infty)$)
- `StatusCode::InconsistentModel` / `StatusCode::InvalidArgument`:
  - Reserved strictly for actual structural defects (dimension mismatch, invalid basis variable indices, out-of-range bounds, malformed input).

---

## 7. Independent Reference Oracle Architecture

To validate Phase 3C without circularity:
1. **Primary Reference Oracle:** Complete-Pivoting Gaussian Elimination (`CompletePivotingOracle`).
   - Independently implemented small-matrix reference.
   - Does NOT call `BasisFactorization`.
   - Does NOT use the Phase 3C pivot selection code (searches active 2D submatrix at every step: $(p, q) = \operatorname{argmax}_{i \ge k, j \ge k} |M_{i,j}|$).
   - Solves both $B x = \text{rhs}$ and $B^T y = \text{rhs}$.
   - Independently computes backward residuals.
2. **Cramer's Rule:**
   - Appears strictly as an optional analytical sanity check for tiny matrices ($m \le 2$).
   - Never used as the primary numerical oracle.
