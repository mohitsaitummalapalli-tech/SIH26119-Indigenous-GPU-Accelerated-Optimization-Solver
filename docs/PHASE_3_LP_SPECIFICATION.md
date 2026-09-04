# Phase 3: LP Solver Core — Mathematical & Engineering Specification

**Document Version:** 1.2.0
**Status:** AUDITED & APPROVED SPECIFICATION (Final Gate)
**Authoritative Baseline Commit:** `7bfa19a097b674d83ca79ce3886c1ed36db9eb33`
**Repository:** `SIH26119-Indigenous-GPU-Accelerated-Optimization-Solver`

---

## 1. Scope & Non-Goals

### 1.1 Objective of Phase 3
Phase 3 establishes the linear programming (LP) solver core for the SIH26119 optimization suite. This phase encompasses:
1. Exact mathematical standardization mapping canonical Phase 1 models into internal standard equality form.
2. Basis state representation and algebraic partitioning.
3. First-principles revised simplex method for canonical minimization.
4. Primal simplex algorithm with deterministic anti-cycling.
5. Two-Phase (Phase I / Phase II) method for artificial-variable initialization, cleanup, and redundant constraint removal.
6. Dual simplex algorithm specification.
7. Abstract basis-factorization numerical contracts integrated with Phase 2 zero-allocation workspaces.
8. Rigorous isolation of numerical decision thresholds.
9. Independent mathematical verification and vertex-enumeration oracle architectures.
10. Deterministic test matrix defining exact benchmark instances.

### 1.2 Non-Goals & Absolute Restrictions
- **NO Simplex Implementation in this Milestone**: No production simplex code, no temporary stubs, and no mock solver routines are permitted in this specification milestone.
- **NO Third-Party Solver Dependencies**: Wrapping or linking external solvers (Gurobi, CPLEX, HiGHS, SCIP, COIN-OR/Clp, GLPK, OR-Tools) is strictly prohibited.
- **NO GPU / CUDA Implementation**: Phase 3 core remains strictly CPU-based. GPU acceleration contracts belong to later phases.
- **NO Weakening of Phase 2 Contracts**: Zero-allocation hot paths and transactional error rollbacks established in Phase 2 remain frozen and inviolable.

---

## 2. Repository Architecture & Subsystem Reuse Inventory

### 2.1 Reusable Phase 1 & Phase 2 Subsystems
The Phase 3 LP specification directly builds on existing, frozen abstractions:

1. **Canonical Model Representation (`src/model/`)**:
   - `Model` ([`src/model/model.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/model/model.hpp)): Provides the immutable problem definition: variables, bounds $[l_j, u_j]$, linear constraints $[L_i, U_i]$, objective sense $\min / \max$, linear coefficients $c_j$, and constant scalar offset $c_0$.
   - `Variable` & `Constraint`: Rich bound queries (`is_equality()`, `is_less_equal()`, `is_greater_equal()`, `is_range()`, `is_free()`, `is_fixed()`).
   - Types ([`src/model/types.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/model/types.hpp)):
     - `VariableIndex` (`uint32_t`)
     - `ConstraintIndex` (`uint32_t`)
     - `DimensionCount` (`uint32_t`)
     - `NonzeroCount` (`uint64_t`)
     - Sentinels: `kInvalidVariableIndex = UINT32_MAX`, `kInvalidConstraintIndex = UINT32_MAX`

2. **Core Error & Result System (`src/core/`)**:
   - `Status` & `Result<T>`: Strict, checked error handling with `StatusCode` (`Ok`, `InvalidArgument`, `NumericalFailure`, etc.). No unhandled exceptions or unchecked raw pointers.

3. **Sparse & Dense Numerical Foundations (`src/numerics/`)**:
   - Index types ([`src/numerics/index.hpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/index.hpp)):
     - `Index = uint32_t`: Coordinate index within a dimension (row/column index).
     - `Dimension = uint32_t`: Matrix/vector dimension count.
     - `NonzeroCount = uint64_t`: Nonzero entry count.
     - Checked conversions: `to_index()`, `to_dimension()`, `to_nonzero_count()`.
   - `Scalar`: `double` (IEEE 754 double precision).
   - `DenseVector`: Contiguous real array with bounds checking, raw pointer access, AXPY, dot product, and scaled norm accumulation.
   - `DenseMatrix`: Contiguous column-major storage ($A_{i,j} = \text{data}[i + j \cdot m]$).
   - `SparseMatrix`: Canonical Compressed Sparse Row (CSR) matrix with sorted column indices, structural zero elimination, SpMV ($y = Ax$), and residual ($r = b - Ax$).
   - `Tolerance` & `approx_equal`: Semantic comparison tolerances ($\text{abs\_tol} = 10^{-12}, \text{rel\_tol} = 10^{-12}$).
   - **Hot-Path Workspace Contract**: Three-argument methods `multiply(x, y, scratch)` and `residual(b, x, r, scratch)` providing guaranteed zero dynamic heap allocations and strict transactional rollback on overflow.

### 2.2 Strict Index-Type Consistency Specification
Phase 3 strictly harmonizes with Phase 1 and Phase 2 types:
- Problem variable coordinates in standard form: `Index` (`uint32_t`).
- Row coordinates: `Index` (`uint32_t`).
- Dimension counts (number of rows $\bar{m}$, columns $\bar{n}$): `Dimension` (`uint32_t`).
- Iteration counters: `uint64_t`.
- Nonzero element counts: `NonzeroCount` (`uint64_t`).
- Conversions between `size_t` containers and `Index` / `Dimension` MUST use `to_index()` and `to_dimension()` from `src/numerics/index.hpp`.

### 2.3 Subsystem Invariants & Explicit Assumptions
- **Allowed Assumptions**:
  - The input `Model` passes `model.validate()` and `model.is_lp()` returns `true`.
  - All original variables in an LP instance are continuous (`VariableType::Continuous`).
  - All constraint and variable bounds are valid ($l \le u$).
- **Prohibited Assumptions**:
  - Never assume $A$ is full row rank.
  - Never assume basis matrix $B$ is well-conditioned without numerical verification.
  - Never assume an initial basic feasible solution is known a priori.
  - Never assume floating-point comparisons can use raw `==` or raw `<=`.

---

## 3. Mathematical Standardization

### 3.1 Canonical Model Definition (Phase 1)
The input optimization problem is given by:
$$\begin{aligned}
\text{opt} \quad & f(x) = c^T x + c_0 \\
\text{subject to} \quad & l \le A x \le u \\
& lb \le x \le ub
\end{aligned}$$
where:
- $\text{opt} \in \{\text{minimize}, \text{maximize}\}$
- $A \in \mathbb{R}^{m \times n}$ with $m = \text{num\_constraints}()$ and $n = \text{num\_variables}()$.
- $c \in \mathbb{R}^n$, $c_0 \in \mathbb{R}$.
- $l, u \in (\mathbb{R} \cup \{-\infty, +\infty\})^m$ ($l_i \le u_i$).
- $lb, ub \in (\mathbb{R} \cup \{-\infty, +\infty\})^n$ ($lb_j \le ub_j$).

### 3.2 Authoritative Internal Standard Form
Phase 3 adopts the **Standard Equality Form with Non-Negative Variables**:
$$\begin{aligned}
\min \quad & z(\bar{x}) = \bar{c}^T \bar{x} + \bar{c}_0 \\
\text{subject to} \quad & \bar{A} \bar{x} = \bar{b} \\
& \bar{x} \ge 0, \quad \bar{b} \ge 0
\end{aligned}$$
where $\bar{A} \in \mathbb{R}^{\bar{m} \times \bar{n}}$, $\bar{b} \in \mathbb{R}^{\bar{m}}_{\ge 0}$, $\bar{x} \in \mathbb{R}^{\bar{n}}_{\ge 0}$, $\bar{c} \in \mathbb{R}^{\bar{n}}$, $\bar{c}_0 \in \mathbb{R}$.

#### Mathematical Justification:
Standard equality form with $\bar{x} \ge 0$ is selected over bounded-variable simplex because:
1. **Unambiguous Basis State**: Nonbasic variables are strictly identically zero ($\bar{x}_N = 0$). In bounded-variable simplex, nonbasic variables can reside at lower or upper bounds, requiring tri-state pivot logic and piecewise ratio tests.
2. **Deterministic Factorization Dimension**: Basis matrix $B$ is always an $\bar{m} \times \bar{m}$ square invertible matrix.
3. **Independent Oracle Equivalence**: Vertices of $\{\bar{x} : \bar{A}\bar{x} = \bar{b}, \bar{x} \ge 0\}$ correspond directly to extreme points of the original polyhedron.

---

### 3.3 Objective Sense Normalization
- If $\text{opt} == \text{minimize}$:
  $$\bar{c} \leftarrow c, \quad \bar{c}_0 \leftarrow c_0$$
- If $\text{opt} == \text{maximize}$:
  $$\bar{c} \leftarrow -c, \quad \bar{c}_0 \leftarrow -c_0$$
  The original optimal objective value is recovered via:
  $$f^*(x^*) = -z^*(\bar{x}^*)$$

---

### 3.4 Complete Variable-Bound Transformations
Every variable $x_j$ ($j \in \{0, \dots, n-1\}$) is mapped to non-negative variables $\bar{x}$ based on its bounds:

| Case | Original Bounds | Transformed Variables | Reconstruction Formula | Objective Coeff Contribution | Objective Constant Offset $\Delta \bar{c}_0$ |
|---|---|---|---|---|---|
| **(a) Standard Non-negative** | $lb_j = 0, ub_j = +\infty$ | $\bar{x}_k \ge 0$ | $x_j = \bar{x}_k$ | $\bar{c}_k = c_j$ | $0$ |
| **(b) Shifted Lower Bound** | $lb_j > -\infty, ub_j = +\infty$ | $\bar{x}_k \ge 0$ | $x_j = \bar{x}_k + lb_j$ | $\bar{c}_k = c_j$ | $+c_j \cdot lb_j$ |
| **(c) Shifted Upper Bound** | $lb_j = -\infty, ub_j < +\infty$ | $\bar{x}_k \ge 0$ | $x_j = ub_j - \bar{x}_k$ | $\bar{c}_k = -c_j$ | $+c_j \cdot ub_j$ |
| **(d) Box Bounded** | $-\infty < lb_j < ub_j < +\infty$ | $\bar{x}_k \ge 0$, $s_j \ge 0$ | $x_j = \bar{x}_k + lb_j$ | $\bar{c}_k = c_j$, $\bar{c}(s_j) = 0$ | $+c_j \cdot lb_j$ |
| **(e) Free Variable** | $lb_j = -\infty, ub_j = +\infty$ | $\bar{x}_k^+ \ge 0, \bar{x}_k^- \ge 0$ | $x_j = \bar{x}_k^+ - \bar{x}_k^-$ | $\bar{c}_k^+ = c_j, \bar{c}_k^- = -c_j$ | $0$ |
| **(f) Fixed Variable** | $lb_j = ub_j = C_j$ | None (eliminated) | $x_j = C_j$ | None | $+c_j \cdot C_j$ |

---

### 3.5 Complete Constraint & Two-Sided Range Transformation

Let constraint $i$ be $l_i \le a_i^T x \le u_i$ with $a_i^T x = \sum_j A_{i,j} x_j$.
After substituting transformed structural variables, let:
$$a_i^T x = \bar{a}_i^T \bar{x}_{\text{struct}} + \kappa_i$$
where $\kappa_i = \sum_{j \in \text{shifted}} A_{i, j} lb_j + \sum_{j \in \text{fixed}} A_{i, j} C_j$.
The constraint becomes:
$$\tilde{l}_i \le \bar{a}_i^T \bar{x}_{\text{struct}} \le \tilde{u}_i$$
where $\tilde{l}_i = l_i - \kappa_i$ and $\tilde{u}_i = u_i - \kappa_i$.

#### Explicit Transformations:
1. **Equality Constraint ($l_i == u_i = b_i$)**:
   $$\bar{a}_i^T \bar{x}_{\text{struct}} = \tilde{b}_i \quad (\text{where } \tilde{b}_i = b_i - \kappa_i)$$
2. **Less-Than-or-Equal Constraint ($l_i = -\infty$, finite $u_i$)**:
   Introduce non-negative slack variable $s_i \ge 0$:
   $$\bar{a}_i^T \bar{x}_{\text{struct}} + s_i = \tilde{u}_i, \quad \bar{c}(s_i) = 0$$
3. **Greater-Than-or-Equal Constraint (finite $l_i$, $u_i = +\infty$)**:
   Introduce non-negative surplus variable $e_i \ge 0$:
   $$\bar{a}_i^T \bar{x}_{\text{struct}} - e_i = \tilde{l}_i, \quad \bar{c}(e_i) = 0$$
4. **Two-Sided Range Constraint ($-\infty < l_i < u_i < +\infty$)**:
   Introduce two explicit non-negative auxiliary variables:
   - Primary surplus variable $s_i \ge 0$
   - Range slack variable $t_i \ge 0$

   Construct two standard equality equations:
   $$\begin{aligned}
   \text{Row 1 (Lower Bound):} \quad & \bar{a}_i^T \bar{x}_{\text{struct}} - s_i = \tilde{l}_i \\
   \text{Row 2 (Range Length):} \quad & s_i + t_i = u_i - l_i
   \end{aligned}$$
   with $\bar{c}(s_i) = 0$, $\bar{c}(t_i) = 0$.
5. **Free Constraint ($l_i = -\infty, u_i = +\infty$)**:
   Mathematically redundant; verified and omitted from standard system.

---

### 3.6 RHS Sign Normalization
For every standard equality row $k \in \{0, \dots, \bar{m}-1\}$ with equation $\bar{a}_k^T \bar{x} = \bar{b}_k$:
- If $\bar{b}_k < 0$:
  $$\bar{a}_k \leftarrow -\bar{a}_k, \quad \bar{b}_k \leftarrow -\bar{b}_k$$
- If $\bar{b}_k \ge 0$: row remains unchanged.

---

## 4. Basis Mathematics & Theoretical Foundations

### 4.1 Partition of the Standard System
Given $\bar{A} \bar{x} = \bar{b}, \bar{x} \ge 0$ with $\bar{A} \in \mathbb{R}^{\bar{m} \times \bar{n}}$ ($\bar{m} \le \bar{n}$) and $\operatorname{rank}(\bar{A}) = \bar{m}$:
- **Basis Index Set**: $\mathcal{B} = \{B(0), B(1), \dots, B(\bar{m}-1)\} \subset \{0, \dots, \bar{n}-1\}$.
- **Nonbasic Index Set**: $\mathcal{N} = \{0, \dots, \bar{n}-1\} \setminus \mathcal{B}$.

$$\bar{A} = \begin{bmatrix} B & N \end{bmatrix}, \quad \bar{x} = \begin{bmatrix} x_B \\ x_N \end{bmatrix}, \quad \bar{c} = \begin{bmatrix} c_B \\ c_N \end{bmatrix}$$
Basic solution: $x_B = B^{-1} \bar{b}, x_N = 0$.

### 4.2 Primal Feasibility & Numerical Singularity Distinction
1. **Basic Feasible Solution (BFS)**:
   $$x_{B, i} \ge -\epsilon_{\text{feas}}, \quad \forall i \in \{0, \dots, \bar{m}-1\}$$
2. **Degenerate BFS**:
   At least one basic coordinate satisfies $|x_{B, i}| \le \epsilon_{\text{feas}}$.
3. **Singularity Threshold vs. Condition Number**:
   - `singularity_threshold` ($\epsilon_{\text{sing}} = 10^{-12}$): A local pivot acceptance criterion during LU factorization. If during Gaussian elimination at step $k$, no available candidate pivot satisfies $|U_{i,k}| > \epsilon_{\text{sing}}$, the basis matrix is numerically rank-deficient / singular at working precision. The solver halts factorization and returns `StatusCode::NumericalFailure`.
   - `condition_number_estimate`: A separate global diagnostic metric ($\kappa(B) = \|B\| \cdot \|B^{-1}\|$). A pivot threshold $\epsilon_{\text{sing}}$ does NOT mathematically imply $\kappa(B) \le \epsilon_{\text{sing}}^{-1}$ (e.g., ill-conditioned triangular matrices can have unit diagonal while exhibiting exponential condition numbers). If the condition estimate exceeds an operational ceiling (e.g. $10^{13}$), the solver flags ill-conditioning to trigger refactorization or basis repair.

---

## 5. Revised Simplex Mathematics (First-Principles Derivation)

### 5.1 Objective Function Expansion
Under the canonical minimization convention:
$$z = c_B^T x_B + c_N^T x_N + \bar{c}_0$$
Substitute $x_B = B^{-1} \bar{b} - B^{-1} N x_N$:
$$z = c_B^T B^{-1} \bar{b} + \bar{c}_0 + \left( c_N^T - c_B^T B^{-1} N \right) x_N$$

### 5.2 Dual Multipliers (BTRAN) & Reduced Costs
$$B^T y = c_B$$
For each nonbasic column $j \in \mathcal{N}$:
$$\bar{c}_j = c_j - y^T \bar{A}_{:, j} = c_j - \bar{A}_{:, j}^T y$$

Optimality: If $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all $j \in \mathcal{N}$, current BFS is optimal.

### 5.3 Direction Vector (FTRAN) & Ratio Test
If $\exists q \in \mathcal{N}$ with $\bar{c}_q < -\epsilon_{\text{opt}}$, solve:
$$B d = \bar{A}_{:, q}$$
- If $d_i \le \epsilon_{\text{pivot}}$ for all $i \in \{0, \dots, \bar{m}-1\}$, problem is **UNBOUNDED**.
- Otherwise:
  $$\theta^* = \min_{i: d_i > \epsilon_{\text{pivot}}} \frac{x_{B, i}}{d_i}, \quad p = \operatorname{argmin}_{i: d_i > \epsilon_{\text{pivot}}} \frac{x_{B, i}}{d_i}$$
Basic updates: $x_{B, i}' = x_{B, i} - \theta^* d_i$ ($\forall i \ne p$), $x_{B, p}' = \theta^*$.
Objective update: $z' = z + \theta^* \bar{c}_q \le z$.

---

## 6. Primal Simplex & Deterministic Pivot Selection

### 6.1 Classical Bland's Rule vs. Numerical Rule
- **Classical Bland's Rule (Exact Arithmetic)**: Select smallest variable index among entering candidates ($\bar{c}_j < 0$) and leaving candidates achieving the exact minimum ratio. Mathematically guarantees zero cycling in exact arithmetic.
- **Tolerance-Aware Deterministic Pivot Selection (Floating-Point Arithmetic)**:
  1. Entering variable:
     $$q = \min \{ j \in \mathcal{N} : \bar{c}_j < -\epsilon_{\text{opt}} \}$$
  2. Leaving candidate set:
     $$\Theta = \left\{ i \in \{0, \dots, \bar{m}-1\} : d_i > \epsilon_{\text{pivot}} \text{ and } \left| \frac{x_{B, i}}{d_i} - \theta^* \right| \le \epsilon_{\text{feas}} \max(1.0, \theta^*) \right\}$$
  3. Deterministic tie-breaking:
     $$p = \operatorname{argmin}_{i \in \Theta} B(i)$$
  In floating-point arithmetic, roundoff error can perturb reduced costs and step lengths; Bland's rule provides **deterministic tie-breaking**, but complete anti-cycling requires periodic refactorization and iteration limits.

---

## 7. Two-Phase Simplex Method (Phase I / Phase II)

### 7.1 Phase I Auxiliary Construction
Given $\bar{A} \bar{x} = \bar{b}$ with $\bar{b} \ge 0$:
$$\min w = \sum_{i=0}^{\bar{m}-1} a_i \quad \text{s.t.} \quad \bar{A} \bar{x} + I_{\bar{m}} a = \bar{b}, \quad \bar{x} \ge 0, \; a \ge 0$$
Initial basis: $B^{(0)} = I_{\bar{m}}$, $x_B = a = \bar{b} \ge 0, x_N = 0$.

### 7.2 Termination Criteria
- If optimal Phase I objective $w^* > \epsilon_{\text{feas}}$: original LP is **INFEASIBLE**. Return `LpSolverStatus::Infeasible`.
- If $w^* \le \epsilon_{\text{feas}}$: feasible solution exists. Proceed to artificial variable cleanup.

### 7.3 Complete Artificial Variable Cleanup & Redundant Row Theorem
Let the final Phase I basis be $B$ with nonbasic partition $\mathcal{N} = \mathcal{N}_{\text{struct}} \cup \mathcal{N}_{\text{art}}$.
For each basic artificial variable $B(k) \in \mathcal{B}$ with $B(k) \ge \bar{n}$ (having $x_{B, k} \le \epsilon_{\text{feas}}$):
The complete tableau row $k$ is:
$$x_{B, k} + \sum_{j \in \mathcal{N}_{\text{struct}}} \alpha_{k, j} \bar{x}_j + \sum_{j \in \mathcal{N}_{\text{art}}} \beta_{k, j} a_{j - \bar{n}} = (B^{-1} \bar{b})_k$$
where $\alpha_{k, j} = e_k^T B^{-1} \bar{A}_{:, j}$ and $\beta_{k, j} = (B^{-1})_{k, j - \bar{n}}$.

In the original problem, artificial variables are restricted to zero ($a = 0$). The row equation restricted to the structural space is:
$$\sum_{j \in \mathcal{N}_{\text{struct}}} \alpha_{k, j} \bar{x}_j = (B^{-1} \bar{b})_k$$

**Structural Pivot vs. Redundancy Cases**:
1. **Case 1 (Structural Pivot Exists)**:
   There exists at least one nonbasic structural variable $j^* \in \mathcal{N}_{\text{struct}}$ such that $|\alpha_{k, j^*}| > \epsilon_{\text{pivot}}$.
   Perform a zero-step pivot ($\theta^* = 0$): variable $j^*$ enters the basis in row $k$, and artificial variable $B(k)$ leaves the basis and is eliminated. The basis size $\bar{m}$ remains unchanged, primal solution $\bar{x}$ remains unchanged, and the basis factorization is updated via standard rank-1 update.
2. **Case 2 (Redundant Row)**:
   For ALL nonbasic structural variables $j \in \mathcal{N}_{\text{struct}}$, $|\alpha_{k, j}| \le \epsilon_{\text{pivot}}$.
   Since basic structural variables have coefficient 0 in row $k$ by definition of canonical tableau, this implies:
   $$(e_k^T B^{-1}) \bar{A} = 0^T$$
   Let vector $\lambda^T = e_k^T B^{-1} \in \mathbb{R}^{\bar{m}}$. Since $B$ is invertible, $\lambda \ne 0$.
   Thus $\lambda^T \bar{A} = 0^T$, and furthermore $\lambda^T \bar{b} = (B^{-1} \bar{b})_k = x_{B, k} \approx 0$.
   This mathematically proves that constraint row $k$ in the standardized system is a linear combination of other rows:
   $$\bar{A}_{k, :} = \sum_{i \ne k} \mu_i \bar{A}_{i, :}, \quad \bar{b}_k = \sum_{i \ne k} \mu_i \bar{b}_i$$
   **Proof of Feasible-Set Preservation**: Any $\bar{x}$ satisfying the other $\bar{m}-1$ equations automatically satisfies row $k$. Deleting row $k$ from $\bar{A}$ and $\bar{b}$ preserves the original feasible set $\{\bar{x} \ge 0 : \bar{A}\bar{x} = \bar{b}\}$ identically.
   **Crucial Property**: The presence of non-zero artificial coefficients $\beta_{k, j} \ne 0$ relates solely to artificial variable dependencies in Phase I and does NOT establish or prevent redundancy of the original LP. Redundancy depends strictly on structural coefficients $\alpha_{k, j}$.
   **Post-Deletion Factorization Rebuild**: Deleting row $k$ reduces the row dimension from $\bar{m}$ to $\bar{m}-1$. The remaining $\bar{m}-1$ basic variables form an $(\bar{m}-1) \times (\bar{m}-1)$ basis matrix. The factorization cannot be updated via rank-1 update; a clean REFACTORIZATION of the reduced $(\bar{m}-1) \times (\bar{m}-1)$ system is required.

---

## 8. Dual Simplex Mathematics

### 8.1 Dual Feasibility Condition
Dual Simplex maintains dual feasibility:
$$\bar{c}_j = c_j - \bar{A}_{:, j}^T y \ge -\epsilon_{\text{opt}}, \quad \forall j \in \mathcal{N}$$
while primal feasibility is violated ($x_{B, i} < -\epsilon_{\text{feas}}$ for some $i$).

### 8.2 Leaving Row & Entering Column Selection
- Leaving row: $l = \operatorname{argmin}_{i: x_{B, i} < -\epsilon_{\text{feas}}} x_{B, i}$.
- BTRAN: $B^T u = e_l \implies v_j = u^T \bar{A}_{:, j}$ for $j \in \mathcal{N}$.
- If $v_j \ge -\epsilon_{\text{pivot}}$ for all $j \in \mathcal{N}$, dual is unbounded $\implies$ primal LP is **INFEASIBLE**.
- Dual ratio test:
  $$q = \operatorname{argmin}_{j \in \mathcal{N} : v_j < -\epsilon_{\text{pivot}}} \left( \frac{\bar{c}_j}{-v_j} \right) \equiv \operatorname{argmin}_{j \in \mathcal{N} : v_j < -\epsilon_{\text{pivot}}} \left( \frac{\bar{c}_j}{|v_j|} \right)$$
  Ties broken deterministically by smallest variable index $j$.

---

## 9. Numerical Linear Algebra & Basis Factorization Contract

### 9.1 Phase 2 Zero-Allocation Hot-Path Integration
Phase 3 adheres strictly to Phase 2 contracts:
- FTRAN ($B d = \bar{A}_{:, q}$) and BTRAN ($B^T y = c_B$) reuse preallocated workspace vectors `DenseVector ftran_workspace` and `DenseVector btran_workspace`.
- SpMV operations utilize the 3-argument API:
  `SparseMatrix::multiply(x, y, scratch)`
- No `std::vector`, `std::make_unique`, `malloc`, or `resize` calls are permitted inside the simplex pivot loop.

### 9.2 Abstract Basis Factorization Contract
Direct matrix inversion ($B^{-1}$) is prohibited.

```cpp
namespace sih26119 {

class BasisFactorization {
public:
    virtual ~BasisFactorization() = default;

    /// Factorizes basis matrix B formed by basic columns.
    [[nodiscard]] virtual Status factorize(
        const SparseMatrix& A,
        std::span<const Index> basic_indices,
        DenseVector& scratch) noexcept = 0;

    /// Solves B * x = rhs (FTRAN) with zero dynamic heap allocation.
    [[nodiscard]] virtual Status solve_primal(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept = 0;

    /// Solves B^T * y = rhs (BTRAN) with zero dynamic heap allocation.
    [[nodiscard]] virtual Status solve_dual(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept = 0;

    /// Updates factorization after a simplex pivot (replacing column p with q).
    [[nodiscard]] virtual Status update(
        Index leaving_row,
        Index entering_col,
        const DenseVector& ftran_dir,
        DenseVector& scratch) noexcept = 0;

    /// Refactorization trigger query.
    [[nodiscard]] virtual bool needs_refactorization(uint64_t pivots_since_factorization) const noexcept = 0;

    /// Condition number estimate for singularity detection.
    [[nodiscard]] virtual Scalar condition_estimate() const noexcept = 0;
};

} // namespace sih26119
```

---

## 10. Distinct Numerical Decision Thresholds

Numerical thresholds are strictly separated across decision boundaries:

```cpp
namespace sih26119 {

struct SimplexTolerances {
    /// Primal feasibility tolerance: x_B >= -feasibility_tol, ||Ax - b||_inf <= feasibility_tol
    Scalar feasibility_tol{1e-8};

    /// Dual optimality tolerance: r_j >= -optimality_tol (minimization)
    Scalar optimality_tol{1e-8};

    /// Pivot selection threshold: minimum denominator magnitude in ratio test
    Scalar pivot_tol{1e-10};

    /// Basis singularity threshold: pivot acceptance threshold during LU factorization
    Scalar singularity_threshold{1e-12};

    /// Decision threshold for comparison to zero (never destructively overwrites numbers)
    Scalar zero_threshold{1e-15};

    /// Phase 1 semantic model comparison tolerance
    Scalar model_bound_tol{1e-12};
};

} // namespace sih26119
```

#### Non-Destructive Zero-Threshold Safety Contract:
- `zero_threshold` ($\epsilon_{\text{zero}} = 10^{-15}$) is a COMPARISON / DECISION threshold only (`std::abs(x) <= eps_zero`).
- The solver NEVER destructively rewrites arbitrary finite computed floating-point numbers to $0.0$.
- Strict classification:
  1. **Stored Value**: Raw IEEE 754 float, preserved without silent alteration.
  2. **Numerical Comparison to Zero**: Query predicate `is_zero(x)` for control flow and branch decisions.
  3. **Structural Sparsity Zero**: Absence of entry in CSR pattern.
  4. **Model Coefficient Zero**: Literal zero terms filtered during immutable model parsing.
  5. **Mathematical Basic/Nonbasic Projection**: Nonbasic variables are algebraically defined to be zero in a basic solution ($x_N \equiv 0$). Basic variables $x_B$ retain their raw computed floating-point values.

---

## 11. Result Semantics & LP Solution Contract

### 11.1 Solver Status Enumeration
```cpp
namespace sih26119 {

enum class LpSolverStatus : uint8_t {
    Optimal,            ///< Globally optimal solution found and verified.
    Infeasible,         ///< Primal problem has no feasible solution (Phase I w* > feas_tol).
    Unbounded,          ///< Objective is unbounded (f(x) -> -inf for min).
    NumericalFailure,   ///< Numerical singularity, excessive round-off, or loss of basis rank.
    IterationLimit,     ///< Maximum pivot limit reached without proving optimality.
    InvalidModel        ///< Input model violates LP invariants or continuous variable requirements.
};

} // namespace sih26119
```

### 11.2 LP Solution Data Transfer Object
```cpp
namespace sih26119 {

struct LpSolution {
    LpSolverStatus status{LpSolverStatus::InvalidModel};
    Scalar objective_value{0.0};   // Includes original Phase 1 c0 constant
    DenseVector primal_variables;   // Size == model.num_variables() (original space)
    DenseVector dual_variables;     // Size == model.num_constraints() (Lagrange multipliers)
    DenseVector reduced_costs;      // Size == model.num_variables()
    uint64_t iterations{0};         // Total simplex pivots executed
    bool is_verified{false};        // True if independent verifier validated certificates
};

} // namespace sih26119
```

---

## 12. Independent LP Verifier & Certificate System

An optimal solution returned by the solver is validated by an independent verifier evaluating the unstandardized `Model` using native Phase 2 SpMV routines.

### 12.1 Independent Primal Verification
1. **Variable Bounds**: $\text{viol}_{\text{var}} = \max_{j=0}^{n-1} \max(0.0, \; lb_j - x_j^*, \; x_j^* - ub_j) \le \epsilon_{\text{feas}}$.
2. **Constraint Bounds**: Compute $r^* = A x^*$ via Phase 2 SpMV: $\text{viol}_{\text{con}} = \max_{i=0}^{m-1} \max(0.0, \; l_i - r_i^*, \; r_i^* - u_i) \le \epsilon_{\text{feas}}$.
3. **Objective Evaluation**: Compute $f(x^*) = c^T x^* + c_0$. Assert $|f(x^*) - z^*| \le \epsilon_{\text{feas}} \max(1.0, |z^*|)$.

### 12.2 Infeasibility & Unboundedness Certificates
1. **Infeasibility (Farkas' Lemma)**: $y \in \mathbb{R}^m$ such that $A^T y \ge -\epsilon_{\text{feas}}, b^T y < -\epsilon_{\text{feas}}$.
2. **Unboundedness**: Extreme ray $d \in \mathbb{R}^n$ with $A d = 0, d \ge 0, c^T d < -\epsilon_{\text{opt}}$.

---

## 13. Independent Reference Oracle for Unit Testing

For testing independence, Phase 3 defines an **Independent Vertex Enumeration Oracle** for small-scale LPs ($n \le 8, m \le 8$):

```
Algorithm VertexEnumerationOracle(A, b, c, c0):
  1. If m == 0 (pure bounds x >= 0):
       If any c_j < -opt_tol: return Unbounded.
       Return Optimal(x = 0, z = c0).
  2. OptimalValue = +Infinity, OptimalVertex = None.
  3. For each combination J of m column indices from {0, ..., n-1}:
       a. Extract square m x m submatrix A_J.
       b. Compute LU factorization with full pivoting.
       c. If rank(A_J) < m or any pivot <= singularity_threshold: skip.
       d. Solve A_J * x_J = b.
       e. If x_J >= -feas_tol for all components:
            Set full vector x with x_J and x_N = 0.
            Compute z = c^T x + c0.
            If z < OptimalValue:
                OptimalValue = z, OptimalVertex = x.
  4. If OptimalVertex is None: return Infeasible.
  5. Check unboundedness by exploring recession rays along active edges.
  6. Return Optimal(OptimalValue, OptimalVertex).
```

---

## 14. Deterministic Test Matrix

The deterministic test matrix defines concrete numerical instances with exact expected outputs:

| Test ID | Category | LP Instance Equations & Bounds | Expected Status | Expected Objective | Expected Solution $x^*$ | Mathematical Purpose |
|---|---|---|---|---|---|---|
| `TEST-LP-01` | Trivial | $\min 2x_1$ s.t. $x_1 \ge 3$ | `Optimal` | $6.0$ | $x_1 = 3.0$ | Bound-to-constraint standardization, 1-step pivot. |
| `TEST-LP-02` | Equality | $\min 3x_1 + 2x_2$ s.t. $x_1 + x_2 = 5, x \ge 0$ | `Optimal` | $10.0$ | $x_1 = 0.0, x_2 = 5.0$ | Direct equality row without slack variables. |
| `TEST-LP-03` | Inequality $\le$ | $\max 5x_1 + 4x_2$ s.t. $6x_1 + 4x_2 \le 24, x_1 + 2x_2 \le 6, x \ge 0$ | `Optimal` | $21.0$ | $x_1 = 3.0, x_2 = 1.5$ | Slack variables, direct Phase II initialization. |
| `TEST-LP-04` | Inequality $\ge$ | $\min 2x_1 + 3x_2$ s.t. $x_1 + x_2 \ge 4, x_1 + 3x_2 \ge 6, x \ge 0$ | `Optimal` | $9.0$ | $x_1 = 3.0, x_2 = 1.0$ | Surplus variables, Two-Phase artificial initialization. |
| `TEST-LP-05` | Free Variable | $\min x_1 + 2x_2$ s.t. $x_1 + x_2 = 10, x_1 \in \mathbb{R}, x_2 \ge 0$ | `Optimal` | $10.0$ | $x_1 = 10.0, x_2 = 0.0$ | $x_1 = x_1^+ - x_1^-$ decomposition and reconstruction. |
| `TEST-LP-06` | Lower Bound Shift | $\min 4x_1$ s.t. $x_1 \ge 5$ | `Optimal` | $20.0$ | $x_1 = 5.0$ | Variable shift $x_1 = x_1' + 5$, objective offset accumulation. |
| `TEST-LP-07` | Upper Bound | $\max x_1$ s.t. $x_1 \le 12, x_1 \ge 0$ | `Optimal` | $12.0$ | $x_1 = 12.0$ | Upper bound slack constraint. |
| `TEST-LP-08` | Box Bounds | $\min 3x_1 - 2x_2$ s.t. $x_1 + x_2 \le 10, 2 \le x_1 \le 5, -3 \le x_2 \le 4$ | `Optimal` | $-2.0$ | $x_1 = 2.0, x_2 = 4.0$ | Combined variable shifts and upper bound slack constraints. |
| `TEST-LP-09` | Fixed Variable | $\min 2x_1 + 5x_2$ s.t. $x_1 = 3, x_1 + x_2 \le 10, x_2 \ge 0$ | `Optimal` | $6.0$ | $x_1 = 3.0, x_2 = 0.0$ | Constant elimination from matrix, RHS shift. |
| `TEST-LP-10` | Infeasible LP | $\min x_1$ s.t. $x_1 \le 2, x_1 \ge 4, x_1 \ge 0$ | `Infeasible` | N/A | N/A | Phase I termination with $w^* > 0$, status `Infeasible`. |
| `TEST-LP-11` | Unbounded LP | $\min -2x_1 + x_2$ s.t. $x_1 - x_2 \ge 0, x \ge 0$ | `Unbounded` | $-\infty$ | N/A | Direction vector $d \le 0$ in ratio test, status `Unbounded`. |
| `TEST-LP-12` | Degeneracy | $\min -x_1 - x_2$ s.t. $x_1 \le 2, x_2 \le 2, x_1 + x_2 \le 2, x \ge 0$ | `Optimal` | $-2.0$ | $(2, 0)$ or $(0, 2)$ | Zero step length $\theta^* = 0$, Bland's rule determinism. |
| `TEST-LP-13` | Cycling (Beale) | Beale's cycling LP: $\min -0.75x_1 + 20x_2 - 0.5x_3 + 6x_4$ | `Optimal` | $-0.05$ | Exact fractional | Proves Bland's rule terminates finitely where Dantzig cycles. |
| `TEST-LP-14` | Redundant Row | $\min x_1 + x_2$ s.t. $x_1 + x_2 = 2, 2x_1 + 2x_2 = 4, x \ge 0$ | `Optimal` | $2.0$ | Sum $= 2.0$ | Phase I artificial variable cleanup in degenerate row. |
| `TEST-LP-15` | Objective $c_0$ | $\min 2x_1 + 3x_2 + 42.5$ s.t. $x_1 + x_2 \ge 1, x \ge 0$ | `Optimal` | $44.5$ | $(1, 0)$ | Exact preservation of $c_0$ in primal and dual objective. |
| `TEST-LP-16` | Range Constraint | $\min x_1 + x_2$ s.t. $5 \le 2x_1 + 3x_2 \le 15, x \ge 0$ | `Optimal` | $1.66666667$ | $(0, 5/3)$ | Range decomposition into surplus and slack. |

---

## 15. Software Architecture & File Layout

Proposed source tree organization under `src/` and `tests/`:
```
src/
├── solver/
│   └── lp/
│       ├── README.md                  # LP solver layer documentation
│       ├── lp_types.hpp               # LpSolverStatus, LpSolution, basis enums
│       ├── lp_standard_form.hpp       # Standard form data structure
│       ├── lp_standard_form.cpp       # Model -> StandardForm mapping and reconstruction
│       ├── basis_factorization.hpp    # Abstract BasisFactorization interface
│       ├── basis_state.hpp            # Basis index state and partition manager
│       ├── revised_simplex.hpp        # RevisedSimplex solver engine
│       ├── revised_simplex.cpp        # FTRAN, BTRAN, pricing, ratio test, pivot loop
│       ├── lp_verifier.hpp            # Independent solution certificate verifier
│       └── lp_verifier.cpp            # Residual, bound, and complementary slackness audit
tests/
├── unit/
│   ├── test_lp_standardization.cpp    # Round-trip and mapping equivalence tests
│   ├── test_revised_simplex.cpp       # Simplex execution against test matrix
│   ├── test_lp_oracle.cpp             # Vertex enumeration oracle comparison tests
│   └── test_lp_verifier.cpp           # Verifier certificate acceptance/rejection tests
```

---

## 16. Acceptance Gate & Conclusion

Phase 3 implementation will proceed only when:
1. `docs/PHASE_3_LP_SPECIFICATION.md` is approved.
2. `docs/PHASE_3_MATHEMATICAL_AUDIT.md` verifies mathematical soundness.
3. No simplex solver code or third-party packages are added prior to specification gate sign-off.
4. Clean build and 9/9 CTest passing status is preserved.
