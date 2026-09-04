# Phase 3: LP Solver Core — Mathematical & Engineering Specification

**Document Version:** 1.1.0
**Status:** APPROVED & AUDITED SPECIFICATION
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
5. Two-Phase (Phase I / Phase II) method for artificial-variable initialization and redundant constraint removal.
6. Dual simplex algorithm specification.
7. Abstract basis-factorization numerical contracts integrated with Phase 2 zero-allocation workspaces.
8. Rigorous isolation of numerical tolerances across decision boundaries.
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
- Conversions between `size_t` containers and `Index` / `Dimension` MUST use `to_index()` and `to_dimension()` from `src/numerics/index.hpp`. Silent signed integer conversions (e.g. `int64_t`) are prohibited.

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

#### Equivalence Proofs:
- **Case (b)**: Since $lb_j$ is finite and fixed, the map $\phi: [lb_j, \infty) \to [0, \infty)$ defined by $\phi(x_j) = x_j - lb_j = \bar{x}_k$ is an affine bijection. Substituting $x_j = \bar{x}_k + lb_j$ into $c_j x_j$ yields $c_j \bar{x}_k + c_j lb_j$. Constraint column $A_{:, j} x_j = A_{:, j} \bar{x}_k + lb_j A_{:, j}$. Moving the constant vector to RHS preserves the constraint set identically.
- **Case (c)**: The map $\psi: (-\infty, ub_j] \to [0, \infty)$ defined by $\psi(x_j) = ub_j - x_j = \bar{x}_k$ is an affine bijection. Substituting $x_j = ub_j - \bar{x}_k$ yields $c_j x_j = -c_j \bar{x}_k + c_j ub_j$. Constraint column $A_{:, j} x_j = -A_{:, j} \bar{x}_k + ub_j A_{:, j}$.
- **Case (d)**: Let $\bar{x}_k = x_j - lb_j \ge 0$. The condition $x_j \le ub_j$ becomes $\bar{x}_k \le ub_j - lb_j$. Introducing slack $s_j \ge 0$ via the equality $\bar{x}_k + s_j = ub_j - lb_j$ restricts $\bar{x}_k \in [0, ub_j - lb_j]$ since $s_j \ge 0$ and $ub_j - lb_j > 0$.
- **Case (e)**: Any real number $x \in \mathbb{R}$ can be expressed as $x = x^+ - x^-$ where $x^+ = \max(x, 0) \ge 0$ and $x^- = \max(-x, 0) \ge 0$. In standard form, columns are $A_{:, j}$ and $-A_{:, j}$. At any basic solution, linear dependence prevents both $x^+$ and $x^-$ from being basic simultaneously unless degenerate at zero.
- **Case (f)**: When $lb_j = ub_j = C_j$, the feasible set restricts $x_j \in \{C_j\}$. Direct substitution eliminates the variable coordinate, shifting constraint RHS by $-C_j A_{:, j}$ and objective offset by $+c_j C_j$.

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
   - New variables: $0$.
   - New equations: $1$.

2. **Less-Than-or-Equal Constraint ($l_i = -\infty$, finite $u_i$)**:
   Introduce non-negative slack variable $s_i \ge 0$:
   $$\bar{a}_i^T \bar{x}_{\text{struct}} + s_i = \tilde{u}_i, \quad \bar{c}(s_i) = 0$$
   - New variables: $1$ ($s_i \ge 0$).
   - New equations: $1$.

3. **Greater-Than-or-Equal Constraint (finite $l_i$, $u_i = +\infty$)**:
   Introduce non-negative surplus variable $e_i \ge 0$:
   $$\bar{a}_i^T \bar{x}_{\text{struct}} - e_i = \tilde{l}_i, \quad \bar{c}(e_i) = 0$$
   - New variables: $1$ ($e_i \ge 0$).
   - New equations: $1$.

4. **Two-Sided Range Constraint ($-\infty < l_i < u_i < +\infty$)**:
   The range constraint enforces both $\bar{a}_i^T \bar{x}_{\text{struct}} \ge \tilde{l}_i$ and $\bar{a}_i^T \bar{x}_{\text{struct}} \le \tilde{u}_i$.
   We introduce **TWO** explicit non-negative auxiliary variables:
   - Primary surplus variable $s_i \ge 0$
   - Range slack variable $t_i \ge 0$

   Construct **TWO** standard equality equations:
   $$\begin{aligned}
   \text{Row 1 (Lower Bound):} \quad & \bar{a}_i^T \bar{x}_{\text{struct}} - s_i = \tilde{l}_i \\
   \text{Row 2 (Range Length):} \quad & s_i + t_i = u_i - l_i
   \end{aligned}$$
   with $\bar{c}(s_i) = 0$, $\bar{c}(t_i) = 0$.

   **Feasible-Set Equivalence Proof**:
   - **Forward**: Suppose $x$ satisfies $l_i \le a_i^T x \le u_i$. Then $\tilde{l}_i \le \bar{a}_i^T \bar{x}_{\text{struct}} \le \tilde{u}_i$. Define $s_i = \bar{a}_i^T \bar{x}_{\text{struct}} - \tilde{l}_i$ and $t_i = \tilde{u}_i - \bar{a}_i^T \bar{x}_{\text{struct}}$. Since $\bar{a}_i^T \bar{x}_{\text{struct}} \ge \tilde{l}_i$, $s_i \ge 0$. Since $\bar{a}_i^T \bar{x}_{\text{struct}} \le \tilde{u}_i$, $t_i \ge 0$. Adding gives $s_i + t_i = \tilde{u}_i - \tilde{l}_i = (u_i - \kappa_i) - (l_i - \kappa_i) = u_i - l_i$. Both equations hold with $s_i, t_i \ge 0$.
   - **Converse**: Suppose $(\bar{x}_{\text{struct}}, s_i, t_i)$ satisfies the two equations with $s_i \ge 0, t_i \ge 0$. From Row 1, $\bar{a}_i^T \bar{x}_{\text{struct}} = \tilde{l}_i + s_i \ge \tilde{l}_i$ (since $s_i \ge 0$). From Row 2, $s_i = (u_i - l_i) - t_i$. Substituting into Row 1 gives $\bar{a}_i^T \bar{x}_{\text{struct}} = \tilde{l}_i + (u_i - l_i) - t_i = \tilde{u}_i - t_i \le \tilde{u}_i$ (since $t_i \ge 0$). Thus $\tilde{l}_i \le \bar{a}_i^T \bar{x}_{\text{struct}} \le \tilde{u}_i$, which is identically $l_i \le a_i^T x \le u_i$.
   - **Objective and RHS Preservation**: $\bar{c}(s_i) = 0$ and $\bar{c}(t_i) = 0$ preserve the objective. Row 2 RHS is $u_i - l_i > 0$, guaranteed strictly non-negative.

5. **Free Constraint ($l_i = -\infty, u_i = +\infty$)**:
   Mathematically vacuous; verified and eliminated from $\bar{A} \bar{x} = \bar{b}$.

---

### 3.6 RHS Sign Normalization
For every standard equality row $k \in \{0, \dots, \bar{m}-1\}$ with equation $\bar{a}_k^T \bar{x} = \bar{b}_k$:
- If $\bar{b}_k < 0$:
  $$\bar{a}_k \leftarrow -\bar{a}_k, \quad \bar{b}_k \leftarrow -\bar{b}_k$$
- If $\bar{b}_k \ge 0$: row remains unchanged.

#### Verification of Invariants:
1. **Feasibility**: For any scalar equation, $u = v \iff -u = -v$. Solution set is invariant.
2. **Basis Initialization in Phase I**: Phase I adds $+a_k$ to row $k$: $\bar{a}_k^T \bar{x} + a_k = \bar{b}_k$. Setting $\bar{x} = 0$ yields $a_k = \bar{b}_k \ge 0$, establishing a valid non-negative initial BFS with identity basis $B^{(0)} = I_{\bar{m}}$.
3. **Objective Invariance**: Constraint row scaling does not affect objective cost vector $\bar{c}$ or constant offset $\bar{c}_0$.

---

## 4. Basis Mathematics & Theoretical Foundations

### 4.1 Partition of the Standard System
Given $\bar{A} \bar{x} = \bar{b}, \bar{x} \ge 0$ with $\bar{A} \in \mathbb{R}^{\bar{m} \times \bar{n}}$ ($\bar{m} \le \bar{n}$) and $\operatorname{rank}(\bar{A}) = \bar{m}$:
- **Basis Index Set**: An ordered sequence of $\bar{m}$ column indices:
  $$\mathcal{B} = \{B(0), B(1), \dots, B(\bar{m}-1)\} \subset \{0, \dots, \bar{n}-1\}$$
- **Nonbasic Index Set**: The remaining $\bar{n} - \bar{m}$ indices:
  $$\mathcal{N} = \{N(0), N(1), \dots, N(\bar{n}-\bar{m}-1)\} = \{0, \dots, \bar{n}-1\} \setminus \mathcal{B}$$

Matrix partition:
$$\bar{A} = \begin{bmatrix} B & N \end{bmatrix}, \quad \bar{x} = \begin{bmatrix} x_B \\ x_N \end{bmatrix}, \quad \bar{c} = \begin{bmatrix} c_B \\ c_N \end{bmatrix}$$
where $B \in \mathbb{R}^{\bar{m} \times \bar{m}}$ is the nonsingular basis matrix.

### 4.2 Derivation of the Basic Solution
From $\bar{A} \bar{x} = \bar{b}$:
$$B x_B + N x_N = \bar{b}$$
Multiplying by $B^{-1}$:
$$x_B = B^{-1} \bar{b} - B^{-1} N x_N$$
Setting nonbasic variables to zero ($x_N = 0$) defines the **basic solution**:
$$x_B = B^{-1} \bar{b}, \quad x_N = 0$$

### 4.3 Primal Feasibility & Degeneracy
1. **Basic Feasible Solution (BFS)**:
   A basic solution is feasible if and only if:
   $$x_B = B^{-1} \bar{b} \ge 0$$
   Numerically evaluated with feasibility tolerance $\epsilon_{\text{feas}}$:
   $$x_{B, i} \ge -\epsilon_{\text{feas}}, \quad \forall i \in \{0, \dots, \bar{m}-1\}$$
2. **Degenerate BFS**:
   A BFS is degenerate if there exists at least one basic coordinate $i$ such that:
   $$|x_{B, i}| \le \epsilon_{\text{feas}}$$
3. **Singular Basis**:
   If the condition estimate $\kappa(B) > \epsilon_{\text{sing}}^{-1}$ or any pivot in the LU factorization has $|U_{k,k}| \le \epsilon_{\text{sing}}$, the basis is numerically singular. The solver terminates with `StatusCode::NumericalFailure`.

---

## 5. Revised Simplex Mathematics (First-Principles Derivation)

### 5.1 Objective Function Expansion
Under the canonical minimization convention:
$$z = c_B^T x_B + c_N^T x_N + \bar{c}_0$$
Substitute $x_B = B^{-1} \bar{b} - B^{-1} N x_N$:
$$\begin{aligned}
z &= c_B^T \left( B^{-1} \bar{b} - B^{-1} N x_N \right) + c_N^T x_N + \bar{c}_0 \\
&= c_B^T B^{-1} \bar{b} + \bar{c}_0 + \left( c_N^T - c_B^T B^{-1} N \right) x_N
\end{aligned}$$

### 5.2 Dual Multiplier Vector (BTRAN)
Define the simplex dual multiplier vector $y \in \mathbb{R}^{\bar{m}}$ by:
$$y^T = c_B^T B^{-1} \iff B^T y = c_B$$
Solved via backward transformation (BTRAN).

### 5.3 Reduced Costs & Optimality Condition
The objective function becomes:
$$z = y^T \bar{b} + \bar{c}_0 + \sum_{j \in \mathcal{N}} \bar{c}_j x_j$$
where the **reduced cost** of nonbasic column $j \in \mathcal{N}$ with column vector $\bar{A}_{:, j}$ is:
$$\bar{c}_j = c_j - y^T \bar{A}_{:, j} = c_j - \bar{A}_{:, j}^T y$$

**Optimality Theorem (Minimization)**:
At the current basic solution, $x_N = 0$. For any feasible solution $x$, $x_N \ge 0$.
If $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all $j \in \mathcal{N}$, then:
$$z(x) - z(x^*) = \sum_{j \in \mathcal{N}} \bar{c}_j x_j \ge 0 \implies z(x) \ge z(x^*)$$
Hence, the current BFS is globally optimal.

### 5.4 Search Direction (FTRAN)
If there exists $q \in \mathcal{N}$ such that $\bar{c}_q < -\epsilon_{\text{opt}}$, increasing $x_q$ from $0$ to $\theta > 0$ decreases $z$.
Let $x_q = \theta$ and $x_j = 0$ for $j \in \mathcal{N} \setminus \{q\}$. The basic variables adjust as:
$$x_B(\theta) = B^{-1} \bar{b} - \theta B^{-1} \bar{A}_{:, q} = x_B - \theta d$$
where $d \in \mathbb{R}^{\bar{m}}$ is the **direction vector** solved via forward transformation (FTRAN):
$$B d = \bar{A}_{:, q}$$

### 5.5 Minimum Ratio Test & Leaving Variable
Primal feasibility requires:
$$x_{B, i}(\theta) = x_{B, i} - \theta d_i \ge 0, \quad \forall i \in \{0, \dots, \bar{m}-1\}$$
- **Unboundedness**: If $d_i \le \epsilon_{\text{pivot}}$ for all $i \in \{0, \dots, \bar{m}-1\}$, $\theta$ can increase to $+\infty$ without violating non-negativity. Since $\bar{c}_q < 0$, $z(\theta) = z_0 + \theta \bar{c}_q \to -\infty$. The LP is **UNBOUNDED**.
- **Ratio Test**: For rows where $d_i > \epsilon_{\text{pivot}}$:
  $$\theta_i = \frac{x_{B, i}}{d_i}$$
  The maximum feasible step length is:
  $$\theta^* = \min_{i : d_i > \epsilon_{\text{pivot}}} \frac{x_{B, i}}{d_i}$$
  Let row $p$ achieve this minimum ratio:
  $$p = \operatorname{argmin}_{i : d_i > \epsilon_{\text{pivot}}} \frac{x_{B, i}}{d_i}$$
  Variable $B(p)$ leaves the basis and becomes nonbasic at zero.

### 5.6 Basis & Objective Updates
1. Basic solution update:
   $$x_{B, i}' = x_{B, i} - \theta^* d_i \quad (\forall i \ne p), \quad x_{B, p}' = \theta^*$$
2. Basis partition update:
   $$\mathcal{B}' = (\mathcal{B} \setminus \{B(p)\}) \cup \{q\}$$
3. Objective value update:
   $$z' = z + \theta^* \bar{c}_q \le z \quad (\text{strictly decreasing if } \theta^* > 0)$$

---

## 6. Primal Simplex & Anti-Cycling Specification

### 6.1 Classical Bland's Rule vs. Numerical Deterministic Rule
- **Classical Bland's Rule (Exact Arithmetic)**:
  1. Entering variable: $q = \min \{ j \in \mathcal{N} : \bar{c}_j < 0 \}$.
  2. Leaving variable: $p = \operatorname{argmin}_{i \in \Theta} B(i)$, where $\Theta = \operatorname{argmin}_{i: d_i > 0} (x_{B, i} / d_i)$.
  In exact rational arithmetic, Bland's rule guarantees finite termination with zero cycling.
- **Tolerance-Aware Deterministic Pivot Selection (Floating-Point Arithmetic)**:
  In IEEE 754 double precision, exact mathematical equality between ratios is rare due to roundoff. Phase 3 specifies the numerical rule:
  1. **Entering Candidate**:
     $$q = \min \{ j \in \mathcal{N} : \bar{c}_j < -\epsilon_{\text{opt}} \}$$
  2. **Tied-Ratio Leaving Candidates**:
     Compute $\theta^* = \min_{i: d_i > \epsilon_{\text{pivot}}} (x_{B, i} / d_i)$. Form the candidate set:
     $$\Theta = \left\{ i \in \{0, \dots, \bar{m}-1\} : d_i > \epsilon_{\text{pivot}} \text{ and } \left| \frac{x_{B, i}}{d_i} - \theta^* \right| \le \epsilon_{\text{feas}} \max(1.0, \theta^*) \right\}$$
  3. **Deterministic Tie-Breaking**:
     $$p = \operatorname{argmin}_{i \in \Theta} B(i)$$
- **Mathematical Limitation**: In floating-point arithmetic, roundoff error can perturb basis updates such that mathematical cycling prevention is NOT guaranteed by Bland's rule alone. Bland's rule provides deterministic tie-breaking. Complete numerical stalling prevention requires periodic basis refactorization and iteration limits.

---

## 7. Two-Phase Simplex Method (Phase I / Phase II)

### 7.1 Phase I Auxiliary Construction
Given $\bar{A} \bar{x} = \bar{b}$ with $\bar{b} \ge 0$:
1. Introduce an artificial variable $a_i \ge 0$ for each row $i \in \{0, \dots, \bar{m}-1\}$:
   $$\bar{A} \bar{x} + I_{\bar{m}} a = \bar{b}, \quad \bar{x} \ge 0, \; a \ge 0$$
2. Phase I objective:
   $$\min w = \sum_{i=0}^{\bar{m}-1} a_i$$
3. Initial Phase I basis: $B = I_{\bar{m}}$, with $x_B = a = \bar{b} \ge 0, x_N = 0$.

### 7.2 Phase I Infeasibility Termination
Solve Phase I to optimality using primal simplex. Let $w^*$ be the optimal value:
- **Case 1: $w^* > \epsilon_{\text{feas}}$**:
  Since $a_i \ge 0$, $\min \sum a_i > 0$ rigorously proves there exists no non-negative $\bar{x}$ satisfying $\bar{A} \bar{x} = \bar{b}$. The original LP is **INFEASIBLE**. Terminate with `LpSolverStatus::Infeasible`.
- **Case 2: $w^* \le \epsilon_{\text{feas}}$**:
  A feasible solution exists. Proceed to artificial variable cleanup.

### 7.3 Artificial Variable Cleanup & Redundant Row Elimination
At Phase I termination with $w^* \le \epsilon_{\text{feas}}$, all artificial variables have value $a_i \le \epsilon_{\text{feas}}$:
1. **Nonbasic Artificials**: Discard all artificial columns from the model.
2. **Basic Artificials (Degenerate Rows)**:
   For each row $k$ where $B(k)$ is an artificial variable (with $x_{B, k} \le \epsilon_{\text{feas}}$):
   - Compute tableau row $v^T = e_k^T B^{-1} \bar{A}$.
   - Inspect coefficients for nonbasic structural variables $j \in \mathcal{N}_{\text{struct}}$:
     - If $\exists j \in \mathcal{N}_{\text{struct}}$ such that $|v_j| > \epsilon_{\text{pivot}}$:
       Perform a degenerate zero-length pivot ($\theta^* = 0$), entering structural variable $j$ and expelling artificial variable $B(k)$.
     - If $|v_j| \le \epsilon_{\text{pivot}}$ for all $j \in \mathcal{N}_{\text{struct}}$:
       Row $k$ has $e_k^T B^{-1} \bar{A} = 0^T$ and $e_k^T B^{-1} \bar{b} = x_{B, k} \approx 0$. This proves constraint row $k$ is a linear combination of other rows—i.e., it is a **REDUNDANT CONSTRAINT**.
       - Row $k$ is deleted from $\bar{A}$ and $\bar{b}$.
       - Row dimension $\bar{m}$ decreases by 1.
       - Basis factorization is rebuilt for the reduced $(\bar{m}-1) \times (\bar{m}-1)$ system.
3. **Phase II Transition**: Restore original objective $\bar{c}$, form basis matrix from remaining structural columns, and solve Phase II to optimality.

---

## 8. Dual Simplex Mathematics

### 8.1 Dual Feasibility Condition
Dual Simplex maintains dual feasibility:
$$\bar{c}_j = c_j - \bar{A}_{:, j}^T y \ge -\epsilon_{\text{opt}}, \quad \forall j \in \mathcal{N}$$
while primal feasibility is violated ($x_{B, i} < -\epsilon_{\text{feas}}$ for some $i$).

### 8.2 Leaving Row Selection
If $x_{B, i} \ge -\epsilon_{\text{feas}}$ for all $i$, solution is primal feasible and optimal.
Otherwise, select leaving row $l$ by deterministic minimum:
$$l = \operatorname{argmin}_{i: x_{B, i} < -\epsilon_{\text{feas}}} x_{B, i}$$
(Ties broken by smallest basic variable index $B(i)$).

### 8.3 Entering Column Selection (Dual Ratio Test)
1. Compute pivot row $v^T = e_l^T B^{-1} N$ via BTRAN ($B^T u = e_l, v_j = u^T \bar{A}_{:, j}$).
2. **Dual Unboundedness (Primal Infeasibility)**:
   If $v_j \ge -\epsilon_{\text{pivot}}$ for all $j \in \mathcal{N}$:
   For any dual step $\gamma \le 0$, $\bar{c}_j - \gamma v_j \ge 0$. As $\gamma \to -\infty$, dual objective $b^T y + \gamma x_{B, l} \to +\infty$ (since $x_{B, l} < 0$).
   The dual problem is unbounded $\implies$ the primal LP is **INFEASIBLE**.
3. **Dual Ratio Test**:
   To preserve dual feasibility $\bar{c}_j - \gamma v_j \ge 0$ for $v_j < -\epsilon_{\text{pivot}}$:
   $$\gamma = \frac{\bar{c}_{j^*}}{v_{j^*}} \implies j^* = \operatorname{argmin}_{j \in \mathcal{N} : v_j < -\epsilon_{\text{pivot}}} \left( \frac{\bar{c}_j}{-v_j} \right) \equiv \operatorname{argmin}_{j \in \mathcal{N} : v_j < -\epsilon_{\text{pivot}}} \left( \frac{\bar{c}_j}{|v_j|} \right)$$
   Ties broken deterministically by smallest variable index $j$.

---

## 9. Numerical Linear Algebra & Basis Factorization Contract

### 9.1 Phase 2 Zero-Allocation Hot-Path Integration
Phase 2 established that numerical hot paths must operate without dynamic heap allocations via caller-owned workspaces. Phase 3 strictly adheres:
- FTRAN ($B d = \bar{A}_{:, q}$) and BTRAN ($B^T y = c_B$) reuse preallocated workspace vectors `DenseVector ftran_workspace` and `DenseVector btran_workspace`.
- SpMV operations utilize the 3-argument API:
  `SparseMatrix::multiply(x, y, scratch)`
- No `std::vector`, `std::make_unique`, `malloc`, or `resize` calls are permitted inside the simplex pivot loop.

### 9.2 Abstract Basis Factorization Contract
Direct matrix inversion ($B^{-1}$) is prohibited:
- Destroys matrix sparsity.
- Requires $O(\bar{m}^3)$ arithmetic per update.
- Suffers from catastrophic numerical instability and round-off accumulation.

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

## 10. Distinct Numerical Tolerances Specification

Simplex decision making requires distinct numerical thresholds tailored to specific mathematical domains:

```cpp
namespace sih26119 {

struct SimplexTolerances {
    /// Primal feasibility tolerance: x_B >= -feasibility_tol, ||Ax - b||_inf <= feasibility_tol
    Scalar feasibility_tol{1e-8};

    /// Dual optimality tolerance: r_j >= -optimality_tol (minimization)
    Scalar optimality_tol{1e-8};

    /// Pivot selection threshold: minimum denominator magnitude in ratio test
    Scalar pivot_tol{1e-10};

    /// Basis singularity threshold: rejection threshold for ill-conditioned bases
    Scalar singularity_threshold{1e-12};

    /// Structural zero tolerance: flush tiny numerical noise to exact zero
    Scalar zero_threshold{1e-15};

    /// Phase 1 semantic model comparison tolerance
    Scalar model_bound_tol{1e-12};
};

} // namespace sih26119
```

#### Mathematical Justification for Double Precision:
1. $\epsilon_{\text{feas}} = 10^{-8}$: Scaled approximately as $\sqrt{\epsilon_{\text{mach}}}$ ($\approx 1.5 \times 10^{-8}$ in IEEE 754). Provides 8 digits of constraint satisfaction while tolerating accumulated rounding error.
2. $\epsilon_{\text{opt}} = 10^{-8}$: Ensures reliable termination without premature convergence or cycling on near-zero reduced costs.
3. $\epsilon_{\text{pivot}} = 10^{-10}$: Roughly $10^6 \times \epsilon_{\text{mach}}$, preventing division by near-zero denominators that would cause catastrophic loss of significance.
4. $\epsilon_{\text{sing}} = 10^{-12}$: Limits condition number $\kappa(B) \le 10^{12}$, retaining at least $\approx 4$ reliable significant digits.
5. $\epsilon_{\text{zero}} = 10^{-15}$: Approximately $4.5 \times \epsilon_{\text{mach}}$, flushing subnormals and machine-level cancellation artifacts.

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

An optimal solution returned by the solver is never accepted on trust. It must be validated by an independent verifier that evaluates the original, unstandardized `Model` using native Phase 2 SpMV routines.

### 12.1 Independent Primal Verification
Given candidate solution $x^* \in \mathbb{R}^n$:
1. **Variable Bounds**:
   $$\text{viol}_{\text{var}} = \max_{j=0}^{n-1} \max(0.0, \; lb_j - x_j^*, \; x_j^* - ub_j) \le \epsilon_{\text{feas}}$$
2. **Constraint Bounds**:
   Compute $r^* = A x^*$ using native Phase 2 SpMV.
   $$\text{viol}_{\text{con}} = \max_{i=0}^{m-1} \max(0.0, \; l_i - r_i^*, \; r_i^* - u_i) \le \epsilon_{\text{feas}}$$
3. **Objective Evaluation**:
   Compute $f(x^*) = c^T x^* + c_0$. Assert $|f(x^*) - z^*| \le \epsilon_{\text{feas}} \max(1.0, |z^*|)$.

### 12.2 Infeasibility Certificate (Farkas' Lemma)
For standard form $Ax = b, x \ge 0$, infeasibility is certified by a dual vector $y \in \mathbb{R}^m$ satisfying:
$$A^T y \ge -\epsilon_{\text{feas}}, \quad b^T y < -\epsilon_{\text{feas}}$$
**Proof**: If $x \ge 0$ existed with $Ax = b$, then $y^T b = y^T A x = (A^T y)^T x \ge 0$, contradicting $b^T y < 0$.

### 12.3 Unboundedness Certificate
Unboundedness for $\min c^T x$ is certified by finding an extreme ray $d \in \mathbb{R}^n$ such that:
$$A d = 0, \quad d \ge 0, \quad c^T d < -\epsilon_{\text{opt}}$$
**Proof**: For any feasible $x^{(0)}$ and $\lambda \ge 0$, $x(\lambda) = x^{(0)} + \lambda d \ge 0$, $A x(\lambda) = b$, and $c^T x(\lambda) \to -\infty$ as $\lambda \to \infty$.

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
       c. If rank(A_J) < m or cond(A_J) > 1e12: skip (singular / rank-deficient).
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
The test suite executes this independent oracle against every small benchmark LP, guaranteeing zero circular dependency between the test assertion and the simplex implementation.

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
