# Phase 3A: LP Standardization Layer — Architecture & Mathematical Specification

**Document Version:** 1.0.0
**Status:** IMPLEMENTED & AUDITED (Phase 3A Milestone)
**Authoritative Baseline Commit:** `b8e9234274bc3c5fdefa07a28a9adf028737346d`
**Repository:** `SIH26119-Indigenous-GPU-Accelerated-Optimization-Solver`

---

> [!IMPORTANT]
> **Phase 3A performs mathematical standardization only. It does not solve LPs.**
>
> The output of this layer is an immutable standard equality representation $\min \bar{c}^T \bar{x} + \bar{c}_0$ subject to $\bar{A} \bar{x} = \bar{b}$, $\bar{x} \ge 0$, $\bar{b} \ge 0$, together with bidirectional affine mapping metadata. Optimization algorithms (primal simplex, dual simplex, Phase I / Phase II basis sequences) are strictly excluded from Phase 3A.

---

## 1. Mathematical Standard Form Representation

Phase 3A maps an arbitrary canonical Linear Program defined in Phase 1 (`model::Model`) into the canonical standard equality form:

$$\begin{aligned}
\min_{\bar{x} \in \mathbb{R}^{\bar{n}}} \quad & \bar{z}(\bar{x}) = \bar{c}^T \bar{x} + \bar{c}_0 \\
\text{subject to} \quad & \bar{A} \bar{x} = \bar{b}, \\
& \bar{x} \ge 0, \\
& \bar{b} \ge 0
\end{aligned}$$

Where:
- $\bar{m} \in \mathbb{N}_0$ is the number of standard equality constraints (`Dimension`).
- $\bar{n} \in \mathbb{N}_0$ is the number of non-negative standard variables (`Dimension`).
- $\bar{A} \in \mathbb{R}^{\bar{m} \times \bar{n}}$ is stored in immutable Compressed Sparse Row (CSR) format (`numerics::SparseMatrix`).
- $\bar{b} \in \mathbb{R}^{\bar{m}}$ is the non-negative right-hand side vector (`numerics::DenseVector`), satisfying $\bar{b}_i \ge 0$ for all $i \in \{0, \dots, \bar{m}-1\}$.
- $\bar{c} \in \mathbb{R}^{\bar{n}}$ is the standard cost vector (`numerics::DenseVector`).
- $\bar{c}_0 \in \mathbb{R}$ is the invariant scalar objective offset (`Scalar`).

---

## 2. Variable Transformation Catalog

Every original variable $x_j$ ($j \in \{0, \dots, n-1\}$) with bounds $[l_j, u_j]$ is classified into one of six mutually exclusive, mathematically exhaustive categories.

| Class | Original Bounds | Transformation $\bar{x} \leftrightarrow x_j$ | Standard Vars | Added Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **Identity** | $l_j = 0, u_j = +\infty$ | $x_j = \bar{x}_k$ | 1 ($\bar{x}_k$) | 0 |
| **LowerShift** | $l_j > -\infty, u_j = +\infty, l_j \ne 0$ | $x_j = \bar{x}_k + l_j$ | 1 ($\bar{x}_k$) | 0 |
| **UpperReflect** | $l_j = -\infty, u_j < +\infty$ | $x_j = u_j - \bar{x}_k$ | 1 ($\bar{x}_k$) | 0 |
| **BoxBound** | $-\infty < l_j < u_j < +\infty$ | $x_j = \bar{x}_k + l_j$ | 2 ($\bar{x}_k, s_j$) | 1 ($\bar{x}_k + s_j = u_j - l_j$) |
| **FreeSplit** | $l_j = -\infty, u_j = +\infty$ | $x_j = \bar{x}_{k^+} - \bar{x}_{k^-}$ | 2 ($\bar{x}_{k^+}, \bar{x}_{k^-}$) | 0 |
| **FixedEliminated** | $l_j = u_j = v_j$ | $x_j = v_j$ | 0 | 0 (substituted out) |

### 2.1 Transformation Details and Cost Adjustments

Let $c_j$ denote the original objective coefficient for variable $x_j$ (after applying objective sense normalization for minimization):

1. **Identity (`VariableTransformType::Identity`)**:
   - $\bar{x}_k = x_j \ge 0$.
   - $\bar{c}_k = c_j$.
   - No offset adjustment.

2. **LowerShift (`VariableTransformType::LowerShift`)**:
   - $x_j = \bar{x}_k + l_j$, where $\bar{x}_k \ge 0$.
   - Contribution to objective: $c_j x_j = c_j (\bar{x}_k + l_j) = c_j \bar{x}_k + c_j l_j$.
   - $\bar{c}_k = c_j$.
   - $\bar{c}_0 \leftarrow \bar{c}_0 + c_j l_j$.
   - Constraint column $a_{\cdot j} x_j = a_{\cdot j} \bar{x}_k + a_{\cdot j} l_j \implies$ right-hand side shift $b_i \leftarrow b_i - a_{ij} l_j$.

3. **UpperReflect (`VariableTransformType::UpperReflect`)**:
   - $x_j = u_j - \bar{x}_k$, where $\bar{x}_k \ge 0$.
   - Contribution to objective: $c_j x_j = c_j (u_j - \bar{x}_k) = -c_j \bar{x}_k + c_j u_j$.
   - $\bar{c}_k = -c_j$.
   - $\bar{c}_0 \leftarrow \bar{c}_0 + c_j u_j$.
   - Constraint column $a_{\cdot j} x_j = -a_{\cdot j} \bar{x}_k + a_{\cdot j} u_j \implies$ column negated, right-hand side shift $b_i \leftarrow b_i - a_{ij} u_j$.

4. **BoxBound (`VariableTransformType::BoxBound`)**:
   - $x_j = \bar{x}_k + l_j$, with bound constraint $\bar{x}_k \le u_j - l_j$.
   - Slack variable $s_j \ge 0$ added via auxiliary equation: $\bar{x}_k + s_j = u_j - l_j$.
   - $\bar{c}_k = c_j$, $\bar{c}_{s_j} = 0$.
   - $\bar{c}_0 \leftarrow \bar{c}_0 + c_j l_j$.
   - Original column coefficients unchanged for $\bar{x}_k$; right-hand side shifted by $-a_{ij} l_j$.

5. **FreeSplit (`VariableTransformType::FreeSplit`)**:
   - $x_j = \bar{x}_{k^+} - \bar{x}_{k^-}$, with $\bar{x}_{k^+} \ge 0, \bar{x}_{k^-} \ge 0$.
   - $\bar{c}_{k^+} = c_j$, $\bar{c}_{k^-} = -c_j$.
   - In constraint matrix: column for $\bar{x}_{k^+}$ is $a_{\cdot j}$; column for $\bar{x}_{k^-}$ is $-a_{\cdot j}$.

6. **FixedEliminated (`VariableTransformType::FixedEliminated`)**:
   - $x_j = v_j = l_j = u_j$.
   - Variable is completely removed from the standard column set.
   - $\bar{c}_0 \leftarrow \bar{c}_0 + c_j v_j$.
   - For all constraints $i$, RHS shifted: $b_i \leftarrow b_i - a_{ij} v_j$.

---

## 3. Constraint Transformation Catalog

Each original constraint $i$ with bounds $[L_i, U_i]$ is transformed into one or two standard equality rows.

| Original Constraint Form | Condition | Added Auxiliaries | Standard Equality System |
| :--- | :--- | :--- | :--- |
| **Equality** | $L_i = U_i = B_i$ | None | $\sum_j \tilde{a}_{ij} \bar{x}_j = B_i'$ |
| **Less-Than-or-Equal** | $L_i = -\infty, U_i < +\infty$ | Slack $s_i \ge 0$ | $\sum_j \tilde{a}_{ij} \bar{x}_j + s_i = U_i'$ |
| **Greater-Than-or-Equal** | $L_i > -\infty, U_i = +\infty$ | Surplus $e_i \ge 0$ | $\sum_j \tilde{a}_{ij} \bar{x}_j - e_i = L_i'$ |
| **Range Constraint** | $-\infty < L_i < U_i < +\infty$ | Surplus $e_i \ge 0$, Slack $s_i \ge 0$ | Row 1: $\sum_j \tilde{a}_{ij} \bar{x}_j - e_i = L_i'$<br>Row 2: $e_i + s_i = U_i - L_i$ |
| **Free Constraint** | $L_i = -\infty, U_i = +\infty$ | None | Redundant (omitted from $\bar{A}$) |

Here, $B_i', L_i', U_i'$ represent the original constraint bounds shifted by substitutions of shifted, reflected, or fixed variables.

### 3.1 Exact Two-Row Formulation for Range Constraints
A two-sided constraint $L_i \le a_i^T x \le U_i$ is mapped without duplicating the dense or sparse structural coefficients $a_i^T$:
1. Row 1 enforces the lower bound: $\sum_j \tilde{a}_{ij} \bar{x}_j - e_i = L_i'$, where $e_i \ge 0$ is a surplus variable.
2. Row 2 enforces the range width on the surplus variable: $e_i + s_i = U_i - L_i$, where $s_i \ge 0$ is a slack variable.
3. This guarantees:
   $$e_i \ge 0 \implies \sum_j \tilde{a}_{ij} \bar{x}_j \ge L_i', \quad e_i \le U_i - L_i \implies \sum_j \tilde{a}_{ij} \bar{x}_j = L_i' + e_i \le U_i'$$
   The structural row is entered into $\bar{A}$ exactly once, preserving matrix sparsity.

---

## 4. Right-Hand Side (RHS) Sign Normalization

Simplex algorithms require standard RHS non-negativity: $\bar{b}_i \ge 0$ for all $i$.

### 4.1 Normalization Algorithm
For each standard row $i \in \{0, \dots, \bar{m}-1\}$:
1. Compute the shifted RHS value $b_i'$.
2. If $b_i' < 0$:
   - Set $\bar{b}_i = -b_i' > 0$.
   - For every nonzero coefficient in row $i$, multiply by $-1$: $\bar{a}_{ij} \leftarrow -\bar{a}_{ij}$.
   - Record in constraint metadata: `row_negated = true`.
3. If $b_i' \ge 0$:
   - Set $\bar{b}_i = b_i'$.
   - Nonzero coefficients in row $i$ remain unchanged.
   - Record in constraint metadata: `row_negated = false`.

### 4.2 Feasible Set Invariance Proof
Multiplying an equality constraint by a non-zero scalar $\alpha \ne 0$ yields:
$$\{x \in \mathbb{R}^n \mid a_i^T x = b_i\} \equiv \{x \in \mathbb{R}^n \mid \alpha a_i^T x = \alpha b_i\}$$
For $\alpha = -1$, the solution set is strictly identical. No information is lost, and the affine manifold defined by the linear equations is invariant.

### 4.3 Dual Multiplier Transformation
When a dual solution $y^{\text{std}} \in \mathbb{R}^{\bar{m}}$ is produced by a subsequent simplex solver, the original Lagrange multiplier $y_i^{\text{orig}}$ is recovered by:
$$y_i^{\text{orig}} = \begin{cases} -y_i^{\text{std}} & \text{if row } i \text{ was negated} \\ +y_i^{\text{std}} & \text{otherwise} \end{cases}$$
This ensures exact KKT and duality recovery.

---

## 5. Objective Sense Preservation Proof

### 5.1 Minimization
For original problem $\min c^T x + c_0$:
- No negation is applied to the costs: $\tilde{c} = c$.
- Variable substitutions yield:
  $$\sum_j c_j x_j + c_0 = \sum_{k} \bar{c}_k \bar{x}_k + \bar{c}_0$$
- Duality and value identity: $\bar{z}(\bar{x}) = c^T x + c_0$.

### 5.2 Maximization
For original problem $\max c^T x + c_0 \iff \min -(c^T x + c_0)$:
- The solver internalizes $\min (-c)^T x - c_0$.
- Hence, all costs and initial constant offset are negated:
  $$\tilde{c}_j = -c_j, \quad \tilde{c}_0 = -c_0$$
- Variable substitutions are applied to $\tilde{c}$, yielding $\bar{c}$ and $\bar{c}_0$.
- Let $\bar{x}^*$ be the standard solution and $x^*$ the reconstructed original solution:
  $$\bar{z}(\bar{x}^*) = \bar{c}^T \bar{x}^* + \bar{c}_0 = - (c^T x^* + c_0)$$
- Therefore:
  $$f_{\text{orig}}(x^*) = - \bar{z}(\bar{x}^*)$$
- The original objective value is preserved with exact sign inversion.

---

## 6. Reconstruction and Projection Algorithms

### 6.1 Reconstruction: $\bar{x} \to x$
Given a point $\bar{x} \in \mathbb{R}^{\bar{n}}$, reconstruct the original decision vector $x \in \mathbb{R}^n$:

```text
Algorithm ReconstructPrimal(x_bar):
    Initialize x in R^n with zeros
    For each original variable j from 0 to n - 1:
        mapping = variable_mappings[j]
        Switch mapping.type:
            Case Identity:
                x[j] = x_bar[mapping.standard_index_1]
            Case LowerShift:
                x[j] = x_bar[mapping.standard_index_1] + mapping.shift_offset
            Case UpperReflect:
                x[j] = mapping.shift_offset - x_bar[mapping.standard_index_1]
            Case BoxBound:
                x[j] = x_bar[mapping.standard_index_1] + mapping.shift_offset
            Case FreeSplit:
                x[j] = x_bar[mapping.standard_index_1] - x_bar[mapping.standard_index_2]
            Case FixedEliminated:
                x[j] = mapping.shift_offset
    Return x
```

### 6.2 Projection: $x \to \bar{x}$
Given a feasible point $x \in \mathbb{R}^n$, project it onto the standard space $\bar{x} \in \mathbb{R}^{\bar{n}}$, synthesizing all structural, bound slack, and constraint slack/surplus variables:

```text
Algorithm ProjectPrimal(x):
    Initialize x_bar in R^n_bar with zeros

    // 1. Project Decision Variables
    For each original variable j from 0 to n - 1:
        mapping = variable_mappings[j]
        Switch mapping.type:
            Case Identity:
                x_bar[mapping.standard_index_1] = x[j]
            Case LowerShift:
                x_bar[mapping.standard_index_1] = x[j] - mapping.shift_offset
            Case UpperReflect:
                x_bar[mapping.standard_index_1] = mapping.shift_offset - x[j]
            Case BoxBound:
                x_bar[mapping.standard_index_1] = x[j] - mapping.shift_offset
                x_bar[mapping.auxiliary_index]   = (u_j - l_j) - x_bar[mapping.standard_index_1]
            Case FreeSplit:
                If x[j] >= 0:
                    x_bar[mapping.standard_index_1] = x[j]
                    x_bar[mapping.standard_index_2] = 0
                Else:
                    x_bar[mapping.standard_index_1] = 0
                    x_bar[mapping.standard_index_2] = -x[j]
            Case FixedEliminated:
                // No standard variable to populate

    // 2. Project Constraint Auxiliaries
    For each original constraint i from 0 to m - 1:
        mapping = constraint_mappings[i]
        val = sum_j a_ij * x[j]
        Switch mapping.type:
            Case Equality:
                // No auxiliary variable
            Case LessEqual:
                // a_i^T x + s_i = U_i  =>  s_i = U_i - a_i^T x
                x_bar[mapping.slack_index] = mapping.upper_bound - val
            Case GreaterEqual:
                // a_i^T x - e_i = L_i  =>  e_i = a_i^T x - L_i
                x_bar[mapping.surplus_index] = val - mapping.lower_bound
            Case Range:
                // Row 1: e_i = a_i^T x - L_i
                // Row 2: s_i = (U_i - L_i) - e_i = U_i - a_i^T x
                x_bar[mapping.surplus_index] = val - mapping.lower_bound
                x_bar[mapping.slack_index]   = mapping.upper_bound - val
            Case Free:
                // Omitted

    Return x_bar
```

---

## 7. Numerical Decision Rules & Robustness

1. **Zero Coefficient Threshold (`NUMERICAL_EPSILON = 1e-14`)**:
   - Objective terms and matrix entries satisfying $|a_{ij}| \le 10^{-14}$ are filtered to avoid catastrophic numerical cancellation and fill-in of structural zeros.
2. **Infinite Bound Detection (`INFINITY_THRESHOLD = 1e20`)**:
   - Variables or constraints with bound values $|B| \ge 10^{20}$ are recognized as unbounded ($+\infty$ or $-\infty$).
3. **Empty Rows**:
   - An empty constraint row $\sum_j 0 \cdot x_j$ with shifted RHS satisfying $|b_i'| \le 10^{-14}$ is structurally redundant and safely satisfied; if $|b_i'| > 10^{-14}$, an error `LpSolverStatus::ErrorInfeasible` is returned immediately.
4. **Empty Columns**:
   - A variable with no constraint appearances contributes solely to the objective. If its bound permits unbounded descent, the solver records it; otherwise, standard non-negative mapping is applied.

---

## 8. Ownership, Immutability, and Lifetime Contracts

- **Immutability**: `StandardizedLp` exposes strictly `const` accessors:
  - `const SparseMatrix& A_bar() const noexcept;`
  - `const DenseVector& b_bar() const noexcept;`
  - `const DenseVector& c_bar() const noexcept;`
  - `Scalar c0_bar() const noexcept;`
- **Zero In-Place Side Effects**: Standardization accepts `const model::Model&`. The input model is never mutated.
- **CSR Matrix Assembly**: Matrix entries are accumulated into temporary triplets and converted via `SparseMatrix::from_triplets`, ensuring ordered column indices and valid row pointers in single-pass CSR format.
- **Zero Hot-Path Allocations for Simplex**: The produced `StandardizedLp` is immutable and can be used with preallocated caller-owned workspaces for matrix-vector products and simplex pricing.

---

## 9. Verification & Test Matrix

The Phase 3A test suite is implemented in [`tests/unit/test_lp_standardization.cpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/tests/unit/test_lp_standardization.cpp).

| Test ID | Test Target | Description & Acceptance Criteria | Status |
| :--- | :--- | :--- | :--- |
| `TEST-STD-01` | Already-Standard Form | Canonical non-negative LP with $\le$ transformed cleanly; verifies $\bar{A}, \bar{b}, \bar{c}, \bar{c}_0$. | **PASSED** |
| `TEST-STD-02` | Lower-Bound Shift | Variable $x_j \ge L_j$ shifted; verifies offset $\bar{c}_0 += c_j L_j$, RHS shift, and reconstruction. | **PASSED** |
| `TEST-STD-03` | Upper-Bound Reflection | Variable $x_j \le U_j$ reflected; verifies negated cost, offset $\bar{c}_0 += c_j U_j$, and reconstruction. | **PASSED** |
| `TEST-STD-04` | Box Bounds | Variable $L_j \le x_j \le U_j$; verifies primary shifted variable + bound slack row $\bar{x} + s = U - L$. | **PASSED** |
| `TEST-STD-05` | Free Variable | Unbounded $x_j \in \mathbb{R}$; verifies split into $x_j^+ - x_j^-$ with opposing costs and reconstruction. | **PASSED** |
| `TEST-STD-06` | Fixed Variable | $x_j = v_j$; verifies variable elimination from column set, RHS substitution, and offset update. | **PASSED** |
| `TEST-STD-07` | Less-Equal Row | $a_i^T x \le U_i$; verifies addition of positive slack variable $+s_i$. | **PASSED** |
| `TEST-STD-08` | Greater-Equal Row | $a_i^T x \ge L_i$; verifies addition of surplus variable $-e_i$. | **PASSED** |
| `TEST-STD-09` | Equality Row | $a_i^T x = B_i$; verifies no auxiliary variables added, exact equality retained. | **PASSED** |
| `TEST-STD-10` | Range Constraint | $L_i \le a_i^T x \le U_i$; verifies two-row formulation with surplus and slack variables. | **PASSED** |
| `TEST-STD-11` | Range with $L_i = -\infty$ | One-sided range upper bound treated as LessEqual constraint. | **PASSED** |
| `TEST-STD-12` | Range with $U_i = +\infty$ | One-sided range lower bound treated as GreaterEqual constraint. | **PASSED** |
| `TEST-STD-13` | Negative RHS Normalization | Constraint with $b_i < 0$; verifies row scaling by $-1$, positive RHS, and metadata recording. | **PASSED** |
| `TEST-STD-14` | Mixed Complex Model | Model combining all 6 variable classes and all 4 constraint types simultaneously. | **PASSED** |
| `TEST-STD-15` | Maximization Conversion | Model with $\max f(x)$; verifies cost negation, offset negation, and objective value preservation. | **PASSED** |
| `TEST-STD-16` | Bidirectional Round-Trip | Verifies $x \to \bar{x} \to x \equiv x$, and $\bar{A} \bar{x} = \bar{b}$ feasibility of projected point. | **PASSED** |
| `TEST-STD-17` | Large Scale Staircase | 100-variable, 50-constraint band matrix; verifies sparsity pattern and CSR indexing integrity. | **PASSED** |
| `TEST-STD-18` | Error Handling | Rejection of models with $lb > ub$ or integer/binary variables with descriptive error codes. | **PASSED** |
| `TEST-PROP-01`| Deterministic Property Suite | 30 randomly generated LPs; verifies non-negative RHS, non-empty CSR, and exact round-trip. | **PASSED** |

---

## 10. Conclusion

Phase 3A provides the mathematically exact, verified, zero-regression foundation required for future Phase 3 milestones. All standard form algebraic invariants, bidirectional reconstruction mappings, and sign-normalization proofs are fully validated under strict `-Wall -Wextra -Wpedantic -Wconversion` compilation settings and automated unit tests.
