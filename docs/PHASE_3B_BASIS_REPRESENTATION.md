# Phase 3B: Basis Representation and State Layer

> **Authoritative Specification & Architecture Documentation**  
> **Repository:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver  
> **Status:** Phase 3B Complete  
> **Mandatory Scope Declaration:** *Phase 3B represents simplex basis state. It does not perform simplex optimization.*

---

## 1. Executive Summary & Mathematical Foundation

In the standard form linear program:

$$\begin{aligned}
\min \quad & c^T x \\
\text{s.t.} \quad & A x = b \\
& x \ge 0
\end{aligned}$$

where $A \in \mathbb{R}^{m \times n}$ has rank $m \le n$, a **simplex basis** corresponds to a choice of $m$ linearly independent columns of $A$. These columns form the basis matrix:

$$B = \begin{bmatrix} A_{:, B(0)} & A_{:, B(1)} & \cdots & A_{:, B(m-1)} \end{bmatrix} \in \mathbb{R}^{m \times m}$$

The nonbasic columns form $N \in \mathbb{R}^{m \times (n - m)}$. Correspondingly, the decision variable vector $x$ partitions into basic variables $x_B \in \mathbb{R}^m$ and nonbasic variables $x_N \in \mathbb{R}^{n - m}$:

$$A x = B x_B + N x_N = b$$

For any basic solution, nonbasic variables are held at zero ($x_N \equiv 0$).

---

## 2. Invariant Guarantees: Structural Validity vs. Numerical Nonsingularity

### Crucial Architectural Separation
Phase 3B enforces a strict separation between **structural basis validity** and **numerical nonsingularity**:

> [!IMPORTANT]
> **Phase 3B Basis validates ONLY structural invariants:**
> 1. Dimension compatibility ($0 \le m \le n \le \text{IndexMax}$).
> 2. Exactly $m$ basic variables: $|B| = m$.
> 3. Strict variable bounds: for all $i \in \{0, \dots, m - 1\}$, $0 \le B(i) < n$.
> 4. Uniqueness: all $m$ basic column indices are strictly distinct ($B(i) \ne B(k)$ for $i \ne k$).
> 5. Exact partition: every column $j \in \{0, \dots, n - 1\}$ is either basic or nonbasic ($B \cup N = \{0, \dots, n - 1\}$, $B \cap N = \emptyset$).
> 6. Bijection: the row-to-variable map $B: \{0, \dots, m - 1\} \to B$ and variable-to-row map $B^{-1}: B \to \{0, \dots, m - 1\}$ are mutual inverses.
>
> **Phase 3B MUST NOT and DOES NOT claim to prove that $B$ is numerically nonsingular.**  
> Numerical rank determination, singularity checking, and condition estimation are strictly the responsibility of **Phase 3C (Basis Factorization Layer)**.

---

## 3. Basis State Storage & $O(1)$ Hot-Path Contracts

The [`Basis`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis.hpp) class manages both forward and reverse representations to guarantee constant-time queries without any dynamic memory allocations:

```cpp
class Basis {
    Dimension num_rows_{0};                      // m
    Dimension num_cols_{0};                      // n
    std::vector<Index> basic_vars_;             // size m: row -> basic column
    std::vector<Index> var_to_row_;             // size n: column -> row (or kInvalidIndex)
    std::vector<Index> nonbasic_vars_;          // size n - m: list of nonbasic columns
    std::vector<Index> var_to_nonbasic_pos_;    // size n: column -> position in nonbasic_vars_
    uint64_t version_{0};                       // incremented upon each pivot
    bool is_valid_{false};
};
```

### Hot-Path Query Performance Contracts
1. **`is_basic(Index j) noexcept`**: $O(1)$ table lookup in `var_to_row_`. Zero allocations.
2. **`is_nonbasic(Index j) noexcept`**: $O(1)$ table lookup in `var_to_row_`. Zero allocations.
3. **`row_of_basic(Index j) noexcept`**: $O(1)$ lookup returning `Result<Index>`. Zero allocations.
4. **`basic_variable(Index row) noexcept`**: $O(1)$ array index in `basic_vars_`. Zero allocations.
5. **`basic_variables()` & `nonbasic_variables()`**: $O(1)$ const reference return. Zero allocations.

### Nonbasic Column Ordering Contract
The list returned by `nonbasic_variables()` adheres to a strict deterministic ordering contract:
1. **Initial Construction (`Basis::create`):** Nonbasic columns are stored in strictly ascending numerical order:
   $$\{j \in [0, n) \mid j \text{ is nonbasic}\}$$
2. **Transactional Replacement (`replace_basic_variable`):** When entering column $j_{\text{enter}}$ enters the basis, it vacates slot $p = \text{var\_to\_nonbasic\_pos\_}[j_{\text{enter}}]$ in `nonbasic_vars_`. The leaving column $j_{\text{leave}}$ is placed directly into slot $p$. This guarantees $O(1)$ constant-time update without dynamic allocations or vector re-sorting, ensuring 100% deterministic ordering across all compilers and execution runs.

---

## 4. Transactional Basis Replacement & Mutation Semantics

Simplex pivot steps exchange a single basic variable for an entering nonbasic variable:

$$B(r) \leftarrow j_{\text{enter}}$$

The method `replace_basic_variable(Index entering_col, Index leaving_row)` adheres to **strict transactional semantics**:

1. **Pre-mutation Invariant Verification:**
   - Basis is structurally valid (`is_valid_ == true`).
   - `leaving_row < num_rows_`.
   - `entering_col < num_cols_`.
   - `entering_col` is currently nonbasic (`var_to_row_[entering_col] == kInvalidIndex`).
   - `leaving_col = basic_vars_[leaving_row]` is verified to map back to `leaving_row`.
   - Position of `entering_col` in `nonbasic_vars_` is verified.
2. **Strong Exception Safety:**
   - If any validation check fails, the method returns an error `Status` immediately.
   - The basis state, mappings, vectors, and `version_` counter remain **completely unaltered**.
3. **Atomic State Commit:**
   - `basic_vars_[leaving_row] = entering_col`
   - `var_to_row_[entering_col] = leaving_row`
   - `var_to_row_[leaving_col] = kInvalidIndex`
   - Nonbasic list updated in place: `nonbasic_vars_[pos] = leaving_col`
   - `var_to_nonbasic_pos_[leaving_col] = pos`
   - `var_to_nonbasic_pos_[entering_col] = kInvalidIndex`
   - `version_` incremented by 1.

---

## 5. BasicSolution Architecture & Allocation Behavior

The [`BasicSolution`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basic_solution.hpp) class stores coordinate values corresponding to a basis:

### Primary Representation
- **$x_B \in \mathbb{R}^m$**: The primary representation stores **only** the basic coordinates $x_B$ as a [`DenseVector`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/numerics/dense_vector.hpp) of length $m$.
- Nonbasic variables are implicitly $x_N \equiv 0$.
- **Allocation Rule:** Ordinary basis-state operations, evaluations, and checks do **not** require full-vector ($n$-dimensional) materialization.

### Optional Convenience Representation ($x \in \mathbb{R}^n$)
- `expand_full_primal()`: Materializes the $n$-vector $x$ such that:
  $$x_j = \begin{cases} x_{B(i)} & \text{if } j = B(i) \\ 0 & \text{if } j \in N \end{cases}$$
- **Allocation Behavior:**
  - `Result<DenseVector> expand_full_primal() const`: Allocates a new `DenseVector` of dimension $n$. Suitable for output, file serialization, and high-level verification.
  - `Status expand_full_primal(DenseVector& x_full) const noexcept`: In-place overload writing into a caller-supplied preallocated workspace of size $n$. Performs **zero** dynamic heap allocations.

### Independent Primal Feasibility Verification
`BasicSolution::check_primal_feasibility` verifies:
1. Dimension compatibility between $A$ ($m \times n$), $b$ ($m$), and solution ($m$ basic, $n$ total).
2. Data validity: Rejection of NaN / non-finite values in $b$ and $x_B$.
3. Primal non-negativity:
   $$x_{B(i)} \ge -\tau_{\text{feas}}, \quad \forall i \in \{0, \dots, m - 1\}$$
4. Constraint residual:
   $$\|A x - b\|_\infty \le \tau_{\text{feas}}$$
   computed using Phase 2's authoritative `SparseMatrix::residual` without materializing dense matrices or invoking an optimization solver.
5. **Zero-Allocation Hot-Path Contract:**
   The 7-argument overload:
   ```cpp
   Status check_primal_feasibility(
       const SparseMatrix& A,
       const DenseVector& b,
       const BasicSolution& sol,
       DenseVector& x_workspace,
       DenseVector& residual_scratch,
       DenseVector& residual_out,
       Scalar feas_tol = 1e-7);
   ```
   operates with **zero heap allocations** and strictly enforces Phase 2 pairwise distinctness/aliasing across `b`, `x_workspace`, `residual_scratch`, and `residual_out`.

---

## 6. BasisMatrixView: Non-Owning Logical View

The [`BasisMatrixView`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/src/solver/lp/basis_matrix_view.hpp) provides a square $m \times m$ logical view over the sparse constraint matrix $A$:

$$B[:, k] = A[:, B(k)], \quad k \in \{0, \dots, m - 1\}$$

### Non-Owning Contract & Lifetime Invariants
1. **Zero Data Duplication:**
   `BasisMatrixView` contains only `const SparseMatrix& A_` and `const Basis& basis_`. It never copies, allocates, or duplicates the matrix coefficients of $B$.
2. **Explicit Lifetime Invariant:**
   The caller must guarantee that both the referenced `SparseMatrix` $A$ and `Basis` object outlive the `BasisMatrixView` instance.
3. **Dynamic View Synchronization:**
   Because `BasisMatrixView` references `Basis`, any successful transactional pivot on `Basis` is immediately and automatically reflected in subsequent queries on the existing view without requiring view re-instantiation.
4. **Phase 3C Integration Boundary:**
   Column retrieval (`original_column_index(k)`) and entry inspection (`get(row, k)`) map directly to $A$'s CSR structure, forming the foundation for Phase 3C LU factorization without premature solver coupling.

---

## 7. Verification & Test Suite Audit

The Phase 3B implementation is validated by a rigorous test suite in [`tests/unit/test_basis.cpp`](file:///c:/Users/mohit/OneDrive/Desktop/SIH26119%20%E2%80%94%20Indigenous%20GPU-Accelerated%20Optimization%20Solver/tests/unit/test_basis.cpp):

| Test ID | Test Category | Validation Contract | Status |
| :--- | :--- | :--- | :--- |
| `TEST-BASIS-01` | Construction | Valid basis construction ($m=3, n=6$), structural validity, version = 1 | **PASSED** |
| `TEST-BASIS-02` | Invariant | Rejects duplicate basic column index (`InvalidArgument`) | **PASSED** |
| `TEST-BASIS-03` | Invariant | Rejects out-of-range column index $j \ge n$ (`InvalidArgument`) | **PASSED** |
| `TEST-BASIS-04` | Dimension | Rejects $m > n$ and basic variable list size $\ne m$ | **PASSED** |
| `TEST-BASIS-05` | Partition | Verifies strict basic/nonbasic partition and complement set | **PASSED** |
| `TEST-BASIS-06` | Bijective Map | Verifies row-to-variable and variable-to-row bijectivity | **PASSED** |
| `TEST-BASIS-07` | Replacement | Successful variable swap, nonbasic position swap, version increment | **PASSED** |
| `TEST-BASIS-08` | Replacement | Rejects entering variable that is already basic | **PASSED** |
| `TEST-BASIS-09` | Rollback | Rejects out-of-bounds leaving row / entering col; zero state mutation | **PASSED** |
| `TEST-BASIS-10` | Degenerate Edge | Valid $m=0$ empty basis construction, all $n$ columns nonbasic | **PASSED** |
| `TEST-BASIS-11` | Edge Dimension | Single-row constraint system ($m=1, n=3$), pivot and mapping verification | **PASSED** |
| `TEST-BASIS-12` | BasicSolution | Rejects dimension mismatch between $x_B$ and basis row count $m$ | **PASSED** |
| `TEST-BASIS-13` | BasicSolution | Primal feasibility check on feasible point; full primal expansion | **PASSED** |
| `TEST-BASIS-14` | Infeasibility | Detects violation of non-negativity bound ($x_B < -\tau$) | **PASSED** |
| `TEST-BASIS-15` | Infeasibility | Detects violation of equality constraint ($\|Ax - b\|_\infty > \tau$) | **PASSED** |
| `TEST-BASIS-16` | BasisMatrixView| View dimensions ($m \times m$), column mapping, entry extraction | **PASSED** |
| `TEST-BASIS-17` | Stress / Cycle | 5 sequential pivots on $3 \times 6$ system, version tracking, state consistency | **PASSED** |
| `TEST-BASIS-18` | Scalability | Large basis ($m=50, n=200$), full partition and bijectivity audit | **PASSED** |
| `TEST-BASIS-19` | Boundary Edge | Square basis $m = n$ (all columns basic, empty nonbasic set) | **PASSED** |
| `TEST-BASIS-20` | Boundary Edge | Degenerate $m = 0, n = 0$ basis system validation | **PASSED** |
| `TEST-BASIS-21` | Transactional | Comprehensive state immutability on failed replacement (every field verified) | **PASSED** |
| `TEST-BASIS-22` | Dynamic View | BasisMatrixView dynamic synchronization across Basis pivot without re-creation | **PASSED** |
| `TEST-BASIS-23` | Workspace/Alias| Zero-allocation feasibility overload, pairwise aliasing rejection, NaN rejection | **PASSED** |
| `TEST-PROP-01` | Property Test | 30 independent randomized trials with 5 random pivots each against independent `std::unordered_set` oracle | **PASSED** |

### Complete Test Suite Execution
- **Unit Test Execution:** 100% pass (0 failures across all 23 test fixtures and 30 property trials).
- **Full CTest Suite:** 11/11 test targets passed (Phase 0, Phase 1, Phase 2, Phase 3A, and Phase 3B).
- **Compiler Diagnostics:** 0 warnings under `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion`.
