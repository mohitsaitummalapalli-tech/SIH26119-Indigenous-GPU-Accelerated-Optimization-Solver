# Canonical Mathematical Formulation & Interchange Semantics

## 1. Canonical Optimization Problem Representation

The solver core represents continuous and mixed-integer linear and quadratic mathematical optimization problems in the canonical formulation:

$$\begin{aligned}
\text{minimize or maximize} \quad & f(x) = c^T x + \frac{1}{2} x^T Q x + c_0 \\
\text{subject to} \quad & l_i \le a_i^T x \le u_i, \quad i = 0, \dots, m-1 \\
& lb_j \le x_j \le ub_j, \quad j = 0, \dots, n-1 \\
& x_j \in \begin{cases} 
\mathbb{R} & \text{if } \text{VariableType::Continuous} \\
\mathbb{Z} & \text{if } \text{VariableType::Integer} \\
\{0, 1\} & \text{if } \text{VariableType::Binary}
\end{cases}
\end{aligned}$$

where:
- $Q \in \mathbb{R}^{n \times n}$ is a symmetric matrix ($Q_{ij} = Q_{ji}$).
- $c_0 \in \mathbb{R}$ is a scalar objective offset.
- $l_i, u_i \in [-\infty, +\infty]$ represent general ranged two-sided constraints.
- $lb_j, ub_j \in [-\infty, +\infty]$ represent variable bounds.

---

## 2. Quadratic Objective Convention

### 2.1. Canonical Storage Contract
The quadratic objective contribution is:
$$\frac{1}{2} x^T Q x = \frac{1}{2} \sum_{i=0}^{n-1} Q_{ii} x_i^2 + \sum_{0 \le i < j < n} Q_{ij} x_i x_j$$

- **Stored Canonical Entry**: Unordered variable pair $(i, j)$ with $i \le j$.
  The stored value represents $Q_{ij}$, the exact entry of the symmetric matrix $Q$.
- **Polynomial Mapping**:
  - Diagonal ($i = j$): Stored value $Q_{ii}$. Objective term: $\frac{1}{2} Q_{ii} x_i^2$.
  - Off-Diagonal ($i < j$): Stored value $Q_{ij} = Q_{ji}$. Objective term: $Q_{ij} x_i x_j$.

### 2.2. CPLEX LP Format Syntax: `[ P(x) ] / 2`
In LP format, the quadratic objective expression is strictly:
$$[ P(x) ] / 2$$
where $P(x) = x^T Q x = \sum_i Q_{ii} x_i^2 + 2 \sum_{i<j} Q_{ij} x_i x_j$.

Therefore:
- Diagonal term $+ a \cdot x_i^2$ inside $[ P(x) ] / 2 \implies Q_{ii} = a$.
- Off-diagonal term $+ b \cdot x_i * x_j$ inside $[ P(x) ] / 2 \implies \mathbf{Q_{ij} = \frac{b}{2}}$.
- Any bracketed quadratic expression lacking the `/ 2` divisor is strictly rejected with `StatusCode::ParseError`.

**Regression Verification**:
`[ 4 x * y ] / 2` produces stored canonical $Q_{xy} = 2$.
At $x = 3, y = 5$, the quadratic contribution is $2(3)(5) = 30$ (not $60$).

### 2.3. MPS `QUADOBJ` Format
In standard MPS `QUADOBJ`, records specify the matrix entries directly:
$$\text{VAR1} \quad \text{VAR2} \quad \text{VALUE} \implies Q_{\text{var1}, \text{var2}} = \text{VALUE}$$
- Diagonal: $Q_{ii} = \text{VALUE}$.
- Off-diagonal: $Q_{ij} = Q_{ji} = \text{VALUE}$.
- Conflicting symmetric definitions $(i, j, v_1)$ and $(j, i, v_2)$ with $v_1 \ne v_2$ are rejected.

---

## 3. MPS Format Semantics

### 3.1. Authoritative RANGES Arithmetic Contract
Let $b_i$ be the RHS value, $r_i$ be the RANGE value, and $q_i = |r_i|$:

| Row Type in `ROWS` | Range Condition | Canonical Interval $[l_i, u_i]$ | Formula |
|---|---|---|---|
| **`G`** ($\ge$) | $r_i \ne 0$ | $[b_i, b_i + q_i]$ | $l_i = b_i, \quad u_i = b_i + \|r_i\|$ |
| **`L`** ($\le$) | $r_i \ne 0$ | $[b_i - q_i, b_i]$ | $l_i = b_i - \|r_i\|, \quad u_i = b_i$ |
| **`E`** ($=$) | $r_i > 0$ | $[b_i, b_i + r_i]$ | $l_i = b_i, \quad u_i = b_i + r_i$ |
| **`E`** ($=$) | $r_i < 0$ | $[b_i + r_i, b_i]$ | $l_i = b_i + r_i, \quad u_i = b_i$ |
| **`E`** ($=$) | $r_i = 0$ | $[b_i, b_i]$ | $l_i = b_i, \quad u_i = b_i$ |
| **`N`** (free) | Any | $[-\infty, +\infty]$ | Free row, range ignored |

### 3.2. Multiple Vector Set Selection Rule
When multiple named vectors appear in `RHS`, `RANGES`, or `BOUNDS`, the parser selects the **first vector name** encountered in the respective section. Records matching subsequent vector names are deterministically skipped.

---

## 4. Unsupported Interchange Features in Phase 1

The following advanced or non-standard format sections are strictly identified and rejected in Phase 1:
- `QCMATRIX` (Quadratically Constrained Programming constraints) $\implies$ `StatusCode::UnsupportedFeature`
- `SOS` (Special Ordered Sets Type 1 and Type 2) $\implies$ `StatusCode::UnsupportedFeature`
- `INDICATORS` (Indicator constraints) $\implies$ `StatusCode::UnsupportedFeature`
- Quadratic expressions in LP constraints $\implies$ `StatusCode::UnsupportedFeature`
