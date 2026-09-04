# Phase 3C: Mathematical Audit — Basis Factorization & Solve Duality

> **Independent Mathematical Audit & Algebraic Proofs**  
> **Repository:** SIH26119 — Indigenous GPU-Accelerated Optimization Solver  
> **Scope:** Verification of $P B Q = L U$ Algebra, Transpose Permutations, and Error Bounds

---

## 1. Algebraic Soundness of Permutation Transpose Duality

### 1.1 Theorem (Transpose Solve Permutation Invariance)
Let $B \in \mathbb{R}^{m \times m}$ be nonsingular. Let $P, Q \in \mathbb{R}^{m \times m}$ be permutation matrices, $L \in \mathbb{R}^{m \times m}$ be unit lower triangular, and $U \in \mathbb{R}^{m \times m}$ be upper triangular with non-zero diagonal entries, such that:

$$P B Q = L U$$

Then:
1. The unique solution to $B x = r$ is given by:
   $$x = Q U^{-1} L^{-1} P r$$
2. The unique solution to $B^T y = r$ is given by:
   $$y = P^T L^{-T} U^{-T} Q^T r$$

### 1.2 Proof of (1) — Forward Solve
Since $P$ and $Q$ are permutation matrices, $P^T P = P P^T = I$ and $Q^T Q = Q Q^T = I$.
$$P B Q = L U \iff B = P^T L U Q^T$$
Substitute $B$ into $B x = r$:
$$P^T L U Q^T x = r$$
Multiply on the left by $P$:
$$L U (Q^T x) = P r$$
Since $L$ is triangular with diagonal entries $1$, it is nonsingular and $L^{-1}$ exists. Multiply by $L^{-1}$:
$$U (Q^T x) = L^{-1} P r$$
Since $U$ has non-zero diagonal entries, it is nonsingular and $U^{-1}$ exists. Multiply by $U^{-1}$:
$$Q^T x = U^{-1} L^{-1} P r$$
Multiply on the left by $Q$:
$$x = Q U^{-1} L^{-1} P r$$
This matches the forward solve sequence:
$$r \xrightarrow{P} z \xrightarrow{L^{-1}} w \xrightarrow{U^{-1}} v \xrightarrow{Q} x \quad \blacksquare$$

### 1.3 Proof of (2) — Dual / Transpose Solve
From $B = P^T L U Q^T$, compute the matrix transpose:
$$B^T = (P^T L U Q^T)^T = (Q^T)^T U^T L^T (P^T)^T = Q U^T L^T P$$
Substitute $B^T$ into $B^T y = r$:
$$Q U^T L^T P y = r$$
Multiply on the left by $Q^T$ (since $Q^T Q = I$):
$$U^T L^T (P y) = Q^T r$$
Since $U^T$ is lower triangular with non-zero diagonal entries, multiply on the left by $U^{-T} = (U^T)^{-1}$:
$$L^T (P y) = U^{-T} Q^T r$$
Since $L^T$ is unit upper triangular, multiply on the left by $L^{-T} = (L^T)^{-1}$:
$$P y = L^{-T} U^{-T} Q^T r$$
Multiply on the left by $P^T$ (since $P^T P = I$):
$$y = P^T L^{-T} U^{-T} Q^T r$$
This matches the transpose solve sequence:
$$r \xrightarrow{Q^T} z \xrightarrow{U^{-T}} w \xrightarrow{L^{-T}} v \xrightarrow{P^T} y \quad \blacksquare$$

---

## 2. Condition Number Estimation via Hager-Higham Algorithm

### 2.1 Theoretical Foundation
Inverting $B$ to compute $\|B^{-1}\|_1$ requires $O(m^3)$ work, which is unacceptable for hot-path simplex. Instead, Phase 3C specifies the **Hager-Higham 1-norm estimator**, which estimates:

$$\|B^{-1}\|_1 = \max_{\|x\|_1 = 1} \|B^{-1} x\|_1$$

using only $4$ to $5$ solves of $B x = v$ and $B^T y = w$ ($O(m^2)$ total operations).

### 2.2 Algorithmic Steps
1. Initialize $x = (\frac{1}{m}, \frac{1}{m}, \dots, \frac{1}{m})^T \in \mathbb{R}^m$ so that $\|x\|_1 = 1$.
2. Solve $B w = x$.
3. Compute $\xi = \operatorname{sign}(w)$ where $\operatorname{sign}(0) = 1$.
4. Solve $B^T z = \xi$.
5. If $\|z\|_\infty \le z^T x$, stop; estimate is $\|w\|_1$.
6. Otherwise, set $j = \operatorname{argmax}_i |z_i|$, set $x = e_j$, and repeat step 2.
7. Return estimate:
   $$\kappa^*(B) = \|B\|_1 \cdot \|w\|_1$$

### 2.3 Mathematical Proof of Independence from Pivot Threshold
- **Theorem:** There exists a family of matrices $T_m \in \mathbb{R}^{m \times m}$ with unit diagonal pivots $|U_{ii}| = 1$ whose condition number grows exponentially: $\kappa(T_m) = O(2^m)$.
- **Proof:** Consider the Ostrowski matrix:
  $$T_m = \begin{bmatrix}
  1 & 0 & 0 & \cdots & 0 \\
  -1 & 1 & 0 & \cdots & 0 \\
  -1 & -1 & 1 & \cdots & 0 \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  -1 & -1 & -1 & \cdots & 1
  \end{bmatrix}$$
  All pivots during elimination are $U_{ii} = 1 > \varepsilon_{\text{sing}}$. However, $(T_m^{-1})_{m, 1} = 2^{m-2}$, so $\|T_m^{-1}\|_\infty = 2^{m-1}$.
  For $m = 60$, $\kappa(T_m) > 10^{17}$, rendering the matrix numerically singular despite every pivot being exactly $1.0$.
- **Conclusion:** A local pivot acceptance threshold $\varepsilon_{\text{sing}}$ **does not and cannot bound the condition number**. The global condition number estimate $\kappa^*(B)$ is mathematically necessary as a distinct diagnostic ceiling.

---

## 3. Backward Error & Componentwise Residual Analysis

### 3.1 Theorem (Oettli-Prager Backward Error)
For any computed approximate solution $\hat{x}$ to $B x = r$, the smallest perturbations $\Delta B, \Delta r$ satisfying $(B + \Delta B) \hat{x} = r + \Delta r$ with $|\Delta B| \le \omega |B|$ and $|\Delta r| \le \omega |r|$ have componentwise magnitude:

$$\omega = \max_{i \in \{0, \dots, m-1\}} \frac{|r_i - (B \hat{x})_i|}{(|B| |\hat{x}| + |r|)_i}$$

### 3.2 Audit Verification Condition
Phase 3C accepts a solve if:

$$\|B \hat{x} - r\|_\infty \le \tau_{\text{resid}} \left( \|B\|_\infty \|\hat{x}\|_\infty + \|r\|_\infty \right)$$

where $\tau_{\text{resid}} = 10^{-8}$. This guarantees that the computed solution corresponds to an exact solution of a problem perturbed by at most $\tau_{\text{resid}}$ in backward error, fully satisfying IEEE 754 numerical stability standards.
