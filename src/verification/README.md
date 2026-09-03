# Verification Layer (`src/verification/`)

## Purpose
The Verification module provides an independent mathematical validation and solution audit engine. It validates candidate solutions against original problem formulations without trusting intermediate solver states.

## Planned Responsibilities
- **Primal Feasibility Verification**:
  - Evaluation of constraint residuals: $\|Ax - b\|_{\infty}$.
  - Verification of variable bounds: $\max(0, l_j - x_j, x_j - u_j) \le \epsilon_{feas}$.
  - Verification of integer integrality constraints for MILP: $|x_j - \text{round}(x_j)| \le \epsilon_{int}$.
- **Dual Feasibility & Residuals**:
  - Evaluation of dual constraint residuals: $\|A^Ty + z - c\|_{\infty} \le \epsilon_{dual}$.
  - Complementary slackness verification: $|x^T z| \le \epsilon_{comp}$.
- **Optimality Auditing**:
  - Independent objective re-evaluation: $f(x) = c^T x$ (LP/MILP) or $\frac{1}{2}x^T Q x + c^T x$ (QP).
  - Karush-Kuhn-Tucker (KKT) condition certificates.

## Architectural Boundaries
- Consumes raw models from `src/model/` and candidate solutions from `src/algorithms/`.
- Must remain strictly decoupled from solver internals to guarantee unbiased verification.
