# Mathematical Foundations & Formulations

This directory contains mathematical formulations, optimality theory, duality derivations, and Karush-Kuhn-Tucker (KKT) conditions for the optimization problem classes targeted by **SIH26119**:

## 1. Linear Programming (LP)
Standard Primal Form:
$$\min_{x} \quad c^T x \quad \text{subject to} \quad A x = b, \quad l \le x \le u$$

Dual Form:
$$\max_{y, z_l, z_u} \quad b^T y + l^T z_l - u^T z_u \quad \text{subject to} \quad A^T y + z_l - z_u = c, \quad z_l \ge 0, \quad z_u \ge 0$$

## 2. Mixed-Integer Linear Programming (MILP)
$$\min_{x} \quad c^T x \quad \text{subject to} \quad A x \le b, \quad x_j \in \mathbb{Z} \quad \forall j \in I$$

## 3. Convex Quadratic Programming (QP)
$$\min_{x} \quad \frac{1}{2} x^T Q x + c^T x \quad \text{subject to} \quad A x = b, \quad x \ge 0 \quad (Q \succeq 0)$$

## Document Index
- Mathematical theory notes will be added during algorithmic development phases.
