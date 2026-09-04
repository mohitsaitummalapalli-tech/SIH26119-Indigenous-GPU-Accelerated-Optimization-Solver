#include "solver/lp/basic_solution.hpp"
#include <cmath>
#include <algorithm>

namespace sih26119 {

Result<BasicSolution> BasicSolution::create(
    const Basis& basis,
    DenseVector x_B)
{
    if (!basis.is_structurally_valid()) {
        return Status::error(StatusCode::InconsistentModel, "Cannot create solution from invalid basis");
    }

    if (x_B.size() != basis.num_rows()) {
        return Status::error(StatusCode::InvalidArgument,
            "BasicSolution dimension mismatch: x_B.size() != basis.num_rows()");
    }

    for (Dimension i = 0; i < x_B.size(); ++i) {
        auto val_res = x_B.at(i);
        if (!val_res.is_ok()) return val_res.status();
        if (!std::isfinite(val_res.value())) {
            return Status::error(StatusCode::InvalidArgument,
                "Non-finite value detected in basic solution vector x_B");
        }
    }

    BasicSolution sol;
    sol.num_cols_ = basis.num_cols();
    sol.basic_vars_ = basis.basic_variables();
    sol.x_B_ = std::move(x_B);
    return sol;
}

Result<DenseVector> BasicSolution::expand_full_primal() const {
    auto res = DenseVector::create(num_cols_);
    if (!res.is_ok()) {
        return res.status();
    }
    DenseVector x_full = std::move(res.value());
    auto st = expand_full_primal(x_full);
    if (!st.is_ok()) {
        return st;
    }
    return x_full;
}

Status BasicSolution::expand_full_primal(DenseVector& x_full) const noexcept {
    if (x_full.size() != num_cols_) {
        return Status::error(StatusCode::InvalidArgument,
            "Destination vector size does not match num_cols()");
    }

    // Zero out the entire vector (ensuring x_N = 0)
    for (Dimension j = 0; j < num_cols_; ++j) {
        auto st = x_full.set(j, 0.0);
        if (!st.is_ok()) return st;
    }

    // Scatter x_B[i] into x_full[B[i]]
    const Dimension m = x_B_.size();
    for (Dimension i = 0; i < m; ++i) {
        Index col = basic_vars_[i];
        if (col >= num_cols_) {
            return Status::error(StatusCode::InconsistentModel, "Basic variable index out of range");
        }
        auto val_res = x_B_.at(i);
        if (!val_res.is_ok()) return val_res.status();
        auto st = x_full.set(col, val_res.value());
        if (!st.is_ok()) return st;
    }

    return Status::ok();
}

Status BasicSolution::check_primal_feasibility(
    const SparseMatrix& A,
    const DenseVector& b,
    const BasicSolution& sol,
    Scalar feas_tol)
{
    if (A.rows() != b.size() || A.rows() != sol.num_rows() || A.cols() != sol.num_cols()) {
        return Status::error(StatusCode::InvalidArgument,
            "Dimension mismatch during primal feasibility verification");
    }

    auto x_res = DenseVector::create(A.cols());
    if (!x_res.is_ok()) return x_res.status();
    DenseVector x_workspace = std::move(x_res.value());

    auto r_res = DenseVector::create(A.rows());
    if (!r_res.is_ok()) return r_res.status();
    DenseVector residual_out = std::move(r_res.value());

    auto s_res = DenseVector::create(A.rows());
    if (!s_res.is_ok()) return s_res.status();
    DenseVector residual_scratch = std::move(s_res.value());

    return check_primal_feasibility(A, b, sol, x_workspace, residual_scratch, residual_out, feas_tol);
}

Status BasicSolution::check_primal_feasibility(
    const SparseMatrix& A,
    const DenseVector& b,
    const BasicSolution& sol,
    DenseVector& residual_scratch,
    DenseVector& residual_out,
    Scalar feas_tol)
{
    auto x_res = DenseVector::create(A.cols());
    if (!x_res.is_ok()) return x_res.status();
    DenseVector x_workspace = std::move(x_res.value());

    return check_primal_feasibility(A, b, sol, x_workspace, residual_scratch, residual_out, feas_tol);
}

Status BasicSolution::check_primal_feasibility(
    const SparseMatrix& A,
    const DenseVector& b,
    const BasicSolution& sol,
    DenseVector& x_workspace,
    DenseVector& residual_scratch,
    DenseVector& residual_out,
    Scalar feas_tol)
{
    const Dimension m = A.rows();
    const Dimension n = A.cols();

    if (!std::isfinite(feas_tol) || feas_tol < 0.0) {
        return Status::error(StatusCode::InvalidArgument,
            "feas_tol must be finite and non-negative");
    }

    // 1. Dimension validation
    if (m != b.size() || m != sol.num_rows() || n != sol.num_cols()) {
        return Status::error(StatusCode::InvalidArgument,
            "Dimension mismatch in check_primal_feasibility");
    }
    if (x_workspace.size() != n) {
        return Status::error(StatusCode::InvalidArgument,
            "x_workspace.size() != A.cols()");
    }
    if (residual_scratch.size() < m) {
        return Status::error(StatusCode::InvalidArgument,
            "residual_scratch.size() < A.rows()");
    }
    if (residual_out.size() != m) {
        return Status::error(StatusCode::InvalidArgument,
            "residual_out.size() != A.rows()");
    }

    // 2. Strict pairwise aliasing contract (Phase 2 compliance)
    if (&b == &x_workspace || (b.size() > 0 && x_workspace.size() > 0 && b.data() == x_workspace.data())) {
        return Status::error(StatusCode::InvalidArgument, "b and x_workspace cannot alias");
    }
    if (&b == &residual_scratch || (b.size() > 0 && residual_scratch.size() > 0 && b.data() == residual_scratch.data())) {
        return Status::error(StatusCode::InvalidArgument, "b and residual_scratch cannot alias");
    }
    if (&b == &residual_out || (b.size() > 0 && residual_out.size() > 0 && b.data() == residual_out.data())) {
        return Status::error(StatusCode::InvalidArgument, "b and residual_out cannot alias");
    }
    if (&x_workspace == &residual_scratch || (x_workspace.size() > 0 && residual_scratch.size() > 0 && x_workspace.data() == residual_scratch.data())) {
        return Status::error(StatusCode::InvalidArgument, "x_workspace and residual_scratch cannot alias");
    }
    if (&x_workspace == &residual_out || (x_workspace.size() > 0 && residual_out.size() > 0 && x_workspace.data() == residual_out.data())) {
        return Status::error(StatusCode::InvalidArgument, "x_workspace and residual_out cannot alias");
    }
    if (&residual_scratch == &residual_out || (residual_scratch.size() > 0 && residual_out.size() > 0 && residual_scratch.data() == residual_out.data())) {
        return Status::error(StatusCode::InvalidArgument, "residual_scratch and residual_out cannot alias");
    }

    // 3. Reject non-finite data in b
    for (Dimension i = 0; i < m; ++i) {
        Scalar b_val = b.at(i).value();
        if (!std::isfinite(b_val)) {
            return Status::error(StatusCode::InvalidArgument,
                "Non-finite value in RHS vector b");
        }
    }

    // 4. Check primal non-negativity: x_B[i] >= -feas_tol and finite
    for (Dimension i = 0; i < m; ++i) {
        Scalar val = sol.x_B_.at(i).value();
        if (!std::isfinite(val)) {
            return Status::error(StatusCode::InvalidArgument,
                "Non-finite value in basic solution coordinate");
        }
        if (val < -feas_tol) {
            return Status::error(StatusCode::InvalidBounds,
                "Basic solution violates non-negativity: coordinate " + std::to_string(i) +
                " has value " + std::to_string(val));
        }
    }

    // 5. If m == 0, system has no constraints; vacuously feasible
    if (m == 0) {
        return Status::ok();
    }

    // 6. Zero-allocation expansion into x_workspace
    auto exp_st = sol.expand_full_primal(x_workspace);
    if (!exp_st.is_ok()) {
        return exp_st;
    }

    // 7. Compute residual r = b - Ax via Phase 2 authoritative SparseMatrix::residual
    auto res_st = A.residual(b, x_workspace, residual_out, residual_scratch);
    if (!res_st.is_ok()) {
        return res_st;
    }

    // 8. Verify max infinity norm of residual
    Scalar max_residual = 0.0;
    for (Dimension i = 0; i < m; ++i) {
        Scalar r_i = std::abs(residual_out.at(i).value());
        if (r_i > max_residual) {
            max_residual = r_i;
        }
    }

    if (max_residual > feas_tol) {
        return Status::error(StatusCode::InvalidBounds,
            "Basic solution violates equality constraints: max residual " +
            std::to_string(max_residual) + " exceeds tolerance " + std::to_string(feas_tol));
    }

    return Status::ok();
}

} // namespace sih26119
