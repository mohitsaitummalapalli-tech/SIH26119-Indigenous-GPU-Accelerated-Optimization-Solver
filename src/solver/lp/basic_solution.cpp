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

    auto r_res = DenseVector::create(A.rows());
    if (!r_res.is_ok()) return r_res.status();
    DenseVector residual_out = std::move(r_res.value());

    auto s_res = DenseVector::create(A.rows());
    if (!s_res.is_ok()) return s_res.status();
    DenseVector residual_scratch = std::move(s_res.value());

    return check_primal_feasibility(A, b, sol, residual_scratch, residual_out, feas_tol);
}

Status BasicSolution::check_primal_feasibility(
    const SparseMatrix& A,
    const DenseVector& b,
    const BasicSolution& sol,
    DenseVector& residual_scratch,
    DenseVector& residual_out,
    Scalar feas_tol)
{
    const Dimension m = A.rows();
    const Dimension n = A.cols();

    if (m != b.size() || m != sol.num_rows() || n != sol.num_cols()) {
        return Status::error(StatusCode::InvalidArgument,
            "Dimension mismatch in check_primal_feasibility");
    }
    if (residual_scratch.size() < m || residual_out.size() < m) {
        return Status::error(StatusCode::InvalidArgument,
            "Residual workspace buffers smaller than matrix rows");
    }

    // 1. Non-negativity check: x_B[i] >= -feas_tol
    for (Dimension i = 0; i < m; ++i) {
        Scalar val = sol.x_B_.at(i).value();
        if (val < -feas_tol) {
            return Status::error(StatusCode::InvalidBounds,
                "Basic solution violates non-negativity: coordinate " + std::to_string(i) +
                " has value " + std::to_string(val));
        }
    }

    // 2. Compute residual directly: r = b - A x
    // Since x_N = 0, for each row i:
    // (A x)_i = sum_{nz in row i} A.values[nz] * x[A.col[nz]]
    // Where x[col] is x_B[row_of_col] if col is basic, and 0 if col is nonbasic.
    // Build a temporary fast column lookup from basic_vars
    std::vector<Scalar> col_values(static_cast<std::size_t>(n), 0.0);
    for (Dimension i = 0; i < m; ++i) {
        col_values[sol.basic_vars_[i]] = sol.x_B_.at(i).value();
    }

    Scalar max_residual = 0.0;
    for (Dimension i = 0; i < m; ++i) {
        Scalar ax_i = 0.0;
        const NonzeroCount r_start = A.row_ptr()[i];
        const NonzeroCount r_end = A.row_ptr()[static_cast<std::size_t>(i) + 1];

        for (NonzeroCount nz = r_start; nz < r_end; ++nz) {
            const Index col = A.col_idx()[static_cast<std::size_t>(nz)];
            const Scalar a_val = A.values()[static_cast<std::size_t>(nz)];
            ax_i += a_val * col_values[col];
        }

        const Scalar b_i = b.at(i).value();
        const Scalar res_i = b_i - ax_i;
        auto st_set = residual_out.set(i, res_i);
        if (!st_set.is_ok()) return st_set;

        max_residual = std::max(max_residual, std::abs(res_i));
    }

    if (max_residual > feas_tol) {
        return Status::error(StatusCode::InvalidBounds,
            "Basic solution violates equality constraints: max residual " +
            std::to_string(max_residual) + " exceeds tolerance " + std::to_string(feas_tol));
    }

    return Status::ok();
}

} // namespace sih26119
