#include "solver/lp/basis_factorization.hpp"
#include "numerics/norms.hpp"
#include "numerics/tolerances.hpp"
#include <cmath>
#include <algorithm>

namespace sih26119 {

void BasisFactorization::invalidate() noexcept {
    state_ = FactorizationState::Failed;
    m_ = 0;
    basis_version_ = 0;
    basis_view_ = nullptr;
    lu_data_.clear();
    b_dense_.clear();
    pi_r_.clear();
    pi_r_inv_.clear();
    b_norm_inf_ = 0.0;
    b_norm_1_ = 0.0;
    pivot_growth_ = 1.0;
    condition_estimate_ = 1.0;
}

Status BasisFactorization::factorize(
    const BasisMatrixView& basis_view,
    FactorizationTolerances tols)
{
    if (!basis_view.basis().is_structurally_valid()) {
        invalidate();
        return Status::error(StatusCode::InvalidArgument, "Basis is not structurally valid");
    }

    const Dimension m = basis_view.num_rows();
    if (m == 0) {
        m_ = 0;
        basis_version_ = basis_view.basis().version();
        tols_ = tols;
        basis_view_ = &basis_view;
        lu_data_.clear();
        b_dense_.clear();
        pi_r_.clear();
        pi_r_inv_.clear();
        b_norm_inf_ = 0.0;
        b_norm_1_ = 0.0;
        pivot_growth_ = 1.0;
        condition_estimate_ = 1.0;
        state_ = FactorizationState::Factored;
        return Status::ok();
    }

    const std::size_t m_sz = static_cast<std::size_t>(m);
    std::vector<Scalar> b_dense(m_sz * m_sz, 0.0);

    // Materialize dense B in column-major order from BasisMatrixView
    for (Index j = 0; j < m; ++j) {
        const auto orig_col_res = basis_view.original_column_index(j);
        if (!orig_col_res.is_ok()) {
            invalidate();
            return orig_col_res.status();
        }
        for (Index i = 0; i < m; ++i) {
            const auto val_res = basis_view.get(i, j);
            if (!val_res.is_ok()) {
                invalidate();
                return val_res.status();
            }
            const Scalar val = val_res.value();
            if (!is_finite_scalar(val)) {
                invalidate();
                return Status::error(StatusCode::NumericalFailure, "Non-finite entry encountered in basis matrix");
            }
            b_dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] = val;
        }
    }

    // Compute ||B||_inf, ||B||_1, and initial max element magnitude
    Scalar b_max = 0.0;
    Scalar b_norm_inf = 0.0;
    for (Index i = 0; i < m; ++i) {
        Scalar row_sum = 0.0;
        for (Index j = 0; j < m; ++j) {
            const Scalar abs_val = std::abs(b_dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz]);
            row_sum += abs_val;
            if (abs_val > b_max) {
                b_max = abs_val;
            }
        }
        if (row_sum > b_norm_inf) {
            b_norm_inf = row_sum;
        }
    }

    Scalar b_norm_1 = 0.0;
    for (Index j = 0; j < m; ++j) {
        Scalar col_sum = 0.0;
        for (Index i = 0; i < m; ++i) {
            col_sum += std::abs(b_dense[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz]);
        }
        if (col_sum > b_norm_1) {
            b_norm_1 = col_sum;
        }
    }

    std::vector<Scalar> lu_data = b_dense;
    std::vector<Index> pi_r(m_sz);
    for (Index i = 0; i < m; ++i) {
        pi_r[static_cast<std::size_t>(i)] = i;
    }

    Scalar max_observed_elem = b_max;

    // Row partial pivoting Gaussian elimination: P B = L U
    for (Index k = 0; k < m; ++k) {
        // Step k: find pivot row p = argmax_{i >= k} |M(i, k)| with deterministic tie-breaking (smallest row index)
        const std::size_t k_col_offset = static_cast<std::size_t>(k) * m_sz;
        Scalar pivot_max = std::abs(lu_data[static_cast<std::size_t>(k) + k_col_offset]);
        Index p = k;

        for (Index i = k + 1; i < m; ++i) {
            const Scalar cand = std::abs(lu_data[static_cast<std::size_t>(i) + k_col_offset]);
            if (cand > pivot_max) {
                pivot_max = cand;
                p = i;
            }
        }

        const Scalar pivot_val = lu_data[static_cast<std::size_t>(p) + k_col_offset];
        if (!is_finite_scalar(pivot_val)) {
            invalidate();
            return Status::error(StatusCode::NumericalFailure, "Non-finite pivot encountered during elimination");
        }
        if (std::abs(pivot_val) <= tols.singularity_tol) {
            invalidate();
            return Status::error(StatusCode::NumericalFailure, "Numerically singular basis: pivot magnitude below threshold");
        }

        // Swap row k and row p in both working matrix and permutation vector
        if (p != k) {
            for (std::size_t j = 0; j < m_sz; ++j) {
                std::swap(
                    lu_data[static_cast<std::size_t>(k) + j * m_sz],
                    lu_data[static_cast<std::size_t>(p) + j * m_sz]);
            }
            std::swap(pi_r[static_cast<std::size_t>(k)], pi_r[static_cast<std::size_t>(p)]);
        }

        const Scalar diag = lu_data[static_cast<std::size_t>(k) + k_col_offset];

        // Elimination and multiplier computation: L(i, k) = M(i, k) / U(k, k)
        for (Index i = k + 1; i < m; ++i) {
            const std::size_t i_idx = static_cast<std::size_t>(i) + k_col_offset;
            const Scalar mult = lu_data[i_idx] / diag;
            lu_data[i_idx] = mult; // Store L(i, k) below diagonal

            for (Index j = k + 1; j < m; ++j) {
                const std::size_t j_col_offset = static_cast<std::size_t>(j) * m_sz;
                const Scalar updated = lu_data[static_cast<std::size_t>(i) + j_col_offset] - mult * lu_data[static_cast<std::size_t>(k) + j_col_offset];
                lu_data[static_cast<std::size_t>(i) + j_col_offset] = updated;

                const Scalar entry_mag = std::abs(updated);
                if (entry_mag > max_observed_elem) {
                    max_observed_elem = entry_mag;
                }
            }
        }
    }

    const Scalar pivot_growth = (b_max > 0.0) ? (max_observed_elem / b_max) : 1.0;
    if (!is_finite_scalar(pivot_growth) || pivot_growth > tols.max_growth_tol) {
        invalidate();
        return Status::error(StatusCode::NumericalFailure, "Excessive pivot growth in basis factorization");
    }

    std::vector<Index> pi_r_inv(m_sz);
    for (Index i = 0; i < m; ++i) {
        pi_r_inv[static_cast<std::size_t>(pi_r[static_cast<std::size_t>(i)])] = i;
    }

    m_ = m;
    basis_version_ = basis_view.basis().version();
    tols_ = tols;
    basis_view_ = &basis_view;
    b_dense_ = std::move(b_dense);
    lu_data_ = std::move(lu_data);
    pi_r_ = std::move(pi_r);
    pi_r_inv_ = std::move(pi_r_inv);
    b_norm_inf_ = b_norm_inf;
    b_norm_1_ = b_norm_1;
    pivot_growth_ = pivot_growth;
    state_ = FactorizationState::Factored;

    estimate_condition_1norm();

    return Status::ok();
}

Status BasisFactorization::solve(
    const DenseVector& rhs,
    DenseVector& solution,
    DenseVector& scratch) const noexcept
{
    if (state_ != FactorizationState::Factored) {
        return Status::error(StatusCode::InconsistentModel, "Factorization is not in Factored state");
    }
    if (basis_view_ == nullptr || basis_view_->basis().version() != basis_version_) {
        return Status::error(StatusCode::InconsistentModel, "Basis version mismatch: factorization is stale");
    }

    // Strict pairwise distinctness check
    if (&rhs == &solution || &rhs == &scratch || &solution == &scratch) {
        return Status::error(StatusCode::InvalidArgument, "Aliasing detected: rhs, solution, and scratch must be pairwise distinct");
    }

    if (rhs.size() != m_ || solution.size() != m_) {
        return Status::error(StatusCode::InvalidArgument, "Vector dimension mismatch with basis dimension");
    }
    if (scratch.size() < m_) {
        return Status::error(StatusCode::InvalidArgument, "Scratch capacity is smaller than basis dimension");
    }

    // Finite inputs check
    for (Index i = 0; i < m_; ++i) {
        if (!is_finite_scalar(rhs[i])) {
            return Status::error(StatusCode::InvalidArgument, "Non-finite entry encountered in RHS vector");
        }
    }

    if (m_ == 0) {
        return Status::ok();
    }

    const std::size_t m_sz = static_cast<std::size_t>(m_);

    // FTRAN Solve: L U x = P rhs
    // Step 1 & 2: Apply P to rhs and forward solve through unit lower triangular L into scratch
    for (Index i = 0; i < m_; ++i) {
        const Index orig_row = pi_r_[static_cast<std::size_t>(i)];
        Scalar sum = rhs[orig_row];
        for (Index j = 0; j < i; ++j) {
            sum -= lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * scratch[j];
        }
        scratch[i] = sum;
    }

    // Step 3: Backward solve through upper triangular U in-place in scratch
    for (Index step = m_; step > 0; --step) {
        const Index i = step - 1;
        Scalar sum = scratch[i];
        for (Index j = i + 1; j < m_; ++j) {
            sum -= lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * scratch[j];
        }
        const Scalar diag = lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m_sz];
        scratch[i] = sum / diag;
    }

    // Step 4: Validate finite candidate solution in scratch
    Scalar x_norm_inf = 0.0;
    for (Index i = 0; i < m_; ++i) {
        const Scalar val = scratch[i];
        if (!is_finite_scalar(val)) {
            return Status::error(StatusCode::NumericalFailure, "Non-finite value produced during solve substitution");
        }
        const Scalar abs_val = std::abs(val);
        if (abs_val > x_norm_inf) {
            x_norm_inf = abs_val;
        }
    }

    // Step 5: Independent scale-aware backward residual test ||B x - rhs||_inf
    Scalar rhs_norm_inf = 0.0;
    for (Index i = 0; i < m_; ++i) {
        const Scalar abs_rhs = std::abs(rhs[i]);
        if (abs_rhs > rhs_norm_inf) {
            rhs_norm_inf = abs_rhs;
        }
    }

    Scalar max_res = 0.0;
    for (Index i = 0; i < m_; ++i) {
        Scalar b_x_i = 0.0;
        for (Index j = 0; j < m_; ++j) {
            b_x_i += b_dense_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * scratch[j];
        }
        const Scalar res = std::abs(b_x_i - rhs[i]);
        if (res > max_res) {
            max_res = res;
        }
    }

    const Scalar denom = b_norm_inf_ * x_norm_inf + rhs_norm_inf;
    if (denom == 0.0) {
        if (max_res != 0.0) {
            return Status::error(StatusCode::NumericalFailure, "Zero-denominator residual check failed for solve");
        }
    } else {
        if (max_res > tols_.residual_tol * denom) {
            return Status::error(StatusCode::NumericalFailure, "Solve backward residual exceeded numerical tolerance");
        }
    }

    // Step 6: Commit solution only after successful verification (transactional guarantee)
    for (Index i = 0; i < m_; ++i) {
        solution[i] = scratch[i];
    }

    return Status::ok();
}

Status BasisFactorization::solve_transpose(
    const DenseVector& rhs,
    DenseVector& solution,
    DenseVector& scratch) const noexcept
{
    if (state_ != FactorizationState::Factored) {
        return Status::error(StatusCode::InconsistentModel, "Factorization is not in Factored state");
    }
    if (basis_view_ == nullptr || basis_view_->basis().version() != basis_version_) {
        return Status::error(StatusCode::InconsistentModel, "Basis version mismatch: factorization is stale");
    }

    // Strict pairwise distinctness check
    if (&rhs == &solution || &rhs == &scratch || &solution == &scratch) {
        return Status::error(StatusCode::InvalidArgument, "Aliasing detected: rhs, solution, and scratch must be pairwise distinct");
    }

    if (rhs.size() != m_ || solution.size() != m_) {
        return Status::error(StatusCode::InvalidArgument, "Vector dimension mismatch with basis dimension");
    }
    if (scratch.size() < m_) {
        return Status::error(StatusCode::InvalidArgument, "Scratch capacity is smaller than basis dimension");
    }

    // Finite inputs check
    for (Index i = 0; i < m_; ++i) {
        if (!is_finite_scalar(rhs[i])) {
            return Status::error(StatusCode::InvalidArgument, "Non-finite entry encountered in RHS vector");
        }
    }

    if (m_ == 0) {
        return Status::ok();
    }

    const std::size_t m_sz = static_cast<std::size_t>(m_);

    // BTRAN Solve: B^T y = rhs <=> U^T L^T P y = rhs
    // Step 1: Forward solve through lower triangular U^T: U^T w = rhs
    for (Index i = 0; i < m_; ++i) {
        Scalar sum = rhs[i];
        for (Index j = 0; j < i; ++j) {
            // (U^T)_{i, j} = U_{j, i} = lu_data[j + i * m_sz]
            sum -= lu_data_[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * m_sz] * scratch[j];
        }
        const Scalar diag = lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m_sz];
        scratch[i] = sum / diag;
    }

    // Step 2: Backward solve through unit upper triangular L^T: L^T v = w
    for (Index step = m_; step > 0; --step) {
        const Index i = step - 1;
        Scalar sum = scratch[i];
        for (Index j = i + 1; j < m_; ++j) {
            // (L^T)_{i, j} = L_{j, i} = lu_data[j + i * m_sz]
            sum -= lu_data_[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * m_sz] * scratch[j];
        }
        scratch[i] = sum; // Unit diagonal
    }

    // Step 3: Validate finite candidate v in scratch
    Scalar y_norm_inf = 0.0;
    for (Index i = 0; i < m_; ++i) {
        const Scalar val = scratch[i];
        if (!is_finite_scalar(val)) {
            return Status::error(StatusCode::NumericalFailure, "Non-finite value produced during transpose solve substitution");
        }
        const Scalar abs_val = std::abs(val);
        if (abs_val > y_norm_inf) {
            y_norm_inf = abs_val;
        }
    }

    // Step 4: Independent scale-aware backward residual test ||B^T y - rhs||_inf
    // Here y = P^T v, so y[k] = v[pi_r_inv[k]].
    Scalar rhs_norm_inf = 0.0;
    for (Index i = 0; i < m_; ++i) {
        const Scalar abs_rhs = std::abs(rhs[i]);
        if (abs_rhs > rhs_norm_inf) {
            rhs_norm_inf = abs_rhs;
        }
    }

    Scalar max_res = 0.0;
    for (Index j = 0; j < m_; ++j) {
        Scalar b_t_y_j = 0.0;
        for (Index i = 0; i < m_; ++i) {
            // (B^T)_{j, i} = B_{i, j} = b_dense[i + j * m_sz]
            // y[i] = scratch[pi_r_inv_[i]]
            const Index v_idx = pi_r_inv_[static_cast<std::size_t>(i)];
            b_t_y_j += b_dense_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * scratch[v_idx];
        }
        const Scalar res = std::abs(b_t_y_j - rhs[j]);
        if (res > max_res) {
            max_res = res;
        }
    }

    const Scalar denom = b_norm_inf_ * y_norm_inf + rhs_norm_inf;
    if (denom == 0.0) {
        if (max_res != 0.0) {
            return Status::error(StatusCode::NumericalFailure, "Zero-denominator residual check failed for transpose solve");
        }
    } else {
        if (max_res > tols_.residual_tol * denom) {
            return Status::error(StatusCode::NumericalFailure, "Transpose solve backward residual exceeded numerical tolerance");
        }
    }

    // Step 5: Commit solution: y = P^T v <=> y[pi_r[i]] = v[i]
    for (Index i = 0; i < m_; ++i) {
        const Index orig_row = pi_r_[static_cast<std::size_t>(i)];
        solution[orig_row] = scratch[i];
    }

    return Status::ok();
}

Result<Scalar> BasisFactorization::compute_factorization_residual() const {
    if (state_ != FactorizationState::Factored) {
        return Status::error(StatusCode::InconsistentModel, "Factorization is not in Factored state");
    }
    if (m_ == 0) {
        return 0.0;
    }

    const std::size_t m_sz = static_cast<std::size_t>(m_);
    Scalar max_row_diff = 0.0;
    Scalar pb_norm_inf = 0.0;
    Scalar lu_norm_inf = 0.0;

    for (Index i = 0; i < m_; ++i) {
        const Index orig_row = pi_r_[static_cast<std::size_t>(i)];
        Scalar row_diff = 0.0;
        Scalar pb_row_sum = 0.0;
        Scalar lu_row_sum = 0.0;

        for (Index j = 0; j < m_; ++j) {
            const Scalar pb_ij = b_dense_[static_cast<std::size_t>(orig_row) + static_cast<std::size_t>(j) * m_sz];
            pb_row_sum += std::abs(pb_ij);

            // Compute (L * U)_{i, j} = sum_{k=0}^{min(i, j)} L_{i, k} * U_{k, j}
            Scalar lu_ij = 0.0;
            const Index k_max = std::min(i, j);
            for (Index k = 0; k <= k_max; ++k) {
                const Scalar l_ik = (i == k) ? 1.0 : lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * m_sz];
                const Scalar u_kj = lu_data_[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * m_sz];
                lu_ij += l_ik * u_kj;
            }

            lu_row_sum += std::abs(lu_ij);
            row_diff += std::abs(pb_ij - lu_ij);
        }

        if (row_diff > max_row_diff) {
            max_row_diff = row_diff;
        }
        if (pb_row_sum > pb_norm_inf) {
            pb_norm_inf = pb_row_sum;
        }
        if (lu_row_sum > lu_norm_inf) {
            lu_norm_inf = lu_row_sum;
        }
    }

    const Scalar denom = pb_norm_inf + lu_norm_inf;
    if (denom == 0.0) {
        if (max_row_diff != 0.0) {
            return Status::error(StatusCode::NumericalFailure, "Zero-denominator factorization residual check failed");
        }
    } else {
        if (max_row_diff > tols_.fact_residual_tol * denom) {
            return Status::error(StatusCode::NumericalFailure, "Factorization residual ||P B - L U|| exceeded tolerance");
        }
    }

    return max_row_diff;
}

void BasisFactorization::estimate_condition_1norm() {
    if (m_ <= 1) {
        condition_estimate_ = 1.0;
        return;
    }

    const std::size_t m_sz = static_cast<std::size_t>(m_);
    std::vector<Scalar> x(m_sz, 1.0 / static_cast<Scalar>(m_));
    std::vector<Scalar> w(m_sz, 0.0);
    std::vector<Scalar> z(m_sz, 0.0);

    Scalar gamma = 0.0;
    const int max_iters = 5;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Solve B w = x: L U w = P x
        for (Index i = 0; i < m_; ++i) {
            const Index orig_row = pi_r_[static_cast<std::size_t>(i)];
            Scalar sum = x[static_cast<std::size_t>(orig_row)];
            for (Index j = 0; j < i; ++j) {
                sum -= lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * w[static_cast<std::size_t>(j)];
            }
            w[static_cast<std::size_t>(i)] = sum;
        }
        for (Index step = m_; step > 0; --step) {
            const Index i = step - 1;
            Scalar sum = w[static_cast<std::size_t>(i)];
            for (Index j = i + 1; j < m_; ++j) {
                sum -= lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m_sz] * w[static_cast<std::size_t>(j)];
            }
            const Scalar diag = lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m_sz];
            w[static_cast<std::size_t>(i)] = sum / diag;
        }

        gamma = 0.0;
        for (Index i = 0; i < m_; ++i) {
            gamma += std::abs(w[static_cast<std::size_t>(i)]);
        }

        // xi_i = sign(w_i)
        std::vector<Scalar> xi(m_sz);
        for (Index i = 0; i < m_; ++i) {
            xi[static_cast<std::size_t>(i)] = (w[static_cast<std::size_t>(i)] >= 0.0) ? 1.0 : -1.0;
        }

        // Solve B^T z = xi: U^T L^T P z = xi
        for (Index i = 0; i < m_; ++i) {
            Scalar sum = xi[static_cast<std::size_t>(i)];
            for (Index j = 0; j < i; ++j) {
                sum -= lu_data_[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * m_sz] * z[static_cast<std::size_t>(j)];
            }
            const Scalar diag = lu_data_[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m_sz];
            z[static_cast<std::size_t>(i)] = sum / diag;
        }
        for (Index step = m_; step > 0; --step) {
            const Index i = step - 1;
            Scalar sum = z[static_cast<std::size_t>(i)];
            for (Index j = i + 1; j < m_; ++j) {
                sum -= lu_data_[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * m_sz] * z[static_cast<std::size_t>(j)];
            }
            z[static_cast<std::size_t>(i)] = sum;
        }

        // Unpermute z: z_orig[pi_r[i]] = z[i]
        Scalar z_norm_inf = 0.0;
        Index max_idx = 0;
        Scalar z_dot_x = 0.0;

        for (Index i = 0; i < m_; ++i) {
            const Index orig_row = pi_r_[static_cast<std::size_t>(i)];
            const Scalar z_val = z[static_cast<std::size_t>(i)];
            const Scalar abs_z = std::abs(z_val);
            if (abs_z > z_norm_inf) {
                z_norm_inf = abs_z;
                max_idx = orig_row;
            }
            z_dot_x += z_val * x[static_cast<std::size_t>(orig_row)];
        }

        if (z_norm_inf <= z_dot_x || iter == max_iters - 1) {
            break;
        }

        // Set x = e_{max_idx}
        std::fill(x.begin(), x.end(), 0.0);
        x[static_cast<std::size_t>(max_idx)] = 1.0;
    }

    condition_estimate_ = b_norm_1_ * gamma;
    if (!is_finite_scalar(condition_estimate_) || condition_estimate_ < 1.0) {
        condition_estimate_ = 1.0;
    }
}

} // namespace sih26119
