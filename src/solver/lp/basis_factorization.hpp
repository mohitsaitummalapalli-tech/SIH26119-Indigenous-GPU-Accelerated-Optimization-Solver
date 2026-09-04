#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include "numerics/dense_vector.hpp"
#include "solver/lp/basis_matrix_view.hpp"
#include <vector>
#include <cstdint>

namespace sih26119 {

/// State of the basis factorization object.
enum class FactorizationState : uint8_t {
    Empty,      ///< Uninitialized or reset.
    Factored,   ///< Successfully factorized and valid for solves.
    Failed      ///< Factorization failed or invalidated.
};

/// Configurable numerical tolerances for basis factorization and solves.
struct FactorizationTolerances {
    Scalar singularity_tol{1e-12};    ///< Min acceptable diagonal pivot |U_kk|
    Scalar residual_tol{1e-8};        ///< Relative solve backward error ceiling
    Scalar max_growth_tol{1e12};      ///< Operational safeguard against extreme growth (empirical cutoff, not a stability theorem)
    Scalar condition_ceiling{1e13};   ///< Operational diagnostic signal for ill-conditioning (non-terminating diagnostic)
    Scalar fact_residual_tol{1e-12};  ///< Scale-aware factorization quality tolerance (reconciled default: 1e-12)
};

/**
 * @brief Basis Factorization layer for LP basis systems:
 *
 *     B x = rhs   (FTRAN)
 *     B^T y = rhs (BTRAN)
 *
 * Factorization Convention:
 *     P B = L U
 *
 * where:
 * - P is a row permutation matrix (represented as a permutation vector pi_r)
 * - L is unit lower triangular (L_ii = 1.0, |L_ij| <= 1.0 in exact arithmetic)
 * - U is upper triangular (U_ij = 0 for i > j)
 * - Standard row partial pivoting is used (no column permutation Q)
 *
 * Zero-Allocation Contract:
 * - After factorization setup, solve() and solve_transpose() perform ZERO dynamic heap allocations.
 * - Caller-owned scratch workspace is used.
 * - Strict pairwise distinctness (no aliasing between rhs, solution, scratch).
 * - Transactional guarantee: On any failure, destination solution is left strictly unmodified.
 */
class BasisFactorization {
public:
    BasisFactorization() = default;

    /**
     * @brief Performs row partial pivoting LU factorization: P B = L U
     *
     * @param basis_view Non-owning logical view of basis columns over constraint matrix A.
     * @param tols Configurable numerical tolerances.
     * @return Status::ok() on success, or StatusCode::NumericalFailure upon breakdown.
     */
    [[nodiscard]] Status factorize(
        const BasisMatrixView& basis_view,
        FactorizationTolerances tols = FactorizationTolerances{});

    /**
     * @brief Solves B * x = rhs (FTRAN) with ZERO dynamic heap allocations.
     *
     * Preconditions:
     * - is_factored() == true
     * - basis.version() == basis_version()
     * - rhs.size() == m
     * - solution.size() == m
     * - scratch.size() >= m
     * - Strict pairwise distinctness (no aliasing between rhs, solution, scratch).
     */
    [[nodiscard]] Status solve(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept;

    /**
     * @brief Solves B^T * y = rhs (BTRAN) with ZERO dynamic heap allocations.
     *
     * Preconditions:
     * - Same as solve().
     */
    [[nodiscard]] Status solve_transpose(
        const DenseVector& rhs,
        DenseVector& solution,
        DenseVector& scratch) const noexcept;

    /// Query whether factorization is ready for solves.
    [[nodiscard]] bool is_factored() const noexcept {
        return state_ == FactorizationState::Factored;
    }

    /// Query current factorization state.
    [[nodiscard]] FactorizationState state() const noexcept {
        return state_;
    }

    /// Recorded Basis version bound to this factorization.
    [[nodiscard]] uint64_t basis_version() const noexcept {
        return basis_version_;
    }

    /// Basis dimension m.
    [[nodiscard]] Dimension dimension() const noexcept {
        return m_;
    }

    /// Condition number estimate kappa^*(B) (diagnostic only).
    [[nodiscard]] Scalar condition_estimate() const noexcept {
        return condition_estimate_;
    }

    /// Observed pivot growth factor rho_m = max|M^(k)| / max|B|.
    [[nodiscard]] Scalar pivot_growth() const noexcept {
        return pivot_growth_;
    }

    /// Infinity norm ||B||_inf.
    [[nodiscard]] Scalar b_norm_inf() const noexcept {
        return b_norm_inf_;
    }

    /// 1-norm ||B||_1.
    [[nodiscard]] Scalar b_norm_1() const noexcept {
        return b_norm_1_;
    }

    /// Active tolerances.
    [[nodiscard]] const FactorizationTolerances& tolerances() const noexcept {
        return tols_;
    }

    /// Explicitly invalidates the factorization, resetting state to Failed.
    void invalidate() noexcept;

    /// Verifies scale-aware factorization residual ||P B - L U||_inf.
    [[nodiscard]] Result<Scalar> compute_factorization_residual() const;

    /// Access row permutation pi_r (length m).
    [[nodiscard]] const std::vector<Index>& row_permutation() const noexcept {
        return pi_r_;
    }

    /// Access inverse row permutation pi_r_inv (length m).
    [[nodiscard]] const std::vector<Index>& row_permutation_inv() const noexcept {
        return pi_r_inv_;
    }

    /// Unchecked element access into LU packed storage (row, col).
    [[nodiscard]] Scalar lu_entry(Index row, Index col) const noexcept {
        return lu_data_[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * m_];
    }

    /// Unchecked element access into materialized dense B (row, col).
    [[nodiscard]] Scalar b_entry(Index row, Index col) const noexcept {
        return b_dense_[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * m_];
    }

private:
    FactorizationState state_{FactorizationState::Empty};
    Dimension m_{0};
    uint64_t basis_version_{0};
    FactorizationTolerances tols_{};

    const BasisMatrixView* basis_view_{nullptr};

    std::vector<Scalar> lu_data_;        ///< m x m column-major: L below diagonal, U on/above diagonal
    std::vector<Scalar> b_dense_;        ///< Copy of materialized B (m x m column-major)
    std::vector<Index> pi_r_;            ///< Row permutation: row i of PB is row pi_r[i] of B
    std::vector<Index> pi_r_inv_;        ///< Inverse row permutation: pi_r_inv[pi_r[i]] = i

    Scalar b_norm_inf_{0.0};
    Scalar b_norm_1_{0.0};
    Scalar pivot_growth_{1.0};
    Scalar condition_estimate_{1.0};

    // Helper for condition estimation
    void estimate_condition_1norm();
};

} // namespace sih26119
