#pragma once

#include "core/result.hpp"
#include "numerics/dense_vector.hpp"
#include "numerics/sparse_matrix.hpp"
#include "solver/lp/basis.hpp"
#include <vector>

namespace sih26119 {

/**
 * @brief Representation of an LP Basic Solution:
 *
 *     x_B in R^m  (Primary representation)
 *     x_N = 0     (Implicitly zero)
 *
 * Design contracts:
 * - Primary representation is solely x_B in R^m, avoiding full-vector materialization
 *   for ordinary basis operations.
 * - Optional convenience methods allow expanding to full primal vector x in R^n
 *   with explicitly documented allocation behavior.
 * - Independent primal feasibility checking verifies Ax = b and x >= 0 without
 *   invoking an optimization solver.
 */
class BasicSolution {
public:
    BasicSolution() = default;

    /**
     * @brief Creates a BasicSolution from a validated Basis and basic values x_B.
     *
     * @param basis Validated LP basis.
     * @param x_B DenseVector of length m containing the basic coordinate values.
     * @return Result containing BasicSolution or error if dimensions mismatch or values non-finite.
     */
    [[nodiscard]] static Result<BasicSolution> create(
        const Basis& basis,
        DenseVector x_B);

    /// Primary representation: basic values x_B in R^m.
    [[nodiscard]] const DenseVector& x_B() const noexcept { return x_B_; }

    /// Number of basic variables / constraint rows m.
    [[nodiscard]] Dimension num_rows() const noexcept { return x_B_.size(); }

    /// Number of total standard variables n.
    [[nodiscard]] Dimension num_cols() const noexcept { return num_cols_; }

    /// The basic variable indices in row order.
    [[nodiscard]] const std::vector<Index>& basic_variables() const noexcept { return basic_vars_; }

    /// Retrieves basic value x_B[row].
    [[nodiscard]] Result<Scalar> basic_value(Index row) const noexcept {
        return x_B_.at(row);
    }

    /**
     * @brief Materializes the full primal vector x in R^n, where x[B[i]] = x_B[i]
     *        and nonbasic coordinates are strictly zero (x_N = 0).
     *
     * ALLOCATION CONTRACT: Dynamically allocates a new DenseVector of size n.
     * Intended for export, serialization, and high-level verification, NOT for hot-path simplex loops.
     */
    [[nodiscard]] Result<DenseVector> expand_full_primal() const;

    /**
     * @brief In-place zero-allocation expansion into a caller-owned DenseVector of size n.
     *
     * Precondition: x_full.size() == num_cols().
     * Performs ZERO dynamic heap allocations.
     */
    [[nodiscard]] Status expand_full_primal(DenseVector& x_full) const noexcept;

    /**
     * @brief Performs independent primal feasibility verification for the basic solution:
     *
     *     A x = b,  x >= 0
     *
     * Verifies:
     * - Dimension consistency between A, b, and the solution.
     * - Non-negativity: x_B[i] >= -feas_tol for all i.
     * - Equality residual: ||A x - b||_inf <= feas_tol.
     *
     * Does NOT invoke a solver.
     */
    [[nodiscard]] static Status check_primal_feasibility(
        const SparseMatrix& A,
        const DenseVector& b,
        const BasicSolution& sol,
        Scalar feas_tol = 1e-7);

    /**
     * @brief Zero-allocation overload of check_primal_feasibility using preallocated workspaces.
     *
     * Preconditions:
     * - residual_scratch.size() == A.rows()
     * - residual_out.size() == A.rows()
     */
    [[nodiscard]] static Status check_primal_feasibility(
        const SparseMatrix& A,
        const DenseVector& b,
        const BasicSolution& sol,
        DenseVector& residual_scratch,
        DenseVector& residual_out,
        Scalar feas_tol = 1e-7);

private:
    DenseVector x_B_;
    std::vector<Index> basic_vars_;
    Dimension num_cols_{0};
};

} // namespace sih26119
