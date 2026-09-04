#pragma once

#include "core/result.hpp"
#include "numerics/sparse_matrix.hpp"
#include "solver/lp/basis.hpp"

namespace sih26119 {

/**
 * @brief Non-owning logical view of the basis matrix:
 *
 *     B = [ A[:, B(0)], A[:, B(1)], ..., A[:, B(m-1)] ]
 *
 * B is an m x m square submatrix formed by selecting the m basic columns of A
 * in the exact row order specified by the Basis mapping.
 *
 * DESIGN CONTRACTS:
 * 1. Non-owning logical view: Holds references to an external SparseMatrix A
 *    and a Basis. Performs NO duplication or materialization of B.
 * 2. Lifetime requirement: The referenced SparseMatrix A and Basis MUST outlive
 *    the BasisMatrixView object.
 * 3. Phase 3C boundary: Provides column-mapping access for future basis
 *    factorization (e.g. LU factorization). Does NOT implement solve or factorization.
 */
class BasisMatrixView {
public:
    /**
     * @brief Constructs a non-owning BasisMatrixView.
     *
     * @param A Reference to the standard form sparse constraint matrix A (m x n).
     * @param basis Reference to the validated Basis mapping.
     */
    BasisMatrixView(const SparseMatrix& A, const Basis& basis);

    /// Number of rows in B (m).
    [[nodiscard]] Dimension num_rows() const noexcept { return basis_.num_rows(); }

    /// Number of columns in B (m).
    [[nodiscard]] Dimension num_cols() const noexcept { return basis_.num_rows(); }

    /// Reference to the underlying constraint matrix A.
    [[nodiscard]] const SparseMatrix& matrix_A() const noexcept { return A_; }

    /// Reference to the underlying Basis mapping.
    [[nodiscard]] const Basis& basis() const noexcept { return basis_; }

    /**
     * @brief Returns the original column index in A for the k-th column of B:
     *
     *     orig_col = B(k)
     *
     * @param basis_col Index k in [0, m - 1].
     */
    [[nodiscard]] Result<Index> original_column_index(Index basis_col) const noexcept;

    /**
     * @brief Retrieves the entry B(row, basis_col) = A(row, B(basis_col)).
     */
    [[nodiscard]] Result<Scalar> get(Index row, Index basis_col) const noexcept;

private:
    const SparseMatrix& A_;
    const Basis& basis_;
};

} // namespace sih26119
