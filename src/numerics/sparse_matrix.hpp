#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include "numerics/dense_vector.hpp"
#include <span>
#include <vector>

namespace sih26119 {

/// Coordinate triplet representation for sparse matrix construction.
struct Triplet {
    Dimension row = 0;
    Dimension col = 0;
    Scalar value = 0.0;
};

/// Mathematically defined Compressed Sparse Row (CSR) sparse matrix.
class SparseMatrix {
public:
    SparseMatrix() noexcept = default;

    /// Builds a CSR SparseMatrix from coordinate triplets.
    /// Semantics:
    /// 1. Triplet coordinates must be within [0, rows) x [0, cols) and values must be finite.
    /// 2. Duplicates at the same (row, col) are accumulated by addition.
    /// 3. Entries with exact value == 0.0 are eliminated from CSR storage.
    /// 4. Column indices within each row are sorted strictly increasing.
    [[nodiscard]] static Result<SparseMatrix> from_triplets(
        Dimension rows, Dimension cols, std::span<const Triplet> triplets);

    /// Matrix dimensions and nonzeros.
    [[nodiscard]] Dimension rows() const noexcept { return rows_; }
    [[nodiscard]] Dimension cols() const noexcept { return cols_; }
    [[nodiscard]] NonzeroCount nnz() const noexcept { return static_cast<NonzeroCount>(values_.size()); }

    /// CSR buffer views.
    [[nodiscard]] std::span<const NonzeroCount> row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] std::span<const Index> col_idx() const noexcept { return col_idx_; }
    [[nodiscard]] std::span<const Scalar> values() const noexcept { return values_; }

    /// Element query for coordinate (row, col).
    /// Returns StatusCode::InvalidArgument if (row >= rows() || col >= cols()).
    /// Returns 0.0 for valid coordinates structurally absent from CSR.
    [[nodiscard]] Result<Scalar> get(Index row, Index col) const noexcept;

    /// Verifies all mathematical CSR invariants.
    [[nodiscard]] Status validate_invariants() const noexcept;

    /// Sparse matrix-vector product in-place: y = Ax.
    /// Requirements:
    /// - cols() == x.size()
    /// - rows() == y.size()
    /// - x and y must NOT alias (&x != &y and x.data() != y.data()).
    [[nodiscard]] Status multiply(const DenseVector& x, DenseVector& y) const noexcept;

    /// Sparse matrix-vector product allocating a new DenseVector: returns y = Ax.
    [[nodiscard]] Result<DenseVector> multiply(const DenseVector& x) const;

    /// Computes residual in-place: r = b - Ax.
    /// Requirements:
    /// - cols() == x.size()
    /// - rows() == b.size() == r.size()
    /// - x and r must NOT alias (&x != &r and x.data() != r.data()).
    [[nodiscard]] Status residual(const DenseVector& b, const DenseVector& x, DenseVector& r) const noexcept;

    /// Computes residual allocating a new DenseVector: returns r = b - Ax.
    [[nodiscard]] Result<DenseVector> residual(const DenseVector& b, const DenseVector& x) const;

private:
    std::vector<NonzeroCount> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Scalar> values_;
    Dimension rows_ = 0;
    Dimension cols_ = 0;
};

} // namespace sih26119
