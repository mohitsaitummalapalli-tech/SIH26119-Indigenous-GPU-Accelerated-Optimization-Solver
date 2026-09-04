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

    /// Hot-path zero-allocation sparse matrix-vector product in-place: y = Ax using caller-owned scratch workspace.
    ///
    /// Contract:
    /// - Zero dynamic heap allocations during execution.
    /// - Dimensions: cols() == x.size(), rows() == y.size().
    /// - Scratch capacity: scratch.size() >= rows(). Only elements [0, rows()) may be modified.
    ///   Extra entries [rows(), scratch.size()) remain untouched.
    /// - Aliasing: x, y, and scratch must all be distinct storage objects (no pairwise aliasing).
    /// - Transactional guarantee: y is modified if and only if all outputs [0, rows()) are finite.
    ///   On arithmetic overflow or error, returns StatusCode::InvalidArgument and y remains strictly unmodified.
    ///   (scratch may contain intermediate values on failure).
    [[nodiscard]] Status multiply(const DenseVector& x, DenseVector& y, DenseVector& scratch) const noexcept;

    /// Convenience in-place SpMV: y = Ax.
    /// Allocates temporary scratch workspace internally and forwards to the 3-argument overload.
    [[nodiscard]] Status multiply(const DenseVector& x, DenseVector& y) const;

    /// Convenience value-returning SpMV: returns y = Ax.
    /// Allocates destination vector and temporary scratch workspace.
    [[nodiscard]] Result<DenseVector> multiply(const DenseVector& x) const;

    /// Hot-path zero-allocation residual in-place: r = b - Ax using caller-owned scratch workspace.
    ///
    /// Contract:
    /// - Zero dynamic heap allocations during execution.
    /// - Dimensions: cols() == x.size(), rows() == b.size() == r.size().
    /// - Scratch capacity: scratch.size() >= rows(). Only elements [0, rows()) may be modified.
    ///   Extra entries [rows(), scratch.size()) remain untouched.
    /// - Aliasing: b, x, r, and scratch must all be distinct storage objects (strict pairwise distinctness).
    /// - Transactional guarantee: r is modified if and only if all residuals [0, rows()) are finite.
    ///   On arithmetic overflow or error, returns StatusCode::InvalidArgument and r remains strictly unmodified.
    ///   (scratch may contain intermediate values on failure).
    [[nodiscard]] Status residual(const DenseVector& b, const DenseVector& x, DenseVector& r, DenseVector& scratch) const noexcept;

    /// Convenience in-place residual: r = b - Ax.
    /// Allocates temporary scratch workspace internally and forwards to the 4-argument overload.
    [[nodiscard]] Status residual(const DenseVector& b, const DenseVector& x, DenseVector& r) const;

    /// Convenience value-returning residual: returns r = b - Ax.
    /// Allocates destination vector and temporary scratch workspace.
    [[nodiscard]] Result<DenseVector> residual(const DenseVector& b, const DenseVector& x) const;

private:
    std::vector<NonzeroCount> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Scalar> values_;
    Dimension rows_ = 0;
    Dimension cols_ = 0;
};

} // namespace sih26119
