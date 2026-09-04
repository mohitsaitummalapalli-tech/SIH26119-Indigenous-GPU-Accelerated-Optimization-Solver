#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include "numerics/dense_vector.hpp"
#include <span>
#include <vector>

namespace sih26119 {

/// Mathematically defined column-major dense matrix.
/// Memory layout: linear_index(i, j) = i + j * rows.
class DenseMatrix {
public:
    DenseMatrix() noexcept = default;

    /// Creates a DenseMatrix of dimension (rows x cols) initialized with a finite scalar.
    [[nodiscard]] static Result<DenseMatrix> create(Dimension rows, Dimension cols, Scalar initial_value = 0.0);

    /// Matrix dimensions.
    [[nodiscard]] Dimension rows() const noexcept { return rows_; }
    [[nodiscard]] Dimension cols() const noexcept { return cols_; }

    /// Raw pointer to underlying contiguous memory buffer.
    [[nodiscard]] Scalar* data() noexcept { return storage_.data(); }
    [[nodiscard]] const Scalar* data() const noexcept { return storage_.data(); }

    /// Unchecked element accessor (hot-path).
    /// Precondition: 0 <= row < rows() && 0 <= col < cols().
    [[nodiscard]] Scalar& operator()(Index row, Index col) noexcept {
        return storage_[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * rows_];
    }

    /// Unchecked const element accessor (hot-path).
    /// Precondition: 0 <= row < rows() && 0 <= col < cols().
    [[nodiscard]] const Scalar& operator()(Index row, Index col) const noexcept {
        return storage_[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * rows_];
    }

    /// Checked element access. Returns StatusCode::InvalidArgument if out of bounds.
    [[nodiscard]] Result<Scalar> at(Index row, Index col) const noexcept;

    /// Checked element modification. Returns error if out of bounds or val is non-finite.
    [[nodiscard]] Status set(Index row, Index col, Scalar val) noexcept;

    /// Fills all matrix elements with a finite scalar value.
    [[nodiscard]] Status fill(Scalar val) noexcept;

    /// Hot-path zero-allocation matrix-vector product in-place: y = Ax using caller-owned scratch workspace.
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

    /// Convenience in-place matrix-vector product: y = Ax.
    /// Allocates temporary scratch workspace internally and forwards to the 3-argument overload.
    [[nodiscard]] Status multiply(const DenseVector& x, DenseVector& y) const;

    /// Convenience value-returning matrix-vector product: returns y = Ax.
    /// Allocates destination vector and temporary scratch workspace.
    [[nodiscard]] Result<DenseVector> multiply(const DenseVector& x) const;

private:
    std::vector<Scalar> storage_;
    Dimension rows_ = 0;
    Dimension cols_ = 0;
};

} // namespace sih26119
