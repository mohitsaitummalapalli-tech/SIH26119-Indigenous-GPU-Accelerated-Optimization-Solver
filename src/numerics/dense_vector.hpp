#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include <span>
#include <vector>

namespace sih26119 {

/// Minimal, mathematically rigorous dense vector abstraction.
class DenseVector {
public:
    /// Default constructs an empty vector.
    DenseVector() noexcept = default;

    /// Creates a DenseVector with specified dimension and initial finite scalar value.
    [[nodiscard]] static Result<DenseVector> create(Dimension size, Scalar initial_value = 0.0);

    /// Creates a DenseVector from a contiguous span of finite scalar values.
    [[nodiscard]] static Result<DenseVector> from_values(std::span<const Scalar> values);

    /// Returns the dimension of the vector.
    [[nodiscard]] Dimension size() const noexcept { return size_; }

    /// Returns whether the vector has dimension zero.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Raw pointer to underlying contiguous memory buffer.
    [[nodiscard]] Scalar* data() noexcept { return storage_.data(); }
    [[nodiscard]] const Scalar* data() const noexcept { return storage_.data(); }

    /// Span representation of the contiguous elements.
    [[nodiscard]] std::span<Scalar> as_span() noexcept { return std::span<Scalar>(storage_.data(), size_); }
    [[nodiscard]] std::span<const Scalar> as_span() const noexcept { return std::span<const Scalar>(storage_.data(), size_); }

    /// Unchecked hot-path element access.
    /// Precondition: 0 <= i < size().
    [[nodiscard]] Scalar& operator[](Index i) noexcept {
        return storage_[i];
    }

    /// Unchecked hot-path const element access.
    /// Precondition: 0 <= i < size().
    [[nodiscard]] const Scalar& operator[](Index i) const noexcept {
        return storage_[i];
    }

    /// Checked element access. Returns StatusCode::InvalidArgument if out of bounds.
    [[nodiscard]] Result<Scalar> at(Index i) const noexcept;

    /// Checked element modification. Returns error if out of bounds or if val is non-finite.
    [[nodiscard]] Status set(Index i, Scalar val) noexcept;

    /// Fills all elements with a finite scalar value.
    [[nodiscard]] Status fill(Scalar val) noexcept;

    /// Scales all elements in-place by alpha (y <- alpha * y).
    [[nodiscard]] Status scale(Scalar alpha) noexcept;

    /// AXPY operation in-place: y <- alpha * x + y.
    /// Returns an explicit error if dimensions mismatch or alpha is non-finite.
    [[nodiscard]] Status axpy(Scalar alpha, const DenseVector& x) noexcept;

    /// Vector dot product: dot(x, y) = sum_i (x_i * y_i).
    /// Returns an explicit error if dimensions mismatch.
    [[nodiscard]] Result<Scalar> dot(const DenseVector& other) const noexcept;

    /// Euclidean 2-norm with scaled accumulation.
    [[nodiscard]] Result<Scalar> norm2() const noexcept;

    /// Infinity norm (maximum absolute value).
    [[nodiscard]] Result<Scalar> norm_inf() const noexcept;

private:
    std::vector<Scalar> storage_;
    Dimension size_ = 0;
};

} // namespace sih26119
