#include "numerics/dense_vector.hpp"
#include "numerics/norms.hpp"

namespace sih26119 {

Result<DenseVector> DenseVector::create(Dimension size, Scalar initial_value) {
    if (!is_finite_scalar(initial_value)) {
        return Status::error(StatusCode::InvalidArgument, "Initial value for DenseVector must be a finite Scalar");
    }
    DenseVector vec;
    vec.size_ = size;
    vec.storage_.assign(static_cast<std::size_t>(size), initial_value);
    return vec;
}

Result<DenseVector> DenseVector::from_values(std::span<const Scalar> values) {
    auto dim_res = to_dimension(values.size());
    if (!dim_res.ok()) {
        return dim_res.status();
    }
    for (const Scalar v : values) {
        if (!is_finite_scalar(v)) {
            return Status::error(StatusCode::InvalidArgument, "Values for DenseVector must all be finite Scalars");
        }
    }
    DenseVector vec;
    vec.size_ = dim_res.value();
    vec.storage_.assign(values.begin(), values.end());
    return vec;
}

Result<Scalar> DenseVector::at(Index i) const noexcept {
    if (i >= size_) {
        return Status::error(StatusCode::InvalidArgument, "Index out of bounds in DenseVector::at");
    }
    return storage_[i];
}

Status DenseVector::set(Index i, Scalar val) noexcept {
    if (i >= size_) {
        return Status::error(StatusCode::InvalidArgument, "Index out of bounds in DenseVector::set");
    }
    if (!is_finite_scalar(val)) {
        return Status::error(StatusCode::InvalidArgument, "Value in DenseVector::set must be a finite Scalar");
    }
    storage_[i] = val;
    return Status::ok();
}

Status DenseVector::fill(Scalar val) noexcept {
    if (!is_finite_scalar(val)) {
        return Status::error(StatusCode::InvalidArgument, "Fill value for DenseVector must be a finite Scalar");
    }
    for (Scalar& elem : storage_) {
        elem = val;
    }
    return Status::ok();
}

Status DenseVector::scale(Scalar alpha) noexcept {
    if (!is_finite_scalar(alpha)) {
        return Status::error(StatusCode::InvalidArgument, "Scale factor alpha must be a finite Scalar");
    }
    for (Scalar& elem : storage_) {
        elem *= alpha;
    }
    return Status::ok();
}

Status DenseVector::axpy(Scalar alpha, const DenseVector& x) noexcept {
    if (!is_finite_scalar(alpha)) {
        return Status::error(StatusCode::InvalidArgument, "Alpha scalar in axpy must be finite");
    }
    if (x.size() != size_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in DenseVector::axpy");
    }
    const Scalar* x_data = x.data();
    Scalar* y_data = storage_.data();
    for (Dimension i = 0; i < size_; ++i) {
        y_data[i] += alpha * x_data[i];
    }
    return Status::ok();
}

Result<Scalar> DenseVector::dot(const DenseVector& other) const noexcept {
    if (other.size() != size_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in DenseVector::dot");
    }
    if (size_ == 0) {
        return kScalarZero;
    }
    Scalar sum = kScalarZero;
    const Scalar* x_data = storage_.data();
    const Scalar* y_data = other.data();
    for (Dimension i = 0; i < size_; ++i) {
        sum += x_data[i] * y_data[i];
    }
    if (!is_finite_scalar(sum)) {
        return Status::error(StatusCode::InvalidArgument, "Dot product produced non-finite result");
    }
    return sum;
}

Result<Scalar> DenseVector::norm2() const noexcept {
    return sih26119::norm2(as_span());
}

Result<Scalar> DenseVector::norm_inf() const noexcept {
    return sih26119::norm_inf(as_span());
}

} // namespace sih26119
