#include "numerics/dense_matrix.hpp"
#include <limits>

namespace sih26119 {

Result<DenseMatrix> DenseMatrix::create(Dimension rows, Dimension cols, Scalar initial_value) {
    if (!is_finite_scalar(initial_value)) {
        return Status::error(StatusCode::InvalidArgument, "Initial value for DenseMatrix must be a finite Scalar");
    }
    const uint64_t total_elements = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    if (total_elements > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::InvalidArgument, "DenseMatrix dimension overflow (rows * cols exceeds size_t max)");
    }

    DenseMatrix mat;
    mat.rows_ = rows;
    mat.cols_ = cols;
    mat.storage_.assign(static_cast<std::size_t>(total_elements), initial_value);
    return mat;
}

Result<Scalar> DenseMatrix::at(Index row, Index col) const noexcept {
    if (row >= rows_ || col >= cols_) {
        return Status::error(StatusCode::InvalidArgument, "Index out of bounds in DenseMatrix::at");
    }
    const std::size_t idx = static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * rows_;
    return storage_[idx];
}

Status DenseMatrix::set(Index row, Index col, Scalar val) noexcept {
    if (row >= rows_ || col >= cols_) {
        return Status::error(StatusCode::InvalidArgument, "Index out of bounds in DenseMatrix::set");
    }
    if (!is_finite_scalar(val)) {
        return Status::error(StatusCode::InvalidArgument, "Value in DenseMatrix::set must be a finite Scalar");
    }
    const std::size_t idx = static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * rows_;
    storage_[idx] = val;
    return Status::ok();
}

Status DenseMatrix::fill(Scalar val) noexcept {
    if (!is_finite_scalar(val)) {
        return Status::error(StatusCode::InvalidArgument, "Fill value for DenseMatrix must be a finite Scalar");
    }
    for (Scalar& elem : storage_) {
        elem = val;
    }
    return Status::ok();
}

Status DenseMatrix::multiply(const DenseVector& x, DenseVector& y) const noexcept {
    if (x.size() != cols_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in DenseMatrix::multiply: x.size() != cols");
    }
    if (y.size() != rows_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in DenseMatrix::multiply: y.size() != rows");
    }
    // Explicit aliasing check
    if (&x == &y || (x.size() > 0 && y.size() > 0 && x.data() == y.data())) {
        return Status::error(StatusCode::InvalidArgument, "x and y cannot alias in DenseMatrix::multiply");
    }

    if (rows_ == 0 || cols_ == 0) {
        return y.fill(kScalarZero);
    }

    const Scalar* x_data = x.data();
    const Scalar* mat_data = storage_.data();

    std::vector<Scalar> temp(static_cast<std::size_t>(rows_), kScalarZero);

    // Column-major traversal: contiguous memory reads in inner loop
    for (Dimension j = 0; j < cols_; ++j) {
        const Scalar xj = x_data[j];
        if (xj == kScalarZero) {
            continue;
        }
        const Scalar* col_data = mat_data + static_cast<std::size_t>(j) * rows_;
        for (Dimension i = 0; i < rows_; ++i) {
            temp[i] += col_data[i] * xj;
        }
    }

    // Verify output finiteness transactionally before committing to y
    for (Dimension i = 0; i < rows_; ++i) {
        if (!is_finite_scalar(temp[i])) {
            return Status::error(StatusCode::InvalidArgument, "DenseMatrix::multiply produced non-finite value (arithmetic overflow)");
        }
    }

    Scalar* y_data = y.data();
    for (Dimension i = 0; i < rows_; ++i) {
        y_data[i] = temp[i];
    }

    return Status::ok();
}

Result<DenseVector> DenseMatrix::multiply(const DenseVector& x) const {
    auto y_res = DenseVector::create(rows_, kScalarZero);
    if (!y_res.ok()) {
        return y_res.status();
    }
    DenseVector y = std::move(y_res.value());
    auto status = multiply(x, y);
    if (!status.is_ok()) {
        return status;
    }
    return y;
}

} // namespace sih26119
