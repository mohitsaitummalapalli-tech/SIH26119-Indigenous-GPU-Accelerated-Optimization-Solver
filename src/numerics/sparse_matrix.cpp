#include "numerics/sparse_matrix.hpp"
#include <algorithm>
#include <limits>

namespace sih26119 {

Result<SparseMatrix> SparseMatrix::from_triplets(
    Dimension rows, Dimension cols, std::span<const Triplet> triplets) {
    
    // Dimension overflow safety check for rows + 1
    if (static_cast<uint64_t>(rows) + 1 > static_cast<uint64_t>(std::numeric_limits<Dimension>::max())) {
        return Status::error(StatusCode::InvalidArgument, "rows + 1 exceeds maximum Dimension in SparseMatrix::from_triplets");
    }

    // Validate coordinates and finite values
    for (const auto& t : triplets) {
        if (t.row >= rows) {
            return Status::error(StatusCode::InvalidArgument, "Triplet row index out of matrix bounds");
        }
        if (t.col >= cols) {
            return Status::error(StatusCode::InvalidArgument, "Triplet col index out of matrix bounds");
        }
        if (!is_finite_scalar(t.value)) {
            return Status::error(StatusCode::InvalidArgument, "Triplet value must be a finite Scalar");
        }
    }

    SparseMatrix mat;
    mat.rows_ = rows;
    mat.cols_ = cols;

    if (rows == 0) {
        mat.row_ptr_.assign(1, 0);
        return mat;
    }

    // Deterministic sort: row major, then col
    std::vector<Triplet> sorted(triplets.begin(), triplets.end());
    std::sort(sorted.begin(), sorted.end(), [](const Triplet& a, const Triplet& b) {
        if (a.row != b.row) {
            return a.row < b.row;
        }
        return a.col < b.col;
    });

    // Accumulate duplicate coordinates exactly
    std::vector<Triplet> accumulated;
    accumulated.reserve(sorted.size());
    for (const auto& t : sorted) {
        if (!accumulated.empty() && accumulated.back().row == t.row && accumulated.back().col == t.col) {
            accumulated.back().value += t.value;
        } else {
            accumulated.push_back(t);
        }
    }

    // Exact structural zero elimination (value == 0.0)
    mat.row_ptr_.assign(static_cast<std::size_t>(rows) + 1, 0);
    for (const auto& t : accumulated) {
        if (t.value != kScalarZero) {
            mat.row_ptr_[static_cast<std::size_t>(t.row) + 1]++;
        }
    }

    // Prefix sum to establish row_ptr offsets
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows); ++i) {
        mat.row_ptr_[i + 1] += mat.row_ptr_[i];
    }

    const NonzeroCount total_nnz = mat.row_ptr_[rows];
    if (total_nnz > static_cast<NonzeroCount>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::InvalidArgument, "Total nnz exceeds container size_t limit");
    }

    mat.col_idx_.resize(static_cast<std::size_t>(total_nnz));
    mat.values_.resize(static_cast<std::size_t>(total_nnz));

    std::vector<NonzeroCount> current_row_pos = mat.row_ptr_;
    for (const auto& t : accumulated) {
        if (t.value != kScalarZero) {
            const std::size_t pos = static_cast<std::size_t>(current_row_pos[t.row]++);
            mat.col_idx_[pos] = t.col;
            mat.values_[pos] = t.value;
        }
    }

    // Validate internal invariants before returning
    auto inv_status = mat.validate_invariants();
    if (!inv_status.is_ok()) {
        return inv_status;
    }

    return mat;
}

Status SparseMatrix::validate_invariants() const noexcept {
    if (row_ptr_.size() != static_cast<std::size_t>(rows_) + 1) {
        return Status::error(StatusCode::InvalidArgument, "row_ptr_.size() != rows + 1");
    }
    if (row_ptr_[0] != 0) {
        return Status::error(StatusCode::InvalidArgument, "row_ptr_[0] must be 0");
    }
    if (row_ptr_[rows_] != values_.size()) {
        return Status::error(StatusCode::InvalidArgument, "row_ptr_[rows] != values_.size()");
    }
    if (values_.size() != col_idx_.size()) {
        return Status::error(StatusCode::InvalidArgument, "values_.size() != col_idx_.size()");
    }

    for (Dimension i = 0; i < rows_; ++i) {
        const NonzeroCount start = row_ptr_[i];
        const NonzeroCount end = row_ptr_[static_cast<std::size_t>(i) + 1];
        if (start > end) {
            return Status::error(StatusCode::InvalidArgument, "row_ptr offsets are not monotonic");
        }
        for (NonzeroCount k = start; k < end; ++k) {
            const std::size_t idx = static_cast<std::size_t>(k);
            if (col_idx_[idx] >= cols_) {
                return Status::error(StatusCode::InvalidArgument, "col_idx out of bounds (< cols)");
            }
            if (!is_finite_scalar(values_[idx])) {
                return Status::error(StatusCode::InvalidArgument, "Stored sparse value is not finite");
            }
            if (k + 1 < end) {
                if (col_idx_[idx] >= col_idx_[idx + 1]) {
                    return Status::error(StatusCode::InvalidArgument, "col_idx within row is not strictly increasing");
                }
            }
        }
    }

    return Status::ok();
}

Result<Scalar> SparseMatrix::get(Index row, Index col) const noexcept {
    if (row >= rows_ || col >= cols_) {
        return Status::error(StatusCode::InvalidArgument, "Index out of bounds in SparseMatrix::get");
    }

    const NonzeroCount start = row_ptr_[row];
    const NonzeroCount end = row_ptr_[static_cast<std::size_t>(row) + 1];
    if (start == end) {
        return kScalarZero;
    }

    // Binary search in sorted column indices of row
    auto it_begin = col_idx_.begin() + static_cast<std::ptrdiff_t>(start);
    auto it_end = col_idx_.begin() + static_cast<std::ptrdiff_t>(end);
    auto it = std::lower_bound(it_begin, it_end, col);

    if (it != it_end && *it == col) {
        const std::size_t offset = static_cast<std::size_t>(std::distance(col_idx_.begin(), it));
        return values_[offset];
    }

    return kScalarZero;
}

Status SparseMatrix::multiply(const DenseVector& x, DenseVector& y) const noexcept {
    if (x.size() != cols_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in SparseMatrix::multiply: x.size() != cols");
    }
    if (y.size() != rows_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in SparseMatrix::multiply: y.size() != rows");
    }
    // Explicit aliasing check
    if (&x == &y || (x.size() > 0 && y.size() > 0 && x.data() == y.data())) {
        return Status::error(StatusCode::InvalidArgument, "x and y cannot alias in SparseMatrix::multiply");
    }

    if (rows_ == 0) {
        return Status::ok();
    }

    const Scalar* x_data = x.data();
    Scalar* y_data = y.data();

    for (Dimension i = 0; i < rows_; ++i) {
        const NonzeroCount start = row_ptr_[i];
        const NonzeroCount end = row_ptr_[static_cast<std::size_t>(i) + 1];
        Scalar sum = kScalarZero;
        for (NonzeroCount k = start; k < end; ++k) {
            const std::size_t idx = static_cast<std::size_t>(k);
            sum += values_[idx] * x_data[col_idx_[idx]];
        }
        if (!is_finite_scalar(sum)) {
            return Status::error(StatusCode::InvalidArgument, "SpMV produced non-finite result in y");
        }
        y_data[i] = sum;
    }

    return Status::ok();
}

Result<DenseVector> SparseMatrix::multiply(const DenseVector& x) const {
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

Status SparseMatrix::residual(const DenseVector& b, const DenseVector& x, DenseVector& r) const noexcept {
    if (b.size() != rows_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in SparseMatrix::residual: b.size() != rows");
    }
    if (x.size() != cols_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in SparseMatrix::residual: x.size() != cols");
    }
    if (r.size() != rows_) {
        return Status::error(StatusCode::InvalidArgument, "Dimension mismatch in SparseMatrix::residual: r.size() != rows");
    }
    // Explicit aliasing check: r cannot alias x
    if (&r == &x || (r.size() > 0 && x.size() > 0 && r.data() == x.data())) {
        return Status::error(StatusCode::InvalidArgument, "x and r cannot alias in SparseMatrix::residual");
    }

    if (rows_ == 0) {
        return Status::ok();
    }

    const Scalar* b_data = b.data();
    const Scalar* x_data = x.data();
    Scalar* r_data = r.data();

    for (Dimension i = 0; i < rows_; ++i) {
        const NonzeroCount start = row_ptr_[i];
        const NonzeroCount end = row_ptr_[static_cast<std::size_t>(i) + 1];
        Scalar ax_i = kScalarZero;
        for (NonzeroCount k = start; k < end; ++k) {
            const std::size_t idx = static_cast<std::size_t>(k);
            ax_i += values_[idx] * x_data[col_idx_[idx]];
        }
        const Scalar res_val = b_data[i] - ax_i;
        if (!is_finite_scalar(res_val)) {
            return Status::error(StatusCode::InvalidArgument, "SparseMatrix::residual produced non-finite residual");
        }
        r_data[i] = res_val;
    }

    return Status::ok();
}

Result<DenseVector> SparseMatrix::residual(const DenseVector& b, const DenseVector& x) const {
    auto r_res = DenseVector::create(rows_, kScalarZero);
    if (!r_res.ok()) {
        return r_res.status();
    }
    DenseVector r = std::move(r_res.value());
    auto status = residual(b, x, r);
    if (!status.is_ok()) {
        return status;
    }
    return r;
}

} // namespace sih26119
