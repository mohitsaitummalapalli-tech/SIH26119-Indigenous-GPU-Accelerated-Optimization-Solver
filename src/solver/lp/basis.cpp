#include "solver/lp/basis.hpp"
#include <algorithm>

namespace sih26119 {

Result<Basis> Basis::create(
    Dimension num_rows,
    Dimension num_cols,
    std::vector<Index> basic_vars)
{
    // 1. Invariant 1: Dimensions must satisfy m <= n
    if (num_rows > num_cols) {
        return Status::error(StatusCode::InvalidArgument,
            "Basis invariant violated: num_rows exceeds num_cols (m > n)");
    }

    // Invariant 2: Size of basic_vars must be exactly m
    if (basic_vars.size() != static_cast<std::size_t>(num_rows)) {
        return Status::error(StatusCode::InvalidArgument,
            "Basis invariant violated: basic_vars.size() != num_rows");
    }

    Basis b;
    b.num_rows_ = num_rows;
    b.num_cols_ = num_cols;
    b.basic_vars_ = std::move(basic_vars);

    b.var_to_row_.assign(static_cast<std::size_t>(num_cols), kInvalidIndex);
    b.var_to_nonbasic_pos_.assign(static_cast<std::size_t>(num_cols), kInvalidIndex);

    // 2. Invariants 3, 4, 6: Validate basic indices, bounds, and uniqueness
    for (Dimension i = 0; i < num_rows; ++i) {
        const Index var = b.basic_vars_[i];
        if (var >= num_cols) {
            return Status::error(StatusCode::InvalidArgument,
                "Basis invariant violated: basic variable index out of range (j >= n)");
        }
        if (b.var_to_row_[var] != kInvalidIndex) {
            return Status::error(StatusCode::InvalidArgument,
                "Basis invariant violated: duplicate basic variable detected");
        }
        b.var_to_row_[var] = i;
    }

    // 3. Invariant 5: Partition all columns into basic and nonbasic sets
    const Dimension nonbasic_count = num_cols - num_rows;
    b.nonbasic_vars_.reserve(static_cast<std::size_t>(nonbasic_count));

    for (Dimension j = 0; j < num_cols; ++j) {
        if (b.var_to_row_[j] == kInvalidIndex) {
            const auto pos = static_cast<Index>(b.nonbasic_vars_.size());
            b.nonbasic_vars_.push_back(j);
            b.var_to_nonbasic_pos_[j] = pos;
        }
    }

    b.version_ = 1;
    b.is_valid_ = true;
    return b;
}

Status Basis::replace_basic_variable(Index entering_col, Index leaving_row) {
    if (!is_valid_) {
        return Status::error(StatusCode::InconsistentModel, "Basis is invalid");
    }

    // 1. Validation checks (MUST not modify state if any fail)
    if (leaving_row >= num_rows_) {
        return Status::error(StatusCode::InvalidArgument, "Leaving row index out of range");
    }
    if (entering_col >= num_cols_) {
        return Status::error(StatusCode::InvalidArgument, "Entering column index out of range");
    }
    if (var_to_row_[entering_col] != kInvalidIndex) {
        return Status::error(StatusCode::InvalidArgument,
            "Entering variable is already basic in the basis");
    }

    const Index leaving_col = basic_vars_[leaving_row];
    if (leaving_col >= num_cols_ || var_to_row_[leaving_col] != leaving_row) {
        return Status::error(StatusCode::InconsistentModel,
            "Basis invariant corruption: leaving variable not mapped to leaving row");
    }

    const Index nonbasic_pos = var_to_nonbasic_pos_[entering_col];
    if (nonbasic_pos == kInvalidIndex ||
        nonbasic_pos >= static_cast<Index>(nonbasic_vars_.size()) ||
        nonbasic_vars_[nonbasic_pos] != entering_col)
    {
        return Status::error(StatusCode::InconsistentModel,
            "Basis invariant corruption: nonbasic index mapping inconsistent");
    }

    // 2. Transactional state mutation
    basic_vars_[leaving_row] = entering_col;
    var_to_row_[entering_col] = leaving_row;
    var_to_row_[leaving_col] = kInvalidIndex;

    // Swap leaving variable into the nonbasic slot vacated by entering variable
    nonbasic_vars_[nonbasic_pos] = leaving_col;
    var_to_nonbasic_pos_[leaving_col] = nonbasic_pos;
    var_to_nonbasic_pos_[entering_col] = kInvalidIndex;

    version_++;
    return Status::ok();
}

} // namespace sih26119
