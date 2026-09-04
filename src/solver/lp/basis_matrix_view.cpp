#include "solver/lp/basis_matrix_view.hpp"

namespace sih26119 {

BasisMatrixView::BasisMatrixView(const SparseMatrix& A, const Basis& basis)
    : A_(A), basis_(basis)
{
}

Result<Index> BasisMatrixView::original_column_index(Index basis_col) const noexcept {
    if (basis_col >= basis_.num_rows()) {
        return Status::error(StatusCode::InvalidArgument,
            "Basis column index out of range (basis_col >= m)");
    }
    return basis_.basic_variable(basis_col);
}

Result<Scalar> BasisMatrixView::get(Index row, Index basis_col) const noexcept {
    if (row >= basis_.num_rows()) {
        return Status::error(StatusCode::InvalidArgument,
            "Basis row index out of range (row >= m)");
    }
    auto orig_col_res = original_column_index(basis_col);
    if (!orig_col_res.is_ok()) {
        return orig_col_res.status();
    }
    return A_.get(row, orig_col_res.value());
}

} // namespace sih26119
