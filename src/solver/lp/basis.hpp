#pragma once

#include "core/result.hpp"
#include "numerics/index.hpp"
#include <vector>
#include <cstdint>

namespace sih26119 {

/**
 * @brief Representation of an LP basis for a standard form system:
 *
 *     min c^T x  s.t.  A x = b,  x >= 0
 *
 * where A is m x n. A basis is a bijective mapping from row indices
 * {0, ..., m - 1} to a subset of m distinct column indices {B(0), ..., B(m - 1)}.
 *
 * NOTE: Phase 3B validates STRUCTURAL validity only:
 *   - Dimension bounds (m <= n)
 *   - Exactly m basic variables
 *   - Uniqueness and range of indices
 *   - Basic/nonbasic partition and row bijection
 *
 * It does NOT prove numerical nonsingularity of B. Numerical rank/singularity
 * verification is the strict responsibility of Phase 3C factorization.
 */
class Basis {
public:
    Basis() = default;

    /**
     * @brief Constructs and validates a basis from a list of basic column indices.
     *
     * Invariants enforced:
     * 1. m <= n (with m, n within project index bounds).
     * 2. basic_vars.size() == m.
     * 3. Every index in basic_vars satisfies 0 <= j < n.
     * 4. All m basic variables are distinct.
     * 5. Every column 0 <= j < n is partitioned as basic or nonbasic.
     * 6. Row-to-variable and variable-to-row mappings are strictly bijective.
     * 7. For m = 0, creates a valid empty basis with all n columns nonbasic.
     *
     * @param num_rows Number of constraint rows m.
     * @param num_cols Number of standard variables n.
     * @param basic_vars List of m basic column indices in row order.
     * @return Result containing the validated Basis or an error status.
     */
    [[nodiscard]] static Result<Basis> create(
        Dimension num_rows,
        Dimension num_cols,
        std::vector<Index> basic_vars);

    // --- Hot-path queries (Zero dynamic heap allocations, O(1)) ---

    /// Checks if variable j is basic. Returns false for out-of-range indices.
    [[nodiscard]] bool is_basic(Index j) const noexcept {
        if (j >= num_cols_) return false;
        return var_to_row_[j] != kInvalidIndex;
    }

    /// Checks if variable j is nonbasic. Returns false for out-of-range indices.
    [[nodiscard]] bool is_nonbasic(Index j) const noexcept {
        if (j >= num_cols_) return false;
        return var_to_row_[j] == kInvalidIndex;
    }

    /// Returns the row index i assigned to basic variable j (where B(i) == j).
    [[nodiscard]] Result<Index> row_of_basic(Index j) const noexcept {
        if (j >= num_cols_) {
            return Status::error(StatusCode::InvalidArgument, "Variable index out of range");
        }
        Index row = var_to_row_[j];
        if (row == kInvalidIndex) {
            return Status::error(StatusCode::InvalidArgument, "Variable is nonbasic");
        }
        return row;
    }

    /// Returns the basic variable index B(row) assigned to the given row.
    [[nodiscard]] Result<Index> basic_variable(Index row) const noexcept {
        if (row >= num_rows_) {
            return Status::error(StatusCode::InvalidArgument, "Row index out of range");
        }
        return basic_vars_[row];
    }

    /// Access the full list of basic column indices (size m).
    [[nodiscard]] const std::vector<Index>& basic_variables() const noexcept {
        return basic_vars_;
    }

    /// Access the full list of nonbasic column indices (size n - m).
    [[nodiscard]] const std::vector<Index>& nonbasic_variables() const noexcept {
        return nonbasic_vars_;
    }

    /// Number of rows m.
    [[nodiscard]] Dimension num_rows() const noexcept { return num_rows_; }

    /// Number of columns n.
    [[nodiscard]] Dimension num_cols() const noexcept { return num_cols_; }

    /// Deterministic modification version (incremented on each successful replacement).
    [[nodiscard]] uint64_t version() const noexcept { return version_; }

    /// Returns true if basis is structurally valid.
    [[nodiscard]] bool is_structurally_valid() const noexcept { return is_valid_; }

    // --- Transactional Basis Replacement ---

    /**
     * @brief Replaces basic variable B[leaving_row] with entering_col:
     *
     *     B[leaving_row] <- entering_col
     *
     * Validates:
     * - leaving_row < m
     * - entering_col < n
     * - entering_col is currently nonbasic
     * - current variable at leaving_row is basic
     *
     * Transactional guarantee:
     * If validation fails, the Basis object remains completely unmodified.
     * On success, both mappings (row->col, col->row, nonbasic list) are updated transactionally.
     */
    [[nodiscard]] Status replace_basic_variable(Index entering_col, Index leaving_row);

private:
    Dimension num_rows_{0};
    Dimension num_cols_{0};

    std::vector<Index> basic_vars_;            ///< size m: row -> basic column
    std::vector<Index> var_to_row_;            ///< size n: column -> row (or kInvalidIndex)
    std::vector<Index> nonbasic_vars_;         ///< size n - m: list of nonbasic columns
    std::vector<Index> var_to_nonbasic_pos_;   ///< size n: column -> index in nonbasic_vars_

    uint64_t version_{0};
    bool is_valid_{false};
};

} // namespace sih26119
