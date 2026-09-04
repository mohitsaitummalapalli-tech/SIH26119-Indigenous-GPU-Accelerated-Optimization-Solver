#include "numerics/sparse_matrix.hpp"
#include "numerics/dense_vector.hpp"
#include "numerics/tolerances.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace {

int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            ++g_failures; \
        } \
    } while (false)

// Independent Non-Circular Dense Reference Oracle:
// Computes y = Ax directly using a 2D dense nested array/loop, completely independent of CSR.
std::vector<sih26119::Scalar> dense_spmv_oracle(
    sih26119::Dimension rows,
    sih26119::Dimension cols,
    const std::vector<std::vector<sih26119::Scalar>>& dense_A,
    const sih26119::DenseVector& x) {
    
    std::vector<sih26119::Scalar> y_ref(rows, 0.0);
    for (sih26119::Dimension i = 0; i < rows; ++i) {
        sih26119::Scalar sum = 0.0;
        for (sih26119::Dimension j = 0; j < cols; ++j) {
            sum += dense_A[i][j] * x[j];
        }
        y_ref[i] = sum;
    }
    return y_ref;
}

void test_empty_and_zero_matrix() {
    std::cout << "[TEST] Empty and Zero CSR Matrices\n";
    // 0x0
    auto m0_res = sih26119::SparseMatrix::from_triplets(0, 0, {});
    CHECK(m0_res.ok(), "Empty 0x0 sparse matrix creation");
    CHECK(m0_res.value().rows() == 0 && m0_res.value().cols() == 0, "0x0 dimensions");
    CHECK(m0_res.value().nnz() == 0, "0x0 nnz must be 0");
    CHECK(m0_res.value().validate_invariants().is_ok(), "0x0 invariants valid");

    // 3x4 completely zero matrix (no triplets)
    auto m_zero_res = sih26119::SparseMatrix::from_triplets(3, 4, {});
    CHECK(m_zero_res.ok(), "3x4 zero sparse matrix creation");
    auto& m_zero = m_zero_res.value();
    CHECK(m_zero.rows() == 3 && m_zero.cols() == 4, "3x4 dimensions");
    CHECK(m_zero.nnz() == 0, "3x4 nnz must be 0");
    CHECK(m_zero.validate_invariants().is_ok(), "3x4 zero matrix invariants valid");

    // get() on absent-but-valid coordinate must return exactly 0.0
    auto get_0_0 = m_zero.get(0, 0);
    CHECK(get_0_0.ok() && get_0_0.value() == sih26119::kScalarZero, "get(0,0) on zero matrix == 0.0");
    auto get_2_3 = m_zero.get(2, 3);
    CHECK(get_2_3.ok() && get_2_3.value() == sih26119::kScalarZero, "get(2,3) on zero matrix == 0.0");

    // get() on invalid coordinates must return error status
    CHECK(!m_zero.get(3, 0).ok(), "get(3,0) out of bounds must fail");
    CHECK(!m_zero.get(0, 4).ok(), "get(0,4) out of bounds must fail");

    // Multiply zero matrix by vector [1, 2, 3, 4]^T => [0, 0, 0]^T
    const std::vector<sih26119::Scalar> x_vals = {1.0, 2.0, 3.0, 4.0};
    auto x = sih26119::DenseVector::from_values(x_vals).value();
    auto y = m_zero.multiply(x).value();
    CHECK(y.size() == 3, "y size is 3");
    CHECK(y[0] == 0.0 && y[1] == 0.0 && y[2] == 0.0, "Zero matrix * x == 0 vector");
}

void test_triplet_accumulation_and_zero_elimination() {
    std::cout << "[TEST] Triplet Duplicate Accumulation & Exact Zero Elimination\n";
    // 3 rows, 3 cols
    std::vector<sih26119::Triplet> triplets = {
        {0, 1, 1.5},
        {0, 1, 2.5},        // duplicate coordinate (0, 1) accumulated: 1.5 + 2.5 = 4.0
        {1, 0, 3.0},
        {1, 0, -3.0},       // duplicate coordinate (1, 0) cancelling out to exact 0.0 -> must be eliminated!
        {1, 2, 1e-15},      // small non-zero -> MUST be preserved (structural zero != numerical zero)
        {2, 2, 5.0}
    };

    auto mat_res = sih26119::SparseMatrix::from_triplets(3, 3, triplets);
    CHECK(mat_res.ok(), "Sparse matrix from triplets with duplicates and zeros");
    auto& mat = mat_res.value();
    CHECK(mat.validate_invariants().is_ok(), "CSR invariants must hold");

    // Exactly 3 nonzeros should remain: (0, 1)=4.0, (1, 2)=1e-15, (2, 2)=5.0
    CHECK(mat.nnz() == 3, "nnz must be exactly 3 after duplicate accumulation and zero elimination");

    // Check values
    CHECK(mat.get(0, 1).value() == 4.0, "Accumulated (0, 1) == 4.0");
    CHECK(mat.get(1, 0).value() == 0.0, "Cancelled (1, 0) == 0.0 (eliminated from CSR)");
    CHECK(mat.get(1, 2).value() == 1e-15, "Small non-zero (1, 2) is preserved");
    CHECK(mat.get(2, 2).value() == 5.0, "Entry (2, 2) == 5.0");

    // Check empty row or absent entries return 0.0
    CHECK(mat.get(0, 0).value() == 0.0, "Absent (0, 0) == 0.0");
    CHECK(mat.get(2, 0).value() == 0.0, "Absent (2, 0) == 0.0");
}

void test_unsorted_triplets_and_column_ordering() {
    std::cout << "[TEST] Unsorted Triplets and Strictly Sorted Column Ordering\n";
    // Provide triplets in intentionally disordered row and column order
    std::vector<sih26119::Triplet> triplets = {
        {2, 1, 10.0},
        {0, 2, 20.0},
        {1, 3, 30.0},
        {0, 0, 40.0},
        {1, 1, 50.0},
        {0, 1, 60.0}
    };

    auto mat = sih26119::SparseMatrix::from_triplets(3, 4, triplets).value();
    CHECK(mat.validate_invariants().is_ok(), "Invariants after sorting");
    CHECK(mat.nnz() == 6, "All 6 nonzeros present");

    // Row 0 has cols: 0, 1, 2 in strictly increasing order
    CHECK(mat.row_ptr()[0] == 0 && mat.row_ptr()[1] == 3, "Row 0 has 3 nonzeros");
    CHECK(mat.col_idx()[0] == 0 && mat.values()[0] == 40.0, "Row 0 entry 0");
    CHECK(mat.col_idx()[1] == 1 && mat.values()[1] == 60.0, "Row 0 entry 1");
    CHECK(mat.col_idx()[2] == 2 && mat.values()[2] == 20.0, "Row 0 entry 2");

    // Row 1 has cols: 1, 3
    CHECK(mat.row_ptr()[2] == 5, "Row 1 end offset is 5");
    CHECK(mat.col_idx()[3] == 1 && mat.values()[3] == 50.0, "Row 1 entry 0");
    CHECK(mat.col_idx()[4] == 3 && mat.values()[4] == 30.0, "Row 1 entry 1");

    // Row 2 has col: 1
    CHECK(mat.row_ptr()[3] == 6, "Row 2 end offset is 6");
    CHECK(mat.col_idx()[5] == 1 && mat.values()[5] == 10.0, "Row 2 entry 0");
}

void test_spmv_against_independent_dense_oracle() {
    std::cout << "[TEST] SpMV against Independent Dense Reference Oracle\n";
    const sih26119::Dimension rows = 4;
    const sih26119::Dimension cols = 5;

    // Build dense 2D reference matrix
    std::vector<std::vector<sih26119::Scalar>> dense_A(rows, std::vector<sih26119::Scalar>(cols, 0.0));
    std::vector<sih26119::Triplet> triplets;

    // Specific nonzeros (including negative and mixed magnitude entries)
    dense_A[0][1] = 2.5;    triplets.push_back({0, 1, 2.5});
    dense_A[0][4] = -1.2;   triplets.push_back({0, 4, -1.2});
    dense_A[1][0] = -3.0;   triplets.push_back({1, 0, -3.0});
    dense_A[1][2] = 4.0;    triplets.push_back({1, 2, 4.0});
    // row 2 is completely empty in dense_A
    dense_A[3][1] = 0.5;    triplets.push_back({3, 1, 0.5});
    dense_A[3][3] = -8.0;   triplets.push_back({3, 3, -8.0});
    dense_A[3][4] = 10.0;   triplets.push_back({3, 4, 10.0});

    auto mat = sih26119::SparseMatrix::from_triplets(rows, cols, triplets).value();

    // Test vector x
    std::vector<sih26119::Scalar> x_data = {1.5, -2.0, 3.5, 0.25, -4.0};
    auto x = sih26119::DenseVector::from_values(x_data).value();

    // Compute via independent dense oracle
    auto y_oracle = dense_spmv_oracle(rows, cols, dense_A, x);

    // Compute via CSR SpMV
    auto y_csr = mat.multiply(x).value();

    for (sih26119::Dimension i = 0; i < rows; ++i) {
        CHECK(sih26119::approx_equal(y_csr[i], y_oracle[i]),
              "CSR SpMV must match independent dense oracle result");
    }

    // Residual test: r = b - Ax
    std::vector<sih26119::Scalar> b_data = {10.0, 20.0, 30.0, 40.0};
    auto b = sih26119::DenseVector::from_values(b_data).value();
    auto r = mat.residual(b, x).value();

    for (sih26119::Dimension i = 0; i < rows; ++i) {
        sih26119::Scalar expected_r_i = b[i] - y_oracle[i];
        CHECK(sih26119::approx_equal(r[i], expected_r_i),
              "Sparse residual must equal b_i - (Ax)_i");
    }
}

void test_invalid_input_and_aliasing() {
    std::cout << "[TEST] Invalid Input and Aliasing Rejection in SparseMatrix\n";
    const sih26119::Scalar nan_val = std::numeric_limits<sih26119::Scalar>::quiet_NaN();

    // Triplet with out of bounds coordinates
    std::vector<sih26119::Triplet> bad_coords = {{3, 1, 1.0}};
    CHECK(!sih26119::SparseMatrix::from_triplets(3, 2, bad_coords).ok(), "Triplet row >= rows must fail");

    std::vector<sih26119::Triplet> bad_cols = {{1, 4, 1.0}};
    CHECK(!sih26119::SparseMatrix::from_triplets(3, 4, bad_cols).ok(), "Triplet col >= cols must fail");

    // Triplet with NaN
    std::vector<sih26119::Triplet> bad_val = {{1, 1, nan_val}};
    CHECK(!sih26119::SparseMatrix::from_triplets(3, 3, bad_val).ok(), "Triplet with NaN must fail");

    // Aliasing check in multiply
    const std::vector<sih26119::Triplet> unit_trip = {{0, 0, 1.0}};
    auto mat = sih26119::SparseMatrix::from_triplets(3, 3, unit_trip).value();
    auto v = sih26119::DenseVector::create(3, 1.0).value();
    CHECK(!mat.multiply(v, v).is_ok(), "Aliased multiply(v, v) must fail");

    // Aliasing check in residual: r cannot alias x
    auto b = sih26119::DenseVector::create(3, 2.0).value();
    CHECK(!mat.residual(b, v, v).is_ok(), "Aliased residual(b, v, v) must fail");
}

void test_triplet_accumulation_overflow_rejection() {
    std::cout << "[TEST] Triplet Duplicate Accumulation Overflow Rejection\n";
    // Duplicate triplets at (0, 0) with values 1e308 + 1e308 -> overflow to +Inf
    const std::vector<sih26119::Triplet> overflow_triplets = {
        {0, 0, 1e308},
        {0, 0, 1e308}
    };
    auto res = sih26119::SparseMatrix::from_triplets(2, 2, overflow_triplets);
    CHECK(!res.ok(), "from_triplets with duplicate accumulation overflow must fail with error");
}

void test_transactional_sparse_multiply_and_residual_overflow() {
    std::cout << "[TEST] SparseMatrix Transactional Multiply & Residual on Overflow\n";
    // 2 rows, 1 col: (0, 0, 1.0) and (1, 0, 1e308)
    const std::vector<sih26119::Triplet> triplets = {
        {0, 0, 1.0},
        {1, 0, 1e308}
    };
    auto mat = sih26119::SparseMatrix::from_triplets(2, 1, triplets).value();
    auto x = sih26119::DenseVector::create(1, 1e308).value();

    // Multiply test:
    // Row 0: 1.0 * 1e308 = 1e308 (finite)
    // Row 1: 1e308 * 1e308 = +Inf (overflow)
    auto y = sih26119::DenseVector::create(2, 0.0).value();
    CHECK(y.set(0, 33.0).is_ok(), "y.set(0)");
    CHECK(y.set(1, 44.0).is_ok(), "y.set(1)");

    auto status_mul = mat.multiply(x, y);
    CHECK(!status_mul.is_ok(), "SparseMatrix::multiply must fail on row 1 overflow");
    CHECK(y[0] == 33.0, "y[0] must remain 33.0 (unmodified) despite row 0 being finite");
    CHECK(y[1] == 44.0, "y[1] must remain 44.0 (unmodified)");

    // Residual test: r = b - Ax
    auto b = sih26119::DenseVector::create(2, 0.0).value();
    auto r = sih26119::DenseVector::create(2, 0.0).value();
    CHECK(r.set(0, 55.0).is_ok(), "r.set(0)");
    CHECK(r.set(1, 66.0).is_ok(), "r.set(1)");

    auto status_res = mat.residual(b, x, r);
    CHECK(!status_res.is_ok(), "SparseMatrix::residual must fail on row 1 overflow");
    CHECK(r[0] == 55.0, "r[0] must remain 55.0 (unmodified) despite row 0 being finite");
    CHECK(r[1] == 66.0, "r[1] must remain 66.0 (unmodified)");
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "SIH26119 SparseMatrix Unit Tests\n";
    std::cout << "========================================\n";

    test_empty_and_zero_matrix();
    test_triplet_accumulation_and_zero_elimination();
    test_unsorted_triplets_and_column_ordering();
    test_spmv_against_independent_dense_oracle();
    test_invalid_input_and_aliasing();
    test_triplet_accumulation_overflow_rejection();
    test_transactional_sparse_multiply_and_residual_overflow();

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] FAILED with " << g_failures << " failure(s).\n";
        return 1;
    }

    std::cout << "\n[RESULT] All SparseMatrix tests PASSED.\n";
    return 0;
}
