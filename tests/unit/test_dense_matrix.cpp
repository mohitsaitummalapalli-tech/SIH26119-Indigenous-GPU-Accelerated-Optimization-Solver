#include "numerics/dense_matrix.hpp"
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

void test_empty_and_single_element() {
    std::cout << "[TEST] 0x0 and 1x1 DenseMatrix\n";
    auto m0 = sih26119::DenseMatrix::create(0, 0);
    CHECK(m0.ok(), "Create 0x0 matrix");
    CHECK(m0.value().rows() == 0 && m0.value().cols() == 0, "Dimensions 0x0");

    // Out of bounds checked access on empty matrix
    CHECK(!m0.value().at(0, 0).ok(), "m0.at(0,0) must fail");
    CHECK(!m0.value().set(0, 0, 1.0).is_ok(), "m0.set(0,0) must fail");

    auto m1 = sih26119::DenseMatrix::create(1, 1, 7.5);
    CHECK(m1.ok(), "Create 1x1 matrix");
    CHECK(m1.value().rows() == 1 && m1.value().cols() == 1, "Dimensions 1x1");
    CHECK(m1.value()(0, 0) == 7.5, "m1(0,0) == 7.5");
    CHECK(m1.value().at(0, 0).value() == 7.5, "m1.at(0,0) == 7.5");
    CHECK(!m1.value().at(1, 0).ok(), "m1.at(1,0) out of bounds");
    CHECK(!m1.value().at(0, 1).ok(), "m1.at(0,1) out of bounds");
}

void test_column_major_layout_verification() {
    std::cout << "[TEST] Direct Column-Major Storage Verification\n";
    // 3 rows, 4 columns (distinct rows and cols to distinguish row-major from column-major)
    const sih26119::Dimension rows = 3;
    const sih26119::Dimension cols = 4;
    auto mat = sih26119::DenseMatrix::create(rows, cols, 0.0).value();

    // Populate each entry with a distinct value: A(i, j) = 10 * i + j
    for (sih26119::Dimension i = 0; i < rows; ++i) {
        for (sih26119::Dimension j = 0; j < cols; ++j) {
            auto s = mat.set(i, j, static_cast<sih26119::Scalar>(10 * i + j));
            CHECK(s.is_ok(), "Set element in matrix");
        }
    }

    const sih26119::Scalar* raw_data = mat.data();

    // Verify each element in raw contiguous buffer strictly matches linear_index(i, j) = i + j * rows
    for (sih26119::Dimension i = 0; i < rows; ++i) {
        for (sih26119::Dimension j = 0; j < cols; ++j) {
            const std::size_t col_major_offset = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * rows;
            const sih26119::Scalar expected = static_cast<sih26119::Scalar>(10 * i + j);
            CHECK(raw_data[col_major_offset] == expected,
                  "Raw contiguous memory at i + j * rows must match column-major indexing");
        }
    }

    // Explicitly verify this differs from row-major layout i * cols + j for asymmetric element
    // For (i=1, j=0):
    // column-major offset = 1 + 0*3 = 1 -> value is 10
    // row-major offset = 1*4 + 0 = 4 -> raw_data[4] is A(1, 1) = 11 in column-major!
    const std::size_t row_major_offset_1_0 = 1 * cols + 0; // 4
    CHECK(raw_data[row_major_offset_1_0] != static_cast<sih26119::Scalar>(10 * 1 + 0),
          "Memory layout must NOT follow row-major indexing");
}

void test_matrix_vector_multiplication() {
    std::cout << "[TEST] Dense Matrix-Vector Multiplication\n";
    // 2 rows, 3 cols:
    // A = [ 1.0  2.0 -1.0 ]
    //     [ 3.0  0.0  4.0 ]
    auto mat = sih26119::DenseMatrix::create(2, 3, 0.0).value();
    CHECK(mat.set(0, 0, 1.0).is_ok(), "set(0, 0)");
    CHECK(mat.set(0, 1, 2.0).is_ok(), "set(0, 1)");
    CHECK(mat.set(0, 2, -1.0).is_ok(), "set(0, 2)");
    CHECK(mat.set(1, 0, 3.0).is_ok(), "set(1, 0)");
    CHECK(mat.set(1, 1, 0.0).is_ok(), "set(1, 1)");
    CHECK(mat.set(1, 2, 4.0).is_ok(), "set(1, 2)");

    // x = [ 2.0, -1.0, 3.0 ]^T
    const std::vector<sih26119::Scalar> x_vals = {2.0, -1.0, 3.0};
    auto x = sih26119::DenseVector::from_values(x_vals).value();

    // Expected y:
    // y_0 = 1.0*2.0 + 2.0*(-1.0) + (-1.0)*3.0 = 2 - 2 - 3 = -3.0
    // y_1 = 3.0*2.0 + 0.0*(-1.0) + 4.0*3.0    = 6 + 0 + 12 = 18.0
    auto y = sih26119::DenseVector::create(2, 0.0).value();
    auto status = mat.multiply(x, y);
    CHECK(status.is_ok(), "Matrix multiply in-place should succeed");
    CHECK(sih26119::approx_equal(y[0], -3.0), "y[0] == -3.0");
    CHECK(sih26119::approx_equal(y[1], 18.0), "y[1] == 18.0");

    // Value-returning multiply overload
    auto y2_res = mat.multiply(x);
    CHECK(y2_res.ok(), "Matrix multiply value-returning should succeed");
    CHECK(sih26119::approx_equal(y2_res.value()[0], -3.0), "y2[0] == -3.0");
    CHECK(sih26119::approx_equal(y2_res.value()[1], 18.0), "y2[1] == 18.0");
}

void test_dimension_mismatch_and_aliasing() {
    std::cout << "[TEST] Dimension Mismatch and Aliasing Contract\n";
    auto mat = sih26119::DenseMatrix::create(2, 3, 1.0).value();

    auto x_bad = sih26119::DenseVector::create(4, 1.0).value();
    auto y = sih26119::DenseVector::create(2, 0.0).value();

    // x dimension mismatch
    CHECK(!mat.multiply(x_bad, y).is_ok(), "multiply with wrong x.size() must fail");

    // y dimension mismatch
    auto x_good = sih26119::DenseVector::create(3, 1.0).value();
    auto y_bad = sih26119::DenseVector::create(3, 0.0).value();
    CHECK(!mat.multiply(x_good, y_bad).is_ok(), "multiply with wrong y.size() must fail");

    // Square matrix aliasing test: y = Ax where &x == &y
    auto square_mat = sih26119::DenseMatrix::create(3, 3, 1.0).value();
    auto v = sih26119::DenseVector::create(3, 1.0).value();
    CHECK(!square_mat.multiply(v, v).is_ok(), "Aliased multiply(&v, v) must be rejected");
}

void test_non_finite_rejection() {
    std::cout << "[TEST] Non-Finite Input Rejection in DenseMatrix\n";
    const sih26119::Scalar nan_val = std::numeric_limits<sih26119::Scalar>::quiet_NaN();
    const sih26119::Scalar inf_val = std::numeric_limits<sih26119::Scalar>::infinity();

    CHECK(!sih26119::DenseMatrix::create(2, 2, nan_val).ok(), "create with NaN must fail");
    CHECK(!sih26119::DenseMatrix::create(2, 2, inf_val).ok(), "create with Inf must fail");

    auto mat = sih26119::DenseMatrix::create(2, 2, 0.0).value();
    CHECK(!mat.set(0, 0, nan_val).is_ok(), "set with NaN must fail");
    CHECK(!mat.fill(inf_val).is_ok(), "fill with Inf must fail");
}

void test_transactional_multiply_overflow() {
    std::cout << "[TEST] DenseMatrix Transactional Multiply on Overflow\n";
    // 2 rows, 1 col
    // A[0, 0] = 1.0
    // A[1, 0] = 1e308
    auto mat = sih26119::DenseMatrix::create(2, 1, 0.0).value();
    CHECK(mat.set(0, 0, 1.0).is_ok(), "set(0, 0)");
    CHECK(mat.set(1, 0, 1e308).is_ok(), "set(1, 0)");

    // x = [1e308]
    // Row 0: 1.0 * 1e308 = 1e308 (finite)
    // Row 1: 1e308 * 1e308 = +Inf (overflow)
    auto x = sih26119::DenseVector::create(1, 1e308).value();

    // Destination vector initialized to [77.0, 88.0]
    auto y = sih26119::DenseVector::create(2, 0.0).value();
    CHECK(y.set(0, 77.0).is_ok(), "y.set(0)");
    CHECK(y.set(1, 88.0).is_ok(), "y.set(1)");

    auto status = mat.multiply(x, y);
    CHECK(!status.is_ok(), "multiply must fail with error on overflow in row 1");

    // Transactional verification: y must remain byte-for-byte / value-for-value unchanged!
    CHECK(y[0] == 77.0, "y[0] must remain 77.0 (unmodified) despite row 0 being finite");
    CHECK(y[1] == 88.0, "y[1] must remain 88.0 (unmodified)");
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "SIH26119 DenseMatrix Unit Tests\n";
    std::cout << "========================================\n";

    test_empty_and_single_element();
    test_column_major_layout_verification();
    test_matrix_vector_multiplication();
    test_dimension_mismatch_and_aliasing();
    test_non_finite_rejection();
    test_transactional_multiply_overflow();

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] FAILED with " << g_failures << " failure(s).\n";
        return 1;
    }

    std::cout << "\n[RESULT] All DenseMatrix tests PASSED.\n";
    return 0;
}
