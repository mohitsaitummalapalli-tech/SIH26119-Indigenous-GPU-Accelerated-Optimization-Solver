#include "numerics/dense_vector.hpp"
#include "numerics/norms.hpp"
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

void test_empty_vector() {
    std::cout << "[TEST] Empty DenseVector\n";
    auto empty_res = sih26119::DenseVector::create(0);
    CHECK(empty_res.ok(), "Empty vector creation should succeed");
    auto& vec = empty_res.value();
    CHECK(vec.size() == 0, "Empty vector size must be 0");
    CHECK(vec.empty(), "Empty vector empty() must be true");

    auto norm2_res = vec.norm2();
    CHECK(norm2_res.ok() && norm2_res.value() == sih26119::kScalarZero, "Empty vector norm2 must be 0.0");
    auto norm_inf_res = vec.norm_inf();
    CHECK(norm_inf_res.ok() && norm_inf_res.value() == sih26119::kScalarZero, "Empty vector norm_inf must be 0.0");

    // Checked access out of bounds
    CHECK(!vec.at(0).ok(), "Empty vector at(0) must fail with error status");
    CHECK(!vec.set(0, 1.0).is_ok(), "Empty vector set(0) must fail with error status");

    // Dot product with another empty vector
    auto other_empty = sih26119::DenseVector::create(0).value();
    auto dot_res = vec.dot(other_empty);
    CHECK(dot_res.ok() && dot_res.value() == sih26119::kScalarZero, "Empty vector dot empty vector must be 0.0");
}

void test_size_one_and_basic_operations() {
    std::cout << "[TEST] Size 1 and Basic Operations\n";
    auto v_res = sih26119::DenseVector::create(1, 4.0);
    CHECK(v_res.ok(), "Size 1 vector creation");
    auto& v = v_res.value();
    CHECK(v.size() == 1, "Size must be 1");
    CHECK(!v.empty(), "Vector must not be empty");

    // Checked access
    auto at_0 = v.at(0);
    CHECK(at_0.ok() && at_0.value() == 4.0, "v.at(0) == 4.0");
    CHECK(!v.at(1).ok(), "v.at(1) must fail");

    // Unchecked access (valid index only)
    CHECK(v[0] == 4.0, "v[0] == 4.0");
    v[0] = 5.0;
    CHECK(v[0] == 5.0, "v[0] updated to 5.0");

    // Checked set
    CHECK(v.set(0, -3.0).is_ok(), "v.set(0, -3.0)");
    CHECK(v[0] == -3.0, "v[0] == -3.0");
    CHECK(!v.set(1, 2.0).is_ok(), "v.set(1, ...) must fail");

    // Norms
    CHECK(sih26119::approx_equal(v.norm2().value(), 3.0), "Norm2 of [-3] is 3.0");
    CHECK(sih26119::approx_equal(v.norm_inf().value(), 3.0), "Norm_inf of [-3] is 3.0");

    // Scale
    CHECK(v.scale(-2.0).is_ok(), "Scale by -2.0");
    CHECK(v[0] == 6.0, "v[0] after scale is 6.0");

    // Fill
    CHECK(v.fill(42.0).is_ok(), "Fill with 42.0");
    CHECK(v[0] == 42.0, "v[0] after fill is 42.0");
}

void test_from_values_and_arithmetic() {
    std::cout << "[TEST] from_values, AXPY, Dot, Norms\n";
    const std::vector<sih26119::Scalar> init_x = {1.0, -2.0, 3.0, -4.0};
    auto x_res = sih26119::DenseVector::from_values(init_x);
    CHECK(x_res.ok(), "from_values creation");
    auto x = x_res.value();
    CHECK(x.size() == 4, "Dimension must be 4");

    const std::vector<sih26119::Scalar> init_y = {10.0, 20.0, 30.0, 40.0};
    auto y_res = sih26119::DenseVector::from_values(init_y);
    CHECK(y_res.ok(), "from_values y creation");
    auto y = y_res.value();

    // Dot product: 1*10 + (-2)*20 + 3*30 + (-4)*40 = 10 - 40 + 90 - 160 = -100
    auto dot_res = x.dot(y);
    CHECK(dot_res.ok(), "Dot product should succeed");
    CHECK(sih26119::approx_equal(dot_res.value(), -100.0), "Dot product value must be -100.0");

    // Symmetry: dot(x, y) == dot(y, x)
    auto dot_yx = y.dot(x);
    CHECK(dot_yx.ok(), "Dot product y.dot(x) should succeed");
    CHECK(sih26119::approx_equal(dot_res.value(), dot_yx.value()), "dot(x, y) == dot(y, x)");

    // Norms of x: 1^2 + 4 + 9 + 16 = 30 => sqrt(30)
    auto n2 = x.norm2();
    CHECK(n2.ok(), "norm2 should succeed");
    CHECK(sih26119::approx_equal(n2.value(), std::sqrt(30.0)), "norm2(x) == sqrt(30)");
    auto n_inf = x.norm_inf();
    CHECK(n_inf.ok(), "norm_inf should succeed");
    CHECK(sih26119::approx_equal(n_inf.value(), 4.0), "norm_inf(x) == 4.0");

    // AXPY: y <- 0.5 * x + y
    // y_0 = 0.5*1 + 10 = 10.5
    // y_1 = 0.5*(-2) + 20 = 19.0
    // y_2 = 0.5*3 + 30 = 31.5
    // y_3 = 0.5*(-4) + 40 = 38.0
    auto axpy_status = y.axpy(0.5, x);
    CHECK(axpy_status.is_ok(), "AXPY execution");
    CHECK(sih26119::approx_equal(y[0], 10.5), "y[0] == 10.5");
    CHECK(sih26119::approx_equal(y[1], 19.0), "y[1] == 19.0");
    CHECK(sih26119::approx_equal(y[2], 31.5), "y[2] == 31.5");
    CHECK(sih26119::approx_equal(y[3], 38.0), "y[3] == 38.0");
}

void test_dimension_mismatch_rejection() {
    std::cout << "[TEST] Dimension Mismatch Handling\n";
    auto v3 = sih26119::DenseVector::create(3, 1.0).value();
    auto v4 = sih26119::DenseVector::create(4, 1.0).value();

    // AXPY mismatch
    auto axpy_status = v3.axpy(2.0, v4);
    CHECK(!axpy_status.is_ok(), "AXPY with mismatched dimensions must fail");

    // Dot mismatch
    auto dot_res = v3.dot(v4);
    CHECK(!dot_res.ok(), "Dot product with mismatched dimensions must fail");
}

void test_non_finite_rejection() {
    std::cout << "[TEST] Non-Finite (NaN / Inf) Rejection\n";
    const sih26119::Scalar nan_val = std::numeric_limits<sih26119::Scalar>::quiet_NaN();
    const sih26119::Scalar inf_val = std::numeric_limits<sih26119::Scalar>::infinity();

    // create with non-finite
    CHECK(!sih26119::DenseVector::create(3, nan_val).ok(), "create(3, NaN) must fail");
    CHECK(!sih26119::DenseVector::create(3, inf_val).ok(), "create(3, Inf) must fail");

    // from_values with non-finite
    std::vector<sih26119::Scalar> bad_vals = {1.0, nan_val, 3.0};
    CHECK(!sih26119::DenseVector::from_values(bad_vals).ok(), "from_values with NaN must fail");

    auto vec = sih26119::DenseVector::create(3, 1.0).value();

    // set non-finite
    CHECK(!vec.set(0, nan_val).is_ok(), "set with NaN must fail");
    CHECK(!vec.set(1, -inf_val).is_ok(), "set with -Inf must fail");

    // fill non-finite
    CHECK(!vec.fill(nan_val).is_ok(), "fill with NaN must fail");

    // scale non-finite
    CHECK(!vec.scale(nan_val).is_ok(), "scale with NaN must fail");

    // axpy non-finite alpha
    auto other = sih26119::DenseVector::create(3, 2.0).value();
    CHECK(!vec.axpy(nan_val, other).is_ok(), "axpy with NaN alpha must fail");
}

void test_large_value_norm_stability() {
    std::cout << "[TEST] Large Value Norm Stability (Scaled Hypot)\n";
    // Values that would overflow a naive sum of squares (1e200^2 = 1e400 -> overflow to Inf)
    const sih26119::Scalar large_val = 1e200;
    std::vector<sih26119::Scalar> large_vec_data = {3.0 * large_val, 4.0 * large_val};
    auto v = sih26119::DenseVector::from_values(large_vec_data).value();

    auto n2 = v.norm2();
    CHECK(n2.ok(), "norm2 on large vector must not return error");
    CHECK(sih26119::is_finite_scalar(n2.value()), "norm2 on large vector must remain finite");
    CHECK(sih26119::approx_equal(n2.value(), 5.0 * large_val), "norm2([3e200, 4e200]) == 5e200");
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "SIH26119 DenseVector Unit Tests\n";
    std::cout << "========================================\n";

    test_empty_vector();
    test_size_one_and_basic_operations();
    test_from_values_and_arithmetic();
    test_dimension_mismatch_rejection();
    test_non_finite_rejection();
    test_large_value_norm_stability();

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] FAILED with " << g_failures << " failure(s).\n";
        return 1;
    }

    std::cout << "\n[RESULT] All DenseVector tests PASSED.\n";
    return 0;
}
