#include "numerics/scalar.hpp"
#include "numerics/tolerances.hpp"
#include "numerics/norms.hpp"
#include "numerics/dense_vector.hpp"
#include "numerics/dense_matrix.hpp"
#include "numerics/sparse_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
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

void test_tolerances_and_approx_equal() {
    std::cout << "[TEST] Foundational Tolerances and approx_equal\n";
    sih26119::Tolerance tol; // default 1e-12, 1e-12

    // Exact equal
    CHECK(sih26119::approx_equal(1.0, 1.0, tol), "1.0 == 1.0");
    CHECK(sih26119::approx_equal(0.0, 0.0, tol), "0.0 == 0.0");
    CHECK(sih26119::approx_equal(-5.2, -5.2, tol), "-5.2 == -5.2");

    // Near boundary (1e-12 difference on order 1 magnitude)
    // formula: |a - b| <= abs_tol + rel_tol * max(1, |a|, |b|)
    // For a = 1.0, b = 1.0 + 1.5e-12:
    // max(1, 1.0, 1.0) = 1.0 -> threshold = 1e-12 + 1e-12*1.0 = 2e-12.
    // diff = 1.5e-12 <= 2e-12 -> approx_equal is true.
    CHECK(sih26119::approx_equal(1.0, 1.0 + 1.5e-12, tol), "1.5e-12 diff is within combined 2e-12 threshold");

    // diff = 3e-12 > 2e-12 -> approx_equal is false.
    CHECK(!sih26119::approx_equal(1.0, 1.0 + 3e-12, tol), "3e-12 diff exceeds combined threshold");

    // Large numbers: relative tolerance dominance
    const sih26119::Scalar large_a = 1e8;
    // max_mag = 1e8 -> rel_tol * 1e8 = 1e-4
    CHECK(sih26119::approx_equal(large_a, large_a + 5e-5, tol), "Relative tolerance handles 1e8 magnitude");
    CHECK(!sih26119::approx_equal(large_a, large_a + 5e-4, tol), "Exceeds relative tolerance on 1e8 magnitude");

    // approx_zero
    CHECK(sih26119::approx_zero(0.0), "0.0 is approx_zero");
    CHECK(sih26119::approx_zero(5e-13), "5e-13 is approx_zero with 1e-12 tol");
    CHECK(!sih26119::approx_zero(5e-11), "5e-11 is not approx_zero with 1e-12 tol");
}

void test_norm_mathematical_properties() {
    std::cout << "[TEST] Norm Mathematical Properties\n";
    std::vector<sih26119::Scalar> x_data = {3.0, -4.0, 0.0, 12.0};
    auto x = sih26119::DenseVector::from_values(x_data).value();

    std::vector<sih26119::Scalar> y_data = {-1.0, 2.0, -5.0, 4.0};
    auto y = sih26119::DenseVector::from_values(y_data).value();

    auto n2_x = x.norm2().value();
    auto n2_y = y.norm2().value();

    CHECK(n2_x >= 0.0, "||x||_2 >= 0");
    CHECK(x.norm_inf().value() >= 0.0, "||x||_inf >= 0");

    // Triangle inequality: ||x + y||_2 <= ||x||_2 + ||y||_2
    auto x_plus_y = x;
    CHECK(x_plus_y.axpy(1.0, y).is_ok(), "Compute x + y");
    auto n2_sum = x_plus_y.norm2().value();

    CHECK(n2_sum <= n2_x + n2_y + 1e-12, "Triangle inequality: ||x+y||_2 <= ||x||_2 + ||y||_2");
}

void test_deterministic_property_tests() {
    std::cout << "[TEST] Deterministic Property Tests (seed = 42)\n";
    // Fixed seed 42
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<sih26119::Scalar> val_dist(-10.0, 10.0);
    std::uniform_real_distribution<sih26119::Scalar> prob_dist(0.0, 1.0);

    struct TestConfig {
        sih26119::Dimension rows;
        sih26119::Dimension cols;
        sih26119::Scalar density;
    };

    const std::vector<TestConfig> configs = {
        {1, 1, 1.0},
        {5, 5, 0.4},
        {10, 10, 0.2},
        {20, 12, 0.15},
        {15, 25, 0.25},
        {30, 30, 0.05}
    };

    int cases_run = 0;

    for (const auto& cfg : configs) {
        for (int repeat = 0; repeat < 5; ++repeat) {
            ++cases_run;
            // 1. Build dense matrix and triplet list simultaneously
            auto dense_mat = sih26119::DenseMatrix::create(cfg.rows, cfg.cols, 0.0).value();
            std::vector<sih26119::Triplet> triplets;

            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                for (sih26119::Dimension j = 0; j < cfg.cols; ++j) {
                    if (prob_dist(rng) <= cfg.density) {
                        const sih26119::Scalar val = val_dist(rng);
                        CHECK(dense_mat.set(i, j, val).is_ok(), "dense_mat.set");
                        triplets.push_back({i, j, val});
                    }
                }
            }

            auto csr_mat = sih26119::SparseMatrix::from_triplets(cfg.rows, cfg.cols, triplets).value();
            CHECK(csr_mat.validate_invariants().is_ok(), "CSR invariants in randomized test");

            // Generate vectors x and y
            std::vector<sih26119::Scalar> x_vals(cfg.cols);
            std::vector<sih26119::Scalar> y_vals(cfg.cols);
            for (sih26119::Dimension j = 0; j < cfg.cols; ++j) {
                x_vals[j] = val_dist(rng);
                y_vals[j] = val_dist(rng);
            }
            auto x = sih26119::DenseVector::from_values(x_vals).value();
            auto y = sih26119::DenseVector::from_values(y_vals).value();

            // Property 1: CSR(A)x == dense(A)x
            auto y_csr = csr_mat.multiply(x).value();
            auto y_dense = dense_mat.multiply(x).value();
            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                sih26119::Tolerance loose_tol{1e-9, 1e-9}; // allow small accumulated FP drift on sums
                CHECK(sih26119::approx_equal(y_csr[i], y_dense[i], loose_tol),
                      "Property 1: CSR(A)x == dense(A)x");
            }

            // Property 2: A(x + y) == Ax + Ay
            auto x_plus_y = x;
            CHECK(x_plus_y.axpy(1.0, y).is_ok(), "x_plus_y.axpy");
            auto A_x_plus_y = csr_mat.multiply(x_plus_y).value();
            auto Ax = y_csr;
            auto Ay = csr_mat.multiply(y).value();
            auto Ax_plus_Ay = Ax;
            CHECK(Ax_plus_Ay.axpy(1.0, Ay).is_ok(), "Ax_plus_Ay.axpy");

            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                sih26119::Tolerance loose_tol{1e-9, 1e-9};
                CHECK(sih26119::approx_equal(A_x_plus_y[i], Ax_plus_Ay[i], loose_tol),
                      "Property 2: A(x + y) == Ax + Ay");
            }

            // Property 3: A(alpha * x) == alpha * (Ax)
            const sih26119::Scalar alpha = -2.5;
            auto alpha_x = x;
            CHECK(alpha_x.scale(alpha).is_ok(), "alpha_x.scale");
            auto A_alpha_x = csr_mat.multiply(alpha_x).value();
            auto alpha_Ax = Ax;
            CHECK(alpha_Ax.scale(alpha).is_ok(), "alpha_Ax.scale");

            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                sih26119::Tolerance loose_tol{1e-9, 1e-9};
                CHECK(sih26119::approx_equal(A_alpha_x[i], alpha_Ax[i], loose_tol),
                      "Property 3: A(alpha*x) == alpha*(Ax)");
            }

            // Property 4: dot(x, y) == dot(y, x)
            auto dot_xy = x.dot(y).value();
            auto dot_yx = y.dot(x).value();
            CHECK(sih26119::approx_equal(dot_xy, dot_yx), "Property 4: dot(x, y) == dot(y, x)");

            // Property 5: Residual correctness: r = b - Ax
            std::vector<sih26119::Scalar> b_vals(cfg.rows);
            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                b_vals[i] = val_dist(rng);
            }
            auto b = sih26119::DenseVector::from_values(b_vals).value();
            auto r = csr_mat.residual(b, x).value();

            for (sih26119::Dimension i = 0; i < cfg.rows; ++i) {
                sih26119::Scalar expected_ri = b[i] - Ax[i];
                CHECK(sih26119::approx_equal(r[i], expected_ri),
                      "Property 5: residual r_i == b_i - (Ax)_i");
            }
        }
    }

    std::cout << "  [INFO] Successfully completed " << cases_run << " randomized property verification cases.\n";
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "SIH26119 Numerics & Property Tests\n";
    std::cout << "========================================\n";

    test_tolerances_and_approx_equal();
    test_norm_mathematical_properties();
    test_deterministic_property_tests();

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] FAILED with " << g_failures << " failure(s).\n";
        return 1;
    }

    std::cout << "\n[RESULT] All Numerics property tests PASSED.\n";
    return 0;
}
