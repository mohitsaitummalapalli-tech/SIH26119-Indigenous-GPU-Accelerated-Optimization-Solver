#include "solver/lp/lp_standard_form.hpp"
#include "model/model.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

using namespace sih26119;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Phase 3A LP Standardization Layer\n";
    std::cout << "=========================================================\n";

    int failures = 0;

    auto check = [&](bool condition, const std::string& desc) {
        if (condition) {
            std::cout << "  [PASS] " << desc << "\n";
        } else {
            std::cerr << "  [FAIL] " << desc << "\n";
            ++failures;
        }
    };

    auto approx_eq = [](Scalar a, Scalar b, Scalar tol = 1e-9) -> bool {
        return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b)));
    };

    auto set_vec = [](DenseVector& vec, Index idx, Scalar val) {
        Status s = vec.set(idx, val);
        (void)s;
    };

    // TEST-STD-01: Simple x >= 0 model (identity mapping)
    {
        Model m("TestStd01");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto x2 = m.add_variable("x2", 0.0, kInfinity).value();
        (void)m.set_objective_coefficient(x1, 2.0);
        (void)m.set_objective_coefficient(x2, 3.0);
        (void)m.set_objective_offset(1.5);
        m.set_objective_sense(ObjectiveSense::Minimize);

        auto c1 = m.add_constraint("c1", 5.0, 5.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0);
        (void)m.add_constraint_term(c1, x2, 1.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-01: Standardization succeeds");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 2, "TEST-STD-01: 2 standard variables");
        check(slp.standardized_constraints() == 1, "TEST-STD-01: 1 standard constraint");
        check(approx_eq(slp.c0(), 1.5), "TEST-STD-01: Objective offset preserved");
        check(approx_eq(slp.c().at(0).value(), 2.0) && approx_eq(slp.c().at(1).value(), 3.0),
              "TEST-STD-01: Objective coefficients preserved");
        check(approx_eq(slp.b().at(0).value(), 5.0), "TEST-STD-01: RHS preserved");
    }

    // TEST-STD-02: Lower-bound shift (x >= 5)
    {
        Model m("TestStd02");
        auto x1 = m.add_variable("x1", 5.0, kInfinity).value();
        (void)m.set_objective_coefficient(x1, 4.0);
        (void)m.set_objective_offset(2.0);

        auto c1 = m.add_constraint("c1", 10.0, 10.0).value();
        (void)m.add_constraint_term(c1, x1, 2.0); // 2*(x1' + 5) = 10 => 2*x1' = 0

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-02: Lower-bound shift standardizes");
        const auto& slp = std_res.value();
        check(approx_eq(slp.c0(), 2.0 + 4.0 * 5.0), "TEST-STD-02: Objective offset c0 + c*L = 22");
        check(approx_eq(slp.b().at(0).value(), 10.0 - 2.0 * 5.0), "TEST-STD-02: Shifted RHS = 0");

        // Test reconstruction: x1' = 3 => x1 = 8
        auto x_bar = DenseVector::create(1).value();
        set_vec(x_bar, 0, 3.0);
        auto x_orig = slp.reconstruct_primal(x_bar).value();
        check(approx_eq(x_orig.at(0).value(), 8.0), "TEST-STD-02: Reconstruct shifted variable");
    }

    // TEST-STD-03: Upper-bound reflection (x <= 8)
    {
        Model m("TestStd03");
        auto x1 = m.add_variable("x1", -kInfinity, 8.0).value();
        (void)m.set_objective_coefficient(x1, 3.0);
        (void)m.set_objective_offset(0.0);

        auto c1 = m.add_constraint("c1", 6.0, 6.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0); // 1*(8 - x1') = 6 => -x1' = -2 => x1' = 2

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-03: Upper-bound reflection standardizes");
        const auto& slp = std_res.value();
        check(approx_eq(slp.c().at(0).value(), -3.0), "TEST-STD-03: Cost negated to -3.0");
        check(approx_eq(slp.c0(), 3.0 * 8.0), "TEST-STD-03: Offset shifted by +24.0");

        // Test reconstruction: x1' = 2 => x1 = 8 - 2 = 6
        auto x_bar = DenseVector::create(1).value();
        set_vec(x_bar, 0, 2.0);
        auto x_orig = slp.reconstruct_primal(x_bar).value();
        check(approx_eq(x_orig.at(0).value(), 6.0), "TEST-STD-03: Reconstruct reflected variable");
    }

    // TEST-STD-04: Finite two-sided variable bound (2 <= x <= 7)
    {
        Model m("TestStd04");
        auto x1 = m.add_variable("x1", 2.0, 7.0).value();
        (void)m.set_objective_coefficient(x1, 5.0);

        auto c1 = m.add_constraint("c1", 4.0, 4.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0); // 1*(x1' + 2) = 4 => x1' = 2

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-04: Box-bound variable standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 2, "TEST-STD-04: 2 standard variables (primary + slack)");
        check(slp.standardized_constraints() == 2, "TEST-STD-04: 2 constraints (structural + box slack row)");
        // Box row RHS = U - L = 7 - 2 = 5
        check(approx_eq(slp.b().at(1).value(), 5.0), "TEST-STD-04: Box slack row RHS is U - L = 5");

        // Reconstruction: x1' = 1.0 => x1 = 3.0
        auto x_bar = DenseVector::create(2).value();
        set_vec(x_bar, 0, 1.0);
        set_vec(x_bar, 1, 4.0);
        auto x_orig = slp.reconstruct_primal(x_bar).value();
        check(approx_eq(x_orig.at(0).value(), 3.0), "TEST-STD-04: Box reconstruction");
    }

    // TEST-STD-05: Free variable split (x in R)
    {
        Model m("TestStd05");
        auto x1 = m.add_variable("x1", -kInfinity, kInfinity).value();
        (void)m.set_objective_coefficient(x1, -3.5);

        auto c1 = m.add_constraint("c1", 10.0, 10.0).value();
        (void)m.add_constraint_term(c1, x1, 2.0); // 2*(x+ - x-) = 10

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-05: Free variable standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 2, "TEST-STD-05: 2 variables (x+, x-)");
        check(approx_eq(slp.c().at(0).value(), -3.5), "TEST-STD-05: c(x+) = -3.5");
        check(approx_eq(slp.c().at(1).value(), 3.5), "TEST-STD-05: c(x-) = +3.5");

        // Reconstruction: x+ = 7, x- = 2 => x = 5
        auto x_bar = DenseVector::create(2).value();
        set_vec(x_bar, 0, 7.0);
        set_vec(x_bar, 1, 2.0);
        auto x_orig = slp.reconstruct_primal(x_bar).value();
        check(approx_eq(x_orig.at(0).value(), 5.0), "TEST-STD-05: Free reconstruction");
    }

    // TEST-STD-06: Fixed variable elimination (x == 4)
    {
        Model m("TestStd06");
        auto x1 = m.add_variable("x1", 4.0, 4.0).value();
        auto x2 = m.add_variable("x2", 0.0, kInfinity).value();
        (void)m.set_objective_coefficient(x1, 10.0);
        (void)m.set_objective_coefficient(x2, 2.0);
        (void)m.set_objective_offset(5.0);

        auto c1 = m.add_constraint("c1", 20.0, 20.0).value();
        (void)m.add_constraint_term(c1, x1, 3.0); // 3*4 + 1*x2 = 20 => x2 = 8
        (void)m.add_constraint_term(c1, x2, 1.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-06: Fixed variable standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 1, "TEST-STD-06: 1 variable remaining (x2)");
        check(approx_eq(slp.c0(), 5.0 + 10.0 * 4.0), "TEST-STD-06: Offset shifted by 40 = 45");
        check(approx_eq(slp.b().at(0).value(), 20.0 - 3.0 * 4.0), "TEST-STD-06: RHS shifted to 8");

        // Reconstruction
        auto x_bar = DenseVector::create(1).value();
        set_vec(x_bar, 0, 8.0);
        auto x_orig = slp.reconstruct_primal(x_bar).value();
        check(approx_eq(x_orig.at(0).value(), 4.0) && approx_eq(x_orig.at(1).value(), 8.0),
              "TEST-STD-06: Fixed variable reconstructed correctly");
    }

    // TEST-STD-07: Less-than-or-equal row (a^T x <= u)
    {
        Model m("TestStd07");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", -kInfinity, 15.0).value();
        (void)m.add_constraint_term(c1, x1, 2.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-07: <= row standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 2, "TEST-STD-07: 2 variables (x1, slack)");
        check(slp.standardized_constraints() == 1, "TEST-STD-07: 1 equality row");
        check(approx_eq(slp.b().at(0).value(), 15.0), "TEST-STD-07: RHS = 15");
    }

    // TEST-STD-08: Greater-than-or-equal row (a^T x >= l)
    {
        Model m("TestStd08");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", 7.0, kInfinity).value();
        (void)m.add_constraint_term(c1, x1, 3.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-08: >= row standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 2, "TEST-STD-08: 2 variables (x1, surplus)");
        check(approx_eq(slp.b().at(0).value(), 7.0), "TEST-STD-08: RHS = 7");
    }

    // TEST-STD-09: Equality row (a^T x == b)
    {
        Model m("TestStd09");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", 12.0, 12.0).value();
        (void)m.add_constraint_term(c1, x1, 4.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-09: == row standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 1, "TEST-STD-09: 1 variable (0 auxiliaries)");
        check(approx_eq(slp.b().at(0).value(), 12.0), "TEST-STD-09: RHS = 12");
    }

    // TEST-STD-10: Finite range row (5 <= 2*x1 + 3*x2 <= 15)
    {
        Model m("TestStd10");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto x2 = m.add_variable("x2", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", 5.0, 15.0).value();
        (void)m.add_constraint_term(c1, x1, 2.0);
        (void)m.add_constraint_term(c1, x2, 3.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-10: Range row standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_variables() == 4, "TEST-STD-10: 4 variables (x1, x2, surplus, slack)");
        check(slp.standardized_constraints() == 2, "TEST-STD-10: 2 rows (lower eq, range length eq)");
        check(approx_eq(slp.b().at(0).value(), 5.0), "TEST-STD-10: Row 1 RHS = 5");
        check(approx_eq(slp.b().at(1).value(), 10.0), "TEST-STD-10: Row 2 RHS = 10 (u - l)");
    }

    // TEST-STD-11: Infinite lower range side (-inf <= x1 <= 10)
    {
        Model m("TestStd11");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", -kInfinity, 10.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-11: Infinite lower range side standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_constraints() == 1, "TEST-STD-11: 1 constraint row");
    }

    // TEST-STD-12: Infinite upper range side (3 <= x1 <= inf)
    {
        Model m("TestStd12");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", 3.0, kInfinity).value();
        (void)m.add_constraint_term(c1, x1, 1.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-12: Infinite upper range side standardizes");
        const auto& slp = std_res.value();
        check(slp.standardized_constraints() == 1, "TEST-STD-12: 1 constraint row");
    }

    // TEST-STD-13: Negative RHS normalization
    {
        Model m("TestStd13");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto x2 = m.add_variable("x2", 0.0, kInfinity).value();
        auto c1 = m.add_constraint("c1", -10.0, -10.0).value(); // 2*x1 - 4*x2 = -10 => -2*x1 + 4*x2 = 10
        (void)m.add_constraint_term(c1, x1, 2.0);
        (void)m.add_constraint_term(c1, x2, -4.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-13: Negative RHS standardizes");
        const auto& slp = std_res.value();
        check(approx_eq(slp.b().at(0).value(), 10.0), "TEST-STD-13: Normalized RHS is +10.0");
        check(approx_eq(slp.A().values()[0], -2.0) && approx_eq(slp.A().values()[1], 4.0),
              "TEST-STD-13: Coefficients flipped to -2.0 and +4.0");
        check(slp.constraint_mappings()[0].row_negated[0] == true, "TEST-STD-13: Recorded row_negated == true");
    }

    // TEST-STD-14: Mixed variables and mixed constraints
    {
        Model m("TestStd14");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        auto x2 = m.add_variable("x2", 1.0, 5.0).value();
        auto x3 = m.add_variable("x3", -kInfinity, kInfinity).value();
        auto x4 = m.add_variable("x4", 3.0, 3.0).value();

        (void)m.set_objective_coefficient(x1, 1.0);
        (void)m.set_objective_coefficient(x2, -2.0);
        (void)m.set_objective_coefficient(x3, 4.0);
        (void)m.set_objective_coefficient(x4, 0.5);

        auto c1 = m.add_constraint("c1", -kInfinity, 20.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0);
        (void)m.add_constraint_term(c1, x2, 2.0);
        (void)m.add_constraint_term(c1, x4, 1.0);

        auto c2 = m.add_constraint("c2", 2.0, 10.0).value();
        (void)m.add_constraint_term(c2, x2, 1.0);
        (void)m.add_constraint_term(c2, x3, 1.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-14: Mixed model standardizes successfully");
        const auto& slp = std_res.value();
        check(slp.original_variables() == 4, "TEST-STD-14: 4 original variables");
        check(slp.original_constraints() == 2, "TEST-STD-14: 2 original constraints");
    }

    // TEST-STD-15: Objective constant and Maximize sense preservation
    {
        Model m("TestStd15");
        auto x1 = m.add_variable("x1", 0.0, kInfinity).value();
        (void)m.set_objective_coefficient(x1, 5.0);
        (void)m.set_objective_offset(42.5);
        m.set_objective_sense(ObjectiveSense::Maximize);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-15: Maximize model standardizes");
        const auto& slp = std_res.value();
        check(approx_eq(slp.c0(), -42.5), "TEST-STD-15: c0_bar = -42.5 under Maximize");
        check(approx_eq(slp.c().at(0).value(), -5.0), "TEST-STD-15: c_bar = -5.0 under Maximize");

        auto x_orig = DenseVector::create(1).value();
        set_vec(x_orig, 0, 2.0);
        Scalar f_orig = slp.evaluate_original_objective(x_orig).value();
        check(approx_eq(f_orig, 5.0 * 2.0 + 42.5), "TEST-STD-15: Original objective evaluated = 52.5");

        auto x_bar = slp.project_primal(x_orig).value();
        Scalar z_std = slp.evaluate_standard_objective(x_bar).value();
        check(approx_eq(z_std, -52.5), "TEST-STD-15: Standard objective evaluated = -52.5");
        check(approx_eq(f_orig, -z_std), "TEST-STD-15: f_orig == -z_std duality holds");
    }

    // TEST-STD-16: Bidirectional reconstruction round-trip
    {
        Model m("TestStd16");
        auto x1 = m.add_variable("x1", 2.0, kInfinity).value();
        auto x2 = m.add_variable("x2", 1.0, 6.0).value();
        (void)m.add_variable("x3", -kInfinity, kInfinity).value();
        (void)m.add_variable("x4", 5.0, 5.0).value();

        auto c1 = m.add_constraint("c1", 10.0, 30.0).value();
        (void)m.add_constraint_term(c1, x1, 1.0);
        (void)m.add_constraint_term(c1, x2, 2.0);

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-16: Model standardizes");
        const auto& slp = std_res.value();

        // Create a feasible point in original space: x1 = 4.0, x2 = 3.0, x3 = -2.5, x4 = 5.0
        auto x_orig = DenseVector::create(4).value();
        set_vec(x_orig, 0, 4.0);
        set_vec(x_orig, 1, 3.0);
        set_vec(x_orig, 2, -2.5);
        set_vec(x_orig, 3, 5.0);

        auto x_bar = slp.project_primal(x_orig).value();
        auto x_reconstructed = slp.reconstruct_primal(x_bar).value();

        bool roundtrip_ok = true;
        for (Dimension j = 0; j < 4; ++j) {
            if (!approx_eq(x_orig.at(j).value(), x_reconstructed.at(j).value())) {
                roundtrip_ok = false;
            }
        }
        check(roundtrip_ok, "TEST-STD-16: Bidirectional reconstruction round-trip exact match");

        // Verify that A_bar * x_bar == b_bar
        auto residual_scratch = DenseVector::create(slp.standardized_constraints()).value();
        auto residual = DenseVector::create(slp.standardized_constraints()).value();
        auto res_st = slp.A().residual(slp.b(), x_bar, residual, residual_scratch);
        check(res_st.is_ok(), "TEST-STD-16: Standard residual computed");
        Scalar max_res = 0.0;
        for (Dimension r = 0; r < slp.standardized_constraints(); ++r) {
            max_res = std::max(max_res, std::abs(residual.at(r).value()));
        }
        check(max_res <= 1e-9, "TEST-STD-16: Projected point satisfies standard constraints A_bar * x_bar = b_bar");
    }

    // TEST-STD-17: Sparse large-index structural case (100 variables, staircase matrix)
    {
        Model m("TestStd17");
        const Dimension N = 100;
        std::vector<VariableIndex> vars;
        for (Dimension i = 0; i < N; ++i) {
            vars.push_back(m.add_variable("x_" + std::to_string(i), 0.0, 10.0).value());
            (void)m.set_objective_coefficient(vars.back(), static_cast<double>(i + 1));
        }

        for (Dimension i = 0; i < N - 1; ++i) {
            auto c = m.add_constraint("c_" + std::to_string(i), -kInfinity, 15.0).value();
            (void)m.add_constraint_term(c, vars[i], 1.0);
            (void)m.add_constraint_term(c, vars[i + 1], 1.0);
        }

        auto std_res = StandardizedLp::standardize(m);
        check(std_res.is_ok(), "TEST-STD-17: 100-variable staircase model standardizes");
        const auto& slp = std_res.value();
        check(slp.original_variables() == N, "TEST-STD-17: Original variables count 100");
        check(slp.standardized_nonzeros() > 0, "TEST-STD-17: Standardized nonzeros verified");
    }

    // TEST-STD-18: Invalid / non-finite inputs rejected
    {
        Model m("TestStd18");
        auto r_invalid_var = m.add_variable("x_inv", 5.0, 2.0); // Invalid lb > ub
        check(!r_invalid_var.is_ok() && r_invalid_var.status().code() == StatusCode::InvalidBounds,
              "TEST-STD-18: Reject variable with lb > ub at Model level");

        Model m2("TestStd18_Int");
        (void)m2.add_variable("y", 0.0, 1.0, VariableType::Integer);
        auto std_res2 = StandardizedLp::standardize(m2);
        check(!std_res2.is_ok() && std_res2.status().code() == StatusCode::InvalidArgument,
              "TEST-STD-18: Reject integer variable in LP standardization");
    }

    // TEST-PROP-01: Deterministic property test across 30 random LP topologies
    {
        std::mt19937_64 rng(424242);
        std::uniform_real_distribution<double> coeff_dist(-10.0, 10.0);
        std::uniform_real_distribution<double> bound_dist(1.0, 20.0);
        std::uniform_int_distribution<int> type_dist(0, 4);

        int prop_passed = 0;
        const int num_trials = 30;

        for (int trial = 0; trial < num_trials; ++trial) {
            Model m("PropModel_" + std::to_string(trial));
            const Dimension n = 5;
            const Dimension mc = 4;

            std::vector<VariableIndex> v_indices;
            for (Dimension j = 0; j < n; ++j) {
                int t = type_dist(rng);
                VariableIndex v = kInvalidVariableIndex;
                if (t == 0) {
                    v = m.add_variable("x" + std::to_string(j), 0.0, kInfinity).value();
                } else if (t == 1) {
                    v = m.add_variable("x" + std::to_string(j), bound_dist(rng), kInfinity).value();
                } else if (t == 2) {
                    v = m.add_variable("x" + std::to_string(j), -kInfinity, bound_dist(rng)).value();
                } else if (t == 3) {
                    double lb = bound_dist(rng);
                    v = m.add_variable("x" + std::to_string(j), lb, lb + bound_dist(rng)).value();
                } else {
                    v = m.add_variable("x" + std::to_string(j), -kInfinity, kInfinity).value();
                }
                (void)m.set_objective_coefficient(v, coeff_dist(rng));
                v_indices.push_back(v);
            }

            for (Dimension i = 0; i < mc; ++i) {
                double l = bound_dist(rng);
                double u = l + bound_dist(rng);
                auto c = m.add_constraint("c" + std::to_string(i), l, u).value();
                for (Dimension j = 0; j < n; ++j) {
                    (void)m.add_constraint_term(c, v_indices[j], coeff_dist(rng));
                }
            }

            auto std_res = StandardizedLp::standardize(m);
            if (!std_res.is_ok()) continue;
            const auto& slp = std_res.value();

            // Pick a candidate original point that respects variable bounds
            auto x_cand = DenseVector::create(n).value();
            for (Dimension j = 0; j < n; ++j) {
                const auto& v = m.get_variable(v_indices[j]);
                double val = 0.0;
                if (v.is_free()) val = coeff_dist(rng);
                else if (std::isinf(v.lower_bound)) val = v.upper_bound - 1.0;
                else if (std::isinf(v.upper_bound)) val = v.lower_bound + 1.0;
                else val = 0.5 * (v.lower_bound + v.upper_bound);
                set_vec(x_cand, j, val);
            }

            auto x_bar = slp.project_primal(x_cand).value();
            auto x_rec = slp.reconstruct_primal(x_bar).value();

            bool match = true;
            for (Dimension j = 0; j < n; ++j) {
                if (!approx_eq(x_cand.at(j).value(), x_rec.at(j).value(), 1e-8)) {
                    match = false;
                }
            }

            // Verify objective correspondence
            Scalar f_orig = slp.evaluate_original_objective(x_cand).value();
            Scalar z_std = slp.evaluate_standard_objective(x_bar).value();
            if (!approx_eq(f_orig, z_std, 1e-7)) {
                match = false;
            }

            if (match) {
                prop_passed++;
            }
        }

        check(prop_passed == num_trials, "TEST-PROP-01: 30/30 deterministic property tests passed");
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "ALL PHASE 3A STANDARDIZATION TESTS PASSED (0 failures)\n";
    } else {
        std::cerr << "FAILURES DETECTED: " << failures << " tests failed\n";
    }
    std::cout << "=========================================================\n";

    return failures;
}
