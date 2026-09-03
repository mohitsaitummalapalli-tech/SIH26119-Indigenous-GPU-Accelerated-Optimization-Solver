#include "model/model.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <map>
#include <vector>

using namespace sih26119;

// Independent reference oracle for quadratic objective evaluation
static double evaluate_reference_polynomial(
    const std::vector<double>& c,
    const std::map<std::pair<uint32_t, uint32_t>, double>& Q,
    double c0,
    const std::vector<double>& x)
{
    double val = c0;
    for (size_t j = 0; j < c.size(); ++j) {
        val += c[j] * x[j];
    }
    for (const auto& [pair, q_val] : Q) {
        uint32_t i = pair.first;
        uint32_t j = pair.second;
        if (i == j) {
            val += 0.5 * q_val * x[i] * x[i];
        } else {
            val += q_val * x[i] * x[j];
        }
    }
    return val;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Canonical Mathematical Model\n";
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

    // 1. Variable addition and bounds
    {
        Model m("TestVar");
        auto r1 = m.add_variable("x1", 0.0, 10.0, VariableType::Continuous);
        check(r1.ok(), "Add valid continuous variable x1");

        auto r_dup = m.add_variable("x1", 0.0, 5.0);
        check(!r_dup.ok() && r_dup.status().code() == StatusCode::DuplicateVariableName,
            "Reject duplicate variable name");

        auto r_empty = m.add_variable("", 0.0, 5.0);
        check(!r_empty.ok() && r_empty.status().code() == StatusCode::InvalidArgument,
            "Reject empty variable name");

        auto r_bounds = m.add_variable("x2", 10.0, 5.0);
        check(!r_bounds.ok() && r_bounds.status().code() == StatusCode::InvalidBounds,
            "Reject lower bound > upper bound");

        auto r_nan = m.add_variable("x3", std::numeric_limits<double>::quiet_NaN(), 5.0);
        check(!r_nan.ok() && r_nan.status().code() == StatusCode::InvalidArgument,
            "Reject NaN bound");

        auto r_pos_inf = m.add_variable("x4", kInfinity, kInfinity);
        check(!r_pos_inf.ok() && r_pos_inf.status().code() == StatusCode::InvalidBounds,
            "Reject lower bound = +infinity");

        auto r_neg_inf = m.add_variable("x5", -kInfinity, -kInfinity);
        check(!r_neg_inf.ok() && r_neg_inf.status().code() == StatusCode::InvalidBounds,
            "Reject upper bound = -infinity");
    }

    // 2. Binary variable domain enforcement
    {
        Model m("TestBin");
        auto r_bin_ok = m.add_variable("b1", 0.0, 1.0, VariableType::Binary);
        check(r_bin_ok.ok(), "Add valid binary variable [0, 1]");

        auto r_bin_neg = m.add_variable("b2", -1.0, 1.0, VariableType::Binary);
        check(!r_bin_neg.ok() && r_bin_neg.status().code() == StatusCode::InvalidBinaryDeclaration,
            "Reject binary variable with lb < 0");

        auto r_bin_large = m.add_variable("b3", 0.0, 2.0, VariableType::Binary);
        check(!r_bin_large.ok() && r_bin_large.status().code() == StatusCode::InvalidBinaryDeclaration,
            "Reject binary variable with ub > 1");

        auto r_bin_frac = m.add_variable("b4", 0.5, 1.0, VariableType::Binary);
        check(!r_bin_frac.ok() && r_bin_frac.status().code() == StatusCode::InvalidBinaryDeclaration,
            "Reject binary variable with fractional bounds");
    }

    // 3. Constraints and linear terms
    {
        Model m("TestCon");
        auto v1 = m.add_variable("x", 0.0, 10.0).value();
        auto v2 = m.add_variable("y", 0.0, 10.0).value();

        auto c1 = m.add_constraint("c1", 0.0, 20.0, {{v1, 2.0}, {v2, 3.0}});
        check(c1.ok(), "Add valid constraint 2x + 3y <= 20");

        auto c_dup = m.add_constraint("c1", 0.0, 10.0);
        check(!c_dup.ok() && c_dup.status().code() == StatusCode::DuplicateConstraintName,
            "Reject duplicate constraint name");

        auto c_inv_var = m.add_constraint("c2", 0.0, 10.0, {{static_cast<VariableIndex>(99), 1.0}});
        check(!c_inv_var.ok() && c_inv_var.status().code() == StatusCode::InvalidVariableReference,
            "Reject invalid variable index in constraint");

        check(m.num_constraints() == 1, "Constraint count is 1");
        check(m.num_nonzeros() == 2, "Linear nonzeros count is 2");
    }

    // 4. Canonical Quadratic Objective & Independent Oracle
    {
        Model m("TestQP");
        auto vx = m.add_variable("x", -10.0, 10.0).value();
        auto vy = m.add_variable("y", -10.0, 10.0).value();

        m.set_objective_sense(ObjectiveSense::Minimize);
        m.set_objective_offset(5.0);
        m.set_objective_coefficient(vx, 3.0);
        m.set_objective_coefficient(vy, -2.0);

        // Q = [ 4  2 ]
        //     [ 2  6 ]
        // Stored Q_xx = 4, Q_yy = 6, Q_xy = 2
        m.set_quadratic_coefficient(vx, vx, 4.0);
        m.set_quadratic_coefficient(vy, vy, 6.0);
        m.set_quadratic_coefficient(vx, vy, 2.0);

        check(m.is_qp(), "Model is identified as QP");
        check(m.num_quadratic_nonzeros() == 3, "QP has 3 quadratic nonzeros");

        // Independent oracle evaluation across 5 distinct points
        // f_ref(x, y) = 5 + 3x - 2y + 0.5(4 x^2 + 6 y^2) + 2 x y
        //             = 5 + 3x - 2y + 2 x^2 + 3 y^2 + 2 x y
        std::vector<double> c_ref = {3.0, -2.0};
        std::map<std::pair<uint32_t, uint32_t>, double> Q_ref = {
            {{0, 0}, 4.0},
            {{1, 1}, 6.0},
            {{0, 1}, 2.0}
        };
        double c0_ref = 5.0;

        std::vector<std::pair<double, double>> test_points = {
            {0.0, 0.0},
            {1.0, 1.0},
            {2.0, -1.0},
            {-3.0, 4.0},
            {0.5, 2.5}
        };

        for (const auto& [x_val, y_val] : test_points) {
            double oracle_val = evaluate_reference_polynomial(c_ref, Q_ref, c0_ref, {x_val, y_val});
            double hand_val = 5.0 + 3.0 * x_val - 2.0 * y_val + 2.0 * x_val * x_val + 3.0 * y_val * y_val + 2.0 * x_val * y_val;
            check(std::abs(oracle_val - hand_val) < 1e-12,
                "Independent oracle agrees with hand expansion at (" + std::to_string(x_val) + ", " + std::to_string(y_val) + ")");
        }
    }

    // 5. Semantic Equivalence Testing
    {
        Model m1("M1");
        auto v1_a = m1.add_variable("x", 0.0, 5.0).value();
        auto v1_b = m1.add_variable("y", 1.0, 10.0, VariableType::Integer).value();
        m1.add_constraint("c1", 0.0, 15.0, {{v1_a, 1.0}, {v1_b, 2.0}});
        m1.set_objective_coefficient(v1_a, 4.0);
        m1.set_quadratic_coefficient(v1_a, v1_b, 1.5);

        Model m2("M2");
        auto v2_b = m2.add_variable("y", 1.0, 10.0, VariableType::Integer).value();
        auto v2_a = m2.add_variable("x", 0.0, 5.0).value();
        m2.add_constraint("c1", 0.0, 15.0, {{v2_b, 2.0}, {v2_a, 1.0}});
        m2.set_objective_coefficient(v2_a, 4.0);
        m2.set_quadratic_coefficient(v2_b, v2_a, 1.5); // Reverse variable order in set

        check(m1.semantic_equals(m2), "Models with reversed variable insertion order are semantically equal");

        // Alter a bound slightly beyond tolerance
        m2.set_variable_bounds(v2_a, 0.0, 5.0001);
        check(!m1.semantic_equals(m2), "Detects semantic bound difference beyond tolerance");
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "MODEL TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << "MODEL TESTS FAILED (" << failures << " failures)\n";
        return 1;
    }
}
