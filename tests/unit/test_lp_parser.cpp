#include "io/lp_reader.hpp"
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

using namespace sih26119;

// Independent non-circular reference evaluator for quadratic contribution
static double eval_independent_quad(double Q_xx, double Q_yy, double Q_xy, double x, double y) {
    // 1/2 x^T Q x = 1/2 Q_xx x^2 + 1/2 Q_yy y^2 + Q_xy x y
    return 0.5 * Q_xx * x * x + 0.5 * Q_yy * y * y + Q_xy * x * y;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Native LP Parser & Factor-of-Two Gate\n";
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

    LpReader reader;

    // 1. MANDATORY FACTOR-OF-TWO REGRESSION TEST: [ 4 x * y ] / 2
    {
        std::string lp_reg =
            "Minimize\n"
            " obj: [ 4 x * y ] / 2\n"
            "Subject To\n"
            " c1: x + y >= 1\n"
            "End\n";

        std::istringstream iss(lp_reg);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse [ 4 x * y ] / 2");

        if (res.ok()) {
            const auto& model = res.value();
            check(model.is_qp(), "Identified as QP");
            check(model.num_quadratic_nonzeros() == 1, "Exactly one quadratic term");

            const auto& qterm = model.objective().quadratic_terms[0];
            // Rule: algebraic coefficient b = 4 in P -> Q_xy = b / 2 = 2.0
            check(qterm.coefficient == 2.0, "Stored canonical Q_xy == 2.0 (not 4.0)");

            // Evaluate at test point x = 3, y = 5
            double x_val = 3.0;
            double y_val = 5.0;
            double actual_quad_contrib = eval_independent_quad(0.0, 0.0, qterm.coefficient, x_val, y_val);
            check(actual_quad_contrib == 30.0,
                "Quadratic contribution at (x=3, y=5) is exactly 30.0 (NOT 60.0!)");
        }
    }

    // 2. Factor-of-Two Comprehensive Test Suite: Cases A, B, C, D
    // Case A: [ x ^ 2 ] / 2 => Q_xx = 1
    {
        std::string lp_a =
            "Minimize\n"
            " obj: [ x ^ 2 ] / 2\n"
            "Subject To\n"
            " c1: x >= 0\n"
            "End\n";
        std::istringstream iss(lp_a);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse Case A: [ x ^ 2 ] / 2");
        if (res.ok()) {
            const auto& qterm = res.value().objective().quadratic_terms[0];
            check(qterm.is_diagonal() && qterm.coefficient == 1.0, "Case A: Q_xx == 1.0");
        }
    }

    // Case C: [ x ^ 2 + 4 x * y + 7 y ^ 2 ] / 2 => Q_xx = 1, Q_xy = 2, Q_yy = 7
    {
        std::string lp_c =
            "Minimize\n"
            " obj: [ x ^ 2 + 4 x * y + 7 y ^ 2 ] / 2\n"
            "Subject To\n"
            " c1: x + y >= 1\n"
            "End\n";
        std::istringstream iss(lp_c);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse Case C: [ x ^ 2 + 4 x * y + 7 y ^ 2 ] / 2");
        if (res.ok()) {
            const auto& model = res.value();
            double qxx = 0.0, qyy = 0.0, qxy = 0.0;
            VariableIndex vx = model.get_variable_index("x").value();
            VariableIndex vy = model.get_variable_index("y").value();
            for (const auto& qt : model.objective().quadratic_terms) {
                if (qt.var1 == vx && qt.var2 == vx) qxx = qt.coefficient;
                else if (qt.var1 == vy && qt.var2 == vy) qyy = qt.coefficient;
                else qxy = qt.coefficient;
            }
            check(qxx == 1.0, "Case C: Q_xx == 1.0");
            check(qxy == 2.0, "Case C: Q_xy == 2.0 (4 / 2)");
            check(qyy == 7.0, "Case C: Q_yy == 7.0");

            // Independent oracle check: 0.5(1)(3^2) + 0.5(7)(5^2) + 2(3)(5) = 4.5 + 87.5 + 30 = 122.0
            double val = eval_independent_quad(qxx, qyy, qxy, 3.0, 5.0);
            check(std::abs(val - 122.0) < 1e-12, "Case C oracle at (3, 5): 122.0");
        }
    }

    // Case D: [ 2 x ^ 2 + 6 x * y + 8 y ^ 2 ] / 2 => Q_xx = 2, Q_xy = 3, Q_yy = 8
    {
        std::string lp_d =
            "Minimize\n"
            " obj: [ 2 x ^ 2 + 6 x * y + 8 y ^ 2 ] / 2\n"
            "Subject To\n"
            " c1: x + y >= 1\n"
            "End\n";
        std::istringstream iss(lp_d);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse Case D: [ 2 x ^ 2 + 6 x * y + 8 y ^ 2 ] / 2");
        if (res.ok()) {
            const auto& model = res.value();
            double qxx = 0.0, qyy = 0.0, qxy = 0.0;
            VariableIndex vx = model.get_variable_index("x").value();
            VariableIndex vy = model.get_variable_index("y").value();
            for (const auto& qt : model.objective().quadratic_terms) {
                if (qt.var1 == vx && qt.var2 == vx) qxx = qt.coefficient;
                else if (qt.var1 == vy && qt.var2 == vy) qyy = qt.coefficient;
                else qxy = qt.coefficient;
            }
            check(qxx == 2.0, "Case D: Q_xx == 2.0");
            check(qxy == 3.0, "Case D: Q_xy == 3.0 (6 / 2)");
            check(qyy == 8.0, "Case D: Q_yy == 8.0");
        }
    }

    // 3. Strict Rejection of Quadratic Objective Lacking '/ 2'
    {
        std::string lp_bad =
            "Minimize\n"
            " obj: [ 4 x * y ]\n"
            "Subject To\n"
            " c1: x >= 0\n"
            "End\n";
        std::istringstream iss(lp_bad);
        auto res = reader.read_stream(iss);
        check(!res.ok(), "Strictly reject quadratic objective missing '/ 2' divisor");
    }

    // 4. Mixed Integer LP Features
    {
        std::string lp_mip =
            "Maximize\n"
            " obj: 5 x + 4 y + 3 z\n"
            "Subject To\n"
            " c1: 2 x + 3 y <= 15\n"
            " c2: 1.5 <= x + z <= 8.5\n"
            "Bounds\n"
            " 0 <= x <= 10\n"
            " y free\n"
            " z >= 2\n"
            "Generals\n"
            " x\n"
            "Binaries\n"
            " z\n"
            "End\n";
        std::istringstream iss(lp_mip);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse Mixed-Integer LP model");
        if (res.ok()) {
            const auto& model = res.value();
            check(model.is_milp(), "Model identified as MILP");
            check(model.objective().sense == ObjectiveSense::Maximize, "Objective sense Maximize");

            const auto& vx = model.get_variable(model.get_variable_index("x").value());
            check(vx.type == VariableType::Integer && vx.lower_bound == 0.0 && vx.upper_bound == 10.0,
                "Variable x is Integer [0, 10]");

            const auto& vy = model.get_variable(model.get_variable_index("y").value());
            check(vy.is_free(), "Variable y is free");

            const auto& vz = model.get_variable(model.get_variable_index("z").value());
            check(vz.is_binary() && vz.lower_bound == 0.0 && vz.upper_bound == 1.0,
                "Variable z is Binary [0, 1]");

            const auto& c2 = model.get_constraint(model.get_constraint_index("c2").value());
            check(c2.is_range() && c2.lower_bound == 1.5 && c2.upper_bound == 8.5,
                "Range constraint c2: [1.5, 8.5]");
        }
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "LP PARSER TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << "LP PARSER TESTS FAILED (" << failures << " failures)\n";
        return 1;
    }
}
