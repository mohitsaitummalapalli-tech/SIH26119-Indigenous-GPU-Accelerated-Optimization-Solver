#include "model/model.hpp"
#include "io/mps_reader.hpp"
#include "io/mps_writer.hpp"
#include "io/lp_reader.hpp"
#include "io/lp_writer.hpp"
#include <iostream>
#include <sstream>
#include <cassert>

using namespace sih26119;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Model Serialization & Round-Trip\n";
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

    MpsWriter mps_writer;
    MpsReader mps_reader;
    LpWriter lp_writer;
    LpReader lp_reader;

    // 1. Mixed-Integer LP Model Round-Trip (MPS and LP)
    {
        Model original("MILP_Test");
        auto vx = original.add_variable("x", 0.0, 10.0, VariableType::Integer).value();
        auto vy = original.add_variable("y", -5.0, 15.0, VariableType::Continuous).value();
        auto vb = original.add_variable("b", 0.0, 1.0, VariableType::Binary).value();

        original.set_objective_sense(ObjectiveSense::Maximize);
        original.add_objective_term(vx, 3.5);
        original.add_objective_term(vy, -2.0);
        original.add_objective_term(vb, 10.0);

        original.add_constraint("c1", -kInfinity, 25.0, {{vx, 1.0}, {vy, 2.0}});
        original.add_constraint("c2", 5.0, kInfinity, {{vx, 3.0}, {vb, -1.0}});
        original.add_constraint("c3", 10.0, 20.0, {{vy, 1.0}, {vb, 4.0}}); // Range constraint

        // Test MPS round-trip
        std::ostringstream mps_ss;
        auto w_status = mps_writer.write_stream(original, mps_ss);
        check(w_status.is_ok(), "Write MILP model to MPS");

        std::istringstream mps_iss(mps_ss.str());
        auto r_res = mps_reader.read_stream(mps_iss);
        check(r_res.ok(), "Read MILP model from generated MPS");
        if (r_res.ok()) {
            check(original.semantic_equals(r_res.value()), "MPS round-trip semantic equality verified");
        }

        // Test LP round-trip
        std::ostringstream lp_ss;
        auto lp_w_status = lp_writer.write_stream(original, lp_ss);
        check(lp_w_status.is_ok(), "Write MILP model to LP");

        std::istringstream lp_iss(lp_ss.str());
        auto lp_r_res = lp_reader.read_stream(lp_iss);
        check(lp_r_res.ok(), "Read MILP model from generated LP");
        if (lp_r_res.ok()) {
            check(original.semantic_equals(lp_r_res.value()), "LP round-trip semantic equality verified");
        }
    }

    // 2. Quadratic Programming (QP) Model Round-Trip (MPS and LP)
    {
        Model original("QP_RoundTrip");
        auto vx = original.add_variable("x", 0.0, 50.0).value();
        auto vy = original.add_variable("y", 0.0, 50.0).value();
        auto vz = original.add_variable("z", -10.0, 10.0).value();

        original.set_objective_sense(ObjectiveSense::Minimize);
        original.add_objective_term(vx, 1.0);
        original.add_objective_term(vy, 2.0);

        // Canonical Q matrix:
        // Q_xx = 4.0, Q_yy = 8.0, Q_xy = 3.0, Q_yz = -1.5
        original.set_quadratic_coefficient(vx, vx, 4.0);
        original.set_quadratic_coefficient(vy, vy, 8.0);
        original.set_quadratic_coefficient(vx, vy, 3.0);
        original.set_quadratic_coefficient(vy, vz, -1.5);

        original.add_constraint("con1", 0.0, 100.0, {{vx, 1.0}, {vy, 1.0}, {vz, 1.0}});

        // MPS round-trip
        std::ostringstream mps_ss;
        check(mps_writer.write_stream(original, mps_ss).is_ok(), "Write QP to MPS stream");
        std::istringstream mps_iss(mps_ss.str());
        auto mps_res = mps_reader.read_stream(mps_iss);
        check(mps_res.ok(), "Read QP from MPS stream");
        if (mps_res.ok()) {
            check(original.semantic_equals(mps_res.value()), "QP MPS round-trip semantic equality");
        }

        // LP round-trip
        std::ostringstream lp_ss;
        check(lp_writer.write_stream(original, lp_ss).is_ok(), "Write QP to LP stream");
        std::istringstream lp_iss(lp_ss.str());
        auto lp_res = lp_reader.read_stream(lp_iss);
        check(lp_res.ok(), "Read QP from LP stream");
        if (lp_res.ok()) {
            check(original.semantic_equals(lp_res.value()), "QP LP round-trip semantic equality");
        }
    }

    // 3. Cross-Format Interchange: LP -> Model -> MPS -> Model
    {
        std::string initial_lp =
            "Minimize\n"
            " obj: 2 x + 3 y + [ 2 x ^ 2 + 6 x * y + 8 y ^ 2 ] / 2\n"
            "Subject To\n"
            " c1: x + y >= 5\n"
            " c2: 2 x - y <= 10\n"
            "Bounds\n"
            " x >= 0\n"
            " y >= 0\n"
            "End\n";

        std::istringstream lp_in(initial_lp);
        auto m1_res = lp_reader.read_stream(lp_in);
        check(m1_res.ok(), "Parse initial cross-format LP");

        if (m1_res.ok()) {
            const auto& m1 = m1_res.value();
            std::ostringstream mps_out;
            check(mps_writer.write_stream(m1, mps_out).is_ok(), "Serialize model to MPS");

            std::istringstream mps_in(mps_out.str());
            auto m2_res = mps_reader.read_stream(mps_in);
            check(m2_res.ok(), "Deserialize model from MPS");

            if (m2_res.ok()) {
                check(m1.semantic_equals(m2_res.value()),
                    "Cross-format interchange (LP -> Model -> MPS -> Model) semantically identical");
            }
        }
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "SERIALIZATION ROUND-TRIP TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << "SERIALIZATION ROUND-TRIP TESTS FAILED (" << failures << " failures)\n";
        return 1;
    }
}
