#include "io/mps_reader.hpp"
#include <iostream>
#include <sstream>
#include <cassert>

using namespace sih26119;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Native MPS Parser & Semantics\n";
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

    MpsReader reader;

    // 1. Authoritative MPS RANGES Arithmetic Verification
    {
        std::string mps_data =
            "NAME          RANGES_TEST\n"
            "ROWS\n"
            " N  OBJ\n"
            " G  ROW_G_POS\n"
            " G  ROW_G_NEG\n"
            " L  ROW_L_POS\n"
            " L  ROW_L_NEG\n"
            " E  ROW_E_POS\n"
            " E  ROW_E_NEG\n"
            " E  ROW_E_ZERO\n"
            "COLUMNS\n"
            "    X         OBJ       1.0       ROW_G_POS 1.0\n"
            "    X         ROW_G_NEG 1.0       ROW_L_POS 1.0\n"
            "    X         ROW_L_NEG 1.0       ROW_E_POS 1.0\n"
            "    X         ROW_E_NEG 1.0       ROW_E_ZERO 1.0\n"
            "RHS\n"
            "    RHS1      ROW_G_POS 10.0      ROW_G_NEG 10.0\n"
            "    RHS1      ROW_L_POS 10.0      ROW_L_NEG 10.0\n"
            "    RHS1      ROW_E_POS 10.0      ROW_E_NEG 10.0\n"
            "    RHS1      ROW_E_ZERO 10.0\n"
            "RANGES\n"
            "    RNG1      ROW_G_POS 4.0       ROW_G_NEG -4.0\n"
            "    RNG1      ROW_L_POS 4.0       ROW_L_NEG -4.0\n"
            "    RNG1      ROW_E_POS 4.0       ROW_E_NEG -4.0\n"
            "    RNG1      ROW_E_ZERO 0.0\n"
            "BOUNDS\n"
            " UP BND1      X         100.0\n"
            "ENDATA\n";

        std::istringstream iss(mps_data);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse MPS with full RANGES test matrix");

        if (res.ok()) {
            const auto& model = res.value();

            // Hand-computed oracle checks:
            // 1) G with b = 10, r = 4 => [10, 14]
            const auto& c_g_pos = model.get_constraint(model.get_constraint_index("ROW_G_POS").value());
            check(c_g_pos.lower_bound == 10.0 && c_g_pos.upper_bound == 14.0,
                "Hand-computed G row (r=4.0): [10.0, 14.0]");

            // 2) G with b = 10, r = -4 => [10, 14]
            const auto& c_g_neg = model.get_constraint(model.get_constraint_index("ROW_G_NEG").value());
            check(c_g_neg.lower_bound == 10.0 && c_g_neg.upper_bound == 14.0,
                "Hand-computed G row (r=-4.0): [10.0, 14.0]");

            // 3) L with b = 10, r = 4 => [6, 10]
            const auto& c_l_pos = model.get_constraint(model.get_constraint_index("ROW_L_POS").value());
            check(c_l_pos.lower_bound == 6.0 && c_l_pos.upper_bound == 10.0,
                "Hand-computed L row (r=4.0): [6.0, 10.0]");

            // 4) L with b = 10, r = -4 => [6, 10]
            const auto& c_l_neg = model.get_constraint(model.get_constraint_index("ROW_L_NEG").value());
            check(c_l_neg.lower_bound == 6.0 && c_l_neg.upper_bound == 10.0,
                "Hand-computed L row (r=-4.0): [6.0, 10.0]");

            // 5) E with b = 10, r = 4 => [10, 14]
            const auto& c_e_pos = model.get_constraint(model.get_constraint_index("ROW_E_POS").value());
            check(c_e_pos.lower_bound == 10.0 && c_e_pos.upper_bound == 14.0,
                "Hand-computed E row (r=4.0): [10.0, 14.0]");

            // 6) E with b = 10, r = -4 => [6, 10]
            const auto& c_e_neg = model.get_constraint(model.get_constraint_index("ROW_E_NEG").value());
            check(c_e_neg.lower_bound == 6.0 && c_e_neg.upper_bound == 10.0,
                "Hand-computed E row (r=-4.0): [6.0, 10.0]");

            // 7) E with b = 10, r = 0 => [10, 10]
            const auto& c_e_zero = model.get_constraint(model.get_constraint_index("ROW_E_ZERO").value());
            check(c_e_zero.lower_bound == 10.0 && c_e_zero.upper_bound == 10.0,
                "Hand-computed E row (r=0.0): [10.0, 10.0]");
        }
    }

    // 2. Multiple Vector Sets Selection Rule (First vector selected)
    {
        std::string mps_multi =
            "NAME          MULTI_SET\n"
            "ROWS\n"
            " N  OBJ\n"
            " L  C1\n"
            "COLUMNS\n"
            "    X         OBJ       1.0       C1        1.0\n"
            "RHS\n"
            "    RHS_FIRST C1        25.0\n"
            "    RHS_OTHER C1        999.0\n"
            "BOUNDS\n"
            " UP BND_FIRST X         50.0\n"
            " UP BND_OTHER X         888.0\n"
            "ENDATA\n";

        std::istringstream iss(mps_multi);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse multi-set MPS file");
        if (res.ok()) {
            const auto& model = res.value();
            const auto& c1 = model.get_constraint(0);
            check(c1.upper_bound == 25.0, "Selected first RHS set (25.0, not 999.0)");
            const auto& x = model.get_variable(0);
            check(x.upper_bound == 50.0, "Selected first BOUNDS set (50.0, not 888.0)");
        }
    }

    // 3. QUADOBJ matrix entries
    {
        std::string mps_qp =
            "NAME          QP_MPS\n"
            "ROWS\n"
            " N  OBJ\n"
            "COLUMNS\n"
            "    X         OBJ       1.0\n"
            "    Y         OBJ       2.0\n"
            "RHS\n"
            "BOUNDS\n"
            " FR BND1      X\n"
            " FR BND1      Y\n"
            "QUADOBJ\n"
            "    X         X         4.0\n"
            "    Y         Y         6.0\n"
            "    X         Y         3.0\n"
            "    Y         X         3.0\n" // Symmetric redundant record
            "ENDATA\n";

        std::istringstream iss(mps_qp);
        auto res = reader.read_stream(iss);
        check(res.ok(), "Parse QUADOBJ with symmetric redundant entry");
        if (res.ok()) {
            const auto& model = res.value();
            check(model.is_qp(), "Identified as QP");
            check(model.num_quadratic_nonzeros() == 3, "Stored 3 unique canonical quadratic terms");
        }
    }

    // 4. QUADOBJ Conflicting Symmetric Entry Rejection
    {
        std::string mps_conflict =
            "NAME          QP_CONFLICT\n"
            "ROWS\n"
            " N  OBJ\n"
            "COLUMNS\n"
            "    X         OBJ       1.0\n"
            "    Y         OBJ       2.0\n"
            "RHS\n"
            "QUADOBJ\n"
            "    X         Y         3.0\n"
            "    Y         X         5.0\n" // Conflict!
            "ENDATA\n";

        std::istringstream iss(mps_conflict);
        auto res = reader.read_stream(iss);
        check(!res.ok(), "Reject conflicting symmetric values in QUADOBJ");
    }

    // 5. Unsupported Features Rejection (QCMATRIX & SOS)
    {
        std::string mps_qc =
            "NAME          QC_TEST\n"
            "ROWS\n"
            " N  OBJ\n"
            "COLUMNS\n"
            "    X         OBJ       1.0\n"
            "RHS\n"
            "QCMATRIX      C1\n"
            "    X         X         1.0\n"
            "ENDATA\n";

        std::istringstream iss(mps_qc);
        auto res = reader.read_stream(iss);
        check(!res.ok(), "Reject unsupported QCMATRIX section");
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "MPS PARSER TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << "MPS PARSER TESTS FAILED (" << failures << " failures)\n";
        return 1;
    }
}
