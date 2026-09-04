#include "solver/lp/basis.hpp"
#include "solver/lp/basic_solution.hpp"
#include "solver/lp/basis_matrix_view.hpp"
#include "numerics/sparse_matrix.hpp"
#include "numerics/dense_vector.hpp"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <random>
#include <algorithm>

using namespace sih26119;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "SIH26119 Test Suite: Phase 3B Basis Representation\n";
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

    // TEST-BASIS-01: Valid basis construction
    {
        // m = 3, n = 6, B = [1, 3, 5]
        auto b_res = Basis::create(3, 6, {1, 3, 5});
        check(b_res.is_ok(), "TEST-BASIS-01: Valid basis construction succeeds");
        const auto& b = b_res.value();
        check(b.num_rows() == 3 && b.num_cols() == 6, "TEST-BASIS-01: Correct dimensions m=3, n=6");
        check(b.is_structurally_valid(), "TEST-BASIS-01: Basis is structurally valid");
        check(b.version() == 1, "TEST-BASIS-01: Initial version is 1");
    }

    // TEST-BASIS-02: Duplicate basic variable rejection
    {
        auto b_res = Basis::create(3, 6, {1, 3, 1}); // Variable 1 repeated
        check(!b_res.is_ok() && b_res.status().code() == StatusCode::InvalidArgument,
              "TEST-BASIS-02: Rejects duplicate basic variable");
    }

    // TEST-BASIS-03: Out-of-range rejection
    {
        auto b_res = Basis::create(2, 4, {1, 4}); // Column 4 >= num_cols (4)
        check(!b_res.is_ok() && b_res.status().code() == StatusCode::InvalidArgument,
              "TEST-BASIS-03: Rejects out-of-range column index");
    }

    // TEST-BASIS-04: Wrong basis dimension rejection
    {
        // More rows than columns: m = 5, n = 3
        auto b_res1 = Basis::create(5, 3, {0, 1, 2});
        check(!b_res1.is_ok() && b_res1.status().code() == StatusCode::InvalidArgument,
              "TEST-BASIS-04: Rejects m > n");

        // basic_vars.size() != m: m = 3, size = 2
        auto b_res2 = Basis::create(3, 5, {0, 1});
        check(!b_res2.is_ok() && b_res2.status().code() == StatusCode::InvalidArgument,
              "TEST-BASIS-04: Rejects basic_vars.size() != m");
    }

    // TEST-BASIS-05: Basic/nonbasic partition
    {
        // m = 3, n = 6, B = [0, 2, 4] => N should be [1, 3, 5]
        auto b = Basis::create(3, 6, {0, 2, 4}).value();
        check(b.is_basic(0) && b.is_basic(2) && b.is_basic(4), "TEST-BASIS-05: Basic variables verified");
        check(b.is_nonbasic(1) && b.is_nonbasic(3) && b.is_nonbasic(5), "TEST-BASIS-05: Nonbasic variables verified");
        check(!b.is_basic(1) && !b.is_nonbasic(0), "TEST-BASIS-05: Strict complement partition");
        check(b.nonbasic_variables().size() == 3, "TEST-BASIS-05: Nonbasic size is n - m = 3");
    }

    // TEST-BASIS-06: Basic-variable row mapping
    {
        // B(0) = 4, B(1) = 1, B(2) = 3
        auto b = Basis::create(3, 5, {4, 1, 3}).value();
        check(b.basic_variable(0).value() == 4, "TEST-BASIS-06: Row 0 -> Col 4");
        check(b.basic_variable(1).value() == 1, "TEST-BASIS-06: Row 1 -> Col 1");
        check(b.basic_variable(2).value() == 3, "TEST-BASIS-06: Row 2 -> Col 3");

        check(b.row_of_basic(4).value() == 0, "TEST-BASIS-06: Col 4 -> Row 0");
        check(b.row_of_basic(1).value() == 1, "TEST-BASIS-06: Col 1 -> Row 1");
        check(b.row_of_basic(3).value() == 2, "TEST-BASIS-06: Col 3 -> Row 2");

        check(!b.row_of_basic(0).is_ok(), "TEST-BASIS-06: Nonbasic Col 0 has no row");
    }

    // TEST-BASIS-07: Successful basis replacement
    {
        // m = 2, n = 4, B = [1, 3], N = [0, 2]
        auto b = Basis::create(2, 4, {1, 3}).value();
        // Replace leaving row 0 (leaving var 1) with entering col 2
        auto st = b.replace_basic_variable(2, 0);
        check(st.is_ok(), "TEST-BASIS-07: Successful basis replacement");
        check(b.basic_variable(0).value() == 2, "TEST-BASIS-07: Row 0 now maps to Col 2");
        check(b.basic_variable(1).value() == 3, "TEST-BASIS-07: Row 1 still maps to Col 3");
        check(b.is_basic(2), "TEST-BASIS-07: Col 2 is now basic");
        check(b.is_nonbasic(1), "TEST-BASIS-07: Col 1 is now nonbasic");
        check(b.row_of_basic(2).value() == 0, "TEST-BASIS-07: Col 2 row is 0");
        check(b.version() == 2, "TEST-BASIS-07: Version incremented to 2");
    }

    // TEST-BASIS-08: Replacement with already-basic entering variable
    {
        auto b = Basis::create(2, 4, {1, 3}).value();
        // Attempt entering col 3 (which is already basic at row 1)
        auto st = b.replace_basic_variable(3, 0);
        check(!st.is_ok() && st.code() == StatusCode::InvalidArgument,
              "TEST-BASIS-08: Rejects entering variable that is already basic");
    }

    // TEST-BASIS-09: Transactional failed replacement
    {
        auto b = Basis::create(2, 4, {1, 3}).value();
        const uint64_t v_orig = b.version();
        const auto b_orig = b.basic_variables();
        const auto nb_orig = b.nonbasic_variables();

        // 1. Invalid leaving row out of bounds
        auto st1 = b.replace_basic_variable(0, 5);
        check(!st1.is_ok(), "TEST-BASIS-09: Rejects out-of-bounds leaving row");
        check(b.version() == v_orig, "TEST-BASIS-09: Version unchanged after failed row");
        check(b.basic_variables() == b_orig && b.nonbasic_variables() == nb_orig,
              "TEST-BASIS-09: State fully preserved after failed row");

        // 2. Invalid entering column out of bounds
        auto st2 = b.replace_basic_variable(10, 0);
        check(!st2.is_ok(), "TEST-BASIS-09: Rejects out-of-bounds entering column");
        check(b.version() == v_orig, "TEST-BASIS-09: Version unchanged after failed col");
        check(b.basic_variables() == b_orig && b.nonbasic_variables() == nb_orig,
              "TEST-BASIS-09: State fully preserved after failed col");
    }

    // TEST-BASIS-10: Empty-row case m = 0
    {
        auto b = Basis::create(0, 5, {}).value();
        check(b.num_rows() == 0 && b.num_cols() == 5, "TEST-BASIS-10: m = 0 basis created");
        check(b.basic_variables().empty(), "TEST-BASIS-10: 0 basic variables");
        check(b.nonbasic_variables().size() == 5, "TEST-BASIS-10: All 5 columns are nonbasic");
        check(b.is_nonbasic(0) && b.is_nonbasic(4), "TEST-BASIS-10: Variables 0 and 4 nonbasic");
    }

    // TEST-BASIS-11: Single-row basis (m = 1)
    {
        auto b = Basis::create(1, 3, {2}).value();
        check(b.basic_variable(0).value() == 2, "TEST-BASIS-11: Row 0 -> Col 2");
        check(b.row_of_basic(2).value() == 0, "TEST-BASIS-11: Col 2 -> Row 0");
        check(b.is_nonbasic(0) && b.is_nonbasic(1), "TEST-BASIS-11: Cols 0 and 1 nonbasic");

        auto st = b.replace_basic_variable(0, 0);
        check(st.is_ok(), "TEST-BASIS-11: Single-row replacement succeeds");
        check(b.basic_variable(0).value() == 0, "TEST-BASIS-11: Row 0 -> Col 0");
        check(b.is_nonbasic(2), "TEST-BASIS-11: Col 2 is now nonbasic");
    }

    // TEST-BASIS-12: Basic solution dimension validation
    {
        auto b = Basis::create(2, 4, {0, 2}).value();
        auto x_b_bad = DenseVector::create(3).value(); // size 3 != m (2)
        auto sol_res = BasicSolution::create(b, std::move(x_b_bad));
        check(!sol_res.is_ok() && sol_res.status().code() == StatusCode::InvalidArgument,
              "TEST-BASIS-12: Rejects x_B size mismatch");
    }

    // TEST-BASIS-13: Independent primal feasibility
    {
        // A: 2 x 4
        // [ 1  2  1  0 ] [x0]   [ 5 ]
        // [ 3  1  0  1 ] [x1] = [ 8 ]
        //                [x2]
        //                [x3]
        // Basis: B = [2, 3] (slacks x2, x3)
        // With x_N = 0 (x0 = 0, x1 = 0) => x2 = 5, x3 = 8
        std::vector<Triplet> trips = {
            {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 1.0},
            {1, 0, 3.0}, {1, 1, 1.0}, {1, 3, 1.0}
        };
        auto A = SparseMatrix::from_triplets(2, 4, trips).value();
        auto b_vec = DenseVector::create(2).value();
        set_vec(b_vec, 0, 5.0);
        set_vec(b_vec, 1, 8.0);

        auto basis = Basis::create(2, 4, {2, 3}).value();
        auto x_B = DenseVector::create(2).value();
        set_vec(x_B, 0, 5.0);
        set_vec(x_B, 1, 8.0);

        auto sol = BasicSolution::create(basis, std::move(x_B)).value();
        auto feas_st = BasicSolution::check_primal_feasibility(A, b_vec, sol);
        check(feas_st.is_ok(), "TEST-BASIS-13: Primal feasibility independently verified");

        // Verify full vector expansion: [0, 0, 5, 8]
        auto x_full = sol.expand_full_primal().value();
        check(approx_eq(x_full.at(0).value(), 0.0) && approx_eq(x_full.at(1).value(), 0.0) &&
              approx_eq(x_full.at(2).value(), 5.0) && approx_eq(x_full.at(3).value(), 8.0),
              "TEST-BASIS-13: Full primal expansion matches x_B and x_N = 0");
    }

    // TEST-BASIS-14: Negative basic-value rejection
    {
        std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0}};
        auto A = SparseMatrix::from_triplets(2, 2, trips).value();
        auto b_vec = DenseVector::create(2).value();
        set_vec(b_vec, 0, 1.0);
        set_vec(b_vec, 1, -2.0);

        auto basis = Basis::create(2, 2, {0, 1}).value();
        auto x_B = DenseVector::create(2).value();
        set_vec(x_B, 0, 1.0);
        set_vec(x_B, 1, -2.0); // Negative basic coordinate violates x >= 0

        auto sol = BasicSolution::create(basis, std::move(x_B)).value();
        auto feas_st = BasicSolution::check_primal_feasibility(A, b_vec, sol);
        check(!feas_st.is_ok() && feas_st.code() == StatusCode::InvalidBounds,
              "TEST-BASIS-14: Rejects negative basic coordinate");
    }

    // TEST-BASIS-15: Equality residual rejection
    {
        std::vector<Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0}};
        auto A = SparseMatrix::from_triplets(2, 2, trips).value();
        auto b_vec = DenseVector::create(2).value();
        set_vec(b_vec, 0, 5.0);
        set_vec(b_vec, 1, 5.0);

        auto basis = Basis::create(2, 2, {0, 1}).value();
        auto x_B = DenseVector::create(2).value();
        set_vec(x_B, 0, 5.0);
        set_vec(x_B, 1, 2.0); // A x = [5, 2] != b [5, 5]

        auto sol = BasicSolution::create(basis, std::move(x_B)).value();
        auto feas_st = BasicSolution::check_primal_feasibility(A, b_vec, sol);
        check(!feas_st.is_ok() && feas_st.code() == StatusCode::InvalidBounds,
              "TEST-BASIS-15: Rejects equality residual violation");
    }

    // TEST-BASIS-16: Basis-column ordering determinism & BasisMatrixView
    {
        // A: 2 x 4
        // [ 10  20  30  40 ]
        // [ 50  60  70  80 ]
        std::vector<Triplet> trips = {
            {0, 0, 10.0}, {0, 1, 20.0}, {0, 2, 30.0}, {0, 3, 40.0},
            {1, 0, 50.0}, {1, 1, 60.0}, {1, 2, 70.0}, {1, 3, 80.0}
        };
        auto A = SparseMatrix::from_triplets(2, 4, trips).value();
        auto basis = Basis::create(2, 4, {3, 1}).value(); // Col 3 in row 0, Col 1 in row 1

        BasisMatrixView bview(A, basis);
        check(bview.num_rows() == 2 && bview.num_cols() == 2, "TEST-BASIS-16: BasisMatrixView dimensions 2x2");
        check(bview.original_column_index(0).value() == 3, "TEST-BASIS-16: View col 0 is original col 3");
        check(bview.original_column_index(1).value() == 1, "TEST-BASIS-16: View col 1 is original col 1");

        // B = [ A[:, 3], A[:, 1] ]
        // B = [ 40  20 ]
        //     [ 80  60 ]
        check(approx_eq(bview.get(0, 0).value(), 40.0), "TEST-BASIS-16: B(0, 0) == 40");
        check(approx_eq(bview.get(1, 0).value(), 80.0), "TEST-BASIS-16: B(1, 0) == 80");
        check(approx_eq(bview.get(0, 1).value(), 20.0), "TEST-BASIS-16: B(0, 1) == 20");
        check(approx_eq(bview.get(1, 1).value(), 60.0), "TEST-BASIS-16: B(1, 1) == 60");
    }

    // TEST-BASIS-17: Repeated basis replacement invariant preservation
    {
        // m = 3, n = 6. Start with B = [0, 1, 2]
        auto b = Basis::create(3, 6, {0, 1, 2}).value();

        // Sequence of 5 replacements
        check(b.replace_basic_variable(3, 0).is_ok(), "TEST-BASIS-17: Pivot 1 ok"); // B = [3, 1, 2]
        check(b.replace_basic_variable(4, 1).is_ok(), "TEST-BASIS-17: Pivot 2 ok"); // B = [3, 4, 2]
        check(b.replace_basic_variable(5, 2).is_ok(), "TEST-BASIS-17: Pivot 3 ok"); // B = [3, 4, 5]
        check(b.replace_basic_variable(0, 0).is_ok(), "TEST-BASIS-17: Pivot 4 ok"); // B = [0, 4, 5]
        check(b.replace_basic_variable(1, 1).is_ok(), "TEST-BASIS-17: Pivot 5 ok"); // B = [0, 1, 5]

        check(b.version() == 6, "TEST-BASIS-17: Version is 6");
        check(b.is_basic(0) && b.is_basic(1) && b.is_basic(5), "TEST-BASIS-17: Basics [0, 1, 5] confirmed");
        check(b.is_nonbasic(2) && b.is_nonbasic(3) && b.is_nonbasic(4), "TEST-BASIS-17: Nonbasics [2, 3, 4] confirmed");
        check(b.row_of_basic(0).value() == 0, "TEST-BASIS-17: Col 0 at row 0");
        check(b.row_of_basic(1).value() == 1, "TEST-BASIS-17: Col 1 at row 1");
        check(b.row_of_basic(5).value() == 2, "TEST-BASIS-17: Col 5 at row 2");
    }

    // TEST-BASIS-18: Large valid index-range structural test (m = 50, n = 200)
    {
        const Dimension m = 50;
        const Dimension n = 200;
        std::vector<Index> basic_vars;
        for (Dimension i = 0; i < m; ++i) {
            basic_vars.push_back(i * 3); // 0, 3, 6, ..., 147 < 200
        }

        auto b = Basis::create(m, n, basic_vars).value();
        check(b.num_rows() == m && b.num_cols() == n, "TEST-BASIS-18: Large basis created");
        check(b.nonbasic_variables().size() == n - m, "TEST-BASIS-18: Nonbasic size is 150");

        bool bijection_ok = true;
        for (Dimension i = 0; i < m; ++i) {
            Index var = basic_vars[i];
            if (b.basic_variable(i).value() != var || b.row_of_basic(var).value() != i) {
                bijection_ok = false;
            }
        }
        check(bijection_ok, "TEST-BASIS-18: Bijective mapping verified across all 50 rows");
    }

    // TEST-BASIS-19: Square basis boundary condition m = n
    {
        // m = 4, n = 4, B = [2, 0, 3, 1]
        auto b_res = Basis::create(4, 4, {2, 0, 3, 1});
        check(b_res.is_ok(), "TEST-BASIS-19: Square basis (m=n) construction succeeds");
        const auto& b = b_res.value();
        check(b.num_rows() == 4 && b.num_cols() == 4, "TEST-BASIS-19: Dimensions m=4, n=4");
        check(b.nonbasic_variables().empty(), "TEST-BASIS-19: Nonbasic set is strictly empty for m=n");
        for (Index j = 0; j < 4; ++j) {
            check(b.is_basic(j) && !b.is_nonbasic(j), "TEST-BASIS-19: All columns are basic");
        }
    }

    // TEST-BASIS-20: Degenerate boundary condition m = 0, n = 0
    {
        auto b_res = Basis::create(0, 0, {});
        check(b_res.is_ok(), "TEST-BASIS-20: m=0, n=0 basis succeeds");
        const auto& b = b_res.value();
        check(b.num_rows() == 0 && b.num_cols() == 0, "TEST-BASIS-20: Dimensions 0x0");
        check(b.basic_variables().empty() && b.nonbasic_variables().empty(), "TEST-BASIS-20: Both sets empty");
    }

    // TEST-BASIS-21: Comprehensive state immutability on failed replacement
    {
        // m = 3, n = 5, B = [0, 2, 4], N = [1, 3]
        auto b = Basis::create(3, 5, {0, 2, 4}).value();
        const auto orig_basic = b.basic_variables();
        const auto orig_nonbasic = b.nonbasic_variables();
        const uint64_t orig_ver = b.version();

        // Attempt invalid replacement: entering variable 2 is already basic!
        auto st = b.replace_basic_variable(2, 0);
        check(!st.is_ok(), "TEST-BASIS-21: Rejects basic variable as entering");
        check(b.version() == orig_ver, "TEST-BASIS-21: Version unchanged on failure");
        check(b.basic_variables() == orig_basic, "TEST-BASIS-21: Basic list unchanged");
        check(b.nonbasic_variables() == orig_nonbasic, "TEST-BASIS-21: Nonbasic list unchanged");
        for (Index j = 0; j < 5; ++j) {
            check(b.is_basic(j) == (j == 0 || j == 2 || j == 4), "TEST-BASIS-21: Basic status unchanged");
            check(b.is_nonbasic(j) == (j == 1 || j == 3), "TEST-BASIS-21: Nonbasic status unchanged");
        }
    }

    // TEST-BASIS-22: Dynamic view behavior of BasisMatrixView after replacement
    {
        // A is 2 x 4
        // row 0: [10, 20, 30, 40]
        // row 1: [50, 60, 70, 80]
        // initial basis: B = [3, 1] (row 0 -> col 3, row 1 -> col 1)
        auto b = Basis::create(2, 4, {3, 1}).value();
        std::vector<Triplet> trips = {
            {0, 0, 10.0}, {0, 1, 20.0}, {0, 2, 30.0}, {0, 3, 40.0},
            {1, 0, 50.0}, {1, 1, 60.0}, {1, 2, 70.0}, {1, 3, 80.0}
        };
        auto A = SparseMatrix::from_triplets(2, 4, trips).value();

        BasisMatrixView view(A, b);
        check(view.original_column_index(0).value() == 3, "TEST-BASIS-22: View col 0 is col 3 initially");
        check(approx_eq(view.get(0, 0).value(), 40.0), "TEST-BASIS-22: B(0,0) is 40.0");

        // Pivot: entering col 0, leaving row 0 (so B becomes [0, 1])
        auto rep_st = b.replace_basic_variable(0, 0);
        check(rep_st.is_ok(), "TEST-BASIS-22: Pivot succeeds");

        // The view MUST dynamically reflect the updated basis without re-instantiation
        check(view.original_column_index(0).value() == 0, "TEST-BASIS-22: View col 0 dynamically updated to col 0");
        check(approx_eq(view.get(0, 0).value(), 10.0), "TEST-BASIS-22: B(0,0) dynamically updated to 10.0");
        check(view.original_column_index(1).value() == 1, "TEST-BASIS-22: View col 1 unchanged");
    }

    // TEST-BASIS-23: Zero-allocation workspace feasibility and aliasing rejection
    {
        // 2 x 3 system:
        // x_0 + 2*x_1 + 3*x_2 = 14
        // 2*x_0 + x_1 + 4*x_2 = 13
        // Basics: B = [0, 1], Nonbasic: [2]
        // Solution: x_B = [4, 5], x_2 = 0
        // Ax: 4 + 10 = 14, 8 + 5 = 13
        auto b_basis = Basis::create(2, 3, {0, 1}).value();
        std::vector<Triplet> trips = {
            {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0},
            {1, 0, 2.0}, {1, 1, 1.0}, {1, 2, 4.0}
        };
        auto A = SparseMatrix::from_triplets(2, 3, trips).value();

        auto b_vec = DenseVector::create(2).value();
        set_vec(b_vec, 0, 14.0);
        set_vec(b_vec, 1, 13.0);

        auto x_B = DenseVector::create(2).value();
        set_vec(x_B, 0, 4.0);
        set_vec(x_B, 1, 5.0);

        auto sol = BasicSolution::create(b_basis, std::move(x_B)).value();

        auto x_work = DenseVector::create(3).value();
        auto res_scratch = DenseVector::create(2).value();
        auto res_out = DenseVector::create(2).value();

        // Zero-allocation overload execution
        auto st = BasicSolution::check_primal_feasibility(A, b_vec, sol, x_work, res_scratch, res_out, 1e-7);
        check(st.is_ok(), "TEST-BASIS-23: Zero-allocation feasibility check passes");

        // Aliasing rejection: pass b_vec as residual_out
        auto alias_st = BasicSolution::check_primal_feasibility(A, b_vec, sol, x_work, res_scratch, b_vec, 1e-7);
        check(!alias_st.is_ok() && alias_st.code() == StatusCode::InvalidArgument,
              "TEST-BASIS-23: Rejects aliasing between b and residual_out");

        // Invalid tolerance rejection: negative feas_tol
        auto neg_tol_st = BasicSolution::check_primal_feasibility(A, b_vec, sol, x_work, res_scratch, res_out, -1.0);
        check(!neg_tol_st.is_ok() && neg_tol_st.code() == StatusCode::InvalidArgument,
              "TEST-BASIS-23: Rejects negative feas_tol");

        // Invalid tolerance rejection: NaN feas_tol
        auto nan_tol_st = BasicSolution::check_primal_feasibility(A, b_vec, sol, x_work, res_scratch, res_out, std::numeric_limits<Scalar>::quiet_NaN());
        check(!nan_tol_st.is_ok() && nan_tol_st.code() == StatusCode::InvalidArgument,
              "TEST-BASIS-23: Rejects NaN feas_tol");

        // Workspace size mismatch rejection: x_work size != n
        auto x_bad = DenseVector::create(4).value();
        auto bad_x_st = BasicSolution::check_primal_feasibility(A, b_vec, sol, x_bad, res_scratch, res_out, 1e-7);
        check(!bad_x_st.is_ok() && bad_x_st.code() == StatusCode::InvalidArgument,
              "TEST-BASIS-23: Rejects mismatched x_workspace dimension");
    }

    // TEST-PROP-01: Deterministic property testing with independent oracle
    {
        std::mt19937_64 rng(424242);
        const int num_trials = 30;
        int passed = 0;

        for (int trial = 0; trial < num_trials; ++trial) {
            const Dimension m = static_cast<Dimension>(3 + (rng() % 8)); // 3 to 10 rows
            const Dimension n = m + static_cast<Dimension>(2 + (rng() % 10)); // m + 2 to m + 11 cols

            // Randomly select m distinct column indices
            std::vector<Index> all_cols(n);
            for (Dimension j = 0; j < n; ++j) all_cols[j] = j;
            std::shuffle(all_cols.begin(), all_cols.end(), rng);

            std::vector<Index> basic_vars(all_cols.begin(), all_cols.begin() + m);

            // Construct independent oracle data structures
            std::unordered_set<Index> oracle_basic_set(basic_vars.begin(), basic_vars.end());
            std::unordered_set<Index> oracle_nonbasic_set;
            std::unordered_map<Index, Index> oracle_row_map;
            for (Dimension i = 0; i < m; ++i) {
                oracle_row_map[basic_vars[i]] = i;
            }
            for (Dimension j = 0; j < n; ++j) {
                if (oracle_basic_set.find(j) == oracle_basic_set.end()) {
                    oracle_nonbasic_set.insert(j);
                }
            }

            auto b_res = Basis::create(m, n, basic_vars);
            if (!b_res.is_ok()) continue;
            auto b = b_res.value();

            bool trial_ok = true;

            // 1. Check every column classified exactly once
            for (Dimension j = 0; j < n; ++j) {
                const bool is_b = b.is_basic(j);
                const bool is_nb = b.is_nonbasic(j);
                if (is_b == is_nb) trial_ok = false; // Must be complementary

                if (oracle_basic_set.count(j) > 0) {
                    if (!is_b || is_nb) trial_ok = false;
                    if (b.row_of_basic(j).value() != oracle_row_map[j]) trial_ok = false;
                } else {
                    if (is_b || !is_nb) trial_ok = false;
                }
            }

            // 2. Perform 5 random replacements and verify against updated oracle
            for (int rep = 0; rep < 5; ++rep) {
                Index leaving_row = static_cast<Index>(rng() % m);
                Index leaving_var = b.basic_variable(leaving_row).value();

                // Pick a random nonbasic variable
                const auto& nb_list = b.nonbasic_variables();
                Index entering_var = nb_list[rng() % nb_list.size()];

                // Update oracle
                oracle_basic_set.erase(leaving_var);
                oracle_basic_set.insert(entering_var);
                oracle_nonbasic_set.erase(entering_var);
                oracle_nonbasic_set.insert(leaving_var);
                oracle_row_map.erase(leaving_var);
                oracle_row_map[entering_var] = leaving_row;

                auto rep_st = b.replace_basic_variable(entering_var, leaving_row);
                if (!rep_st.is_ok()) {
                    trial_ok = false;
                    break;
                }

                // Verify against oracle
                if (b.basic_variable(leaving_row).value() != entering_var) trial_ok = false;
                if (b.row_of_basic(entering_var).value() != leaving_row) trial_ok = false;
                if (!b.is_basic(entering_var) || !b.is_nonbasic(leaving_var)) trial_ok = false;
            }

            if (trial_ok) {
                passed++;
            }
        }

        check(passed == num_trials, "TEST-PROP-01: 30/30 deterministic property tests passed");
    }

    std::cout << "=========================================================\n";
    if (failures == 0) {
        std::cout << "ALL PHASE 3B BASIS TESTS PASSED (0 failures)\n";
    } else {
        std::cerr << "FAILURES DETECTED: " << failures << " tests failed\n";
    }
    std::cout << "=========================================================\n";

    return failures;
}
