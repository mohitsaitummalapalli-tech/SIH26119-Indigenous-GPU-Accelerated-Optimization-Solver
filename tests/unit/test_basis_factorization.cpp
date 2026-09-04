#include "solver/lp/basis_factorization.hpp"
#include "numerics/sparse_matrix.hpp"
#include "numerics/dense_vector.hpp"
#include "numerics/tolerances.hpp"
#include "solver/lp/basis.hpp"
#include "solver/lp/basis_matrix_view.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <random>
#include <algorithm>
#include <new>
#include <atomic>
#include <cstdlib>

// --- Zero-Allocation Instrumentation ---
static std::atomic<int64_t> g_alloc_count{0};
static std::atomic<bool> g_track_allocs{false};

void* operator new(std::size_t size) {
    if (g_track_allocs.load(std::memory_order_relaxed)) {
        g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

namespace sih26119 {
namespace test {

// Helper: sets up a SparseMatrix, Basis, and BasisMatrixView for testing
struct TestBasisEnv {
    SparseMatrix A;
    Basis basis;
    std::unique_ptr<BasisMatrixView> view;
};

static TestBasisEnv make_test_basis_env(Dimension m, const std::vector<Scalar>& dense_col_major) {
    std::vector<Triplet> triplets;
    for (Dimension j = 0; j < m; ++j) {
        for (Dimension i = 0; i < m; ++i) {
            const Scalar val = dense_col_major[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
            if (val != 0.0) {
                triplets.push_back(Triplet{i, j, val});
            }
        }
    }
    auto A_res = SparseMatrix::from_triplets(m, m, triplets);
    assert(A_res.is_ok());

    std::vector<Index> basic_vars(m);
    for (Dimension i = 0; i < m; ++i) {
        basic_vars[i] = i;
    }
    auto basis_res = Basis::create(m, m, basic_vars);
    assert(basis_res.is_ok());

    TestBasisEnv env;
    env.A = std::move(A_res.value());
    env.basis = std::move(basis_res.value());
    env.view = std::make_unique<BasisMatrixView>(env.A, env.basis);
    return env;
}

// Helper: sets up an m x n constraint matrix A with a specified basis mapping
static TestBasisEnv make_rectangular_basis_env(
    Dimension m,
    Dimension n,
    const std::vector<Triplet>& triplets,
    const std::vector<Index>& basic_vars)
{
    auto A_res = SparseMatrix::from_triplets(m, n, triplets);
    assert(A_res.is_ok());

    auto basis_res = Basis::create(m, n, basic_vars);
    assert(basis_res.is_ok());

    TestBasisEnv env;
    env.A = std::move(A_res.value());
    env.basis = std::move(basis_res.value());
    env.view = std::make_unique<BasisMatrixView>(env.A, env.basis);
    return env;
}

// ============================================================================
// 16. INDEPENDENT COMPLETE-PIVOTING GAUSSIAN ELIMINATION ORACLE
// ============================================================================
struct CompletePivotingOracle {
    // Independent primal solve: M x = rhs using complete pivoting
    static Result<std::vector<Scalar>> solve(
        Dimension m,
        const std::vector<Scalar>& M_col_major,
        const std::vector<Scalar>& rhs,
        Scalar singularity_tol = 1e-12)
    {
        if (m == 0) {
            return std::vector<Scalar>{};
        }

        const std::size_t m_sz = static_cast<std::size_t>(m);
        // Working matrix in row-major for oracle
        std::vector<std::vector<Scalar>> A(m_sz, std::vector<Scalar>(m_sz, 0.0));
        for (std::size_t i = 0; i < m_sz; ++i) {
            for (std::size_t j = 0; j < m_sz; ++j) {
                A[i][j] = M_col_major[i + j * m_sz];
            }
        }
        std::vector<Scalar> b = rhs;
        std::vector<std::size_t> col_order(m_sz);
        for (std::size_t j = 0; j < m_sz; ++j) col_order[j] = j;

        for (std::size_t k = 0; k < m_sz; ++k) {
            // Find global maximum in active submatrix [k..m-1] x [k..m-1]
            Scalar max_val = 0.0;
            std::size_t best_row = k;
            std::size_t best_col = k;
            for (std::size_t i = k; i < m_sz; ++i) {
                for (std::size_t j = k; j < m_sz; ++j) {
                    const Scalar val = std::abs(A[i][j]);
                    if (val > max_val) {
                        max_val = val;
                        best_row = i;
                        best_col = j;
                    }
                }
            }

            if (max_val <= singularity_tol || !std::isfinite(max_val)) {
                return Status::error(StatusCode::NumericalFailure, "Oracle: singular matrix");
            }

            // Swap row k and best_row
            if (best_row != k) {
                std::swap(A[k], A[best_row]);
                std::swap(b[k], b[best_row]);
            }
            // Swap col k and best_col
            if (best_col != k) {
                for (std::size_t i = 0; i < m_sz; ++i) {
                    std::swap(A[i][k], A[i][best_col]);
                }
                std::swap(col_order[k], col_order[best_col]);
            }

            const Scalar pivot = A[k][k];
            for (std::size_t i = k + 1; i < m_sz; ++i) {
                const Scalar factor = A[i][k] / pivot;
                A[i][k] = 0.0;
                for (std::size_t j = k + 1; j < m_sz; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Backward substitution
        std::vector<Scalar> y(m_sz, 0.0);
        for (std::size_t step = m_sz; step > 0; --step) {
            const std::size_t i = step - 1;
            Scalar sum = b[i];
            for (std::size_t j = i + 1; j < m_sz; ++j) {
                sum -= A[i][j] * y[j];
            }
            y[i] = sum / A[i][i];
        }

        // Unpermute columns: x[col_order[i]] = y[i]
        std::vector<Scalar> x(m_sz, 0.0);
        for (std::size_t i = 0; i < m_sz; ++i) {
            x[col_order[i]] = y[i];
        }
        return x;
    }

    // Independent transpose solve: M^T y = rhs
    static Result<std::vector<Scalar>> solve_transpose(
        Dimension m,
        const std::vector<Scalar>& M_col_major,
        const std::vector<Scalar>& rhs,
        Scalar singularity_tol = 1e-12)
    {
        if (m == 0) {
            return std::vector<Scalar>{};
        }
        const std::size_t m_sz = static_cast<std::size_t>(m);
        // Transpose the column-major matrix
        std::vector<Scalar> M_transpose(m_sz * m_sz, 0.0);
        for (std::size_t i = 0; i < m_sz; ++i) {
            for (std::size_t j = 0; j < m_sz; ++j) {
                M_transpose[j + i * m_sz] = M_col_major[i + j * m_sz];
            }
        }
        return solve(m, M_transpose, rhs, singularity_tol);
    }

    // Independent residual: ||M x - rhs||_inf
    static Scalar compute_residual(
        Dimension m,
        const std::vector<Scalar>& M_col_major,
        const std::vector<Scalar>& x,
        const std::vector<Scalar>& rhs)
    {
        const std::size_t m_sz = static_cast<std::size_t>(m);
        Scalar max_res = 0.0;
        for (std::size_t i = 0; i < m_sz; ++i) {
            Scalar sum = 0.0;
            for (std::size_t j = 0; j < m_sz; ++j) {
                sum += M_col_major[i + j * m_sz] * x[j];
            }
            const Scalar diff = std::abs(sum - rhs[i]);
            if (diff > max_res) max_res = diff;
        }
        return max_res;
    }

    // Independent transpose residual: ||M^T y - rhs||_inf
    static Scalar compute_transpose_residual(
        Dimension m,
        const std::vector<Scalar>& M_col_major,
        const std::vector<Scalar>& y,
        const std::vector<Scalar>& rhs)
    {
        const std::size_t m_sz = static_cast<std::size_t>(m);
        Scalar max_res = 0.0;
        for (std::size_t j = 0; j < m_sz; ++j) {
            Scalar sum = 0.0;
            for (std::size_t i = 0; i < m_sz; ++i) {
                sum += M_col_major[i + j * m_sz] * y[i];
            }
            const Scalar diff = std::abs(sum - rhs[j]);
            if (diff > max_res) max_res = diff;
        }
        return max_res;
    }
};

// ============================================================================
// CONCRETE TEST MATRIX (TEST-FACT-01 to TEST-FACT-24)
// ============================================================================

void test_fact_01_0x0() {
    auto env = make_test_basis_env(0, {});
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());
    assert(fact.is_factored());
    assert(fact.dimension() == 0);

    DenseVector rhs, solution, scratch;
    status = fact.solve(rhs, solution, scratch);
    assert(status.is_ok());

    status = fact.solve_transpose(rhs, solution, scratch);
    assert(status.is_ok());
    std::cout << "  [PASS] TEST-FACT-01: 0x0 empty basis\n";
}

void test_fact_02_1x1() {
    auto env = make_test_basis_env(1, {5.0});
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());
    assert(fact.is_factored());
    assert(fact.dimension() == 1);

    auto rhs = DenseVector::create(1, 10.0).value();
    auto sol = DenseVector::create(1, 0.0).value();
    auto scr = DenseVector::create(1, 0.0).value();

    g_alloc_count.store(0);
    g_track_allocs.store(true);
    status = fact.solve(rhs, sol, scr);
    g_track_allocs.store(false);
    assert(status.is_ok());
    assert(g_alloc_count.load() == 0);
    assert(std::abs(sol[0] - 2.0) < 1e-12);

    auto rhs_t = DenseVector::create(1, 15.0).value();
    g_alloc_count.store(0);
    g_track_allocs.store(true);
    status = fact.solve_transpose(rhs_t, sol, scr);
    g_track_allocs.store(false);
    assert(status.is_ok());
    assert(g_alloc_count.load() == 0);
    assert(std::abs(sol[0] - 3.0) < 1e-12);

    std::cout << "  [PASS] TEST-FACT-02: 1x1 scalar basis\n";
}

void test_fact_03_diagonal() {
    // Diag(2, -4, 5)
    std::vector<Scalar> D = {
        2.0, 0.0, 0.0,
        0.0, -4.0, 0.0,
        0.0, 0.0, 5.0
    };
    auto env = make_test_basis_env(3, D);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 4.0; rhs[1] = -12.0; rhs[2] = 25.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    g_alloc_count.store(0);
    g_track_allocs.store(true);
    status = fact.solve(rhs, sol, scr);
    g_track_allocs.store(false);
    assert(status.is_ok());
    assert(g_alloc_count.load() == 0);
    assert(std::abs(sol[0] - 2.0) < 1e-12);
    assert(std::abs(sol[1] - 3.0) < 1e-12);
    assert(std::abs(sol[2] - 5.0) < 1e-12);

    status = fact.solve_transpose(rhs, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0] - 2.0) < 1e-12);
    assert(std::abs(sol[1] - 3.0) < 1e-12);
    assert(std::abs(sol[2] - 5.0) < 1e-12);

    std::cout << "  [PASS] TEST-FACT-03: diagonal matrix\n";
}

void test_fact_04_triangular() {
    // Upper triangular:
    // [ 2  3  1 ]
    // [ 0  4  2 ]
    // [ 0  0  5 ]
    std::vector<Scalar> U = {
        2.0, 0.0, 0.0,
        3.0, 4.0, 0.0,
        1.0, 2.0, 5.0
    };
    auto env = make_test_basis_env(3, U);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 6.0; rhs[1] = 8.0; rhs[2] = 10.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());

    auto oracle_sol = CompletePivotingOracle::solve(3, U, {6.0, 8.0, 10.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }
    std::cout << "  [PASS] TEST-FACT-04: triangular matrix\n";
}

void test_fact_05_dense_nonsingular() {
    // 3x3 dense nonsingular
    std::vector<Scalar> M = {
        2.0, 4.0, 8.0,
        1.0, 3.0, 2.0,
        3.0, 1.0, 1.0
    };
    auto env = make_test_basis_env(3, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 10.0; rhs[1] = 12.0; rhs[2] = 4.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());

    auto oracle_sol = CompletePivotingOracle::solve(3, M, {10.0, 12.0, 4.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }

    status = fact.solve_transpose(rhs, sol, scr);
    assert(status.is_ok());
    auto oracle_trans = CompletePivotingOracle::solve_transpose(3, M, {10.0, 12.0, 4.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_trans[static_cast<std::size_t>(i)]) < 1e-10);
    }
    std::cout << "  [PASS] TEST-FACT-05: dense nonsingular 3x3\n";
}

void test_fact_06_sparse_nonsingular() {
    // 5x5 tridiagonal
    const Dimension m = 5;
    std::vector<Scalar> M(m * m, 0.0);
    for (Dimension i = 0; i < m; ++i) {
        M[i + i * m] = 4.0;
        if (i > 0) M[i + (i - 1) * m] = -1.0;
        if (i + 1 < m) M[i + (i + 1) * m] = -1.0;
    }
    auto env = make_test_basis_env(m, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(m, 1.0).value();
    auto sol = DenseVector::create(m, 0.0).value();
    auto scr = DenseVector::create(m, 0.0).value();

    g_alloc_count.store(0);
    g_track_allocs.store(true);
    status = fact.solve(rhs, sol, scr);
    g_track_allocs.store(false);
    assert(status.is_ok());
    assert(g_alloc_count.load() == 0);

    auto oracle_sol = CompletePivotingOracle::solve(m, M, std::vector<Scalar>(m, 1.0)).value();
    for (Dimension i = 0; i < m; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }
    std::cout << "  [PASS] TEST-FACT-06: sparse tridiagonal 5x5\n";
}

void test_fact_07_permutation_matrix() {
    // Permutation matrix: row 0->1, row 1->2, row 2->0
    std::vector<Scalar> P = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };
    auto env = make_test_basis_env(3, P);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 10.0; rhs[1] = 20.0; rhs[2] = 30.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());
    auto oracle_sol = CompletePivotingOracle::solve(3, P, {10.0, 20.0, 30.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-12);
    }
    assert(std::abs(sol[0] - 20.0) < 1e-12);
    assert(std::abs(sol[1] - 30.0) < 1e-12);
    assert(std::abs(sol[2] - 10.0) < 1e-12);

    status = fact.solve_transpose(rhs, sol, scr);
    assert(status.is_ok());
    auto oracle_trans = CompletePivotingOracle::solve_transpose(3, P, {10.0, 20.0, 30.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_trans[static_cast<std::size_t>(i)]) < 1e-12);
    }
    assert(std::abs(sol[0] - 30.0) < 1e-12);
    assert(std::abs(sol[1] - 10.0) < 1e-12);
    assert(std::abs(sol[2] - 20.0) < 1e-12);

    std::cout << "  [PASS] TEST-FACT-07: permutation matrix\n";
}

void test_fact_08_negative_pivots() {
    // Matrix where pivots are negative
    std::vector<Scalar> M = {
        -5.0, 2.0, 0.0,
        1.0, -4.0, 1.0,
        0.0, 3.0, -6.0
    };
    auto env = make_test_basis_env(3, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 1.0).value();
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());
    auto oracle_sol = CompletePivotingOracle::solve(3, M, {1.0, 1.0, 1.0}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }
    std::cout << "  [PASS] TEST-FACT-08: negative pivots\n";
}

void test_fact_09_mixed_sign_matrix() {
    std::vector<Scalar> M = {
        10.5, -2.3, 4.1,
        -7.8, 12.0, -1.5,
        3.2, -6.4, -9.1
    };
    auto env = make_test_basis_env(3, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = -5.0; rhs[1] = 14.2; rhs[2] = 3.3;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());
    auto oracle_sol = CompletePivotingOracle::solve(3, M, {-5.0, 14.2, 3.3}).value();
    for (Dimension i = 0; i < 3; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }
    std::cout << "  [PASS] TEST-FACT-09: mixed-sign matrix\n";
}

void test_fact_10_singular() {
    // Zero row
    std::vector<Scalar> M_zero_row = {
        1.0, 0.0, 2.0,
        2.0, 0.0, 4.0,
        3.0, 0.0, 5.0
    };
    auto env1 = make_test_basis_env(3, M_zero_row);
    BasisFactorization fact1;
    auto status1 = fact1.factorize(*env1.view);
    assert(!status1.is_ok());
    assert(status1.code() == StatusCode::NumericalFailure);
    assert(!fact1.is_factored());
    assert(fact1.state() == FactorizationState::Failed);

    // Dependent columns
    std::vector<Scalar> M_dep_cols = {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        1.0, 2.0, 5.0
    };
    auto env2 = make_test_basis_env(3, M_dep_cols);
    BasisFactorization fact2;
    auto status2 = fact2.factorize(*env2.view);
    assert(!status2.is_ok());
    assert(status2.code() == StatusCode::NumericalFailure);

    std::cout << "  [PASS] TEST-FACT-10: singular matrices rejected with NumericalFailure\n";
}

void test_fact_11_nearly_singular() {
    // Pivot below singularity tolerance 1e-12
    std::vector<Scalar> M = {
        1.0, 0.0,
        0.0, 1e-14
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::NumericalFailure);
    assert(!fact.is_factored());
    std::cout << "  [PASS] TEST-FACT-11: nearly singular basis rejected with NumericalFailure\n";
}

void test_fact_12_repeated_close_pivots() {
    // Equal candidate pivots: verifies deterministic tie breaking
    std::vector<Scalar> M = {
        5.0, 5.0,
        1.0, 2.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());
    // Col 0 has M[0, 0] = 5.0 and M[1, 0] = 5.0. Tie breaker selects smallest index 0.
    assert(fact.row_permutation()[0] == 0);
    std::cout << "  [PASS] TEST-FACT-12: repeated/close pivots deterministic tie-breaking\n";
}

void test_fact_13_badly_scaled_matrix() {
    // Badly scaled entries: 1e-6 and 1e6
    std::vector<Scalar> M = {
        1e6, 0.0,
        0.0, 1e-6
    };
    FactorizationTolerances tols;
    tols.singularity_tol = 1e-8; // 1e-6 is above 1e-8
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view, tols);
    assert(status.is_ok());

    auto rhs = DenseVector::create(2, 0.0).value();
    rhs[0] = 2e6; rhs[1] = 3e-6;
    auto sol = DenseVector::create(2, 0.0).value();
    auto scr = DenseVector::create(2, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0] - 2.0) < 1e-10);
    assert(std::abs(sol[1] - 3.0) < 1e-10);

    status = fact.solve_transpose(rhs, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0] - 2.0) < 1e-10);
    assert(std::abs(sol[1] - 3.0) < 1e-10);
    std::cout << "  [PASS] TEST-FACT-13: badly scaled matrix with scale-aware residual test\n";
}

void test_fact_14_b_solve() {
    std::vector<Scalar> M = {
        3.0, 1.0, 2.0,
        2.0, 4.0, 1.0,
        1.0, 2.0, 5.0
    };
    auto env = make_test_basis_env(3, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 11.0; rhs[1] = 11.0; rhs[2] = 17.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(status.is_ok());

    // Residual verification
    Scalar res = CompletePivotingOracle::compute_residual(3, M, {sol[0], sol[1], sol[2]}, {11.0, 11.0, 17.0});
    assert(res < 1e-10);
    std::cout << "  [PASS] TEST-FACT-14: B solve (FTRAN) with residual check\n";
}

void test_fact_15_transpose_solve() {
    std::vector<Scalar> M = {
        3.0, 1.0, 2.0,
        2.0, 4.0, 1.0,
        1.0, 2.0, 5.0
    };
    auto env = make_test_basis_env(3, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(3, 0.0).value();
    rhs[0] = 7.0; rhs[1] = 13.0; rhs[2] = 16.0;
    auto sol = DenseVector::create(3, 0.0).value();
    auto scr = DenseVector::create(3, 0.0).value();

    status = fact.solve_transpose(rhs, sol, scr);
    assert(status.is_ok());

    Scalar res = CompletePivotingOracle::compute_transpose_residual(3, M, {sol[0], sol[1], sol[2]}, {7.0, 13.0, 16.0});
    assert(res < 1e-10);
    std::cout << "  [PASS] TEST-FACT-15: transpose solve (BTRAN) with residual check\n";
}

void test_fact_16_multiple_rhs() {
    std::vector<Scalar> M = {
        4.0, 1.0,
        2.0, 3.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto sol = DenseVector::create(2, 0.0).value();
    auto scr = DenseVector::create(2, 0.0).value();

    // RHS 1
    auto rhs1 = DenseVector::create(2, 0.0).value();
    rhs1[0] = 6.0; rhs1[1] = 7.0;
    status = fact.solve(rhs1, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0] - 0.4) < 1e-10); // 4*0.4 + 2*2.2 = 1.6 + 4.4 = 6.0
    assert(std::abs(sol[1] - 2.2) < 1e-10); // 1*0.4 + 3*2.2 = 0.4 + 6.6 = 7.0

    // RHS 2
    auto rhs2 = DenseVector::create(2, 0.0).value();
    rhs2[0] = 10.0; rhs2[1] = 10.0;
    status = fact.solve(rhs2, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0] - 1.0) < 1e-10);
    assert(std::abs(sol[1] - 3.0) < 1e-10);

    // RHS 3
    auto rhs3 = DenseVector::create(2, 0.0).value();
    rhs3[0] = 0.0; rhs3[1] = 0.0;
    status = fact.solve(rhs3, sol, scr);
    assert(status.is_ok());
    assert(std::abs(sol[0]) < 1e-12);
    assert(std::abs(sol[1]) < 1e-12);

    std::cout << "  [PASS] TEST-FACT-16: multiple RHS solves with same factorization\n";
}

void test_fact_17_nan() {
    std::vector<Scalar> M = {
        1.0, 0.0,
        0.0, 1.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(2, 0.0).value();
    rhs[0] = std::numeric_limits<Scalar>::quiet_NaN();
    auto sol = DenseVector::create(2, 42.0).value();
    auto scr = DenseVector::create(2, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);
    assert(sol[0] == 42.0); // Destination unmodified!

    status = fact.solve_transpose(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);
    assert(sol[0] == 42.0); // Destination unmodified!

    std::cout << "  [PASS] TEST-FACT-17: NaN in RHS rejected, destination unmodified\n";
}

void test_fact_18_inf() {
    std::vector<Scalar> M = {
        1.0, 0.0,
        0.0, 1.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto rhs = DenseVector::create(2, 0.0).value();
    rhs[1] = std::numeric_limits<Scalar>::infinity();
    auto sol = DenseVector::create(2, 99.0).value();
    auto scr = DenseVector::create(2, 0.0).value();

    status = fact.solve(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);
    assert(sol[0] == 99.0);
    assert(sol[1] == 99.0);

    status = fact.solve_transpose(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);
    assert(sol[0] == 99.0);

    std::cout << "  [PASS] TEST-FACT-18: Inf in RHS rejected, destination unmodified\n";
}

void test_fact_19_aliasing_rejection() {
    std::vector<Scalar> M = {
        2.0, 0.0,
        0.0, 3.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto v1 = DenseVector::create(2, 1.0).value();
    auto v2 = DenseVector::create(2, 0.0).value();

    // &rhs == &solution
    status = fact.solve(v1, v1, v2);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);

    // &solution == &scratch
    status = fact.solve(v1, v2, v2);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);

    // &rhs == &scratch
    status = fact.solve(v1, v2, v1);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "  [PASS] TEST-FACT-19: pairwise aliasing rejected with InvalidArgument\n";
}

void test_fact_20_stale_basis_version() {
    // Construct 2x4 constraint matrix A
    std::vector<Triplet> triplets = {
        Triplet{0, 0, 1.0}, Triplet{0, 1, 2.0}, Triplet{0, 2, 3.0}, Triplet{0, 3, 4.0},
        Triplet{1, 0, 5.0}, Triplet{1, 1, 6.0}, Triplet{1, 2, 7.0}, Triplet{1, 3, 8.0}
    };
    std::vector<Index> basic_vars = {0, 1}; // columns 0 and 1 basic
    auto env = make_rectangular_basis_env(2, 4, triplets, basic_vars);

    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());
    assert(fact.basis_version() == env.basis.version());

    auto rhs = DenseVector::create(2, 1.0).value();
    auto sol = DenseVector::create(2, 77.0).value();
    auto scr = DenseVector::create(2, 0.0).value();

    // Modify the basis! Entering col 2 leaves row 0
    status = env.basis.replace_basic_variable(2, 0);
    assert(status.is_ok());
    assert(env.basis.version() > fact.basis_version());

    // Subsequent solve MUST be rejected with InconsistentModel
    status = fact.solve(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InconsistentModel);
    assert(sol[0] == 77.0); // Destination unchanged!

    status = fact.solve_transpose(rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InconsistentModel);
    assert(sol[0] == 77.0); // Destination unchanged!

    std::cout << "  [PASS] TEST-FACT-20: stale basis version rejected with InconsistentModel\n";
}

void test_fact_21_solution_unchanged_on_failure() {
    std::vector<Scalar> M = {
        2.0, 0.0,
        0.0, 2.0
    };
    auto env = make_test_basis_env(2, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto sol = DenseVector::create(2, 12345.67).value();
    auto scr = DenseVector::create(2, 0.0).value();

    // 1. Dimension mismatch
    auto bad_rhs = DenseVector::create(3, 1.0).value();
    status = fact.solve(bad_rhs, sol, scr);
    assert(!status.is_ok());
    assert(sol[0] == 12345.67 && sol[1] == 12345.67);

    // 2. Insufficient scratch
    auto good_rhs = DenseVector::create(2, 1.0).value();
    auto small_scr = DenseVector::create(1, 0.0).value();
    status = fact.solve(good_rhs, sol, small_scr);
    assert(!status.is_ok());
    assert(sol[0] == 12345.67 && sol[1] == 12345.67);

    // 3. Unfactored object
    BasisFactorization unfactored;
    status = unfactored.solve(good_rhs, sol, scr);
    assert(!status.is_ok());
    assert(status.code() == StatusCode::InconsistentModel);
    assert(sol[0] == 12345.67 && sol[1] == 12345.67);

    std::cout << "  [PASS] TEST-FACT-21: transactional guarantee: solution unchanged on failure\n";
}

void test_fact_22_factorization_residual() {
    // 4x4 matrix
    std::vector<Scalar> M = {
        5.0, 2.0, 1.0, 0.5,
        1.0, 6.0, 2.0, 1.0,
        0.5, 1.5, 4.0, 0.8,
        0.2, 0.3, 0.4, 3.0
    };
    auto env = make_test_basis_env(4, M);
    BasisFactorization fact;
    auto status = fact.factorize(*env.view);
    assert(status.is_ok());

    auto res = fact.compute_factorization_residual();
    assert(res.is_ok());
    assert(res.value() < 1e-12);
    std::cout << "  [PASS] TEST-FACT-22: scale-aware factorization check ||PB - LU|| passed (" << res.value() << ")\n";
}

void test_fact_23_deterministic_pivoting() {
    std::vector<Scalar> M = {
        1.0, 4.0, 2.0,
        3.0, 2.0, 1.0,
        2.0, 1.0, 5.0
    };
    auto env1 = make_test_basis_env(3, M);
    BasisFactorization fact1;
    assert(fact1.factorize(*env1.view).is_ok());

    auto env2 = make_test_basis_env(3, M);
    BasisFactorization fact2;
    assert(fact2.factorize(*env2.view).is_ok());

    assert(fact1.row_permutation() == fact2.row_permutation());
    assert(fact1.row_permutation_inv() == fact2.row_permutation_inv());
    for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 3; ++j) {
            assert(fact1.lu_entry(i, j) == fact2.lu_entry(i, j));
        }
    }
    std::cout << "  [PASS] TEST-FACT-23: deterministic pivoting repeatability across runs\n";
}

void test_fact_24_independent_oracle_agreement() {
    // Compare BasisFactorization against CompletePivotingOracle for 4x4 matrix
    std::vector<Scalar> M = {
        4.0, 2.0, 1.0, 0.5,
        1.0, 5.0, 3.0, 1.0,
        0.5, 2.0, 6.0, 2.0,
        0.2, 1.0, 1.5, 7.0
    };
    auto env = make_test_basis_env(4, M);
    BasisFactorization fact;
    assert(fact.factorize(*env.view).is_ok());

    std::vector<Scalar> rhs_data = {2.5, -4.0, 8.2, 1.1};
    auto rhs = DenseVector::create(4, 0.0).value();
    for (Dimension i = 0; i < 4; ++i) rhs[i] = rhs_data[static_cast<std::size_t>(i)];

    auto sol = DenseVector::create(4, 0.0).value();
    auto scr = DenseVector::create(4, 0.0).value();

    // Primal solve
    assert(fact.solve(rhs, sol, scr).is_ok());
    auto oracle_sol = CompletePivotingOracle::solve(4, M, rhs_data).value();
    for (Dimension i = 0; i < 4; ++i) {
        assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-10);
    }

    // Transpose solve
    assert(fact.solve_transpose(rhs, sol, scr).is_ok());
    auto oracle_trans = CompletePivotingOracle::solve_transpose(4, M, rhs_data).value();
    for (Dimension i = 0; i < 4; ++i) {
        assert(std::abs(sol[i] - oracle_trans[static_cast<std::size_t>(i)]) < 1e-10);
    }

    std::cout << "  [PASS] TEST-FACT-24: independent oracle agreement\n";
}

// ============================================================================
// 18. DETERMINISTIC PROPERTY TESTS (Fixed seed: 0x3C3C3C)
// ============================================================================
void run_property_tests() {
    std::mt19937_64 rng(0x3C3C3C);
    std::uniform_real_distribution<Scalar> dist_val(-10.0, 10.0);
    std::uniform_real_distribution<Scalar> dist_scale(0.1, 5.0);

    const std::vector<Dimension> dims = {2, 3, 4, 5, 8};
    int passed_cases = 0;

    for (Dimension m : dims) {
        for (int trial = 0; trial < 10; ++trial) {
            const std::size_t m_sz = static_cast<std::size_t>(m);
            std::vector<Scalar> M(m_sz * m_sz, 0.0);

            // Generate mixed-magnitude nonsingular matrix (not purely diagonally dominant)
            for (std::size_t i = 0; i < m_sz; ++i) {
                for (std::size_t j = 0; j < m_sz; ++j) {
                    M[i + j * m_sz] = dist_val(rng) * dist_scale(rng);
                }
                // Add non-zero offset to avoid exact singularity
                M[i + i * m_sz] += ((i % 2 == 0) ? 5.0 : -5.0);
            }

            auto env = make_test_basis_env(m, M);
            BasisFactorization fact;
            auto fact_status = fact.factorize(*env.view);
            if (!fact_status.is_ok()) {
                // If matrix happened to be singular, oracle must also detect singularity
                auto oracle_check = CompletePivotingOracle::solve(m, M, std::vector<Scalar>(m_sz, 1.0));
                assert(!oracle_check.is_ok());
                continue;
            }

            // Generate random RHS
            std::vector<Scalar> rhs_raw(m_sz);
            auto rhs = DenseVector::create(m, 0.0).value();
            for (std::size_t i = 0; i < m_sz; ++i) {
                rhs_raw[i] = dist_val(rng);
                rhs[static_cast<Index>(i)] = rhs_raw[i];
            }

            auto sol = DenseVector::create(m, 0.0).value();
            auto scr = DenseVector::create(m, 0.0).value();

            // Zero-allocation instrumented solve (FTRAN)
            g_alloc_count.store(0);
            g_track_allocs.store(true);
            auto solve_status = fact.solve(rhs, sol, scr);
            g_track_allocs.store(false);
            assert(solve_status.is_ok());
            assert(g_alloc_count.load() == 0); // ZERO ALLOCATION CONTRACT

            auto oracle_sol = CompletePivotingOracle::solve(m, M, rhs_raw).value();
            for (Dimension i = 0; i < m; ++i) {
                assert(std::abs(sol[i] - oracle_sol[static_cast<std::size_t>(i)]) < 1e-8);
            }

            // Zero-allocation instrumented transpose solve (BTRAN)
            g_alloc_count.store(0);
            g_track_allocs.store(true);
            auto trans_status = fact.solve_transpose(rhs, sol, scr);
            g_track_allocs.store(false);
            assert(trans_status.is_ok());
            assert(g_alloc_count.load() == 0); // ZERO ALLOCATION CONTRACT

            auto oracle_trans = CompletePivotingOracle::solve_transpose(m, M, rhs_raw).value();
            for (Dimension i = 0; i < m; ++i) {
                assert(std::abs(sol[i] - oracle_trans[static_cast<std::size_t>(i)]) < 1e-8);
            }

            // Factorization residual check
            auto fact_res = fact.compute_factorization_residual();
            assert(fact_res.is_ok());

            passed_cases++;
        }
    }
    std::cout << "  [PASS] Property Tests: " << passed_cases << " randomized cases passed (Fixed seed 0x3C3C3C)\n";
}

} // namespace test
} // namespace sih26119

int main() {
    std::cout << "Running Phase 3C Basis Factorization Tests...\n";

    sih26119::test::test_fact_01_0x0();
    sih26119::test::test_fact_02_1x1();
    sih26119::test::test_fact_03_diagonal();
    sih26119::test::test_fact_04_triangular();
    sih26119::test::test_fact_05_dense_nonsingular();
    sih26119::test::test_fact_06_sparse_nonsingular();
    sih26119::test::test_fact_07_permutation_matrix();
    sih26119::test::test_fact_08_negative_pivots();
    sih26119::test::test_fact_09_mixed_sign_matrix();
    sih26119::test::test_fact_10_singular();
    sih26119::test::test_fact_11_nearly_singular();
    sih26119::test::test_fact_12_repeated_close_pivots();
    sih26119::test::test_fact_13_badly_scaled_matrix();
    sih26119::test::test_fact_14_b_solve();
    sih26119::test::test_fact_15_transpose_solve();
    sih26119::test::test_fact_16_multiple_rhs();
    sih26119::test::test_fact_17_nan();
    sih26119::test::test_fact_18_inf();
    sih26119::test::test_fact_19_aliasing_rejection();
    sih26119::test::test_fact_20_stale_basis_version();
    sih26119::test::test_fact_21_solution_unchanged_on_failure();
    sih26119::test::test_fact_22_factorization_residual();
    sih26119::test::test_fact_23_deterministic_pivoting();
    sih26119::test::test_fact_24_independent_oracle_agreement();

    sih26119::test::run_property_tests();

    std::cout << "All Phase 3C Basis Factorization tests passed successfully!\n";
    return 0;
}
