#pragma once

#include "core/result.hpp"
#include "model/model.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include "numerics/dense_vector.hpp"
#include "numerics/sparse_matrix.hpp"
#include "solver/lp/lp_types.hpp"
#include <vector>

namespace sih26119 {

/**
 * @brief Standardized Linear Program representation in standard equality form:
 *
 *     minimize    c_bar^T x_bar + c0_bar
 *     subject to  A_bar * x_bar = b_bar
 *                 x_bar >= 0
 *                 b_bar >= 0
 *
 * Encapsulates full bidirectional mapping metadata to reconstruct original
 * Phase 1 model variables x and original objective values.
 *
 * Note: Phase 3A performs mathematical standardization only. It does not solve LPs.
 */
class StandardizedLp {
public:
    StandardizedLp() = default;

    /// Factory method: standardizes a canonical Phase 1 continuous LP Model.
    /// Returns StatusCode::InvalidArgument on invalid models or non-finite inputs.
    [[nodiscard]] static Result<StandardizedLp> standardize(const Model& model);

    // Standard form linear algebra components
    [[nodiscard]] const SparseMatrix& A() const noexcept { return A_bar_; }
    [[nodiscard]] const DenseVector& b() const noexcept { return b_bar_; }
    [[nodiscard]] const DenseVector& c() const noexcept { return c_bar_; }
    [[nodiscard]] Scalar c0() const noexcept { return c0_bar_; }

    // Original model metadata
    [[nodiscard]] ObjectiveSense original_sense() const noexcept { return original_sense_; }
    [[nodiscard]] Scalar original_c0() const noexcept { return original_c0_; }
    [[nodiscard]] Dimension original_variables() const noexcept { return num_original_vars_; }
    [[nodiscard]] Dimension original_constraints() const noexcept { return num_original_cons_; }
    [[nodiscard]] NonzeroCount original_nonzeros() const noexcept { return original_nnz_; }

    // Standardized dimension metrics
    [[nodiscard]] Dimension standardized_variables() const noexcept { return A_bar_.cols(); }
    [[nodiscard]] Dimension standardized_constraints() const noexcept { return A_bar_.rows(); }
    [[nodiscard]] NonzeroCount standardized_nonzeros() const noexcept { return A_bar_.nnz(); }

    // Mapping metadata
    [[nodiscard]] const std::vector<VariableMapping>& variable_mappings() const noexcept { return var_mappings_; }
    [[nodiscard]] const std::vector<ConstraintMapping>& constraint_mappings() const noexcept { return con_mappings_; }

    /**
     * @brief Reconstructs original variable vector x in R^n from standardized vector x_bar in R^m_bar.
     *
     * Precondition: x_bar.size() == standardized_variables().
     */
    [[nodiscard]] Result<DenseVector> reconstruct_primal(const DenseVector& x_bar) const;

    /**
     * @brief Projects a feasible original variable vector x in R^n into standard vector x_bar in R^m_bar.
     *
     * Computes all structural transformed variables and auxiliary slack/surplus values.
     * Precondition: x_orig.size() == original_variables().
     */
    [[nodiscard]] Result<DenseVector> project_primal(const DenseVector& x_orig) const;

    /**
     * @brief Evaluates the standard minimization objective: z = c_bar^T x_bar + c0_bar.
     */
    [[nodiscard]] Result<Scalar> evaluate_standard_objective(const DenseVector& x_bar) const;

    /**
     * @brief Evaluates the original objective value in original problem sense.
     */
    [[nodiscard]] Result<Scalar> evaluate_original_objective(const DenseVector& x_orig) const;

private:
    SparseMatrix A_bar_;
    DenseVector b_bar_;
    DenseVector c_bar_;
    Scalar c0_bar_{0.0};

    ObjectiveSense original_sense_{ObjectiveSense::Minimize};
    Scalar original_c0_{0.0};
    Dimension num_original_vars_{0};
    Dimension num_original_cons_{0};
    NonzeroCount original_nnz_{0};

    std::vector<VariableMapping> var_mappings_;
    std::vector<ConstraintMapping> con_mappings_;
};

} // namespace sih26119
