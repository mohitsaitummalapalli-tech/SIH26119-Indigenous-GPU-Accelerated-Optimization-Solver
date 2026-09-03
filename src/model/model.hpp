#pragma once

#include "core/status.hpp"
#include "core/result.hpp"
#include "model/types.hpp"
#include "model/variable.hpp"
#include "model/constraint.hpp"
#include "model/objective.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace sih26119 {

class Model {
public:
    explicit Model(std::string name = "") : name_(std::move(name)) {}

    // Model identification
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    // Variable management
    Result<VariableIndex> add_variable(
        std::string name,
        double lower_bound = 0.0,
        double upper_bound = kInfinity,
        VariableType type = VariableType::Continuous);

    [[nodiscard]] DimensionCount num_variables() const noexcept {
        return static_cast<DimensionCount>(variables_.size());
    }

    [[nodiscard]] const Variable& get_variable(VariableIndex index) const;
    [[nodiscard]] Variable& get_variable(VariableIndex index);
    [[nodiscard]] const std::vector<Variable>& variables() const noexcept { return variables_; }
    [[nodiscard]] Result<VariableIndex> get_variable_index(const std::string& name) const;
    [[nodiscard]] bool has_variable(const std::string& name) const noexcept;

    Status set_variable_bounds(VariableIndex index, double lb, double ub);
    Status set_variable_type(VariableIndex index, VariableType type);

    // Constraint management
    Result<ConstraintIndex> add_constraint(
        std::string name,
        double lower_bound = -kInfinity,
        double upper_bound = kInfinity,
        std::vector<LinearTerm> terms = {});

    [[nodiscard]] DimensionCount num_constraints() const noexcept {
        return static_cast<DimensionCount>(constraints_.size());
    }

    [[nodiscard]] const Constraint& get_constraint(ConstraintIndex index) const;
    [[nodiscard]] Constraint& get_constraint(ConstraintIndex index);
    [[nodiscard]] const std::vector<Constraint>& constraints() const noexcept { return constraints_; }
    [[nodiscard]] Result<ConstraintIndex> get_constraint_index(const std::string& name) const;
    [[nodiscard]] bool has_constraint(const std::string& name) const noexcept;

    Status add_constraint_term(ConstraintIndex con_idx, VariableIndex var_idx, double coeff);
    Status set_constraint_bounds(ConstraintIndex con_idx, double lb, double ub);

    // Objective management
    [[nodiscard]] const Objective& objective() const noexcept { return objective_; }
    [[nodiscard]] Objective& objective() noexcept { return objective_; }
    void set_objective_sense(ObjectiveSense sense) noexcept { objective_.sense = sense; }
    void set_objective_offset(double offset) noexcept { objective_.offset = offset; }

    Status set_objective_coefficient(VariableIndex var_idx, double coeff);
    Status add_objective_term(VariableIndex var_idx, double coeff);

    /**
     * @brief Sets or updates a canonical quadratic objective matrix entry Q_ij.
     *
     * Canonical objective: f(x) = c^T x + 1/2 x^T Q x + c0 with symmetric Q.
     * For diagonal: coefficient = Q_ii.
     * For off-diagonal: coefficient = Q_ij = Q_ji.
     */
    Status set_quadratic_coefficient(VariableIndex var1, VariableIndex var2, double coeff);
    Status add_quadratic_coefficient(VariableIndex var1, VariableIndex var2, double coeff);

    // Classification queries
    [[nodiscard]] bool is_lp() const noexcept;
    [[nodiscard]] bool is_milp() const noexcept;
    [[nodiscard]] bool is_qp() const noexcept;
    [[nodiscard]] NonzeroCount num_nonzeros() const noexcept;
    [[nodiscard]] NonzeroCount num_quadratic_nonzeros() const noexcept;

    // Mathematical validation
    [[nodiscard]] Status validate() const;

    // Semantic comparison against another model with numerical tolerances
    [[nodiscard]] bool semantic_equals(const Model& other, double abs_tol = kAbsTol, double rel_tol = kRelTol) const;

    void clear();

private:
    std::string name_;
    std::vector<Variable> variables_;
    std::vector<Constraint> constraints_;
    Objective objective_;

    std::unordered_map<std::string, VariableIndex> name_to_var_;
    std::unordered_map<std::string, ConstraintIndex> name_to_con_;

    // Fast map for canonical quadratic terms: (var1, var2) -> index in objective_.quadratic_terms
    std::map<std::pair<VariableIndex, VariableIndex>, size_t> quad_index_map_;
};

} // namespace sih26119
