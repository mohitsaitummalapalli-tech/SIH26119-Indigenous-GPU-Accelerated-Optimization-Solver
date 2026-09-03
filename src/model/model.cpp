#include "model/model.hpp"
#include <cmath>
#include <algorithm>

namespace sih26119 {

namespace {

bool float_equals(double a, double b, double abs_tol, double rel_tol) noexcept {
    if (std::isinf(a) && std::isinf(b)) {
        return (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0);
    }
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    double diff = std::abs(a - b);
    double max_mag = std::max({1.0, std::abs(a), std::abs(b)});
    return diff <= (abs_tol + rel_tol * max_mag);
}

bool is_valid_bound_value(double val) noexcept {
    return !std::isnan(val);
}

} // anonymous namespace

Result<VariableIndex> Model::add_variable(
    std::string name,
    double lower_bound,
    double upper_bound,
    VariableType type)
{
    if (name.empty()) {
        return Status::error(StatusCode::InvalidArgument, "Variable name cannot be empty");
    }
    if (name_to_var_.find(name) != name_to_var_.end()) {
        return Status::error(StatusCode::DuplicateVariableName, "Duplicate variable name: " + name);
    }
    if (!is_valid_bound_value(lower_bound) || !is_valid_bound_value(upper_bound)) {
        return Status::error(StatusCode::InvalidArgument, "Variable bounds cannot be NaN for variable: " + name);
    }
    if (lower_bound > upper_bound) {
        return Status::error(StatusCode::InvalidBounds, "Lower bound exceeds upper bound for variable: " + name);
    }
    if (std::isinf(lower_bound) && lower_bound > 0.0) {
        return Status::error(StatusCode::InvalidBounds, "Lower bound cannot be +infinity for variable: " + name);
    }
    if (std::isinf(upper_bound) && upper_bound < 0.0) {
        return Status::error(StatusCode::InvalidBounds, "Upper bound cannot be -infinity for variable: " + name);
    }

    if (type == VariableType::Binary) {
        if (lower_bound < 0.0 || upper_bound > 1.0) {
            return Status::error(StatusCode::InvalidBinaryDeclaration,
                "Binary variable bounds must be within [0, 1] for variable: " + name);
        }
        double int_part;
        if (std::modf(lower_bound, &int_part) != 0.0 || std::modf(upper_bound, &int_part) != 0.0) {
            return Status::error(StatusCode::InvalidBinaryDeclaration,
                "Binary variable bounds cannot be fractional for variable: " + name);
        }
    }

    auto var_idx = static_cast<VariableIndex>(variables_.size());
    variables_.emplace_back(name, lower_bound, upper_bound, type);
    name_to_var_.emplace(std::move(name), var_idx);
    return var_idx;
}

const Variable& Model::get_variable(VariableIndex index) const {
    if (index >= variables_.size()) {
        throw std::out_of_range("Variable index out of range: " + std::to_string(index));
    }
    return variables_[index];
}

Variable& Model::get_variable(VariableIndex index) {
    if (index >= variables_.size()) {
        throw std::out_of_range("Variable index out of range: " + std::to_string(index));
    }
    return variables_[index];
}

Result<VariableIndex> Model::get_variable_index(const std::string& name) const {
    auto it = name_to_var_.find(name);
    if (it == name_to_var_.end()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable not found: " + name);
    }
    return it->second;
}

bool Model::has_variable(const std::string& name) const noexcept {
    return name_to_var_.find(name) != name_to_var_.end();
}

Status Model::set_variable_bounds(VariableIndex index, double lb, double ub) {
    if (index >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable index out of range");
    }
    if (!is_valid_bound_value(lb) || !is_valid_bound_value(ub)) {
        return Status::error(StatusCode::InvalidArgument, "Variable bounds cannot be NaN");
    }
    if (lb > ub) {
        return Status::error(StatusCode::InvalidBounds, "Lower bound exceeds upper bound");
    }
    if (variables_[index].type == VariableType::Binary) {
        if (lb < 0.0 || ub > 1.0) {
            return Status::error(StatusCode::InvalidBinaryDeclaration, "Binary variable bounds must be within [0, 1]");
        }
    }
    variables_[index].lower_bound = lb;
    variables_[index].upper_bound = ub;
    return Status::ok();
}

Status Model::set_variable_type(VariableIndex index, VariableType type) {
    if (index >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable index out of range");
    }
    if (type == VariableType::Binary) {
        if (variables_[index].lower_bound < 0.0 || variables_[index].upper_bound > 1.0) {
            return Status::error(StatusCode::InvalidBinaryDeclaration, "Existing bounds incompatible with Binary type");
        }
    }
    variables_[index].type = type;
    return Status::ok();
}

Result<ConstraintIndex> Model::add_constraint(
    std::string name,
    double lower_bound,
    double upper_bound,
    std::vector<LinearTerm> terms)
{
    if (name.empty()) {
        return Status::error(StatusCode::InvalidArgument, "Constraint name cannot be empty");
    }
    if (name_to_con_.find(name) != name_to_con_.end()) {
        return Status::error(StatusCode::DuplicateConstraintName, "Duplicate constraint name: " + name);
    }
    if (!is_valid_bound_value(lower_bound) || !is_valid_bound_value(upper_bound)) {
        return Status::error(StatusCode::InvalidArgument, "Constraint bounds cannot be NaN for constraint: " + name);
    }
    if (lower_bound > upper_bound) {
        return Status::error(StatusCode::InvalidBounds, "Constraint lower bound exceeds upper bound: " + name);
    }
    if (std::isinf(lower_bound) && lower_bound > 0.0) {
        return Status::error(StatusCode::InvalidBounds, "Lower bound cannot be +infinity for constraint: " + name);
    }
    if (std::isinf(upper_bound) && upper_bound < 0.0) {
        return Status::error(StatusCode::InvalidBounds, "Upper bound cannot be -infinity for constraint: " + name);
    }

    for (const auto& term : terms) {
        if (term.variable_index >= variables_.size()) {
            return Status::error(StatusCode::InvalidVariableReference,
                "Constraint references invalid variable index: " + std::to_string(term.variable_index));
        }
        if (std::isnan(term.coefficient)) {
            return Status::error(StatusCode::InvalidArgument, "Constraint coefficient cannot be NaN");
        }
    }

    auto con_idx = static_cast<ConstraintIndex>(constraints_.size());
    constraints_.emplace_back(name, lower_bound, upper_bound, std::move(terms));
    name_to_con_.emplace(std::move(name), con_idx);
    return con_idx;
}

const Constraint& Model::get_constraint(ConstraintIndex index) const {
    if (index >= constraints_.size()) {
        throw std::out_of_range("Constraint index out of range: " + std::to_string(index));
    }
    return constraints_[index];
}

Constraint& Model::get_constraint(ConstraintIndex index) {
    if (index >= constraints_.size()) {
        throw std::out_of_range("Constraint index out of range: " + std::to_string(index));
    }
    return constraints_[index];
}

Result<ConstraintIndex> Model::get_constraint_index(const std::string& name) const {
    auto it = name_to_con_.find(name);
    if (it == name_to_con_.end()) {
        return Status::error(StatusCode::InvalidConstraintReference, "Constraint not found: " + name);
    }
    return it->second;
}

bool Model::has_constraint(const std::string& name) const noexcept {
    return name_to_con_.find(name) != name_to_con_.end();
}

Status Model::add_constraint_term(ConstraintIndex con_idx, VariableIndex var_idx, double coeff) {
    if (con_idx >= constraints_.size()) {
        return Status::error(StatusCode::InvalidConstraintReference, "Constraint index out of range");
    }
    if (var_idx >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable index out of range");
    }
    if (std::isnan(coeff)) {
        return Status::error(StatusCode::InvalidArgument, "Constraint coefficient cannot be NaN");
    }

    auto& terms = constraints_[con_idx].terms;
    for (auto& term : terms) {
        if (term.variable_index == var_idx) {
            term.coefficient += coeff;
            return Status::ok();
        }
    }
    terms.emplace_back(var_idx, coeff);
    return Status::ok();
}

Status Model::set_constraint_bounds(ConstraintIndex con_idx, double lb, double ub) {
    if (con_idx >= constraints_.size()) {
        return Status::error(StatusCode::InvalidConstraintReference, "Constraint index out of range");
    }
    if (!is_valid_bound_value(lb) || !is_valid_bound_value(ub)) {
        return Status::error(StatusCode::InvalidArgument, "Constraint bounds cannot be NaN");
    }
    if (lb > ub) {
        return Status::error(StatusCode::InvalidBounds, "Lower bound exceeds upper bound");
    }
    constraints_[con_idx].lower_bound = lb;
    constraints_[con_idx].upper_bound = ub;
    return Status::ok();
}

Status Model::set_objective_offset(double offset) {
    if (std::isnan(offset) || std::isinf(offset)) {
        return Status::error(StatusCode::InvalidArgument, "Objective offset cannot be NaN or Infinite");
    }
    objective_.offset = offset;
    return Status::ok();
}

Status Model::set_objective_coefficient(VariableIndex var_idx, double coeff) {
    if (var_idx >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable index out of range");
    }
    if (std::isnan(coeff)) {
        return Status::error(StatusCode::InvalidArgument, "Objective coefficient cannot be NaN");
    }

    for (auto& term : objective_.linear_terms) {
        if (term.variable_index == var_idx) {
            term.coefficient = coeff;
            return Status::ok();
        }
    }
    objective_.linear_terms.emplace_back(var_idx, coeff);
    return Status::ok();
}

Status Model::add_objective_term(VariableIndex var_idx, double coeff) {
    if (var_idx >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Variable index out of range");
    }
    if (std::isnan(coeff)) {
        return Status::error(StatusCode::InvalidArgument, "Objective coefficient cannot be NaN");
    }

    for (auto& term : objective_.linear_terms) {
        if (term.variable_index == var_idx) {
            term.coefficient += coeff;
            return Status::ok();
        }
    }
    objective_.linear_terms.emplace_back(var_idx, coeff);
    return Status::ok();
}

Status Model::set_quadratic_coefficient(VariableIndex var1, VariableIndex var2, double coeff) {
    if (var1 >= variables_.size() || var2 >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Quadratic term references invalid variable index");
    }
    if (std::isnan(coeff)) {
        return Status::error(StatusCode::InvalidArgument, "Quadratic coefficient cannot be NaN");
    }

    VariableIndex v1 = std::min(var1, var2);
    VariableIndex v2 = std::max(var1, var2);
    auto key = std::make_pair(v1, v2);

    auto it = quad_index_map_.find(key);
    if (it != quad_index_map_.end()) {
        objective_.quadratic_terms[it->second].coefficient = coeff;
    } else {
        size_t idx = objective_.quadratic_terms.size();
        objective_.quadratic_terms.emplace_back(v1, v2, coeff);
        quad_index_map_[key] = idx;
    }
    return Status::ok();
}

Status Model::add_quadratic_coefficient(VariableIndex var1, VariableIndex var2, double coeff) {
    if (var1 >= variables_.size() || var2 >= variables_.size()) {
        return Status::error(StatusCode::InvalidVariableReference, "Quadratic term references invalid variable index");
    }
    if (std::isnan(coeff)) {
        return Status::error(StatusCode::InvalidArgument, "Quadratic coefficient cannot be NaN");
    }

    VariableIndex v1 = std::min(var1, var2);
    VariableIndex v2 = std::max(var1, var2);
    auto key = std::make_pair(v1, v2);

    auto it = quad_index_map_.find(key);
    if (it != quad_index_map_.end()) {
        objective_.quadratic_terms[it->second].coefficient += coeff;
    } else {
        size_t idx = objective_.quadratic_terms.size();
        objective_.quadratic_terms.emplace_back(v1, v2, coeff);
        quad_index_map_[key] = idx;
    }
    return Status::ok();
}

bool Model::is_lp() const noexcept {
    return !is_qp() && !is_milp();
}

bool Model::is_milp() const noexcept {
    for (const auto& var : variables_) {
        if (var.is_integer()) {
            return true;
        }
    }
    return false;
}

bool Model::is_qp() const noexcept {
    return objective_.is_quadratic();
}

NonzeroCount Model::num_nonzeros() const noexcept {
    NonzeroCount count = 0;
    for (const auto& con : constraints_) {
        count += con.terms.size();
    }
    return count;
}

NonzeroCount Model::num_quadratic_nonzeros() const noexcept {
    return objective_.quadratic_terms.size();
}

Status Model::validate() const {
    // 1. Verify Variable Invariants
    for (size_t j = 0; j < variables_.size(); ++j) {
        const auto& var = variables_[j];
        if (var.name.empty()) {
            return Status::error(StatusCode::InvalidArgument, "Empty variable name at index " + std::to_string(j));
        }
        if (var.lower_bound > var.upper_bound) {
            return Status::error(StatusCode::InvalidBounds, "Invalid bounds for variable " + var.name);
        }
        if (var.type == VariableType::Binary) {
            if (var.lower_bound < 0.0 || var.upper_bound > 1.0) {
                return Status::error(StatusCode::InvalidBinaryDeclaration,
                    "Binary variable bounds outside [0, 1] for variable " + var.name);
            }
        }
    }

    // 2. Verify Constraint Invariants
    for (size_t i = 0; i < constraints_.size(); ++i) {
        const auto& con = constraints_[i];
        if (con.name.empty()) {
            return Status::error(StatusCode::InvalidArgument, "Empty constraint name at index " + std::to_string(i));
        }
        if (con.lower_bound > con.upper_bound) {
            return Status::error(StatusCode::InvalidBounds, "Invalid bounds for constraint " + con.name);
        }
        for (const auto& term : con.terms) {
            if (term.variable_index >= variables_.size()) {
                return Status::error(StatusCode::InvalidVariableReference,
                    "Constraint " + con.name + " references invalid variable index " + std::to_string(term.variable_index));
            }
            if (std::isnan(term.coefficient)) {
                return Status::error(StatusCode::InvalidArgument, "NaN coefficient in constraint " + con.name);
            }
        }
    }

    // 3. Verify Objective Invariants
    for (const auto& term : objective_.linear_terms) {
        if (term.variable_index >= variables_.size()) {
            return Status::error(StatusCode::InvalidVariableReference,
                "Objective references invalid variable index " + std::to_string(term.variable_index));
        }
        if (std::isnan(term.coefficient)) {
            return Status::error(StatusCode::InvalidArgument, "NaN linear objective coefficient");
        }
    }
    for (const auto& qterm : objective_.quadratic_terms) {
        if (qterm.var1 >= variables_.size() || qterm.var2 >= variables_.size()) {
            return Status::error(StatusCode::InvalidVariableReference, "Quadratic objective references invalid variable index");
        }
        if (qterm.var1 > qterm.var2) {
            return Status::error(StatusCode::MalformedQuadraticTerm, "Quadratic term violates canonical index ordering var1 <= var2");
        }
        if (std::isnan(qterm.coefficient)) {
            return Status::error(StatusCode::InvalidArgument, "NaN quadratic objective coefficient");
        }
    }

    return Status::ok();
}

bool Model::semantic_equals(const Model& other, double abs_tol, double rel_tol) const {
    if (objective_.sense != other.objective_.sense) {
        return false;
    }
    if (!float_equals(objective_.offset, other.objective_.offset, abs_tol, rel_tol)) {
        return false;
    }
    if (variables_.size() != other.variables_.size()) {
        return false;
    }
    if (constraints_.size() != other.constraints_.size()) {
        return false;
    }

    // Compare variables by name
    for (const auto& [name, var_idx] : name_to_var_) {
        auto other_it = other.name_to_var_.find(name);
        if (other_it == other.name_to_var_.end()) {
            return false;
        }
        const auto& v1 = variables_[var_idx];
        const auto& v2 = other.variables_[other_it->second];
        if (v1.type != v2.type) {
            return false;
        }
        if (!float_equals(v1.lower_bound, v2.lower_bound, abs_tol, rel_tol) ||
            !float_equals(v1.upper_bound, v2.upper_bound, abs_tol, rel_tol)) {
            return false;
        }
    }

    // Compare linear objective coefficients
    std::map<std::string, double> obj1, obj2;
    for (const auto& term : objective_.linear_terms) {
        if (term.variable_index < variables_.size()) {
            obj1[variables_[term.variable_index].name] += term.coefficient;
        }
    }
    for (const auto& term : other.objective_.linear_terms) {
        if (term.variable_index < other.variables_.size()) {
            obj2[other.variables_[term.variable_index].name] += term.coefficient;
        }
    }
    for (const auto& [name, coeff] : obj1) {
        double c2 = obj2[name];
        if (!float_equals(coeff, c2, abs_tol, rel_tol)) {
            return false;
        }
    }
    for (const auto& [name, coeff] : obj2) {
        double c1 = obj1[name];
        if (!float_equals(c1, coeff, abs_tol, rel_tol)) {
            return false;
        }
    }

    if (!float_equals(objective_.offset, other.objective_.offset, abs_tol, rel_tol)) {
        return false;
    }

    // Compare quadratic objective matrix entries Q_ij
    std::map<std::pair<std::string, std::string>, double> q1, q2;
    for (const auto& qterm : objective_.quadratic_terms) {
        if (qterm.var1 < variables_.size() && qterm.var2 < variables_.size()) {
            std::string n1 = variables_[qterm.var1].name;
            std::string n2 = variables_[qterm.var2].name;
            if (n1 > n2) std::swap(n1, n2);
            q1[{n1, n2}] += qterm.coefficient;
        }
    }
    for (const auto& qterm : other.objective_.quadratic_terms) {
        if (qterm.var1 < other.variables_.size() && qterm.var2 < other.variables_.size()) {
            std::string n1 = other.variables_[qterm.var1].name;
            std::string n2 = other.variables_[qterm.var2].name;
            if (n1 > n2) std::swap(n1, n2);
            q2[{n1, n2}] += qterm.coefficient;
        }
    }
    for (const auto& [pair, coeff] : q1) {
        double c2 = q2[pair];
        if (!float_equals(coeff, c2, abs_tol, rel_tol)) {
            return false;
        }
    }
    for (const auto& [pair, coeff] : q2) {
        double c1 = q1[pair];
        if (!float_equals(c1, coeff, abs_tol, rel_tol)) {
            return false;
        }
    }

    // Compare constraints by name
    for (const auto& [name, con_idx] : name_to_con_) {
        auto other_it = other.name_to_con_.find(name);
        if (other_it == other.name_to_con_.end()) {
            return false;
        }
        const auto& c1 = constraints_[con_idx];
        const auto& c2 = other.constraints_[other_it->second];
        if (!float_equals(c1.lower_bound, c2.lower_bound, abs_tol, rel_tol) ||
            !float_equals(c1.upper_bound, c2.upper_bound, abs_tol, rel_tol)) {
            return false;
        }

        std::map<std::string, double> terms1, terms2;
        for (const auto& t : c1.terms) {
            if (t.variable_index < variables_.size()) {
                terms1[variables_[t.variable_index].name] += t.coefficient;
            }
        }
        for (const auto& t : c2.terms) {
            if (t.variable_index < other.variables_.size()) {
                terms2[other.variables_[t.variable_index].name] += t.coefficient;
            }
        }
        for (const auto& [vname, coeff] : terms1) {
            if (!float_equals(coeff, terms2[vname], abs_tol, rel_tol)) {
                return false;
            }
        }
        for (const auto& [vname, coeff] : terms2) {
            if (!float_equals(terms1[vname], coeff, abs_tol, rel_tol)) {
                return false;
            }
        }
    }

    return true;
}

void Model::clear() {
    name_.clear();
    variables_.clear();
    constraints_.clear();
    objective_.clear();
    name_to_var_.clear();
    name_to_con_.clear();
    quad_index_map_.clear();
}

} // namespace sih26119
