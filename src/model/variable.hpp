#pragma once

#include "model/types.hpp"
#include "model/variable_type.hpp"
#include <string>
#include <cmath>

namespace sih26119 {

struct Variable {
    std::string name;
    double lower_bound{0.0};
    double upper_bound{kInfinity};
    VariableType type{VariableType::Continuous};

    Variable() = default;

    Variable(std::string var_name, double lb, double ub, VariableType var_type = VariableType::Continuous)
        : name(std::move(var_name)), lower_bound(lb), upper_bound(ub), type(var_type) {}

    [[nodiscard]] bool is_free() const noexcept {
        return std::isinf(lower_bound) && lower_bound < 0.0 &&
               std::isinf(upper_bound) && upper_bound > 0.0;
    }

    [[nodiscard]] bool is_fixed() const noexcept {
        return lower_bound == upper_bound;
    }

    [[nodiscard]] bool is_binary() const noexcept {
        return type == VariableType::Binary;
    }

    [[nodiscard]] bool is_integer() const noexcept {
        return type == VariableType::Integer || type == VariableType::Binary;
    }
};

} // namespace sih26119
