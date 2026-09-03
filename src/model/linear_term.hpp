#pragma once

#include "model/types.hpp"

namespace sih26119 {

struct LinearTerm {
    VariableIndex variable_index{kInvalidVariableIndex};
    double coefficient{0.0};

    constexpr LinearTerm() noexcept = default;
    constexpr LinearTerm(VariableIndex var_idx, double coeff) noexcept
        : variable_index(var_idx), coefficient(coeff) {}

    bool operator==(const LinearTerm& other) const noexcept {
        return variable_index == other.variable_index && coefficient == other.coefficient;
    }
};

} // namespace sih26119
