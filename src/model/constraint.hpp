#pragma once

#include "model/types.hpp"
#include "model/linear_term.hpp"
#include <string>
#include <vector>
#include <cmath>

namespace sih26119 {

struct Constraint {
    std::string name;
    double lower_bound{-kInfinity};
    double upper_bound{kInfinity};
    std::vector<LinearTerm> terms;

    Constraint() = default;

    Constraint(std::string con_name, double lb, double ub, std::vector<LinearTerm> linear_terms = {})
        : name(std::move(con_name)), lower_bound(lb), upper_bound(ub), terms(std::move(linear_terms)) {}

    [[nodiscard]] bool is_equality() const noexcept {
        return lower_bound == upper_bound && !std::isinf(lower_bound);
    }

    [[nodiscard]] bool is_less_equal() const noexcept {
        return std::isinf(lower_bound) && lower_bound < 0.0 && !std::isinf(upper_bound);
    }

    [[nodiscard]] bool is_greater_equal() const noexcept {
        return !std::isinf(lower_bound) && std::isinf(upper_bound) && upper_bound > 0.0;
    }

    [[nodiscard]] bool is_range() const noexcept {
        return !std::isinf(lower_bound) && !std::isinf(upper_bound) && lower_bound < upper_bound;
    }

    [[nodiscard]] bool is_free() const noexcept {
        return std::isinf(lower_bound) && lower_bound < 0.0 &&
               std::isinf(upper_bound) && upper_bound > 0.0;
    }
};

} // namespace sih26119
