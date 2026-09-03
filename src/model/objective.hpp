#pragma once

#include "model/types.hpp"
#include "model/objective_sense.hpp"
#include "model/linear_term.hpp"
#include "model/quadratic_term.hpp"
#include <vector>

namespace sih26119 {

struct Objective {
    ObjectiveSense sense{ObjectiveSense::Minimize};
    double offset{0.0}; // c0 constant offset
    std::vector<LinearTerm> linear_terms;
    std::vector<QuadraticTerm> quadratic_terms;

    [[nodiscard]] bool is_quadratic() const noexcept {
        return !quadratic_terms.empty();
    }

    void clear() {
        sense = ObjectiveSense::Minimize;
        offset = 0.0;
        linear_terms.clear();
        quadratic_terms.clear();
    }
};

} // namespace sih26119
