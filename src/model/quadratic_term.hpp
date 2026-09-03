#pragma once

#include "model/types.hpp"
#include <algorithm>

namespace sih26119 {

/**
 * @brief Canonical representation of a quadratic objective term.
 *
 * Canonical objective: f(x) = c^T x + 1/2 x^T Q x + c0 with symmetric Q.
 * Stored coefficient:
 *   - Diagonal (var1 == var2): represents Q_ii. Polynomial contribution is 1/2 * Q_ii * x_i^2.
 *   - Off-diagonal (var1 < var2): represents Q_ij (where Q_ij = Q_ji).
 *     Polynomial contribution is Q_ij * x_i * x_j.
 */
struct QuadraticTerm {
    VariableIndex var1{kInvalidVariableIndex};
    VariableIndex var2{kInvalidVariableIndex};
    double coefficient{0.0}; // Stored Q_ij entry

    constexpr QuadraticTerm() noexcept = default;

    constexpr QuadraticTerm(VariableIndex v1, VariableIndex v2, double coeff) noexcept
        : var1(v1 <= v2 ? v1 : v2),
          var2(v1 <= v2 ? v2 : v1),
          coefficient(coeff) {}

    [[nodiscard]] constexpr bool is_diagonal() const noexcept {
        return var1 == var2;
    }

    bool operator==(const QuadraticTerm& other) const noexcept {
        return var1 == other.var1 && var2 == other.var2 && coefficient == other.coefficient;
    }

    bool operator<(const QuadraticTerm& other) const noexcept {
        if (var1 != other.var1) return var1 < other.var1;
        return var2 < other.var2;
    }
};

} // namespace sih26119
