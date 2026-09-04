#pragma once

#include "numerics/scalar.hpp"
#include <algorithm>
#include <cmath>

namespace sih26119 {

/// Semantic tolerance configuration adhering to project-wide standard.
struct Tolerance {
    Scalar abs_tol = 1e-12;
    Scalar rel_tol = 1e-12;
};

/// Authoritative semantic floating-point approximate equality comparison:
/// |a - b| <= abs_tol + rel_tol * max(1.0, |a|, |b|)
[[nodiscard]] inline bool approx_equal(Scalar a, Scalar b, Tolerance tol = Tolerance{}) noexcept {
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    const Scalar diff = std::abs(a - b);
    const Scalar max_mag = std::max({kScalarOne, std::abs(a), std::abs(b)});
    return diff <= (tol.abs_tol + tol.rel_tol * max_mag);
}

/// Checks if a scalar is approximately zero within absolute tolerance.
[[nodiscard]] inline bool approx_zero(Scalar a, Scalar abs_tol = 1e-12) noexcept {
    if (std::isnan(a) || std::isinf(a)) {
        return false;
    }
    return std::abs(a) <= abs_tol;
}

} // namespace sih26119
