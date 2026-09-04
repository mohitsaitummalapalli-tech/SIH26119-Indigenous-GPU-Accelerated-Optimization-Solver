#pragma once

#include <cmath>
#include <limits>

namespace sih26119 {

/// Foundational real scalar type for the numerical layer.
using Scalar = double;

/// Validates whether a scalar value is finite (not NaN, +Inf, or -Inf).
[[nodiscard]] constexpr inline bool is_finite_scalar(Scalar val) noexcept {
    // std::isfinite is not constexpr in all C++20 implementations; use std::isnan / std::isinf or __builtin_isfinite
    return !std::isnan(val) && !std::isinf(val);
}

inline constexpr Scalar kScalarZero = 0.0;
inline constexpr Scalar kScalarOne = 1.0;
inline constexpr Scalar kScalarInfinity = std::numeric_limits<Scalar>::infinity();

} // namespace sih26119
