#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include <span>
#include <cmath>
#include <algorithm>

namespace sih26119 {

/// Computes the Euclidean 2-norm (L2 norm) using stable scaled accumulation.
/// Rejects non-finite values with an explicit StatusCode::InvalidArgument error.
/// An empty span returns 0.0.
[[nodiscard]] inline Result<Scalar> norm2(std::span<const Scalar> data) noexcept {
    if (data.empty()) {
        return kScalarZero;
    }

    Scalar max_abs = kScalarZero;
    for (const Scalar x : data) {
        if (!is_finite_scalar(x)) {
            return Status::error(StatusCode::InvalidArgument, "Non-finite element encountered during norm2 calculation");
        }
        const Scalar ax = std::abs(x);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    if (max_abs == kScalarZero) {
        return kScalarZero;
    }

    Scalar sum_sq = kScalarZero;
    for (const Scalar x : data) {
        const Scalar scaled = x / max_abs;
        sum_sq += scaled * scaled;
    }

    return max_abs * std::sqrt(sum_sq);
}

/// Computes the Infinity norm (L-inf norm: max absolute value).
/// Rejects non-finite values with an explicit StatusCode::InvalidArgument error.
/// An empty span returns 0.0.
[[nodiscard]] inline Result<Scalar> norm_inf(std::span<const Scalar> data) noexcept {
    if (data.empty()) {
        return kScalarZero;
    }

    Scalar max_abs = kScalarZero;
    for (const Scalar x : data) {
        if (!is_finite_scalar(x)) {
            return Status::error(StatusCode::InvalidArgument, "Non-finite element encountered during norm_inf calculation");
        }
        const Scalar ax = std::abs(x);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    return max_abs;
}

} // namespace sih26119
