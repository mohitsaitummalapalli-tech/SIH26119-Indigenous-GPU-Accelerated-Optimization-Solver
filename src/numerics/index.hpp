#pragma once

#include "core/result.hpp"
#include <cstdint>
#include <cstddef>
#include <limits>

namespace sih26119 {

/// Coordinate index type within a dimension (e.g. column index, row index).
using Index = uint32_t;

/// Matrix or vector dimension count (number of rows, columns, or vector length).
using Dimension = uint32_t;

/// Nonzero entry count for sparse matrix representation.
using NonzeroCount = uint64_t;

inline constexpr Index kInvalidIndex = std::numeric_limits<Index>::max();
inline constexpr Dimension kInvalidDimension = std::numeric_limits<Dimension>::max();
inline constexpr NonzeroCount kInvalidNonzeroCount = std::numeric_limits<NonzeroCount>::max();

/// Safe checked conversion from size_t to Dimension.
[[nodiscard]] inline Result<Dimension> to_dimension(std::size_t val) noexcept {
    if (val > static_cast<std::size_t>(std::numeric_limits<Dimension>::max())) {
        return Status::error(StatusCode::InvalidArgument, "Value exceeds maximum supported Dimension (uint32_t overflow)");
    }
    return static_cast<Dimension>(val);
}

/// Safe checked conversion from size_t to Index.
[[nodiscard]] inline Result<Index> to_index(std::size_t val) noexcept {
    if (val > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        return Status::error(StatusCode::InvalidArgument, "Value exceeds maximum supported Index (uint32_t overflow)");
    }
    return static_cast<Index>(val);
}

/// Safe checked conversion from size_t to NonzeroCount.
[[nodiscard]] inline Result<NonzeroCount> to_nonzero_count(std::size_t val) noexcept {
    if (val > static_cast<std::size_t>(std::numeric_limits<NonzeroCount>::max())) {
        return Status::error(StatusCode::InvalidArgument, "Value exceeds maximum supported NonzeroCount (uint64_t overflow)");
    }
    return static_cast<NonzeroCount>(val);
}

} // namespace sih26119
