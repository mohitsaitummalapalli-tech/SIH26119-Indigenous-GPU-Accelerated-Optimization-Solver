#pragma once

#include <cstdint>
#include <limits>

namespace sih26119 {

// Index and count types with strict index vs. count separation
using VariableIndex = uint32_t;
using ConstraintIndex = uint32_t;
using DimensionCount = uint32_t;
using NonzeroCount = uint64_t;

// Sentinel invalid indices
inline constexpr VariableIndex kInvalidVariableIndex = std::numeric_limits<VariableIndex>::max();
inline constexpr ConstraintIndex kInvalidConstraintIndex = std::numeric_limits<ConstraintIndex>::max();

// Canonical IEEE 754 infinity
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Semantic comparison tolerances
inline constexpr double kAbsTol = 1e-12;
inline constexpr double kRelTol = 1e-12;

} // namespace sih26119
