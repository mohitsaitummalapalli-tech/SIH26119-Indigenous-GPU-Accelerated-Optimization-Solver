#pragma once

#include "core/result.hpp"
#include "numerics/scalar.hpp"
#include "numerics/index.hpp"
#include "model/types.hpp"
#include "model/objective.hpp"
#include <vector>
#include <cstdint>

namespace sih26119 {

/// Status of an LP solver execution.
enum class LpSolverStatus : uint8_t {
    Optimal,            ///< Globally optimal solution found and verified.
    Infeasible,         ///< Primal problem has no feasible solution.
    Unbounded,          ///< Objective is unbounded (f(x) -> -inf for min).
    NumericalFailure,   ///< Numerical singularity, excessive round-off, or loss of basis rank.
    IterationLimit,     ///< Maximum pivot limit reached without proving optimality.
    InvalidModel        ///< Input model violates LP invariants or continuous variable requirements.
};

/// Type of transformation applied to an original model variable.
enum class VariableTransformType : uint8_t {
    Identity,           ///< x >= 0 -> x_bar = x (identity mapping)
    LowerShift,         ///< x >= L (L != 0) -> x = x_bar + L, x_bar >= 0
    UpperReflect,       ///< x <= U -> x = U - x_bar, x_bar >= 0
    BoxBound,           ///< L <= x <= U -> x = x_bar + L, x_bar + s = U - L, x_bar, s >= 0
    FreeSplit,          ///< x free -> x = x_plus - x_minus, x_plus, x_minus >= 0
    FixedEliminated     ///< x == C -> eliminated from standard columns, shifted to RHS
};

/// Exact mapping metadata for an original variable.
struct VariableMapping {
    VariableIndex original_index{kInvalidVariableIndex};
    VariableTransformType transform_type{VariableTransformType::Identity};
    Index std_var_primary{kInvalidIndex};      ///< Primary standard variable coordinate
    Index std_var_secondary{kInvalidIndex};    ///< Secondary variable (e.g. x_minus for FreeSplit)
    Index std_var_slack{kInvalidIndex};        ///< Slack variable (e.g. s for BoxBound)
    Scalar original_lb{0.0};
    Scalar original_ub{kInfinity};
    Scalar fixed_value{0.0};                   ///< Stored constant for FixedEliminated
    Scalar original_cost{0.0};                  ///< Linear objective coefficient in original Model
};

/// Type of transformation applied to an original constraint row.
enum class ConstraintTransformType : uint8_t {
    Equality,           ///< a^T x == b -> a_bar^T x_bar = b_bar
    LessEqual,          ///< a^T x <= u -> a_bar^T x_bar + s = u_bar, s >= 0
    GreaterEqual,       ///< a^T x >= l -> a_bar^T x_bar - e = l_bar, e >= 0
    Range,              ///< l <= a^T x <= u -> a_bar^T x_bar - s = l_bar, s + t = u - l, s, t >= 0
    Free                ///< -inf <= a^T x <= inf -> mathematically redundant, omitted
};

/// Exact mapping metadata for an original constraint row.
struct ConstraintMapping {
    ConstraintIndex original_index{kInvalidConstraintIndex};
    ConstraintTransformType transform_type{ConstraintTransformType::Equality};
    std::vector<Index> generated_rows;         ///< Standard equality row indices generated
    Index auxiliary_primary{kInvalidIndex};    ///< Primary auxiliary variable (slack s or surplus e)
    Index auxiliary_secondary{kInvalidIndex};  ///< Secondary auxiliary variable (range slack t)
    Scalar original_lb{-kInfinity};
    Scalar original_ub{kInfinity};
    std::vector<bool> row_negated;             ///< Whether standard row was multiplied by -1
};

} // namespace sih26119
