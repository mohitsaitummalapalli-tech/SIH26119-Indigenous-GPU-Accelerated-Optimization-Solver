#pragma once

#include <string_view>

namespace sih26119 {

enum class VariableType {
    Continuous,
    Integer,
    Binary
};

inline std::string_view variable_type_to_string(VariableType type) noexcept {
    switch (type) {
        case VariableType::Continuous: return "Continuous";
        case VariableType::Integer: return "Integer";
        case VariableType::Binary: return "Binary";
    }
    return "Unknown";
}

} // namespace sih26119
