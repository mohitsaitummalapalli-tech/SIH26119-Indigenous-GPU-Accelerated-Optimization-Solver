#pragma once

#include <string_view>

namespace sih26119 {

enum class ObjectiveSense {
    Minimize,
    Maximize
};

inline std::string_view objective_sense_to_string(ObjectiveSense sense) noexcept {
    switch (sense) {
        case ObjectiveSense::Minimize: return "Minimize";
        case ObjectiveSense::Maximize: return "Maximize";
    }
    return "Unknown";
}

} // namespace sih26119
