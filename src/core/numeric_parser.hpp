#pragma once

#include <charconv>
#include <string_view>
#include <cmath>
#include <cctype>

namespace sih26119 {

/**
 * @brief Parses a floating-point number strictly from the entire string view.
 * 
 * Requirements enforced:
 * - Entire token must be consumed (rejects "12abc")
 * - Disallows double signs (rejects "--3", "++5", "+-1")
 * - Rejects malformed exponents (rejects "1.2e", "1e+")
 * - Rejects NaN and unsupported Infinity tokens
 * - Supports scientific notation with positive/negative exponents (e.g., "1.5e-3", "-1e-3", "+2.5E+4")
 * - Disallows internal whitespace
 * 
 * @param str The string view to parse
 * @param out_val The output parsed double
 * @return true if string represents a strictly valid IEEE 754 finite double
 */
inline bool parse_strict_double(std::string_view str, double& out_val) noexcept {
    if (str.empty()) {
        return false;
    }

    const char* start = str.data();
    const char* end = str.data() + str.size();

    // Check for internal or trailing whitespace
    for (char c : str) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    // Leading sign validation
    if (*start == '+') {
        ++start;
        if (start == end) {
            return false; // Single '+'
        }
        if (*start == '+' || *start == '-') {
            return false; // Double sign like '++' or '+-'
        }
    } else if (*start == '-') {
        if (start + 1 < end && (*(start + 1) == '-' || *(start + 1) == '+')) {
            return false; // Double sign like '--' or '-+'
        }
    }

    auto [ptr, ec] = std::from_chars(start, end, out_val, std::chars_format::general);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }

    if (std::isnan(out_val) || std::isinf(out_val)) {
        return false;
    }

    return true;
}

} // namespace sih26119
