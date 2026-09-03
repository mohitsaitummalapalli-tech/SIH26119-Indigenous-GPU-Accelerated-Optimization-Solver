#include "io/lp_reader.hpp"
#include <fstream>
#include <vector>
#include <cctype>
#include <cmath>

namespace sih26119 {

namespace {

std::string to_upper(std::string_view s) {
    std::string res;
    res.reserve(s.size());
    for (char c : s) {
        res.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return res;
}

enum class LpSection {
    None,
    Objective,
    SubjectTo,
    Bounds,
    Generals,
    Binaries,
    End
};

bool parse_num(std::string_view str, double& out_val) {
    std::string s(str);
    char* end = nullptr;
    out_val = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && !std::isnan(out_val);
}

} // anonymous namespace

Result<Model> LpReader::read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Status::error(StatusCode::IOError, "Failed to open LP file: " + filepath);
    }
    return read_stream(file, filepath);
}

Result<Model> LpReader::read_stream(std::istream& is, const std::string& sourcename) {
    (void)sourcename;
    Model model;
    std::string full_text;
    std::string line;

    // First pass: strip comments starting with '\'
    while (std::getline(is, line)) {
        size_t comment_pos = line.find('\\');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        full_text += line + " \n ";
    }

    // Tokenize stream into tokens, preserving punctuation symbols: +, -, *, ^, [, ], /, :, <=, >=, =, <, >
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < full_text.size()) {
        while (i < full_text.size() && std::isspace(static_cast<unsigned char>(full_text[i]))) {
            ++i;
        }
        if (i >= full_text.size()) break;

        char c = full_text[i];
        if (c == '<' || c == '>') {
            if (i + 1 < full_text.size() && full_text[i + 1] == '=') {
                tokens.emplace_back(full_text.substr(i, 2));
                i += 2;
            } else {
                tokens.emplace_back(1, c);
                ++i;
            }
        } else if (c == ':' || c == '[' || c == ']' || c == '*' || c == '^' || c == '=') {
            tokens.emplace_back(1, c);
            ++i;
        } else if (c == '+' || c == '-') {
            // Check if this +/- is part of a number or standalone operator
            // If next chars look like digits or decimal, could be sign of number
            tokens.emplace_back(1, c);
            ++i;
        } else if (c == '/') {
            tokens.emplace_back(1, c);
            ++i;
        } else {
            // Word or number
            size_t start = i;
            while (i < full_text.size() &&
                   !std::isspace(static_cast<unsigned char>(full_text[i])) &&
                   full_text[i] != ':' && full_text[i] != '[' && full_text[i] != ']' &&
                   full_text[i] != '*' && full_text[i] != '^' && full_text[i] != '=' &&
                   full_text[i] != '<' && full_text[i] != '>' && full_text[i] != '+' &&
                   full_text[i] != '-' && full_text[i] != '/') {
                ++i;
            }
            tokens.emplace_back(full_text.substr(start, i - start));
        }
    }

    // Variable tracking
    std::unordered_map<std::string, VariableIndex> var_map;
    auto get_or_add_var = [&](const std::string& name) -> VariableIndex {
        auto it = var_map.find(name);
        if (it != var_map.end()) return it->second;
        auto res = model.add_variable(name, 0.0, kInfinity, VariableType::Continuous);
        VariableIndex idx = res.value();
        var_map[name] = idx;
        return idx;
    };

    LpSection current_section = LpSection::None;
    size_t t_idx = 0;

    auto is_section_header = [&](size_t idx, LpSection& next_sec, size_t& advance) -> bool {
        if (idx >= tokens.size()) return false;
        std::string u1 = to_upper(tokens[idx]);

        if (u1 == "MIN" || u1 == "MINIMIZE" || u1 == "MINIMUM") {
            next_sec = LpSection::Objective;
            model.set_objective_sense(ObjectiveSense::Minimize);
            advance = 1;
            return true;
        }
        if (u1 == "MAX" || u1 == "MAXIMIZE" || u1 == "MAXIMUM") {
            next_sec = LpSection::Objective;
            model.set_objective_sense(ObjectiveSense::Maximize);
            advance = 1;
            return true;
        }
        if (u1 == "SUBJECT" && idx + 1 < tokens.size() && to_upper(tokens[idx + 1]) == "TO") {
            next_sec = LpSection::SubjectTo;
            advance = 2;
            return true;
        }
        if (u1 == "SUCH" && idx + 1 < tokens.size() && to_upper(tokens[idx + 1]) == "THAT") {
            next_sec = LpSection::SubjectTo;
            advance = 2;
            return true;
        }
        if (u1 == "ST" || u1 == "S.T.") {
            next_sec = LpSection::SubjectTo;
            advance = 1;
            return true;
        }
        if (u1 == "BOUNDS" || u1 == "BOUND") {
            next_sec = LpSection::Bounds;
            advance = 1;
            return true;
        }
        if (u1 == "GENERALS" || u1 == "GENERAL" || u1 == "INTEGERS" || u1 == "INTEGER" || u1 == "GEN" || u1 == "INT") {
            next_sec = LpSection::Generals;
            advance = 1;
            return true;
        }
        if (u1 == "BINARIES" || u1 == "BINARY" || u1 == "BIN") {
            next_sec = LpSection::Binaries;
            advance = 1;
            return true;
        }
        if (u1 == "END") {
            next_sec = LpSection::End;
            advance = 1;
            return true;
        }
        return false;
    };

    while (t_idx < tokens.size()) {
        LpSection next_sec;
        size_t advance = 0;
        if (is_section_header(t_idx, next_sec, advance)) {
            current_section = next_sec;
            t_idx += advance;
            if (current_section == LpSection::End) break;
            continue;
        }

        switch (current_section) {
            case LpSection::Objective: {
                // Check if optional objective name: obj_name :
                if (t_idx + 1 < tokens.size() && tokens[t_idx + 1] == ":") {
                    // Objective name
                    t_idx += 2;
                    continue;
                }

                // Check if start of quadratic part: '['
                if (tokens[t_idx] == "[") {
                    ++t_idx; // consume '['
                    // Parse quadratic polynomial P(x) until ']'
                    while (t_idx < tokens.size() && tokens[t_idx] != "]") {
                        double sign = 1.0;
                        if (tokens[t_idx] == "+") {
                            sign = 1.0;
                            ++t_idx;
                        } else if (tokens[t_idx] == "-") {
                            sign = -1.0;
                            ++t_idx;
                        }

                        if (t_idx >= tokens.size() || tokens[t_idx] == "]") break;

                        double coeff = sign;
                        double parsed_c = 0.0;
                        if (parse_num(tokens[t_idx], parsed_c)) {
                            coeff = sign * parsed_c;
                            ++t_idx;
                        }

                        if (t_idx >= tokens.size()) {
                            return Status::parse_error(0, "Unexpected end of quadratic objective expression");
                        }

                        // Variable name
                        std::string var1_name = tokens[t_idx];
                        ++t_idx;
                        VariableIndex v1 = get_or_add_var(var1_name);

                        // Could be var1 ^ 2, var1 * var1, or var1 * var2
                        if (t_idx < tokens.size() && tokens[t_idx] == "^") {
                            ++t_idx; // consume '^'
                            if (t_idx >= tokens.size() || tokens[t_idx] != "2") {
                                return Status::parse_error(0, "Expected '2' after '^' in quadratic objective");
                            }
                            ++t_idx; // consume '2'
                            // Diagonal term: coeff * x_i^2 inside [ P(x) ] / 2
                            // Rule: diagonal coeff a in P -> Q_ii = a
                            model.add_quadratic_coefficient(v1, v1, coeff);
                        } else if (t_idx < tokens.size() && tokens[t_idx] == "*") {
                            ++t_idx; // consume '*'
                            if (t_idx >= tokens.size()) {
                                return Status::parse_error(0, "Expected variable name after '*' in quadratic objective");
                            }
                            std::string var2_name = tokens[t_idx];
                            ++t_idx;
                            VariableIndex v2 = get_or_add_var(var2_name);
                            if (v1 == v2) {
                                // Diagonal term written as x * x
                                model.add_quadratic_coefficient(v1, v1, coeff);
                            } else {
                                // Off-diagonal term: coeff * x_i * x_j inside [ P(x) ] / 2
                                // Rule: algebraic coeff b in P -> Q_ij = b / 2
                                model.add_quadratic_coefficient(v1, v2, coeff / 2.0);
                            }
                        } else {
                            return Status::parse_error(0, "Malformed quadratic objective term: " + var1_name);
                        }
                    }

                    if (t_idx >= tokens.size() || tokens[t_idx] != "]") {
                        return Status::parse_error(0, "Missing closing ']' for quadratic objective");
                    }
                    ++t_idx; // consume ']'

                    // CRITICAL MATHEMATICAL GATE: Divisor '/ 2' MUST be present immediately following ']'
                    if (t_idx + 1 >= tokens.size() || tokens[t_idx] != "/" || tokens[t_idx + 1] != "2") {
                        return Status::parse_error(0, "Quadratic objective syntax must strictly be [ P(x) ] / 2");
                    }
                    t_idx += 2; // consume '/' and '2'
                    continue;
                }

                // Linear objective term
                double sign = 1.0;
                if (tokens[t_idx] == "+") {
                    sign = 1.0;
                    ++t_idx;
                } else if (tokens[t_idx] == "-") {
                    sign = -1.0;
                    ++t_idx;
                }

                if (t_idx >= tokens.size()) break;

                // Peek if next token is section header
                LpSection test_sec;
                size_t test_adv = 0;
                if (is_section_header(t_idx, test_sec, test_adv)) {
                    break;
                }

                double coeff = sign;
                double parsed_c = 0.0;
                if (parse_num(tokens[t_idx], parsed_c)) {
                    coeff = sign * parsed_c;
                    ++t_idx;
                }

                if (t_idx >= tokens.size()) break;
                if (tokens[t_idx] == "[") {
                    // Handled in next iteration
                    if (coeff != 1.0) {
                        return Status::parse_error(0, "Leading scalar before '[' in quadratic objective is invalid");
                    }
                    continue;
                }

                std::string var_name = tokens[t_idx];
                ++t_idx;
                VariableIndex v_idx = get_or_add_var(var_name);
                model.add_objective_term(v_idx, coeff);
                break;
            }

            case LpSection::SubjectTo: {
                // Check if constraint name: cname :
                std::string con_name;
                if (t_idx + 1 < tokens.size() && tokens[t_idx + 1] == ":") {
                    con_name = tokens[t_idx];
                    t_idx += 2;
                } else {
                    con_name = "c" + std::to_string(model.num_constraints() + 1);
                }

                // Check for ranged constraint starting with LHS: num <= expr <= num
                double lhs_val = -kInfinity;
                bool has_lhs = false;
                if (t_idx + 1 < tokens.size() && (tokens[t_idx + 1] == "<=" || tokens[t_idx + 1] == "<")) {
                    double num = 0.0;
                    if (parse_num(tokens[t_idx], num)) {
                        lhs_val = num;
                        has_lhs = true;
                        t_idx += 2; // consume num and '<='
                    }
                }

                // Parse linear expression of constraint
                std::vector<LinearTerm> terms;
                while (t_idx < tokens.size()) {
                    if (tokens[t_idx] == "<=" || tokens[t_idx] == ">=" || tokens[t_idx] == "=" ||
                        tokens[t_idx] == "<" || tokens[t_idx] == ">") {
                        break;
                    }

                    double sign = 1.0;
                    if (tokens[t_idx] == "+") {
                        sign = 1.0;
                        ++t_idx;
                    } else if (tokens[t_idx] == "-") {
                        sign = -1.0;
                        ++t_idx;
                    }

                    if (t_idx >= tokens.size()) break;

                    double coeff = sign;
                    double parsed_c = 0.0;
                    if (parse_num(tokens[t_idx], parsed_c)) {
                        coeff = sign * parsed_c;
                        ++t_idx;
                    }

                    if (t_idx >= tokens.size()) break;
                    std::string var_name = tokens[t_idx];
                    ++t_idx;
                    VariableIndex v_idx = get_or_add_var(var_name);
                    terms.emplace_back(v_idx, coeff);
                }

                if (t_idx >= tokens.size()) {
                    return Status::parse_error(0, "Unexpected end of constraint: " + con_name);
                }

                std::string op = tokens[t_idx];
                ++t_idx;

                double rhs_val = 0.0;
                double sign_rhs = 1.0;
                if (t_idx < tokens.size() && (tokens[t_idx] == "+" || tokens[t_idx] == "-")) {
                    if (tokens[t_idx] == "-") sign_rhs = -1.0;
                    ++t_idx;
                }
                if (t_idx >= tokens.size() || !parse_num(tokens[t_idx], rhs_val)) {
                    return Status::parse_error(0, "Expected numeric RHS for constraint: " + con_name);
                }
                rhs_val *= sign_rhs;
                ++t_idx;

                double lb = -kInfinity;
                double ub = kInfinity;

                if (has_lhs) {
                    lb = lhs_val;
                    ub = rhs_val;
                } else if (op == "<=" || op == "<") {
                    ub = rhs_val;
                } else if (op == ">=" || op == ">") {
                    lb = rhs_val;
                } else if (op == "=") {
                    lb = rhs_val;
                    ub = rhs_val;
                }

                auto add_con_res = model.add_constraint(con_name, lb, ub, std::move(terms));
                if (!add_con_res.ok()) {
                    return add_con_res.status();
                }
                break;
            }

            case LpSection::Bounds: {
                // Patterns:
                // 1) var free
                // 2) lb <= var <= ub
                // 3) lb <= var
                // 4) var <= ub
                // 5) var >= lb
                // 6) var = val
                // Peek if start with number:
                double num1 = 0.0;
                double num1_sign = 1.0;
                size_t p = t_idx;
                if (tokens[p] == "+" || tokens[p] == "-") {
                    if (tokens[p] == "-") num1_sign = -1.0;
                    ++p;
                }

                if (p < tokens.size() && (parse_num(tokens[p], num1) || to_upper(tokens[p]) == "INFINITY" || to_upper(tokens[p]) == "INF")) {
                    bool is_inf = (to_upper(tokens[p]) == "INFINITY" || to_upper(tokens[p]) == "INF");
                    num1 = is_inf ? (num1_sign > 0 ? kInfinity : -kInfinity) : (num1 * num1_sign);
                    t_idx = p + 1; // consumed num1

                    if (t_idx < tokens.size() && (tokens[t_idx] == "<=" || tokens[t_idx] == "<")) {
                        ++t_idx; // consumed '<='
                        std::string var_name = tokens[t_idx];
                        ++t_idx;
                        VariableIndex v_idx = get_or_add_var(var_name);

                        if (t_idx < tokens.size() && (tokens[t_idx] == "<=" || tokens[t_idx] == "<")) {
                            ++t_idx; // consumed '<='
                            double num2 = 0.0;
                            double num2_sign = 1.0;
                            if (tokens[t_idx] == "+" || tokens[t_idx] == "-") {
                                if (tokens[t_idx] == "-") num2_sign = -1.0;
                                ++t_idx;
                            }
                            if (to_upper(tokens[t_idx]) == "INFINITY" || to_upper(tokens[t_idx]) == "INF") {
                                num2 = num2_sign > 0 ? kInfinity : -kInfinity;
                            } else {
                                parse_num(tokens[t_idx], num2);
                                num2 *= num2_sign;
                            }
                            ++t_idx;
                            model.set_variable_bounds(v_idx, num1, num2);
                        } else {
                            // num1 <= var  =>  var >= num1
                            model.set_variable_bounds(v_idx, num1, model.get_variable(v_idx).upper_bound);
                        }
                    }
                    break;
                }

                // Starts with variable name
                std::string var_name = tokens[t_idx];
                ++t_idx;
                VariableIndex v_idx = get_or_add_var(var_name);

                if (t_idx < tokens.size() && to_upper(tokens[t_idx]) == "FREE") {
                    ++t_idx;
                    model.set_variable_bounds(v_idx, -kInfinity, kInfinity);
                    break;
                }

                if (t_idx < tokens.size()) {
                    std::string op = tokens[t_idx];
                    ++t_idx;
                    double bval = 0.0;
                    double bsign = 1.0;
                    if (tokens[t_idx] == "+" || tokens[t_idx] == "-") {
                        if (tokens[t_idx] == "-") bsign = -1.0;
                        ++t_idx;
                    }
                    if (to_upper(tokens[t_idx]) == "INFINITY" || to_upper(tokens[t_idx]) == "INF") {
                        bval = bsign > 0 ? kInfinity : -kInfinity;
                    } else {
                        parse_num(tokens[t_idx], bval);
                        bval *= bsign;
                    }
                    ++t_idx;

                    if (op == "<=" || op == "<") {
                        model.set_variable_bounds(v_idx, model.get_variable(v_idx).lower_bound, bval);
                    } else if (op == ">=" || op == ">") {
                        model.set_variable_bounds(v_idx, bval, model.get_variable(v_idx).upper_bound);
                    } else if (op == "=") {
                        model.set_variable_bounds(v_idx, bval, bval);
                    }
                }
                break;
            }

            case LpSection::Generals: {
                std::string var_name = tokens[t_idx];
                ++t_idx;
                VariableIndex v_idx = get_or_add_var(var_name);
                model.set_variable_type(v_idx, VariableType::Integer);
                break;
            }

            case LpSection::Binaries: {
                std::string var_name = tokens[t_idx];
                ++t_idx;
                VariableIndex v_idx = get_or_add_var(var_name);
                model.set_variable_bounds(v_idx, 0.0, 1.0);
                model.set_variable_type(v_idx, VariableType::Binary);
                break;
            }

            default:
                ++t_idx;
                break;
        }
    }

    auto val_status = model.validate();
    if (!val_status.is_ok()) {
        return val_status;
    }

    return model;
}

} // namespace sih26119
