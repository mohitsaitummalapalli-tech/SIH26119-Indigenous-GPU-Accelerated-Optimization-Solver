#include "io/lp_reader.hpp"
#include "core/numeric_parser.hpp"
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

struct Token {
    std::string text;
    uint64_t line = 1;
};

enum class LpSection {
    None,
    Objective,
    SubjectTo,
    Bounds,
    Generals,
    Binaries,
    End
};

std::vector<Token> tokenize_lp_stream(std::istream& is) {
    std::vector<Token> tokens;
    std::string line_str;
    uint64_t line_num = 0;

    while (std::getline(is, line_str)) {
        ++line_num;
        size_t comment_pos = line_str.find('\\');
        if (comment_pos != std::string::npos) {
            line_str = line_str.substr(0, comment_pos);
        }

        size_t i = 0;
        while (i < line_str.size()) {
            while (i < line_str.size() && std::isspace(static_cast<unsigned char>(line_str[i]))) {
                ++i;
            }
            if (i >= line_str.size()) break;

            char c = line_str[i];
            if (c == '<' || c == '>') {
                if (i + 1 < line_str.size() && line_str[i + 1] == '=') {
                    tokens.push_back({line_str.substr(i, 2), line_num});
                    i += 2;
                } else {
                    tokens.push_back({std::string(1, c), line_num});
                    ++i;
                }
            } else if (c == ':' || c == '[' || c == ']' || c == '*' || c == '^' || c == '=') {
                tokens.push_back({std::string(1, c), line_num});
                ++i;
            } else if (c == '/') {
                tokens.push_back({std::string(1, c), line_num});
                ++i;
            } else if (c == '+' || c == '-') {
                bool is_leading_sign = false;
                if (i + 1 < line_str.size()) {
                    char next_c = line_str[i + 1];
                    if (std::isdigit(static_cast<unsigned char>(next_c)) ||
                        (next_c == '.' && i + 2 < line_str.size() && std::isdigit(static_cast<unsigned char>(line_str[i + 2])))) {
                        if (tokens.empty()) {
                            is_leading_sign = true;
                        } else {
                            const std::string& prev = tokens.back().text;
                            if (prev == ":" || prev == "=" || prev == "<=" || prev == ">=" || prev == "<" || prev == ">" ||
                                prev == "[" || prev == "(") {
                                is_leading_sign = true;
                            }
                        }
                    }
                }

                if (is_leading_sign) {
                    size_t start = i;
                    ++i;
                    while (i < line_str.size() && !std::isspace(static_cast<unsigned char>(line_str[i])) &&
                           line_str[i] != ':' && line_str[i] != '[' && line_str[i] != ']' &&
                           line_str[i] != '*' && line_str[i] != '^' && line_str[i] != '=' &&
                           line_str[i] != '<' && line_str[i] != '>' && line_str[i] != '/') {
                        if ((line_str[i] == 'e' || line_str[i] == 'E') && i + 1 < line_str.size() &&
                            (line_str[i + 1] == '+' || line_str[i + 1] == '-')) {
                            i += 2;
                        } else if (line_str[i] == '+' || line_str[i] == '-') {
                            break;
                        } else {
                            ++i;
                        }
                    }
                    tokens.push_back({line_str.substr(start, i - start), line_num});
                } else {
                    tokens.push_back({std::string(1, c), line_num});
                    ++i;
                }
            } else {
                size_t start = i;
                while (i < line_str.size() && !std::isspace(static_cast<unsigned char>(line_str[i])) &&
                       line_str[i] != ':' && line_str[i] != '[' && line_str[i] != ']' &&
                       line_str[i] != '*' && line_str[i] != '^' && line_str[i] != '=' &&
                       line_str[i] != '<' && line_str[i] != '>' && line_str[i] != '/') {
                    if ((line_str[i] == 'e' || line_str[i] == 'E') && i + 1 < line_str.size() &&
                        (line_str[i + 1] == '+' || line_str[i + 1] == '-')) {
                        i += 2;
                    } else if (line_str[i] == '+' || line_str[i] == '-') {
                        break;
                    } else {
                        ++i;
                    }
                }
                tokens.push_back({line_str.substr(start, i - start), line_num});
            }
        }
    }
    return tokens;
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

    std::vector<Token> tokens = tokenize_lp_stream(is);
    if (tokens.empty()) {
        return Status::parse_error(1, "Empty LP model input");
    }

    auto get_token_line = [&](size_t idx) -> uint64_t {
        if (idx < tokens.size()) return tokens[idx].line;
        return tokens.empty() ? 1 : tokens.back().line;
    };

    // Variable tracking
    std::unordered_map<std::string, VariableIndex> var_map;
    auto get_or_add_var = [&](const std::string& name) -> Result<VariableIndex> {
        auto it = var_map.find(name);
        if (it != var_map.end()) return it->second;
        auto res = model.add_variable(name, 0.0, kInfinity, VariableType::Continuous);
        if (!res.is_ok()) return res.status();
        VariableIndex idx = res.value();
        var_map[name] = idx;
        return idx;
    };

    LpSection current_section = LpSection::None;
    size_t t_idx = 0;

    auto is_section_header = [&](size_t idx, LpSection& next_sec, size_t& advance) -> bool {
        if (idx >= tokens.size()) return false;
        std::string u1 = to_upper(tokens[idx].text);

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
        if (u1 == "SUBJECT" && idx + 1 < tokens.size() && to_upper(tokens[idx + 1].text) == "TO") {
            next_sec = LpSection::SubjectTo;
            advance = 2;
            return true;
        }
        if (u1 == "SUCH" && idx + 1 < tokens.size() && to_upper(tokens[idx + 1].text) == "THAT") {
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
                if (t_idx + 1 < tokens.size() && tokens[t_idx + 1].text == ":") {
                    t_idx += 2;
                    continue;
                }

                // Check if start of quadratic part: '['
                if (tokens[t_idx].text == "[") {
                    uint64_t bracket_line = tokens[t_idx].line;
                    ++t_idx; // consume '['

                    // Parse quadratic polynomial P(x) until ']'
                    while (t_idx < tokens.size() && tokens[t_idx].text != "]") {
                        double sign = 1.0;
                        if (tokens[t_idx].text == "+") {
                            sign = 1.0;
                            ++t_idx;
                        } else if (tokens[t_idx].text == "-") {
                            sign = -1.0;
                            ++t_idx;
                        }

                        if (t_idx >= tokens.size() || tokens[t_idx].text == "]") break;

                        double coeff = sign;
                        double parsed_c = 0.0;
                        if (parse_strict_double(tokens[t_idx].text, parsed_c)) {
                            coeff = sign * parsed_c;
                            ++t_idx;
                        }

                        if (t_idx >= tokens.size() || tokens[t_idx].text == "]") {
                            return Status::parse_error(get_token_line(t_idx), "Unexpected end of quadratic objective expression");
                        }

                        // Variable name
                        std::string var1_name = tokens[t_idx].text;
                        uint64_t v1_line = tokens[t_idx].line;
                        ++t_idx;
                        auto v1_res = get_or_add_var(var1_name);
                        if (!v1_res.is_ok()) return v1_res.status();
                        VariableIndex v1 = v1_res.value();

                        // Could be var1 ^ 2, var1 * var1, or var1 * var2
                        if (t_idx < tokens.size() && tokens[t_idx].text == "^") {
                            uint64_t pow_line = tokens[t_idx].line;
                            ++t_idx; // consume '^'
                            if (t_idx >= tokens.size() || tokens[t_idx].text != "2") {
                                return Status::parse_error(pow_line, "Expected '2' after '^' in quadratic objective");
                            }
                            ++t_idx; // consume '2'
                            // Diagonal term: coeff * x_i^2 inside [ P(x) ] / 2
                            // Rule: diagonal coeff a in P -> Q_ii = a
                            auto qst = model.add_quadratic_coefficient(v1, v1, coeff);
                            if (!qst.is_ok()) return qst;
                        } else if (t_idx < tokens.size() && tokens[t_idx].text == "*") {
                            uint64_t mul_line = tokens[t_idx].line;
                            ++t_idx; // consume '*'
                            if (t_idx >= tokens.size()) {
                                return Status::parse_error(mul_line, "Expected variable name after '*' in quadratic objective");
                            }
                            std::string var2_name = tokens[t_idx].text;
                            ++t_idx;
                            auto v2_res = get_or_add_var(var2_name);
                            if (!v2_res.is_ok()) return v2_res.status();
                            VariableIndex v2 = v2_res.value();
                            if (v1 == v2) {
                                auto qst = model.add_quadratic_coefficient(v1, v1, coeff);
                                if (!qst.is_ok()) return qst;
                            } else {
                                // Off-diagonal term: coeff * x_i * x_j inside [ P(x) ] / 2
                                // Rule: algebraic coeff b in P -> Q_ij = b / 2
                                auto qst = model.add_quadratic_coefficient(v1, v2, coeff / 2.0);
                                if (!qst.is_ok()) return qst;
                            }
                        } else {
                            return Status::parse_error(v1_line, "Malformed quadratic objective term: " + var1_name);
                        }
                    }

                    if (t_idx >= tokens.size() || tokens[t_idx].text != "]") {
                        return Status::parse_error(bracket_line, "Missing closing ']' for quadratic objective");
                    }
                    uint64_t close_line = tokens[t_idx].line;
                    ++t_idx; // consume ']'

                    // CRITICAL MATHEMATICAL GATE: Divisor '/ 2' MUST be present immediately following ']'
                    if (t_idx + 1 >= tokens.size() || tokens[t_idx].text != "/" || tokens[t_idx + 1].text != "2") {
                        return Status::parse_error(close_line, "Quadratic objective syntax must strictly be [ P(x) ] / 2");
                    }
                    t_idx += 2; // consume '/' and '2'
                    continue;
                }

                // Linear objective term or objective constant
                double sign = 1.0;
                bool had_explicit_sign = false;
                if (tokens[t_idx].text == "+") {
                    sign = 1.0;
                    had_explicit_sign = true;
                    ++t_idx;
                    if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                        return Status::parse_error(tokens[t_idx].line, "Consecutive sign operators in objective");
                    }
                } else if (tokens[t_idx].text == "-") {
                    sign = -1.0;
                    had_explicit_sign = true;
                    ++t_idx;
                    if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                        return Status::parse_error(tokens[t_idx].line, "Consecutive sign operators in objective");
                    }
                }

                if (t_idx >= tokens.size()) break;

                // Peek if next token is section header
                LpSection test_sec;
                size_t test_adv = 0;
                if (is_section_header(t_idx, test_sec, test_adv)) {
                    break;
                }

                const std::string& tok_str = tokens[t_idx].text;
                if (had_explicit_sign && !tok_str.empty() && (tok_str[0] == '+' || tok_str[0] == '-')) {
                    return Status::parse_error(tokens[t_idx].line, "Malformed consecutive signs in objective: " + tok_str);
                }
                if (tok_str.size() >= 2 && (tok_str[0] == '+' || tok_str[0] == '-') && (tok_str[1] == '+' || tok_str[1] == '-')) {
                    return Status::parse_error(tokens[t_idx].line, "Malformed sign operator in token: " + tok_str);
                }

                bool looks_numeric = !tok_str.empty() && (
                    std::isdigit(static_cast<unsigned char>(tok_str[0])) ||
                    ((tok_str[0] == '+' || tok_str[0] == '-') && tok_str.size() > 1 &&
                     (std::isdigit(static_cast<unsigned char>(tok_str[1])) || tok_str[1] == '.')) ||
                    (tok_str[0] == '.' && tok_str.size() > 1 && std::isdigit(static_cast<unsigned char>(tok_str[1])))
                );

                double coeff = sign;
                double parsed_c = 0.0;
                bool has_explicit_num = false;

                if (looks_numeric) {
                    if (!parse_strict_double(tok_str, parsed_c)) {
                        return Status::parse_error(tokens[t_idx].line, "Malformed numeric literal in objective: " + tok_str);
                    }
                    coeff = sign * parsed_c;
                    has_explicit_num = true;
                    ++t_idx;
                }

                if (t_idx < tokens.size() && tokens[t_idx].text == "[") {
                    if (has_explicit_num && coeff != 1.0) {
                        return Status::parse_error(tokens[t_idx].line, "Leading scalar before '[' in quadratic objective is invalid");
                    }
                    continue;
                }

                // Check if this is an objective constant or variable
                if (has_explicit_num) {
                    bool is_var = false;
                    if (t_idx < tokens.size()) {
                        const std::string& nxt = tokens[t_idx].text;
                        if (nxt != "+" && nxt != "-" && nxt != "[" && !is_section_header(t_idx, test_sec, test_adv)) {
                            is_var = true;
                        }
                    }
                    if (!is_var) {
                        // Objective constant
                        auto ost = model.set_objective_offset(model.objective().offset + coeff);
                        if (!ost.is_ok()) return ost;
                        continue;
                    }
                }

                if (t_idx >= tokens.size()) {
                    return Status::parse_error(get_token_line(t_idx), "Expected variable in objective expression");
                }

                std::string var_name = tokens[t_idx].text;
                if (var_name.empty() || std::isdigit(static_cast<unsigned char>(var_name[0]))) {
                    return Status::parse_error(tokens[t_idx].line, "Invalid variable name in objective: " + var_name);
                }
                ++t_idx;
                auto v_res = get_or_add_var(var_name);
                if (!v_res.is_ok()) return v_res.status();
                auto ost = model.add_objective_term(v_res.value(), coeff);
                if (!ost.is_ok()) return ost;
                break;
            }

            case LpSection::SubjectTo: {
                // Check if constraint name: cname :
                std::string con_name;
                uint64_t con_line = tokens[t_idx].line;
                if (t_idx + 1 < tokens.size() && tokens[t_idx + 1].text == ":") {
                    con_name = tokens[t_idx].text;
                    con_line = tokens[t_idx].line;
                    t_idx += 2;
                } else {
                    con_name = "c" + std::to_string(model.num_constraints() + 1);
                }

                // Check for ranged constraint starting with LHS: num <= expr <= num
                double lhs_val = -kInfinity;
                bool has_lhs = false;
                if (t_idx + 1 < tokens.size() && (tokens[t_idx + 1].text == "<=" || tokens[t_idx + 1].text == "<")) {
                    double num = 0.0;
                    if (parse_strict_double(tokens[t_idx].text, num)) {
                        lhs_val = num;
                        has_lhs = true;
                        t_idx += 2; // consume num and '<='
                    }
                }

                // Parse linear expression of constraint
                std::vector<LinearTerm> terms;
                while (t_idx < tokens.size()) {
                    if (tokens[t_idx].text == "<=" || tokens[t_idx].text == ">=" || tokens[t_idx].text == "=" ||
                        tokens[t_idx].text == "<" || tokens[t_idx].text == ">") {
                        break;
                    }

                    double sign = 1.0;
                    bool had_explicit_sign = false;
                    if (tokens[t_idx].text == "+") {
                        sign = 1.0;
                        had_explicit_sign = true;
                        ++t_idx;
                        if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                            return Status::parse_error(tokens[t_idx].line, "Consecutive sign operators in constraint");
                        }
                    } else if (tokens[t_idx].text == "-") {
                        sign = -1.0;
                        had_explicit_sign = true;
                        ++t_idx;
                        if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                            return Status::parse_error(tokens[t_idx].line, "Consecutive sign operators in constraint");
                        }
                    }

                    if (t_idx >= tokens.size()) break;

                    const std::string& con_tok_str = tokens[t_idx].text;
                    if (had_explicit_sign && !con_tok_str.empty() && (con_tok_str[0] == '+' || con_tok_str[0] == '-')) {
                        return Status::parse_error(tokens[t_idx].line, "Malformed consecutive signs in constraint: " + con_tok_str);
                    }
                    if (con_tok_str.size() >= 2 && (con_tok_str[0] == '+' || con_tok_str[0] == '-') && (con_tok_str[1] == '+' || con_tok_str[1] == '-')) {
                        return Status::parse_error(tokens[t_idx].line, "Malformed sign operator in token: " + con_tok_str);
                    }

                    bool con_looks_numeric = !con_tok_str.empty() && (
                        std::isdigit(static_cast<unsigned char>(con_tok_str[0])) ||
                        ((con_tok_str[0] == '+' || con_tok_str[0] == '-') && con_tok_str.size() > 1 &&
                         (std::isdigit(static_cast<unsigned char>(con_tok_str[1])) || con_tok_str[1] == '.')) ||
                        (con_tok_str[0] == '.' && con_tok_str.size() > 1 && std::isdigit(static_cast<unsigned char>(con_tok_str[1])))
                    );

                    double coeff = sign;
                    double parsed_c = 0.0;
                    if (con_looks_numeric) {
                        if (!parse_strict_double(con_tok_str, parsed_c)) {
                            return Status::parse_error(tokens[t_idx].line, "Malformed numeric literal in constraint: " + con_tok_str);
                        }
                        coeff = sign * parsed_c;
                        ++t_idx;
                    }

                    if (t_idx >= tokens.size()) break;
                    std::string var_name = tokens[t_idx].text;
                    if (var_name.empty() || std::isdigit(static_cast<unsigned char>(var_name[0]))) {
                        return Status::parse_error(tokens[t_idx].line, "Invalid variable name in constraint: " + var_name);
                    }
                    ++t_idx;
                    auto v_res = get_or_add_var(var_name);
                    if (!v_res.is_ok()) return v_res.status();
                    terms.emplace_back(v_res.value(), coeff);
                }

                if (t_idx >= tokens.size()) {
                    return Status::parse_error(con_line, "Unexpected end of constraint: " + con_name);
                }

                std::string op = tokens[t_idx].text;
                uint64_t op_line = tokens[t_idx].line;
                ++t_idx;

                double rhs_val = 0.0;
                double sign_rhs = 1.0;
                if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                    if (tokens[t_idx].text == "-") sign_rhs = -1.0;
                    ++t_idx;
                }
                if (t_idx >= tokens.size() || !parse_strict_double(tokens[t_idx].text, rhs_val)) {
                    return Status::parse_error(op_line, "Expected numeric RHS for constraint: " + con_name);
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
                if (!add_con_res.is_ok()) {
                    return add_con_res.status();
                }
                break;
            }

            case LpSection::Bounds: {
                // Patterns:
                // 1) num1 <= var <= num2
                // 2) num1 <= var
                // 3) var free
                // 4) var op val
                double num1 = 0.0;
                double num1_sign = 1.0;
                size_t p = t_idx;
                if (tokens[p].text == "+" || tokens[p].text == "-") {
                    if (tokens[p].text == "-") num1_sign = -1.0;
                    ++p;
                }

                if (p < tokens.size() && (parse_strict_double(tokens[p].text, num1) ||
                                          to_upper(tokens[p].text) == "INFINITY" ||
                                          to_upper(tokens[p].text) == "INF")) {
                    bool is_inf = (to_upper(tokens[p].text) == "INFINITY" || to_upper(tokens[p].text) == "INF");
                    num1 = is_inf ? (num1_sign > 0 ? kInfinity : -kInfinity) : (num1 * num1_sign);
                    t_idx = p + 1; // consumed num1

                    if (t_idx < tokens.size() && (tokens[t_idx].text == "<=" || tokens[t_idx].text == "<")) {
                        ++t_idx; // consumed '<='
                        if (t_idx >= tokens.size()) {
                            return Status::parse_error(get_token_line(t_idx), "Expected variable name in bound declaration");
                        }
                        std::string var_name = tokens[t_idx].text;
                        ++t_idx;
                        auto v_res = get_or_add_var(var_name);
                        if (!v_res.is_ok()) return v_res.status();
                        VariableIndex v_idx = v_res.value();

                        if (t_idx < tokens.size() && (tokens[t_idx].text == "<=" || tokens[t_idx].text == "<")) {
                            ++t_idx; // consumed '<='
                            double num2 = 0.0;
                            double num2_sign = 1.0;
                            if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                                if (tokens[t_idx].text == "-") num2_sign = -1.0;
                                ++t_idx;
                            }
                            if (t_idx >= tokens.size()) {
                                return Status::parse_error(get_token_line(t_idx), "Expected upper bound value");
                            }
                            if (to_upper(tokens[t_idx].text) == "INFINITY" || to_upper(tokens[t_idx].text) == "INF") {
                                num2 = num2_sign > 0 ? kInfinity : -kInfinity;
                            } else if (!parse_strict_double(tokens[t_idx].text, num2)) {
                                return Status::parse_error(tokens[t_idx].line, "Invalid upper bound value: " + tokens[t_idx].text);
                            } else {
                                num2 *= num2_sign;
                            }
                            ++t_idx;
                            auto st = model.set_variable_bounds(v_idx, num1, num2);
                            if (!st.is_ok()) return st;
                        } else {
                            // num1 <= var  =>  var >= num1
                            auto st = model.set_variable_bounds(v_idx, num1, model.get_variable(v_idx).upper_bound);
                            if (!st.is_ok()) return st;
                        }
                    }
                    break;
                }

                // Starts with variable name
                std::string var_name = tokens[t_idx].text;
                uint64_t vline = tokens[t_idx].line;
                ++t_idx;
                auto v_res = get_or_add_var(var_name);
                if (!v_res.is_ok()) return v_res.status();
                VariableIndex v_idx = v_res.value();

                if (t_idx < tokens.size() && to_upper(tokens[t_idx].text) == "FREE") {
                    ++t_idx;
                    auto st = model.set_variable_bounds(v_idx, -kInfinity, kInfinity);
                    if (!st.is_ok()) return st;
                    break;
                }

                if (t_idx < tokens.size()) {
                    std::string op = tokens[t_idx].text;
                    uint64_t op_line = tokens[t_idx].line;
                    ++t_idx;
                    double bval = 0.0;
                    double bsign = 1.0;
                    if (t_idx < tokens.size() && (tokens[t_idx].text == "+" || tokens[t_idx].text == "-")) {
                        if (tokens[t_idx].text == "-") bsign = -1.0;
                        ++t_idx;
                    }
                    if (t_idx >= tokens.size()) {
                        return Status::parse_error(op_line, "Expected bound value after operator: " + op);
                    }
                    if (to_upper(tokens[t_idx].text) == "INFINITY" || to_upper(tokens[t_idx].text) == "INF") {
                        bval = bsign > 0 ? kInfinity : -kInfinity;
                    } else if (!parse_strict_double(tokens[t_idx].text, bval)) {
                        return Status::parse_error(tokens[t_idx].line, "Invalid bound value: " + tokens[t_idx].text);
                    } else {
                        bval *= bsign;
                    }
                    ++t_idx;

                    if (op == "<=" || op == "<") {
                        auto st = model.set_variable_bounds(v_idx, model.get_variable(v_idx).lower_bound, bval);
                        if (!st.is_ok()) return st;
                    } else if (op == ">=" || op == ">") {
                        auto st = model.set_variable_bounds(v_idx, bval, model.get_variable(v_idx).upper_bound);
                        if (!st.is_ok()) return st;
                    } else if (op == "=") {
                        auto st = model.set_variable_bounds(v_idx, bval, bval);
                        if (!st.is_ok()) return st;
                    } else {
                        return Status::parse_error(op_line, "Unrecognized bound operator: " + op);
                    }
                } else {
                    return Status::parse_error(vline, "Incomplete bound line for variable: " + var_name);
                }
                break;
            }

            case LpSection::Generals: {
                std::string var_name = tokens[t_idx].text;
                ++t_idx;
                auto v_res = get_or_add_var(var_name);
                if (!v_res.is_ok()) return v_res.status();
                auto st = model.set_variable_type(v_res.value(), VariableType::Integer);
                if (!st.is_ok()) return st;
                break;
            }

            case LpSection::Binaries: {
                std::string var_name = tokens[t_idx].text;
                ++t_idx;
                auto v_res = get_or_add_var(var_name);
                if (!v_res.is_ok()) return v_res.status();
                auto bst = model.set_variable_bounds(v_res.value(), 0.0, 1.0);
                if (!bst.is_ok()) return bst;
                auto tst = model.set_variable_type(v_res.value(), VariableType::Binary);
                if (!tst.is_ok()) return tst;
                break;
            }

            default:
                return Status::parse_error(tokens[t_idx].line, "Unexpected token outside valid section: " + tokens[t_idx].text);
        }
    }

    auto val_status = model.validate();
    if (!val_status.is_ok()) {
        return val_status;
    }

    return model;
}

} // namespace sih26119
