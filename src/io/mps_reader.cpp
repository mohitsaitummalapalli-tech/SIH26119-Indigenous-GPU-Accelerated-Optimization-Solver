#include "io/mps_reader.hpp"
#include "core/numeric_parser.hpp"
#include <fstream>
#include <vector>
#include <cctype>
#include <cmath>
#include <map>

namespace sih26119 {

namespace {

std::vector<std::string> tokenize_line(std::string_view line) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) {
            ++i;
        }
        if (i >= line.size()) break;
        if (line[i] == '$') break; // Inline comment
        size_t start = i;
        while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i])) && line[i] != '$') {
            ++i;
        }
        tokens.emplace_back(line.substr(start, i - start));
    }
    return tokens;
}

enum class MpsSection {
    None,
    Name,
    Objsense,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Quadobj,
    Endata
};

struct RawRow {
    char type{' '};
    std::string name;
};

} // anonymous namespace

Result<Model> MpsReader::read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Status::error(StatusCode::IOError, "Failed to open MPS file: " + filepath);
    }
    return read_stream(file, filepath);
}

Result<Model> MpsReader::read_stream(std::istream& is, const std::string& sourcename) {
    (void)sourcename;
    Model model;
    std::string line;
    uint64_t line_number = 0;

    MpsSection section = MpsSection::None;
    std::string obj_name;
    bool has_explicit_obj = false;

    std::vector<RawRow> raw_rows;
    std::unordered_map<std::string, size_t> row_name_to_idx;

    // Track active vector names for multiple sets
    std::string active_rhs_set;
    std::string active_ranges_set;
    std::string active_bounds_set;

    bool in_integer_marker = false;

    // Intermediate RHS and Range values per constraint name
    std::unordered_map<std::string, double> rhs_values;
    std::unordered_map<std::string, double> range_values;

    // Intermediate Column coefficients: con_name -> list of (var_idx, coeff)
    std::unordered_map<std::string, std::vector<LinearTerm>> con_terms;

    struct ColRecord {
        std::string var_name;
        std::string row_name;
        double value;
    };
    std::vector<ColRecord> col_records;

    // Map of variables encountered
    std::vector<std::string> var_order;
    std::unordered_map<std::string, VariableType> var_types;
    std::unordered_map<std::string, double> var_lb;
    std::unordered_map<std::string, double> var_ub;

    struct BoundTracker {
        bool has_lo = false;
        bool has_up = false;
        bool has_fx = false;
        bool has_fr = false;
        bool has_bv = false;
        bool has_ui = false;
        bool has_li = false;
    };
    std::unordered_map<std::string, BoundTracker> bound_trackers;

    // QUADOBJ storage: (var1, var2) -> value
    std::map<std::pair<std::string, std::string>, double> quadobj_entries;

    while (std::getline(is, line)) {
        ++line_number;
        if (line.empty() || line[0] == '*' || line[0] == '$') {
            continue; // Skip comments and empty lines
        }

        // Section header lines have no leading whitespace
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            auto tokens = tokenize_line(line);
            if (tokens.empty()) continue;
            const std::string& header = tokens[0];

            if (header == "NAME") {
                section = MpsSection::Name;
                if (tokens.size() > 1) {
                    model.set_name(tokens[1]);
                }
            } else if (header == "OBJSENSE") {
                section = MpsSection::Objsense;
            } else if (header == "ROWS") {
                section = MpsSection::Rows;
            } else if (header == "COLUMNS") {
                section = MpsSection::Columns;
            } else if (header == "RHS") {
                section = MpsSection::Rhs;
            } else if (header == "RANGES") {
                section = MpsSection::Ranges;
            } else if (header == "BOUNDS") {
                section = MpsSection::Bounds;
            } else if (header == "QUADOBJ") {
                section = MpsSection::Quadobj;
            } else if (header == "QCMATRIX") {
                return Status::parse_error(line_number, "QCMATRIX is unsupported in Phase 1 solver core");
            } else if (header == "SOS") {
                return Status::parse_error(line_number, "SOS section is unsupported in Phase 1 solver core");
            } else if (header == "ENDATA") {
                section = MpsSection::Endata;
                break;
            } else {
                return Status::parse_error(line_number, "Unrecognized or unsupported MPS section: " + header);
            }
            continue;
        }

        // Data lines within sections
        auto tokens = tokenize_line(line);
        if (tokens.empty()) continue;

        switch (section) {
            case MpsSection::Objsense: {
                std::string sense_str = tokens[0];
                for (auto& c : sense_str) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                if (sense_str == "MIN" || sense_str == "MINIMIZE") {
                    model.set_objective_sense(ObjectiveSense::Minimize);
                } else if (sense_str == "MAX" || sense_str == "MAXIMIZE") {
                    model.set_objective_sense(ObjectiveSense::Maximize);
                } else {
                    return Status::parse_error(line_number, "Invalid OBJSENSE value: " + tokens[0]);
                }
                break;
            }

            case MpsSection::Rows: {
                if (tokens.size() < 2) {
                    return Status::parse_error(line_number, "Malformed row declaration in ROWS section");
                }
                char rtype = tokens[0][0];
                const std::string& rname = tokens[1];
                if (rtype != 'N' && rtype != 'G' && rtype != 'L' && rtype != 'E') {
                    return Status::parse_error(line_number, "Invalid row type: " + tokens[0]);
                }
                if (row_name_to_idx.find(rname) != row_name_to_idx.end()) {
                    return Status::parse_error(line_number, "Duplicate row name in ROWS: " + rname);
                }

                if (rtype == 'N' && !has_explicit_obj) {
                    obj_name = rname;
                    has_explicit_obj = true;
                }

                row_name_to_idx[rname] = raw_rows.size();
                raw_rows.push_back({rtype, rname});
                break;
            }

            case MpsSection::Columns: {
                // COLUMNS format:
                // VARNAME  ROWNAME1  VAL1  [ROWNAME2  VAL2]
                // OR 'MARKER' record
                if (tokens.size() >= 3 && tokens[1] == "'MARKER'") {
                    std::string marker_type = tokens[2];
                    if (marker_type == "'INTORG'") {
                        in_integer_marker = true;
                    } else if (marker_type == "'INTEND'") {
                        in_integer_marker = false;
                    }
                    break;
                }

                if (tokens.size() < 3) {
                    return Status::parse_error(line_number, "Incomplete entry in COLUMNS section");
                }

                const std::string& var_name = tokens[0];
                if (var_types.find(var_name) == var_types.end()) {
                    var_order.push_back(var_name);
                    var_types[var_name] = in_integer_marker ? VariableType::Integer : VariableType::Continuous;
                    var_lb[var_name] = 0.0;
                    var_ub[var_name] = kInfinity;
                }

                for (size_t k = 1; k + 1 < tokens.size(); k += 2) {
                    const std::string& rname = tokens[k];
                    double val = 0.0;
                    if (!parse_strict_double(tokens[k + 1], val)) {
                        return Status::parse_error(line_number, "Invalid numeric value in COLUMNS: " + tokens[k + 1]);
                    }

                    if (row_name_to_idx.find(rname) == row_name_to_idx.end()) {
                        return Status::parse_error(line_number, "Referenced row in COLUMNS not defined in ROWS: " + rname);
                    }

                    col_records.push_back({var_name, rname, val});
                }
                break;
            }

            case MpsSection::Rhs: {
                // RHS format: [RHS_NAME] ROWNAME1 VAL1 [ROWNAME2 VAL2]
                size_t offset = 0;
                if (tokens.size() % 2 == 1) {
                    if (active_rhs_set.empty()) {
                        active_rhs_set = tokens[0];
                    }
                    if (tokens[0] != active_rhs_set) {
                        break; // Skip non-active RHS vector
                    }
                    offset = 1;
                }

                for (size_t k = offset; k + 1 < tokens.size(); k += 2) {
                    const std::string& rname = tokens[k];
                    double val = 0.0;
                    if (!parse_strict_double(tokens[k + 1], val)) {
                        return Status::parse_error(line_number, "Invalid numeric value in RHS: " + tokens[k + 1]);
                    }
                    if (rname == obj_name) {
                        auto st = model.set_objective_offset(val);
                        if (!st.is_ok()) return st;
                    } else {
                        if (row_name_to_idx.find(rname) == row_name_to_idx.end()) {
                            return Status::parse_error(line_number, "Referenced row in RHS not defined in ROWS: " + rname);
                        }
                        rhs_values[rname] = val;
                    }
                }
                break;
            }

            case MpsSection::Ranges: {
                // RANGES format: [RANGES_NAME] ROWNAME1 VAL1 [ROWNAME2 VAL2]
                size_t offset = 0;
                if (tokens.size() % 2 == 1) {
                    if (active_ranges_set.empty()) {
                        active_ranges_set = tokens[0];
                    }
                    if (tokens[0] != active_ranges_set) {
                        break; // Skip non-active RANGES vector
                    }
                    offset = 1;
                }

                for (size_t k = offset; k + 1 < tokens.size(); k += 2) {
                    const std::string& rname = tokens[k];
                    double val = 0.0;
                    if (!parse_strict_double(tokens[k + 1], val)) {
                        return Status::parse_error(line_number, "Invalid numeric value in RANGES: " + tokens[k + 1]);
                    }
                    if (row_name_to_idx.find(rname) == row_name_to_idx.end()) {
                        return Status::parse_error(line_number, "Referenced row in RANGES not defined in ROWS: " + rname);
                    }
                    range_values[rname] = val;
                }
                break;
            }

            case MpsSection::Bounds: {
                // BOUNDS format: BOUND_TYPE [BOUND_SET] VARNAME [VALUE]
                if (tokens.size() < 2) {
                    return Status::parse_error(line_number, "Incomplete BOUNDS line");
                }
                std::string btype = tokens[0];
                size_t var_idx = 1;
                double dummy_val = 0.0;
                if (tokens.size() >= 3 && !parse_strict_double(tokens[2], dummy_val)) { // If token 2 is not number, token 1 is set name
                    if (active_bounds_set.empty()) {
                        active_bounds_set = tokens[1];
                    }
                    if (tokens[1] != active_bounds_set) {
                        break; // Skip non-active BOUNDS vector
                    }
                    var_idx = 2;
                }

                const std::string& var_name = tokens[var_idx];
                if (var_types.find(var_name) == var_types.end()) {
                    var_order.push_back(var_name);
                    var_types[var_name] = VariableType::Continuous;
                    var_lb[var_name] = 0.0;
                    var_ub[var_name] = kInfinity;
                }

                double bval = 0.0;
                if (tokens.size() > var_idx + 1) {
                    if (!parse_strict_double(tokens[var_idx + 1], bval)) {
                        return Status::parse_error(line_number, "Invalid bound value: " + tokens[var_idx + 1]);
                    }
                }

                auto& bt = bound_trackers[var_name];

                if (btype == "UP") {
                    if (bt.has_up) {
                        return Status::parse_error(line_number, "Duplicate UP bound specified for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting UP bound on already fixed (FX) variable: " + var_name);
                    }
                    if (bt.has_fr) {
                        return Status::parse_error(line_number, "Conflicting UP bound on already free (FR) variable: " + var_name);
                    }
                    bt.has_up = true;
                    var_ub[var_name] = bval;
                } else if (btype == "LO") {
                    if (bt.has_lo) {
                        return Status::parse_error(line_number, "Duplicate LO bound specified for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting LO bound on already fixed (FX) variable: " + var_name);
                    }
                    if (bt.has_fr) {
                        return Status::parse_error(line_number, "Conflicting LO bound on already free (FR) variable: " + var_name);
                    }
                    bt.has_lo = true;
                    var_lb[var_name] = bval;
                } else if (btype == "FX") {
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Duplicate FX bound specified for variable: " + var_name);
                    }
                    if (bt.has_lo) {
                        return Status::parse_error(line_number, "Conflicting FX bound on variable with prior LO bound: " + var_name);
                    }
                    if (bt.has_up) {
                        return Status::parse_error(line_number, "Conflicting FX bound on variable with prior UP bound: " + var_name);
                    }
                    if (bt.has_fr) {
                        return Status::parse_error(line_number, "Conflicting FX bound on already free (FR) variable: " + var_name);
                    }
                    bt.has_fx = true;
                    bt.has_lo = true;
                    bt.has_up = true;
                    var_lb[var_name] = bval;
                    var_ub[var_name] = bval;
                } else if (btype == "FR") {
                    if (bt.has_fr) {
                        return Status::parse_error(line_number, "Duplicate FR bound specified for variable: " + var_name);
                    }
                    if (bt.has_lo || bt.has_up || bt.has_fx || bt.has_bv) {
                        return Status::parse_error(line_number, "Conflicting FR bound on variable with prior bound specifications: " + var_name);
                    }
                    bt.has_fr = true;
                    var_lb[var_name] = -kInfinity;
                    var_ub[var_name] = kInfinity;
                } else if (btype == "MI") {
                    if (bt.has_lo) {
                        return Status::parse_error(line_number, "Duplicate or conflicting lower bound (MI) for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting MI bound on fixed variable: " + var_name);
                    }
                    bt.has_lo = true;
                    var_lb[var_name] = -kInfinity;
                } else if (btype == "PL") {
                    if (bt.has_up) {
                        return Status::parse_error(line_number, "Duplicate or conflicting upper bound (PL) for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting PL bound on fixed variable: " + var_name);
                    }
                    bt.has_up = true;
                    var_ub[var_name] = kInfinity;
                } else if (btype == "BV") {
                    if (bt.has_bv || bt.has_fx || bt.has_lo || bt.has_up) {
                        return Status::parse_error(line_number, "Conflicting BV bound declaration on variable: " + var_name);
                    }
                    bt.has_bv = true;
                    bt.has_lo = true;
                    bt.has_up = true;
                    var_types[var_name] = VariableType::Binary;
                    var_lb[var_name] = 0.0;
                    var_ub[var_name] = 1.0;
                } else if (btype == "UI") {
                    if (bt.has_up || bt.has_ui) {
                        return Status::parse_error(line_number, "Duplicate upper bound (UI) for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting UI bound on fixed variable: " + var_name);
                    }
                    bt.has_up = true;
                    bt.has_ui = true;
                    var_types[var_name] = VariableType::Integer;
                    var_ub[var_name] = bval;
                } else if (btype == "LI") {
                    if (bt.has_lo || bt.has_li) {
                        return Status::parse_error(line_number, "Duplicate lower bound (LI) for variable: " + var_name);
                    }
                    if (bt.has_fx) {
                        return Status::parse_error(line_number, "Conflicting LI bound on fixed variable: " + var_name);
                    }
                    bt.has_lo = true;
                    bt.has_li = true;
                    var_types[var_name] = VariableType::Integer;
                    var_lb[var_name] = bval;
                } else {
                    return Status::parse_error(line_number, "Unrecognized bound type: " + btype);
                }
                break;
            }

            case MpsSection::Quadobj: {
                // QUADOBJ format: VAR1  VAR2  VALUE
                if (tokens.size() < 3) {
                    return Status::parse_error(line_number, "Malformed QUADOBJ entry");
                }
                const std::string& v1 = tokens[0];
                const std::string& v2 = tokens[1];
                double val = 0.0;
                if (!parse_strict_double(tokens[2], val)) {
                    return Status::parse_error(line_number, "Invalid numeric value in QUADOBJ: " + tokens[2]);
                }

                std::string s1 = v1 <= v2 ? v1 : v2;
                std::string s2 = v1 <= v2 ? v2 : v1;
                auto key = std::make_pair(s1, s2);

                auto it = quadobj_entries.find(key);
                if (it != quadobj_entries.end()) {
                    if (std::abs(it->second - val) > 1e-9) {
                        return Status::parse_error(line_number, "Conflicting symmetric QUADOBJ values for " + v1 + ", " + v2);
                    }
                } else {
                    quadobj_entries[key] = val;
                }
                break;
            }

            default:
                break;
        }
    }

    // Register all variables in Model
    for (const auto& var_name : var_order) {
        double lb = var_lb[var_name];
        double ub = var_ub[var_name];
        VariableType vtype = var_types[var_name];
        auto res = model.add_variable(var_name, lb, ub, vtype);
        if (!res.is_ok()) {
            return res.status();
        }
    }

    // Populate constraint linear terms and objective from col_records
    for (const auto& entry : col_records) {
        auto var_idx_res = model.get_variable_index(entry.var_name);
        if (!var_idx_res.is_ok()) continue;
        VariableIndex v_idx = var_idx_res.value();

        if (entry.row_name == obj_name) {
            auto st = model.add_objective_term(v_idx, entry.value);
            if (!st.is_ok()) return st;
        } else {
            con_terms[entry.row_name].emplace_back(v_idx, entry.value);
        }
    }

    // Create Constraints in Model according to ROWS and RANGES semantics
    for (const auto& r : raw_rows) {
        if (r.name == obj_name) {
            continue; // Objective row
        }

        double rhs = 0.0;
        auto rhs_it = rhs_values.find(r.name);
        if (rhs_it != rhs_values.end()) {
            rhs = rhs_it->second;
        }

        double lb = -kInfinity;
        double ub = kInfinity;

        auto range_it = range_values.find(r.name);
        bool has_range = (range_it != range_values.end());
        double r_val = has_range ? range_it->second : 0.0;
        double q_val = std::abs(r_val);

        switch (r.type) {
            case 'G':
                if (has_range) {
                    // Authoritative G range: [b, b + |r|]
                    lb = rhs;
                    ub = rhs + q_val;
                } else {
                    lb = rhs;
                    ub = kInfinity;
                }
                break;

            case 'L':
                if (has_range) {
                    // Authoritative L range: [b - |r|, b]
                    lb = rhs - q_val;
                    ub = rhs;
                } else {
                    lb = -kInfinity;
                    ub = rhs;
                }
                break;

            case 'E':
                if (has_range) {
                    // Authoritative E range:
                    // r > 0 -> [b, b + r]
                    // r < 0 -> [b + r, b]
                    // r = 0 -> [b, b]
                    if (r_val > 0.0) {
                        lb = rhs;
                        ub = rhs + r_val;
                    } else if (r_val < 0.0) {
                        lb = rhs + r_val;
                        ub = rhs;
                    } else {
                        lb = rhs;
                        ub = rhs;
                    }
                } else {
                    lb = rhs;
                    ub = rhs;
                }
                break;

            case 'N':
                // Free row
                lb = -kInfinity;
                ub = kInfinity;
                break;

            default:
                break;
        }

        std::vector<LinearTerm> terms = con_terms[r.name];
        auto add_con_res = model.add_constraint(r.name, lb, ub, std::move(terms));
        if (!add_con_res.ok()) {
            return add_con_res.status();
        }
    }

    // Populate QUADOBJ terms in Model
    for (const auto& [pair, val] : quadobj_entries) {
        auto v1_res = model.get_variable_index(pair.first);
        auto v2_res = model.get_variable_index(pair.second);
        if (!v1_res.ok() || !v2_res.ok()) {
            return Status::error(StatusCode::InvalidVariableReference, "QUADOBJ references undefined variable");
        }
        auto q_status = model.set_quadratic_coefficient(v1_res.value(), v2_res.value(), val);
        if (!q_status.is_ok()) {
            return q_status;
        }
    }

    auto val_status = model.validate();
    if (!val_status.is_ok()) {
        return val_status;
    }

    return model;
}

} // namespace sih26119
