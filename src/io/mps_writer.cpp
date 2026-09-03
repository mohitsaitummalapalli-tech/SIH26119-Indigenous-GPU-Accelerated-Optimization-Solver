#include "io/mps_writer.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <map>

namespace sih26119 {

namespace {

void format_mps_field(std::ostream& os, const std::string& str, int width) {
    os << std::left << std::setw(width) << str;
}

void format_mps_num(std::ostream& os, double val) {
    std::ostringstream ss;
    ss << std::setprecision(16) << val;
    os << std::left << std::setw(12) << ss.str();
}

} // anonymous namespace

Status MpsWriter::write_file(const Model& model, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return Status::error(StatusCode::IOError, "Failed to open file for writing: " + filepath);
    }
    return write_stream(model, file);
}

Status MpsWriter::write_stream(const Model& model, std::ostream& os) {
    Status val_status = model.validate();
    if (!val_status.ok()) {
        return val_status;
    }

    // 1. NAME
    std::string model_name = model.name().empty() ? "SIH26119" : model.name();
    os << "NAME          " << model_name << "\n";

    // 2. OBJSENSE
    os << "OBJSENSE\n";
    if (model.objective().sense == ObjectiveSense::Maximize) {
        os << "  MAX\n";
    } else {
        os << "  MIN\n";
    }

    // 3. ROWS
    os << "ROWS\n";
    std::string obj_row_name = "OBJ";
    os << " N  " << obj_row_name << "\n";

    for (const auto& con : model.constraints()) {
        char rtype = 'E';
        if (con.is_equality()) {
            rtype = 'E';
        } else if (con.is_range()) {
            rtype = 'E'; // Handled via RANGES
        } else if (con.is_greater_equal()) {
            rtype = 'G';
        } else if (con.is_less_equal()) {
            rtype = 'L';
        } else if (con.is_free()) {
            rtype = 'N';
        }
        os << " " << rtype << "  " << con.name << "\n";
    }

    // 4. COLUMNS
    os << "COLUMNS\n";
    const auto& vars = model.variables();
    const auto& cons = model.constraints();

    // Map column entries: var_idx -> list of (row_name, coeff)
    std::map<VariableIndex, std::vector<std::pair<std::string, double>>> col_entries;

    // Add objective coefficients
    for (const auto& term : model.objective().linear_terms) {
        if (term.coefficient != 0.0) {
            col_entries[term.variable_index].emplace_back(obj_row_name, term.coefficient);
        }
    }

    // Add constraint coefficients
    for (const auto& con : cons) {
        for (const auto& term : con.terms) {
            if (term.coefficient != 0.0) {
                col_entries[term.variable_index].emplace_back(con.name, term.coefficient);
            }
        }
    }

    bool int_mode = false;
    for (VariableIndex j = 0; j < vars.size(); ++j) {
        const auto& var = vars[j];
        if (var.is_integer() && !int_mode) {
            os << "    MARK0000  'MARKER'                 'INTORG'\n";
            int_mode = true;
        } else if (!var.is_integer() && int_mode) {
            os << "    MARK0001  'MARKER'                 'INTEND'\n";
            int_mode = false;
        }

        auto it = col_entries.find(j);
        if (it != col_entries.end()) {
            const auto& entries = it->second;
            for (size_t k = 0; k < entries.size(); k += 2) {
                os << "    ";
                format_mps_field(os, var.name, 10);
                format_mps_field(os, entries[k].first, 10);
                format_mps_num(os, entries[k].second);

                if (k + 1 < entries.size()) {
                    format_mps_field(os, entries[k + 1].first, 10);
                    format_mps_num(os, entries[k + 1].second);
                }
                os << "\n";
            }
        } else {
            // Variable with 0 coefficients in all rows
            os << "    ";
            format_mps_field(os, var.name, 10);
            format_mps_field(os, obj_row_name, 10);
            format_mps_num(os, 0.0);
            os << "\n";
        }
    }
    if (int_mode) {
        os << "    MARK0002  'MARKER'                 'INTEND'\n";
    }

    // 5. RHS
    os << "RHS\n";
    for (const auto& con : cons) {
        double rhs_val = 0.0;
        bool has_rhs = false;

        if (con.is_equality()) {
            rhs_val = con.lower_bound;
            has_rhs = true;
        } else if (con.is_range()) {
            rhs_val = con.lower_bound; // with range r = ub - lb
            has_rhs = true;
        } else if (con.is_greater_equal()) {
            rhs_val = con.lower_bound;
            has_rhs = true;
        } else if (con.is_less_equal()) {
            rhs_val = con.upper_bound;
            has_rhs = true;
        }

        if (has_rhs && rhs_val != 0.0) {
            os << "    RHS1      ";
            format_mps_field(os, con.name, 10);
            format_mps_num(os, rhs_val);
            os << "\n";
        }
    }

    // 6. RANGES (only if range constraints exist)
    bool has_ranges = false;
    for (const auto& con : cons) {
        if (con.is_range()) {
            has_ranges = true;
            break;
        }
    }
    if (has_ranges) {
        os << "RANGES\n";
        for (const auto& con : cons) {
            if (con.is_range()) {
                double r_val = con.upper_bound - con.lower_bound;
                os << "    RNG1      ";
                format_mps_field(os, con.name, 10);
                format_mps_num(os, r_val);
                os << "\n";
            }
        }
    }

    // 7. BOUNDS
    os << "BOUNDS\n";
    for (const auto& var : vars) {
        if (var.type == VariableType::Binary) {
            os << " BV BND1      " << var.name << "\n";
        } else if (var.is_free()) {
            os << " FR BND1      " << var.name << "\n";
        } else if (var.is_fixed()) {
            os << " FX BND1      ";
            format_mps_field(os, var.name, 10);
            format_mps_num(os, var.lower_bound);
            os << "\n";
        } else {
            // Lower bound
            if (var.lower_bound != 0.0 && !std::isinf(var.lower_bound)) {
                os << (var.is_integer() ? " LI BND1      " : " LO BND1      ");
                format_mps_field(os, var.name, 10);
                format_mps_num(os, var.lower_bound);
                os << "\n";
            } else if (std::isinf(var.lower_bound) && var.lower_bound < 0.0) {
                os << " MI BND1      " << var.name << "\n";
            }

            // Upper bound
            if (!std::isinf(var.upper_bound)) {
                os << (var.is_integer() ? " UI BND1      " : " UP BND1      ");
                format_mps_field(os, var.name, 10);
                format_mps_num(os, var.upper_bound);
                os << "\n";
            }
        }
    }

    // 8. QUADOBJ (if QP)
    if (model.objective().is_quadratic()) {
        os << "QUADOBJ\n";
        for (const auto& qterm : model.objective().quadratic_terms) {
            const std::string& n1 = vars[qterm.var1].name;
            const std::string& n2 = vars[qterm.var2].name;
            os << "    ";
            format_mps_field(os, n1, 10);
            format_mps_field(os, n2, 10);
            format_mps_num(os, qterm.coefficient);
            os << "\n";
        }
    }

    // 9. ENDATA
    os << "ENDATA\n";
    return Status::ok();
}

} // namespace sih26119
