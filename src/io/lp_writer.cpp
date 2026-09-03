#include "io/lp_writer.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>

namespace sih26119 {

namespace {

void write_term(std::ostream& os, double coeff, const std::string& var_name, bool first_term) {
    if (coeff == 0.0) return;

    if (coeff < 0.0) {
        os << (first_term ? "- " : " - ");
        coeff = -coeff;
    } else {
        if (!first_term) os << " + ";
    }

    if (coeff != 1.0) {
        os << std::setprecision(16) << coeff << " ";
    }
    os << var_name;
}

} // anonymous namespace

Status LpWriter::write_file(const Model& model, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return Status::error(StatusCode::IOError, "Failed to open LP file for writing: " + filepath);
    }
    return write_stream(model, file);
}

Status LpWriter::write_stream(const Model& model, std::ostream& os) {
    Status val_status = model.validate();
    if (!val_status.ok()) {
        return val_status;
    }

    const auto& vars = model.variables();
    const auto& cons = model.constraints();
    const auto& obj = model.objective();

    // 1. Objective Sense
    if (obj.sense == ObjectiveSense::Maximize) {
        os << "Maximize\n";
    } else {
        os << "Minimize\n";
    }

    // Linear objective terms
    os << " obj: ";
    bool first_obj = true;
    for (const auto& term : obj.linear_terms) {
        if (term.coefficient != 0.0) {
            write_term(os, term.coefficient, vars[term.variable_index].name, first_obj);
            first_obj = false;
        }
    }
    if (first_obj && !obj.is_quadratic()) {
        os << "0 " << (vars.empty() ? "x" : vars[0].name);
    }

    // Quadratic objective terms written inside strictly [ P(x) ] / 2
    if (obj.is_quadratic()) {
        os << " + [ ";
        bool first_q = true;
        for (const auto& qterm : obj.quadratic_terms) {
            const std::string& n1 = vars[qterm.var1].name;
            const std::string& n2 = vars[qterm.var2].name;

            if (qterm.is_diagonal()) {
                // Diagonal: inside P(x), coeff is Q_ii
                double coeff = qterm.coefficient;
                if (coeff < 0.0) {
                    os << (first_q ? "- " : " - ");
                    coeff = -coeff;
                } else if (!first_q) {
                    os << " + ";
                }
                if (coeff != 1.0) {
                    os << std::setprecision(16) << coeff << " ";
                }
                os << n1 << " ^ 2";
            } else {
                // Off-diagonal: inside P(x), algebraic coeff b = 2 * Q_ij
                double b_coeff = 2.0 * qterm.coefficient;
                if (b_coeff < 0.0) {
                    os << (first_q ? "- " : " - ");
                    b_coeff = -b_coeff;
                } else if (!first_q) {
                    os << " + ";
                }
                if (b_coeff != 1.0) {
                    os << std::setprecision(16) << b_coeff << " ";
                }
                os << n1 << " * " << n2;
            }
            first_q = false;
        }
        os << " ] / 2";
    }

    if (obj.offset > 0.0) {
        os << " + " << std::setprecision(16) << obj.offset;
    } else if (obj.offset < 0.0) {
        os << " - " << std::setprecision(16) << -obj.offset;
    }
    os << "\n";

    // 2. Constraints Section
    os << "Subject To\n";
    for (const auto& con : cons) {
        if (con.is_free()) continue;

        if (con.is_range()) {
            // Range constraint written as: lb <= expr <= ub
            os << " " << con.name << ": " << std::setprecision(16) << con.lower_bound << " <= ";
            bool first_t = true;
            for (const auto& term : con.terms) {
                if (term.coefficient != 0.0) {
                    write_term(os, term.coefficient, vars[term.variable_index].name, first_t);
                    first_t = false;
                }
            }
            if (first_t) os << "0";
            os << " <= " << std::setprecision(16) << con.upper_bound << "\n";
        } else {
            os << " " << con.name << ": ";
            bool first_t = true;
            for (const auto& term : con.terms) {
                if (term.coefficient != 0.0) {
                    write_term(os, term.coefficient, vars[term.variable_index].name, first_t);
                    first_t = false;
                }
            }
            if (first_t) os << "0";

            if (con.is_equality()) {
                os << " = " << std::setprecision(16) << con.lower_bound << "\n";
            } else if (con.is_greater_equal()) {
                os << " >= " << std::setprecision(16) << con.lower_bound << "\n";
            } else if (con.is_less_equal()) {
                os << " <= " << std::setprecision(16) << con.upper_bound << "\n";
            }
        }
    }

    // 3. Bounds Section
    os << "Bounds\n";
    for (const auto& var : vars) {
        if (var.type == VariableType::Binary) {
            // Handled under Binaries section
            continue;
        }

        if (var.is_free()) {
            os << " " << var.name << " free\n";
        } else if (var.is_fixed()) {
            os << " " << var.name << " = " << std::setprecision(16) << var.lower_bound << "\n";
        } else {
            bool non_default_lb = (var.lower_bound != 0.0);
            bool finite_ub = (!std::isinf(var.upper_bound));

            if (non_default_lb && finite_ub) {
                os << " " << std::setprecision(16) << var.lower_bound << " <= " << var.name
                   << " <= " << std::setprecision(16) << var.upper_bound << "\n";
            } else if (non_default_lb) {
                os << " " << var.name << " >= " << std::setprecision(16) << var.lower_bound << "\n";
            } else if (finite_ub) {
                os << " " << var.name << " <= " << std::setprecision(16) << var.upper_bound << "\n";
            }
        }
    }

    // 4. Generals (Integer) Section
    bool has_generals = false;
    for (const auto& var : vars) {
        if (var.type == VariableType::Integer) {
            has_generals = true;
            break;
        }
    }
    if (has_generals) {
        os << "Generals\n";
        for (const auto& var : vars) {
            if (var.type == VariableType::Integer) {
                os << " " << var.name << "\n";
            }
        }
    }

    // 5. Binaries Section
    bool has_binaries = false;
    for (const auto& var : vars) {
        if (var.type == VariableType::Binary) {
            has_binaries = true;
            break;
        }
    }
    if (has_binaries) {
        os << "Binaries\n";
        for (const auto& var : vars) {
            if (var.type == VariableType::Binary) {
                os << " " << var.name << "\n";
            }
        }
    }

    // 6. End
    os << "End\n";
    return Status::ok();
}

} // namespace sih26119
