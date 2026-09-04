#include "solver/lp/lp_standard_form.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace sih26119 {

Result<StandardizedLp> StandardizedLp::standardize(const Model& model) {
    // 1. Validate input model preconditions
    Status val_status = model.validate();
    if (!val_status.is_ok()) {
        return val_status;
    }

    if (!model.is_lp()) {
        return Status::error(StatusCode::InvalidArgument, "Model is not a continuous Linear Program");
    }

    StandardizedLp std_lp;
    std_lp.original_sense_ = model.objective().sense;
    std_lp.original_c0_ = model.objective().offset;
    std_lp.original_nnz_ = model.num_nonzeros();

    auto num_vars_res = to_dimension(model.num_variables());
    if (!num_vars_res.is_ok()) {
        return num_vars_res.status();
    }
    std_lp.num_original_vars_ = num_vars_res.value();

    auto num_cons_res = to_dimension(model.num_constraints());
    if (!num_cons_res.is_ok()) {
        return num_cons_res.status();
    }
    std_lp.num_original_cons_ = num_cons_res.value();

    const bool is_maximize = (std_lp.original_sense_ == ObjectiveSense::Maximize);
    std_lp.c0_bar_ = is_maximize ? -std_lp.original_c0_ : std_lp.original_c0_;

    // Pre-accumulate original linear objective coefficients
    std::vector<Scalar> raw_obj(std_lp.num_original_vars_, 0.0);
    for (const auto& term : model.objective().linear_terms) {
        if (term.variable_index < std_lp.num_original_vars_) {
            raw_obj[term.variable_index] += term.coefficient;
        }
    }

    // 2. Classify original variables and allocate standard column indices
    std_lp.var_mappings_.resize(std_lp.num_original_vars_);
    Dimension std_col_count = 0;

    // Track box-bounded variables to create auxiliary rows later
    struct BoxBoundVar {
        VariableIndex var_idx;
        Index primary_col;
        Index slack_col;
        Scalar range;
    };
    std::vector<BoxBoundVar> box_bound_vars;

    // Temporary storage for standard costs
    std::vector<Scalar> c_bar_builder;

    auto allocate_col = [&](Scalar cost) -> Result<Index> {
        auto idx_res = to_index(std_col_count);
        if (!idx_res.is_ok()) {
            return idx_res.status();
        }
        std_col_count++;
        c_bar_builder.push_back(cost);
        return idx_res.value();
    };

    for (Dimension j = 0; j < std_lp.num_original_vars_; ++j) {
        VariableIndex v_idx = static_cast<VariableIndex>(j);
        const auto& v = model.get_variable(v_idx);
        auto& mapping = std_lp.var_mappings_[j];
        mapping.original_index = v_idx;
        mapping.original_lb = v.lower_bound;
        mapping.original_ub = v.upper_bound;

        const Scalar raw_cost = raw_obj[j];
        mapping.original_cost = raw_cost;
        const Scalar c_j = is_maximize ? -raw_cost : raw_cost;

        if (v.is_fixed()) {
            mapping.transform_type = VariableTransformType::FixedEliminated;
            mapping.fixed_value = v.lower_bound;
            std_lp.c0_bar_ += c_j * mapping.fixed_value;
        } else if (v.is_free()) {
            mapping.transform_type = VariableTransformType::FreeSplit;
            auto col_plus = allocate_col(c_j);
            if (!col_plus.is_ok()) return col_plus.status();
            auto col_minus = allocate_col(-c_j);
            if (!col_minus.is_ok()) return col_minus.status();
            mapping.std_var_primary = col_plus.value();
            mapping.std_var_secondary = col_minus.value();
        } else if (!std::isinf(v.lower_bound) && !std::isinf(v.upper_bound)) {
            // Box bound: L <= x <= U
            mapping.transform_type = VariableTransformType::BoxBound;
            auto col_primary = allocate_col(c_j);
            if (!col_primary.is_ok()) return col_primary.status();
            auto col_slack = allocate_col(0.0);
            if (!col_slack.is_ok()) return col_slack.status();
            mapping.std_var_primary = col_primary.value();
            mapping.std_var_slack = col_slack.value();
            std_lp.c0_bar_ += c_j * v.lower_bound;

            box_bound_vars.push_back({v_idx, mapping.std_var_primary, mapping.std_var_slack, v.upper_bound - v.lower_bound});
        } else if (!std::isinf(v.lower_bound) && std::isinf(v.upper_bound)) {
            if (v.lower_bound == 0.0) {
                mapping.transform_type = VariableTransformType::Identity;
                auto col_prim = allocate_col(c_j);
                if (!col_prim.is_ok()) return col_prim.status();
                mapping.std_var_primary = col_prim.value();
            } else {
                mapping.transform_type = VariableTransformType::LowerShift;
                auto col_prim = allocate_col(c_j);
                if (!col_prim.is_ok()) return col_prim.status();
                mapping.std_var_primary = col_prim.value();
                std_lp.c0_bar_ += c_j * v.lower_bound;
            }
        } else if (std::isinf(v.lower_bound) && !std::isinf(v.upper_bound)) {
            mapping.transform_type = VariableTransformType::UpperReflect;
            auto col_prim = allocate_col(-c_j);
            if (!col_prim.is_ok()) return col_prim.status();
            mapping.std_var_primary = col_prim.value();
            std_lp.c0_bar_ += c_j * v.upper_bound;
        } else {
            return Status::error(StatusCode::InvalidArgument, "Unsupported variable bound state");
        }
    }

    // 3. Process constraints and allocate standard row indices
    std_lp.con_mappings_.resize(std_lp.num_original_cons_);
    Dimension std_row_count = 0;

    // Temporary data structures for constraint construction
    std::vector<Scalar> b_bar_builder;
    std::vector<Triplet> triplets;

    auto allocate_row = [&](Scalar rhs) -> Result<Index> {
        auto idx_res = to_index(std_row_count);
        if (!idx_res.is_ok()) {
            return idx_res.status();
        }
        std_row_count++;
        b_bar_builder.push_back(rhs);
        return idx_res.value();
    };

    for (Dimension i = 0; i < std_lp.num_original_cons_; ++i) {
        ConstraintIndex c_idx = static_cast<ConstraintIndex>(i);
        const auto& con = model.get_constraint(c_idx);
        auto& mapping = std_lp.con_mappings_[i];
        mapping.original_index = c_idx;
        mapping.original_lb = con.lower_bound;
        mapping.original_ub = con.upper_bound;

        if (con.is_free()) {
            mapping.transform_type = ConstraintTransformType::Free;
            continue;
        }

        if (con.is_equality()) {
            mapping.transform_type = ConstraintTransformType::Equality;
            auto row_res = allocate_row(con.lower_bound);
            if (!row_res.is_ok()) return row_res.status();
            mapping.generated_rows.push_back(row_res.value());
        } else if (con.is_less_equal()) {
            mapping.transform_type = ConstraintTransformType::LessEqual;
            auto row_res = allocate_row(con.upper_bound);
            if (!row_res.is_ok()) return row_res.status();
            mapping.generated_rows.push_back(row_res.value());

            auto slack_res = allocate_col(0.0);
            if (!slack_res.is_ok()) return slack_res.status();
            mapping.auxiliary_primary = slack_res.value();
            triplets.push_back({row_res.value(), mapping.auxiliary_primary, 1.0});
        } else if (con.is_greater_equal()) {
            mapping.transform_type = ConstraintTransformType::GreaterEqual;
            auto row_res = allocate_row(con.lower_bound);
            if (!row_res.is_ok()) return row_res.status();
            mapping.generated_rows.push_back(row_res.value());

            auto surplus_res = allocate_col(0.0);
            if (!surplus_res.is_ok()) return surplus_res.status();
            mapping.auxiliary_primary = surplus_res.value();
            triplets.push_back({row_res.value(), mapping.auxiliary_primary, -1.0});
        } else if (con.is_range()) {
            mapping.transform_type = ConstraintTransformType::Range;
            // Row 1: Lower bound equality with surplus variable
            auto row1_res = allocate_row(con.lower_bound);
            if (!row1_res.is_ok()) return row1_res.status();
            mapping.generated_rows.push_back(row1_res.value());

            // Row 2: Range length equality with surplus + range slack
            auto row2_res = allocate_row(con.upper_bound - con.lower_bound);
            if (!row2_res.is_ok()) return row2_res.status();
            mapping.generated_rows.push_back(row2_res.value());

            auto surplus_res = allocate_col(0.0);
            if (!surplus_res.is_ok()) return surplus_res.status();
            auto slack_res = allocate_col(0.0);
            if (!slack_res.is_ok()) return slack_res.status();

            mapping.auxiliary_primary = surplus_res.value();
            mapping.auxiliary_secondary = slack_res.value();

            // Row 1 surplus entry: -1.0 * s_i
            triplets.push_back({row1_res.value(), mapping.auxiliary_primary, -1.0});
            // Row 2 entries: 1.0 * s_i + 1.0 * t_i = u_i - l_i
            triplets.push_back({row2_res.value(), mapping.auxiliary_primary, 1.0});
            triplets.push_back({row2_res.value(), mapping.auxiliary_secondary, 1.0});
        }

        // Add structural terms to the primary generated row
        Index primary_row = mapping.generated_rows[0];
        for (const auto& term : con.terms) {
            VariableIndex v_idx = term.variable_index;
            const auto& v_map = std_lp.var_mappings_[v_idx];
            const Scalar a_ij = term.coefficient;

            switch (v_map.transform_type) {
                case VariableTransformType::FixedEliminated:
                    b_bar_builder[primary_row] -= a_ij * v_map.fixed_value;
                    break;
                case VariableTransformType::Identity:
                    triplets.push_back({primary_row, v_map.std_var_primary, a_ij});
                    break;
                case VariableTransformType::LowerShift:
                    triplets.push_back({primary_row, v_map.std_var_primary, a_ij});
                    b_bar_builder[primary_row] -= a_ij * v_map.original_lb;
                    break;
                case VariableTransformType::UpperReflect:
                    triplets.push_back({primary_row, v_map.std_var_primary, -a_ij});
                    b_bar_builder[primary_row] -= a_ij * v_map.original_ub;
                    break;
                case VariableTransformType::BoxBound:
                    triplets.push_back({primary_row, v_map.std_var_primary, a_ij});
                    b_bar_builder[primary_row] -= a_ij * v_map.original_lb;
                    break;
                case VariableTransformType::FreeSplit:
                    triplets.push_back({primary_row, v_map.std_var_primary, a_ij});
                    triplets.push_back({primary_row, v_map.std_var_secondary, -a_ij});
                    break;
            }
        }
    }

    // 4. Add auxiliary rows for box-bounded variables: x_bar + s = U - L
    for (const auto& bb : box_bound_vars) {
        auto row_res = allocate_row(bb.range);
        if (!row_res.is_ok()) return row_res.status();
        Index box_row = row_res.value();
        triplets.push_back({box_row, bb.primary_col, 1.0});
        triplets.push_back({box_row, bb.slack_col, 1.0});
    }

    // 5. Apply RHS Sign Normalization (ensure b_bar[i] >= 0)
    std::vector<bool> rows_negated(std_row_count, false);
    for (Dimension r = 0; r < std_row_count; ++r) {
        if (b_bar_builder[r] < 0.0) {
            b_bar_builder[r] = -b_bar_builder[r];
            rows_negated[r] = true;
        }
    }

    // Record row negation in constraint mappings
    for (auto& con_map : std_lp.con_mappings_) {
        con_map.row_negated.resize(con_map.generated_rows.size(), false);
        for (size_t k = 0; k < con_map.generated_rows.size(); ++k) {
            Index r = con_map.generated_rows[k];
            if (rows_negated[r]) {
                con_map.row_negated[k] = true;
            }
        }
    }

    // Multiply matrix entries in negated rows by -1.0
    for (auto& trip : triplets) {
        if (rows_negated[trip.row]) {
            trip.value = -trip.value;
        }
    }

    // 6. Build SparseMatrix A_bar
    auto mat_res = SparseMatrix::from_triplets(std_row_count, std_col_count, triplets);
    if (!mat_res.is_ok()) {
        return mat_res.status();
    }
    std_lp.A_bar_ = std::move(mat_res.value());

    // 7. Assemble DenseVector b_bar and c_bar
    auto b_res = DenseVector::create(std_row_count);
    if (!b_res.is_ok()) return b_res.status();
    std_lp.b_bar_ = std::move(b_res.value());
    for (Dimension r = 0; r < std_row_count; ++r) {
        auto st = std_lp.b_bar_.set(r, b_bar_builder[r]);
        if (!st.is_ok()) return st;
    }

    auto c_res = DenseVector::create(std_col_count);
    if (!c_res.is_ok()) return c_res.status();
    std_lp.c_bar_ = std::move(c_res.value());
    for (Dimension c = 0; c < std_col_count; ++c) {
        auto st = std_lp.c_bar_.set(c, c_bar_builder[c]);
        if (!st.is_ok()) return st;
    }

    return std_lp;
}

Result<DenseVector> StandardizedLp::reconstruct_primal(const DenseVector& x_bar) const {
    if (x_bar.size() != standardized_variables()) {
        return Status::error(StatusCode::InvalidArgument, "Standard vector dimension mismatch during reconstruction");
    }

    auto res = DenseVector::create(num_original_vars_);
    if (!res.is_ok()) {
        return res.status();
    }
    DenseVector x_orig = std::move(res.value());

    for (Dimension j = 0; j < num_original_vars_; ++j) {
        const auto& mapping = var_mappings_[j];
        Scalar val = 0.0;

        switch (mapping.transform_type) {
            case VariableTransformType::FixedEliminated:
                val = mapping.fixed_value;
                break;
            case VariableTransformType::Identity: {
                auto get_res = x_bar.at(mapping.std_var_primary);
                if (!get_res.is_ok()) return get_res.status();
                val = get_res.value();
                break;
            }
            case VariableTransformType::LowerShift: {
                auto get_res = x_bar.at(mapping.std_var_primary);
                if (!get_res.is_ok()) return get_res.status();
                val = get_res.value() + mapping.original_lb;
                break;
            }
            case VariableTransformType::UpperReflect: {
                auto get_res = x_bar.at(mapping.std_var_primary);
                if (!get_res.is_ok()) return get_res.status();
                val = mapping.original_ub - get_res.value();
                break;
            }
            case VariableTransformType::BoxBound: {
                auto get_res = x_bar.at(mapping.std_var_primary);
                if (!get_res.is_ok()) return get_res.status();
                val = get_res.value() + mapping.original_lb;
                break;
            }
            case VariableTransformType::FreeSplit: {
                auto get_plus = x_bar.at(mapping.std_var_primary);
                if (!get_plus.is_ok()) return get_plus.status();
                auto get_minus = x_bar.at(mapping.std_var_secondary);
                if (!get_minus.is_ok()) return get_minus.status();
                val = get_plus.value() - get_minus.value();
                break;
            }
        }

        auto st = x_orig.set(j, val);
        if (!st.is_ok()) return st;
    }

    return x_orig;
}

Result<DenseVector> StandardizedLp::project_primal(const DenseVector& x_orig) const {
    if (x_orig.size() != num_original_vars_) {
        return Status::error(StatusCode::InvalidArgument, "Original vector dimension mismatch during projection");
    }

    auto res = DenseVector::create(standardized_variables());
    if (!res.is_ok()) {
        return res.status();
    }
    DenseVector x_bar = std::move(res.value());

    // 1. Project variables
    for (Dimension j = 0; j < num_original_vars_; ++j) {
        const auto& mapping = var_mappings_[j];
        auto get_res = x_orig.at(j);
        if (!get_res.is_ok()) return get_res.status();
        const Scalar xj = get_res.value();

        switch (mapping.transform_type) {
            case VariableTransformType::FixedEliminated:
                // Eliminated from x_bar
                break;
            case VariableTransformType::Identity: {
                auto st = x_bar.set(mapping.std_var_primary, xj);
                if (!st.is_ok()) return st;
                break;
            }
            case VariableTransformType::LowerShift: {
                auto st = x_bar.set(mapping.std_var_primary, xj - mapping.original_lb);
                if (!st.is_ok()) return st;
                break;
            }
            case VariableTransformType::UpperReflect: {
                auto st = x_bar.set(mapping.std_var_primary, mapping.original_ub - xj);
                if (!st.is_ok()) return st;
                break;
            }
            case VariableTransformType::BoxBound: {
                auto st1 = x_bar.set(mapping.std_var_primary, xj - mapping.original_lb);
                if (!st1.is_ok()) return st1;
                auto st2 = x_bar.set(mapping.std_var_slack, mapping.original_ub - xj);
                if (!st2.is_ok()) return st2;
                break;
            }
            case VariableTransformType::FreeSplit: {
                if (xj >= 0.0) {
                    auto st1 = x_bar.set(mapping.std_var_primary, xj);
                    if (!st1.is_ok()) return st1;
                    auto st2 = x_bar.set(mapping.std_var_secondary, 0.0);
                    if (!st2.is_ok()) return st2;
                } else {
                    auto st1 = x_bar.set(mapping.std_var_primary, 0.0);
                    if (!st1.is_ok()) return st1;
                    auto st2 = x_bar.set(mapping.std_var_secondary, -xj);
                    if (!st2.is_ok()) return st2;
                }
                break;
            }
        }
    }

    // 2. Project constraint auxiliaries
    for (Dimension i = 0; i < num_original_cons_; ++i) {
        const auto& mapping = con_mappings_[i];
        if (mapping.transform_type == ConstraintTransformType::Free ||
            mapping.transform_type == ConstraintTransformType::Equality) {
            continue;
        }

        Index primary_row = mapping.generated_rows[0];
        Scalar structural_sum = 0.0;
        auto r_start = A_bar_.row_ptr()[primary_row];
        auto r_end = A_bar_.row_ptr()[primary_row + 1];
        for (auto nz = r_start; nz < r_end; ++nz) {
            Index col = A_bar_.col_idx()[nz];
            if (col != mapping.auxiliary_primary && col != mapping.auxiliary_secondary) {
                auto val_res = x_bar.at(col);
                if (!val_res.is_ok()) return val_res.status();
                structural_sum += A_bar_.values()[nz] * val_res.value();
            }
        }

        auto b_val_res = b_bar_.at(primary_row);
        if (!b_val_res.is_ok()) return b_val_res.status();
        Scalar b_val = b_val_res.value();

        Scalar aux_coeff = 0.0;
        for (auto nz = r_start; nz < r_end; ++nz) {
            if (A_bar_.col_idx()[nz] == mapping.auxiliary_primary) {
                aux_coeff = A_bar_.values()[nz];
                break;
            }
        }

        if (std::abs(aux_coeff) > 0.0) {
            Scalar aux_val = (b_val - structural_sum) / aux_coeff;
            auto st = x_bar.set(mapping.auxiliary_primary, aux_val);
            if (!st.is_ok()) return st;
        }

        if (mapping.transform_type == ConstraintTransformType::Range) {
            Index range_row = mapping.generated_rows[1];
            auto b_range_res = b_bar_.at(range_row);
            if (!b_range_res.is_ok()) return b_range_res.status();
            Scalar b_range = b_range_res.value();

            auto s_res = x_bar.at(mapping.auxiliary_primary);
            if (!s_res.is_ok()) return s_res.status();
            Scalar s_val = s_res.value();

            Scalar t_coeff = 1.0;
            auto r2_start = A_bar_.row_ptr()[range_row];
            auto r2_end = A_bar_.row_ptr()[range_row + 1];
            for (auto nz = r2_start; nz < r2_end; ++nz) {
                if (A_bar_.col_idx()[nz] == mapping.auxiliary_secondary) {
                    t_coeff = A_bar_.values()[nz];
                    break;
                }
            }

            Scalar t_val = (b_range - (t_coeff > 0.0 ? s_val : -s_val)) / t_coeff;
            auto st = x_bar.set(mapping.auxiliary_secondary, t_val);
            if (!st.is_ok()) return st;
        }
    }

    return x_bar;
}

Result<Scalar> StandardizedLp::evaluate_standard_objective(const DenseVector& x_bar) const {
    if (x_bar.size() != standardized_variables()) {
        return Status::error(StatusCode::InvalidArgument, "Standard vector dimension mismatch");
    }

    auto dot_res = c_bar_.dot(x_bar);
    if (!dot_res.is_ok()) {
        return dot_res.status();
    }
    return dot_res.value() + c0_bar_;
}

Result<Scalar> StandardizedLp::evaluate_original_objective(const DenseVector& x_orig) const {
    if (x_orig.size() != num_original_vars_) {
        return Status::error(StatusCode::InvalidArgument, "Original vector dimension mismatch");
    }

    Scalar obj_sum = original_c0_;
    for (Dimension j = 0; j < num_original_vars_; ++j) {
        const auto& mapping = var_mappings_[j];
        auto xj_res = x_orig.at(j);
        if (!xj_res.is_ok()) return xj_res.status();
        obj_sum += mapping.original_cost * xj_res.value();
    }
    return obj_sum;
}

} // namespace sih26119
