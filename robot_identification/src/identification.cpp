#include "robot_identification/identification.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "name_validation.h"
#include "robot_identification/detail/regression_assembly.h"

namespace robot_identification {

RegressionProblem BuildRegressionProblem(const Dataset& dataset, const RegressorBackend& backend) {
    return detail::AssembleRegressionProblem(dataset, backend,
                                             [&](const Sample& sample) { return backend.Compute(sample); });
}

JointActuatorRegressor::JointActuatorRegressor(std::shared_ptr<const RegressorBackend> base,
                                               std::vector<std::string> joint_names,
                                               JointActuatorRegressorOptions options)
    : base_(std::move(base)), joint_names_(std::move(joint_names)), options_(options) {
    if (!base_ || joint_names_.empty() ||
        std::any_of(joint_names_.begin(), joint_names_.end(), [](const std::string& name) { return name.empty(); }) ||
        !std::isfinite(options_.coulomb_velocity_deadband) || options_.coulomb_velocity_deadband < 0.0) {
        throw std::invalid_argument("joint actuator regressor configuration is invalid");
    }
    if (!detail::HasUniqueNonemptyNames(joint_names_)) {
        throw std::invalid_argument("joint actuator regressor names must be unique");
    }
    terms_per_joint_ =
        static_cast<std::size_t>(options_.drive_inertia) + static_cast<std::size_t>(options_.viscous_friction) +
        static_cast<std::size_t>(options_.coulomb_friction) + static_cast<std::size_t>(options_.torque_bias);
    if (terms_per_joint_ == 0) throw std::invalid_argument("joint actuator regressor enables no terms");
}

std::size_t JointActuatorRegressor::parameter_count() const noexcept {
    return base_->parameter_count() + terms_per_joint_ * joint_names_.size();
}

std::vector<std::string> JointActuatorRegressor::parameter_names() const {
    auto names = base_->parameter_names();
    names.reserve(parameter_count());
    constexpr std::string_view kParameterPrefix = "actuator/";
    for (const auto& joint : joint_names_) {
        const auto append_name = [&](std::string_view suffix) {
            auto& name = names.emplace_back();
            name.reserve(kParameterPrefix.size() + joint.size() + suffix.size());
            name.append(kParameterPrefix).append(joint).append(suffix);
        };
        if (options_.drive_inertia) append_name("/drive_inertia");
        if (options_.viscous_friction) append_name("/viscous_friction");
        if (options_.coulomb_friction) append_name("/coulomb_friction");
        if (options_.torque_bias) append_name("/torque_bias");
    }
    return names;
}

Eigen::MatrixXd JointActuatorRegressor::Compute(const Sample& sample) const {
    const Eigen::Index dof = static_cast<Eigen::Index>(joint_names_.size());
    if (sample.position.size() != dof || sample.velocity.size() != dof || sample.acceleration.size() != dof ||
        !sample.position.allFinite() || !sample.velocity.allFinite() || !sample.acceleration.allFinite()) {
        throw std::invalid_argument("joint actuator regressor sample dimension is invalid");
    }
    const Eigen::MatrixXd base_block = base_->Compute(sample);
    if (base_block.rows() != dof || base_block.cols() != static_cast<Eigen::Index>(base_->parameter_count()) ||
        !base_block.allFinite()) {
        throw std::runtime_error("base regressor returned an invalid block");
    }
    Eigen::MatrixXd output(dof, static_cast<Eigen::Index>(parameter_count()));
    output.leftCols(base_block.cols()) = base_block;
    output.rightCols(output.cols() - base_block.cols()).setZero();
    Eigen::Index column = base_block.cols();
    for (Eigen::Index joint = 0; joint < dof; ++joint) {
        if (options_.drive_inertia) output(joint, column++) = sample.acceleration[joint];
        if (options_.viscous_friction) output(joint, column++) = sample.velocity[joint];
        if (options_.coulomb_friction) {
            output(joint, column++) = std::abs(sample.velocity[joint]) <= options_.coulomb_velocity_deadband
                                          ? 0.0
                                          : std::copysign(1.0, sample.velocity[joint]);
        }
        if (options_.torque_bias) output(joint, column++) = 1.0;
    }
    return output;
}

PhysicalValidationResult ValidateMassMatrices(const Eigen::MatrixXd& joint_positions,
                                              const Eigen::VectorXd& parameters,
                                              const MassMatrixFunction& mass_matrix,
                                              double required_minimum_eigenvalue) {
    if (joint_positions.rows() == 0 || joint_positions.cols() == 0 || !joint_positions.allFinite() ||
        parameters.size() == 0 || !parameters.allFinite() || !mass_matrix ||
        !std::isfinite(required_minimum_eigenvalue) || required_minimum_eigenvalue <= 0.0) {
        throw std::invalid_argument("mass-matrix validation inputs are invalid");
    }
    PhysicalValidationResult result;
    result.minimum_eigenvalue = std::numeric_limits<double>::infinity();
    Eigen::VectorXd joint_position(joint_positions.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenvalue_solver(joint_positions.cols());
    for (Eigen::Index row = 0; row < joint_positions.rows(); ++row) {
        joint_position = joint_positions.row(row).transpose();
        Eigen::MatrixXd scaled_mass_matrix = mass_matrix(joint_position, parameters);
        ++result.tested;
        if (scaled_mass_matrix.rows() != joint_positions.cols() ||
            scaled_mass_matrix.cols() != joint_positions.cols() || !scaled_mass_matrix.allFinite()) {
            ++result.failed;
            continue;
        }
        // Normalize the symmetry comparison so squared Frobenius norms cannot
        // hide asymmetry through overflow or underflow. The zero matrix keeps
        // its zero eigenvalue and fails the positive threshold below.
        // Eigenvalues are computed at this normalized scale, then restored before comparison.
        const double mass_matrix_scale = scaled_mass_matrix.cwiseAbs().maxCoeff();
        if (mass_matrix_scale != 0.0) scaled_mass_matrix /= mass_matrix_scale;
        if (!scaled_mass_matrix.isApprox(scaled_mass_matrix.transpose(), 1e-9)) {
            ++result.failed;
            continue;
        }
        eigenvalue_solver.compute(scaled_mass_matrix, Eigen::EigenvaluesOnly);
        if (eigenvalue_solver.info() != Eigen::Success ||
            !(eigenvalue_solver.eigenvalues() * mass_matrix_scale).allFinite()) {
            ++result.failed;
            continue;
        }
        const double smallest_sample_eigenvalue = eigenvalue_solver.eigenvalues().minCoeff() * mass_matrix_scale;
        result.minimum_eigenvalue = std::min(result.minimum_eigenvalue, smallest_sample_eigenvalue);
        if (smallest_sample_eigenvalue < required_minimum_eigenvalue) {
            ++result.failed;
        }
    }
    result.passed = result.failed == 0;
    return result;
}

}  // namespace robot_identification
