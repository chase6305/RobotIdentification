#include "robot_identification_pinocchio/pinocchio_regressor.h"

#include <array>
#include <pinocchio/algorithm/regressor.hpp>
#include <stdexcept>
#include <utility>

namespace robot_identification_pinocchio {

PinocchioRegressor::PinocchioRegressor(std::shared_ptr<const pinocchio::Model> model) : model_(std::move(model)) {
    if (!model_ || model_->nq == 0 || model_->nv == 0 || model_->nq != model_->nv || model_->njoints <= 1) {
        throw std::invalid_argument("Pinocchio regressor requires a non-empty fixed-base model with nq equal to nv");
    }
    static constexpr std::array<const char*, 10> kParameterSuffixes{"mass", "mcx", "mcy", "mcz", "Ixx",
                                                                    "Ixy",  "Iyy", "Ixz", "Iyz", "Izz"};
    parameter_names_.reserve(10 * (model_->njoints - 1));
    for (pinocchio::JointIndex joint = 1; joint < model_->njoints; ++joint) {
        if (model_->names[joint].empty()) throw std::invalid_argument("Pinocchio model contains an unnamed joint");
        for (const char* suffix : kParameterSuffixes) {
            parameter_names_.push_back("rigid_body/" + model_->names[joint] + "/" + suffix);
        }
    }
}

std::size_t PinocchioRegressor::parameter_count() const noexcept { return parameter_names_.size(); }

std::vector<std::string> PinocchioRegressor::parameter_names() const { return parameter_names_; }

Eigen::MatrixXd PinocchioRegressor::Compute(const robot_identification::Sample& sample) const {
    if (sample.position.size() != model_->nq || sample.velocity.size() != model_->nv ||
        sample.acceleration.size() != model_->nv || !sample.position.allFinite() || !sample.velocity.allFinite() ||
        !sample.acceleration.allFinite()) {
        throw std::invalid_argument("sample dimensions do not match the Pinocchio model");
    }
    // Data is deliberately local: identification is offline, and this keeps a
    // shared backend safe when independent datasets are assembled concurrently.
    pinocchio::Data data(*model_);
    Eigen::MatrixXd regressor =
        pinocchio::computeJointTorqueRegressor(*model_, data, sample.position, sample.velocity, sample.acceleration);
    if (regressor.rows() != model_->nv || regressor.cols() != static_cast<Eigen::Index>(parameter_count()) ||
        !regressor.allFinite()) {
        throw std::runtime_error("Pinocchio produced an invalid joint-torque regressor");
    }
    return regressor;
}

}  // namespace robot_identification_pinocchio
