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
    for (pinocchio::JointIndex joint = 1; joint < static_cast<pinocchio::JointIndex>(model_->njoints); ++joint) {
        if (model_->names[joint].empty()) throw std::invalid_argument("Pinocchio model contains an unnamed joint");
        for (const char* suffix : kParameterSuffixes) {
            parameter_names_.push_back("rigid_body/" + model_->names[joint] + "/" + suffix);
        }
    }
}

std::size_t PinocchioRegressor::parameter_count() const noexcept { return parameter_names_.size(); }

std::vector<std::string> PinocchioRegressor::parameter_names() const { return parameter_names_; }

namespace {
void ValidateSample(const pinocchio::Model& model, const robot_identification::Sample& sample) {
    if (sample.position.size() != model.nq || sample.velocity.size() != model.nv ||
        sample.acceleration.size() != model.nv || !sample.position.allFinite() || !sample.velocity.allFinite() ||
        !sample.acceleration.allFinite()) {
        throw std::invalid_argument("sample dimensions do not match the Pinocchio model");
    }
}

Eigen::MatrixXd ComputeBlock(const pinocchio::Model& model,
                             pinocchio::Data& data,
                             const robot_identification::Sample& sample,
                             std::size_t parameter_count) {
    Eigen::MatrixXd regressor =
        pinocchio::computeJointTorqueRegressor(model, data, sample.position, sample.velocity, sample.acceleration);
    if (regressor.rows() != model.nv || regressor.cols() != static_cast<Eigen::Index>(parameter_count) ||
        !regressor.allFinite()) {
        throw std::runtime_error("Pinocchio produced an invalid joint-torque regressor");
    }
    return regressor;
}

class DatasetBackend final : public robot_identification::RegressorBackend {
public:
    explicit DatasetBackend(const PinocchioRegressor& owner) : owner_(owner), data_(owner.model()) {}
    std::size_t parameter_count() const noexcept override { return owner_.parameter_count(); }
    std::vector<std::string> parameter_names() const override { return owner_.parameter_names(); }
    Eigen::MatrixXd Compute(const robot_identification::Sample& sample) const override {
        ValidateSample(owner_.model(), sample);
        return ComputeBlock(owner_.model(), data_, sample, parameter_count());
    }

private:
    const PinocchioRegressor& owner_;
    mutable pinocchio::Data data_;
};
}  // namespace

Eigen::MatrixXd PinocchioRegressor::Compute(const robot_identification::Sample& sample) const {
    ValidateSample(*model_, sample);
    // Keep the individual-sample interface safe for concurrent callers.
    pinocchio::Data data(*model_);
    return ComputeBlock(*model_, data, sample, parameter_count());
}

robot_identification::RegressionProblem PinocchioRegressor::BuildProblem(
    const robot_identification::Dataset& dataset) const {
    // The mutable workspace is owned by this call and cannot escape or alias
    // another concurrent assembly. Core validation and row ordering stay shared.
    DatasetBackend local(*this);
    return robot_identification::BuildRegressionProblem(dataset, local);
}

}  // namespace robot_identification_pinocchio
