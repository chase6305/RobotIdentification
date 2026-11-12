#include "data_validation.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>

#include "name_validation.h"
#include "robot_identification/identification.h"

namespace robot_identification {
namespace {

bool Fail(std::string* reason, const char* message) {
    // Only write the diagnostic string when the caller requests it.
    if (reason) *reason = message;
    return false;
}

bool HasFinitePositiveWeights(const Eigen::VectorXd& weights) {
    for (Eigen::Index row = 0; row < weights.size(); ++row) {
        const double weight = weights[row];
        if (!std::isfinite(weight) || weight <= 0.0) return false;
    }
    return true;
}

}  // namespace

namespace detail {

bool ValidateRawInputs(const std::vector<std::string>& joint_names,
                       const Eigen::VectorXd& time_seconds,
                       const Eigen::MatrixXd& position,
                       const Eigen::MatrixXd& torque,
                       std::string* reason) {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    if (!HasUniqueNonemptyNames(joint_names)) {
        return Fail(reason, "joint names must be non-empty and unique");
    }
    if (time_seconds.size() < 3 || position.rows() != time_seconds.size() || torque.rows() != time_seconds.size() ||
        position.cols() != static_cast<Eigen::Index>(joint_names.size()) || torque.cols() != position.cols() ||
        !time_seconds.allFinite() || !position.allFinite() || !torque.allFinite()) {
        return Fail(reason, "raw dataset dimensions are inconsistent or values are non-finite");
    }
    for (Eigen::Index index = 1; index < time_seconds.size(); ++index) {
        if (time_seconds[index] <= time_seconds[index - 1]) {
            return Fail(reason, "raw timestamps must be strictly increasing");
        }
    }
    return true;
}

}  // namespace detail

bool Dataset::Validate(std::string* reason) const {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    if (samples.empty()) return Fail(reason, "dataset must contain samples");
    if (!detail::HasUniqueNonemptyNames(joint_names)) {
        return Fail(reason, "joint names must be non-empty and unique");
    }
    double previous_time = -std::numeric_limits<double>::infinity();
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.time_seconds) || sample.time_seconds <= previous_time) {
            return Fail(reason, "sample timestamps must be finite and strictly increasing");
        }
        previous_time = sample.time_seconds;
        for (const auto* vector : {&sample.position, &sample.velocity, &sample.acceleration, &sample.torque}) {
            if (vector->size() != static_cast<Eigen::Index>(dof()) || !vector->allFinite()) {
                return Fail(reason, "sample vectors must match the joint count and be finite");
            }
        }
    }
    return true;
}

bool RawDataset::Validate(std::string* reason) const {
    return detail::ValidateRawInputs(joint_names, time_seconds, position, torque, reason);
}

bool RegressionProblem::Validate(std::string* reason) const {
    if (sample_count == 0 || dof == 0 || regressor.rows() == 0 || regressor.cols() == 0) {
        return Fail(reason, "regression problem must not be empty");
    }
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) / dof ||
        regressor.rows() != observation.size() || row_weights.size() != observation.size() ||
        regressor.rows() != static_cast<Eigen::Index>(sample_count * dof)) {
        return Fail(reason, "regression problem dimensions are inconsistent");
    }
    if (!regressor.allFinite() || !observation.allFinite() || !HasFinitePositiveWeights(row_weights)) {
        return Fail(reason, "regression values must be finite and row weights positive");
    }
    return true;
}

}  // namespace robot_identification
