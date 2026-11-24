#include "forward_validation.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace arm_identification {
namespace {
constexpr std::size_t kSamplesPerSplit = 81;
}

ForwardValidation::ForwardValidation(const robot_identification::RegressorBackend& backend,
                                     const robot_identification::Dataset& calibration,
                                     const robot_identification::Dataset& holdout) {
    AddPoints(backend, calibration, false);
    AddPoints(backend, holdout, true);
}

void ForwardValidation::AddPoints(const robot_identification::RegressorBackend& backend,
                                  const robot_identification::Dataset& dataset,
                                  bool measure_acceleration) {
    const auto span = dataset.samples.size() - 1;
    const auto intervals = std::min(kSamplesPerSplit - 1, span);
    const auto dof = static_cast<Eigen::Index>(dataset.dof());
    for (std::size_t index = 0; index <= intervals; ++index) {
        const auto source = index * (span / intervals) + index * (span % intervals) / intervals;
        auto sample = dataset.samples[source];
        Point point;
        point.position = sample.position.transpose();
        if (measure_acceleration) {
            point.torque = sample.torque;
            point.acceleration = sample.acceleration;
        }
        sample.acceleration.setZero();
        const auto bias = backend.Compute(sample);
        point.mass_regressor.resize(dof * dof, bias.cols());
        for (Eigen::Index column = 0; column < dof; ++column) {
            sample.acceleration[column] = 1.;
            point.mass_regressor.middleRows(column * dof, dof) = backend.Compute(sample) - bias;
            sample.acceleration[column] = 0.;
        }
        if (measure_acceleration) point.bias_regressor = bias;
        points_.push_back(std::move(point));
    }
}

ForwardMetrics ForwardValidation::Evaluate(const Eigen::VectorXd& parameters) const {
    ForwardMetrics result;
    result.mass.minimum_eigenvalue = std::numeric_limits<double>::infinity();
    std::size_t acceleration_terms = 0;
    double squared_error = 0.;
    for (const auto& point : points_) {
        const Eigen::Index dof = point.position.cols();
        const Eigen::VectorXd coefficients = point.mass_regressor * parameters;
        const Eigen::Map<const Eigen::MatrixXd> mass(coefficients.data(), dof, dof);
        const auto physical = robot_identification::ValidateMassMatrices(
            point.position, parameters, [&](const auto&, const auto&) { return Eigen::MatrixXd(mass); });
        result.mass.tested += physical.tested;
        result.mass.failed += physical.failed;
        result.mass.minimum_eigenvalue = std::min(result.mass.minimum_eigenvalue, physical.minimum_eigenvalue);
        if (point.acceleration.size() == 0) continue;
        ++result.holdout_samples;
        if (!physical.passed) continue;
        const Eigen::VectorXd right_hand_side = point.torque - point.bias_regressor * parameters;
        const Eigen::VectorXd predicted = mass.ldlt().solve(right_hand_side);
        squared_error += (predicted - point.acceleration).squaredNorm();
        acceleration_terms += dof;
    }
    result.mass.passed = result.mass.failed == 0;
    if (acceleration_terms != 0 &&
        acceleration_terms == result.holdout_samples * static_cast<std::size_t>(points_.front().position.cols()) &&
        std::isfinite(squared_error)) {
        result.holdout_acceleration_rmse = std::sqrt(squared_error / acceleration_terms);
    }
    return result;
}

}  // namespace arm_identification
