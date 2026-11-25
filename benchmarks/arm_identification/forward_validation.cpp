#include "forward_validation.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace arm_identification {
namespace {
constexpr std::size_t kSamplesPerSplit = 81;
constexpr std::size_t kReplayWindowCount = 16;
constexpr double kReplayWindowDurationSeconds = .1;

std::size_t EvenIndex(std::size_t span, std::size_t index, std::size_t intervals) {
    if (intervals == 0) return 0;
    return index * (span / intervals) + index * (span % intervals) / intervals;
}

bool ReplayStep(const robot_identification::RegressorBackend& backend,
                const Eigen::VectorXd& parameters,
                const robot_identification::Sample& observation,
                double dt,
                const Eigen::VectorXd& minimum_position,
                const Eigen::VectorXd& maximum_position,
                robot_identification::Sample& state) {
    state.time_seconds = observation.time_seconds;
    state.torque = observation.torque;
    const auto dof = state.position.size();
    state.acceleration.setZero();
    const Eigen::VectorXd bias = backend.Compute(state) * parameters;
    Eigen::MatrixXd mass(dof, dof);
    for (Eigen::Index joint = 0; joint < dof; ++joint) {
        state.acceleration[joint] = 1.;
        mass.col(joint) = backend.Compute(state) * parameters - bias;
        state.acceleration[joint] = 0.;
    }
    if (!mass.allFinite() || !mass.isApprox(mass.transpose(), 1e-9)) return false;
    const Eigen::LDLT<Eigen::MatrixXd> factor(mass);
    if (factor.info() != Eigen::Success || !factor.isPositive() || factor.vectorD().minCoeff() < 1e-9) return false;
    const Eigen::VectorXd acceleration = factor.solve(observation.torque - bias);
    state.velocity += dt * acceleration;
    state.position += dt * state.velocity;
    return state.velocity.allFinite() && state.position.allFinite() &&
           (state.position.array() >= minimum_position.array()).all() &&
           (state.position.array() <= maximum_position.array()).all();
}
}  // namespace

ReplayMetrics ReplayForwardWindows(const robot_identification::RegressorBackend& backend,
                                   const Eigen::VectorXd& parameters,
                                   const robot_identification::Dataset& dense_holdout,
                                   const Eigen::VectorXd& minimum_position,
                                   const Eigen::VectorXd& maximum_position) {
    const auto& samples = dense_holdout.samples;
    const auto compare_time = [](const auto& sample, double time) { return sample.time_seconds < time; };
    const double last_start_time = samples.back().time_seconds - kReplayWindowDurationSeconds;
    if (last_start_time <= samples.front().time_seconds)
        throw std::invalid_argument("dense holdout is too short for forward replay");
    const auto last_start = static_cast<std::size_t>(
        std::lower_bound(samples.begin(), samples.end(), last_start_time, compare_time) - samples.begin() - 1);
    ReplayMetrics result;
    result.requested_window_seconds = kReplayWindowDurationSeconds;
    result.windows = std::min(kReplayWindowCount, last_start + 1);
    double squared_error = 0.;
    double maximum_final_error = 0.;
    std::size_t position_terms = 0;
    for (std::size_t window = 0; window < result.windows; ++window) {
        const auto start = EvenIndex(last_start, window, result.windows - 1);
        const auto end = static_cast<std::size_t>(
            std::lower_bound(samples.begin() + start, samples.end(),
                             samples[start].time_seconds + kReplayWindowDurationSeconds, compare_time) -
            samples.begin());
        if (end >= samples.size()) throw std::invalid_argument("dense holdout does not cover the requested window");
        auto state = samples[start];
        double window_error = 0.;
        std::size_t index = start;
        for (; index < end; ++index) {
            const double dt = samples[index + 1].time_seconds - samples[index].time_seconds;
            if (!ReplayStep(backend, parameters, samples[index], dt, minimum_position, maximum_position, state)) break;
            window_error += (state.position - samples[index + 1].position).squaredNorm();
        }
        if (index != end) continue;
        ++result.complete_windows;
        squared_error += window_error;
        maximum_final_error =
            std::max(maximum_final_error, (state.position - samples[end].position).cwiseAbs().maxCoeff());
        position_terms += (end - start) * dense_holdout.dof();
    }
    if (result.complete_windows == result.windows && position_terms != 0 && std::isfinite(squared_error)) {
        result.position_rmse = std::sqrt(squared_error / position_terms);
        result.maximum_final_position_error = maximum_final_error;
    }
    return result;
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
        const auto source = EvenIndex(span, index, intervals);
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
