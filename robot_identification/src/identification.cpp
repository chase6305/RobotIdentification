#include "robot_identification/identification.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

namespace robot_identification {
namespace {

bool Fail(std::string* reason, const std::string& message) {
    if (reason) *reason = message;
    return false;
}

Eigen::MatrixXd SmoothSymmetric(const Eigen::MatrixXd& values, std::size_t window) {
    if (window == 1) return values;
    const Eigen::Index radius = static_cast<Eigen::Index>(window / 2);
    Eigen::MatrixXd output(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
        const Eigen::Index end = std::min<Eigen::Index>(values.rows() - 1, row + radius);
        output.row(row) = values.middleRows(begin, end - begin + 1).colwise().mean();
    }
    return output;
}

Eigen::MatrixXd Differentiate(const Eigen::VectorXd& time, const Eigen::MatrixXd& values) {
    if (time.size() < 3 || values.rows() != time.size()) {
        throw std::invalid_argument("differentiation requires at least three matching samples");
    }
    for (Eigen::Index row = 1; row < time.size(); ++row) {
        if (!std::isfinite(time[row]) || time[row] <= time[row - 1]) {
            throw std::invalid_argument("timestamps must be finite and strictly increasing");
        }
    }
    Eigen::MatrixXd derivative(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const Eigen::Index first = row == 0 ? 0 : row + 1 == values.rows() ? row - 2 : row - 1;
        const Eigen::Index second = first + 1;
        const Eigen::Index third = first + 2;
        const double x0 = time[first] - time[row];
        const double x1 = time[second] - time[row];
        const double x2 = time[third] - time[row];
        const double c0 = -(x1 + x2) / ((x0 - x1) * (x0 - x2));
        const double c1 = -(x0 + x2) / ((x1 - x0) * (x1 - x2));
        const double c2 = -(x0 + x1) / ((x2 - x0) * (x2 - x1));
        if (!std::isfinite(c0) || !std::isfinite(c1) || !std::isfinite(c2)) {
            throw std::overflow_error("timestamp spacing is numerically unusable for differentiation");
        }
        derivative.row(row) = c0 * values.row(first) + c1 * values.row(second) + c2 * values.row(third);
    }
    return derivative;
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::string normalized = line;
    if (!normalized.empty() && normalized.back() == '\r') normalized.pop_back();
    std::vector<std::string> fields;
    std::stringstream stream(normalized);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!normalized.empty() && normalized.back() == ',') fields.emplace_back();
    return fields;
}

double ParseFinite(const std::string& field) {
    std::size_t consumed = 0;
    const double value = std::stod(field, &consumed);
    if (consumed != field.size() || !std::isfinite(value)) {
        throw std::runtime_error("CSV numeric field is invalid: " + field);
    }
    return value;
}

struct LinearSolveResult {
    Eigen::VectorXd parameters;
    Eigen::VectorXd singular_values;
    std::size_t rank{0};
};

LinearSolveResult SolveWeighted(const RegressionProblem& problem,
                                const Eigen::VectorXd& weights,
                                const SolverOptions& options) {
    const Eigen::VectorXd sqrt_weights = weights.array().sqrt();
    Eigen::MatrixXd weighted_regressor = problem.regressor.array().colwise() * sqrt_weights.array();
    Eigen::VectorXd weighted_observation = problem.observation.array() * sqrt_weights.array();

    // Observability is a property of measured data. Diagnose it before adding
    // ridge rows, otherwise regularization would make every problem appear
    // full rank even when the experiment did not excite some parameters.
    Eigen::JacobiSVD<Eigen::MatrixXd> diagnostic_svd(weighted_regressor);
    diagnostic_svd.setThreshold(options.relative_rank_tolerance);
    const Eigen::VectorXd diagnostic_singular_values = diagnostic_svd.singularValues();
    const std::size_t diagnostic_rank = static_cast<std::size_t>(diagnostic_svd.rank());

    if (options.ridge > 0.0) {
        const Eigen::Index old_rows = weighted_regressor.rows();
        const Eigen::Index parameter_count = weighted_regressor.cols();
        weighted_regressor.conservativeResize(old_rows + parameter_count, Eigen::NoChange);
        weighted_regressor.bottomRows(parameter_count) =
            std::sqrt(options.ridge) * Eigen::MatrixXd::Identity(parameter_count, parameter_count);
        weighted_observation.conservativeResize(old_rows + parameter_count);
        weighted_observation.tail(parameter_count).setZero();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted_regressor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(options.relative_rank_tolerance);
    LinearSolveResult result;
    result.parameters = svd.solve(weighted_observation);
    result.singular_values = diagnostic_singular_values;
    result.rank = diagnostic_rank;
    return result;
}

}  // namespace

bool Dataset::Validate(std::string* reason) const {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    if (samples.empty()) return Fail(reason, "dataset must contain samples");
    std::vector<std::string> names = joint_names;
    std::sort(names.begin(), names.end());
    if (names.front().empty() || std::adjacent_find(names.begin(), names.end()) != names.end()) {
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
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    std::vector<std::string> names = joint_names;
    std::sort(names.begin(), names.end());
    if (names.front().empty() || std::adjacent_find(names.begin(), names.end()) != names.end()) {
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

HarmonicTrajectory::HarmonicTrajectory(Eigen::VectorXd offset,
                                       Eigen::MatrixXd sin_coefficients,
                                       Eigen::MatrixXd cos_coefficients,
                                       double fundamental_frequency_hz)
    : offset_(std::move(offset)),
      sin_coefficients_(std::move(sin_coefficients)),
      cos_coefficients_(std::move(cos_coefficients)),
      angular_frequency_(2.0 * std::acos(-1.0) * fundamental_frequency_hz) {
    if (offset_.size() == 0 || sin_coefficients_.rows() != offset_.size() ||
        cos_coefficients_.rows() != offset_.size() || sin_coefficients_.cols() == 0 ||
        cos_coefficients_.cols() != sin_coefficients_.cols() || !offset_.allFinite() ||
        !sin_coefficients_.allFinite() || !cos_coefficients_.allFinite() ||
        !std::isfinite(angular_frequency_) || angular_frequency_ <= 0.0) {
        throw std::invalid_argument("harmonic trajectory configuration is invalid");
    }
}

std::size_t HarmonicTrajectory::dof() const noexcept { return static_cast<std::size_t>(offset_.size()); }

std::size_t HarmonicTrajectory::harmonic_count() const noexcept {
    return static_cast<std::size_t>(sin_coefficients_.cols());
}

double HarmonicTrajectory::period_seconds() const noexcept {
    return 2.0 * std::acos(-1.0) / angular_frequency_;
}

ExcitationSample HarmonicTrajectory::Evaluate(double time_seconds) const {
    if (!std::isfinite(time_seconds)) throw std::invalid_argument("trajectory time must be finite");
    ExcitationSample sample{offset_, Eigen::VectorXd::Zero(offset_.size()),
                            Eigen::VectorXd::Zero(offset_.size()), Eigen::VectorXd::Zero(offset_.size())};
    for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
        const double frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
        const double phase = frequency * time_seconds;
        const double sine = std::sin(phase);
        const double cosine = std::cos(phase);
        const Eigen::VectorXd position_term =
            sin_coefficients_.col(harmonic) * sine + cos_coefficients_.col(harmonic) * cosine;
        const Eigen::VectorXd derivative_term =
            sin_coefficients_.col(harmonic) * cosine - cos_coefficients_.col(harmonic) * sine;
        sample.position += position_term;
        sample.velocity += frequency * derivative_term;
        sample.acceleration -= frequency * frequency * position_term;
        sample.jerk -= frequency * frequency * frequency * derivative_term;
    }
    if (!sample.position.allFinite() || !sample.velocity.allFinite() || !sample.acceleration.allFinite() ||
        !sample.jerk.allFinite()) {
        throw std::overflow_error("harmonic trajectory evaluation overflowed");
    }
    return sample;
}

HarmonicBounds HarmonicTrajectory::ConservativeBounds() const {
    HarmonicBounds bounds{offset_, offset_, Eigen::VectorXd::Zero(offset_.size()),
                          Eigen::VectorXd::Zero(offset_.size()), Eigen::VectorXd::Zero(offset_.size())};
    Eigen::VectorXd position_envelope = Eigen::VectorXd::Zero(offset_.size());
    for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
        const double frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
        const Eigen::VectorXd amplitude =
            (sin_coefficients_.col(harmonic).array().square() +
             cos_coefficients_.col(harmonic).array().square())
                .sqrt()
                .matrix();
        position_envelope += amplitude;
        bounds.maximum_absolute_velocity += frequency * amplitude;
        bounds.maximum_absolute_acceleration += frequency * frequency * amplitude;
        bounds.maximum_absolute_jerk += frequency * frequency * frequency * amplitude;
    }
    bounds.minimum_position -= position_envelope;
    bounds.maximum_position += position_envelope;
    return bounds;
}

ExcitationValidationResult ValidateExcitationTrajectory(const HarmonicTrajectory& trajectory,
                                                        const ExcitationLimits& limits,
                                                        double duration_seconds,
                                                        double sample_period_seconds) {
    const Eigen::Index dof = static_cast<Eigen::Index>(trajectory.dof());
    const bool dimensions_match = limits.minimum_position.size() == dof && limits.maximum_position.size() == dof &&
                                  limits.maximum_velocity.size() == dof &&
                                  limits.maximum_acceleration.size() == dof && limits.maximum_jerk.size() == dof;
    if (!dimensions_match || !limits.minimum_position.allFinite() || !limits.maximum_position.allFinite() ||
        !limits.maximum_velocity.allFinite() || !limits.maximum_acceleration.allFinite() ||
        !limits.maximum_jerk.allFinite() ||
        (limits.minimum_position.array() >= limits.maximum_position.array()).any() ||
        (limits.maximum_velocity.array() <= 0.0).any() || (limits.maximum_acceleration.array() <= 0.0).any() ||
        (limits.maximum_jerk.array() <= 0.0).any() || !std::isfinite(duration_seconds) ||
        duration_seconds <= 0.0 || !std::isfinite(sample_period_seconds) || sample_period_seconds <= 0.0) {
        throw std::invalid_argument("excitation validation configuration is invalid");
    }
    const double requested_steps = std::ceil(duration_seconds / sample_period_seconds);
    if (!std::isfinite(requested_steps) || requested_steps > 10000000.0) {
        throw std::invalid_argument("excitation validation requests too many samples");
    }
    const std::size_t steps = static_cast<std::size_t>(requested_steps);
    ExcitationValidationResult result;
    result.passed = true;
    result.peak_absolute_velocity = Eigen::VectorXd::Zero(dof);
    result.peak_absolute_acceleration = Eigen::VectorXd::Zero(dof);
    result.peak_absolute_jerk = Eigen::VectorXd::Zero(dof);
    result.conservative_position_envelope = Eigen::VectorXd::Zero(dof);
    result.conservative_velocity_bound = Eigen::VectorXd::Zero(dof);
    result.conservative_acceleration_bound = Eigen::VectorXd::Zero(dof);
    result.conservative_jerk_bound = Eigen::VectorXd::Zero(dof);
    const HarmonicBounds conservative = trajectory.ConservativeBounds();
    result.conservative_position_envelope =
        (conservative.maximum_position - conservative.minimum_position) * 0.5;
    result.conservative_velocity_bound = conservative.maximum_absolute_velocity;
    result.conservative_acceleration_bound = conservative.maximum_absolute_acceleration;
    result.conservative_jerk_bound = conservative.maximum_absolute_jerk;
    if ((conservative.minimum_position.array() < limits.minimum_position.array()).any() ||
        (conservative.maximum_position.array() > limits.maximum_position.array()).any()) {
        result.passed = false;
        result.reason = "conservative position bound exceeded";
        return result;
    }
    if ((conservative.maximum_absolute_velocity.array() > limits.maximum_velocity.array()).any()) {
        result.passed = false;
        result.reason = "conservative velocity bound exceeded";
        return result;
    }
    if ((conservative.maximum_absolute_acceleration.array() > limits.maximum_acceleration.array()).any()) {
        result.passed = false;
        result.reason = "conservative acceleration bound exceeded";
        return result;
    }
    if ((conservative.maximum_absolute_jerk.array() > limits.maximum_jerk.array()).any()) {
        result.passed = false;
        result.reason = "conservative jerk bound exceeded";
        return result;
    }
    for (std::size_t step = 0; step <= steps; ++step) {
        const double time = std::min(duration_seconds, static_cast<double>(step) * sample_period_seconds);
        const ExcitationSample sample = trajectory.Evaluate(time);
        ++result.sample_count;
        result.peak_absolute_velocity = result.peak_absolute_velocity.cwiseMax(sample.velocity.cwiseAbs());
        result.peak_absolute_acceleration =
            result.peak_absolute_acceleration.cwiseMax(sample.acceleration.cwiseAbs());
        result.peak_absolute_jerk = result.peak_absolute_jerk.cwiseMax(sample.jerk.cwiseAbs());
        if ((sample.position.array() < limits.minimum_position.array()).any() ||
            (sample.position.array() > limits.maximum_position.array()).any()) {
            result.passed = false;
            result.reason = "position limit exceeded";
            break;
        }
        if ((sample.velocity.cwiseAbs().array() > limits.maximum_velocity.array()).any()) {
            result.passed = false;
            result.reason = "velocity limit exceeded";
            break;
        }
        if ((sample.acceleration.cwiseAbs().array() > limits.maximum_acceleration.array()).any()) {
            result.passed = false;
            result.reason = "acceleration limit exceeded";
            break;
        }
        if ((sample.jerk.cwiseAbs().array() > limits.maximum_jerk.array()).any()) {
            result.passed = false;
            result.reason = "jerk limit exceeded";
            break;
        }
    }
    return result;
}

void WriteRawDatasetCsv(const RawDataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid raw dataset: " + reason);
    if (path.empty()) throw std::invalid_argument("raw dataset output path must not be empty");
    for (const auto& name : dataset.joint_names) {
        if (name.find(',') != std::string::npos || name.find('\n') != std::string::npos ||
            name.find('\r') != std::string::npos) {
            throw std::invalid_argument("joint names in CSV must not contain commas or newlines");
        }
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open raw dataset output: " + path);
    output << "time_s";
    for (const char* prefix : {"q/", "tau/"}) {
        for (const auto& name : dataset.joint_names) output << ',' << prefix << name;
    }
    output << '\n' << std::setprecision(17);
    for (Eigen::Index row = 0; row < dataset.time_seconds.size(); ++row) {
        output << dataset.time_seconds[row];
        for (Eigen::Index joint = 0; joint < dataset.position.cols(); ++joint) {
            output << ',' << dataset.position(row, joint);
        }
        for (Eigen::Index joint = 0; joint < dataset.torque.cols(); ++joint) {
            output << ',' << dataset.torque(row, joint);
        }
        output << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing raw dataset: " + path);
}

RawDataset ReadRawDatasetCsv(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("raw dataset input path must not be empty");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open raw dataset input: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("raw dataset CSV is empty");
    const auto header = SplitCsvLine(line);
    if (header.size() < 3 || header[0] != "time_s" || (header.size() - 1) % 2 != 0) {
        throw std::runtime_error("raw dataset CSV header has an invalid shape");
    }
    const std::size_t dof = (header.size() - 1) / 2;
    RawDataset dataset;
    for (std::size_t joint = 0; joint < dof; ++joint) {
        if (header[1 + joint].rfind("q/", 0) != 0 || header[1 + joint].size() <= 2) {
            throw std::runtime_error("invalid raw q column name");
        }
        dataset.joint_names.push_back(header[1 + joint].substr(2));
        if (header[1 + dof + joint] != "tau/" + dataset.joint_names.back()) {
            throw std::runtime_error("raw torque columns do not share the q joint order");
        }
    }
    std::vector<double> times;
    std::vector<Eigen::VectorXd> positions;
    std::vector<Eigen::VectorXd> torques;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = SplitCsvLine(line);
        if (fields.empty()) continue;
        if (fields.size() != header.size()) {
            throw std::runtime_error("raw dataset row has the wrong field count at line " +
                                     std::to_string(line_number));
        }
        times.push_back(ParseFinite(fields[0]));
        positions.emplace_back(static_cast<Eigen::Index>(dof));
        torques.emplace_back(static_cast<Eigen::Index>(dof));
        for (std::size_t joint = 0; joint < dof; ++joint) {
            positions.back()[static_cast<Eigen::Index>(joint)] = ParseFinite(fields[1 + joint]);
            torques.back()[static_cast<Eigen::Index>(joint)] = ParseFinite(fields[1 + dof + joint]);
        }
    }
    dataset.time_seconds.resize(static_cast<Eigen::Index>(times.size()));
    dataset.position.resize(static_cast<Eigen::Index>(times.size()), static_cast<Eigen::Index>(dof));
    dataset.torque.resize(dataset.position.rows(), dataset.position.cols());
    for (std::size_t row = 0; row < times.size(); ++row) {
        dataset.time_seconds[static_cast<Eigen::Index>(row)] = times[row];
        dataset.position.row(static_cast<Eigen::Index>(row)) = positions[row].transpose();
        dataset.torque.row(static_cast<Eigen::Index>(row)) = torques[row].transpose();
    }
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::runtime_error("raw dataset CSV is invalid: " + reason);
    return dataset;
}

void WriteDatasetCsv(const Dataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid dataset: " + reason);
    if (path.empty()) throw std::invalid_argument("dataset output path must not be empty");
    for (const auto& name : dataset.joint_names) {
        if (name.find(',') != std::string::npos || name.find('\n') != std::string::npos ||
            name.find('\r') != std::string::npos) {
            throw std::invalid_argument("joint names in CSV must not contain commas or newlines");
        }
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open dataset output: " + path);
    output << "time_s";
    for (const char* prefix : {"q/", "dq/", "ddq/", "tau/"}) {
        for (const auto& name : dataset.joint_names) output << ',' << prefix << name;
    }
    output << '\n' << std::setprecision(17);
    for (const auto& sample : dataset.samples) {
        output << sample.time_seconds;
        for (const auto* values : {&sample.position, &sample.velocity, &sample.acceleration, &sample.torque}) {
            for (Eigen::Index index = 0; index < values->size(); ++index) output << ',' << (*values)[index];
        }
        output << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing dataset: " + path);
}

Dataset ReadDatasetCsv(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("dataset input path must not be empty");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open dataset input: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("dataset CSV is empty");
    const auto header = SplitCsvLine(line);
    if (header.size() < 5 || header[0] != "time_s" || (header.size() - 1) % 4 != 0) {
        throw std::runtime_error("dataset CSV header has an invalid shape");
    }
    const std::size_t dof = (header.size() - 1) / 4;
    Dataset dataset;
    dataset.joint_names.reserve(dof);
    for (std::size_t joint = 0; joint < dof; ++joint) {
        constexpr const char* prefix = "q/";
        const auto& field = header[1 + joint];
        if (field.rfind(prefix, 0) != 0 || field.size() <= 2) throw std::runtime_error("invalid q column name");
        dataset.joint_names.push_back(field.substr(2));
    }
    for (std::size_t group = 0; group < 4; ++group) {
        const char* prefix = group == 0 ? "q/" : group == 1 ? "dq/" : group == 2 ? "ddq/" : "tau/";
        for (std::size_t joint = 0; joint < dof; ++joint) {
            if (header[1 + group * dof + joint] != prefix + dataset.joint_names[joint]) {
                throw std::runtime_error("dataset CSV columns do not share the same joint order");
            }
        }
    }
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = SplitCsvLine(line);
        if (fields.empty()) continue;
        if (fields.size() != header.size()) {
            throw std::runtime_error("dataset CSV row has the wrong field count at line " +
                                     std::to_string(line_number));
        }
        Sample sample;
        sample.time_seconds = ParseFinite(fields[0]);
        sample.position.resize(static_cast<Eigen::Index>(dof));
        sample.velocity.resize(static_cast<Eigen::Index>(dof));
        sample.acceleration.resize(static_cast<Eigen::Index>(dof));
        sample.torque.resize(static_cast<Eigen::Index>(dof));
        std::array<Eigen::VectorXd*, 4> vectors{&sample.position, &sample.velocity, &sample.acceleration,
                                               &sample.torque};
        for (std::size_t group = 0; group < vectors.size(); ++group) {
            for (std::size_t joint = 0; joint < dof; ++joint) {
                (*vectors[group])[static_cast<Eigen::Index>(joint)] = ParseFinite(fields[1 + group * dof + joint]);
            }
        }
        dataset.samples.push_back(std::move(sample));
    }
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::runtime_error("dataset CSV is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                    const Eigen::VectorXd& time_seconds,
                                    const Eigen::MatrixXd& position,
                                    const Eigen::MatrixXd& torque,
                                    const PreprocessOptions& options) {
    if (joint_names.empty() || time_seconds.size() < 3 || position.rows() != time_seconds.size() ||
        torque.rows() != time_seconds.size() || position.cols() != static_cast<Eigen::Index>(joint_names.size()) ||
        torque.cols() != position.cols() || !time_seconds.allFinite() || !position.allFinite() || !torque.allFinite()) {
        throw std::invalid_argument("preprocessing inputs have inconsistent dimensions or non-finite values");
    }
    if (options.smoothing_window == 0 || options.smoothing_window % 2 == 0 ||
        options.torque_smoothing_window == 0 || options.torque_smoothing_window % 2 == 0) {
        throw std::invalid_argument("position and torque smoothing windows must be positive odd numbers");
    }
    if (options.trim_front + options.trim_back >= static_cast<std::size_t>(time_seconds.size())) {
        throw std::invalid_argument("trimming removes every sample");
    }
    const Eigen::MatrixXd filtered_position = SmoothSymmetric(position, options.smoothing_window);
    const Eigen::MatrixXd filtered_velocity =
        SmoothSymmetric(Differentiate(time_seconds, filtered_position), options.smoothing_window);
    const Eigen::MatrixXd filtered_acceleration =
        SmoothSymmetric(Differentiate(time_seconds, filtered_velocity), options.smoothing_window);
    const Eigen::MatrixXd filtered_torque = SmoothSymmetric(torque, options.torque_smoothing_window);

    Dataset dataset;
    dataset.joint_names = joint_names;
    const Eigen::Index begin = static_cast<Eigen::Index>(options.trim_front);
    const Eigen::Index end = time_seconds.size() - static_cast<Eigen::Index>(options.trim_back);
    dataset.samples.reserve(static_cast<std::size_t>(end - begin));
    for (Eigen::Index row = begin; row < end; ++row) {
        dataset.samples.push_back(Sample{time_seconds[row], filtered_position.row(row).transpose(),
                                         filtered_velocity.row(row).transpose(),
                                         filtered_acceleration.row(row).transpose(),
                                         filtered_torque.row(row).transpose()});
    }
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("preprocessed dataset is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const RawDataset& dataset, const PreprocessOptions& options) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid raw dataset: " + reason);
    return PreprocessPositionAndTorque(dataset.joint_names, dataset.time_seconds, dataset.position, dataset.torque,
                                       options);
}

bool RegressionProblem::Validate(std::string* reason) const {
    if (sample_count == 0 || dof == 0 || regressor.rows() == 0 || regressor.cols() == 0) {
        return Fail(reason, "regression problem must not be empty");
    }
    if (regressor.rows() != observation.size() || row_weights.size() != observation.size() ||
        regressor.rows() != static_cast<Eigen::Index>(sample_count * dof)) {
        return Fail(reason, "regression problem dimensions are inconsistent");
    }
    if (!regressor.allFinite() || !observation.allFinite() || !row_weights.allFinite() ||
        (row_weights.array() <= 0.0).any()) {
        return Fail(reason, "regression values must be finite and row weights positive");
    }
    return true;
}

RegressionProblem BuildRegressionProblem(const Dataset& dataset, const RegressorBackend& backend) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid dataset: " + reason);
    const std::size_t parameter_count = backend.parameter_count();
    auto parameter_names = backend.parameter_names();
    auto sorted_parameter_names = parameter_names;
    std::sort(sorted_parameter_names.begin(), sorted_parameter_names.end());
    if (parameter_count == 0 || parameter_names.size() != parameter_count ||
        sorted_parameter_names.front().empty() ||
        std::adjacent_find(sorted_parameter_names.begin(), sorted_parameter_names.end()) !=
            sorted_parameter_names.end()) {
        throw std::invalid_argument("regressor backend has an invalid parameter contract");
    }
    RegressionProblem problem;
    problem.sample_count = dataset.samples.size();
    problem.dof = dataset.dof();
    problem.regressor.resize(static_cast<Eigen::Index>(problem.sample_count * problem.dof),
                             static_cast<Eigen::Index>(parameter_count));
    problem.observation.resize(problem.regressor.rows());
    problem.row_weights = Eigen::VectorXd::Ones(problem.regressor.rows());
    for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
        const Eigen::MatrixXd block = backend.Compute(dataset.samples[index]);
        if (block.rows() != static_cast<Eigen::Index>(problem.dof) ||
            block.cols() != static_cast<Eigen::Index>(parameter_count) || !block.allFinite()) {
            throw std::runtime_error("regressor backend returned an invalid block");
        }
        const Eigen::Index row = static_cast<Eigen::Index>(index * problem.dof);
        problem.regressor.middleRows(row, static_cast<Eigen::Index>(problem.dof)) = block;
        problem.observation.segment(row, static_cast<Eigen::Index>(problem.dof)) = dataset.samples[index].torque;
    }
    return problem;
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
    auto sorted_names = joint_names_;
    std::sort(sorted_names.begin(), sorted_names.end());
    if (std::adjacent_find(sorted_names.begin(), sorted_names.end()) != sorted_names.end()) {
        throw std::invalid_argument("joint actuator regressor names must be unique");
    }
    terms_per_joint_ = static_cast<std::size_t>(options_.drive_inertia) +
                       static_cast<std::size_t>(options_.viscous_friction) +
                       static_cast<std::size_t>(options_.coulomb_friction) +
                       static_cast<std::size_t>(options_.torque_bias);
    if (terms_per_joint_ == 0) throw std::invalid_argument("joint actuator regressor enables no terms");
}

std::size_t JointActuatorRegressor::parameter_count() const noexcept {
    return base_->parameter_count() + terms_per_joint_ * joint_names_.size();
}

std::vector<std::string> JointActuatorRegressor::parameter_names() const {
    auto names = base_->parameter_names();
    names.reserve(parameter_count());
    for (const auto& joint : joint_names_) {
        if (options_.drive_inertia) names.push_back("actuator/" + joint + "/drive_inertia");
        if (options_.viscous_friction) names.push_back("actuator/" + joint + "/viscous_friction");
        if (options_.coulomb_friction) names.push_back("actuator/" + joint + "/coulomb_friction");
        if (options_.torque_bias) names.push_back("actuator/" + joint + "/torque_bias");
    }
    return names;
}

Eigen::MatrixXd JointActuatorRegressor::Compute(const Sample& sample) const {
    const Eigen::Index dof = static_cast<Eigen::Index>(joint_names_.size());
    if (sample.position.size() != dof || sample.velocity.size() != dof || sample.acceleration.size() != dof ||
        !sample.position.allFinite() || !sample.velocity.allFinite() || !sample.acceleration.allFinite()) {
        throw std::invalid_argument("joint actuator regressor sample dimension is invalid");
    }
    const Eigen::MatrixXd base = base_->Compute(sample);
    if (base.rows() != dof || base.cols() != static_cast<Eigen::Index>(base_->parameter_count()) ||
        !base.allFinite()) {
        throw std::runtime_error("base regressor returned an invalid block");
    }
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(dof, static_cast<Eigen::Index>(parameter_count()));
    output.leftCols(base.cols()) = base;
    Eigen::Index column = base.cols();
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

IdentificationResult Solve(const RegressionProblem& problem, const SolverOptions& options) {
    std::string reason;
    if (!problem.Validate(&reason)) throw std::invalid_argument("invalid regression problem: " + reason);
    if (!std::isfinite(options.ridge) || options.ridge < 0.0 ||
        !std::isfinite(options.relative_rank_tolerance) || options.relative_rank_tolerance <= 0.0 ||
        !std::isfinite(options.huber_delta) || options.huber_delta < 0.0 || options.max_iterations == 0 ||
        !std::isfinite(options.convergence_tolerance) || options.convergence_tolerance <= 0.0) {
        throw std::invalid_argument("solver options are invalid");
    }
    Eigen::VectorXd weights = problem.row_weights;
    Eigen::VectorXd solve_weights = weights;
    Eigen::VectorXd previous;
    LinearSolveResult linear;
    std::size_t iterations = 0;
    bool converged = options.huber_delta == 0.0;
    const std::size_t limit = options.huber_delta > 0.0 ? options.max_iterations : 1;
    for (; iterations < limit; ++iterations) {
        solve_weights = weights;
        linear = SolveWeighted(problem, weights, options);
        if (!linear.parameters.allFinite()) throw std::runtime_error("parameter solution is non-finite");
        if (previous.size() == linear.parameters.size()) {
            const double scale = std::max(1.0, previous.norm());
            if ((linear.parameters - previous).norm() <= options.convergence_tolerance * scale) {
                converged = true;
                ++iterations;
                break;
            }
        }
        previous = linear.parameters;
        if (options.huber_delta > 0.0) {
            const Eigen::VectorXd residual = problem.observation - problem.regressor * linear.parameters;
            for (Eigen::Index row = 0; row < residual.size(); ++row) {
                const double absolute = std::abs(residual[row]);
                const double robust = absolute <= options.huber_delta ? 1.0 : options.huber_delta / absolute;
                weights[row] = problem.row_weights[row] * robust;
            }
        }
    }
    IdentificationResult result;
    result.parameters = linear.parameters;
    result.predicted = problem.regressor * result.parameters;
    result.residual = problem.observation - result.predicted;
    // These are the exact weights used for the returned parameter vector. If
    // IRLS reaches its iteration limit, a newly proposed weight vector must
    // not be reported as though another solve had consumed it.
    result.final_row_weights = solve_weights;
    result.rank = linear.rank;
    result.condition_number = linear.rank < static_cast<std::size_t>(problem.regressor.cols()) ||
                                      linear.singular_values.size() == 0 || linear.singular_values.tail(1)[0] <= 0.0
                                  ? std::numeric_limits<double>::infinity()
                                  : linear.singular_values[0] / linear.singular_values.tail(1)[0];
    result.rmse = std::sqrt(result.residual.squaredNorm() / static_cast<double>(result.residual.size()));
    result.max_absolute_error = result.residual.cwiseAbs().maxCoeff();
    result.iterations = iterations;
    result.converged = converged;
    const Eigen::MatrixXd information =
        problem.regressor.transpose() * solve_weights.asDiagonal() * problem.regressor +
                                        options.ridge * Eigen::MatrixXd::Identity(problem.regressor.cols(),
                                                                                  problem.regressor.cols());
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition(information);
    const double weighted_squared_error =
        result.residual.dot(solve_weights.asDiagonal() * result.residual);
    const double variance = weighted_squared_error /
                            static_cast<double>(std::max<Eigen::Index>(1, problem.regressor.rows() -
                                                                             static_cast<Eigen::Index>(result.rank)));
    result.covariance = variance * decomposition.pseudoInverse();
    return result;
}

ValidationMetrics ValidateParameters(const RegressionProblem& problem, const Eigen::VectorXd& parameters) {
    std::string reason;
    if (!problem.Validate(&reason) || parameters.size() != problem.regressor.cols() || !parameters.allFinite()) {
        throw std::invalid_argument("validation inputs are invalid");
    }
    const Eigen::VectorXd residual = problem.observation - problem.regressor * parameters;
    ValidationMetrics metrics;
    metrics.row_count = static_cast<std::size_t>(residual.size());
    metrics.rmse = std::sqrt(residual.squaredNorm() / static_cast<double>(residual.size()));
    metrics.mean_absolute_error = residual.cwiseAbs().mean();
    metrics.max_absolute_error = residual.cwiseAbs().maxCoeff();
    const double scale = std::sqrt(problem.observation.squaredNorm() / static_cast<double>(problem.observation.size()));
    metrics.normalized_rmse =
        scale > 0.0
            ? metrics.rmse / scale
            : (metrics.rmse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    return metrics;
}

PhysicalValidationResult ValidateMassMatrices(const Eigen::MatrixXd& joint_positions,
                                              const Eigen::VectorXd& parameters,
                                              const MassMatrixFunction& mass_matrix,
                                              double minimum_eigenvalue) {
    if (joint_positions.rows() == 0 || joint_positions.cols() == 0 || !joint_positions.allFinite() ||
        parameters.size() == 0 || !parameters.allFinite() || !mass_matrix || !std::isfinite(minimum_eigenvalue) ||
        minimum_eigenvalue <= 0.0) {
        throw std::invalid_argument("mass-matrix validation inputs are invalid");
    }
    PhysicalValidationResult result;
    result.passed = true;
    result.minimum_eigenvalue = std::numeric_limits<double>::infinity();
    for (Eigen::Index row = 0; row < joint_positions.rows(); ++row) {
        const Eigen::MatrixXd mass = mass_matrix(joint_positions.row(row).transpose(), parameters);
        ++result.tested;
        if (mass.rows() != joint_positions.cols() || mass.cols() != joint_positions.cols() || !mass.allFinite() ||
            !mass.isApprox(mass.transpose(), 1e-9)) {
            ++result.failed;
            result.passed = false;
            continue;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success) {
            ++result.failed;
            result.passed = false;
            continue;
        }
        const double value = solver.eigenvalues().minCoeff();
        result.minimum_eigenvalue = std::min(result.minimum_eigenvalue, value);
        if (value < minimum_eigenvalue) {
            ++result.failed;
            result.passed = false;
        }
    }
    return result;
}

}  // namespace robot_identification
