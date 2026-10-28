#include "robot_identification/identification.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace robot_identification {
namespace {

bool Fail(std::string* reason, const std::string& message) {
    if (reason) *reason = message;
    return false;
}

bool ValidateRawInputs(const std::vector<std::string>& joint_names,
                       const Eigen::VectorXd& time_seconds,
                       const Eigen::MatrixXd& position,
                       const Eigen::MatrixXd& torque,
                       std::string* reason) {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    auto names = joint_names;
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

struct ErrorMagnitudes {
    double rms{0.0};
    double mean_absolute{0.0};
    double maximum{0.0};
};

bool ParametersConverged(const Eigen::VectorXd& current, const Eigen::VectorXd& previous, double tolerance) {
    const double scale = std::max({1.0, current.cwiseAbs().maxCoeff(), previous.cwiseAbs().maxCoeff()});
    double previous_norm = 0.0;
    double change_norm = 0.0;
    for (Eigen::Index row = 0; row < current.size(); ++row) {
        const double difference = current[row] - previous[row];
        // Subtract first to retain small updates; normalize first only when
        // opposite-sign finite parameters overflow their difference.
        const double change =
            std::isfinite(difference) ? difference / scale : current[row] / scale - previous[row] / scale;
        previous_norm = std::hypot(previous_norm, previous[row] / scale);
        change_norm = std::hypot(change_norm, change);
    }
    return change_norm / std::max(1.0 / scale, previous_norm) <= tolerance;
}

ErrorMagnitudes MeasureErrors(const Eigen::VectorXd& values) {
    if (!values.allFinite()) throw std::overflow_error("error metrics require finite predictions and residuals");
    ErrorMagnitudes result;
    result.maximum = values.cwiseAbs().maxCoeff();
    if (result.maximum == 0.0) return result;
    // Scale before squaring or summing: the final metrics are bounded by the
    // largest magnitude even when the unscaled squared norm is not representable.
    long double absolute_sum = 0.0L;
    long double squared_sum = 0.0L;
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const long double normalized = std::abs(static_cast<long double>(values[row])) / result.maximum;
        absolute_sum += normalized;
        squared_sum += normalized * normalized;
    }
    result.rms = static_cast<double>(std::min(1.0L, std::sqrt(squared_sum / values.size())) * result.maximum);
    result.mean_absolute = static_cast<double>(std::min(1.0L, absolute_sum / values.size()) * result.maximum);
    return result;
}

Eigen::MatrixXd SmoothSymmetric(const Eigen::MatrixXd& values, std::size_t window) {
    if (window == 1) return values;
    const Eigen::Index radius =
        static_cast<Eigen::Index>(std::min(window / 2, static_cast<std::size_t>(values.rows() - 1)));
    Eigen::MatrixXd output(values.rows(), values.cols());
    // Eigen's short reductions are faster for the usual small windows.
    if (radius <= 32) {
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
            output.row(row) = values.middleRows(begin, end - begin + 1).colwise().mean();
            for (Eigen::Index column = 0; column < values.cols(); ++column) {
                if (std::isfinite(output(row, column))) continue;
                // A finite mean can have an overflowing intermediate sum.
                // Normalize only exceptional windows, keeping the common path fast.
                const double scale = values.col(column).segment(begin, end - begin + 1).cwiseAbs().maxCoeff();
                long double normalized_sum = 0.0L;
                for (Eigen::Index index = begin; index <= end; ++index) {
                    normalized_sum += static_cast<long double>(values(index, column)) / scale;
                }
                const long double mean = normalized_sum / (end - begin + 1);
                output(row, column) = static_cast<double>(std::clamp(mean, -1.0L, 1.0L) * scale);
            }
        }
        return output;
    }
    const Eigen::Index rebase_interval =
        static_cast<Eigen::Index>(std::min(window, static_cast<std::size_t>(values.rows())));
    for (Eigen::Index column = 0; column < values.cols(); ++column) {
        long double sum = 0.0L;
        long double correction = 0.0L;
        const auto add = [&](long double value) {
            const long double next = sum + value;
            correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
            sum = next;
        };
        Eigen::Index previous_begin = 0;
        Eigen::Index previous_end = -1;
        Eigen::Index until_rebase = 0;
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
            if (until_rebase == 0) {
                // Rebuild once per window to bound roundoff accumulation;
                // the total work remains linear in the number of samples.
                sum = correction = 0.0L;
                for (Eigen::Index index = begin; index <= end; ++index) add(values(index, column));
                until_rebase = rebase_interval;
            } else {
                if (begin != previous_begin) add(-static_cast<long double>(values(previous_begin, column)));
                if (end != previous_end) add(values(end, column));
            }
            output(row, column) = static_cast<double>((sum + correction) / (end - begin + 1));
            previous_begin = begin;
            previous_end = end;
            --until_rebase;
        }
    }
    return output;
}

Eigen::MatrixXd Differentiate(const Eigen::VectorXd& time, const Eigen::MatrixXd& values) {
    // Internal preprocessing helper: dimensions and timestamps were checked by
    // ValidateRawInputs before smoothing. Keep arithmetic checks below.
    Eigen::MatrixXd derivative(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const Eigen::Index first = row == 0 ? 0 : row + 1 == values.rows() ? row - 2 : row - 1;
        const Eigen::Index second = first + 1;
        const Eigen::Index third = first + 2;
        const double left_step = time[second] - time[first];
        const double right_step = time[third] - time[second];
        const double span = left_step + right_step;
        if (!std::isfinite(span)) {
            throw std::overflow_error("timestamp spacing is numerically unusable for differentiation");
        }
        // Differentiate the same three-point quadratic through its secant
        // slopes, so a constant position offset cancels before division.
        const double ratio = row == first ? -left_step / span : row == third ? right_step / span : left_step / span;
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            const double left_slope = (values(second, column) - values(first, column)) / left_step;
            const double right_slope = (values(third, column) - values(second, column)) / right_step;
            const double base = row == third ? right_slope : left_slope;
            derivative(row, column) = base + ratio * (right_slope - left_slope);
        }
    }
    if (!derivative.allFinite()) throw std::overflow_error("differentiation produced non-finite values");
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

void ValidateCsvJointNames(const std::vector<std::string>& names) {
    // Sort views to detect duplicates without copying every joint-name string.
    std::vector<std::string_view> ordered;
    ordered.reserve(names.size());
    for (const auto& name : names) {
        if (name.empty() || name.find_first_of(",\r\n") != std::string::npos) {
            throw std::runtime_error("CSV joint names must be non-empty and contain no commas or newlines");
        }
        ordered.emplace_back(name);
    }
    std::sort(ordered.begin(), ordered.end());
    if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end()) {
        throw std::runtime_error("CSV joint names must be unique");
    }
}

bool ParseCsvRow(std::string& line, std::vector<double>& values, std::size_t line_number, double& previous_time) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const char* cursor = line.data();
    const char* end = cursor + line.size();
    if (cursor == end) return false;

    // from_chars bounds conversion to each field without changing the buffer
    // or consulting the process locale. Preserve leading ASCII whitespace,
    // an explicit plus sign, and the existing 0x-prefixed hexadecimal syntax.
    std::size_t column = 0;
    while (true) {
        if (column == values.size()) {
            throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
        }
        const char* field_end = std::find(cursor, end, ',');
        const char* number = cursor;
        while (number != field_end && (*number == ' ' || (*number >= '\t' && *number <= '\r'))) ++number;
        const bool negative = number != field_end && *number == '-';
        if (number != field_end && (*number == '+' || *number == '-')) ++number;
        auto format = std::chars_format::general;
        if (field_end - number >= 2 && number[0] == '0' && (number[1] == 'x' || number[1] == 'X')) {
            number += 2;
            format = std::chars_format::hex;
        }
        const auto location = [&]() {
            return " at line " + std::to_string(line_number) + ", column " + std::to_string(column + 1);
        };
        if (number == field_end || *number == '+' || *number == '-') {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        double value = 0.0;
        const auto conversion = std::from_chars(number, field_end, value, format);
        if (conversion.ec == std::errc::invalid_argument) {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        if (conversion.ec == std::errc::result_out_of_range) {
            throw std::out_of_range("CSV numeric field is out of range" + location());
        }
        if (conversion.ptr != field_end || !std::isfinite(value)) {
            throw std::runtime_error("CSV numeric field is invalid" + location());
        }
        values[column++] = negative ? -value : value;
        if (field_end == end) break;
        cursor = field_end + 1;
    }
    if (column != values.size()) {
        throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
    }
    if (values[0] <= previous_time) {
        throw std::runtime_error("CSV timestamp must be strictly increasing at line " + std::to_string(line_number));
    }
    previous_time = values[0];
    return true;
}

struct LinearSolveResult {
    Eigen::VectorXd parameters;
    Eigen::VectorXd singular_values;
    std::size_t rank{0};
};

class WeightedSolveWorkspace {
public:
    WeightedSolveWorkspace(const RegressionProblem& problem, const SolverOptions& options)
        : data_rows_(problem.regressor.rows()),
          parameter_count_(problem.regressor.cols()),
          ridge_rows_(options.ridge > 0.0 ? parameter_count_ : 0) {
        if (data_rows_ > std::numeric_limits<Eigen::Index>::max() - ridge_rows_) {
            throw std::overflow_error("regularized regression row count overflowed");
        }
        weighted_regressor_.resize(data_rows_ + ridge_rows_, parameter_count_);
        weighted_observation_.resize(data_rows_ + ridge_rows_);
        sqrt_weights_.resize(data_rows_);
        const unsigned int diagnostics_options = ridge_rows_ == 0 ? Eigen::ComputeThinU | Eigen::ComputeThinV : 0;
        diagnostic_svd_ = std::make_unique<Svd>(data_rows_, parameter_count_, diagnostics_options);
        diagnostic_svd_->setThreshold(options.relative_rank_tolerance);
        if (ridge_rows_ > 0) {
            weighted_regressor_.bottomRows(ridge_rows_).setZero();
            weighted_regressor_.bottomRows(ridge_rows_).diagonal().setConstant(std::sqrt(options.ridge));
            weighted_observation_.tail(ridge_rows_).setZero();
            regularized_svd_ = std::make_unique<Svd>(data_rows_ + ridge_rows_, parameter_count_,
                                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
            regularized_svd_->setThreshold(options.relative_rank_tolerance);
        }
    }

    void Compute(const RegressionProblem& problem, const Eigen::VectorXd& weights, LinearSolveResult& result) {
        sqrt_weights_ = weights.array().sqrt();
        weighted_regressor_.topRows(data_rows_) = problem.regressor.array().colwise() * sqrt_weights_.array();
        weighted_observation_.head(data_rows_) = problem.observation.array() * sqrt_weights_.array();
        if (!weighted_regressor_.topRows(data_rows_).allFinite() ||
            !weighted_observation_.head(data_rows_).allFinite()) {
            throw std::overflow_error("weighted regression values are non-finite");
        }
        // Diagnose measured data before ridge rows: regularization must not
        // disguise an unobservable parameter. Reuse each decomposition's
        // scratch across IRLS iterations without changing its input shape.
        if (ridge_rows_ == 0) {
            diagnostic_svd_->compute(weighted_regressor_);
        } else {
            diagnostic_svd_->compute(weighted_regressor_.topRows(data_rows_));
        }
        if (diagnostic_svd_->info() != Eigen::Success || !diagnostic_svd_->singularValues().allFinite()) {
            throw std::runtime_error("regression diagnostic decomposition failed");
        }
        result.singular_values = diagnostic_svd_->singularValues();
        result.rank = static_cast<std::size_t>(diagnostic_svd_->rank());
        if (ridge_rows_ == 0) {
            result.parameters = diagnostic_svd_->solve(weighted_observation_);
        } else {
            regularized_svd_->compute(weighted_regressor_);
            if (regularized_svd_->info() != Eigen::Success || !regularized_svd_->singularValues().allFinite()) {
                throw std::runtime_error("regularized regression decomposition failed");
            }
            result.parameters = regularized_svd_->solve(weighted_observation_);
        }
    }

private:
    using Svd = Eigen::JacobiSVD<Eigen::MatrixXd>;
    Eigen::Index data_rows_, parameter_count_, ridge_rows_;
    Eigen::MatrixXd weighted_regressor_;
    Eigen::VectorXd weighted_observation_, sqrt_weights_;
    std::unique_ptr<Svd> diagnostic_svd_, regularized_svd_;
};

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
    return ValidateRawInputs(joint_names, time_seconds, position, torque, reason);
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
        !sin_coefficients_.allFinite() || !cos_coefficients_.allFinite() || !std::isfinite(angular_frequency_) ||
        angular_frequency_ <= 0.0) {
        throw std::invalid_argument("harmonic trajectory configuration is invalid");
    }
}

std::size_t HarmonicTrajectory::dof() const noexcept { return static_cast<std::size_t>(offset_.size()); }

std::size_t HarmonicTrajectory::harmonic_count() const noexcept {
    return static_cast<std::size_t>(sin_coefficients_.cols());
}

double HarmonicTrajectory::period_seconds() const noexcept { return 2.0 * std::acos(-1.0) / angular_frequency_; }

ExcitationSample HarmonicTrajectory::Evaluate(double time_seconds) const {
    if (!std::isfinite(time_seconds)) throw std::invalid_argument("trajectory time must be finite");
    ExcitationSample sample{offset_, Eigen::VectorXd::Zero(offset_.size()), Eigen::VectorXd::Zero(offset_.size()),
                            Eigen::VectorXd::Zero(offset_.size())};
    for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
        const double frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
        const double phase = frequency * time_seconds;
        const double sine = std::sin(phase);
        const double cosine = std::cos(phase);
        for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
            const double position_term =
                sin_coefficients_(joint, harmonic) * sine + cos_coefficients_(joint, harmonic) * cosine;
            const double derivative_term =
                sin_coefficients_(joint, harmonic) * cosine - cos_coefficients_(joint, harmonic) * sine;
            sample.position[joint] += position_term;
            // Multiply the finite coefficient before successive frequency
            // factors: powers of frequency may overflow or underflow even
            // when every requested derivative remains representable.
            const double velocity_term = frequency * derivative_term;
            sample.velocity[joint] += velocity_term;
            sample.acceleration[joint] -= frequency * (frequency * position_term);
            sample.jerk[joint] -= frequency * (frequency * velocity_term);
        }
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
        for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
            const double amplitude = std::hypot(sin_coefficients_(joint, harmonic), cos_coefficients_(joint, harmonic));
            const double velocity = amplitude * frequency;
            const double acceleration = velocity * frequency;
            const double jerk = acceleration * frequency;
            position_envelope[joint] += amplitude;
            bounds.maximum_absolute_velocity[joint] += velocity;
            bounds.maximum_absolute_acceleration[joint] += acceleration;
            bounds.maximum_absolute_jerk[joint] += jerk;
        }
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
                                  limits.maximum_velocity.size() == dof && limits.maximum_acceleration.size() == dof &&
                                  limits.maximum_jerk.size() == dof;
    if (!dimensions_match || !limits.minimum_position.allFinite() || !limits.maximum_position.allFinite() ||
        !limits.maximum_velocity.allFinite() || !limits.maximum_acceleration.allFinite() ||
        !limits.maximum_jerk.allFinite() ||
        (limits.minimum_position.array() >= limits.maximum_position.array()).any() ||
        (limits.maximum_velocity.array() <= 0.0).any() || (limits.maximum_acceleration.array() <= 0.0).any() ||
        (limits.maximum_jerk.array() <= 0.0).any() || !std::isfinite(duration_seconds) || duration_seconds <= 0.0 ||
        !std::isfinite(sample_period_seconds) || sample_period_seconds <= 0.0) {
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
    for (Eigen::Index joint = 0; joint < dof; ++joint) {
        const double width = conservative.maximum_position[joint] - conservative.minimum_position[joint];
        result.conservative_position_envelope[joint] =
            std::isfinite(width)
                ? width * 0.5
                : conservative.maximum_position[joint] * 0.5 - conservative.minimum_position[joint] * 0.5;
    }
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
        result.peak_absolute_acceleration = result.peak_absolute_acceleration.cwiseMax(sample.acceleration.cwiseAbs());
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
    output.imbue(std::locale::classic());
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
    output.close();
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
    const auto maximum_values = static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
    if (header.size() > maximum_values) throw std::overflow_error("raw dataset CSV has too many columns");
    RawDataset dataset;
    dataset.joint_names.reserve(dof);
    for (std::size_t joint = 0; joint < dof; ++joint) {
        if (header[1 + joint].rfind("q/", 0) != 0 || header[1 + joint].size() <= 2) {
            throw std::runtime_error("invalid raw q column name");
        }
        dataset.joint_names.push_back(header[1 + joint].substr(2));
        if (header[1 + dof + joint] != "tau/" + dataset.joint_names.back()) {
            throw std::runtime_error("raw torque columns do not share the q joint order");
        }
    }
    ValidateCsvJointNames(dataset.joint_names);
    std::vector<double> rows;
    std::vector<double> values(header.size());
    std::size_t line_number = 1;
    double previous_time = -std::numeric_limits<double>::infinity();
    while (std::getline(input, line)) {
        ++line_number;
        if (!ParseCsvRow(line, values, line_number, previous_time)) continue;
        if (rows.size() > maximum_values - values.size()) {
            throw std::overflow_error("raw dataset CSV contains too many values");
        }
        rows.insert(rows.end(), values.begin(), values.end());
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("failed while reading raw dataset: " + path);
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Map<const RowMatrix> parsed(rows.data(), static_cast<Eigen::Index>(rows.size() / header.size()),
                                             static_cast<Eigen::Index>(header.size()));
    dataset.time_seconds = parsed.col(0);
    dataset.position = parsed.middleCols(1, static_cast<Eigen::Index>(dof));
    dataset.torque = parsed.rightCols(static_cast<Eigen::Index>(dof));
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
    output.imbue(std::locale::classic());
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
    output.close();
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
    if (dof > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error("dataset CSV has too many joints");
    }
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
    ValidateCsvJointNames(dataset.joint_names);
    std::vector<double> values(header.size());
    std::size_t line_number = 1;
    double previous_time = -std::numeric_limits<double>::infinity();
    while (std::getline(input, line)) {
        ++line_number;
        if (!ParseCsvRow(line, values, line_number, previous_time)) continue;
        Sample sample;
        sample.time_seconds = values[0];
        std::array<Eigen::VectorXd*, 4> vectors{&sample.position, &sample.velocity, &sample.acceleration,
                                                &sample.torque};
        for (std::size_t group = 0; group < vectors.size(); ++group) {
            *vectors[group] =
                Eigen::Map<const Eigen::VectorXd>(values.data() + 1 + group * dof, static_cast<Eigen::Index>(dof));
        }
        dataset.samples.push_back(std::move(sample));
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("failed while reading dataset: " + path);
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::runtime_error("dataset CSV is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                    const Eigen::VectorXd& time_seconds,
                                    const Eigen::MatrixXd& position,
                                    const Eigen::MatrixXd& torque,
                                    const PreprocessOptions& options) {
    std::string reason;
    if (!ValidateRawInputs(joint_names, time_seconds, position, torque, &reason)) {
        throw std::invalid_argument("invalid raw dataset: " + reason);
    }
    if (options.smoothing_window == 0 || options.smoothing_window % 2 == 0 || options.torque_smoothing_window == 0 ||
        options.torque_smoothing_window % 2 == 0) {
        throw std::invalid_argument("position and torque smoothing windows must be positive odd numbers");
    }
    const std::size_t sample_count = static_cast<std::size_t>(time_seconds.size());
    if (options.trim_front >= sample_count || options.trim_back >= sample_count - options.trim_front) {
        throw std::invalid_argument("trimming removes every sample");
    }
    // Keep caller-owned position/torque as views when smoothing is disabled.
    // Derivative matrices already own their storage and can be reused directly.
    const Eigen::MatrixXd position_storage =
        options.smoothing_window == 1 ? Eigen::MatrixXd{} : SmoothSymmetric(position, options.smoothing_window);
    const auto& filtered_position = options.smoothing_window == 1 ? position : position_storage;
    Eigen::MatrixXd filtered_velocity = Differentiate(time_seconds, filtered_position);
    if (options.smoothing_window != 1) {
        filtered_velocity = SmoothSymmetric(filtered_velocity, options.smoothing_window);
    }
    Eigen::MatrixXd filtered_acceleration = Differentiate(time_seconds, filtered_velocity);
    if (options.smoothing_window != 1) {
        filtered_acceleration = SmoothSymmetric(filtered_acceleration, options.smoothing_window);
    }
    const Eigen::MatrixXd torque_storage = options.torque_smoothing_window == 1
                                               ? Eigen::MatrixXd{}
                                               : SmoothSymmetric(torque, options.torque_smoothing_window);
    const auto& filtered_torque = options.torque_smoothing_window == 1 ? torque : torque_storage;

    Dataset dataset;
    dataset.joint_names = joint_names;
    const Eigen::Index begin = static_cast<Eigen::Index>(options.trim_front);
    const Eigen::Index end = time_seconds.size() - static_cast<Eigen::Index>(options.trim_back);
    dataset.samples.reserve(static_cast<std::size_t>(end - begin));
    for (Eigen::Index row = begin; row < end; ++row) {
        dataset.samples.push_back(
            Sample{time_seconds[row], filtered_position.row(row).transpose(), filtered_velocity.row(row).transpose(),
                   filtered_acceleration.row(row).transpose(), filtered_torque.row(row).transpose()});
    }
    if (!dataset.Validate(&reason)) throw std::invalid_argument("preprocessed dataset is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const RawDataset& dataset, const PreprocessOptions& options) {
    return PreprocessPositionAndTorque(dataset.joint_names, dataset.time_seconds, dataset.position, dataset.torque,
                                       options);
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
    if (parameter_count == 0 || parameter_names.size() != parameter_count || sorted_parameter_names.front().empty() ||
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
    if (base.rows() != dof || base.cols() != static_cast<Eigen::Index>(base_->parameter_count()) || !base.allFinite()) {
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
    if (!std::isfinite(options.ridge) || options.ridge < 0.0 || !std::isfinite(options.relative_rank_tolerance) ||
        options.relative_rank_tolerance <= 0.0 || !std::isfinite(options.huber_delta) || options.huber_delta < 0.0 ||
        options.max_iterations == 0 || !std::isfinite(options.convergence_tolerance) ||
        options.convergence_tolerance <= 0.0) {
        throw std::invalid_argument("solver options are invalid");
    }
    Eigen::VectorXd weights = problem.row_weights;
    Eigen::VectorXd solve_weights = weights;
    Eigen::VectorXd previous;
    LinearSolveResult linear;
    WeightedSolveWorkspace workspace(problem, options);
    Eigen::VectorXd robust_residual;
    if (options.huber_delta > 0.0 && options.max_iterations > 1) robust_residual.resize(problem.regressor.rows());
    std::size_t iterations = 0;
    bool converged = options.huber_delta == 0.0;
    const std::size_t limit = options.huber_delta > 0.0 ? options.max_iterations : 1;
    for (; iterations < limit; ++iterations) {
        solve_weights = weights;
        workspace.Compute(problem, weights, linear);
        if (!linear.parameters.allFinite()) throw std::runtime_error("parameter solution is non-finite");
        if (previous.size() == linear.parameters.size()) {
            if (ParametersConverged(linear.parameters, previous, options.convergence_tolerance)) {
                converged = true;
                ++iterations;
                break;
            }
        }
        // Only prepare another weighted solve when an iteration remains.
        if (options.huber_delta > 0.0 && iterations + 1 < limit) {
            previous = linear.parameters;
            robust_residual.noalias() = problem.regressor * linear.parameters;
            robust_residual = problem.observation - robust_residual;
            if (!robust_residual.allFinite()) throw std::overflow_error("robust regression residuals are non-finite");
            for (Eigen::Index row = 0; row < robust_residual.size(); ++row) {
                const double absolute = std::abs(robust_residual[row]);
                const double robust = absolute <= options.huber_delta ? 1.0 : options.huber_delta / absolute;
                weights[row] = problem.row_weights[row] * robust;
            }
        }
    }
    IdentificationResult result;
    result.parameters = std::move(linear.parameters);
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
    const auto errors = MeasureErrors(result.residual);
    result.rmse = errors.rms;
    result.max_absolute_error = errors.maximum;
    result.iterations = iterations;
    result.converged = converged;
    const Eigen::MatrixXd information =
        problem.regressor.transpose() * solve_weights.asDiagonal() * problem.regressor +
        options.ridge * Eigen::MatrixXd::Identity(problem.regressor.cols(), problem.regressor.cols());
    if (!information.allFinite()) throw std::overflow_error("covariance information matrix is non-finite");
    const double weighted_squared_error = result.residual.dot(solve_weights.asDiagonal() * result.residual);
    const double variance =
        weighted_squared_error / static_cast<double>(std::max<Eigen::Index>(
                                     1, problem.regressor.rows() - static_cast<Eigen::Index>(result.rank)));
    if (!std::isfinite(variance)) throw std::overflow_error("covariance residual variance is non-finite");
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition(information);
    result.covariance = variance * decomposition.pseudoInverse();
    if (!result.covariance.allFinite()) throw std::overflow_error("parameter covariance is non-finite");
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
    const auto errors = MeasureErrors(residual);
    metrics.rmse = errors.rms;
    metrics.mean_absolute_error = errors.mean_absolute;
    metrics.max_absolute_error = errors.maximum;
    const double scale = MeasureErrors(problem.observation).rms;
    metrics.normalized_rmse =
        scale > 0.0 ? metrics.rmse / scale : (metrics.rmse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
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
    Eigen::VectorXd position(joint_positions.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(joint_positions.cols());
    for (Eigen::Index row = 0; row < joint_positions.rows(); ++row) {
        position = joint_positions.row(row).transpose();
        Eigen::MatrixXd mass = mass_matrix(position, parameters);
        ++result.tested;
        if (mass.rows() != joint_positions.cols() || mass.cols() != joint_positions.cols() || !mass.allFinite()) {
            ++result.failed;
            result.passed = false;
            continue;
        }
        // Normalize the symmetry comparison so squared Frobenius norms cannot
        // hide asymmetry through overflow or underflow. The zero matrix keeps
        // its zero eigenvalue and fails the positive threshold below.
        const double scale = mass.cwiseAbs().maxCoeff();
        if (scale != 0.0) mass /= scale;
        if (!mass.isApprox(mass.transpose(), 1e-9)) {
            ++result.failed;
            result.passed = false;
            continue;
        }
        solver.compute(mass, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success || !(solver.eigenvalues() * scale).allFinite()) {
            ++result.failed;
            result.passed = false;
            continue;
        }
        const double value = solver.eigenvalues().minCoeff() * scale;
        result.minimum_eigenvalue = std::min(result.minimum_eigenvalue, value);
        if (value < minimum_eigenvalue) {
            ++result.failed;
            result.passed = false;
        }
    }
    return result;
}

}  // namespace robot_identification
