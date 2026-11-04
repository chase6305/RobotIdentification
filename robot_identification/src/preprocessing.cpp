#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "data_validation.h"
#include "robot_identification/identification.h"

namespace robot_identification {
namespace {

Eigen::MatrixXd SmoothSymmetric(const Eigen::MatrixXd& values, std::size_t window) {
    const Eigen::Index radius =
        static_cast<Eigen::Index>(std::min(window / 2, static_cast<std::size_t>(values.rows() - 1)));
    Eigen::MatrixXd output(values.rows(), values.cols());
    // Eigen's short reductions are faster for the usual small windows.
    if (radius <= 32) {
        // Visit contiguous joint columns in Eigen's default storage order.
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            for (Eigen::Index row = 0; row < values.rows(); ++row) {
                const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
                const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
                const auto window_values = values.col(column).segment(begin, end - begin + 1);
                output(row, column) = window_values.mean();
                if (std::isfinite(output(row, column))) continue;
                // A finite mean can have an overflowing intermediate sum.
                // Normalize only exceptional windows, keeping the common path fast.
                const double scale = window_values.cwiseAbs().maxCoeff();
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
        Eigen::Index rows_until_rebase = 0;
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
            if (rows_until_rebase == 0) {
                // Rebuild periodically to bound roundoff accumulation;
                // the total work remains linear in the number of samples.
                sum = correction = 0.0L;
                for (Eigen::Index index = begin; index <= end; ++index) add(values(index, column));
                rows_until_rebase = rebase_interval;
            } else {
                if (begin != previous_begin) add(-static_cast<long double>(values(previous_begin, column)));
                if (end != previous_end) add(values(end, column));
            }
            output(row, column) = static_cast<double>((sum + correction) / (end - begin + 1));
            previous_begin = begin;
            previous_end = end;
            --rows_until_rebase;
        }
    }
    return output;
}

Eigen::MatrixXd Differentiate(const Eigen::VectorXd& time_seconds, const Eigen::MatrixXd& signal_values) {
    // Internal preprocessing helper: dimensions and timestamps were checked by
    // ValidateRawInputs before smoothing. Keep arithmetic checks below.
    Eigen::MatrixXd derivative(signal_values.rows(), signal_values.cols());
    for (Eigen::Index row = 0; row < signal_values.rows(); ++row) {
        const Eigen::Index first_row = row == 0 ? 0 : row + 1 == signal_values.rows() ? row - 2 : row - 1;
        const Eigen::Index middle_row = first_row + 1;
        const Eigen::Index last_row = first_row + 2;
        const double left_interval_seconds = time_seconds[middle_row] - time_seconds[first_row];
        const double right_interval_seconds = time_seconds[last_row] - time_seconds[middle_row];
        const double interval_span_seconds = left_interval_seconds + right_interval_seconds;
        if (!std::isfinite(interval_span_seconds)) {
            throw std::overflow_error("timestamp spacing is numerically unusable for differentiation");
        }
        // Differentiate the same three-point quadratic through its secant
        // slopes, so a constant position offset cancels before division.
        // At an interior point, start from the slope with the larger weight.
        // Otherwise a weight rounded to one can cancel the base slope and
        // erase the smaller contribution entirely on strongly unequal steps.
        const bool use_right_slope =
            row == last_row || (row != first_row && left_interval_seconds > right_interval_seconds);
        const double blend_interval_seconds = row == first_row  ? -left_interval_seconds
                                              : row == last_row ? right_interval_seconds
                                              : use_right_slope ? -right_interval_seconds
                                                                : left_interval_seconds;
        const double blend_weight = blend_interval_seconds / interval_span_seconds;
        const bool tiny_blend_weight = std::abs(blend_weight) < std::numeric_limits<double>::min();
        for (Eigen::Index column = 0; column < signal_values.cols(); ++column) {
            const double left_slope =
                (signal_values(middle_row, column) - signal_values(first_row, column)) / left_interval_seconds;
            const double right_slope =
                (signal_values(last_row, column) - signal_values(middle_row, column)) / right_interval_seconds;
            const double base_slope = use_right_slope ? right_slope : left_slope;
            derivative(row, column) = base_slope + blend_weight * (right_slope - left_slope);
            if (tiny_blend_weight || !std::isfinite(derivative(row, column))) {
                // Finite final derivatives can have overflowing differences or
                // individual secants, or an underflowed weight whose product
                // is representable. Retry this derivative in extended range.
                const long double wide_left_slope =
                    (static_cast<long double>(signal_values(middle_row, column)) - signal_values(first_row, column)) /
                    left_interval_seconds;
                const long double wide_right_slope =
                    (static_cast<long double>(signal_values(last_row, column)) - signal_values(middle_row, column)) /
                    right_interval_seconds;
                const long double wide_interval_span =
                    static_cast<long double>(left_interval_seconds) + right_interval_seconds;
                const long double wide_weight = static_cast<long double>(blend_interval_seconds) / wide_interval_span;
                const long double wide_base_slope = use_right_slope ? wide_right_slope : wide_left_slope;
                const long double wide_derivative =
                    wide_base_slope + wide_weight * (wide_right_slope - wide_left_slope);
                if (!std::isfinite(wide_derivative) || std::abs(wide_derivative) > std::numeric_limits<double>::max()) {
                    throw std::overflow_error("differentiation produced non-finite values");
                }
                derivative(row, column) = static_cast<double>(wide_derivative);
            }
        }
    }
    return derivative;
}

}  // namespace

Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                    const Eigen::VectorXd& time_seconds,
                                    const Eigen::MatrixXd& position,
                                    const Eigen::MatrixXd& torque,
                                    const PreprocessOptions& options) {
    std::string reason;
    if (!detail::ValidateRawInputs(joint_names, time_seconds, position, torque, &reason)) {
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

}  // namespace robot_identification
