// Independent window-sum and Lagrange-weight checks; no timing claim.
#include <robot_identification/identification.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

#include "stress_duration.h"

namespace {
using ErrorBound = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
bool Reference(const Eigen::VectorXd& time,
               const Eigen::MatrixXd& values,
               Eigen::MatrixXd& output,
               ErrorBound& bounds,
               const ErrorBound* input_bounds = nullptr) {
    output.resize(values.rows(), values.cols());
    bounds.resize(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const Eigen::Index first = row == 0 ? 0 : row + 1 == values.rows() ? row - 2 : row - 1;
        std::array<long double, 3> weights;
        for (int term = 0; term < 3; ++term) {
            const auto index = first + term;
            const auto a = first + (term + 1) % 3, b = first + (term + 2) % 3;
            weights[term] = (2.L * time[row] - time[a] - time[b]) / ((static_cast<long double>(time[index]) - time[a]) *
                                                                     (static_cast<long double>(time[index]) - time[b]));
        }
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            long double value = 0, magnitude = 0, propagated = 0;
            for (int term = 0; term < 3; ++term) {
                const long double contribution = weights[term] * values(first + term, column);
                value += contribution;
                magnitude += std::abs(contribution);
                if (input_bounds) propagated += std::abs(weights[term]) * (*input_bounds)(first + term, column);
            }
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) return false;
            output(row, column) = static_cast<double>(value);
            // Secant evaluation can cancel two large weighted slopes even
            // when the equivalent middle Lagrange contribution is small.
            const long double left_change = static_cast<long double>(values(first + 1, column)) - values(first, column);
            const long double right_change =
                static_cast<long double>(values(first + 2, column)) - values(first + 1, column);
            const long double weighted_secant_magnitude =
                std::abs(weights[0] * left_change) + std::abs(weights[2] * right_change);
            // Preserve small weights on highly unequal intervals, and propagate
            // earlier-stage error through the same linear derivative operator.
            bounds(row, column) =
                64.L * std::numeric_limits<double>::epsilon() * std::max(magnitude, weighted_secant_magnitude) +
                propagated + 32.L * std::numeric_limits<double>::denorm_min();
        }
    }
    return true;
}
void Compare(const Eigen::MatrixXd& actual,
             const Eigen::MatrixXd& reference,
             const ErrorBound& bounds,
             const char* label) {
    if (!actual.allFinite()) throw std::runtime_error("accepted output is nonfinite");
    for (Eigen::Index column = 0; column < actual.cols(); ++column) {
        const long double scale = reference.col(column).cwiseAbs().maxCoeff();
        for (Eigen::Index row = 0; row < actual.rows(); ++row) {
            const long double tolerance = bounds(row, column);
            if (std::abs(static_cast<long double>(actual(row, column)) - reference(row, column)) > tolerance) {
                std::ostringstream message;
                message << std::setprecision(20) << label << " row=" << row << " column=" << column
                        << " actual=" << actual(row, column) << " reference=" << reference(row, column)
                        << " scale=" << scale << " tolerance=" << tolerance;
                throw std::runtime_error(message.str());
            }
        }
    }
}

// Deliberately recompute every window in extended precision. This oracle does
// not use the production running sum, rebasing, or Eigen's reduction order.
void SmoothReference(const Eigen::MatrixXd& values,
                     std::size_t window,
                     Eigen::MatrixXd& output,
                     ErrorBound& bounds,
                     const ErrorBound* input_bounds = nullptr) {
    output.resize(values.rows(), values.cols());
    bounds.resize(values.rows(), values.cols());
    const Eigen::Index radius =
        static_cast<Eigen::Index>(std::min(window / 2, static_cast<std::size_t>(values.rows() - 1)));
    for (Eigen::Index column = 0; column < values.cols(); ++column) {
        long double column_maximum = 0;
        for (Eigen::Index row = 0; row < values.rows(); ++row)
            column_maximum = std::max(column_maximum, std::abs(static_cast<long double>(values(row, column))));
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = std::min<Eigen::Index>(values.rows(), row + radius + 1);
            long double sum = 0, propagated = 0;
            for (Eigen::Index term = begin; term < end; ++term) {
                sum += static_cast<long double>(values(term, column));
                if (input_bounds) propagated += (*input_bounds)(term, column);
            }
            const long double count = end - begin;
            output(row, column) = static_cast<double>(sum / count);
            // A conservative absolute forward-error bound also covers a
            // rolling reduction whose cancelled terms exceed this window's
            // current sum. Propagate earlier-stage uncertainty separately.
            bounds(row, column) = propagated / count;
            if (window != 1) {
                bounds(row, column) +=
                    8.L * (values.rows() + 1) * std::numeric_limits<double>::epsilon() * column_maximum +
                    32.L * std::numeric_limits<double>::denorm_min();
            }
        }
    }
}

struct SmoothingCoverage {
    unsigned long long cases = 0, extreme_cases = 0, trimmed_cases = 0, long_window_cases = 0;
    std::array<unsigned long long, 7> windows{};
    std::array<unsigned long long, 3> dofs{};
};

void ValidateSmoothedCase(unsigned long long index, SmoothingCoverage& coverage) {
    constexpr std::array<std::size_t, 7> windows{1, 3, 5, 31, 65, 67, 129};
    constexpr std::array<int, 3> dofs{1, 7, 14};
    // Each extra case has its own seed. It neither consumes the legacy RNG
    // nor changes which original window-one case FIRST_CASE replays.
    std::mt19937_64 random(0x20261103a6e7ULL ^ (index * 0x9e3779b97f4a7c15ULL));
    std::uniform_real_distribution<double> unit(0., 1.);
    const auto window_index = index % windows.size();
    const auto dof_index = (index / windows.size()) % dofs.size();
    const auto window = windows[window_index];
    const auto torque_window = windows[(index / 3) % windows.size()];
    const int dof = dofs[dof_index];
    const int rows = index % 11 == 0 ? 3 : 3 + random() % 203;
    const bool extreme = index % 13 == 0;
    const double step = extreme ? 1e100 : std::ldexp(1., static_cast<int>(random() % 321) - 160);
    const double scale =
        extreme ? std::numeric_limits<double>::max() : std::ldexp(1., static_cast<int>(random() % 1001) - 500);
    robot_identification::RawDataset raw;
    for (int joint = 0; joint < dof; ++joint) raw.joint_names.push_back("axis_" + std::to_string(joint));
    raw.time_seconds.resize(rows);
    raw.position.resize(rows, dof);
    raw.torque.resize(rows, dof);
    raw.time_seconds[0] = 0.;
    for (int row = 0; row < rows; ++row) {
        if (row > 0) raw.time_seconds[row] = raw.time_seconds[row - 1] + step * (.25 + 4. * unit(random));
        for (int joint = 0; joint < dof; ++joint) {
            double shape = .4 * std::sin(.17 * row + .3 * joint) + .1 * unit(random);
            if (extreme) {
                switch ((index / 13) % 4) {
                    case 0:
                        shape = 1.;
                        break;
                    case 1:
                        shape = row % 2 == 0 ? 1. : -1.;
                        break;
                    case 2:
                        shape = row % 3 == 0 ? -1. : .75;
                        break;
                    default:
                        shape = row == rows / 2 ? 1. : 0.;
                        break;
                }
            }
            raw.position(row, joint) = scale * shape;
            raw.torque(row, joint) =
                scale * (extreme ? (row % 3 == 0 ? -.5 : 1.) : .7 * std::cos(.11 * row + .2 * joint));
        }
    }
    const std::size_t trim_front = index % 17 == 0 ? rows - 1 : index % 3;
    const std::size_t trim_back = index % 17 == 0 ? 0 : std::min<std::size_t>(index % 2, rows - trim_front - 1);
    const Eigen::Index retained_rows = rows - trim_front - trim_back;
    Eigen::MatrixXd position, velocity_before_smoothing, velocity, acceleration_before_smoothing, acceleration, torque;
    ErrorBound position_bound, velocity_before_bound, velocity_bound, acceleration_before_bound, acceleration_bound,
        torque_bound;
    SmoothReference(raw.position, window, position, position_bound);
    if (!Reference(raw.time_seconds, position, velocity_before_smoothing, velocity_before_bound, &position_bound))
        throw std::runtime_error("smoothed velocity oracle exceeded its intended finite range");
    SmoothReference(velocity_before_smoothing, window, velocity, velocity_bound, &velocity_before_bound);
    if (!Reference(raw.time_seconds, velocity, acceleration_before_smoothing, acceleration_before_bound,
                   &velocity_bound))
        throw std::runtime_error("smoothed acceleration oracle exceeded its intended finite range");
    SmoothReference(acceleration_before_smoothing, window, acceleration, acceleration_bound,
                    &acceleration_before_bound);
    SmoothReference(raw.torque, torque_window, torque, torque_bound);
    try {
        const auto result =
            robot_identification::PreprocessPositionAndTorque(raw, {window, trim_front, trim_back, torque_window});
        if (result.joint_names != raw.joint_names || result.samples.size() != static_cast<std::size_t>(retained_rows))
            throw std::runtime_error("smoothing changed joint names or retained sample count");
        Eigen::MatrixXd actual_position(retained_rows, dof), actual_velocity(retained_rows, dof),
            actual_acceleration(retained_rows, dof), actual_torque(retained_rows, dof);
        for (Eigen::Index row = 0; row < retained_rows; ++row) {
            const auto& sample = result.samples[static_cast<std::size_t>(row)];
            if (sample.time_seconds != raw.time_seconds[row + trim_front])
                throw std::runtime_error("smoothing or trimming changed a retained timestamp");
            actual_position.row(row) = sample.position.transpose();
            actual_velocity.row(row) = sample.velocity.transpose();
            actual_acceleration.row(row) = sample.acceleration.transpose();
            actual_torque.row(row) = sample.torque.transpose();
        }
        Compare(actual_position, position.middleRows(trim_front, retained_rows),
                position_bound.middleRows(trim_front, retained_rows), "smoothed_position");
        Compare(actual_velocity, velocity.middleRows(trim_front, retained_rows),
                velocity_bound.middleRows(trim_front, retained_rows), "smoothed_velocity");
        Compare(actual_acceleration, acceleration.middleRows(trim_front, retained_rows),
                acceleration_bound.middleRows(trim_front, retained_rows), "smoothed_acceleration");
        Compare(actual_torque, torque.middleRows(trim_front, retained_rows),
                torque_bound.middleRows(trim_front, retained_rows), "smoothed_torque");
    } catch (const std::exception& error) {
        std::ostringstream message;
        message << "smoothing case=" << index << " window=" << window << " torque_window=" << torque_window
                << " rows=" << rows << " dof=" << dof << " trim=" << trim_front << ',' << trim_back
                << " extreme=" << extreme << ": " << error.what();
        throw std::runtime_error(message.str());
    }
    ++coverage.cases;
    ++coverage.windows[window_index];
    ++coverage.dofs[dof_index];
    if (extreme) ++coverage.extreme_cases;
    if (trim_front || trim_back) ++coverage.trimmed_cases;
    if (window > 65 && rows > 33) ++coverage.long_window_cases;
}

void PrintSmoothingCoverage(const SmoothingCoverage& coverage) {
    std::cout << " smoothing_cases=" << coverage.cases << " smoothing_extreme_cases=" << coverage.extreme_cases
              << " smoothing_trimmed_cases=" << coverage.trimmed_cases
              << " smoothing_long_window_cases=" << coverage.long_window_cases << " smoothing_windows=";
    for (std::size_t index = 0; index < coverage.windows.size(); ++index)
        std::cout << (index == 0 ? "" : ",") << coverage.windows[index];
    std::cout << " smoothing_dofs=";
    for (std::size_t index = 0; index < coverage.dofs.size(); ++index)
        std::cout << (index == 0 ? "" : ",") << coverage.dofs[index];
}
}  // namespace

int main(int argc, char** argv) {
    const auto [seconds, first_case] = synthetic_validation::ReplayCampaign(argc, argv, 100000000);
    if (std::numeric_limits<long double>::max_exponent <= 2100) {
        std::cerr << "This validation oracle requires extended-range long double.\n";
        return 2;
    }
    std::mt19937_64 random(0x20261028d1ffeULL);
    std::uniform_real_distribution<double> unit(0., 1.);
    for (unsigned long long index = 0; index < first_case; ++index) {
        if (index % 32 == 0) continue;
        const int rows = 3 + random() % 98, dof = 1 + random() % 7;
        (void)random();
        (void)random();
        for (int draw = 0, count = rows - 1 + rows * dof; draw < count; ++draw) (void)unit(random);
    }
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, accepted = 0, rejected = 0, extreme_cases = 0, unequal_spacing_cases = 0;
    double next_progress = 0;
    SmoothingCoverage smoothing_coverage;
    try {
        // Always recheck the impulse that exposed the weighted-secant error scale.
        // Keep this fixed fixture separate from replay RNG and campaign counters.
        SmoothingCoverage fixed_coverage;
        ValidateSmoothedCase(231959, fixed_coverage);
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            robot_identification::RawDataset raw;
            if ((first_case + cases) % 32 == 0) {
                raw.joint_names = {"extreme"};
                const auto variant = ((first_case + cases) / 32) % 9;
                if (variant == 0) {
                    raw.time_seconds = Eigen::VectorXd::LinSpaced(7, 0., 9.);
                    raw.position.resize(7, 1);
                    const double maximum = std::numeric_limits<double>::max();
                    raw.position << 0., 0., maximum, -maximum, maximum, 0., 0.;
                } else {
                    const std::array<double, 4> short_steps{std::numeric_limits<double>::denorm_min(), 1e-308, 1e-200,
                                                            1e-30};
                    const double short_step = short_steps[(variant - 1) / 2];
                    const bool reverse = variant % 2 == 0;
                    raw.time_seconds =
                        reverse ? Eigen::Vector3d(-2., -short_step, 0.) : Eigen::Vector3d(0., short_step, 2.);
                    raw.position = reverse ? Eigen::Vector3d(2e307, 0., 0.) : Eigen::Vector3d(0., 0., 2e307);
                    ++unequal_spacing_cases;
                }
                ++extreme_cases;
            } else {
                const int rows = 3 + random() % 98, dof = 1 + random() % 7;
                const double step = std::ldexp(1., static_cast<int>(random() % 1001) - 500);
                const double scale = std::ldexp(1., static_cast<int>(random() % 1521) - 500);
                for (int joint = 0; joint < dof; ++joint) raw.joint_names.push_back("axis_" + std::to_string(joint));
                raw.time_seconds.resize(rows);
                raw.time_seconds[0] = 0.;
                raw.position.resize(rows, dof);
                for (int row = 0; row < rows; ++row) {
                    if (row > 0) raw.time_seconds[row] = raw.time_seconds[row - 1] + step * (.8 + .4 * unit(random));
                    for (int joint = 0; joint < dof; ++joint)
                        raw.position(row, joint) = scale * (.4 * std::sin(.17 * row + .3 * joint) + .1 * unit(random));
                }
            }
            raw.torque = Eigen::MatrixXd::Zero(raw.position.rows(), raw.position.cols());
            Eigen::MatrixXd reference_velocity, reference_acceleration;
            ErrorBound velocity_bound, acceleration_bound;
            const bool representable = Reference(raw.time_seconds, raw.position, reference_velocity, velocity_bound) &&
                                       Reference(raw.time_seconds, reference_velocity, reference_acceleration,
                                                 acceleration_bound, &velocity_bound);
            bool failed = false;
            robot_identification::Dataset result;
            try {
                result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
            } catch (const std::overflow_error&) {
                failed = true;
            }
            if (failed == representable) throw std::runtime_error("preprocessing accepted/rejected the wrong range");
            if (failed) {
                ++rejected;
            } else {
                Eigen::MatrixXd velocity(raw.position.rows(), raw.position.cols());
                Eigen::MatrixXd acceleration(raw.position.rows(), raw.position.cols());
                for (Eigen::Index row = 0; row < velocity.rows(); ++row) {
                    velocity.row(row) = result.samples[static_cast<std::size_t>(row)].velocity.transpose();
                    acceleration.row(row) = result.samples[static_cast<std::size_t>(row)].acceleration.transpose();
                }
                try {
                    Compare(velocity, reference_velocity, velocity_bound, "velocity");
                    Compare(acceleration, reference_acceleration, acceleration_bound, "acceleration");
                } catch (const std::exception&) {
                    std::cerr << std::setprecision(17) << "time=" << raw.time_seconds.transpose() << "\nposition=\n"
                              << raw.position << "\nvelocity=\n"
                              << velocity << "\nreference_velocity=\n"
                              << reference_velocity << std::endl;
                    throw;
                }
                ++accepted;
            }
            ValidateSmoothedCase(first_case + cases, smoothing_coverage);
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " accepted=" << accepted
                          << " rejected=" << rejected << " extreme_cases=" << extreme_cases
                          << " unequal_spacing_cases=" << unequal_spacing_cases;
                PrintSmoothingCoverage(smoothing_coverage);
                std::cout << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << first_case + cases << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED first_case=" << first_case << " cases=" << cases << " accepted=" << accepted
              << " rejected=" << rejected << " extreme_cases=" << extreme_cases
              << " unequal_spacing_cases=" << unequal_spacing_cases;
    PrintSmoothingCoverage(smoothing_coverage);
    std::cout << std::endl;
}
