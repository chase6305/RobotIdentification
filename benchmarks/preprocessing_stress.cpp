// Independent Lagrange-weight checks across data/time scales; no timing claim.
#include <robot_identification/identification.h>

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
            // Allow rounding in each differentiated input and propagate the
            // first pass's error through D for acceleration. Relative error in
            // a nearly cancelled acceleration alone is not a useful bound.
            bounds(row, column) = 64.L * std::numeric_limits<double>::epsilon() * magnitude + propagated +
                                  32.L * std::numeric_limits<double>::denorm_min();
        }
    }
    return true;
}
void Compare(const Eigen::MatrixXd& actual,
             const Eigen::MatrixXd& reference,
             const ErrorBound& bounds,
             const char* label) {
    if (!actual.allFinite()) throw std::runtime_error("accepted derivative is nonfinite");
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
    try {
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
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " accepted=" << accepted
                          << " rejected=" << rejected << " extreme_cases=" << extreme_cases
                          << " unequal_spacing_cases=" << unequal_spacing_cases << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << first_case + cases << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED first_case=" << first_case << " cases=" << cases << " accepted=" << accepted
              << " rejected=" << rejected << " extreme_cases=" << extreme_cases
              << " unequal_spacing_cases=" << unequal_spacing_cases << std::endl;
}
