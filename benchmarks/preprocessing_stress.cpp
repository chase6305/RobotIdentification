// Independent Lagrange-weight checks across data/time scales; no timing claim.
#include <robot_identification/identification.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "stress_duration.h"

namespace {
bool Reference(const Eigen::VectorXd& time, const Eigen::MatrixXd& values, Eigen::MatrixXd& output) {
    output.resize(values.rows(), values.cols());
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
            long double value = 0;
            for (int term = 0; term < 3; ++term) value += weights[term] * values(first + term, column);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) return false;
            output(row, column) = static_cast<double>(value);
        }
    }
    return true;
}
void Compare(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& reference) {
    if (!actual.allFinite()) throw std::runtime_error("accepted derivative is nonfinite");
    for (Eigen::Index column = 0; column < actual.cols(); ++column) {
        const long double scale = reference.col(column).cwiseAbs().maxCoeff();
        const long double tolerance = 1e-10L * scale + 32.L * std::numeric_limits<double>::denorm_min();
        for (Eigen::Index row = 0; row < actual.rows(); ++row) {
            if (std::abs(static_cast<long double>(actual(row, column)) - reference(row, column)) > tolerance)
                throw std::runtime_error("derivative disagrees with extended-range Lagrange reference");
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const double seconds = synthetic_validation::DurationSeconds(argc, argv);
    if (std::numeric_limits<long double>::max_exponent <= 2100) {
        std::cerr << "This validation oracle requires extended-range long double.\n";
        return 2;
    }
    std::mt19937_64 random(0x20261028d1ffeULL);
    std::uniform_real_distribution<double> unit(0., 1.);
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, accepted = 0, rejected = 0, extreme_cases = 0;
    double next_progress = 0;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            robot_identification::RawDataset raw;
            if (cases % 32 == 0) {
                raw.joint_names = {"extreme"};
                raw.time_seconds = Eigen::VectorXd::LinSpaced(7, 0., 9.);
                raw.position.resize(7, 1);
                const double maximum = std::numeric_limits<double>::max();
                raw.position << 0., 0., maximum, -maximum, maximum, 0., 0.;
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
            const bool representable = Reference(raw.time_seconds, raw.position, reference_velocity) &&
                                       Reference(raw.time_seconds, reference_velocity, reference_acceleration);
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
                Compare(velocity, reference_velocity);
                Compare(acceleration, reference_acceleration);
                ++accepted;
            }
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " accepted=" << accepted
                          << " rejected=" << rejected << " extreme_cases=" << extreme_cases << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << cases << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED cases=" << cases << " accepted=" << accepted << " rejected=" << rejected
              << " extreme_cases=" << extreme_cases << std::endl;
}
