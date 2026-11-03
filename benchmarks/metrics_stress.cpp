// Independent extended-range metric checks, including subnormal RMS ratios.
#include <robot_identification/identification.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "stress_duration.h"

namespace {
void Compare(double actual, long double reference, const char* label) {
    const double rounded = static_cast<double>(reference);
    if (std::isinf(rounded)) {
        if (actual == rounded) return;
    } else if (std::isfinite(actual)) {
        const long double tolerance = 64.L * std::numeric_limits<double>::epsilon() * std::abs(reference) +
                                      2.L * std::numeric_limits<double>::denorm_min();
        if (std::abs(static_cast<long double>(actual) - reference) <= tolerance) return;
    }
    std::cerr << std::setprecision(21) << label << " actual=" << actual << " reference=" << reference << '\n';
    throw std::runtime_error("metric differs from the independent extended-range norm");
}
}  // namespace

int main(int argc, char** argv) {
    const auto [seconds, first_case] = synthetic_validation::ReplayCampaign(argc, argv);
    if (std::numeric_limits<long double>::max_exponent <= 2200) {
        std::cerr << "This validation oracle requires extended-range long double.\n";
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, rejected = 0, subnormal_cases = 0;
    double next_progress = 0;
    robot_identification::RegressionProblem problem;
    problem.dof = 1;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            const auto index = first_case + cases;
            // Each case owns its seed, so replay needs no prefix RNG work.
            std::mt19937_64 random(0x20261029ee770ULL ^ (index * 0x9e3779b97f4a7c15ULL));
            std::uniform_real_distribution<double> unit(-1., 1.);
            const int rows = 1 + random() % 257;
            problem.sample_count = rows;
            problem.regressor = Eigen::MatrixXd::Zero(rows, 1);
            problem.observation = Eigen::VectorXd::Zero(rows);
            problem.row_weights = Eigen::VectorXd::Ones(rows);
            const bool subnormal = index % 4 == 0;
            const int exponent = static_cast<int>(random() % 2098) - 1074;
            for (int row = 0; row < rows; ++row) {
                if (subnormal) {
                    if (row == 0) {
                        const double quantum = std::ldexp(1., -1074 + static_cast<int>(random() % 53));
                        problem.observation[row] = (1 + random() % 17) * quantum;
                        problem.regressor(row, 0) = (static_cast<int>(random() % 11) - 5) * quantum;
                    }
                } else if (index % 4 != 1) {
                    problem.observation[row] = std::ldexp(unit(random), exponent);
                    problem.regressor(row, 0) =
                        std::ldexp(unit(random), index % 8 == 2 ? static_cast<int>(random() % 2098) - 1074 : exponent);
                } else {
                    problem.regressor(row, 0) = std::ldexp(unit(random), exponent);
                }
                if (index % 16 == 3) problem.regressor(row, 0) = problem.observation[row];
            }
            if (index % 64 == 5) {
                problem.observation[0] = std::numeric_limits<double>::max();
                problem.regressor(0, 0) = -std::numeric_limits<double>::max();
            }
            long double squared_error = 0, squared_observation = 0, absolute_error = 0, maximum = 0;
            bool finite = true;
            for (int row = 0; row < rows; ++row) {
                const double error = problem.observation[row] - problem.regressor(row, 0);
                if (!std::isfinite(error)) {
                    finite = false;
                    break;
                }
                squared_error += static_cast<long double>(error) * error;
                squared_observation += static_cast<long double>(problem.observation[row]) * problem.observation[row];
                absolute_error += std::abs(static_cast<long double>(error));
                maximum = std::max(maximum, std::abs(static_cast<long double>(error)));
            }
            robot_identification::ValidationMetrics actual;
            bool failed = false;
            try {
                actual = robot_identification::ValidateParameters(problem, Eigen::VectorXd::Ones(1));
            } catch (const std::overflow_error&) {
                failed = true;
            }
            if (failed == finite) throw std::runtime_error("metric accepted/rejected the wrong residual range");
            if (failed) {
                ++rejected;
            } else {
                const long double normalized =
                    squared_observation == 0 ? (squared_error == 0 ? 0.L : std::numeric_limits<long double>::infinity())
                                             : std::sqrt(squared_error / squared_observation);
                Compare(actual.rmse, std::sqrt(squared_error / rows), "rmse");
                Compare(actual.mean_absolute_error, absolute_error / rows, "mean_absolute_error");
                Compare(actual.max_absolute_error, maximum, "max_absolute_error");
                Compare(actual.normalized_rmse, normalized, "normalized_rmse");
                if (actual.row_count != static_cast<std::size_t>(rows)) throw std::runtime_error("row count changed");
                if (subnormal) ++subnormal_cases;
            }
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " rejected=" << rejected
                          << " subnormal_cases=" << subnormal_cases << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << first_case + cases << ": " << error.what() << '\n'
                  << std::setprecision(17) << "observation=" << problem.observation.transpose()
                  << "\nprediction=" << problem.regressor.transpose() << std::endl;
        return 1;
    }
    std::cout << "PASSED first_case=" << first_case << " cases=" << cases << " rejected=" << rejected
              << " subnormal_cases=" << subnormal_cases << std::endl;
}
