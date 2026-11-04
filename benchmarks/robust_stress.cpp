// Independent Huber-location score roots and one-column weighted solve checks.
#include <robot_identification/identification.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "stress_duration.h"

int main(int argc, char** argv) {
    const double seconds = synthetic_validation::DurationSeconds(argc, argv);
    if (std::numeric_limits<long double>::max_exponent <= 2200) {
        std::cerr << "This validation oracle requires extended-range long double.\n";
        return 2;
    }
    std::mt19937_64 random(0x20261029b0b057ULL);
    std::uniform_real_distribution<double> unit(-1., 1.);
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, limited_cases = 0, unconverged_cases = 0, ridge_cases = 0;
    double next_progress = 0, maximum_parameter_error = 0, maximum_covariance_error = 0;
    robot_identification::RegressionProblem problem;
    problem.dof = 1;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            const int rows = 5 + 2 * (random() % 50);
            const double scale = std::ldexp(1., static_cast<int>(random() % 901) - 450);
            const double common_weight = std::ldexp(1., static_cast<int>(random() % 2001) - 1000);
            problem.sample_count = rows;
            problem.regressor = Eigen::MatrixXd::Constant(rows, 1, scale);
            problem.observation.resize(rows);
            problem.row_weights.resize(rows);
            for (int row = 0; row < rows; ++row) {
                double value = 2. + .4 * unit(random);
                if (row % 5 == 0) value += std::copysign(10. + 40. * std::abs(unit(random)), unit(random));
                problem.observation[row] = scale * value;
                problem.row_weights[row] = common_weight * (1. + .25 * unit(random));
            }
            robot_identification::SolverOptions options;
            options.huber_delta = scale * .25;
            options.max_iterations = cases % 16 == 0 ? 1 : cases % 16 == 1 ? 2 : 200;
            options.convergence_tolerance = 1e-12;
            // Anchor ridge to the measured information scale. Use a normal
            // ridge value; keep the remaining cases unregularized.
            const long double proposed_ridge = .3L * common_weight * scale * scale;
            if (cases % 3 == 0 && proposed_ridge >= std::numeric_limits<double>::min() &&
                proposed_ridge <= std::numeric_limits<double>::max()) {
                options.ridge = static_cast<double>(proposed_ridge);
                ++ridge_cases;
            }
            const long double ridge_weight = static_cast<long double>(options.ridge) / scale / scale;
            const long double normalized_ridge = ridge_weight / common_weight;
            const auto actual = robot_identification::Solve(problem, options);
            if (actual.rank != 1 || actual.condition_number != 1. || !actual.final_row_weights.allFinite() ||
                (actual.final_row_weights.array() <= 0.).any())
                throw std::runtime_error("robust result has invalid rank, condition or weights");

            // The returned parameters must solve the exact reported weight set,
            // including when IRLS stopped before the Huber score reached zero.
            long double weight_sum = 0, weighted_observation = 0, weighted_squared_residual = 0;
            for (int row = 0; row < rows; ++row) {
                const long double weight = actual.final_row_weights[row];
                weight_sum += weight;
                weighted_observation += weight * (static_cast<long double>(problem.observation[row]) / scale);
                const long double residual = static_cast<long double>(actual.residual[row]) / scale;
                weighted_squared_residual += weight * residual * residual;
            }
            const long double fitted = weighted_observation / (weight_sum + ridge_weight);
            if (std::abs(actual.parameters[0] - fitted) > 1e-11L * std::max(1.L, std::abs(fitted)))
                throw std::runtime_error("returned parameters do not solve their reported weights");
            const long double covariance = weighted_squared_residual / ((rows - 1) * (weight_sum + ridge_weight));
            const double covariance_error =
                static_cast<double>(std::abs(actual.covariance(0, 0) - covariance) / covariance);
            if (!std::isfinite(covariance_error) || covariance_error > 5e-12)
                throw std::runtime_error("robust covariance differs from the scalar weighted oracle");
            maximum_covariance_error = std::max(maximum_covariance_error, covariance_error);

            long double original_weight_sum = 0, original_weighted_observation = 0;
            for (int row = 0; row < rows; ++row) {
                const long double weight = static_cast<long double>(problem.row_weights[row]) / common_weight;
                original_weight_sum += weight;
                original_weighted_observation += weight * (static_cast<long double>(problem.observation[row]) / scale);
            }
            const auto objective = [&](long double parameter) {
                long double loss = 0;
                const long double delta = static_cast<long double>(options.huber_delta) / scale;
                for (int row = 0; row < rows; ++row) {
                    const long double residual =
                        std::abs(static_cast<long double>(problem.observation[row]) / scale - parameter);
                    const long double value =
                        residual <= delta ? .5L * residual * residual : delta * (residual - .5L * delta);
                    loss += (static_cast<long double>(problem.row_weights[row]) / common_weight) * value;
                }
                return loss + .5L * normalized_ridge * parameter * parameter;
            };
            const long double initial_loss =
                objective(original_weighted_observation / (original_weight_sum + normalized_ridge));
            if (objective(actual.parameters[0]) >
                initial_loss + 256.L * std::numeric_limits<double>::epsilon() * std::max(1.L, initial_loss))
                throw std::runtime_error("IRLS increased the Huber loss above its initial weighted fit");
            if (!actual.converged && actual.iterations != options.max_iterations)
                throw std::runtime_error("unconverged result stopped before consuming its iteration budget");

            if (options.max_iterations == 200 && actual.converged) {
                long double lower = std::min(0.L, static_cast<long double>(problem.observation.minCoeff()) / scale);
                long double upper = std::max(0.L, static_cast<long double>(problem.observation.maxCoeff()) / scale);
                const long double delta = static_cast<long double>(options.huber_delta) / scale;
                for (int iteration = 0; iteration < 256; ++iteration) {
                    const long double middle = .5L * lower + .5L * upper;
                    if (middle == lower || middle == upper) break;
                    long double score = -normalized_ridge * middle;
                    for (int row = 0; row < rows; ++row) {
                        const long double residual =
                            static_cast<long double>(problem.observation[row]) / scale - middle;
                        score += (static_cast<long double>(problem.row_weights[row]) / common_weight) *
                                 std::clamp(residual, -delta, delta);
                    }
                    if (score > 0)
                        lower = middle;
                    else
                        upper = middle;
                }
                const long double reference = .5L * lower + .5L * upper;
                const double error = static_cast<double>(std::abs(actual.parameters[0] - reference));
                if (!std::isfinite(error) || error > 1e-9)
                    throw std::runtime_error("IRLS differs from the independent Huber score root");
                maximum_parameter_error = std::max(maximum_parameter_error, error);
            } else if (options.max_iterations == 200) {
                ++unconverged_cases;
            } else {
                ++limited_cases;
            }
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " limited_cases=" << limited_cases
                          << " unconverged_cases=" << unconverged_cases << " ridge_cases=" << ridge_cases
                          << " max_parameter_error=" << maximum_parameter_error
                          << " max_covariance_error=" << maximum_covariance_error << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << cases << ": " << error.what() << '\n'
                  << std::setprecision(17) << "scale=" << problem.regressor(0, 0)
                  << "\nobservation=" << problem.observation.transpose()
                  << "\nweights=" << problem.row_weights.transpose() << std::endl;
        return 1;
    }
    std::cout << "PASSED cases=" << cases << " limited_cases=" << limited_cases
              << " unconverged_cases=" << unconverged_cases << " ridge_cases=" << ridge_cases
              << " max_parameter_error=" << maximum_parameter_error
              << " max_covariance_error=" << maximum_covariance_error << std::endl;
}
