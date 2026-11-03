// Opt-in synthetic correctness campaign; no hardware validation.
#include <robot_identification/identification.h>

#include <Eigen/QR>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "stress_duration.h"

static_assert(std::numeric_limits<long double>::max_exponent > 3 * std::numeric_limits<double>::max_exponent,
              "The independent wide-range oracle requires extended long double range");
int main(int argc, char** argv) {
    const double duration = synthetic_validation::DurationSeconds(argc, argv);
    std::mt19937_64 rng(0x630520261029);
    std::normal_distribution<double> normal;
    const auto started = std::chrono::steady_clock::now();
    double next = 0;
    unsigned long long cases = 0;
    double max_parameter_error = 0, max_covariance_error = 0;
    while (true) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (elapsed >= duration) break;
        const int columns = 1 + rng() % 40, rows = columns + 2 + rng() % 128;
        auto gaussian = [&](int n, int m) {
            Eigen::MatrixXd a(n, m);
            for (int i = 0; i < a.size(); ++i) a.data()[i] = normal(rng);
            return a;
        };
        Eigen::HouseholderQR<Eigen::MatrixXd> left_qr(gaussian(rows, columns + 1));
        Eigen::MatrixXd left = left_qr.householderQ() * Eigen::MatrixXd::Identity(rows, columns + 1);
        Eigen::HouseholderQR<Eigen::MatrixXd> right_qr(gaussian(columns, columns));
        Eigen::MatrixXd right = right_qr.householderQ() * Eigen::MatrixXd::Identity(columns, columns);
        Eigen::VectorXd singular(columns);
        const double condition = std::pow(10., rng() % 9), scale = std::pow(10., int(rng() % 261) - 130);
        for (int i = 0; i < columns; ++i)
            singular[i] = scale / std::pow(condition, columns == 1 ? 0. : double(i) / (columns - 1));
        const bool deficient = cases % 7 == 0 && columns > 1;
        if (deficient) singular[columns - 1] = 0;
        Eigen::VectorXd coefficients = Eigen::VectorXd::NullaryExpr(columns, [&] { return normal(rng); });
        robot_identification::RegressionProblem problem;
        problem.dof = 1;
        problem.sample_count = rows;
        problem.row_weights = Eigen::VectorXd::NullaryExpr(rows, [&] { return std::pow(10., int(rng() % 13) - 6); });
        problem.regressor = left.leftCols(columns) * singular.asDiagonal() * right.transpose();
        problem.observation = problem.regressor * coefficients + left.col(columns) * std::min(1., scale / condition);
        for (int row = 0; row < rows; ++row) {
            const double root = std::sqrt(problem.row_weights[row]);
            problem.regressor.row(row) /= root;
            problem.observation[row] /= root;
        }
        const double common_weight = std::pow(10., int(rng() % 601) - 300);
        problem.row_weights *= common_weight;
        robot_identification::SolverOptions options;
        const auto result = robot_identification::Solve(problem, options);
        Eigen::VectorXd factors(columns), retained = right.transpose() * coefficients;
        for (int i = 0; i < columns; ++i) {
            const double s = singular[i], denominator = s * s + options.ridge;
            if (denominator == 0) {
                factors[i] = 0;
                retained[i] = 0;
            } else {
                factors[i] = 1. / std::sqrt(denominator);
                retained[i] *= s * s / denominator;
            }
        }
        Eigen::VectorXd expected = right * retained;
        const double parameter_error = (result.parameters - expected).norm() / std::max(1., expected.norm());
        if (result.rank != std::size_t(columns - deficient) || !std::isfinite(parameter_error) ||
            parameter_error > 2e-6) {
            Eigen::MatrixXd a = problem.row_weights.array().sqrt().matrix().asDiagonal() * problem.regressor;
            Eigen::VectorXd b = problem.row_weights.array().sqrt().matrix().asDiagonal() * problem.observation;
            Eigen::VectorXd qr = a.colPivHouseholderQr().solve(b);
            std::cerr << "rows=" << rows << " columns=" << columns << " ridge=" << options.ridge << " q_orth="
                      << (left.transpose() * left - Eigen::MatrixXd::Identity(columns + 1, columns + 1)).norm()
                      << " qr_error=" << (qr - expected).norm() / std::max(1., expected.norm())
                      << "\nactual=" << result.parameters.transpose() << "\nexpected=" << expected.transpose() << "\n";
            std::cerr << "parameter mismatch case=" << cases << " error=" << parameter_error << " cond=" << condition
                      << " scale=" << scale << '\n';
            return 1;
        }
        long double error = 0;
        for (int i = 0; i < rows; ++i)
            error += static_cast<long double>(result.residual[i]) * result.residual[i] * problem.row_weights[i];
        const long double variance = error / std::max(1, rows - int(result.rank));
        Eigen::MatrixXd covariance(columns, columns);
        for (int i = 0; i < columns; ++i)
            for (int j = 0; j < columns; ++j) {
                long double value = 0;
                for (int k = 0; k < columns; ++k)
                    if (singular[k] > 0) {
                        const long double denominator =
                            static_cast<long double>(singular[k]) * singular[k] * common_weight;
                        value += variance / denominator * static_cast<long double>(right(i, k)) * right(j, k);
                    }
                covariance(i, j) = static_cast<double>(value);
            }
        const Eigen::MatrixXd difference = result.covariance - covariance;
        const double covariance_error =
            Eigen::Map<const Eigen::VectorXd>(difference.data(), difference.size()).stableNorm() /
            std::max(1e-300, Eigen::Map<const Eigen::VectorXd>(covariance.data(), covariance.size()).stableNorm());
        if (!std::isfinite(covariance_error) || covariance_error > 2e-6) {
            std::cerr << "covariance mismatch case=" << cases << " error=" << covariance_error << " cond=" << condition
                      << " scale=" << scale << '\n';
            return 2;
        }
        max_parameter_error = std::max(max_parameter_error, parameter_error);
        max_covariance_error = std::max(max_covariance_error, covariance_error);
        ++cases;
        if (elapsed >= next) {
            std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " max_parameter_error=" << max_parameter_error
                      << " max_covariance_error=" << max_covariance_error << std::endl;
            next = elapsed + 60;
        }
    }
    std::cout << "PASSED cases=" << cases << " max_parameter_error=" << max_parameter_error
              << " max_covariance_error=" << max_covariance_error << std::endl;
}
