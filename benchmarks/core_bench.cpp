// Synthetic in-process benchmark; no hardware timing guarantee.
#include <robot_identification/identification.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allocation_hooks.h"

template <typename Function>
void Measure(const char* name, std::size_t count, Function function) {
    function();
    std::vector<double> elapsed, cpu;
    elapsed.reserve(5);
    cpu.reserve(5);
    std::size_t total_allocations = 0, total_new = 0;
    double checksum = 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
        allocations = 0;
        new_allocations = 0;
        counting = true;
        auto start = std::chrono::steady_clock::now();
        const auto cpu_start = std::clock();
        for (std::size_t index = 0; index < count; ++index) checksum += function();
        const auto cpu_end = std::clock();
        auto end = std::chrono::steady_clock::now();
        counting = false;
        if (cpu_start == static_cast<std::clock_t>(-1) || cpu_end == static_cast<std::clock_t>(-1))
            throw std::runtime_error("process CPU timer is unavailable");
        elapsed.push_back(std::chrono::duration<double, std::micro>(end - start).count() / count);
        cpu.push_back(1e6 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC / count);
        total_allocations += allocations;
        total_new += new_allocations;
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::sort(cpu.begin(), cpu.end());
    std::cout << std::setprecision(12) << name << " median_us=" << elapsed[2] << " cpu_median_us=" << cpu[2]
              << " malloc_calloc_per_call=" << double(total_allocations) / (5 * count)
              << " new_per_call=" << double(total_new) / (5 * count)
              << " tracked_total_per_call=" << double(total_allocations + total_new) / (5 * count)
              << " checksum=" << checksum << '\n';
}

int main(int argc, char** argv) {
    const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
    if (argc != 1 && !smoke) {
        std::cerr << "Usage: " << argv[0] << " [--smoke]\n";
        return 2;
    }
    const auto iterations = [smoke](std::size_t count) { return smoke ? std::size_t{1} : count; };
    robot_identification::RegressionProblem problem;
    problem.sample_count = 1500;
    problem.dof = 1;
    problem.regressor.resize(1500, 12);
    problem.row_weights = Eigen::VectorXd::Ones(1500);
    Eigen::VectorXd truth = Eigen::VectorXd::LinSpaced(12, -1.0, 1.0);
    for (Eigen::Index row = 0; row < problem.regressor.rows(); ++row) {
        for (Eigen::Index col = 0; col < problem.regressor.cols(); ++col) {
            problem.regressor(row, col) =
                std::sin((row + 1) * (col + 1) * 0.017) + 0.2 * std::cos((row + 1) * (col + 3) * 0.031);
        }
    }
    problem.observation = problem.regressor * truth;
    for (Eigen::Index row = 0; row < 1500; ++row) {
        problem.observation[row] += 0.001 * std::sin(row * 0.77) + (row % 17 == 0 ? 100.0 : 0.0);
    }
    for (double ridge : {0.0, 0.01}) {
        robot_identification::SolverOptions options;
        options.huber_delta = 0.01;
        options.max_iterations = 20;
        options.ridge = ridge;
        Measure(ridge == 0 ? "irls" : "irls_ridge", iterations(10), [&] {
            auto result = robot_identification::Solve(problem, options);
            return result.parameters.sum() + result.iterations;
        });
    }
    Measure("validation_metrics", iterations(1000), [&] {
        const auto metrics = robot_identification::ValidateParameters(problem, truth);
        return metrics.rmse + metrics.mean_absolute_error + metrics.max_absolute_error + metrics.normalized_rmse;
    });
    for (const int columns : {1, 12}) {
        robot_identification::RegressionProblem single;
        single.sample_count = single.dof = 1;
        single.regressor = Eigen::MatrixXd::Constant(1, columns, .125);
        single.row_weights = Eigen::VectorXd::Ones(1);
        const double magnitude = .125 * columns;
        single.observation = Eigen::VectorXd::Constant(1, 2. * magnitude);
        const Eigen::VectorXd parameters = Eigen::VectorXd::Ones(columns);
        Measure(columns == 1 ? "scalar_validation_metrics" : "single_row_validation_metrics", iterations(1000000), [&] {
            const auto metrics = robot_identification::ValidateParameters(single, parameters);
            if (metrics.row_count != 1 || metrics.rmse != magnitude || metrics.mean_absolute_error != magnitude ||
                metrics.max_absolute_error != magnitude || metrics.normalized_rmse != .5)
                throw std::runtime_error("scalar metric benchmark differs from its analytic fixture");
            return metrics.rmse + metrics.mean_absolute_error + metrics.max_absolute_error + metrics.normalized_rmse;
        });
    }
    Eigen::MatrixXd sine(7, 20), cosine(7, 20);
    for (Eigen::Index row = 0; row < 7; ++row)
        for (Eigen::Index col = 0; col < 20; ++col) {
            sine(row, col) = 0.001 * std::sin(row + col);
            cosine(row, col) = 0.001 * std::cos(row - col);
        }
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(7), sine, cosine, 0.1);
    double time = 0;
    Measure("harmonic_evaluate", iterations(50000), [&] {
        time += 0.0001;
        auto sample = trajectory.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
    Measure("harmonic_bounds", iterations(10000), [&] {
        const auto bounds = trajectory.ConservativeBounds();
        return bounds.maximum_position.sum() + bounds.maximum_absolute_jerk.sum();
    });
    Eigen::MatrixXd sparse_sine = Eigen::MatrixXd::Zero(7, 20), sparse_cosine = sparse_sine;
    sparse_sine.col(0) = sine.col(0);
    sparse_cosine.col(0) = cosine.col(0);
    const robot_identification::HarmonicTrajectory sparse(Eigen::VectorXd::Zero(7), sparse_sine, sparse_cosine, .1);
    time = 0.;
    Measure("harmonic_sparse_evaluate", iterations(50000), [&] {
        time += .0001;
        const auto sample = sparse.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
    const robot_identification::HarmonicTrajectory constant(
        Eigen::VectorXd::Constant(7, .2), Eigen::MatrixXd::Zero(7, 20), Eigen::MatrixXd::Zero(7, 20), .1);
    time = 0.;
    Measure("harmonic_constant_evaluate", iterations(50000), [&] {
        time += .0001;
        const auto sample = constant.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
}
