// Synthetic in-process benchmark; no hardware timing guarantee.
#include <robot_identification/identification.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "allocation_hooks.h"

template <typename Function>
void Measure(const char* name, std::size_t count, Function function) {
    function();
    std::vector<double> elapsed;
    std::size_t total_allocations = 0, total_new = 0;
    double checksum = 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
        allocations = 0;
        new_allocations = 0;
        counting = true;
        auto start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < count; ++index) checksum += function();
        auto end = std::chrono::steady_clock::now();
        counting = false;
        elapsed.push_back(std::chrono::duration<double, std::micro>(end - start).count() / count);
        total_allocations += allocations;
        total_new += new_allocations;
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::cout << std::setprecision(12) << name << " median_us=" << elapsed[2]
              << " malloc_calloc_per_call=" << double(total_allocations) / (5 * count)
              << " new_per_call=" << double(total_new) / (5 * count)
              << " tracked_total_per_call=" << double(total_allocations + total_new) / (5 * count)
              << " checksum=" << checksum << '\n';
}

int main() {
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
        Measure(ridge == 0 ? "irls" : "irls_ridge", 10, [&] {
            auto result = robot_identification::Solve(problem, options);
            return result.parameters.sum() + result.iterations;
        });
    }
    Measure("validation_metrics", 1000, [&] {
        const auto metrics = robot_identification::ValidateParameters(problem, truth);
        return metrics.rmse + metrics.mean_absolute_error + metrics.max_absolute_error + metrics.normalized_rmse;
    });
    Eigen::MatrixXd sine(7, 20), cosine(7, 20);
    for (Eigen::Index row = 0; row < 7; ++row)
        for (Eigen::Index col = 0; col < 20; ++col) {
            sine(row, col) = 0.001 * std::sin(row + col);
            cosine(row, col) = 0.001 * std::cos(row - col);
        }
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(7), sine, cosine, 0.1);
    double time = 0;
    Measure("harmonic_evaluate", 50000, [&] {
        time += 0.0001;
        auto sample = trajectory.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
    Measure("harmonic_bounds", 10000, [&] {
        const auto bounds = trajectory.ConservativeBounds();
        return bounds.maximum_position.sum() + bounds.maximum_absolute_jerk.sum();
    });
    Eigen::MatrixXd sparse_sine = Eigen::MatrixXd::Zero(7, 20), sparse_cosine = sparse_sine;
    sparse_sine.col(0) = sine.col(0);
    sparse_cosine.col(0) = cosine.col(0);
    const robot_identification::HarmonicTrajectory sparse(Eigen::VectorXd::Zero(7), sparse_sine, sparse_cosine, .1);
    time = 0.;
    Measure("harmonic_sparse_evaluate", 50000, [&] {
        time += .0001;
        const auto sample = sparse.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
    const robot_identification::HarmonicTrajectory constant(
        Eigen::VectorXd::Constant(7, .2), Eigen::MatrixXd::Zero(7, 20), Eigen::MatrixXd::Zero(7, 20), .1);
    time = 0.;
    Measure("harmonic_constant_evaluate", 50000, [&] {
        time += .0001;
        const auto sample = constant.Evaluate(time);
        return sample.position.sum() + sample.jerk.sum();
    });
}
