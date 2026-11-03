// Independent known-rank, holdout-map and analytic information campaigns.
#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "stress_duration.h"

namespace {
using namespace robot_identification;
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void CheckBasis(unsigned long long index) {
    std::mt19937_64 random(0x20261029ba515ULL ^ (index * 0x9e3779b97f4a7c15ULL));
    std::uniform_real_distribution<double> unit(-1., 1.);
    const bool zero_only = index % 4 == 0;
    const int rank = 1 + random() % 8, columns = rank + 1 + random() % 6, dof = 1 + random() % 5;
    const int samples = rank + 3, rows = samples * dof;
    Eigen::MatrixXd independent(rows, rank), combination(rank, columns);
    combination.setZero();
    combination.leftCols(rank).setIdentity();
    if (!zero_only)
        for (int column = rank; column < columns; ++column)
            for (int direction = 0; direction < rank; ++direction) combination(direction, column) = unit(random);
    if (index % 7 == 0) combination.col(columns - 1).setZero();
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < rank; ++column)
            independent(row, column) = std::cos(std::acos(-1.) * (row + .5) * (column + 1) / rows);
    Eigen::VectorXd scales(columns), truth(columns), unscaled(columns);
    for (int column = 0; column < columns; ++column) {
        scales[column] = std::ldexp(1., static_cast<int>(random() % 601) - 300);
        unscaled[column] = unit(random);
        truth[column] = unscaled[column] / scales[column];
    }
    if (index % 8 == 0) {
        Eigen::MatrixXd diagonal = Eigen::MatrixXd::Zero(rank, rank);
        Eigen::MatrixXd arbitrary(3, rank);
        for (int column = 0; column < rank; ++column) {
            diagonal(column, column) = scales[column] * (.5 + std::abs(unit(random)));
            for (int row = 0; row < arbitrary.rows(); ++row) arbitrary(row, column) = scales[column] * unit(random);
        }
        const BaseParameterBasis independent_basis(diagonal);
        Require(independent_basis.ReconstructionError(arbitrary) == 0.,
                "full-rank independent columns acquired a spurious reconstruction residual");
        const auto independent_reduced = independent_basis.ReduceRegressor(arbitrary, 0.);
        Require(
            (independent_reduced * independent_basis.ToBaseParameters(truth.head(rank)) - arbitrary * truth.head(rank))
                    .cwiseAbs()
                    .maxCoeff() < 1e-12,
            "zero-tolerance independent mapping changed predictions");
    }
    RegressionProblem problem;
    problem.regressor = independent * combination * scales.asDiagonal();
    problem.observation = independent * (combination * unscaled);
    problem.row_weights.resize(rows);
    for (int row = 0; row < rows; ++row) problem.row_weights[row] = 1. + .5 * unit(random);
    problem.sample_count = samples;
    problem.dof = dof;
    const BaseParameterBasis basis(problem.regressor);
    Require(basis.parameter_count() == static_cast<std::size_t>(rank), "known structural rank changed");
    const auto reduced = basis.ReduceProblem(problem);
    const auto fit = Solve(reduced);
    Require(fit.rank == static_cast<std::size_t>(rank), "reduced fit lost a known direction");
    const auto expected = basis.ToBaseParameters(truth);
    Require((fit.parameters - expected).stableNorm() <= 1e-9 * std::max(1., expected.stableNorm()),
            "known parameter mapping changed under column-unit scaling");
    Require((fit.predicted - problem.observation).cwiseAbs().maxCoeff() < 1e-9,
            "reduced weighted fit changed known analytic observations");
    Eigen::MatrixXd holdout(3, rank);
    for (int row = 0; row < holdout.rows(); ++row)
        for (int column = 0; column < rank; ++column) holdout(row, column) = unit(random);
    Eigen::MatrixXd full = holdout * combination * scales.asDiagonal();
    Require((basis.ReduceRegressor(full) * fit.parameters - holdout * (combination * unscaled)).cwiseAbs().maxCoeff() <
                1e-9,
            "base mapping failed independent holdout states");
    if (index % 7 == 0 || zero_only) {
        full(0, columns - 1) = index % 2 == 0 ? std::numeric_limits<double>::denorm_min() : 1e-150;
        bool rejected = false;
        try {
            (void)basis.ReduceRegressor(full);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(rejected, "new excitation in a zero calibration column was silently discarded");
    }
}

class IndependentActuators final : public RegressorBackend {
public:
    explicit IndependentActuators(int dof) : dof_(dof) {}
    std::size_t parameter_count() const noexcept override { return 2 * dof_; }
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> result;
        for (int joint = 0; joint < dof_; ++joint) {
            result.push_back("inertia/" + std::to_string(joint));
            result.push_back("viscous/" + std::to_string(joint));
        }
        return result;
    }
    Eigen::MatrixXd Compute(const Sample& sample) const override {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dof_, 2 * dof_);
        for (int joint = 0; joint < dof_; ++joint) {
            result(joint, 2 * joint) = sample.acceleration[joint];
            result(joint, 2 * joint + 1) = sample.velocity[joint];
        }
        return result;
    }

private:
    int dof_;
};

bool CheckInformation(unsigned long long index) {
    std::mt19937_64 random(0x20261029d3516ULL ^ (index * 0x9e3779b97f4a7c15ULL));
    std::uniform_real_distribution<double> positive(.5, 1.5);
    const int dof = 1 + random() % 5;
    const IndependentActuators backend(dof);
    HarmonicDesign design;
    design.offset = Eigen::VectorXd::Zero(dof);
    design.sin_coefficients.resize(dof, 1);
    design.cos_coefficients = Eigen::MatrixXd::Zero(dof, 1);
    design.fundamental_frequency_hz = std::ldexp(1., static_cast<int>(random() % 5) - 2) / (2. * std::acos(-1.));
    const double angular = 2. * std::acos(-1.) * design.fundamental_frequency_hz;
    ExcitationDesignOptions options;
    options.duration_seconds = 2. * std::acos(-1.) / angular;
    options.sample_count = 33;
    options.parameter_scales.resize(2 * dof);
    options.torque_noise_standard_deviation.resize(dof);
    double expected = 0., minimum_log = 1e300, maximum_log = -1e300;
    const bool unit_scales = index % 3 != 0;
    for (int joint = 0; joint < dof; ++joint) {
        const double amplitude = design.sin_coefficients(joint, 0) = .5 * positive(random);
        const double noise = options.torque_noise_standard_deviation[joint] = unit_scales ? 1. : positive(random);
        const double inertia = options.parameter_scales[2 * joint] = unit_scales ? 1. : positive(random);
        const double viscous = options.parameter_scales[2 * joint + 1] = unit_scales ? 1. : positive(random);
        const double acceleration_log =
            2. * (std::log(amplitude) + 2. * std::log(angular) + std::log(inertia) - std::log(noise)) +
            std::log(16. / 33.);
        const double velocity_log =
            2. * (std::log(amplitude) + std::log(angular) + std::log(viscous) - std::log(noise)) + std::log(17. / 33.);
        expected += acceleration_log + velocity_log;
        minimum_log = std::min({minimum_log, acceleration_log, velocity_log});
        maximum_log = std::max({maximum_log, acceleration_log, velocity_log});
    }
    if (index % 3 == 1) {
        options.parameter_scales.resize(0);
        options.torque_noise_standard_deviation.resize(0);
    }
    expected /= 2 * dof;
    const auto quality = EvaluateExcitationDesign(design.trajectory(), backend, options);
    Require(quality.fully_identifiable() && std::abs(quality.mean_log_information - expected) < 1e-10,
            "multi-axis information differs from analytic Gram eigenvalues");
    const double condition = std::exp(.5 * (maximum_log - minimum_log));
    Require(std::abs(quality.condition_number / condition - 1.) < 1e-10, "analytic information condition changed");
    if (index % 64 != 0) return false;
    design.sin_coefficients *= .05;
    options.max_evaluations = 20;
    const ExcitationLimits limits{Eigen::VectorXd::Constant(dof, -1.), Eigen::VectorXd::Constant(dof, 1.),
                                  Eigen::VectorXd::Constant(dof, 10.), Eigen::VectorXd::Constant(dof, 100.),
                                  Eigen::VectorXd::Constant(dof, 1000.)};
    const auto result = OptimizeExcitationDesign(design, limits, backend, options);
    Require(result.validation.passed && result.quality.fully_identifiable() &&
                result.evaluations <= options.max_evaluations && result.accepted_steps != 0 &&
                result.quality.mean_log_information >= result.initial_quality.mean_log_information,
            "multi-axis pattern search lost feasibility, information or its budget");
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const auto [seconds, first] = synthetic_validation::ReplayCampaign(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, optimized = 0;
    double next_progress = 0.;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            CheckBasis(first + cases);
            if (CheckInformation(first + cases)) ++optimized;
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " optimized=" << optimized << std::endl;
                next_progress = elapsed + 60.;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << first + cases << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED first_case=" << first << " cases=" << cases << " optimized=" << optimized << std::endl;
}
