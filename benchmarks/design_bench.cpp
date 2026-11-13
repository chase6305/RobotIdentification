// Synthetic API benchmarks; compare fresh copies with identical compiler flags.
#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace robot_identification;

class Actuators final : public RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return 14; }
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> result;
        for (int index = 0; index < 14; ++index) result.push_back(std::to_string(index));
        return result;
    }
    Eigen::MatrixXd Compute(const Sample& sample) const override {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(7, 14);
        for (int joint = 0; joint < 7; ++joint) {
            result(joint, 2 * joint) = sample.acceleration[joint];
            result(joint, 2 * joint + 1) = sample.velocity[joint];
        }
        return result;
    }
};

template <typename Function>
void Measure(const std::string& name, std::size_t iterations, Function function) {
    const double warmup = function();
    double checksum = 0.;
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start = std::clock();
    for (std::size_t index = 0; index < iterations; ++index) checksum += function();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto cpu_end = std::clock();
    if (cpu_start == static_cast<std::clock_t>(-1) || cpu_end == static_cast<std::clock_t>(-1))
        throw std::runtime_error("process CPU timer is unavailable");
    const double cpu_seconds = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    if (!std::isfinite(warmup) || !std::isfinite(checksum)) throw std::runtime_error("nonfinite benchmark result");
    std::cout << std::setprecision(17) << "{\"case\":\"" << name << "\",\"iterations\":" << iterations
              << ",\"microseconds_per_call\":" << 1e6 * seconds / iterations
              << ",\"cpu_microseconds_per_call\":" << 1e6 * cpu_seconds / iterations << ",\"checksum\":" << checksum
              << "}\n";
}

void Basis(const std::string& name, std::size_t iterations) {
    const bool scalar = name == "basis-scalar" || name == "basis-zero-scalar";
    const bool zero = name.rfind("basis-zero", 0) == 0;
    const bool mixed = name.rfind("basis-mixed", 0) == 0;
    const Eigen::Index rows =
        scalar ? 1 : (name == "basis-batch" || name == "basis-zero-batch" || name == "basis-mixed-batch" ? 65537 : 14);
    const Eigen::Index rank = scalar ? 1 : 80, columns = scalar ? 2 : (name == "basis-full-rank" ? rank : 140);
    Eigen::MatrixXd calibration(256, rank), holdout(rows, rank), combination(rank, columns);
    combination.setZero();
    combination.leftCols(rank).setIdentity();
    if (!zero)
        for (Eigen::Index row = 0; row < rank; ++row)
            for (Eigen::Index column = rank; column < (mixed ? rank + rank / 8 : columns); ++column)
                combination(row, column) = std::sin(.03 * (row + 1) * (column + 1));
    for (Eigen::Index column = 0; column < rank; ++column) {
        for (Eigen::Index row = 0; row < calibration.rows(); ++row)
            calibration(row, column) = std::cos(std::acos(-1.) * (row + .5) * (column + 1) / calibration.rows());
        for (Eigen::Index row = 0; row < rows; ++row) holdout(row, column) = std::sin(.07 * (row + .4) * (column + 1));
    }
    const BaseParameterBasis basis(calibration * combination);
    if (basis.parameter_count() != static_cast<std::size_t>(rank)) throw std::runtime_error("unexpected basis rank");
    const Eigen::MatrixXd values = holdout * combination;
    Measure(name, iterations, [&] { return basis.ReduceRegressor(values).sum(); });
}

void Reconstruction(const std::string& name, std::size_t iterations) {
    const bool mixed = name.rfind("reconstruction-mixed", 0) == 0;
    const Eigen::Index rank = name == "reconstruction-scalar" || name == "reconstruction-pair" ? 1
                              : name == "reconstruction-small"                                 ? 8
                              : name == "reconstruction-wide"                                  ? 24
                              : name == "reconstruction-large"                                 ? 160
                                                                                               : 80;
    const Eigen::Index rows = name == "reconstruction-scalar"                                          ? 1
                              : name == "reconstruction-pair"                                          ? 2
                              : name == "reconstruction-small"                                         ? 14
                              : name == "reconstruction-254"                                           ? 254
                              : name == "reconstruction-256" || name == "reconstruction-mixed-256"     ? 256
                              : name == "reconstruction-511"                                           ? 511
                              : name == "reconstruction-batch" || name == "reconstruction-mixed-batch" ? 65537
                                                                                                       : 255;
    const Eigen::Index columns = rank == 1 ? 2 : name == "reconstruction-wide" ? 72 : rank + 3 * rank / 4;
    Eigen::MatrixXd calibration(2 * rank + 4, rank), holdout(rows, rank), combination(rank, columns);
    combination.setZero();
    combination.leftCols(rank).setIdentity();
    for (Eigen::Index column = rank; column < (mixed ? rank + rank / 8 : columns); ++column)
        for (Eigen::Index row = 0; row < rank; ++row)
            combination(row, column) = std::sin(.03 * (row + 1) * (column + 1));
    for (Eigen::Index column = 0; column < rank; ++column) {
        for (Eigen::Index row = 0; row < calibration.rows(); ++row)
            calibration(row, column) = std::cos(std::acos(-1.) * (row + .5) * (column + 1) / calibration.rows());
        for (Eigen::Index row = 0; row < rows; ++row) holdout(row, column) = std::sin(.07 * (row + .4) * (column + 1));
    }
    const BaseParameterBasis basis(calibration * combination);
    if (basis.parameter_count() != static_cast<std::size_t>(rank)) throw std::runtime_error("unexpected basis rank");
    const Eigen::MatrixXd values = holdout * combination;
    Measure(name, iterations, [&] { return basis.ReconstructionError(values); });
}

void Bounds(const std::string& name, std::size_t iterations) {
    Eigen::MatrixXd sine(7, 20), cosine = Eigen::MatrixXd::Zero(7, 20);
    for (Eigen::Index row = 0; row < sine.rows(); ++row)
        for (Eigen::Index column = 0; column < sine.cols(); ++column) {
            sine(row, column) = name == "bounds-exact" ? 3. / 1024. : .001 * std::sin(row + column);
            if (name == "bounds-exact")
                cosine(row, column) = 4. / 1024.;
            else if (name == "bounds-dense")
                cosine(row, column) = .001 * std::cos(row - column);
        }
    const double hz = name == "bounds-exact" ? 1. / (2. * std::acos(-1.)) : .1;
    const HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(7), sine, cosine, hz);
    Measure(name, iterations, [&] {
        const auto result = trajectory.ConservativeBounds();
        return result.maximum_position.sum() + result.maximum_absolute_jerk.sum();
    });
}

void Sampling(const std::string& name, std::size_t iterations) {
    const HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(7), Eigen::MatrixXd::Constant(7, 3, .04),
                                        Eigen::MatrixXd::Constant(7, 3, .02), .1);
    ExcitationSample sample;
    if (name == "trajectory-validation") {
        const ExcitationLimits limits{Eigen::VectorXd::Constant(7, -2.), Eigen::VectorXd::Constant(7, 2.),
                                      Eigen::VectorXd::Constant(7, 10.), Eigen::VectorXd::Constant(7, 100.),
                                      Eigen::VectorXd::Constant(7, 1000.)};
        Measure(name, iterations, [&] {
            const auto result = ValidateExcitationTrajectory(trajectory, limits, 10., .05);
            if (!result.passed || result.sample_count != 201) throw std::runtime_error("invalid sampled validation");
            return result.peak_absolute_jerk.sum();
        });
    } else {
        Measure(name, iterations, [&] {
            if (name == "trajectory-owning") {
                sample = trajectory.Evaluate(.123);
            } else {
#ifdef ROBOT_IDENTIFICATION_BENCH_HAS_EVALUATE_INTO
                trajectory.EvaluateInto(.123, sample);
#else
                // The pre-0.2.4 comparison uses the owning API plus assignment.
                sample = trajectory.Evaluate(.123);
#endif
            }
            return sample.position.sum() + sample.velocity.sum() + sample.acceleration.sum() + sample.jerk.sum();
        });
    }
}

void Quality(const std::string& name, std::size_t iterations) {
    HarmonicDesign design{Eigen::VectorXd::Zero(7), Eigen::MatrixXd::Constant(7, 3, .04),
                          Eigen::MatrixXd::Constant(7, 3, .02), .1};
    const Actuators backend;
    ExcitationDesignOptions options;
    if (name == "quality-scaled" || name == "optimize-scaled") {
        options.parameter_scales = Eigen::VectorXd::LinSpaced(14, .5, 2.);
        options.torque_noise_standard_deviation = Eigen::VectorXd::LinSpaced(7, .7, 1.4);
    }
    if (name == "optimize" || name == "optimize-scaled") {
        options.max_evaluations = 15;
        const ExcitationLimits limits{Eigen::VectorXd::Constant(7, -1.), Eigen::VectorXd::Constant(7, 1.),
                                      Eigen::VectorXd::Constant(7, 10.), Eigen::VectorXd::Constant(7, 100.),
                                      Eigen::VectorXd::Constant(7, 1000.)};
        Measure(name, iterations, [&] {
            const auto result = OptimizeExcitationDesign(design, limits, backend, options);
            if (!result.validation.passed || !result.quality.fully_identifiable() || result.evaluations > 15)
                throw std::runtime_error("invalid optimization result");
            return result.quality.mean_log_information;
        });
    } else {
        const auto trajectory = design.trajectory();
        Measure(name, iterations, [&] {
            const auto result = EvaluateExcitationDesign(trajectory, backend, options);
            if (!result.fully_identifiable()) throw std::runtime_error("unexpected information rank");
            return result.mean_log_information;
        });
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::invalid_argument("expected CASE [ITERATIONS]");
        const std::string name(argv[1]);
        std::size_t iterations = name == "basis-batch" ? 1 : 100;
        if (argc == 3) {
            std::size_t consumed;
            const std::string argument(argv[2]);
            iterations = std::stoull(argument, &consumed);
            if (argument.empty() || argument.front() == '-' || consumed != argument.size())
                throw std::invalid_argument("invalid iteration count");
        }
        if (iterations == 0 || iterations > 1000000) throw std::invalid_argument("iteration count out of range");
        if (name == "basis-small" || name == "basis-batch" || name == "basis-full-rank" || name == "basis-scalar" ||
            name == "basis-mixed-small" || name == "basis-mixed-batch" || name == "basis-zero-scalar" ||
            name == "basis-zero-small" || name == "basis-zero-batch")
            Basis(name, iterations);
        else if (name == "reconstruction-scalar" || name == "reconstruction-pair" || name == "reconstruction-small" ||
                 name == "reconstruction-wide" || name == "reconstruction-large" || name == "reconstruction-254" ||
                 name == "reconstruction-255" || name == "reconstruction-256" || name == "reconstruction-511" ||
                 name == "reconstruction-batch" || name == "reconstruction-mixed-255" ||
                 name == "reconstruction-mixed-256" || name == "reconstruction-mixed-batch")
            Reconstruction(name, iterations);
        else if (name == "bounds-dense" || name == "bounds-sine" || name == "bounds-exact")
            Bounds(name, iterations);
        else if (name == "trajectory-owning" || name == "trajectory-reused" || name == "trajectory-validation")
            Sampling(name, iterations);
        else if (name == "quality-unit" || name == "quality-scaled" || name == "optimize" || name == "optimize-scaled")
            Quality(name, iterations);
        else
            throw std::invalid_argument("unknown benchmark case");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
