#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>
#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <pinocchio/config.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <random>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace robot_identification;
using robot_identification_pinocchio::PinocchioRegressor;

constexpr int kTrajectorySamples = 401;
constexpr double kGravity = 9.81;
// MuJoCo diagonalizes the source URDF's full inertia tensors on import.
constexpr double kOracleToleranceNm = 1e-7;
constexpr double kReductionToleranceNm = 1e-8;

std::shared_ptr<const pinocchio::Model> LoadModel(const char* path) {
    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(path, *model);
    model->gravity.linear() = Eigen::Vector3d(0., 0., -kGravity);
    return model;
}

std::vector<std::string> JointNames(const pinocchio::Model& model) {
    return {model.names.begin() + 1, model.names.end()};
}

Dataset Calibration(const pinocchio::Model& model, const ExcitationLimits& limits) {
    Dataset dataset;
    dataset.joint_names = JointNames(model);
    const Eigen::VectorXd midpoint = (limits.minimum_position + limits.maximum_position) / 2.;
    const Eigen::VectorXd radius = (limits.maximum_position - limits.minimum_position) / 2.;
    std::mt19937 random(0x20261124);
    std::uniform_real_distribution<double> unit(-1., 1.);
    for (int index = 0; index < 192; ++index) {
        Sample sample;
        sample.time_seconds = .02 * index;
        sample.position.resize(model.nv);
        sample.velocity.resize(model.nv);
        sample.acceleration.resize(model.nv);
        sample.torque = Eigen::VectorXd::Zero(model.nv);
        for (int joint = 0; joint < model.nv; ++joint) {
            sample.position[joint] = midpoint[joint] + .85 * radius[joint] * unit(random);
            sample.velocity[joint] = .75 * limits.maximum_velocity[joint] * unit(random);
            sample.acceleration[joint] = .75 * limits.maximum_acceleration[joint] * unit(random);
        }
        dataset.samples.push_back(std::move(sample));
    }
    return dataset;
}

HarmonicDesign SeedDesign(const ExcitationLimits& limits, bool holdout) {
    const Eigen::VectorXd center = (limits.minimum_position + limits.maximum_position) / 2.;
    const Eigen::VectorXd radius = (limits.maximum_position - limits.minimum_position) / 2.;
    HarmonicDesign design;
    design.fundamental_frequency_hz = holdout ? .16 : .125;
    design.offset = center;
    design.sin_coefficients.resize(center.size(), 3);
    design.cos_coefficients.resize(center.size(), 3);
    for (Eigen::Index joint = 0; joint < center.size(); ++joint) {
        if (holdout) design.offset[joint] += .18 * radius[joint] * std::cos(1.3 * joint);
        for (int harmonic = 0; harmonic < 3; ++harmonic) {
            const double index = static_cast<double>((joint + 1) * (harmonic + 1));
            const double phase = holdout ? 1.71 * index + .35 * harmonic : .84 * index;
            const double amplitude = (holdout ? .52 : .36) * radius[joint] / (harmonic + 2);
            design.sin_coefficients(joint, harmonic) = amplitude * std::cos(phase);
            design.cos_coefficients(joint, harmonic) = amplitude * std::sin(phase);
        }
    }
    return design;
}

void WriteMotion(const HarmonicTrajectory& trajectory,
                 const ExcitationLimits& limits,
                 const std::vector<std::string>& names,
                 const fs::path& path) {
    const double duration = trajectory.period_seconds();
    const double period = duration / (kTrajectorySamples - 1);
    const auto validation = ValidateExcitationTrajectory(trajectory, limits, duration, period);
    if (!validation.passed) throw std::runtime_error(path.string() + ": " + validation.reason);
    Dataset dataset{names, {}};
    for (int index = 0; index < kTrajectorySamples; ++index) {
        const double time = duration * index / (kTrajectorySamples - 1);
        auto motion = trajectory.Evaluate(time);
        dataset.samples.push_back({time, std::move(motion.position), std::move(motion.velocity),
                                   std::move(motion.acceleration), Eigen::VectorXd::Zero(names.size())});
    }
    WriteDatasetCsv(dataset, path.string());
}

void WriteQuality(const ExcitationDesignQuality& quality) {
    std::cout << "{\"rank\":" << quality.rank << ",\"condition_number\":" << quality.condition_number
              << ",\"mean_log_information\":" << quality.mean_log_information << '}';
}

void Design(const char* nominal_urdf, const fs::path& directory) {
    const auto model = LoadModel(nominal_urdf);
    const auto backend = std::make_shared<PinocchioRegressor>(model);
    const Eigen::VectorXd margin = .05 * (model->upperPositionLimit - model->lowerPositionLimit);
    const ExcitationLimits limits{model->lowerPositionLimit + margin, model->upperPositionLimit - margin,
                                  (.8 * model->velocityLimit).cwiseMin(4.), Eigen::VectorXd::Constant(model->nv, 25.),
                                  Eigen::VectorXd::Constant(model->nv, 160.)};
    const auto calibration = Calibration(*model, limits);
    const auto full = backend->BuildProblem(calibration);
    const BaseParameterBasis basis(full.regressor);
    const BaseParameterRegressor base(backend, basis);
    const auto baseline = SeedDesign(limits, false);
    ExcitationDesignOptions options;
    options.duration_seconds = baseline.trajectory().period_seconds();
    options.sample_count = 101;
    options.max_evaluations = 240;
    const auto result = OptimizeExcitationDesign(baseline, limits, base, options);
    if (!result.initial_quality.fully_identifiable() || !result.quality.fully_identifiable() ||
        !result.validation.passed || result.quality.mean_log_information < result.initial_quality.mean_log_information)
        throw std::runtime_error("excitation design failed rank, information or limit validation");

    WriteDatasetCsv(calibration, (directory / "calibration.csv").string());
    WriteMotion(baseline.trajectory(), limits, calibration.joint_names, directory / "baseline_states.csv");
    WriteMotion(result.design.trajectory(), limits, calibration.joint_names, directory / "optimized_states.csv");
    WriteMotion(SeedDesign(limits, true).trajectory(), limits, calibration.joint_names,
                directory / "holdout_states.csv");
    std::cout << "{\"baseline\":";
    WriteQuality(result.initial_quality);
    std::cout << ",\"optimized\":";
    WriteQuality(result.quality);
    std::cout << ",\"evaluations\":" << result.evaluations << ",\"accepted_steps\":" << result.accepted_steps
              << ",\"converged\":" << std::boolalpha << result.converged << "}\n";
}

Eigen::VectorXd DynamicParameters(const pinocchio::Model& model) {
    Eigen::VectorXd parameters(10 * (model.njoints - 1));
    for (int joint = 1; joint < model.njoints; ++joint)
        parameters.segment<10>(10 * (joint - 1)) = model.inertias[joint].toDynamicParameters();
    return parameters;
}

template <typename Vector>
void WriteVector(const Vector& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < static_cast<std::size_t>(values.size()); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void WriteFit(const RegressionProblem& train,
              const RegressionProblem& holdout,
              const Eigen::VectorXd& oracle,
              double huber_delta) {
    SolverOptions options;
    options.huber_delta = huber_delta;
    const auto result = Solve(train, options);
    if (result.rank != static_cast<std::size_t>(train.regressor.cols()))
        throw std::runtime_error("training regressor lost identifiable directions");
    const auto metrics = ValidateParameters(holdout, result.parameters);
    const Eigen::VectorXd residual = holdout.observation - holdout.regressor * result.parameters;
    const Eigen::Map<const Eigen::MatrixXd> joint_residual(residual.data(), holdout.dof, holdout.sample_count);
    const Eigen::VectorXd per_joint = joint_residual.array().square().rowwise().mean().sqrt();
    const double base_error = (result.parameters - oracle).norm() / oracle.norm();
    std::cout << "{\"rank\":" << result.rank << ",\"condition_number\":" << result.condition_number
              << ",\"train_rmse_nm\":" << result.rmse << ",\"holdout_rmse_nm\":" << metrics.rmse
              << ",\"holdout_max_absolute_error_nm\":" << metrics.max_absolute_error
              << ",\"holdout_normalized_rmse\":" << metrics.normalized_rmse << ",\"relative_base_error\":" << base_error
              << ",\"iterations\":" << result.iterations << ",\"converged\":" << std::boolalpha << result.converged
              << ",\"holdout_joint_rmse_nm\":";
    WriteVector(per_joint);
    std::cout << ",\"parameters\":";
    WriteVector(result.parameters);
    std::cout << '}';
}

void Fit(const char* nominal_urdf, const fs::path& directory, double huber_delta) {
    const auto model = LoadModel(nominal_urdf);
    const PinocchioRegressor backend(model);
    const auto build = [&](const char* filename) {
        return backend.BuildProblem(ReadDatasetCsv((directory / filename).string()));
    };
    const auto calibration = build("calibration.csv");
    const BaseParameterBasis basis(calibration.regressor);
    const auto full_holdout = build("holdout.csv");
    const auto holdout = basis.ReduceProblem(full_holdout);
    const auto nominal = basis.ToBaseParameters(DynamicParameters(*model));
    const double nominal_rmse = ValidateParameters(holdout, nominal).rmse;

    const auto truth = LoadModel((directory / "truth.urdf").c_str());
    if (JointNames(*truth) != JointNames(*model)) throw std::runtime_error("truth model changed joint order");
    const auto full_truth = DynamicParameters(*truth);
    const auto oracle = basis.ToBaseParameters(full_truth);
    const Eigen::VectorXd full_prediction = full_holdout.regressor * full_truth;
    const Eigen::VectorXd reduced_prediction = holdout.regressor * oracle;
    const double engine_error = (holdout.observation - full_prediction).cwiseAbs().maxCoeff();
    const double reduction_error = (full_prediction - reduced_prediction).cwiseAbs().maxCoeff();
    if (!(engine_error <= kOracleToleranceNm) || !(reduction_error <= kReductionToleranceNm)) {
        std::cerr << "holdout maximum torque errors [N m]: cross-engine=" << engine_error
                  << ", reduction=" << reduction_error << '\n';
        throw std::runtime_error("clean holdout oracle check failed");
    }

    std::cout << "{\"pinocchio_version\":" << std::quoted(PINOCCHIO_VERSION)
              << ",\"full_parameter_count\":" << basis.full_parameter_count()
              << ",\"base_parameter_count\":" << basis.parameter_count()
              << ",\"calibration_reconstruction_error\":" << basis.calibration_error()
              << ",\"holdout_reconstruction_error\":" << basis.ReconstructionError(full_holdout.regressor)
              << ",\"nominal_holdout_rmse_nm\":" << nominal_rmse
              << ",\"oracle_holdout_max_absolute_error_nm\":" << engine_error
              << ",\"reduced_oracle_max_difference_nm\":" << reduction_error << ",\"selected_columns\":";
    WriteVector(basis.selected_columns());
    std::cout << ",\"column_scales\":";
    WriteVector(basis.column_scales());
    std::cout << ",\"full_to_base\":[";
    for (Eigen::Index index = 0; index < basis.full_to_base().rows(); ++index) {
        if (index != 0) std::cout << ',';
        WriteVector(basis.full_to_base().row(index));
    }
    std::cout << ']';
    std::cout << ",\"oracle_parameters\":";
    WriteVector(oracle);
    for (const auto& name : {"baseline", "optimized"}) {
        const auto filename = std::string("train_") + name + ".csv";
        const auto train = basis.ReduceProblem(build(filename.c_str()));
        std::cout << ",\"" << name << "\":{\"ols\":";
        WriteFit(train, holdout, oracle, 0.);
        std::cout << ",\"huber\":";
        WriteFit(train, holdout, oracle, huber_delta);
        std::cout << '}';
    }
    std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        const auto mode = argc > 1 ? std::string(argv[1]) : "";
        if (argc == 4 && mode == "design") {
            Design(argv[2], argv[3]);
        } else if (argc == 5 && mode == "fit") {
            std::size_t end = 0;
            const double delta = std::stod(argv[4], &end);
            if (argv[4][end] != '\0' || !std::isfinite(delta) || delta <= 0.)
                throw std::invalid_argument("Huber delta must be finite and positive");
            Fit(argv[2], argv[3], delta);
        } else {
            std::cerr << "usage: arm_identification design NOMINAL_URDF DIRECTORY\n"
                         "       arm_identification fit NOMINAL_URDF DIRECTORY HUBER_DELTA_NM\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "arm_identification: " << error.what() << '\n';
        return 1;
    }
}
