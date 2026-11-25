#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>
#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <pinocchio/config.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <random>
#include <stdexcept>
#include <string_view>

#include "forward_validation.h"
#include "observable_parameters.h"
#include "payload_fit.h"
#include "physical_arm_fit.h"
#include "raw_preprocessing.h"
#include "robust_scale.h"

namespace {
namespace fs = std::filesystem;
using namespace robot_identification;
using robot_identification_pinocchio::PinocchioRegressor;

constexpr std::size_t kTrajectorySamples = 401;
constexpr std::size_t kDesignEvaluations = 240;
constexpr double kTrainingPeriodSeconds = 8.;
constexpr double kHoldoutPeriodSeconds = 6.25;
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

class ArmRegression final : public RegressorBackend {
public:
    ArmRegression(std::shared_ptr<const pinocchio::Model> model, bool actuator_terms)
        : rigid_(std::make_shared<PinocchioRegressor>(std::move(model))), backend_(rigid_) {
        if (actuator_terms) {
            JointActuatorRegressorOptions options;
            options.coulomb_friction = false;
            options.torque_bias = false;
            backend_ = std::make_shared<JointActuatorRegressor>(rigid_, JointNames(rigid_->model()), options);
        }
    }

    [[nodiscard]] std::size_t parameter_count() const noexcept override { return backend_->parameter_count(); }
    [[nodiscard]] std::vector<std::string> parameter_names() const override { return backend_->parameter_names(); }

    [[nodiscard]] Eigen::MatrixXd Compute(const Sample& sample) const override {
        auto regressor = backend_->Compute(sample);
        ClearRoundoffColumns(regressor);
        return regressor;
    }

    [[nodiscard]] RegressionProblem BuildProblem(const Dataset& dataset) const {
        if (dataset.joint_names != JointNames(rigid_->model()))
            throw std::invalid_argument("dataset joint names do not match the loaded arm");
        auto problem = backend_ == rigid_ ? rigid_->BuildProblem(dataset) : BuildRegressionProblem(dataset, *backend_);
        ClearRoundoffColumns(problem.regressor);
        return problem;
    }

    [[nodiscard]] RegressionProblem Calibrate(const Dataset& dataset) {
        auto problem = BuildProblem(dataset);
        const Eigen::RowVectorXd scales = problem.regressor.cwiseAbs().colwise().maxCoeff();
        // Pinocchio can return roundoff in structurally zero inertia columns.
        // Unit column normalization would amplify those values into apparent rank.
        roundoff_tolerance_ = 64. * std::numeric_limits<double>::epsilon() * scales.maxCoeff();
        for (Eigen::Index column = 0; column < scales.size(); ++column)
            if (scales[column] <= roundoff_tolerance_) roundoff_columns_.push_back(column);
        ClearRoundoffColumns(problem.regressor);
        return problem;
    }

private:
    void ClearRoundoffColumns(Eigen::MatrixXd& regressor) const {
        for (const auto column : roundoff_columns_) {
            if (regressor.col(column).cwiseAbs().maxCoeff() > roundoff_tolerance_)
                throw std::runtime_error("a calibration-zero regressor column exceeds the roundoff bound");
            regressor.col(column).setZero();
        }
    }

    std::shared_ptr<const PinocchioRegressor> rigid_;
    std::shared_ptr<const RegressorBackend> backend_;
    std::vector<Eigen::Index> roundoff_columns_;
    double roundoff_tolerance_ = 0.;
};

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

HarmonicDesign SeedDesign(const ExcitationLimits& limits, bool holdout, double period_seconds) {
    const Eigen::VectorXd center = (limits.minimum_position + limits.maximum_position) / 2.;
    const Eigen::VectorXd radius = (limits.maximum_position - limits.minimum_position) / 2.;
    HarmonicDesign design;
    design.fundamental_frequency_hz = 1. / period_seconds;
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
    const auto bounds = design.trajectory().ConservativeBounds();
    for (Eigen::Index joint = 0; joint < center.size(); ++joint) {
        const double scale = std::min({1., limits.maximum_velocity[joint] / bounds.maximum_absolute_velocity[joint],
                                       limits.maximum_acceleration[joint] / bounds.maximum_absolute_acceleration[joint],
                                       limits.maximum_jerk[joint] / bounds.maximum_absolute_jerk[joint]});
        if (scale < 1.) {
            // Leave slack for rounding when the requested period increases derivative bounds.
            design.sin_coefficients.row(joint) *= .95 * scale;
            design.cos_coefficients.row(joint) *= .95 * scale;
        }
    }
    return design;
}

void WriteMotion(const HarmonicTrajectory& trajectory,
                 const ExcitationLimits& limits,
                 const std::vector<std::string>& names,
                 const fs::path& path,
                 std::size_t sample_count) {
    const double duration = trajectory.period_seconds();
    const double period = duration / (sample_count - 1);
    const auto validation = ValidateExcitationTrajectory(trajectory, limits, duration, period);
    if (!validation.passed) throw std::runtime_error(path.string() + ": " + validation.reason);
    Dataset dataset{names, {}};
    dataset.samples.reserve(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const double time = duration * index / (sample_count - 1);
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

template <typename Vector>
void WriteVector(const Vector& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < static_cast<std::size_t>(values.size()); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void WriteMatrix(const Eigen::MatrixXd& values) {
    std::cout << '[';
    for (Eigen::Index index = 0; index < values.rows(); ++index) {
        if (index != 0) std::cout << ',';
        WriteVector(values.row(index));
    }
    std::cout << ']';
}

void WriteReference(const HarmonicDesign& design) {
    std::cout << "{\"frequency_hz\":" << design.fundamental_frequency_hz << ",\"offset_rad\":";
    WriteVector(design.offset);
    std::cout << ",\"sin_rad\":";
    WriteMatrix(design.sin_coefficients);
    std::cout << ",\"cos_rad\":";
    WriteMatrix(design.cos_coefficients);
    std::cout << '}';
}

void Design(const char* nominal_urdf,
            const fs::path& directory,
            std::size_t samples,
            std::size_t evaluations,
            double training_period,
            double holdout_period,
            bool actuator_terms) {
    if (samples < 3 || evaluations == 0 || training_period <= 0. || holdout_period <= 0.)
        throw std::invalid_argument("design requires at least three samples, one evaluation and positive periods");
    const auto model = LoadModel(nominal_urdf);
    const auto arm = std::make_shared<ArmRegression>(model, actuator_terms);
    const Eigen::VectorXd margin = .05 * (model->upperPositionLimit - model->lowerPositionLimit);
#if PINOCCHIO_VERSION_AT_LEAST(4, 0, 0)
    const auto& velocity_limit = model->upperVelocityLimit;
#else
    const auto& velocity_limit = model->velocityLimit;
#endif
    const ExcitationLimits limits{model->lowerPositionLimit + margin, model->upperPositionLimit - margin,
                                  (.8 * velocity_limit).cwiseMin(4.), Eigen::VectorXd::Constant(model->nv, 25.),
                                  Eigen::VectorXd::Constant(model->nv, 160.)};
    const auto calibration = Calibration(*model, limits);
    const auto full = arm->Calibrate(calibration);
    const BaseParameterBasis basis(full.regressor);
    const BaseParameterRegressor base(arm, basis);
    const auto baseline = SeedDesign(limits, false, training_period);
    const auto holdout = SeedDesign(limits, true, holdout_period);
    ExcitationDesignOptions options;
    options.duration_seconds = baseline.trajectory().period_seconds();
    options.sample_count = std::min<std::size_t>(101, samples);
    options.max_evaluations = evaluations;
    const auto result = OptimizeExcitationDesign(baseline, limits, base, options);
    if (!result.initial_quality.fully_identifiable() || !result.quality.fully_identifiable() ||
        !result.validation.passed || result.quality.mean_log_information < result.initial_quality.mean_log_information)
        throw std::runtime_error("excitation design failed rank, information or limit validation");

    WriteDatasetCsv(calibration, (directory / "calibration.csv").string());
    WriteMotion(baseline.trajectory(), limits, calibration.joint_names, directory / "baseline_states.csv", samples);
    WriteMotion(result.design.trajectory(), limits, calibration.joint_names, directory / "optimized_states.csv",
                samples);
    WriteMotion(holdout.trajectory(), limits, calibration.joint_names, directory / "holdout_states.csv", samples);
    std::cout << "{\"baseline\":";
    WriteQuality(result.initial_quality);
    std::cout << ",\"optimized\":";
    WriteQuality(result.quality);
    std::cout << ",\"evaluations\":" << result.evaluations << ",\"accepted_steps\":" << result.accepted_steps
              << ",\"converged\":" << std::boolalpha << result.converged << ",\"references\":{\"baseline\":";
    WriteReference(baseline);
    std::cout << ",\"optimized\":";
    WriteReference(result.design);
    std::cout << ",\"holdout\":";
    WriteReference(holdout);
    std::cout << "},\"kinematic_limits\":{\"minimum_position_rad\":";
    WriteVector(limits.minimum_position);
    std::cout << ",\"maximum_position_rad\":";
    WriteVector(limits.maximum_position);
    std::cout << ",\"maximum_velocity_rad_s\":";
    WriteVector(limits.maximum_velocity);
    std::cout << ",\"maximum_acceleration_rad_s2\":";
    WriteVector(limits.maximum_acceleration);
    std::cout << ",\"maximum_jerk_rad_s3\":";
    WriteVector(limits.maximum_jerk);
    std::cout << "}}\n";
}

Eigen::VectorXd DynamicParameters(const pinocchio::Model& model, const std::vector<double>& drive_parameters) {
    const auto inertia_count = 10 * (model.njoints - 1);
    Eigen::VectorXd parameters(inertia_count + drive_parameters.size());
    for (int joint = 1; joint < model.njoints; ++joint)
        parameters.segment<10>(10 * (joint - 1)) = model.inertias[joint].toDynamicParameters();
    for (std::size_t index = 0; index < drive_parameters.size(); ++index)
        parameters[inertia_count + index] = drive_parameters[index];
    return parameters;
}

void WriteForwardMetrics(const arm_identification::ForwardMetrics& forward) {
    std::cout << "{\"mass_check_passed\":" << std::boolalpha << forward.mass.passed
              << ",\"mass_samples\":" << forward.mass.tested << ",\"mass_failures\":" << forward.mass.failed
              << ",\"minimum_mass_eigenvalue_kg_m2\":";
    if (std::isfinite(forward.mass.minimum_eigenvalue))
        std::cout << forward.mass.minimum_eigenvalue;
    else
        std::cout << "null";
    std::cout << ",\"holdout_samples\":" << forward.holdout_samples << ",\"holdout_acceleration_rmse_rad_s2\":";
    if (forward.holdout_acceleration_rmse)
        std::cout << *forward.holdout_acceleration_rmse;
    else
        std::cout << "null";
    std::cout << '}';
}

void WriteReplayMetrics(const arm_identification::ReplayMetrics& replay) {
    std::cout << "{\"window_seconds\":" << replay.requested_window_seconds << ",\"windows\":" << replay.windows
              << ",\"complete_windows\":" << replay.complete_windows << ",\"position_rmse_rad\":";
    if (replay.position_rmse)
        std::cout << *replay.position_rmse;
    else
        std::cout << "null";
    std::cout << ",\"maximum_final_position_error_rad\":";
    if (replay.maximum_final_position_error)
        std::cout << *replay.maximum_final_position_error;
    else
        std::cout << "null";
    std::cout << '}';
}

void WritePayloadFit(const arm_identification::PayloadFit& payload) {
    std::cout << "{\"estimated_mass_kg\":" << payload.estimated_mass
              << ",\"unconstrained_holdout_rmse_nm\":" << payload.unconstrained_holdout_rmse
              << ",\"physical_candidate\":";
    if (payload.physical) {
        const auto& physical = *payload.physical;
        std::cout << "{\"mass_kg\":" << physical.mass << ",\"center_m\":";
        WriteVector(physical.center);
        std::cout << ",\"inertia_at_center_kg_m2\":";
        WriteMatrix(physical.inertia_at_center);
        std::cout << ",\"holdout_rmse_nm\":" << physical.holdout_rmse << ",\"projected_inertia\":" << std::boolalpha
                  << physical.projected << '}';
    } else {
        std::cout << "null";
    }
    std::cout << '}';
}

void WriteActuatorEstimates(const std::vector<std::string>& names,
                            const std::vector<double>& configured,
                            const std::vector<std::optional<Eigen::VectorXd>>& coordinates,
                            const Eigen::VectorXd& parameters,
                            const Eigen::MatrixXd* covariance) {
    const auto first = names.size() - configured.size();
    std::cout << '[';
    for (std::size_t index = 0; index < configured.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << "{\"name\":" << std::quoted(names[first + index]) << ",\"configured\":" << configured[index]
                  << ",\"estimate\":";
        if (coordinates[index]) {
            const auto& weights = *coordinates[index];
            std::cout << weights.dot(parameters) << ",\"standard_deviation\":";
            const double variance = covariance ? weights.dot(*covariance * weights) : -1.;
            if (variance >= 0. && std::isfinite(variance))
                std::cout << std::sqrt(variance);
            else
                std::cout << "null";
        } else {
            std::cout << "null,\"standard_deviation\":null";
        }
        std::cout << '}';
    }
    std::cout << ']';
}

void WriteFit(const IdentificationResult& result,
              const RegressionProblem& holdout,
              const Eigen::VectorXd& oracle,
              double huber_delta,
              const arm_identification::ForwardValidation& validation) {
    if (result.rank != static_cast<std::size_t>(holdout.regressor.cols()))
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
              << ",\"huber_delta_nm\":" << huber_delta << ",\"iterations\":" << result.iterations
              << ",\"converged\":" << std::boolalpha << result.converged << ",\"holdout_joint_rmse_nm\":";
    WriteVector(per_joint);
    std::cout << ",\"parameters\":";
    WriteVector(result.parameters);
    std::cout << ",\"forward_validation\":";
    WriteForwardMetrics(validation.Evaluate(result.parameters));
    std::cout << '}';
}

void WriteTorquePrediction(Dataset dataset, const Eigen::VectorXd& prediction, const fs::path& path) {
    const auto dof = static_cast<Eigen::Index>(dataset.dof());
    for (std::size_t sample = 0; sample < dataset.samples.size(); ++sample)
        dataset.samples[sample].torque = prediction.segment(sample * dof, dof);
    WriteDatasetCsv(dataset, path.string());
}

void Fit(const char* nominal_urdf,
         const fs::path& directory,
         std::optional<double> huber_delta,
         const std::vector<double>& drive_parameters,
         bool estimate_payload,
         bool estimate_physical_arm) {
    const auto model = LoadModel(nominal_urdf);
    const bool actuator_terms = !drive_parameters.empty();
    if (actuator_terms && drive_parameters.size() != static_cast<std::size_t>(2 * model->nv))
        throw std::invalid_argument("expected one armature and damping pair per joint");
    const auto arm = std::make_shared<ArmRegression>(model, actuator_terms);
    const auto build = [&](const char* filename) {
        return arm->BuildProblem(ReadDatasetCsv((directory / filename).string()));
    };
    const auto calibration_data = ReadDatasetCsv((directory / "calibration.csv").string());
    const auto calibration = arm->Calibrate(calibration_data);
    const BaseParameterBasis basis(calibration.regressor);
    const auto holdout_data = ReadDatasetCsv((directory / "holdout.csv").string());
    auto full_holdout = arm->BuildProblem(holdout_data);
    const auto holdout = basis.ReduceProblem(full_holdout);
    const auto full_nominal = DynamicParameters(*model, std::vector<double>(drive_parameters.size()));
    const auto nominal = basis.ToBaseParameters(full_nominal);
    const double nominal_rmse = ValidateParameters(holdout, nominal).rmse;

    const auto truth = LoadModel((directory / "truth.urdf").c_str());
    if (JointNames(*truth) != JointNames(*model)) throw std::runtime_error("truth model changed joint order");
    const auto full_truth = DynamicParameters(*truth, drive_parameters);
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
    const auto holdout_reconstruction_error = basis.ReconstructionError(full_holdout.regressor);
    const auto payload_column = 10 * (model->njoints - 2);
    Eigen::VectorXd payload_prior;
    RegressionProblem payload_holdout;
    if (estimate_payload) {
        payload_prior = DynamicParameters(*model, drive_parameters);
        payload_holdout = arm_identification::PayloadRegression(full_holdout, payload_prior, payload_column);
    }
    full_holdout = {};
    const BaseParameterRegressor base(arm, basis);
    const arm_identification::ForwardValidation validation(base, calibration_data, holdout_data);

    std::cout << "{\"pinocchio_version\":" << std::quoted(PINOCCHIO_VERSION)
              << ",\"full_parameter_count\":" << basis.full_parameter_count()
              << ",\"base_parameter_count\":" << basis.parameter_count()
              << ",\"calibration_reconstruction_error\":" << basis.calibration_error()
              << ",\"holdout_reconstruction_error\":" << holdout_reconstruction_error
              << ",\"nominal_holdout_rmse_nm\":" << nominal_rmse
              << ",\"oracle_holdout_max_absolute_error_nm\":" << engine_error
              << ",\"reduced_oracle_max_difference_nm\":" << reduction_error << ",\"parameter_names\":[";
    const auto parameter_names = arm->parameter_names();
    for (std::size_t index = 0; index < parameter_names.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << std::quoted(parameter_names[index]);
    }
    std::cout << "],\"selected_columns\":";
    WriteVector(basis.selected_columns());
    std::cout << ",\"column_scales\":";
    WriteVector(basis.column_scales());
    std::cout << ",\"full_to_base\":";
    WriteMatrix(basis.full_to_base());
    std::cout << ",\"oracle_parameters\":";
    WriteVector(oracle);
    std::cout << ",\"nominal_forward_validation\":";
    WriteForwardMetrics(validation.Evaluate(nominal));
    std::vector<std::optional<Eigen::VectorXd>> drive_coordinates;
    if (actuator_terms)
        drive_coordinates = arm_identification::ObservableCoordinates(
            basis.full_to_base(), basis.full_parameter_count() - drive_parameters.size(), drive_parameters.size());
    Eigen::VectorXd optimized_parameters;
    Eigen::MatrixXd optimized_covariance;
    arm_identification::DriveProjection drive_projection;
    double optimized_delta = 0.;
    std::optional<arm_identification::PayloadFit> payload;
    std::optional<arm_identification::PhysicalArmFit> physical_arm;
    for (const auto& name : {"baseline", "optimized"}) {
        const bool optimized = std::string_view(name) == "optimized";
        const auto filename = std::string("train_") + name + ".csv";
        auto full_train = build(filename.c_str());
        const auto train = basis.ReduceProblem(full_train);
        if (!optimized || (!estimate_payload && !estimate_physical_arm)) full_train = {};
        const auto ols = Solve(train);
        SolverOptions options;
        options.huber_delta = huber_delta ? *huber_delta
                                          : arm_identification::HuberDeltaFromResiduals(
                                                ols.residual, train.observation.cwiseAbs().maxCoeff());
        auto robust = Solve(train, options);
        std::cout << ",\"" << name << "\":{\"ols\":";
        WriteFit(ols, holdout, oracle, 0., validation);
        std::cout << ",\"huber\":";
        WriteFit(robust, holdout, oracle, options.huber_delta, validation);
        std::cout << '}';
        if (optimized) {
            WriteTorquePrediction(holdout_data, holdout.regressor * robust.parameters,
                                  directory / "holdout_predicted_optimized_huber.csv");
            if (estimate_physical_arm) {
                const double residual_scale = arm_identification::HuberDeltaFromResiduals(
                                                  robust.residual, train.observation.cwiseAbs().maxCoeff()) /
                                              arm_identification::kHuberGaussianMultiplier;
                physical_arm = arm_identification::FitPhysicalArm(full_train, robust.final_row_weights, full_nominal,
                                                                  model->nv, residual_scale);
            }
            if (actuator_terms) {
                drive_projection = arm_identification::ProjectNonnegativeDrive(
                    train.regressor, robust.final_row_weights, robust.parameters, drive_coordinates);
                optimized_covariance = std::move(robust.covariance);
            }
            optimized_parameters = std::move(robust.parameters);
            optimized_delta = options.huber_delta;
            if (estimate_payload) {
                const auto payload_train =
                    arm_identification::PayloadRegression(full_train, payload_prior, payload_column);
                payload = arm_identification::FitPayload(payload_train, payload_holdout, optimized_delta);
            }
        }
    }
    if (payload) {
        std::cout << ",\"payload_fit\":";
        WritePayloadFit(*payload);
    }
    Eigen::VectorXd physical_base;
    if (physical_arm) {
        physical_base = basis.ToBaseParameters(physical_arm->parameters);
        const auto metrics = ValidateParameters(holdout, physical_base);
        WriteTorquePrediction(holdout_data, holdout.regressor * physical_base,
                              directory / "holdout_predicted_physical_arm.csv");
        std::cout << ",\"physical_arm_fit\":{\"holdout_rmse_nm\":" << metrics.rmse
                  << ",\"iterations\":" << physical_arm->iterations << ",\"converged\":" << std::boolalpha
                  << physical_arm->converged << ",\"prior_weight\":" << physical_arm->prior_weight
                  << ",\"minimum_relative_moment\":" << physical_arm->minimum_relative_moment
                  << ",\"full_parameters\":";
        WriteVector(physical_arm->parameters);
        std::cout << ",\"forward_validation\":";
        WriteForwardMetrics(validation.Evaluate(physical_base));
        std::cout << '}';
    }
    const auto replay_path = directory / "holdout_dense.csv";
    if (fs::is_regular_file(replay_path)) {
        const auto dense = ReadDatasetCsv(replay_path.string());
        if (dense.joint_names != holdout_data.joint_names)
            throw std::invalid_argument("dense holdout joint order does not match captured data");
        const auto replay = [&](const Eigen::VectorXd& parameters) {
            return arm_identification::ReplayForwardWindows(base, parameters, dense, model->lowerPositionLimit,
                                                            model->upperPositionLimit);
        };
        std::cout << ",\"replay\":{\"nominal\":";
        WriteReplayMetrics(replay(nominal));
        std::cout << ",\"optimized_huber\":";
        WriteReplayMetrics(replay(optimized_parameters));
        if (physical_arm) {
            std::cout << ",\"physical_arm\":";
            WriteReplayMetrics(replay(physical_base));
        }
        std::cout << '}';
    }
    if (actuator_terms) {
        std::cout << ",\"optimized_huber_actuator_parameters\":";
        WriteActuatorEstimates(parameter_names, drive_parameters, drive_coordinates, optimized_parameters,
                               &optimized_covariance);
        std::cout << ",\"nonnegative_actuator_projection\":{\"negative_terms_before\":"
                  << drive_projection.negative_before << ",\"candidate\":";
        if (drive_projection.parameters) {
            const auto& projected = *drive_projection.parameters;
            const auto metrics = ValidateParameters(holdout, projected);
            std::cout << "{\"holdout_rmse_nm\":" << metrics.rmse << ",\"parameters\":";
            WriteVector(projected);
            std::cout << ",\"actuator_parameters\":";
            WriteActuatorEstimates(parameter_names, drive_parameters, drive_coordinates, projected, nullptr);
            std::cout << ",\"forward_validation\":";
            WriteForwardMetrics(validation.Evaluate(projected));
            std::cout << '}';
        } else {
            std::cout << "null";
        }
        std::cout << '}';
        ArmRegression rigid_arm(model, false);
        const BaseParameterBasis rigid_basis(rigid_arm.Calibrate(calibration_data).regressor);
        const auto rigid_train = rigid_basis.ReduceProblem(
            rigid_arm.BuildProblem(ReadDatasetCsv((directory / "train_optimized.csv").string())));
        const auto rigid_holdout = rigid_basis.ReduceProblem(rigid_arm.BuildProblem(holdout_data));
        SolverOptions options;
        options.huber_delta = optimized_delta;
        const auto rigid_fit = Solve(rigid_train, options);
        const auto comparison = ValidateParameters(rigid_holdout, rigid_fit.parameters);
        std::cout << ",\"rigid_only_optimized_huber_holdout_rmse_nm\":" << comparison.rmse;
    }
    std::cout << "}\n";
}

Dataset EstimateMovingAverage(const RawDataset& raw, std::size_t window) {
    const auto sample_count = static_cast<std::size_t>(raw.time_seconds.size());
    if (window == 0 || window % 2 == 0 || window > sample_count)
        throw std::invalid_argument("raw smoothing window must be a positive odd number within the dataset length");
    PreprocessOptions options;
    options.smoothing_window = window;
    // Three centered averages and two centered differences extend this far.
    options.trim_front = options.trim_back = 3 * (window / 2) + 2;
    if (options.trim_front > (sample_count - 3) / 2)
        throw std::invalid_argument("raw smoothing leaves fewer than three valid interior samples");
    return PreprocessPositionAndTorque(raw, options);
}

void ProcessRaw(
    const char* input, const char* output, std::size_t size, std::string_view method, double delay_seconds) {
    const auto raw = ReadRawDatasetCsv(input);
    if (delay_seconds != 0. && method != "periodic-fourier" && method != "periodic-fourier-huber")
        throw std::invalid_argument("torque delay alignment requires a periodic Fourier estimator");
    Eigen::VectorXd huber_delta;
    const auto process = [&] {
        if (method == "moving-average") return EstimateMovingAverage(raw, size);
        if (method == "local-cubic") return arm_identification::EstimateLocalCubic(raw, size);
        if (method == "periodic-fourier" || method == "periodic-fourier-huber") {
            auto model = arm_identification::FitPeriodicFourier(raw, size, method == "periodic-fourier-huber");
            huber_delta = std::move(model.huber_delta_rad);
            return arm_identification::ReconstructPeriodicFourier(raw, model, delay_seconds);
        }
        throw std::invalid_argument(
            "raw method must be moving-average, local-cubic, periodic-fourier or periodic-fourier-huber");
    };
    const auto dataset = process();
    WriteDatasetCsv(dataset, output);
    std::cout << "{\"input_samples\":" << raw.time_seconds.size() << ",\"used_samples\":" << dataset.samples.size();
    if (huber_delta.size() != 0) {
        std::cout << ",\"huber_delta_rad\":";
        WriteVector(huber_delta);
    }
    std::cout << "}\n";
}

double MeanHuberLoss(const RegressionProblem& problem, const Eigen::VectorXd& parameters, double delta) {
    const Eigen::VectorXd absolute = (problem.observation - problem.regressor * parameters).cwiseAbs();
    const Eigen::ArrayXd quadratic = absolute.array().min(delta);
    return (.5 * quadratic.square() + delta * (absolute.array() - quadratic)).mean();
}

RawDataset DelaySearchSamples(RawDataset raw) {
    constexpr Eigen::Index kMaximumSearchSamples = 801;
    if (raw.time_seconds.size() <= kMaximumSearchSamples) return raw;
    const Eigen::Index dof = raw.position.cols();
    RawDataset reduced{std::move(raw.joint_names), Eigen::VectorXd(kMaximumSearchSamples),
                       Eigen::MatrixXd(kMaximumSearchSamples, dof), Eigen::MatrixXd(kMaximumSearchSamples, dof)};
    const Eigen::Index span = raw.time_seconds.size() - 1;
    const Eigen::Index intervals = kMaximumSearchSamples - 1;
    for (Eigen::Index row = 0; row < kMaximumSearchSamples; ++row) {
        const Eigen::Index index = row * (span / intervals) + row * (span % intervals) / intervals;
        reduced.time_seconds[row] = raw.time_seconds[index];
        reduced.position.row(row) = raw.position.row(index);
        reduced.torque.row(row) = raw.torque.row(index);
    }
    return reduced;
}

void EstimateTorqueDelay(const char* nominal_urdf,
                         const fs::path& directory,
                         double maximum_delay_ms,
                         std::size_t harmonics,
                         std::string_view method,
                         double huber_delta,
                         bool actuator_terms) {
    if (method != "periodic-fourier" && method != "periodic-fourier-huber")
        throw std::invalid_argument("torque delay search requires a periodic Fourier estimator");
    if (maximum_delay_ms <= 0. || huber_delta <= 0.)
        throw std::invalid_argument("torque delay search requires a positive bound and Huber threshold");
    ArmRegression arm(LoadModel(nominal_urdf), actuator_terms);
    const auto calibration = arm.Calibrate(ReadDatasetCsv((directory / "calibration.csv").string()));
    const BaseParameterBasis basis(calibration.regressor);
    const auto baseline = DelaySearchSamples(ReadRawDatasetCsv((directory / "train_baseline_raw.csv").string()));
    const auto optimized = DelaySearchSamples(ReadRawDatasetCsv((directory / "train_optimized_raw.csv").string()));
    const bool robust = method == "periodic-fourier-huber";
    const auto baseline_model = arm_identification::FitPeriodicFourier(baseline, harmonics, robust);
    const auto optimized_model = arm_identification::FitPeriodicFourier(optimized, harmonics, robust);
    SolverOptions options;
    options.huber_delta = huber_delta;
    options.max_iterations = 12;
    options.convergence_tolerance = 1e-6;
    const auto build = [&](const RawDataset& raw, const arm_identification::PeriodicFourierModel& model,
                           double seconds) {
        const auto states = arm_identification::ReconstructPeriodicFourier(raw, model, seconds);
        return basis.ReduceProblem(arm.BuildProblem(states));
    };
    constexpr int kSearchSteps = 20;
    constexpr int kCoarseStep = 5;
    std::size_t evaluations = 0;
    const auto milliseconds = [&](int index) { return (static_cast<double>(index) / kSearchSteps) * maximum_delay_ms; };
    const auto score = [&](int index) {
        ++evaluations;
        const double seconds = milliseconds(index) / 1000.;
        const auto a = build(baseline, baseline_model, seconds);
        const auto b = build(optimized, optimized_model, seconds);
        const auto fit_a = Solve(a, options);
        const auto fit_b = Solve(b, options);
        if (fit_a.rank != basis.parameter_count() || fit_b.rank != basis.parameter_count())
            throw std::runtime_error("torque delay search lost identifiable directions");
        // Transfer each trajectory's fitted model to the other training trajectory.
        const double loss =
            .5 * (MeanHuberLoss(a, fit_b.parameters, huber_delta) + MeanHuberLoss(b, fit_a.parameters, huber_delta));
        if (!std::isfinite(loss)) throw std::runtime_error("torque delay search loss is non-finite");
        return loss;
    };
    const double zero_loss = score(0);
    int best_index = 0;
    double best_loss = zero_loss;
    const auto test = [&](int index) {
        const double loss = score(index);
        if (loss < best_loss) {
            best_loss = loss;
            best_index = index;
        }
    };
    for (int index = -kSearchSteps; index <= kSearchSteps; index += kCoarseStep)
        if (index != 0) test(index);
    const int lowest_index = std::max(-kSearchSteps, best_index - kCoarseStep + 1);
    const int highest_index = std::min(kSearchSteps, best_index + kCoarseStep - 1);
    for (int index = lowest_index; index <= highest_index; ++index)
        if (index % kCoarseStep != 0) test(index);

    constexpr double kMinimumRelativeImprovement = .02;
    const bool accepted = best_index != 0 && best_loss <= (1. - kMinimumRelativeImprovement) * zero_loss;
    std::cout << "{\"delay_ms\":" << (accepted ? milliseconds(best_index) : 0.)
              << ",\"best_trial_delay_ms\":" << milliseconds(best_index) << ",\"tested_delays\":" << evaluations
              << ",\"baseline_samples\":" << baseline.time_seconds.size()
              << ",\"optimized_samples\":" << optimized.time_seconds.size()
              << ",\"zero_delay_huber_loss\":" << zero_loss << ",\"best_trial_huber_loss\":" << best_loss
              << ",\"minimum_relative_improvement\":" << kMinimumRelativeImprovement
              << ",\"maximum_delay_ms\":" << maximum_delay_ms << ",\"huber_delta_nm\":" << huber_delta << "}\n";
}

std::size_t ParseCount(std::string_view value, const char* label) {
    std::size_t count = 0;
    const auto* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, count);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument(std::string(label) + " must be a nonnegative integer");
    return count;
}

double ParseFinite(const char* value, const char* label) {
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (value[end] != '\0' || !std::isfinite(result))
        throw std::invalid_argument(std::string(label) + " must be finite");
    return result;
}

double ParseNonnegative(const char* value, const char* label) {
    const double result = ParseFinite(value, label);
    if (result < 0.) throw std::invalid_argument(std::string(label) + " must be nonnegative");
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        const auto mode = argc > 1 ? std::string(argv[1]) : "";
        if (argc >= 4 && mode == "design") {
            const bool actuators = argc >= 7 && std::string_view(argv[argc - 1]) == "actuators";
            const int design_arguments = argc - actuators;
            if (design_arguments != 4 && design_arguments != 6 && design_arguments != 8)
                throw std::invalid_argument("design requires samples and evaluations before optional periods");
            const auto samples = design_arguments >= 6 ? ParseCount(argv[4], "sample count") : kTrajectorySamples;
            const auto evaluations =
                design_arguments >= 6 ? ParseCount(argv[5], "evaluation budget") : kDesignEvaluations;
            const double training_period =
                design_arguments == 8 ? ParseNonnegative(argv[6], "training period") : kTrainingPeriodSeconds;
            const double holdout_period =
                design_arguments == 8 ? ParseNonnegative(argv[7], "holdout period") : kHoldoutPeriodSeconds;
            Design(argv[2], argv[3], samples, evaluations, training_period, holdout_period, actuators);
        } else if (argc >= 5 && mode == "fit") {
            std::optional<double> delta;
            if (std::string_view(argv[4]) != "auto") {
                delta = ParseNonnegative(argv[4], "Huber delta");
                if (*delta == 0.) throw std::invalid_argument("Huber delta must be positive");
            }
            bool estimate_payload = false;
            bool estimate_physical_arm = false;
            int first_drive_argument = 5;
            for (; first_drive_argument < argc; ++first_drive_argument) {
                const std::string_view flag(argv[first_drive_argument]);
                if (flag == "payload")
                    estimate_payload = true;
                else if (flag == "physical")
                    estimate_physical_arm = true;
                else
                    break;
            }
            std::vector<double> drive_parameters;
            for (int index = first_drive_argument; index < argc; ++index)
                drive_parameters.push_back(ParseNonnegative(argv[index], "drive parameter"));
            Fit(argv[2], argv[3], delta, drive_parameters, estimate_payload, estimate_physical_arm);
        } else if ((argc == 8 || argc == 9) && mode == "align") {
            const bool actuators = argc == 9 && std::string_view(argv[8]) == "actuators";
            if (argc == 9 && !actuators) throw std::invalid_argument("expected actuators as the final alignment flag");
            EstimateTorqueDelay(argv[2], argv[3], ParseNonnegative(argv[4], "maximum delay"),
                                ParseCount(argv[5], "Fourier harmonics"), argv[6],
                                ParseNonnegative(argv[7], "Huber delta"), actuators);
        } else if (argc >= 5 && argc <= 7 && mode == "preprocess") {
            ProcessRaw(argv[2], argv[3], ParseCount(argv[4], "raw estimator size"),
                       argc >= 6 ? argv[5] : "moving-average",
                       argc == 7 ? ParseFinite(argv[6], "torque delay in milliseconds") / 1000. : 0.);
        } else {
            std::cerr << "usage: arm_identification design NOMINAL_URDF DIRECTORY "
                         "[SAMPLES EVALUATIONS [TRAIN_PERIOD_S HOLDOUT_PERIOD_S] [actuators]]\n"
                         "       arm_identification fit NOMINAL_URDF DIRECTORY HUBER_DELTA_NM_OR_AUTO "
                         "[payload] [physical] [J1 B1 ...]\n"
                         "       arm_identification align NOMINAL_URDF DIRECTORY MAX_DELAY_MS HARMONICS FOURIER_METHOD "
                         "HUBER_DELTA_NM [actuators]\n"
                         "       arm_identification preprocess RAW_CSV PROCESSED_CSV WINDOW_OR_HARMONICS "
                         "[METHOD [TORQUE_DELAY_MS]]\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "arm_identification: " << error.what() << '\n';
        return 1;
    }
}
