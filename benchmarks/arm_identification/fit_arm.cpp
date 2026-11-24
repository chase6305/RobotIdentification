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
#include "raw_preprocessing.h"

namespace {
namespace fs = std::filesystem;
using namespace robot_identification;
using robot_identification_pinocchio::PinocchioRegressor;

constexpr std::size_t kTrajectorySamples = 401;
constexpr std::size_t kDesignEvaluations = 240;
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

class ArmRegression {
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

    [[nodiscard]] const auto& backend() const { return backend_; }
    [[nodiscard]] const PinocchioRegressor& rigid() const { return *rigid_; }

    [[nodiscard]] RegressionProblem BuildProblem(const Dataset& dataset) const {
        if (backend_ == rigid_) return rigid_->BuildProblem(dataset);
        if (dataset.joint_names != JointNames(rigid_->model()))
            throw std::invalid_argument("dataset joint names do not match the loaded arm");
        return BuildRegressionProblem(dataset, *backend_);
    }

private:
    std::shared_ptr<const PinocchioRegressor> rigid_;
    std::shared_ptr<const RegressorBackend> backend_;
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
            bool actuator_terms) {
    if (samples < 3 || evaluations == 0)
        throw std::invalid_argument("design requires at least three samples and one evaluation");
    const auto model = LoadModel(nominal_urdf);
    const ArmRegression arm(model, actuator_terms);
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
    const auto full = arm.BuildProblem(calibration);
    const BaseParameterBasis basis(full.regressor);
    const BaseParameterRegressor base(arm.backend(), basis);
    const auto baseline = SeedDesign(limits, false);
    const auto holdout = SeedDesign(limits, true);
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

Eigen::VectorXd WriteFit(const RegressionProblem& train,
                         const RegressionProblem& holdout,
                         const Eigen::VectorXd& oracle,
                         double huber_delta,
                         const arm_identification::ForwardValidation& validation) {
    SolverOptions options;
    options.huber_delta = huber_delta;
    auto result = Solve(train, options);
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
    std::cout << ",\"forward_validation\":";
    WriteForwardMetrics(validation.Evaluate(result.parameters));
    std::cout << '}';
    return std::move(result.parameters);
}

void Fit(const char* nominal_urdf,
         const fs::path& directory,
         double huber_delta,
         const std::vector<double>& drive_parameters) {
    const auto model = LoadModel(nominal_urdf);
    const bool actuator_terms = !drive_parameters.empty();
    if (actuator_terms && drive_parameters.size() != static_cast<std::size_t>(2 * model->nv))
        throw std::invalid_argument("expected one armature and damping pair per joint");
    const ArmRegression arm(model, actuator_terms);
    const auto build = [&](const char* filename) {
        return arm.BuildProblem(ReadDatasetCsv((directory / filename).string()));
    };
    const auto calibration_data = ReadDatasetCsv((directory / "calibration.csv").string());
    const auto calibration = arm.BuildProblem(calibration_data);
    const BaseParameterBasis basis(calibration.regressor);
    const auto holdout_data = ReadDatasetCsv((directory / "holdout.csv").string());
    const auto full_holdout = arm.BuildProblem(holdout_data);
    const auto holdout = basis.ReduceProblem(full_holdout);
    const auto nominal =
        basis.ToBaseParameters(DynamicParameters(*model, std::vector<double>(drive_parameters.size())));
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
    const BaseParameterRegressor base(arm.backend(), basis);
    const arm_identification::ForwardValidation validation(base, calibration_data, holdout_data);

    std::cout << "{\"pinocchio_version\":" << std::quoted(PINOCCHIO_VERSION)
              << ",\"full_parameter_count\":" << basis.full_parameter_count()
              << ",\"base_parameter_count\":" << basis.parameter_count()
              << ",\"calibration_reconstruction_error\":" << basis.calibration_error()
              << ",\"holdout_reconstruction_error\":" << basis.ReconstructionError(full_holdout.regressor)
              << ",\"nominal_holdout_rmse_nm\":" << nominal_rmse
              << ",\"oracle_holdout_max_absolute_error_nm\":" << engine_error
              << ",\"reduced_oracle_max_difference_nm\":" << reduction_error << ",\"parameter_names\":[";
    const auto parameter_names = arm.backend()->parameter_names();
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
    Eigen::VectorXd optimized_parameters;
    for (const auto& name : {"baseline", "optimized"}) {
        const auto filename = std::string("train_") + name + ".csv";
        const auto train = basis.ReduceProblem(build(filename.c_str()));
        std::cout << ",\"" << name << "\":{\"ols\":";
        WriteFit(train, holdout, oracle, 0., validation);
        std::cout << ",\"huber\":";
        auto parameters = WriteFit(train, holdout, oracle, huber_delta, validation);
        std::cout << '}';
        if (std::string_view(name) == "optimized") {
            const Eigen::VectorXd prediction = holdout.regressor * parameters;
            auto predicted_data = holdout_data;
            const auto dof = static_cast<Eigen::Index>(predicted_data.dof());
            for (std::size_t sample = 0; sample < predicted_data.samples.size(); ++sample)
                predicted_data.samples[sample].torque = prediction.segment(sample * dof, dof);
            WriteDatasetCsv(predicted_data, (directory / "holdout_predicted_optimized_huber.csv").string());
            optimized_parameters = std::move(parameters);
        }
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
        std::cout << '}';
    }
    if (actuator_terms) {
        const auto build_rigid = [&](const char* filename) {
            return arm.rigid().BuildProblem(ReadDatasetCsv((directory / filename).string()));
        };
        const BaseParameterBasis rigid_basis(build_rigid("calibration.csv").regressor);
        const auto rigid_train = rigid_basis.ReduceProblem(build_rigid("train_optimized.csv"));
        const auto rigid_holdout = rigid_basis.ReduceProblem(build_rigid("holdout.csv"));
        SolverOptions options;
        options.huber_delta = huber_delta;
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

void ProcessRaw(const char* input, const char* output, std::size_t size, std::string_view method) {
    const auto raw = ReadRawDatasetCsv(input);
    const auto process = [&] {
        if (method == "moving-average") return EstimateMovingAverage(raw, size);
        if (method == "local-cubic") return arm_identification::EstimateLocalCubic(raw, size);
        if (method == "periodic-fourier") return arm_identification::EstimatePeriodicFourier(raw, size);
        throw std::invalid_argument("raw method must be moving-average, local-cubic or periodic-fourier");
    };
    const auto dataset = process();
    WriteDatasetCsv(dataset, output);
    std::cout << "{\"input_samples\":" << raw.time_seconds.size() << ",\"used_samples\":" << dataset.samples.size()
              << "}\n";
}

std::size_t ParseCount(std::string_view value, const char* label) {
    std::size_t count = 0;
    const auto* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, count);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument(std::string(label) + " must be a nonnegative integer");
    return count;
}

double ParseNonnegative(const char* value, const char* label) {
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (value[end] != '\0' || !std::isfinite(result) || result < 0.)
        throw std::invalid_argument(std::string(label) + " must be finite and nonnegative");
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        const auto mode = argc > 1 ? std::string(argv[1]) : "";
        if ((argc == 4 || argc == 6 || argc == 7) && mode == "design") {
            const auto samples = argc >= 6 ? ParseCount(argv[4], "sample count") : kTrajectorySamples;
            const auto evaluations = argc >= 6 ? ParseCount(argv[5], "evaluation budget") : kDesignEvaluations;
            if (argc == 7 && std::string_view(argv[6]) != "actuators")
                throw std::invalid_argument("design only supports optional actuator terms");
            Design(argv[2], argv[3], samples, evaluations, argc == 7);
        } else if (argc >= 5 && mode == "fit") {
            const double delta = ParseNonnegative(argv[4], "Huber delta");
            if (delta == 0.) throw std::invalid_argument("Huber delta must be positive");
            std::vector<double> drive_parameters;
            for (int index = 5; index < argc; ++index)
                drive_parameters.push_back(ParseNonnegative(argv[index], "drive parameter"));
            Fit(argv[2], argv[3], delta, drive_parameters);
        } else if ((argc == 5 || argc == 6) && mode == "preprocess") {
            ProcessRaw(argv[2], argv[3], ParseCount(argv[4], "raw estimator size"),
                       argc == 6 ? argv[5] : "moving-average");
        } else {
            std::cerr << "usage: arm_identification design NOMINAL_URDF DIRECTORY [SAMPLES EVALUATIONS [actuators]]\n"
                         "       arm_identification fit NOMINAL_URDF DIRECTORY HUBER_DELTA_NM [J1 B1 ...]\n"
                         "       arm_identification preprocess RAW_CSV PROCESSED_CSV WINDOW_OR_HARMONICS [METHOD]\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "arm_identification: " << error.what() << '\n';
        return 1;
    }
}
