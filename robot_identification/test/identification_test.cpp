#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "robot_identification/identification.h"

namespace {
class Backend final : public robot_identification::RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return 3; }
    std::vector<std::string> parameter_names() const override { return {"position", "velocity", "bias"}; }
    Eigen::MatrixXd Compute(const robot_identification::Sample& sample) const override {
        Eigen::MatrixXd block(1, 3);
        block << sample.position[0], sample.velocity[0], 1.0;
        return block;
    }
};

robot_identification::Dataset MakeDataset(bool outlier) {
    robot_identification::Dataset dataset;
    dataset.joint_names = {"joint"};
    for (int i = 0; i < 200; ++i) {
        const double time = 0.01 * i;
        const double position = std::sin(1.7 * time);
        const double velocity = 1.7 * std::cos(1.7 * time);
        double torque = 2.5 * position + 0.4 * velocity - 0.3;
        if (outlier && i == 100) torque += 100.0;
        dataset.samples.push_back({time, Eigen::VectorXd::Constant(1, position),
                                   Eigen::VectorXd::Constant(1, velocity), Eigen::VectorXd::Zero(1),
                                   Eigen::VectorXd::Constant(1, torque)});
    }
    return dataset;
}
}  // namespace

int main() {
    Eigen::Vector2d excitation_offset(0.1, -0.2);
    Eigen::Matrix<double, 2, 1> excitation_sine;
    excitation_sine << 0.2, 0.1;
    const Eigen::Matrix<double, 2, 1> excitation_cosine = Eigen::Matrix<double, 2, 1>::Zero();
    robot_identification::HarmonicTrajectory excitation(excitation_offset, excitation_sine, excitation_cosine, 0.5);
    const auto excitation_at_zero = excitation.Evaluate(0.0);
    const double angular_frequency = std::acos(-1.0);
    if (!excitation_at_zero.position.isApprox(excitation_offset) ||
        !excitation_at_zero.velocity.isApprox(angular_frequency * excitation_sine) ||
        !excitation_at_zero.acceleration.isZero(1e-12)) {
        throw std::runtime_error("harmonic trajectory analytic derivatives mismatch");
    }
    robot_identification::ExcitationLimits excitation_limits;
    excitation_limits.minimum_position = Eigen::Vector2d(-1.0, -1.0);
    excitation_limits.maximum_position = Eigen::Vector2d(1.0, 1.0);
    excitation_limits.maximum_velocity = Eigen::Vector2d::Constant(1.0);
    excitation_limits.maximum_acceleration = Eigen::Vector2d::Constant(3.0);
    excitation_limits.maximum_jerk = Eigen::Vector2d::Constant(7.0);
    const auto safe_excitation = robot_identification::ValidateExcitationTrajectory(
        excitation, excitation_limits, excitation.period_seconds(), 0.001);
    if (!safe_excitation.passed || safe_excitation.sample_count < 1000) {
        throw std::runtime_error("safe harmonic excitation was rejected");
    }
    excitation_limits.maximum_velocity = Eigen::Vector2d::Constant(0.1);
    const auto unsafe_excitation = robot_identification::ValidateExcitationTrajectory(
        excitation, excitation_limits, excitation.period_seconds(), 0.001);
    if (unsafe_excitation.passed || unsafe_excitation.reason != "conservative velocity bound exceeded") {
        throw std::runtime_error("unsafe harmonic excitation was accepted");
    }

    Backend backend;
    auto shared_backend = std::make_shared<Backend>();
    robot_identification::JointActuatorRegressorOptions actuator_options;
    actuator_options.coulomb_velocity_deadband = 0.1;
    robot_identification::JointActuatorRegressor actuator(shared_backend, {"joint"}, actuator_options);
    robot_identification::Sample actuator_sample{0.0, Eigen::VectorXd::Constant(1, 0.2),
                                                  Eigen::VectorXd::Constant(1, -0.5),
                                                  Eigen::VectorXd::Constant(1, 1.25), Eigen::VectorXd::Zero(1)};
    const Eigen::MatrixXd actuator_block = actuator.Compute(actuator_sample);
    if (actuator.parameter_count() != 7 || actuator.parameter_names().size() != 7 ||
        actuator_block.rows() != 1 || actuator_block.cols() != 7 || actuator_block(0, 3) != 1.25 ||
        actuator_block(0, 4) != -0.5 || actuator_block(0, 5) != -1.0 || actuator_block(0, 6) != 1.0) {
        throw std::runtime_error("joint actuator regressor mismatch");
    }
    actuator_sample.velocity[0] = 0.05;
    if (actuator.Compute(actuator_sample)(0, 5) != 0.0) {
        throw std::runtime_error("Coulomb deadband mismatch");
    }
    const auto clean_problem = robot_identification::BuildRegressionProblem(MakeDataset(false), backend);
    const auto clean = robot_identification::Solve(clean_problem);
    const Eigen::Vector3d expected(2.5, 0.4, -0.3);
    if (!clean.converged || clean.rank != 3 || !clean.parameters.isApprox(expected, 1e-9) || clean.rmse > 1e-9) {
        throw std::runtime_error("ordinary least-squares identification mismatch");
    }

    const auto dirty_problem = robot_identification::BuildRegressionProblem(MakeDataset(true), backend);
    robot_identification::SolverOptions robust_options;
    robust_options.huber_delta = 0.05;
    robust_options.max_iterations = 50;
    const auto robust = robot_identification::Solve(dirty_problem, robust_options);
    if (!robust.converged || (robust.parameters - expected).norm() > 0.01 || robust.final_row_weights[100] >= 0.01) {
        throw std::runtime_error("Huber regression did not suppress the injected outlier");
    }
    const auto metrics = robot_identification::ValidateParameters(clean_problem, clean.parameters);
    if (metrics.row_count != 200 || metrics.rmse > 1e-9) throw std::runtime_error("validation metric mismatch");

    const std::string csv_path = "robot_identification_roundtrip.csv";
    const auto clean_dataset = MakeDataset(false);
    robot_identification::WriteDatasetCsv(clean_dataset, csv_path);
    const auto restored = robot_identification::ReadDatasetCsv(csv_path);
    std::remove(csv_path.c_str());
    if (restored.joint_names != clean_dataset.joint_names || restored.samples.size() != clean_dataset.samples.size() ||
        !restored.samples[73].torque.isApprox(clean_dataset.samples[73].torque, 0.0)) {
        throw std::runtime_error("dataset CSV round trip mismatch");
    }

    robot_identification::RawDataset raw;
    raw.joint_names = {"joint"};
    raw.time_seconds = Eigen::VectorXd::LinSpaced(5, 0.0, 0.04);
    raw.position = raw.time_seconds.array().square().matrix();
    raw.torque = Eigen::MatrixXd::Constant(5, 1, 0.25);
    const std::string raw_csv_path = "robot_identification_raw_roundtrip.csv";
    robot_identification::WriteRawDatasetCsv(raw, raw_csv_path);
    const auto restored_raw = robot_identification::ReadRawDatasetCsv(raw_csv_path);
    std::remove(raw_csv_path.c_str());
    if (restored_raw.joint_names != raw.joint_names || !restored_raw.position.isApprox(raw.position, 0.0) ||
        !restored_raw.torque.isApprox(raw.torque, 0.0)) {
        throw std::runtime_error("raw dataset CSV round trip mismatch");
    }
    const std::string crlf_path = "robot_identification_crlf.csv";
    {
        std::ofstream crlf(crlf_path, std::ios::binary | std::ios::trunc);
        crlf << "time_s,q/joint,tau/joint\r\n"
             << "0,0,0\r\n0.01,0.1,0.2\r\n0.02,0.2,0.4\r\n\r\n";
    }
    const auto crlf_raw = robot_identification::ReadRawDatasetCsv(crlf_path);
    std::remove(crlf_path.c_str());
    if (crlf_raw.time_seconds.size() != 3 || crlf_raw.joint_names != raw.joint_names) {
        throw std::runtime_error("CRLF raw dataset was not parsed portably");
    }
    const auto processed_raw = robot_identification::PreprocessPositionAndTorque(restored_raw, {1, 1, 1});
    if (processed_raw.samples.size() != 3) throw std::runtime_error("raw dataset preprocessing mismatch");

    auto deficient_problem = clean_problem;
    deficient_problem.regressor.col(1) = deficient_problem.regressor.col(0);
    const auto deficient = robot_identification::Solve(deficient_problem);
    if (deficient.rank >= 3 || !std::isinf(deficient.condition_number)) {
        throw std::runtime_error("rank-deficient problem was not diagnosed");
    }
    robot_identification::SolverOptions deficient_ridge_options;
    deficient_ridge_options.ridge = 1e-3;
    const auto deficient_with_ridge = robot_identification::Solve(deficient_problem, deficient_ridge_options);
    if (deficient_with_ridge.rank >= 3 || !std::isinf(deficient_with_ridge.condition_number) ||
        !deficient_with_ridge.parameters.allFinite()) {
        throw std::runtime_error("ridge disguised a rank-deficient experiment");
    }

    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(101, 0.0, 1.0);
    Eigen::MatrixXd position(101, 1);
    Eigen::MatrixXd torque = Eigen::MatrixXd::Zero(101, 1);
    for (Eigen::Index i = 0; i < time.size(); ++i) position(i, 0) = time[i] * time[i];
    const auto processed = robot_identification::PreprocessPositionAndTorque({"joint"}, time, position, torque,
                                                                             {1, 1, 1});
    if (processed.samples.size() != 99 || std::abs(processed.samples[49].acceleration[0] - 2.0) > 1e-9) {
        throw std::runtime_error("preprocessing derivative mismatch");
    }
    Eigen::VectorXd irregular_time(7);
    irregular_time << 0.0, 0.011, 0.027, 0.041, 0.068, 0.083, 0.12;
    Eigen::MatrixXd irregular_position(7, 1);
    for (Eigen::Index row = 0; row < irregular_time.size(); ++row) {
        irregular_position(row, 0) = irregular_time[row] * irregular_time[row];
    }
    const auto irregular = robot_identification::PreprocessPositionAndTorque(
        {"joint"}, irregular_time, irregular_position, Eigen::MatrixXd::Zero(7, 1), {1, 0, 0});
    for (Eigen::Index row = 0; row < irregular_time.size(); ++row) {
        if (std::abs(irregular.samples[static_cast<std::size_t>(row)].velocity[0] - 2.0 * irregular_time[row]) >
                1e-12 ||
            std::abs(irregular.samples[static_cast<std::size_t>(row)].acceleration[0] - 2.0) > 1e-10) {
            throw std::runtime_error("nonuniform timestamp differentiation mismatch");
        }
    }

    Eigen::MatrixXd configurations(3, 1);
    configurations << -1.0, 0.0, 1.0;
    const auto physical = robot_identification::ValidateMassMatrices(
        configurations, expected,
        [](const Eigen::VectorXd&, const Eigen::VectorXd&) { return Eigen::MatrixXd::Constant(1, 1, 2.0); });
    if (!physical.passed || physical.tested != 3 || physical.minimum_eigenvalue != 2.0) {
        throw std::runtime_error("mass-matrix validation mismatch");
    }

    auto invalid = MakeDataset(false);
    invalid.samples[0].torque[0] = std::numeric_limits<double>::quiet_NaN();
    std::string reason;
    if (invalid.Validate(&reason)) throw std::runtime_error("non-finite dataset was accepted");
    return 0;
}
