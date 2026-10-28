#include "robot_identification/identification.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

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
        dataset.samples.push_back({time, Eigen::VectorXd::Constant(1, position), Eigen::VectorXd::Constant(1, velocity),
                                   Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, torque)});
    }
    return dataset;
}

void TestHarmonicScaleRobustness() {
    using namespace robot_identification;
    const auto trajectory = [](double amplitude, double angular_frequency) {
        return HarmonicTrajectory(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, amplitude),
                                  Eigen::MatrixXd::Zero(1, 1), angular_frequency / (2.0 * std::acos(-1.0)));
    };
    // Sparse samples can miss every peak; the conservative envelope must not
    // underflow to zero and admit a position-unsafe trajectory.
    const auto tiny = trajectory(1e-200, 2.0 * std::acos(-1.0));
    ExcitationLimits limits{Eigen::VectorXd::Constant(1, -1e-210), Eigen::VectorXd::Constant(1, 1e-210),
                            Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1)};
    if (ValidateExcitationTrajectory(tiny, limits, 1.0, 1.0).passed) {
        throw std::runtime_error("underflowed harmonic envelope admitted an unsafe trajectory");
    }
    for (const auto scales : {std::pair<double, double>{1e300, 1e-150}, {1e-300, 1e110}}) {
        HarmonicTrajectory value(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, scales.first),
                                 Eigen::MatrixXd::Constant(1, 1, scales.first),
                                 scales.second / (2.0 * std::acos(-1.0)));
        const auto sample = value.Evaluate(0);
        const double velocity = scales.first * scales.second;
        const double acceleration = velocity * scales.second;
        const double jerk = acceleration * scales.second;
        if (std::abs(sample.velocity[0] / velocity - 1.0) > 1e-12 ||
            std::abs(sample.acceleration[0] / -acceleration - 1.0) > 1e-12 ||
            std::abs(sample.jerk[0] / -jerk - 1.0) > 1e-12) {
            throw std::runtime_error("representable harmonic derivatives changed with numeric scale");
        }
        const auto bounds = value.ConservativeBounds();
        if (!bounds.maximum_position.allFinite() ||
            std::abs(bounds.maximum_position[0] / scales.first - std::sqrt(2.0)) > 1e-12 ||
            std::abs(bounds.maximum_absolute_jerk[0] / jerk - std::sqrt(2.0)) > 1e-12) {
            throw std::runtime_error("representable harmonic bounds changed with numeric scale");
        }
    }
    const auto wide = trajectory(1e308, 1e-110);
    limits = {Eigen::VectorXd::Constant(1, -1.5e308), Eigen::VectorXd::Constant(1, 1.5e308),
              Eigen::VectorXd::Constant(1, 1.5e198), Eigen::VectorXd::Constant(1, 1.5e88),
              Eigen::VectorXd::Constant(1, 1.5e-22)};
    const auto result = ValidateExcitationTrajectory(wide, limits, 1.0, 1.0);
    if (!result.passed || !result.conservative_position_envelope.allFinite() ||
        std::abs(result.conservative_position_envelope[0] / 1e308 - 1.0) > 1e-12) {
        throw std::runtime_error("representable half-width overflowed excitation diagnostics");
    }
}

void TestMultiharmonicDerivativesAtZero() {
    Eigen::Matrix<double, 2, 3> sine, cosine;
    sine << 1, 2, 3, -2, 1, 4;
    cosine << 4, -1, 2, 1, 3, -2;
    robot_identification::HarmonicTrajectory trajectory(Eigen::Vector2d(0.5, -0.25), sine, cosine,
                                                        1.0 / (2.0 * std::acos(-1.0)));
    const auto sample = trajectory.Evaluate(0);
    if (!sample.position.isApprox(Eigen::Vector2d(5.5, 1.75), 1e-14) ||
        !sample.velocity.isApprox(Eigen::Vector2d(14, 12), 1e-14) ||
        !sample.acceleration.isApprox(Eigen::Vector2d(-18, 5), 1e-14) ||
        !sample.jerk.isApprox(Eigen::Vector2d(-98, -114), 1e-14)) {
        throw std::runtime_error("multiharmonic analytical derivatives are incorrect");
    }
}

void TestRobustWeightScales() {
    robot_identification::RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.regressor = Eigen::MatrixXd::Ones(4, 1);
    problem.row_weights = Eigen::VectorXd::Constant(4, 1e300);
    robot_identification::SolverOptions options;
    options.huber_delta = 1e-300;
    options.max_iterations = 2;
    for (double magnitude : {1e20, 1e100}) {
        problem.observation = Eigen::Vector4d(magnitude, -magnitude, magnitude, -magnitude);
        const auto result = robot_identification::Solve(problem, options);
        if (result.rank != 1 || !result.covariance.allFinite() || result.final_row_weights.minCoeff() <= 0 ||
            (result.final_row_weights.array() * magnitude - 1.0).abs().maxCoeff() > 1e-12) {
            throw std::runtime_error("representable robust weights were lost before multiplying by base weights");
        }
    }
}

void TestWeightedSolves() {
    robot_identification::RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.regressor.resize(4, 2);
    problem.regressor << 1, 0, 0, 1, 1, 0, 0, 1;
    problem.observation = Eigen::Vector4d(1, 2, 3, 4);
    problem.row_weights = Eigen::Vector4d(1, 2, 3, 4);
    for (double ridge : {0.0, 2.0}) {
        robot_identification::SolverOptions options;
        options.ridge = ridge;
        const auto result = robot_identification::Solve(problem, options);
        const Eigen::Vector2d expected(10.0 / (4.0 + ridge), 20.0 / (6.0 + ridge));
        if (!result.parameters.isApprox(expected, 1e-12) || result.rank != 2 ||
            std::abs(result.condition_number - std::sqrt(1.5)) > 1e-12 ||
            !result.final_row_weights.isApprox(problem.row_weights, 0.0)) {
            throw std::runtime_error("weighted/ridge solution or measured-data diagnostics mismatch");
        }
    }

    // An underdetermined solve must retain the minimum-norm solution and
    // diagnose unobservable parameters even when ridge is enabled.
    problem.sample_count = 2;
    problem.regressor.resize(2, 3);
    problem.regressor << 1, 0, 1, 0, 1, 0;
    problem.observation = Eigen::Vector2d(4, 3);
    problem.row_weights = Eigen::Vector2d(1, 4);
    for (double ridge : {0.0, 0.5}) {
        robot_identification::SolverOptions options;
        options.ridge = ridge;
        const auto result = robot_identification::Solve(problem, options);
        const Eigen::Vector3d expected(4.0 / (2.0 + ridge), 12.0 / (4.0 + ridge), 4.0 / (2.0 + ridge));
        if (!result.parameters.isApprox(expected, 1e-12) || result.rank != 2 || !std::isinf(result.condition_number)) {
            throw std::runtime_error("underdetermined minimum-norm/ridge solve mismatch");
        }
    }

    problem.dof = 2;
    problem.sample_count = std::numeric_limits<std::size_t>::max() / 2 + 2;
    if (problem.Validate()) throw std::runtime_error("overflowed regression dimensions were accepted");

    problem.dof = 1;
    problem.sample_count = 2;
    problem.regressor(0, 0) = std::numeric_limits<double>::max();
    problem.row_weights[0] = 4.0;
    bool rejected = false;
    try {
        (void)robot_identification::Solve(problem);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("overflowed weighted data reached the SVD");
}

void TestPreprocessingBounds() {
    robot_identification::RawDataset raw;
    raw.joint_names = {"joint"};
    raw.time_seconds = Eigen::VectorXd::LinSpaced(5, 0.0, 4.0);
    raw.position = Eigen::MatrixXd::Zero(5, 1);
    raw.torque = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);
    robot_identification::PreprocessOptions options;
    options.trim_front = std::numeric_limits<std::size_t>::max();
    options.trim_back = 2;
    bool rejected = false;
    try {
        (void)robot_identification::PreprocessPositionAndTorque(raw, options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("overflowed trim counts were accepted");

    options.trim_front = 0;
    options.trim_back = 0;
    options.smoothing_window = std::numeric_limits<std::size_t>::max();
    options.torque_smoothing_window = options.smoothing_window;
    const auto processed = robot_identification::PreprocessPositionAndTorque(raw, options);
    for (const auto& sample : processed.samples) {
        if (sample.torque[0] != 3.0 || !sample.position.isZero() || !sample.velocity.isZero() ||
            !sample.acceleration.isZero()) {
            throw std::runtime_error("oversized smoothing window did not use the available samples");
        }
    }
}
void TestMetricScales() {
    robot_identification::RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.regressor = Eigen::MatrixXd::Ones(4, 1);
    problem.row_weights = Eigen::VectorXd::Ones(4);
    for (const double magnitude : {0.0, 1e-300, 1.0, 1e200, std::numeric_limits<double>::max()}) {
        problem.observation = Eigen::VectorXd::Constant(4, magnitude);
        const auto metrics = robot_identification::ValidateParameters(problem, Eigen::VectorXd::Zero(1));
        if (!std::isfinite(metrics.rmse) || metrics.rmse != magnitude || metrics.mean_absolute_error != magnitude ||
            metrics.max_absolute_error != magnitude || metrics.normalized_rmse != (magnitude == 0 ? 0.0 : 1.0)) {
            throw std::runtime_error("representable validation metrics changed with numeric scale");
        }
    }
    problem.observation = Eigen::Vector4d(3e200, -4e200, 0, 0);
    const auto mixed = robot_identification::ValidateParameters(problem, Eigen::VectorXd::Zero(1));
    if (std::abs(mixed.rmse / 1e200 - 2.5) > 1e-14 || std::abs(mixed.mean_absolute_error / 1e200 - 1.75) > 1e-14) {
        throw std::runtime_error("scaled mixed-sign validation metrics are incorrect");
    }
    problem.observation.setZero();
    const auto zero_reference = robot_identification::ValidateParameters(problem, Eigen::VectorXd::Ones(1));
    if (zero_reference.rmse != 1.0 || !std::isinf(zero_reference.normalized_rmse)) {
        throw std::runtime_error("zero-reference normalization convention changed");
    }
    problem.observation.setConstant(2.0);
    const auto perfect = robot_identification::ValidateParameters(problem, Eigen::VectorXd::Constant(1, 2.0));
    if (perfect.rmse != 0 || perfect.mean_absolute_error != 0 || perfect.normalized_rmse != 0) {
        throw std::runtime_error("exact predictions did not have zero error");
    }
    problem.regressor.setConstant(std::numeric_limits<double>::max());
    bool rejected = false;
    try {
        (void)robot_identification::ValidateParameters(problem, Eigen::VectorXd::Constant(1, 2));
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("non-finite model prediction was accepted");
}

void TestCovarianceOverflowRejection() {
    robot_identification::RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.row_weights = Eigen::VectorXd::Ones(4);
    for (int stage = 0; stage < 3; ++stage) {
        problem.regressor = Eigen::MatrixXd::Constant(4, 1, stage == 0 ? 1e200 : stage == 1 ? 1.0 : 1e-150);
        const double residual = stage == 0 ? 0.0 : stage == 1 ? 1e200 : 1e10;
        problem.observation = Eigen::Vector4d(residual, -residual, residual, -residual);
        bool rejected = false;
        try {
            (void)robot_identification::Solve(problem);
        } catch (const std::overflow_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("non-finite covariance calculation was accepted");
    }
}

void TestLargeParameterConvergence() {
    robot_identification::RegressionProblem reference;
    reference.sample_count = 4;
    reference.dof = 1;
    reference.row_weights = Eigen::VectorXd::Ones(4);
    reference.regressor = Eigen::MatrixXd::Ones(4, 1);
    reference.observation = Eigen::VectorXd::Constant(4, 1e55);
    reference.observation[3] += 1e48;
    auto scaled = reference;
    scaled.regressor *= 1e-100;
    robot_identification::SolverOptions options;
    options.huber_delta = 1e46;
    options.convergence_tolerance = 1e-12;
    options.max_iterations = 2;
    const auto ordinary = robot_identification::Solve(reference, options);
    const auto large = robot_identification::Solve(scaled, options);
    if (ordinary.converged || large.converged || large.iterations != 2 || !large.covariance.allFinite()) {
        throw std::runtime_error("large parameter norm caused premature IRLS convergence");
    }
    options.max_iterations = 50;
    const auto converged = robot_identification::Solve(scaled, options);
    if (!converged.converged || converged.iterations <= 2 || std::abs(converged.predicted[0] / 1e55 - 1.0) > 1e-9) {
        throw std::runtime_error("large-parameter IRLS did not continue suppressing the outlier");
    }
}

void TestMassMatrixScaleChecks() {
    const Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(1, 2);
    const Eigen::VectorXd parameters = Eigen::VectorXd::Ones(1);
    for (double scale : {1e-200, 1.0, 1e200}) {
        Eigen::Matrix2d mass;
        mass << scale, scale, 0.0, scale;
        auto evaluate = [&]() {
            return robot_identification::ValidateMassMatrices(
                positions, parameters,
                [&](const Eigen::VectorXd&, const Eigen::VectorXd&) -> Eigen::MatrixXd { return mass; }, scale * 0.1);
        };
        const auto asymmetric = evaluate();
        if (asymmetric.passed || asymmetric.failed != 1 || asymmetric.tested != 1) {
            throw std::runtime_error("matrix scale concealed asymmetric mass coefficients");
        }
        mass(0, 1) = mass(1, 0) = scale * 0.25;
        const auto symmetric = evaluate();
        if (!symmetric.passed || symmetric.failed != 0 ||
            std::abs(symmetric.minimum_eigenvalue / scale - 0.75) > 1e-12) {
            throw std::runtime_error("scaled symmetric positive-definite mass matrix was rejected");
        }
    }
    const auto overflow = robot_identification::ValidateMassMatrices(
        positions, parameters, [](const Eigen::VectorXd&, const Eigen::VectorXd&) -> Eigen::MatrixXd {
            Eigen::Matrix2d mass;
            mass << 1.4e308, 7e307, 7e307, 1.4e308;
            return mass;
        });
    if (overflow.passed || overflow.failed != 1) {
        throw std::runtime_error("non-finite mass eigenvalues were accepted");
    }
    for (double diagonal : {0.0, -1.0}) {
        const auto invalid = robot_identification::ValidateMassMatrices(
            positions, parameters, [&](const Eigen::VectorXd&, const Eigen::VectorXd&) -> Eigen::MatrixXd {
                return diagonal * Eigen::Matrix2d::Identity();
            });
        if (invalid.passed || invalid.failed != 1 || invalid.minimum_eigenvalue != diagonal) {
            throw std::runtime_error("non-positive mass eigenvalue diagnostics changed");
        }
    }
}

void TestMassMatrixBatchRecovery() {
    Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(6, 2);
    for (Eigen::Index row = 0; row < positions.rows(); ++row) positions(row, 0) = row;
    const Eigen::VectorXd parameters = Eigen::VectorXd::Constant(1, 7.0);
    int calls = 0;
    const auto result = robot_identification::ValidateMassMatrices(
        positions, parameters,
        [&](const Eigen::VectorXd& position, const Eigen::VectorXd& received) -> Eigen::MatrixXd {
            if (position[0] != calls || position[1] != 0 || received[0] != 7.0) {
                throw std::runtime_error("batch validation changed callback inputs or order");
            }
            const int row = calls++;
            if (row == 1) return Eigen::MatrixXd::Identity(1, 1);
            Eigen::Matrix2d mass = Eigen::Matrix2d::Identity();
            if (row == 0) mass *= 2;
            if (row == 2) mass(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (row == 3) mass(0, 1) = 1;
            if (row == 4) mass(0, 0) = -0.5;
            return mass;
        });
    if (calls != 6 || result.tested != 6 || result.failed != 4 || result.passed || result.minimum_eigenvalue != -0.5) {
        throw std::runtime_error("batch validation lost failures or retained stale eigensolver state");
    }
}

}  // namespace

int main() {
    TestRobustWeightScales();
    TestHarmonicScaleRobustness();
    TestMultiharmonicDerivativesAtZero();
    TestMetricScales();
    TestCovarianceOverflowRejection();
    TestLargeParameterConvergence();
    TestMassMatrixScaleChecks();
    TestMassMatrixBatchRecovery();
    TestWeightedSolves();
    TestPreprocessingBounds();
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
    const auto safe_excitation = robot_identification::ValidateExcitationTrajectory(excitation, excitation_limits,
                                                                                    excitation.period_seconds(), 0.001);
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
                                                 Eigen::VectorXd::Constant(1, -0.5), Eigen::VectorXd::Constant(1, 1.25),
                                                 Eigen::VectorXd::Zero(1)};
    const Eigen::MatrixXd actuator_block = actuator.Compute(actuator_sample);
    if (actuator.parameter_count() != 7 || actuator.parameter_names().size() != 7 || actuator_block.rows() != 1 ||
        actuator_block.cols() != 7 || actuator_block(0, 3) != 1.25 || actuator_block(0, 4) != -0.5 ||
        actuator_block(0, 5) != -1.0 || actuator_block(0, 6) != 1.0) {
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
    auto reweighted_problem = dirty_problem;
    reweighted_problem.row_weights = robust.final_row_weights;
    if (!robot_identification::Solve(reweighted_problem).parameters.isApprox(robust.parameters, 1e-12)) {
        throw std::runtime_error("IRLS result does not match its reported final weights");
    }
    for (std::size_t limit : {std::size_t{1}, std::size_t{2}}) {
        auto limited_options = robust_options;
        limited_options.max_iterations = limit;
        limited_options.convergence_tolerance = 1e-30;
        const auto limited = robot_identification::Solve(dirty_problem, limited_options);
        reweighted_problem.row_weights = limited.final_row_weights;
        if (limited.iterations != limit || limited.converged ||
            !robot_identification::Solve(reweighted_problem).parameters.isApprox(limited.parameters, 1e-12) ||
            (limit == 1 && !limited.final_row_weights.isApprox(dirty_problem.row_weights, 0))) {
            throw std::runtime_error("iteration-limited IRLS changed the returned solve or reported weights");
        }
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
    const auto processed =
        robot_identification::PreprocessPositionAndTorque({"joint"}, time, position, torque, {1, 1, 1});
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
