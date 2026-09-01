#include <cmath>
#include <iomanip>
#include <iostream>

#include "robot_identification/identification.h"

namespace {
class TwoJointBackend final : public robot_identification::RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return 4; }
    std::vector<std::string> parameter_names() const override {
        return {"joint_0_position", "joint_0_viscous", "joint_1_inertia", "joint_1_bias"};
    }
    Eigen::MatrixXd Compute(const robot_identification::Sample& sample) const override {
        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(2, 4);
        block(0, 0) = sample.position[0];
        block(0, 1) = sample.velocity[0];
        block(1, 2) = sample.acceleration[1];
        block(1, 3) = 1.0;
        return block;
    }
};
}  // namespace

int main() {
    constexpr std::size_t sample_count = 240;
    Eigen::VectorXd time(sample_count);
    Eigen::MatrixXd position(sample_count, 2);
    Eigen::MatrixXd torque(sample_count, 2);
    const Eigen::Vector4d expected(2.0, 0.35, 1.4, -0.2);
    for (std::size_t i = 0; i < sample_count; ++i) {
        const double t = 0.01 * static_cast<double>(i);
        time[static_cast<Eigen::Index>(i)] = t;
        position(static_cast<Eigen::Index>(i), 0) = std::sin(2.0 * t);
        position(static_cast<Eigen::Index>(i), 1) = std::cos(1.3 * t);
    }
    const auto provisional = robot_identification::PreprocessPositionAndTorque(
        {"axis_a", "axis_b"}, time, position, Eigen::MatrixXd::Zero(sample_count, 2), {5, 3, 3});
    for (std::size_t i = 0; i < provisional.samples.size(); ++i) {
        const auto& sample = provisional.samples[i];
        torque(static_cast<Eigen::Index>(i + 3), 0) = expected[0] * sample.position[0] +
                                                     expected[1] * sample.velocity[0];
        torque(static_cast<Eigen::Index>(i + 3), 1) = expected[2] * sample.acceleration[1] + expected[3];
    }
    const auto dataset = robot_identification::PreprocessPositionAndTorque(
        {"axis_a", "axis_b"}, time, position, torque, {5, 6, 6});
    TwoJointBackend backend;
    const auto problem = robot_identification::BuildRegressionProblem(dataset, backend);
    robot_identification::SolverOptions options;
    options.ridge = 1e-10;
    const auto result = robot_identification::Solve(problem, options);
    std::cout << std::fixed << std::setprecision(6) << "parameters=" << result.parameters.transpose()
              << " rmse=" << result.rmse << " rank=" << result.rank
              << " condition=" << result.condition_number << '\n';
    return result.rank == 4 && result.parameters.allFinite() ? 0 : 1;
}
