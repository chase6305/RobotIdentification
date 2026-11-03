#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>

#include <cmath>
#include <iostream>
#include <memory>

namespace {
class DependentActuatorModel final : public robot_identification::RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return 3; }
    std::vector<std::string> parameter_names() const override { return {"inertia", "viscous", "coupled"}; }
    Eigen::MatrixXd Compute(const robot_identification::Sample& sample) const override {
        Eigen::MatrixXd result(1, 3);
        const double acceleration = sample.acceleration[0], velocity = sample.velocity[0];
        result << acceleration, velocity, 2. * acceleration + 3. * velocity;
        return result;
    }
};
}  // namespace

int main() {
    using namespace robot_identification;
    const auto model = std::make_shared<DependentActuatorModel>();
    // Independent calibration states, rather than one possibly unexciting motion.
    Eigen::MatrixXd calibration(100, 3);
    Sample sample;
    sample.position = Eigen::VectorXd::Zero(1);
    sample.velocity = sample.acceleration = sample.torque = Eigen::VectorXd::Zero(1);
    for (Eigen::Index row = 0; row < calibration.rows(); ++row) {
        sample.acceleration[0] = std::sin(.31 * row);
        sample.velocity[0] = std::cos(.73 * row);
        calibration.row(row) = model->Compute(sample);
    }
    const BaseParameterBasis basis(calibration);
    const BaseParameterRegressor base(model, basis);
    const HarmonicDesign initial{Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, .05),
                                 Eigen::MatrixXd::Zero(1, 1), 1. / (2. * std::acos(-1.))};
    const ExcitationLimits limits{Eigen::VectorXd::Constant(1, -1.), Eigen::VectorXd::Constant(1, 1.),
                                  Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.),
                                  Eigen::VectorXd::Constant(1, 2.)};
    ExcitationDesignOptions options;
    options.duration_seconds = 2. * std::acos(-1.);
    options.sample_count = 101;
    const auto optimized = OptimizeExcitationDesign(initial, limits, base, options);
    if (!optimized.validation.passed || !optimized.quality.fully_identifiable()) return 1;
    const Eigen::Vector3d truth(2., .3, .4);
    const auto make_data = [&](const HarmonicTrajectory& trajectory) {
        Dataset data;
        data.joint_names = {"joint"};
        for (std::size_t index = 0; index < options.sample_count; ++index) {
            Sample value;
            value.time_seconds = options.duration_seconds * (static_cast<double>(index) / (options.sample_count - 1));
            const auto motion = trajectory.Evaluate(value.time_seconds);
            value.position = motion.position;
            value.velocity = motion.velocity;
            value.acceleration = motion.acceleration;
            value.torque = model->Compute(value) * truth;
            data.samples.push_back(std::move(value));
        }
        return data;
    };
    const auto problem = BuildRegressionProblem(make_data(optimized.design.trajectory()), base);
    const auto result = Solve(problem);
    HarmonicDesign holdout = initial;
    holdout.sin_coefficients(0, 0) = .3;
    holdout.fundamental_frequency_hz *= 1.7;
    const auto metrics =
        ValidateParameters(BuildRegressionProblem(make_data(holdout.trajectory()), base), result.parameters);
    std::cout << "full_parameters=" << model->parameter_count() << " base_parameters=" << base.parameter_count()
              << " initial_log_information=" << optimized.initial_quality.mean_log_information
              << " optimized_log_information=" << optimized.quality.mean_log_information
              << " evaluations=" << optimized.evaluations << " train_rmse=" << result.rmse
              << " holdout_rmse=" << metrics.rmse << '\n';
    // The fitted values describe identifiable combinations; no full inertia
    // vector or physical-feasibility certificate is inferred from this fit.
    return result.rank == base.parameter_count() && result.parameters.isApprox(basis.ToBaseParameters(truth), 1e-10) &&
                   metrics.rmse < 1e-10
               ? 0
               : 1;
}
