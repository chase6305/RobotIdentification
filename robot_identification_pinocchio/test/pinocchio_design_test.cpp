#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>
#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <cmath>
#include <iostream>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <random>
#include <stdexcept>

int main() {
    using namespace robot_identification;
    try {
        auto model = std::make_shared<pinocchio::Model>();
        const pinocchio::SE3 placement(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.3, .1, .2));
        const auto first = model->addJoint(0, pinocchio::JointModelRY(), placement, "shoulder");
        const auto second = model->addJoint(first, pinocchio::JointModelRZ(), placement, "elbow");
        for (const auto joint : {first, second})
            model->appendBodyToJoint(
                joint, pinocchio::Inertia(1.2, Eigen::Vector3d(.1, .02, .03), .2 * Eigen::Matrix3d::Identity()),
                pinocchio::SE3::Identity());
        Eigen::VectorXd truth(20);
        truth.head<10>() = model->inertias[first].toDynamicParameters();
        truth.tail<10>() = model->inertias[second].toDynamicParameters();
        const auto backend = std::make_shared<robot_identification_pinocchio::PinocchioRegressor>(model);
        pinocchio::Data independent(*model);
        std::mt19937_64 random(0x20261029);
        std::uniform_real_distribution<double> unit(-1., 1.);
        Dataset calibration;
        calibration.joint_names = {"shoulder", "elbow"};
        for (int index = 0; index < 200; ++index) {
            Sample sample;
            sample.time_seconds = .01 * index;
            sample.position.resize(2);
            sample.velocity.resize(2);
            sample.acceleration.resize(2);
            for (int joint = 0; joint < 2; ++joint) {
                sample.position[joint] = unit(random);
                sample.velocity[joint] = unit(random);
                sample.acceleration[joint] = unit(random);
            }
            sample.torque = pinocchio::rnea(*model, independent, sample.position, sample.velocity, sample.acceleration);
            calibration.samples.push_back(std::move(sample));
        }
        const auto full = backend->BuildProblem(calibration);
        const BaseParameterBasis basis(full.regressor);
        if (basis.parameter_count() == 0 || basis.parameter_count() >= backend->parameter_count())
            throw std::runtime_error("native model did not expose its redundant inertial directions");
        const BaseParameterRegressor base(backend, basis);
        HarmonicDesign seed;
        seed.offset = Eigen::Vector2d::Zero();
        seed.sin_coefficients.resize(2, 3);
        seed.cos_coefficients.resize(2, 3);
        seed.sin_coefficients << .12, .03, -.02, .08, -.04, .01;
        seed.cos_coefficients << .02, -.01, 0., -.04, .02, -.01;
        seed.fundamental_frequency_hz = .15;
        const ExcitationLimits limits{Eigen::Vector2d::Constant(-1.), Eigen::Vector2d::Constant(1.),
                                      Eigen::Vector2d::Constant(2.), Eigen::Vector2d::Constant(4.),
                                      Eigen::Vector2d::Constant(12.)};
        ExcitationDesignOptions options;
        options.duration_seconds = 10.;
        options.sample_count = 101;
        options.max_evaluations = 60;
        const auto designed = OptimizeExcitationDesign(seed, limits, base, options);
        if (!designed.validation.passed || !designed.quality.fully_identifiable() ||
            designed.quality.mean_log_information < designed.initial_quality.mean_log_information)
            throw std::runtime_error("native excitation optimization lost feasibility or identifiability");
        const auto make_data = [&](const HarmonicTrajectory& trajectory) {
            Dataset result;
            result.joint_names = calibration.joint_names;
            for (int index = 0; index < 101; ++index) {
                Sample sample;
                sample.time_seconds = .1 * index;
                const auto values = trajectory.Evaluate(sample.time_seconds);
                sample.position = values.position;
                sample.velocity = values.velocity;
                sample.acceleration = values.acceleration;
                sample.torque =
                    pinocchio::rnea(*model, independent, sample.position, sample.velocity, sample.acceleration);
                result.samples.push_back(std::move(sample));
            }
            return result;
        };
        const auto training = BuildRegressionProblem(make_data(designed.design.trajectory()), base);
        const auto fit = Solve(training);
        seed.fundamental_frequency_hz = .37;
        const auto holdout = BuildRegressionProblem(make_data(seed.trajectory()), base);
        const auto metrics = ValidateParameters(holdout, fit.parameters);
        if (fit.rank != basis.parameter_count() || fit.rmse > 1e-9 || metrics.rmse > 1e-9 ||
            !fit.parameters.isApprox(basis.ToBaseParameters(truth), 1e-8))
            throw std::runtime_error("base-parameter fit did not predict independent native RNEA torque");
        std::cout << "full_parameters=" << backend->parameter_count() << " base_parameters=" << base.parameter_count()
                  << " initial_log_information=" << designed.initial_quality.mean_log_information
                  << " final_log_information=" << designed.quality.mean_log_information
                  << " holdout_rmse=" << metrics.rmse << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
