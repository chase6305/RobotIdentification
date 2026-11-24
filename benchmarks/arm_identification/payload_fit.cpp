#include "payload_fit.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace arm_identification {
namespace {
using robot_identification::RegressionProblem;

RegressionProblem SubtractKnownArm(const RegressionProblem& original,
                                   const Eigen::VectorXd& known,
                                   Eigen::Index first_column) {
    return {original.regressor.middleCols(first_column, 10), original.observation - original.regressor * known,
            original.row_weights, original.sample_count, original.dof};
}

Eigen::Matrix3d InertiaAboutOrigin(const Eigen::VectorXd& parameters) {
    Eigen::Matrix3d inertia;
    inertia << parameters[4], parameters[5], parameters[7], parameters[5], parameters[6], parameters[8], parameters[7],
        parameters[8], parameters[9];
    return inertia;
}

void SetInertiaAboutOrigin(Eigen::VectorXd& parameters, const Eigen::Matrix3d& inertia) {
    parameters.segment<6>(4) << inertia(0, 0), inertia(0, 1), inertia(1, 1), inertia(0, 2), inertia(1, 2),
        inertia(2, 2);
}
}  // namespace

PayloadFit FitPayload(const RegressionProblem& training,
                      const RegressionProblem& holdout,
                      const Eigen::VectorXd& known_arm_parameters,
                      Eigen::Index payload_column,
                      double huber_delta) {
    const auto train = SubtractKnownArm(training, known_arm_parameters, payload_column);
    const auto validation = SubtractKnownArm(holdout, known_arm_parameters, payload_column);
    robot_identification::SolverOptions options;
    options.huber_delta = huber_delta;
    const auto fit = robot_identification::Solve(train, options);
    if (fit.rank != 10) throw std::runtime_error("payload inertia is not fully identifiable on this trajectory");

    PayloadFit result;
    result.estimated_mass = fit.parameters[0];
    result.unconstrained_holdout_rmse = robot_identification::ValidateParameters(validation, fit.parameters).rmse;
    if (result.estimated_mass <= 0.) return result;

    const Eigen::Vector3d center = fit.parameters.segment<3>(1) / result.estimated_mass;
    const auto identity = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d parallel_axis =
        result.estimated_mass * (center.squaredNorm() * identity - center * center.transpose());
    const Eigen::Matrix3d inertia = InertiaAboutOrigin(fit.parameters) - parallel_axis;
    const Eigen::Matrix3d second_moment = .5 * inertia.trace() * identity - inertia;
    if (!second_moment.allFinite()) return result;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(second_moment);
    if (eigen.info() != Eigen::Success) return result;

    // A positive centered second moment satisfies the principal inertia triangle.
    const double floor =
        std::max(1e-12, 16 * std::numeric_limits<double>::epsilon() * eigen.eigenvalues().cwiseAbs().maxCoeff());
    const bool projected = eigen.eigenvalues().minCoeff() < floor;
    Eigen::Matrix3d physical_inertia = inertia;
    if (projected) {
        const Eigen::Matrix3d physical_moment =
            eigen.eigenvectors() * eigen.eigenvalues().cwiseMax(floor).asDiagonal() * eigen.eigenvectors().transpose();
        physical_inertia = physical_moment.trace() * identity - physical_moment;
    }
    auto physical_parameters = fit.parameters;
    SetInertiaAboutOrigin(physical_parameters, physical_inertia + parallel_axis);
    result.physical =
        PayloadInertia{result.estimated_mass, center, physical_inertia,
                       robot_identification::ValidateParameters(validation, physical_parameters).rmse, projected};
    return result;
}

}  // namespace arm_identification
