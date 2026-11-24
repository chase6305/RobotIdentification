#include "physical_arm_fit.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace arm_identification {
namespace {
constexpr Eigen::Index kBodyCoordinates = 10;
constexpr double kMomentFloor = .02;
constexpr double kMinimumPriorWeight = 1e-11;
constexpr double kMaximumPriorWeight = 1e-6;
constexpr double kDriveScale = .05;
constexpr double kSqrtTwo = 1.4142135623730950488;
using BodyVector = Eigen::Matrix<double, kBodyCoordinates, 1>;

Eigen::Matrix4d PseudoInertia(const BodyVector& parameters) {
    Eigen::Matrix3d inertia;
    inertia << parameters[4], parameters[5], parameters[7], parameters[5], parameters[6], parameters[8], parameters[7],
        parameters[8], parameters[9];
    Eigen::Matrix4d pseudo;
    pseudo.topLeftCorner<3, 3>() = .5 * inertia.trace() * Eigen::Matrix3d::Identity() - inertia;
    pseudo.topRightCorner<3, 1>() = parameters.segment<3>(1);
    pseudo.bottomLeftCorner<1, 3>() = parameters.segment<3>(1).transpose();
    pseudo(3, 3) = parameters[0];
    return pseudo;
}

BodyVector PackSymmetric(const Eigen::Matrix4d& matrix) {
    BodyVector coordinates;
    Eigen::Index entry = 0;
    for (Eigen::Index column = 0; column < 4; ++column) {
        coordinates[entry++] = matrix(column, column);
        for (Eigen::Index row = column + 1; row < 4; ++row) coordinates[entry++] = kSqrtTwo * matrix(row, column);
    }
    return coordinates;
}

Eigen::Matrix4d UnpackSymmetric(const BodyVector& coordinates) {
    Eigen::Matrix4d matrix;
    Eigen::Index entry = 0;
    for (Eigen::Index column = 0; column < 4; ++column) {
        matrix(column, column) = coordinates[entry++];
        for (Eigen::Index row = column + 1; row < 4; ++row)
            matrix(row, column) = matrix(column, row) = coordinates[entry++] / kSqrtTwo;
    }
    return matrix;
}

BodyVector DynamicCoordinates(const Eigen::Matrix4d& pseudo) {
    const Eigen::Matrix3d second_moment = pseudo.topLeftCorner<3, 3>();
    const Eigen::Matrix3d inertia = second_moment.trace() * Eigen::Matrix3d::Identity() - second_moment;
    BodyVector parameters;
    parameters << pseudo(3, 3), pseudo.topRightCorner<3, 1>(), inertia(0, 0), inertia(0, 1), inertia(1, 1),
        inertia(0, 2), inertia(1, 2), inertia(2, 2);
    return parameters;
}

Eigen::MatrixXd ParameterMap(const Eigen::VectorXd& nominal, Eigen::Index joint_count) {
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(nominal.size(), nominal.size());
    for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
        const Eigen::Index offset = kBodyCoordinates * joint;
        const Eigen::LLT<Eigen::Matrix4d> factor(PseudoInertia(nominal.segment<kBodyCoordinates>(offset)));
        if (factor.info() != Eigen::Success) throw std::runtime_error("nominal link has nonphysical inertia");
        const Eigen::Matrix4d lower = factor.matrixL();
        // The positive cone is invariant under congruence. Whitening maps
        // the nominal pseudo-inertia to identity and makes the prior unitless.
        for (Eigen::Index column = 0; column < kBodyCoordinates; ++column) {
            const auto basis = UnpackSymmetric(BodyVector::Unit(column));
            map.block<kBodyCoordinates, 1>(offset, offset + column) =
                DynamicCoordinates(lower * basis * lower.transpose());
        }
    }
    map.diagonal().tail(nominal.size() - kBodyCoordinates * joint_count).setConstant(kDriveScale);
    return map;
}

std::optional<double> EvaluateBarrier(const Eigen::VectorXd& point,
                                      Eigen::Index joint_count,
                                      Eigen::VectorXd* gradient = nullptr,
                                      Eigen::MatrixXd* hessian = nullptr) {
    double barrier = 0.;
    for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
        const Eigen::Index offset = kBodyCoordinates * joint;
        Eigen::Matrix4d margin = UnpackSymmetric(point.segment<kBodyCoordinates>(offset));
        margin.diagonal().array() -= kMomentFloor;
        const Eigen::LLT<Eigen::Matrix4d> factor(margin);
        if (factor.info() != Eigen::Success) return std::nullopt;
        const Eigen::Matrix4d lower = factor.matrixL();
        barrier -= 2. * lower.diagonal().array().log().sum();
        if (gradient) {
            const Eigen::Matrix4d inverse = factor.solve(Eigen::Matrix4d::Identity());
            gradient->segment<kBodyCoordinates>(offset) = -PackSymmetric(inverse);
            for (Eigen::Index column = 0; column < kBodyCoordinates; ++column) {
                const auto basis = UnpackSymmetric(BodyVector::Unit(column));
                hessian->block<kBodyCoordinates, 1>(offset, offset + column) = PackSymmetric(inverse * basis * inverse);
            }
        }
    }
    for (Eigen::Index index = kBodyCoordinates * joint_count; index < point.size(); ++index) {
        if (point[index] <= 0.) return std::nullopt;
        barrier -= std::log(point[index]);
        if (gradient) {
            const double inverse = 1. / point[index];
            (*gradient)[index] = -inverse;
            (*hessian)(index, index) = inverse * inverse;
        }
    }
    return std::isfinite(barrier) ? std::optional<double>(barrier) : std::nullopt;
}

bool FollowBarrier(const Eigen::MatrixXd& information,
                   const Eigen::VectorXd& target,
                   const Eigen::VectorXd& prior,
                   Eigen::Index joint_count,
                   double barrier_weight,
                   Eigen::VectorXd& point,
                   std::size_t& iterations) {
    Eigen::VectorXd barrier_gradient(point.size());
    Eigen::MatrixXd barrier_hessian = Eigen::MatrixXd::Zero(point.size(), point.size());
    for (int iteration = 0; iteration < 50; ++iteration) {
        ++iterations;
        const auto barrier = EvaluateBarrier(point, joint_count, &barrier_gradient, &barrier_hessian);
        if (!barrier) return false;
        const Eigen::VectorXd quadratic_gradient = information * (point - prior) - target;
        const Eigen::VectorXd gradient = quadratic_gradient + barrier_weight * barrier_gradient;
        const Eigen::LLT<Eigen::MatrixXd> factor(information + barrier_weight * barrier_hessian);
        if (factor.info() != Eigen::Success) return false;
        const Eigen::VectorXd direction = factor.solve(-gradient);
        const double slope = gradient.dot(direction);
        if (!std::isfinite(slope) || slope > 0.) return false;
        if (-.5 * slope <= std::max(1e-17, 1e-3 * barrier_weight)) return true;

        const double linear_change = quadratic_gradient.dot(direction);
        const double curvature = direction.dot(information * direction);
        bool accepted = false;
        for (double step = 1.; step >= 1e-10; step *= .5) {
            const Eigen::VectorXd candidate = point + step * direction;
            const auto candidate_barrier = EvaluateBarrier(candidate, joint_count);
            if (!candidate_barrier) continue;
            const double change =
                step * (linear_change + .5 * step * curvature) + barrier_weight * (*candidate_barrier - *barrier);
            if (change <= .01 * step * slope) {
                point = candidate;
                accepted = true;
                break;
            }
        }
        if (!accepted) return false;
    }
    return false;
}
}  // namespace

PhysicalArmFit FitPhysicalArm(const robot_identification::RegressionProblem& training,
                              const Eigen::VectorXd& row_weights,
                              const Eigen::VectorXd& nominal,
                              Eigen::Index joint_count,
                              double residual_scale) {
    const Eigen::MatrixXd map = ParameterMap(nominal, joint_count);
    Eigen::VectorXd prior = Eigen::VectorXd::Zero(nominal.size());
    for (Eigen::Index joint = 0; joint < joint_count; ++joint)
        prior.segment<kBodyCoordinates>(kBodyCoordinates * joint) = PackSymmetric(Eigen::Matrix4d::Identity());

    const Eigen::VectorXd weights = row_weights.array().sqrt();
    const Eigen::MatrixXd weighted = training.regressor.array().colwise() * weights.array();
    const double count = static_cast<double>(weighted.rows());
    // Scale the dimensionless prior with inverse sample count and observed torque variance.
    const double prior_weight =
        std::clamp(residual_scale * residual_scale / count, kMinimumPriorWeight, kMaximumPriorWeight);
    Eigen::MatrixXd information = map.transpose() * (weighted.transpose() * weighted / count) * map;
    information.diagonal().array() += prior_weight;
    const Eigen::VectorXd residual = weights.cwiseProduct(training.observation - training.regressor * nominal);
    const Eigen::VectorXd target = map.transpose() * (weighted.transpose() * residual / count);

    PhysicalArmFit result;
    result.prior_weight = prior_weight;
    result.minimum_relative_moment = kMomentFloor;
    const Eigen::LLT<Eigen::MatrixXd> unconstrained_factor(information);
    if (unconstrained_factor.info() == Eigen::Success) {
        const Eigen::VectorXd unconstrained = prior + unconstrained_factor.solve(target);
        if (EvaluateBarrier(unconstrained, joint_count)) {
            result.parameters = map * unconstrained;
            result.converged = true;
            return result;
        }
    }

    Eigen::VectorXd point = prior;
    point.tail(nominal.size() - kBodyCoordinates * joint_count).setConstant(.1);
    result.converged = true;
    const double final_weight = std::min(1e-11, 1e-3 * prior_weight);
    // Solve a sequence of strictly feasible Newton problems with decreasing barriers.
    double barrier_weight = 1e-3;
    while (true) {
        if (!FollowBarrier(information, target, prior, joint_count, barrier_weight, point, result.iterations)) {
            result.converged = false;
            break;
        }
        if (barrier_weight == final_weight) break;
        barrier_weight = std::max(.1 * barrier_weight, final_weight);
    }
    result.parameters = map * point;
    return result;
}

}  // namespace arm_identification
