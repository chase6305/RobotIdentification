#include <cmath>
#include <stdexcept>

#include "robot_identification/identification.h"

namespace {
void RequireNear(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() || !actual.allFinite() ||
        !actual.isApprox(expected, 2e-13)) {
        throw std::runtime_error("covariance does not retain the analytically observable directions");
    }
}

void TestRidgeDirections() {
    using namespace robot_identification;
    RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.regressor = Eigen::MatrixXd::Zero(4, 3);
    problem.regressor.col(0).setOnes();
    problem.observation = Eigen::Vector4d(1., 1., 3., 3.);
    problem.row_weights = Eigen::Vector4d::Ones();
    SolverOptions options;
    options.ridge = 1.;
    const auto check = [&](std::size_t measured_rank, const Eigen::Vector3d& parameters,
                           const Eigen::Matrix3d& covariance) {
        const auto result = Solve(problem, options);
        if (result.rank != measured_rank || !result.parameters.isApprox(parameters, 5e-13))
            throw std::runtime_error("ridge hid the measured rank or changed the fit");
        RequireNear(result.covariance, covariance);
    };
    // A' A + ridge I is diagonal (5, 1, 1). The residual variance uses
    // four measurements minus the measured rank, giving 116/75.
    Eigen::Matrix3d expected = Eigen::Matrix3d::Zero();
    expected.diagonal() << 116. / 375., 116. / 75., 116. / 75.;
    check(1, Eigen::Vector3d(1.6, 0., 0.), expected);

    // Positive ridge need not retain every direction at the chosen threshold.
    options.ridge = 1e-24;
    expected.setZero();
    expected(0, 0) = 1. / 3.;
    check(1, Eigen::Vector3d(2., 0., 0.), expected);

    options.ridge = 1.;
    options.relative_rank_tolerance = 2.;
    check(0, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Zero());

    // No measured direction is observable, while ridge retains three modes.
    options.relative_rank_tolerance = 1e-10;
    problem.regressor.setZero();
    check(0, Eigen::Vector3d::Zero(), 5. * Eigen::Matrix3d::Identity());
}
}  // namespace

int main() {
    TestRidgeDirections();
    robot_identification::RegressionProblem problem;
    problem.sample_count = 4;
    problem.dof = 1;
    problem.row_weights = Eigen::Vector4d(4, 9, 1, 1);
    problem.regressor = Eigen::MatrixXd::Zero(4, 2);
    problem.regressor(0, 0) = 1;
    problem.regressor(1, 1) = 1e-8;
    problem.observation = Eigen::Vector4d(0, 0, 1, -1);
    Eigen::Matrix2d expected = Eigen::Matrix2d::Zero();
    expected(0, 0) = 0.25;
    expected(1, 1) = 1.0 / 9e-16;
    const auto ordinary = robot_identification::Solve(problem);
    if (ordinary.rank != 2) throw std::runtime_error("weak direction was not observable");
    RequireNear(ordinary.covariance, expected);

    robot_identification::SolverOptions options;
    options.ridge = 1e-14;
    expected(0, 0) = 1.0 / (4.0 + options.ridge);
    expected(1, 1) = 1.0 / (9e-16 + options.ridge);
    RequireNear(robot_identification::Solve(problem, options).covariance, expected);

    Eigen::Matrix2d rotation;
    rotation << 0.6, -0.8, 0.8, 0.6;
    problem.regressor = (problem.regressor * rotation).eval();
    RequireNear(robot_identification::Solve(problem, options).covariance, rotation.transpose() * expected * rotation);

    // With no ridge, a truncated direction has the minimum-norm pseudoinverse
    // convention. A finite covariance is not evidence of full observability.
    problem.regressor = Eigen::MatrixXd::Zero(4, 2);
    problem.regressor(0, 0) = 1;
    const auto deficient = robot_identification::Solve(problem);
    expected.setZero();
    expected(0, 0) = (2.0 / 3.0) / 4.0;
    if (deficient.rank != 1) throw std::runtime_error("rank deficiency was hidden");
    RequireNear(deficient.covariance, expected);

    for (double scale : {1e-150, 1.0, 1e150, 1e200}) {
        problem.regressor = Eigen::MatrixXd::Constant(4, 1, scale);
        problem.row_weights.setOnes();
        const auto result = robot_identification::Solve(problem);
        const double standard_error = std::sqrt(2.0 / 3.0) / (2.0 * scale);
        const double variance = standard_error * standard_error;
        if (std::abs(result.covariance(0, 0) - variance) > std::abs(variance) * 2e-13) {
            throw std::runtime_error("representable covariance changed with regressor scale");
        }
    }
    // Uniform row-weight scaling cancels between residual variance and
    // information. Neither intermediate must erase a representable covariance.
    for (double scale : {1e-100, 1.0, 1e100}) {
        problem.regressor = Eigen::MatrixXd::Constant(4, 1, scale);
        problem.observation = Eigen::Vector4d(scale, -scale, scale, -scale);
        for (double weight : {1e-300, 1.0, 1e300}) {
            problem.row_weights.setConstant(weight);
            const auto result = robot_identification::Solve(problem);
            if (result.rank != 1 || std::abs(result.covariance(0, 0) - 1.0 / 3.0) > 1e-13) {
                throw std::runtime_error("uniform weights changed representable covariance");
            }
        }
    }
    return 0;
}
