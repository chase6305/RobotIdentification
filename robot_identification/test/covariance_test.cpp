#include <cmath>
#include <stdexcept>

#include "robot_identification/identification.h"

namespace {
void RequireNear(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected) {
    if (!actual.allFinite() || !actual.isApprox(expected, 2e-13)) {
        throw std::runtime_error("covariance does not retain the analytically observable directions");
    }
}
}  // namespace

int main() {
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
    return 0;
}
