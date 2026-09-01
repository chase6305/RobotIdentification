#include <robot_identification/identification.h>

int main() {
    robot_identification::RegressionProblem problem;
    problem.regressor = Eigen::MatrixXd::Identity(2, 2);
    problem.observation = Eigen::Vector2d(1.0, 2.0);
    problem.row_weights = Eigen::Vector2d::Ones();
    problem.sample_count = 2;
    problem.dof = 1;
    const auto result = robot_identification::Solve(problem);
    return result.parameters.isApprox(problem.observation, 1e-12) ? 0 : 1;
}
