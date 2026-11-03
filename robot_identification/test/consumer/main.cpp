#include <robot_identification/base_parameters.h>
#include <robot_identification/experiment_design.h>
#include <robot_identification/identification.h>

int main() {
    robot_identification::RegressionProblem problem;
    problem.regressor = Eigen::MatrixXd::Identity(2, 2);
    problem.observation = Eigen::Vector2d(1.0, 2.0);
    problem.row_weights = Eigen::Vector2d::Ones();
    problem.sample_count = 2;
    problem.dof = 1;
    const auto result = robot_identification::Solve(problem);
    const robot_identification::BaseParameterBasis basis(problem.regressor);
    const auto reduced = basis.ReduceProblem(problem);
    const robot_identification::HarmonicDesign design{Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
                                                      Eigen::MatrixXd::Zero(1, 1), .1};
    robot_identification::ExcitationSample sample;
    design.trajectory().EvaluateInto(0., sample);
    return result.parameters.isApprox(problem.observation, 1e-12) && reduced.regressor.rows() == 2 &&
                   design.trajectory().dof() == 1 && sample.position.size() == 1 && sample.position[0] == 0. &&
                   sample.velocity[0] > 0.
               ? 0
               : 1;
}
