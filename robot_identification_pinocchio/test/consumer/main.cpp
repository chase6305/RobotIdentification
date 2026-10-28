#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <memory>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <type_traits>

static_assert(
    std::is_base_of_v<robot_identification::RegressorBackend, robot_identification_pinocchio::PinocchioRegressor>);

int main() {
    auto model = std::make_shared<pinocchio::Model>();
    const auto joint = model->addJoint(0, pinocchio::JointModelRY(), pinocchio::SE3::Identity(), "joint");
    model->appendBodyToJoint(joint, pinocchio::Inertia::Identity(), pinocchio::SE3::Identity());
    robot_identification_pinocchio::PinocchioRegressor backend(model);
    robot_identification::Sample sample;
    sample.position = Eigen::VectorXd::Zero(1);
    sample.velocity = Eigen::VectorXd::Zero(1);
    sample.acceleration = Eigen::VectorXd::Zero(1);
    sample.torque = Eigen::VectorXd::Zero(1);
    const auto regressor = backend.Compute(sample);
    return backend.parameter_count() == 10 && regressor.rows() == 1 && regressor.cols() == 10 && regressor.allFinite()
               ? 0
               : 1;
}
