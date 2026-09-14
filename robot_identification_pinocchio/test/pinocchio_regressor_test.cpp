#include "robot_identification_pinocchio/pinocchio_regressor.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <stdexcept>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "robot_identification_pinocchio_test.urdf";
    {
        std::ofstream output(path, std::ios::trunc);
        output << R"(<robot name="generic_identification_chain">
  <link name="base"/>
  <link name="body"><inertial><origin xyz="0.1 0 0"/><mass value="1.5"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/></inertial></link>
  <joint name="joint" type="revolute"><parent link="base"/><child link="body"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="2"/></joint>
</robot>)";
        if (!output) throw std::runtime_error("failed to create test URDF");
    }

    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(path.string(), *model);
    std::error_code error;
    std::filesystem::remove(path, error);
    robot_identification_pinocchio::PinocchioRegressor backend(model);
    if (backend.parameter_count() != 10 || backend.parameter_names().front() != "rigid_body/joint/mass") {
        throw std::runtime_error("unexpected Pinocchio parameter contract");
    }

    robot_identification::Sample sample;
    sample.position = Eigen::VectorXd::Constant(1, 0.35);
    sample.velocity = Eigen::VectorXd::Constant(1, -0.2);
    sample.acceleration = Eigen::VectorXd::Constant(1, 0.6);
    sample.torque = Eigen::VectorXd::Zero(1);
    const Eigen::MatrixXd regressor = backend.Compute(sample);
    const Eigen::VectorXd parameters = model->inertias[1].toDynamicParameters();
    pinocchio::Data data(*model);
    const Eigen::VectorXd expected =
        pinocchio::rnea(*model, data, sample.position, sample.velocity, sample.acceleration);
    if (!(regressor * parameters).isApprox(expected, 1e-10)) {
        throw std::runtime_error("Pinocchio regressor does not reproduce RNEA torque");
    }

    sample.velocity.resize(2);
    try {
        static_cast<void>(backend.Compute(sample));
    } catch (const std::invalid_argument&) {
        return 0;
    }
    throw std::runtime_error("Pinocchio regressor accepted an invalid sample");
}
