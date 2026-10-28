#include "robot_identification_pinocchio/pinocchio_regressor.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <pinocchio/multibody/joint/joint-translation.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <stdexcept>
#include <utility>

namespace {
void TestFourteenAxisBatch() {
    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::JointIndex parent = 0;
    robot_identification::Dataset dataset;
    Eigen::VectorXd parameters(140);
    for (int joint = 0; joint < 14; ++joint) {
        const auto name = "joint_" + std::to_string(joint);
        const pinocchio::SE3 placement(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.1, .02, .03));
        if (joint % 2)
            parent = model->addJoint(parent, pinocchio::JointModelRZ(), placement, name);
        else
            parent = model->addJoint(parent, pinocchio::JointModelRY(), placement, name);
        model->appendBodyToJoint(
            parent, pinocchio::Inertia(1. + .1 * joint, Eigen::Vector3d(.1, .02, .03), Eigen::Matrix3d::Identity()),
            pinocchio::SE3::Identity());
        parameters.segment<10>(10 * joint) = model->inertias[parent].toDynamicParameters();
        dataset.joint_names.push_back(name);
    }
    for (int index = 0; index < 40; ++index) {
        robot_identification::Sample sample;
        sample.time_seconds = index * .01;
        sample.position.resize(14);
        sample.velocity.resize(14);
        sample.acceleration.resize(14);
        for (int joint = 0; joint < 14; ++joint) {
            sample.position[joint] = .2 * std::sin(index + joint * .3);
            sample.velocity[joint] = index % 3 == 0 ? 0. : .1 * std::cos(index - joint * .2);
            sample.acceleration[joint] = index % 5 == 0 ? 0. : -.2 * std::sin(index + joint * .7);
        }
        pinocchio::Data data(*model);
        sample.torque = pinocchio::rnea(*model, data, sample.position, sample.velocity, sample.acceleration);
        dataset.samples.push_back(std::move(sample));
    }
    robot_identification_pinocchio::PinocchioRegressor backend(model);
    const auto generic = robot_identification::BuildRegressionProblem(dataset, backend);
    const auto actual = backend.BuildProblem(dataset);
    if (!actual.regressor.isApprox(generic.regressor, 0) || !actual.observation.isApprox(generic.observation, 0) ||
        !(actual.regressor * parameters).isApprox(actual.observation, 1e-11)) {
        throw std::runtime_error("fourteen-axis batch changed rows or RNEA torque");
    }
    (void)backend.Compute(dataset.samples.back());
    if (!actual.regressor.isApprox(generic.regressor, 0)) {
        throw std::runtime_error("later calls modified an owning batch result");
    }
}
}  // namespace

int main() {
    TestFourteenAxisBatch();
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
    bool invalid_sample_rejected = false;
    try {
        static_cast<void>(backend.Compute(sample));
    } catch (const std::invalid_argument&) {
        invalid_sample_rejected = true;
    }
    if (!invalid_sample_rejected) throw std::runtime_error("Pinocchio regressor accepted an invalid sample");
    sample.velocity = Eigen::VectorXd::Constant(1, -0.2);
    robot_identification::Dataset dataset;
    dataset.joint_names = {"joint"};
    for (int index = 0; index < 20; ++index) {
        auto value = sample;
        value.time_seconds = index * 0.01;
        value.position[0] = 0.1 * std::sin(index);
        value.velocity[0] = 0.2 * std::cos(index);
        value.acceleration[0] = -0.2 * std::sin(index);
        value.torque[0] = 0.01 * index;
        dataset.samples.push_back(std::move(value));
    }
    const auto generic = robot_identification::BuildRegressionProblem(dataset, backend);
    const auto verify = [&](const robot_identification::RegressionProblem& result) {
        if (!result.Validate() || result.sample_count != generic.sample_count || result.dof != generic.dof ||
            !result.regressor.isApprox(generic.regressor, 0) || !result.observation.isApprox(generic.observation, 0) ||
            !result.row_weights.isApprox(generic.row_weights, 0)) {
            throw std::runtime_error("reused Pinocchio workspace changed dataset assembly");
        }
    };
    verify(backend.BuildProblem(dataset));
    auto concurrent = std::async(std::launch::async, [&] { return backend.BuildProblem(dataset); });
    verify(backend.BuildProblem(dataset));
    verify(concurrent.get());
    const auto require_invalid = [](const auto& target, const robot_identification::Dataset& values) {
        try {
            (void)target.BuildProblem(values);
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error("named batch accepted incompatible model coordinates");
    };
    auto wrong_name = dataset;
    wrong_name.joint_names[0] = "unknown";
    require_invalid(backend, wrong_name);

    auto pair_model = std::make_shared<pinocchio::Model>();
    const auto first = pair_model->addJoint(0, pinocchio::JointModelRY(), pinocchio::SE3::Identity(), "hip");
    pair_model->appendBodyToJoint(first, pinocchio::Inertia::Identity(), pinocchio::SE3::Identity());
    const auto second = pair_model->addJoint(first, pinocchio::JointModelRZ(), pinocchio::SE3::Identity(), "knee");
    pair_model->appendBodyToJoint(second, pinocchio::Inertia::Identity(), pinocchio::SE3::Identity());
    robot_identification_pinocchio::PinocchioRegressor pair_backend(pair_model);
    robot_identification::Dataset pair_dataset;
    pair_dataset.joint_names = {"hip", "knee"};
    auto pair_sample = sample;
    pair_sample.position = Eigen::Vector2d(0.2, -0.1);
    pair_sample.velocity = Eigen::Vector2d(0.3, 0.2);
    pair_sample.acceleration = Eigen::Vector2d(-0.1, 0.5);
    pair_sample.torque = Eigen::Vector2d::Zero();
    for (int index = 0; index < 3; ++index) {
        pair_sample.time_seconds = index * 0.01;
        pair_dataset.samples.push_back(pair_sample);
    }
    const auto pair_problem = pair_backend.BuildProblem(pair_dataset);
    if (!pair_problem.regressor.isApprox(
            robot_identification::BuildRegressionProblem(pair_dataset, pair_backend).regressor, 0)) {
        throw std::runtime_error("correctly ordered named batch changed scalar joint rows");
    }
    std::swap(pair_dataset.joint_names[0], pair_dataset.joint_names[1]);
    require_invalid(pair_backend, pair_dataset);

    auto multi_model = std::make_shared<pinocchio::Model>();
    const auto multi =
        multi_model->addJoint(0, pinocchio::JointModelTranslation(), pinocchio::SE3::Identity(), "cartesian");
    multi_model->appendBodyToJoint(multi, pinocchio::Inertia::Identity(), pinocchio::SE3::Identity());
    robot_identification_pinocchio::PinocchioRegressor multi_backend(multi_model);
    robot_identification::Dataset multi_dataset;
    multi_dataset.joint_names = {"cartesian/x", "cartesian/y", "cartesian/z"};
    auto multi_sample = sample;
    multi_sample.position = Eigen::Vector3d::Zero();
    multi_sample.velocity = Eigen::Vector3d::Zero();
    multi_sample.acceleration = Eigen::Vector3d::Ones();
    multi_sample.torque = Eigen::Vector3d::Zero();
    for (int index = 0; index < 3; ++index) {
        multi_sample.time_seconds = index * 0.01;
        multi_dataset.samples.push_back(multi_sample);
    }
    if (multi_backend.Compute(multi_sample).rows() != 3) {
        throw std::runtime_error("positional multi-DoF interface changed");
    }
    require_invalid(multi_backend, multi_dataset);
    dataset.samples.back().velocity.conservativeResize(0);
    bool invalid_rejected = false;
    try {
        (void)backend.BuildProblem(dataset);
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    if (!invalid_rejected) throw std::runtime_error("batch assembly accepted invalid samples");
    return 0;
}
