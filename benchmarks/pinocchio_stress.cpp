// Deterministic synthetic regression/lifetime campaign; no hardware validation.
#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <mutex>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/joint/joint-prismatic.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <pinocchio/multibody/joint/joint-translation.hpp>
#include <random>
#include <stdexcept>

#include "stress_duration.h"

namespace {
using namespace robot_identification;
using robot_identification_pinocchio::PinocchioRegressor;
struct Source {
    std::shared_ptr<const pinocchio::Model> model;
    std::shared_ptr<const PinocchioRegressor> backend;
    std::vector<std::string> coordinates;
    Eigen::VectorXd parameters;
    bool scalar{true};
};
Source MakeSource(int count, bool translation = false) {
    auto model = std::make_shared<pinocchio::Model>();
    Source source;
    source.scalar = !translation;
    pinocchio::JointIndex parent = 0;
    for (int joint = 0; joint < (translation ? 1 : count); ++joint) {
        const auto name = "body_" + std::to_string(joint);
        const pinocchio::SE3 placement(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.1, .02, .03));
        if (translation) {
            parent = model->addJoint(0, pinocchio::JointModelTranslation(), placement, name);
            source.coordinates = {"cart/x", "cart/y", "cart/z"};
        } else {
            if (joint % 3 == 0)
                parent = model->addJoint(parent, pinocchio::JointModelRY(), placement, name);
            else if (joint % 3 == 1)
                parent = model->addJoint(parent, pinocchio::JointModelRZ(), placement, name);
            else
                parent = model->addJoint(parent, pinocchio::JointModelPX(), placement, name);
            source.coordinates.push_back(name);
        }
        const double mass = 1. + .1 * joint;
        model->appendBodyToJoint(
            parent,
            pinocchio::Inertia(mass, Eigen::Vector3d(.04, .02, .01), Eigen::Matrix3d::Identity() * (.03 * mass)),
            pinocchio::SE3::Identity());
    }
    model->gravity.linear() = Eigen::Vector3d(.3, -.5, -9.81);
    source.parameters.resize(10 * (model->njoints - 1));
    for (int joint = 1; joint < model->njoints; ++joint)
        source.parameters.segment<10>(10 * (joint - 1)) = model->inertias[joint].toDynamicParameters();
    source.model = model;
    source.backend = std::make_shared<PinocchioRegressor>(model);
    return source;
}
struct Result {
    unsigned long long cases{0}, samples{0}, actuator_rows{0}, rejected_batches{0};
    double maximum_error{0};
    bool failed{false};
};
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
double Error(const Eigen::VectorXd& actual, const Eigen::VectorXd& expected) {
    return (actual - expected).norm() / std::max(1., expected.norm());
}
}  // namespace

int main(int argc, char** argv) {
    const double seconds = synthetic_validation::DurationSeconds(argc, argv);
    const std::vector<Source> shared{MakeSource(1), MakeSource(2), MakeSource(7), MakeSource(14), MakeSource(3, true)};
    const auto started = std::chrono::steady_clock::now();
    std::atomic<bool> stop{false};
    std::mutex output_mutex;
    std::vector<std::future<Result>> tasks;
    for (int worker = 0; worker < 4; ++worker) {
        tasks.push_back(std::async(std::launch::async, [&, worker] {
            Result result;
            std::mt19937_64 random(0x20261028ef2dcafULL + worker);
            std::uniform_real_distribution<double> unit(-.5, .5);
            double next_progress = 0;
            try {
                while (!stop &&
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
                    Source transient;
                    const Source* source;
                    if (result.cases % 8 == 0) {
                        transient = MakeSource(1 + random() % 14);
                        source = &transient;
                    } else
                        source = &shared[random() % shared.size()];
                    Dataset dataset;
                    dataset.joint_names = source->coordinates;
                    pinocchio::Data independent(*source->model);
                    const int dof = source->model->nv;
                    for (int sample_index = 0, count = 3 + random() % 5; sample_index < count; ++sample_index) {
                        Sample sample;
                        sample.time_seconds = sample_index * .01;
                        sample.position.resize(dof);
                        sample.velocity.resize(dof);
                        sample.acceleration.resize(dof);
                        for (int joint = 0; joint < dof; ++joint) {
                            sample.position[joint] = unit(random);
                            sample.velocity[joint] = (sample_index + joint) % 4 == 0 ? 0. : .1 * unit(random);
                            sample.acceleration[joint] = .2 * unit(random);
                        }
                        sample.torque = pinocchio::rnea(*source->model, independent, sample.position, sample.velocity,
                                                        sample.acceleration);
                        const auto matrix = source->backend->Compute(sample);
                        const double error = Error(matrix * source->parameters, sample.torque);
                        Require(std::isfinite(error) && error < 1e-10, "single regressor disagrees with native RNEA");
                        result.maximum_error = std::max(result.maximum_error, error);
                        dataset.samples.push_back(std::move(sample));
                        ++result.samples;
                    }
                    const auto generic = BuildRegressionProblem(dataset, *source->backend);
                    if (source->scalar) {
                        const auto batch = source->backend->BuildProblem(dataset);
                        Require(batch.regressor.isApprox(generic.regressor, 0) &&
                                    batch.observation.isApprox(generic.observation, 0),
                                "named batch changed row/column values");
                    } else {
                        bool rejected = false;
                        try {
                            (void)source->backend->BuildProblem(dataset);
                        } catch (const std::invalid_argument&) {
                            rejected = true;
                        }
                        Require(rejected, "named batch accepted a multi-coordinate joint");
                        ++result.rejected_batches;
                    }
                    JointActuatorRegressorOptions options;
                    options.coulomb_velocity_deadband = .02;
                    JointActuatorRegressor actuators(source->backend, dataset.joint_names, options);
                    Eigen::VectorXd parameters(source->parameters.size() + 4 * dof);
                    parameters.head(source->parameters.size()) = source->parameters;
                    for (int joint = 0; joint < dof; ++joint) {
                        parameters.segment<4>(source->parameters.size() + 4 * joint) << .03, .2, .05, -.01;
                        for (auto& sample : dataset.samples) {
                            const double sign = std::abs(sample.velocity[joint]) <= .02
                                                    ? 0.
                                                    : std::copysign(1., sample.velocity[joint]);
                            sample.torque[joint] +=
                                .03 * sample.acceleration[joint] + .2 * sample.velocity[joint] + .05 * sign - .01;
                        }
                    }
                    const auto composed = BuildRegressionProblem(dataset, actuators);
                    const double error = Error(composed.regressor * parameters, composed.observation);
                    Require(composed.Validate() && std::isfinite(error) && error < 1e-10,
                            "actuator composition changed torque/column order");
                    result.maximum_error = std::max(result.maximum_error, error);
                    result.actuator_rows += composed.observation.size();
                    const auto saved = generic.regressor;
                    (void)shared[(result.cases + worker) % shared.size()].backend->Compute([&] {
                        Sample value;
                        const int size = shared[(result.cases + worker) % shared.size()].model->nv;
                        value.position = value.velocity = value.acceleration = Eigen::VectorXd::Zero(size);
                        return value;
                    }());
                    Require(generic.regressor.isApprox(saved, 0), "later model use modified an owning problem");
                    ++result.cases;
                    if (worker == 0) {
                        const double elapsed =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                        if (elapsed >= next_progress) {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            std::cout << "elapsed_s=" << elapsed << " worker0_cases=" << result.cases
                                      << " max_scaled_error=" << result.maximum_error << std::endl;
                            next_progress = elapsed + 60;
                        }
                    }
                }
            } catch (const std::exception& error) {
                stop = result.failed = true;
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cerr << "FAILED worker=" << worker << " case=" << result.cases << ": " << error.what()
                          << std::endl;
            }
            return result;
        }));
    }
    Result total;
    for (auto& task : tasks) {
        const auto result = task.get();
        total.failed = total.failed || result.failed;
        total.cases += result.cases;
        total.samples += result.samples;
        total.actuator_rows += result.actuator_rows;
        total.rejected_batches += result.rejected_batches;
        total.maximum_error = std::max(total.maximum_error, result.maximum_error);
    }
    std::cout << (total.failed ? "FAILED" : "PASSED") << " workers=4 cases=" << total.cases
              << " samples=" << total.samples << " actuator_rows=" << total.actuator_rows
              << " rejected_batches=" << total.rejected_batches << " max_scaled_error=" << total.maximum_error
              << std::endl;
    return total.failed ? 1 : 0;
}
