// Synthetic in-process benchmark; no hardware timing guarantee.
#include <robot_identification_pinocchio/pinocchio_regressor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <vector>
using robot_identification_pinocchio::PinocchioRegressor;
#include "allocation_hooks.h"

template <typename F>
void Measure(const std::string& name, F function) {
    function();
    std::vector<double> elapsed;
    double checksum = 0, total_allocations = 0, total_new = 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
        allocations = 0;
        new_allocations = 0;
        counting = true;
        const auto start = std::chrono::steady_clock::now();
        auto result = function();
        const auto end = std::chrono::steady_clock::now();
        counting = false;
        total_allocations += allocations;
        total_new += new_allocations;
        elapsed.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        checksum += result.regressor.sum();
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::cout << name << " median_ms=" << elapsed[2] << " malloc_calloc_per_batch=" << total_allocations / 5
              << " new_per_batch=" << total_new / 5
              << " tracked_total_per_batch=" << (total_allocations + total_new) / 5 << " checksum=" << checksum << '\n';
}
int main() {
    for (int dof : {2, 14}) {
        auto model = std::make_shared<pinocchio::Model>();
        pinocchio::JointIndex parent = 0;
        robot_identification::Dataset dataset;
        for (int joint = 0; joint < dof; ++joint) {
            const auto name = "joint_" + std::to_string(joint);
            auto placement = pinocchio::SE3::Identity();
            placement.translation().x() = .2;
            parent = model->addJoint(parent, pinocchio::JointModelRY(), placement, name);
            model->appendBodyToJoint(parent, pinocchio::Inertia::Identity(), pinocchio::SE3::Identity());
            dataset.joint_names.push_back(name);
        }
        for (int row = 0; row < 1000; ++row) {
            robot_identification::Sample sample;
            sample.time_seconds = row * .001;
            sample.position = Eigen::VectorXd::Constant(dof, .1 * std::sin(row * .01));
            sample.velocity = Eigen::VectorXd::Constant(dof, std::cos(row * .01));
            sample.acceleration = Eigen::VectorXd::Constant(dof, -10 * std::sin(row * .01));
            sample.torque = Eigen::VectorXd::Zero(dof);
            dataset.samples.push_back(std::move(sample));
        }
        PinocchioRegressor backend(model);
        Measure("generic_dof" + std::to_string(dof),
                [&] { return robot_identification::BuildRegressionProblem(dataset, backend); });
#ifdef ROBOT_IDENTIFICATION_BENCH_HAS_BATCH_API
        Measure("batch_dof" + std::to_string(dof), [&] { return backend.BuildProblem(dataset); });
#endif
    }
}
