#pragma once

#include <robot_identification/identification.h>

namespace arm_identification {

struct PhysicalArmFit {
    Eigen::VectorXd parameters;
    std::size_t iterations{0};
    bool converged{false};
    double prior_weight{0.};
    double minimum_relative_moment{0.};
};

// Learn full inertias using the measured torques, nominal link inertias as
// a regularizing prior, and positive mass distributions for every moving link.
[[nodiscard]] PhysicalArmFit FitPhysicalArm(const robot_identification::RegressionProblem& training,
                                            const Eigen::VectorXd& row_weights,
                                            const Eigen::VectorXd& nominal,
                                            Eigen::Index joint_count,
                                            double residual_scale);

}  // namespace arm_identification
