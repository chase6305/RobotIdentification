#pragma once

#include <robot_identification/identification.h>

#include <optional>

namespace arm_identification {

struct PayloadInertia {
    double mass{0.};
    Eigen::Vector3d center;
    Eigen::Matrix3d inertia_at_center;
    double holdout_rmse{0.};
    bool projected{false};
};

struct PayloadFit {
    double estimated_mass{0.};
    double unconstrained_holdout_rmse{0.};
    std::optional<PayloadInertia> physical;
};

[[nodiscard]] robot_identification::RegressionProblem PayloadRegression(
    const robot_identification::RegressionProblem& arm,
    const Eigen::VectorXd& known_arm_parameters,
    Eigen::Index payload_column);

[[nodiscard]] PayloadFit FitPayload(const robot_identification::RegressionProblem& train,
                                    const robot_identification::RegressionProblem& holdout,
                                    double huber_delta);

}  // namespace arm_identification
