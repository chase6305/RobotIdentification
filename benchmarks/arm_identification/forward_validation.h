#pragma once

#include <robot_identification/identification.h>

#include <optional>

namespace arm_identification {

struct ForwardMetrics {
    robot_identification::PhysicalValidationResult mass;
    std::size_t holdout_samples{0};
    std::optional<double> holdout_acceleration_rmse;
};

struct ReplayMetrics {
    double requested_window_seconds{0.};
    std::size_t windows{0};
    std::size_t complete_windows{0};
    std::optional<double> position_rmse;
    std::optional<double> maximum_final_position_error;
};

[[nodiscard]] ReplayMetrics ReplayForwardWindows(const robot_identification::RegressorBackend& backend,
                                                 const Eigen::VectorXd& parameters,
                                                 const robot_identification::Dataset& dense_holdout,
                                                 const Eigen::VectorXd& minimum_position,
                                                 const Eigen::VectorXd& maximum_position);

class ForwardValidation {
public:
    ForwardValidation(const robot_identification::RegressorBackend& backend,
                      const robot_identification::Dataset& calibration,
                      const robot_identification::Dataset& holdout);

    [[nodiscard]] ForwardMetrics Evaluate(const Eigen::VectorXd& parameters) const;

private:
    struct Point {
        Eigen::MatrixXd position;
        Eigen::MatrixXd mass_regressor;
        Eigen::MatrixXd bias_regressor;
        Eigen::VectorXd torque;
        Eigen::VectorXd acceleration;
    };

    void AddPoints(const robot_identification::RegressorBackend& backend,
                   const robot_identification::Dataset& dataset,
                   bool measure_acceleration);

    std::vector<Point> points_;
};

}  // namespace arm_identification
