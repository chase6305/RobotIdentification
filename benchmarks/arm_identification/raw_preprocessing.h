#pragma once

#include <robot_identification/identification.h>

namespace arm_identification {

// Fit a cubic around each interior timestamp of a validated raw dataset.
robot_identification::Dataset EstimateLocalCubic(const robot_identification::RawDataset& raw, std::size_t window);

// Encoder harmonics and their original sampled basis for one full period.
struct PeriodicFourierModel {
    double origin_seconds;
    double period_seconds;
    Eigen::RowVectorXd position_offset;
    Eigen::MatrixXd sampled_basis;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd huber_delta_rad;
};

PeriodicFourierModel FitPeriodicFourier(const robot_identification::RawDataset& raw,
                                        std::size_t harmonics,
                                        bool robust);

robot_identification::Dataset ReconstructPeriodicFourier(const robot_identification::RawDataset& raw,
                                                         const PeriodicFourierModel& model,
                                                         double torque_delay_seconds);

}  // namespace arm_identification
