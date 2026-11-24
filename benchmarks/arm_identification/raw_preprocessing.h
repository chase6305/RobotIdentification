#pragma once

#include <robot_identification/identification.h>

namespace arm_identification {

// Fit a cubic around each interior timestamp of a validated raw dataset.
robot_identification::Dataset EstimateLocalCubic(const robot_identification::RawDataset& raw, std::size_t window);

// Fit harmonics over one complete period, from the first to the last timestamp.
struct PeriodicFourierEstimate {
    robot_identification::Dataset dataset;
    Eigen::VectorXd huber_delta_rad;
};

PeriodicFourierEstimate EstimatePeriodicFourier(const robot_identification::RawDataset& raw,
                                                std::size_t harmonics,
                                                bool robust,
                                                double torque_delay_seconds = 0.);

}  // namespace arm_identification
