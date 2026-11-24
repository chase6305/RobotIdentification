#pragma once

#include <robot_identification/identification.h>

namespace arm_identification {

// Fit a cubic around each interior timestamp of a validated raw dataset.
robot_identification::Dataset EstimateLocalCubic(const robot_identification::RawDataset& raw, std::size_t window);

}  // namespace arm_identification
