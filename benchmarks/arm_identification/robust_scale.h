#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <vector>

namespace arm_identification {

inline constexpr double kHuberGaussianMultiplier = 1.345;

inline double Median(const Eigen::VectorXd& values) {
    std::vector<double> scratch(values.data(), values.data() + values.size());
    const auto middle = scratch.begin() + scratch.size() / 2;
    std::nth_element(scratch.begin(), middle, scratch.end());
    if (scratch.size() % 2 != 0) return *middle;
    return .5 * *std::max_element(scratch.begin(), middle) + .5 * *middle;
}

inline double HuberDeltaFromResiduals(const Eigen::VectorXd& residual, double maximum_observation) {
    const double center = Median(residual);
    const double mad = Median((residual.array() - center).abs().matrix());
    const double floor = 64 * std::numeric_limits<double>::epsilon() * std::max(1., maximum_observation);
    return std::max(floor, kHuberGaussianMultiplier * 1.4826 * mad);
}

}  // namespace arm_identification
