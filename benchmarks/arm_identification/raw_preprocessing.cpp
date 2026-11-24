#include "raw_preprocessing.h"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace arm_identification {

robot_identification::Dataset EstimateLocalCubic(const robot_identification::RawDataset& raw, std::size_t window) {
    using Eigen::Index;
    const Index rows = raw.time_seconds.size();
    if (window < 5 || window % 2 == 0 || window > static_cast<std::size_t>(rows - 2))
        throw std::invalid_argument("local cubic requires an odd window of at least five and three interior samples");

    const Index radius = static_cast<Index>(window / 2);
    Eigen::Matrix<double, Eigen::Dynamic, 4> polynomial(window, 4);
    polynomial.col(0).setOnes();
    robot_identification::Dataset output{raw.joint_names, {}};
    output.samples.reserve(rows - 2 * radius);
    for (Index center = radius; center < rows - radius; ++center) {
        const Index first = center - radius;
        const double time = raw.time_seconds[center];
        const double scale = std::max(time - raw.time_seconds[first], raw.time_seconds[center + radius] - time);
        if (!std::isfinite(scale) || scale <= 0.)
            throw std::invalid_argument("local cubic requires a finite positive time span");

        // Center the position and scale time before building powers of the offsets.
        polynomial.col(1) = (raw.time_seconds.segment(first, window).array() - time) / scale;
        polynomial.col(2) = polynomial.col(1).array().square();
        polynomial.col(3) = polynomial.col(1).array() * polynomial.col(2).array();
        const auto qr = polynomial.colPivHouseholderQr();
        if (qr.rank() != 4) throw std::invalid_argument("local cubic has a rank-deficient timestamp window");
        const Eigen::MatrixXd displacement =
            raw.position.middleRows(first, window).rowwise() - raw.position.row(center);
        const Eigen::Matrix<double, 4, Eigen::Dynamic> coefficients = qr.solve(displacement);
        output.samples.push_back({time, (raw.position.row(center) + coefficients.row(0)).transpose(),
                                  coefficients.row(1).transpose() / scale,
                                  (2. / scale) * coefficients.row(2).transpose() / scale,
                                  raw.torque.row(center).transpose()});
    }
    return output;
}

}  // namespace arm_identification
