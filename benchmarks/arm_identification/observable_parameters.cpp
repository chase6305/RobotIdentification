#include "observable_parameters.h"

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>

namespace arm_identification {

std::vector<std::optional<Eigen::VectorXd>> ObservableCoordinates(const Eigen::MatrixXd& full_to_base,
                                                                  Eigen::Index first_column,
                                                                  Eigen::Index count) {
    Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(full_to_base.cols(), count);
    targets.middleRows(first_column, count).setIdentity();
    const auto qr = full_to_base.transpose().colPivHouseholderQr();
    const Eigen::MatrixXd weights = qr.solve(targets);
    const Eigen::MatrixXd error = full_to_base.transpose() * weights - targets;
    std::vector<std::optional<Eigen::VectorXd>> result(count);
    for (Eigen::Index column = 0; column < count; ++column) {
        if (error.col(column).cwiseAbs().maxCoeff() <= 1e-8) result[column] = weights.col(column);
    }
    return result;
}

DriveProjection ProjectNonnegativeDrive(const Eigen::MatrixXd& regressor,
                                        const Eigen::VectorXd& row_weights,
                                        const Eigen::VectorXd& estimate,
                                        const std::vector<std::optional<Eigen::VectorXd>>& coordinates) {
    const auto count = static_cast<Eigen::Index>(
        std::count_if(coordinates.begin(), coordinates.end(), [](const auto& item) { return item.has_value(); }));
    Eigen::MatrixXd constraints(count, estimate.size());
    Eigen::Index row = 0;
    for (const auto& item : coordinates)
        if (item) constraints.row(row++) = item->transpose();
    const Eigen::VectorXd margin = constraints * estimate;
    DriveProjection result;
    result.negative_before = static_cast<std::size_t>((margin.array() < 0.).count());
    if (result.negative_before == 0) {
        result.parameters = estimate;
        return result;
    }

    const Eigen::MatrixXd weighted = regressor.array().colwise() * row_weights.array().sqrt();
    const Eigen::LLT<Eigen::MatrixXd> information(weighted.transpose() * weighted);
    if (information.info() != Eigen::Success) return result;
    const Eigen::MatrixXd directions = information.solve(constraints.transpose());
    const Eigen::MatrixXd dual = constraints * directions;
    if (!dual.allFinite() || (dual.diagonal().array() <= 0.).any()) return result;

    // Diagonal scaling gives unit curvature to each dual coordinate.
    const Eigen::VectorXd scales = dual.diagonal().array().rsqrt();
    const Eigen::MatrixXd scaled_dual = scales.asDiagonal() * dual * scales.asDiagonal();
    Eigen::VectorXd gradient = scales.cwiseProduct(margin);
    Eigen::VectorXd multipliers = Eigen::VectorXd::Zero(count);
    const double tolerance = 1e-10 * std::max(1., gradient.cwiseAbs().maxCoeff());
    for (int iteration = 0; iteration < 1000; ++iteration) {
        for (Eigen::Index index = 0; index < count; ++index) {
            const double updated = std::max(0., multipliers[index] - gradient[index] / scaled_dual(index, index));
            const double step = updated - multipliers[index];
            multipliers[index] = updated;
            gradient += step * scaled_dual.col(index);
        }
        double maximum_error = 0.;
        for (Eigen::Index index = 0; index < count; ++index) {
            const double error = multipliers[index] > 0. ? std::abs(gradient[index]) : std::max(0., -gradient[index]);
            maximum_error = std::max(maximum_error, error);
        }
        if (maximum_error <= tolerance) {
            const Eigen::VectorXd projected = estimate + directions * scales.cwiseProduct(multipliers);
            if (projected.allFinite() && (constraints * projected).minCoeff() >= -1e-9) result.parameters = projected;
            break;
        }
    }
    return result;
}

}  // namespace arm_identification
