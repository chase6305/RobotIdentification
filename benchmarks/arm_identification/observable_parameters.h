#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

namespace arm_identification {

// A full coordinate has a unique value when its unit vector belongs to the
// row space of the full-to-base map. Its weights act on a fitted base vector.
[[nodiscard]] std::vector<std::optional<Eigen::VectorXd>> ObservableCoordinates(const Eigen::MatrixXd& full_to_base,
                                                                                Eigen::Index first_column,
                                                                                Eigen::Index count);

struct DriveProjection {
    std::size_t negative_before{0};
    std::optional<Eigen::VectorXd> parameters;
};

[[nodiscard]] DriveProjection ProjectNonnegativeDrive(const Eigen::MatrixXd& regressor,
                                                      const Eigen::VectorXd& row_weights,
                                                      const Eigen::VectorXd& estimate,
                                                      const std::vector<std::optional<Eigen::VectorXd>>& coordinates);

}  // namespace arm_identification
