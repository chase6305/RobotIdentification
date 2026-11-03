#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace robot_identification::detail {

bool ValidateRawInputs(const std::vector<std::string>& joint_names,
                       const Eigen::VectorXd& time_seconds,
                       const Eigen::MatrixXd& position,
                       const Eigen::MatrixXd& torque,
                       std::string* reason);

}  // namespace robot_identification::detail
