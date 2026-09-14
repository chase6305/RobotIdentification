#pragma once

#include <robot_identification/identification.h>

#include <memory>
#include <pinocchio/multibody/model.hpp>
#include <string>
#include <vector>

namespace robot_identification_pinocchio {

// Rigid-body inverse-dynamics regressor backed by Pinocchio. The current
// Dataset contract supports fixed-base models whose nq equals nv.
class PinocchioRegressor final : public robot_identification::RegressorBackend {
public:
    explicit PinocchioRegressor(std::shared_ptr<const pinocchio::Model> model);

    [[nodiscard]] std::size_t parameter_count() const noexcept override;
    [[nodiscard]] std::vector<std::string> parameter_names() const override;
    [[nodiscard]] Eigen::MatrixXd Compute(const robot_identification::Sample& sample) const override;

    [[nodiscard]] const pinocchio::Model& model() const noexcept { return *model_; }

private:
    std::shared_ptr<const pinocchio::Model> model_;
    std::vector<std::string> parameter_names_;
};

}  // namespace robot_identification_pinocchio
