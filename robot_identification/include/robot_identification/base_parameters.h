#pragma once

#include <robot_identification/identification.h>

namespace robot_identification {

// A numerical basis learned from a sufficiently rich calibration regressor.
// Y_base[:,i] = Y[:,selected[i]] / column_scales[selected[i]],
// beta = full_to_base * theta, and Y*theta ~= Y_base*beta.
// These coordinates are parameter combinations, not individual link inertias.
class BaseParameterBasis {
public:
    explicit BaseParameterBasis(const Eigen::MatrixXd& calibration_regressor, double relative_rank_tolerance = 1e-10);
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    [[nodiscard]] std::size_t full_parameter_count() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& selected_columns() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& column_scales() const noexcept;
    [[nodiscard]] const Eigen::MatrixXd& full_to_base() const noexcept;
    [[nodiscard]] double calibration_error() const noexcept;

    // Maximum entry error, normalized per column by the larger of that
    // column's calibration maximum and its current maximum. Zero calibration
    // columns must remain zero. This checks the learned dependency on holdout.
    // Selected independent columns are exact identities and contribute zero.
    // The extended-precision maximum is converted to double with upward
    // rounding, retaining a positive result even below double denorm_min.
    // Quotients at double boundaries also check their fused division residual.
    [[nodiscard]] double ReconstructionError(const Eigen::MatrixXd& regressor) const;
    [[nodiscard]] Eigen::MatrixXd ReduceRegressor(const Eigen::MatrixXd& regressor,
                                                  double maximum_reconstruction_error = 1e-8) const;
    [[nodiscard]] RegressionProblem ReduceProblem(const RegressionProblem& problem,
                                                  double maximum_reconstruction_error = 1e-8) const;
    // On IEEE binary64 with gradual underflow, round the stored-mapping dot once to
    // nearest/ties-even, including cancellation. Exact values beyond max double
    // and nonzero values rounding to zero throw. This does not remove QR error.
    [[nodiscard]] Eigen::VectorXd ToBaseParameters(const Eigen::VectorXd& full_parameters) const;

private:
    std::vector<std::size_t> selected_columns_;
    Eigen::VectorXd column_scales_;
    Eigen::MatrixXd full_to_base_;
    double calibration_error_{0.};
};

// Wrap any existing backend, including Pinocchio and actuator compositions.
// Each new block must respect the learned dependencies before columns are reduced.
class BaseParameterRegressor final : public RegressorBackend {
public:
    BaseParameterRegressor(std::shared_ptr<const RegressorBackend> backend,
                           BaseParameterBasis basis,
                           double maximum_reconstruction_error = 1e-8);
    [[nodiscard]] std::size_t parameter_count() const noexcept override;
    [[nodiscard]] std::vector<std::string> parameter_names() const override;
    [[nodiscard]] Eigen::MatrixXd Compute(const Sample& sample) const override;
    [[nodiscard]] const BaseParameterBasis& basis() const noexcept;

private:
    std::shared_ptr<const RegressorBackend> backend_;
    BaseParameterBasis basis_;
    double maximum_reconstruction_error_;
    std::vector<std::string> parameter_names_;
};

}  // namespace robot_identification
