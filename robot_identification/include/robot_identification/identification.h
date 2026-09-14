#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace robot_identification {

struct Sample {
    double time_seconds{0.0};
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    Eigen::VectorXd acceleration;
    Eigen::VectorXd torque;
};

struct Dataset {
    std::vector<std::string> joint_names;
    std::vector<Sample> samples;

    [[nodiscard]] std::size_t dof() const noexcept { return joint_names.size(); }
    [[nodiscard]] bool Validate(std::string* reason = nullptr) const;
};

struct RawDataset {
    std::vector<std::string> joint_names;
    Eigen::VectorXd time_seconds;
    Eigen::MatrixXd position;
    Eigen::MatrixXd torque;

    [[nodiscard]] bool Validate(std::string* reason = nullptr) const;
};

// Portable processed dataset format:
// time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
void WriteDatasetCsv(const Dataset& dataset, const std::string& path);
[[nodiscard]] Dataset ReadDatasetCsv(const std::string& path);
// Raw recorder format: time_s,q/<joint>...,tau/<joint>...
void WriteRawDatasetCsv(const RawDataset& dataset, const std::string& path);
[[nodiscard]] RawDataset ReadRawDatasetCsv(const std::string& path);

struct PreprocessOptions {
    // Odd window length for a symmetric, zero-phase moving-average smoother.
    // One disables smoothing. Derivatives are recomputed from position with
    // a three-point formula that supports nonuniform timestamps.
    std::size_t smoothing_window{5};
    std::size_t trim_front{0};
    std::size_t trim_back{0};
    // One preserves raw torque, matching the legacy identification default.
    std::size_t torque_smoothing_window{1};
};

struct ExcitationSample {
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    Eigen::VectorXd acceleration;
    Eigen::VectorXd jerk;
};

struct HarmonicBounds {
    Eigen::VectorXd minimum_position;
    Eigen::VectorXd maximum_position;
    Eigen::VectorXd maximum_absolute_velocity;
    Eigen::VectorXd maximum_absolute_acceleration;
    Eigen::VectorXd maximum_absolute_jerk;
};

// q(t) = offset + sum_k(sin_coeff[:,k] sin((k+1)wt)
//                         + cos_coeff[:,k] cos((k+1)wt)).
class HarmonicTrajectory {
public:
    HarmonicTrajectory(Eigen::VectorXd offset,
                       Eigen::MatrixXd sin_coefficients,
                       Eigen::MatrixXd cos_coefficients,
                       double fundamental_frequency_hz);

    [[nodiscard]] std::size_t dof() const noexcept;
    [[nodiscard]] std::size_t harmonic_count() const noexcept;
    [[nodiscard]] double period_seconds() const noexcept;
    [[nodiscard]] ExcitationSample Evaluate(double time_seconds) const;
    [[nodiscard]] HarmonicBounds ConservativeBounds() const;

private:
    Eigen::VectorXd offset_;
    Eigen::MatrixXd sin_coefficients_;
    Eigen::MatrixXd cos_coefficients_;
    double angular_frequency_{0.0};
};

struct ExcitationLimits {
    Eigen::VectorXd minimum_position;
    Eigen::VectorXd maximum_position;
    Eigen::VectorXd maximum_velocity;
    Eigen::VectorXd maximum_acceleration;
    Eigen::VectorXd maximum_jerk;
};

struct ExcitationValidationResult {
    bool passed{false};
    std::size_t sample_count{0};
    Eigen::VectorXd peak_absolute_velocity;
    Eigen::VectorXd peak_absolute_acceleration;
    Eigen::VectorXd peak_absolute_jerk;
    Eigen::VectorXd conservative_position_envelope;
    Eigen::VectorXd conservative_velocity_bound;
    Eigen::VectorXd conservative_acceleration_bound;
    Eigen::VectorXd conservative_jerk_bound;
    std::string reason;
};

[[nodiscard]] ExcitationValidationResult ValidateExcitationTrajectory(const HarmonicTrajectory& trajectory,
                                                                      const ExcitationLimits& limits,
                                                                      double duration_seconds,
                                                                      double sample_period_seconds);

[[nodiscard]] Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                                  const Eigen::VectorXd& time_seconds,
                                                  const Eigen::MatrixXd& position,
                                                  const Eigen::MatrixXd& torque,
                                                  const PreprocessOptions& options = {});
[[nodiscard]] Dataset PreprocessPositionAndTorque(const RawDataset& dataset, const PreprocessOptions& options = {});

class RegressorBackend {
public:
    virtual ~RegressorBackend() = default;
    [[nodiscard]] virtual std::size_t parameter_count() const noexcept = 0;
    [[nodiscard]] virtual std::vector<std::string> parameter_names() const = 0;
    [[nodiscard]] virtual Eigen::MatrixXd Compute(const Sample& sample) const = 0;
};

struct JointActuatorRegressorOptions {
    bool drive_inertia{true};
    bool viscous_friction{true};
    bool coulomb_friction{true};
    bool torque_bias{true};
    double coulomb_velocity_deadband{0.0};
};

// Appends arbitrary-DoF actuator columns to a rigid-body/payload backend:
// ddq (drive inertia), dq (viscous), sign(dq) (Coulomb), and 1 (bias).
class JointActuatorRegressor final : public RegressorBackend {
public:
    JointActuatorRegressor(std::shared_ptr<const RegressorBackend> base,
                           std::vector<std::string> joint_names,
                           JointActuatorRegressorOptions options = {});
    [[nodiscard]] std::size_t parameter_count() const noexcept override;
    [[nodiscard]] std::vector<std::string> parameter_names() const override;
    [[nodiscard]] Eigen::MatrixXd Compute(const Sample& sample) const override;

private:
    std::shared_ptr<const RegressorBackend> base_;
    std::vector<std::string> joint_names_;
    JointActuatorRegressorOptions options_;
    std::size_t terms_per_joint_{0};
};

struct RegressionProblem {
    Eigen::MatrixXd regressor;
    Eigen::VectorXd observation;
    Eigen::VectorXd row_weights;
    std::size_t sample_count{0};
    std::size_t dof{0};

    [[nodiscard]] bool Validate(std::string* reason = nullptr) const;
};

[[nodiscard]] RegressionProblem BuildRegressionProblem(const Dataset& dataset, const RegressorBackend& backend);

struct SolverOptions {
    double ridge{0.0};
    double relative_rank_tolerance{1e-10};
    // Zero selects ordinary weighted least squares. Positive values enable
    // Huber IRLS with this residual threshold in torque units.
    double huber_delta{0.0};
    std::size_t max_iterations{30};
    double convergence_tolerance{1e-9};
};

struct IdentificationResult {
    Eigen::VectorXd parameters;
    Eigen::MatrixXd covariance;
    Eigen::VectorXd predicted;
    Eigen::VectorXd residual;
    Eigen::VectorXd final_row_weights;
    std::size_t rank{0};
    double condition_number{0.0};
    double rmse{0.0};
    double max_absolute_error{0.0};
    std::size_t iterations{0};
    bool converged{false};
};

[[nodiscard]] IdentificationResult Solve(const RegressionProblem& problem, const SolverOptions& options = {});

struct ValidationMetrics {
    double rmse{0.0};
    double mean_absolute_error{0.0};
    double max_absolute_error{0.0};
    double normalized_rmse{0.0};
    std::size_t row_count{0};
};

[[nodiscard]] ValidationMetrics ValidateParameters(const RegressionProblem& problem, const Eigen::VectorXd& parameters);

using MassMatrixFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

struct PhysicalValidationResult {
    bool passed{false};
    std::size_t tested{0};
    std::size_t failed{0};
    double minimum_eigenvalue{0.0};
};

[[nodiscard]] PhysicalValidationResult ValidateMassMatrices(const Eigen::MatrixXd& joint_positions,
                                                            const Eigen::VectorXd& parameters,
                                                            const MassMatrixFunction& mass_matrix,
                                                            double minimum_eigenvalue = 1e-9);

}  // namespace robot_identification
