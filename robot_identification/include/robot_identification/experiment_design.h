#pragma once

#include <robot_identification/identification.h>

#include <limits>

namespace robot_identification {

struct HarmonicDesign {
    Eigen::VectorXd offset;
    Eigen::MatrixXd sin_coefficients;
    Eigen::MatrixXd cos_coefficients;
    double fundamental_frequency_hz{0.1};
    [[nodiscard]] HarmonicTrajectory trajectory() const;
};

struct ExcitationDesignOptions {
    double duration_seconds{10.};
    std::size_t sample_count{200};
    std::size_t max_evaluations{250};
    // Coefficient steps use this fraction of each joint's half position range,
    // divided by the harmonic count. Offsets and frequency remain fixed.
    double initial_step_fraction{0.1};
    double minimum_step_fraction{1e-4};
    double step_contraction{0.5};
    double relative_rank_tolerance{1e-10};
    double improvement_tolerance{1e-9};
    // Fixed across every candidate. Empty vectors mean all ones.
    // Parameter scales have parameter units; noise scales have torque units.
    Eigen::VectorXd parameter_scales;
    Eigen::VectorXd torque_noise_standard_deviation;
};

struct ExcitationDesignQuality {
    std::size_t rank{0};
    std::size_t parameter_count{0};
    double condition_number{std::numeric_limits<double>::infinity()};
    // Mean log eigenvalue of Y_scaled' Y_scaled / N over directions
    // retained by the relative rank threshold.
    // This is a D-optimal information score only at full column rank.
    double mean_log_information{-std::numeric_limits<double>::infinity()};
    [[nodiscard]] bool fully_identifiable() const noexcept { return parameter_count != 0 && rank == parameter_count; }
};

struct ExcitationDesignResult {
    HarmonicDesign design;
    ExcitationDesignQuality initial_quality;
    ExcitationDesignQuality quality;
    ExcitationValidationResult validation;
    std::size_t evaluations{0};
    std::size_t feasible_evaluations{0};
    std::size_t accepted_steps{0};
    double final_step_fraction{0.};
    // Local step-size termination; an exhausted evaluation budget is not convergence.
    bool converged{false};
};

[[nodiscard]] ExcitationDesignQuality EvaluateExcitationDesign(const HarmonicTrajectory& trajectory,
                                                               const RegressorBackend& backend,
                                                               const ExcitationDesignOptions& options = {});

// Deterministic coordinate pattern search. First improve numerical rank, then
// the information score. The initial design must already satisfy the limits;
// every returned design passes the same conservative and sampled limit checks.
[[nodiscard]] ExcitationDesignResult OptimizeExcitationDesign(const HarmonicDesign& initial,
                                                              const ExcitationLimits& limits,
                                                              const RegressorBackend& backend,
                                                              const ExcitationDesignOptions& options = {});

}  // namespace robot_identification
