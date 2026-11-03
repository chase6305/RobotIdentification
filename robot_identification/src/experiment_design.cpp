#include "robot_identification/experiment_design.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace robot_identification {
namespace {
double ValidateOptions(const ExcitationDesignOptions& options, std::size_t dof, std::size_t parameters) {
    if (parameters == 0 || dof == 0 || options.sample_count < 3 || options.sample_count > 10000000 ||
        !std::isfinite(options.duration_seconds) || options.duration_seconds <= 0. || options.max_evaluations == 0 ||
        !std::isfinite(options.initial_step_fraction) || options.initial_step_fraction <= 0. ||
        options.initial_step_fraction > 1. || !std::isfinite(options.minimum_step_fraction) ||
        options.minimum_step_fraction <= 0. || options.minimum_step_fraction > options.initial_step_fraction ||
        !std::isfinite(options.step_contraction) || options.step_contraction <= 0. || options.step_contraction >= 1. ||
        !std::isfinite(options.relative_rank_tolerance) || options.relative_rank_tolerance <= 0. ||
        options.relative_rank_tolerance >= 1. || !std::isfinite(options.improvement_tolerance) ||
        options.improvement_tolerance < 0.)
        throw std::invalid_argument("excitation design options are invalid");
    const auto validate_scales = [](const Eigen::VectorXd& scales, std::size_t count) {
        return scales.size() == 0 ||
               (static_cast<std::size_t>(scales.size()) == count && scales.allFinite() && (scales.array() > 0.).all());
    };
    if (!validate_scales(options.parameter_scales, parameters) ||
        !validate_scales(options.torque_noise_standard_deviation, dof))
        throw std::invalid_argument("excitation design scales must be finite, positive and match the backend");
    const auto maximum = std::min(static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                                  std::numeric_limits<std::size_t>::max() / sizeof(double));
    if (dof > maximum / options.sample_count || parameters > maximum / (dof * options.sample_count))
        throw std::overflow_error("excitation design matrix dimensions overflowed");
    const double period = options.duration_seconds / static_cast<double>(options.sample_count - 1);
    if (!std::isfinite(period) || period <= 0.)
        throw std::invalid_argument("excitation design sample interval is not representable");
    return period;
}

bool Better(const ExcitationDesignQuality& candidate, const ExcitationDesignQuality& current, double tolerance) {
    if (candidate.rank != current.rank) return candidate.rank > current.rank;
    return candidate.mean_log_information > current.mean_log_information + tolerance;
}
}  // namespace

HarmonicTrajectory HarmonicDesign::trajectory() const {
    return HarmonicTrajectory(offset, sin_coefficients, cos_coefficients, fundamental_frequency_hz);
}

ExcitationDesignQuality EvaluateExcitationDesign(const HarmonicTrajectory& trajectory,
                                                 const RegressorBackend& backend,
                                                 const ExcitationDesignOptions& options) {
    const std::size_t dof = trajectory.dof(), parameters = backend.parameter_count();
    (void)ValidateOptions(options, dof, parameters);
    const Eigen::Index rows = static_cast<Eigen::Index>(dof * options.sample_count);
    Eigen::MatrixXd regressor(rows, static_cast<Eigen::Index>(parameters));
    Sample sample;
    sample.torque = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dof));
    ExcitationSample values;
    double previous = -1.;
    for (std::size_t index = 0; index < options.sample_count; ++index) {
        sample.time_seconds =
            options.duration_seconds * (static_cast<double>(index) / static_cast<double>(options.sample_count - 1));
        if (sample.time_seconds <= previous)
            throw std::invalid_argument("excitation design duration cannot represent distinct sample times");
        previous = sample.time_seconds;
        trajectory.EvaluateInto(sample.time_seconds, values);
        // Lend the owning buffers to the backend sample, then restore them
        // for the next evaluation without copies or additional vector storage.
        sample.position.swap(values.position);
        sample.velocity.swap(values.velocity);
        sample.acceleration.swap(values.acceleration);
        const auto block = backend.Compute(sample);
        sample.position.swap(values.position);
        sample.velocity.swap(values.velocity);
        sample.acceleration.swap(values.acceleration);
        if (block.rows() != static_cast<Eigen::Index>(dof) || block.cols() != static_cast<Eigen::Index>(parameters) ||
            !block.allFinite())
            throw std::runtime_error("excitation design backend returned an invalid regressor block");
        regressor.middleRows(static_cast<Eigen::Index>(index * dof), static_cast<Eigen::Index>(dof)) = block;
    }
    const bool unit_scales =
        (options.parameter_scales.size() == 0 || options.parameter_scales.isOnes(0.)) &&
        (options.torque_noise_standard_deviation.size() == 0 || options.torque_noise_standard_deviation.isOnes(0.));
    const auto scaled = [&](Eigen::Index row, Eigen::Index column) {
        const long double parameter_scale =
            options.parameter_scales.size() == 0 ? 1.L : options.parameter_scales[column];
        const long double noise = options.torque_noise_standard_deviation.size() == 0
                                      ? 1.L
                                      : options.torque_noise_standard_deviation[row % static_cast<Eigen::Index>(dof)];
        return static_cast<long double>(regressor(row, column)) * parameter_scale / noise;
    };
    long double maximum = 0.L;
    if (unit_scales)
        maximum = regressor.cwiseAbs().maxCoeff();
    else
        for (Eigen::Index column = 0; column < regressor.cols(); ++column)
            for (Eigen::Index row = 0; row < regressor.rows(); ++row)
                maximum = std::max(maximum, std::abs(scaled(row, column)));
    if (!std::isfinite(maximum)) throw std::overflow_error("scaled excitation design matrix overflowed");
    ExcitationDesignQuality result;
    result.parameter_count = parameters;
    if (maximum == 0.L) return result;
    if (unit_scales)
        // No weighted products are needed; direct division also handles a
        // subnormal maximum without forming an overflowing reciprocal.
        for (Eigen::Index column = 0; column < regressor.cols(); ++column)
            for (Eigen::Index row = 0; row < regressor.rows(); ++row)
                regressor(row, column) /= static_cast<double>(maximum);
    else
        for (Eigen::Index column = 0; column < regressor.cols(); ++column)
            for (Eigen::Index row = 0; row < regressor.rows(); ++row)
                regressor(row, column) = static_cast<double>(scaled(row, column) / maximum);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(regressor);
    svd.setThreshold(options.relative_rank_tolerance);
    result.rank = static_cast<std::size_t>(svd.rank());
    if (result.rank == 0) return result;
    const auto& singular = svd.singularValues();
    long double log_sum = 0.L;
    for (std::size_t index = 0; index < result.rank; ++index)
        log_sum +=
            2.L * (std::log(static_cast<long double>(singular[static_cast<Eigen::Index>(index)])) + std::log(maximum)) -
            std::log(static_cast<long double>(options.sample_count));
    result.mean_log_information = static_cast<double>(log_sum / result.rank);
    if (result.fully_identifiable()) result.condition_number = singular[0] / singular[singular.size() - 1];
    return result;
}

ExcitationDesignResult OptimizeExcitationDesign(const HarmonicDesign& initial,
                                                const ExcitationLimits& limits,
                                                const RegressorBackend& backend,
                                                const ExcitationDesignOptions& options) {
    const auto seed = initial.trajectory();
    const double period = ValidateOptions(options, seed.dof(), backend.parameter_count());
    ExcitationDesignResult result;
    result.design = initial;
    result.validation = ValidateExcitationTrajectory(seed, limits, options.duration_seconds, period);
    if (!result.validation.passed)
        throw std::invalid_argument("initial excitation design violates the configured limits");
    result.initial_quality = result.quality = EvaluateExcitationDesign(seed, backend, options);
    result.evaluations = result.feasible_evaluations = 1;
    double step = options.initial_step_fraction;
    const Eigen::VectorXd half_range = .5 * limits.maximum_position - .5 * limits.minimum_position;
    if ((half_range.array() <= 0.).any() || !half_range.allFinite())
        throw std::invalid_argument("excitation design position range is not representable");
    while (step >= options.minimum_step_fraction && result.evaluations < options.max_evaluations) {
        bool improved = false;
        for (int kind = 0; kind < 2 && result.evaluations < options.max_evaluations; ++kind)
            for (Eigen::Index harmonic = 0;
                 harmonic < initial.sin_coefficients.cols() && result.evaluations < options.max_evaluations; ++harmonic)
                for (Eigen::Index joint = 0;
                     joint < initial.offset.size() && result.evaluations < options.max_evaluations; ++joint)
                    for (double sign : {1., -1.}) {
                        if (result.evaluations == options.max_evaluations) break;
                        ++result.evaluations;
                        auto& coefficient = (kind == 0 ? result.design.sin_coefficients
                                                       : result.design.cos_coefficients)(joint, harmonic);
                        const double original = coefficient;
                        const double candidate =
                            original + sign * step * half_range[joint] / static_cast<double>(seed.harmonic_count());
                        if (!std::isfinite(candidate)) continue;
                        coefficient = candidate;
                        const auto trajectory = result.design.trajectory();
                        // The trajectory owns its copies. Restore the incumbent
                        // before evaluating, then retain only accepted changes.
                        coefficient = original;
                        auto validation =
                            ValidateExcitationTrajectory(trajectory, limits, options.duration_seconds, period);
                        if (!validation.passed) continue;
                        ++result.feasible_evaluations;
                        const auto quality = EvaluateExcitationDesign(trajectory, backend, options);
                        if (Better(quality, result.quality, options.improvement_tolerance)) {
                            coefficient = candidate;
                            result.quality = quality;
                            result.validation = std::move(validation);
                            ++result.accepted_steps;
                            improved = true;
                        }
                    }
        // A budget-truncated poll did not test all coordinate directions.
        // It cannot justify shrinking the step or declaring convergence.
        if (!improved && result.evaluations < options.max_evaluations) step *= options.step_contraction;
    }
    result.final_step_fraction = step;
    result.converged = step < options.minimum_step_fraction;
    return result;
}
}  // namespace robot_identification
