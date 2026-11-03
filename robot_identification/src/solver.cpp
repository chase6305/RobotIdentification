#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "robot_identification/identification.h"

namespace robot_identification {
namespace {

struct MagnitudeStatistics {
    double rms{0.0};
    double mean_absolute{0.0};
    double maximum{0.0};
    // Preserve RMS / maximum for ratios when the rounded RMS is subnormal or zero.
    double rms_over_maximum{0.0};
};

double RobustRowWeight(double base_weight, double huber_delta, double residual_magnitude) {
    if (residual_magnitude <= huber_delta) return base_weight;
    const double factor = huber_delta / residual_magnitude;
    if (factor >= std::numeric_limits<double>::min()) return base_weight * factor;
    // A subnormal ratio can lose precision or become zero before a large
    // base weight restores a representable result. Combine mantissas first.
    int base_exponent, delta_exponent, residual_exponent;
    const double base_fraction = std::frexp(base_weight, &base_exponent);
    const double delta_fraction = std::frexp(huber_delta, &delta_exponent);
    const double residual_fraction = std::frexp(residual_magnitude, &residual_exponent);
    return std::min(base_weight, std::ldexp(base_fraction * delta_fraction / residual_fraction,
                                            base_exponent + delta_exponent - residual_exponent));
}

bool ParametersConverged(const Eigen::VectorXd& current, const Eigen::VectorXd& previous, double tolerance) {
    const double scale = std::max({1.0, current.cwiseAbs().maxCoeff(), previous.cwiseAbs().maxCoeff()});
    double previous_norm = 0.0;
    double change_norm = 0.0;
    for (Eigen::Index row = 0; row < current.size(); ++row) {
        const double difference = current[row] - previous[row];
        // Subtract first to retain small updates; normalize first only when
        // opposite-sign finite parameters overflow their difference.
        const double change =
            std::isfinite(difference) ? difference / scale : current[row] / scale - previous[row] / scale;
        previous_norm = std::hypot(previous_norm, previous[row] / scale);
        change_norm = std::hypot(change_norm, change);
    }
    return change_norm / std::max(1.0 / scale, previous_norm) <= tolerance;
}

MagnitudeStatistics MeasureMagnitudes(const Eigen::VectorXd& values) {
    if (!values.allFinite()) throw std::overflow_error("error metrics require finite predictions and residuals");
    MagnitudeStatistics result;
    result.maximum = values.cwiseAbs().maxCoeff();
    if (result.maximum == 0.0) return result;
    // Scale before squaring or summing: the final metrics are bounded by the
    // largest magnitude even when the unscaled squared norm is not representable.
    long double absolute_sum = 0.0L;
    long double squared_sum = 0.0L;
    for (Eigen::Index row = 0; row < values.size(); ++row) {
        const long double normalized = std::abs(static_cast<long double>(values[row])) / result.maximum;
        absolute_sum += normalized;
        squared_sum += normalized * normalized;
    }
    const long double rms_over_maximum = std::min(1.0L, std::sqrt(squared_sum / values.size()));
    result.rms_over_maximum = static_cast<double>(rms_over_maximum);
    result.rms = static_cast<double>(rms_over_maximum * result.maximum);
    result.mean_absolute = static_cast<double>(std::min(1.0L, absolute_sum / values.size()) * result.maximum);
    return result;
}

double NormalizeRms(const MagnitudeStatistics& error, const MagnitudeStatistics& reference) {
    if (reference.maximum == 0.0) return error.maximum == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    if (error.maximum == 0.0) return 0.0;
    if (error.rms >= std::numeric_limits<double>::min() && reference.rms >= std::numeric_limits<double>::min())
        return error.rms / reference.rms;
    // Rounded subnormal RMS values may lose their ratio, including 0/0
    // for nonzero inputs. Combine the normalized roots and mantissas before
    // restoring the exponent difference, so a finite ratio stays finite.
    int error_exponent, reference_exponent;
    const double error_fraction = std::frexp(error.maximum, &error_exponent);
    const double reference_fraction = std::frexp(reference.maximum, &reference_exponent);
    return std::ldexp((error_fraction / reference_fraction) * (error.rms_over_maximum / reference.rms_over_maximum),
                      error_exponent - reference_exponent);
}

class WeightedSolveWorkspace {
public:
    WeightedSolveWorkspace(const RegressionProblem& problem, const SolverOptions& options)
        : data_rows_(problem.regressor.rows()), ridge_rows_(options.ridge > 0.0 ? problem.regressor.cols() : 0) {
        if (data_rows_ > std::numeric_limits<Eigen::Index>::max() - ridge_rows_) {
            throw std::overflow_error("regularized regression row count overflowed");
        }
        const auto parameter_count = problem.regressor.cols();
        weighted_regressor_.resize(data_rows_ + ridge_rows_, parameter_count);
        weighted_observation_.resize(data_rows_ + ridge_rows_);
        sqrt_weights_.resize(data_rows_);
        const unsigned int diagnostics_options = ridge_rows_ == 0 ? Eigen::ComputeThinU | Eigen::ComputeThinV : 0;
        diagnostic_svd_.emplace(data_rows_, parameter_count, diagnostics_options);
        diagnostic_svd_->setThreshold(options.relative_rank_tolerance);
        if (ridge_rows_ > 0) {
            weighted_regressor_.bottomRows(ridge_rows_).setZero();
            weighted_regressor_.bottomRows(ridge_rows_).diagonal().setConstant(std::sqrt(options.ridge));
            weighted_observation_.tail(ridge_rows_).setZero();
            regularized_svd_.emplace(data_rows_ + ridge_rows_, parameter_count,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
            regularized_svd_->setThreshold(options.relative_rank_tolerance);
        }
    }

    void Compute(const RegressionProblem& problem, const Eigen::VectorXd& weights, IdentificationResult& result) {
        sqrt_weights_ = weights.array().sqrt();
        weighted_regressor_.topRows(data_rows_) = problem.regressor.array().colwise() * sqrt_weights_.array();
        weighted_observation_.head(data_rows_) = problem.observation.array() * sqrt_weights_.array();
        if (!weighted_regressor_.topRows(data_rows_).allFinite() ||
            !weighted_observation_.head(data_rows_).allFinite()) {
            throw std::overflow_error("weighted regression values are non-finite");
        }
        // Diagnose measured data before ridge rows: regularization must not
        // disguise an unobservable parameter. Reuse each decomposition's
        // scratch across IRLS iterations without changing its input shape.
        if (ridge_rows_ == 0) {
            diagnostic_svd_->compute(weighted_regressor_);
        } else {
            diagnostic_svd_->compute(weighted_regressor_.topRows(data_rows_));
        }
        if (!diagnostic_svd_->singularValues().allFinite()) {
            throw std::runtime_error("regression diagnostic decomposition failed");
        }
        if (ridge_rows_ == 0) {
            result.parameters = diagnostic_svd_->solve(weighted_observation_);
        } else {
            regularized_svd_->compute(weighted_regressor_);
            if (!regularized_svd_->singularValues().allFinite()) {
                throw std::runtime_error("regularized regression decomposition failed");
            }
            result.parameters = regularized_svd_->solve(weighted_observation_);
        }
    }

    void SetDiagnostics(IdentificationResult& result) const {
        result.rank = static_cast<std::size_t>(diagnostic_svd_->rank());
        const auto& singular_values = diagnostic_svd_->singularValues();
        result.condition_number = result.rank < static_cast<std::size_t>(weighted_regressor_.cols())
                                      ? std::numeric_limits<double>::infinity()
                                      : singular_values[0] / singular_values[singular_values.size() - 1];
    }

    Eigen::MatrixXd Covariance(const Eigen::VectorXd& residual, std::size_t measured_rank) const {
        const auto degrees_of_freedom =
            std::max<Eigen::Index>(1, data_rows_ - static_cast<Eigen::Index>(measured_rank));
        const double variance =
            (residual.array() * sqrt_weights_.array()).matrix().squaredNorm() / static_cast<double>(degrees_of_freedom);
        double residual_scale = std::sqrt(variance);
        int scale_exponent = 0;
        if (!std::isfinite(variance) || variance < std::numeric_limits<double>::min()) {
            // Keep the root weighted error as a mantissa/exponent pair when
            // squaring it would lose range. Its ratio to a singular value can
            // still yield a representable parameter uncertainty.
            residual_scale = 0.0;
            bool initialized = false;
            for (Eigen::Index row = 0; row < residual.size(); ++row) {
                if (residual[row] == 0.0 || sqrt_weights_[row] == 0.0) continue;
                int residual_exponent, weight_exponent;
                const double fraction = std::frexp(std::abs(residual[row]), &residual_exponent) *
                                        std::frexp(sqrt_weights_[row], &weight_exponent);
                const int exponent = residual_exponent + weight_exponent;
                if (!initialized || exponent > scale_exponent) {
                    residual_scale = std::ldexp(residual_scale, scale_exponent - exponent);
                    scale_exponent = exponent;
                    initialized = true;
                }
                residual_scale = std::hypot(residual_scale, std::ldexp(fraction, exponent - scale_exponent));
            }
            residual_scale /= std::sqrt(static_cast<double>(degrees_of_freedom));
        }
        // Use the same retained singular directions as the parameter solve.
        // Forming A'WA squares its condition number and a second decomposition
        // can incorrectly erase observable, weakly excited directions.
        const auto& decomposition = ridge_rows_ == 0 ? *diagnostic_svd_ : *regularized_svd_;
        const Eigen::Index rank = ridge_rows_ == 0 ? static_cast<Eigen::Index>(measured_rank) : decomposition.rank();
        Eigen::MatrixXd scaled_basis = decomposition.matrixV().leftCols(rank);
        for (Eigen::Index column = 0; column < rank; ++column) {
            double uncertainty;
            if (scale_exponent == 0) {
                uncertainty = residual_scale / decomposition.singularValues()[column];
            } else {
                int singular_exponent;
                const double fraction = std::frexp(decomposition.singularValues()[column], &singular_exponent);
                uncertainty = std::ldexp(residual_scale / fraction, scale_exponent - singular_exponent);
            }
            scaled_basis.col(column) *= uncertainty;
        }
        Eigen::MatrixXd covariance = scaled_basis * scaled_basis.transpose();
        if (!covariance.allFinite()) throw std::overflow_error("parameter covariance is non-finite");
        return covariance;
    }

private:
    using Svd = Eigen::JacobiSVD<Eigen::MatrixXd>;
    Eigen::Index data_rows_, ridge_rows_;
    Eigen::MatrixXd weighted_regressor_;
    Eigen::VectorXd weighted_observation_, sqrt_weights_;
    std::optional<Svd> diagnostic_svd_, regularized_svd_;
};

}  // namespace

IdentificationResult Solve(const RegressionProblem& problem, const SolverOptions& options) {
    std::string reason;
    if (!problem.Validate(&reason)) throw std::invalid_argument("invalid regression problem: " + reason);
    if (!std::isfinite(options.ridge) || options.ridge < 0.0 || !std::isfinite(options.relative_rank_tolerance) ||
        options.relative_rank_tolerance <= 0.0 || !std::isfinite(options.huber_delta) || options.huber_delta < 0.0 ||
        options.max_iterations == 0 || !std::isfinite(options.convergence_tolerance) ||
        options.convergence_tolerance <= 0.0) {
        throw std::invalid_argument("solver options are invalid");
    }
    Eigen::VectorXd weights = problem.row_weights;
    Eigen::VectorXd previous_parameters;
    IdentificationResult result;
    WeightedSolveWorkspace workspace(problem, options);
    Eigen::VectorXd robust_residual;
    if (options.huber_delta > 0.0 && options.max_iterations > 1) robust_residual.resize(problem.regressor.rows());
    std::size_t iterations = 0;
    bool converged = options.huber_delta == 0.0;
    const std::size_t iteration_limit = options.huber_delta > 0.0 ? options.max_iterations : 1;
    for (; iterations < iteration_limit; ++iterations) {
        workspace.Compute(problem, weights, result);
        if (!result.parameters.allFinite()) throw std::runtime_error("parameter solution is non-finite");
        if (previous_parameters.size() == result.parameters.size()) {
            if (ParametersConverged(result.parameters, previous_parameters, options.convergence_tolerance)) {
                converged = true;
                ++iterations;
                break;
            }
        }
        // Only prepare another weighted solve when an iteration remains.
        if (options.huber_delta > 0.0 && iterations + 1 < iteration_limit) {
            // Keep the current solution and recycle the older output buffer.
            previous_parameters.swap(result.parameters);
            robust_residual.noalias() = problem.regressor * previous_parameters;
            robust_residual = problem.observation - robust_residual;
            if (!robust_residual.allFinite()) throw std::overflow_error("robust regression residuals are non-finite");
            for (Eigen::Index row = 0; row < robust_residual.size(); ++row) {
                const double absolute = std::abs(robust_residual[row]);
                weights[row] = RobustRowWeight(problem.row_weights[row], options.huber_delta, absolute);
            }
        }
    }
    result.predicted = problem.regressor * result.parameters;
    result.residual = problem.observation - result.predicted;
    // Weights are updated only when another solve remains, so this is the
    // vector consumed by the final solve, including at the iteration limit.
    result.final_row_weights = std::move(weights);
    workspace.SetDiagnostics(result);
    const auto errors = MeasureMagnitudes(result.residual);
    result.rmse = errors.rms;
    result.max_absolute_error = errors.maximum;
    result.iterations = iterations;
    result.converged = converged;
    result.covariance = workspace.Covariance(result.residual, result.rank);
    return result;
}

ValidationMetrics ValidateParameters(const RegressionProblem& problem, const Eigen::VectorXd& parameters) {
    if (!problem.Validate() || parameters.size() != problem.regressor.cols() || !parameters.allFinite()) {
        throw std::invalid_argument("validation inputs are invalid");
    }
    const Eigen::VectorXd residual = problem.observation - problem.regressor * parameters;
    ValidationMetrics metrics;
    metrics.row_count = static_cast<std::size_t>(residual.size());
    const auto errors = MeasureMagnitudes(residual);
    metrics.rmse = errors.rms;
    metrics.mean_absolute_error = errors.mean_absolute;
    metrics.max_absolute_error = errors.maximum;
    metrics.normalized_rmse = NormalizeRms(errors, MeasureMagnitudes(problem.observation));
    return metrics;
}

}  // namespace robot_identification
