#include "robot_identification/base_parameters.h"

#include <Eigen/QR>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "exact_parameter_dot.h"

namespace robot_identification {
namespace {
void ValidateRegressor(const Eigen::MatrixXd& values, Eigen::Index columns) {
    if (values.rows() == 0 || values.cols() != columns || !values.allFinite())
        throw std::invalid_argument("base parameter regressor dimensions or values are invalid");
}

void ValidateErrorTolerance(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0. || tolerance >= 1.)
        throw std::invalid_argument("base parameter reconstruction tolerance must be in [0, 1)");
}

double CheckedConversion(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("base parameter coordinates exceed the finite double range");
    const double result = static_cast<double>(value);
    if (value != 0.L && result == 0.)
        throw std::underflow_error("base parameter coordinates underflow the double range");
    return result;
}
}  // namespace

BaseParameterBasis::BaseParameterBasis(const Eigen::MatrixXd& calibration, double tolerance) {
    if (!std::isfinite(tolerance) || tolerance <= 0. || tolerance >= 1. || calibration.cols() == 0)
        throw std::invalid_argument("base parameter rank tolerance must be in (0, 1)");
    ValidateRegressor(calibration, calibration.cols());
    column_scales_ = calibration.cwiseAbs().colwise().maxCoeff().transpose();
    Eigen::MatrixXd normalized = calibration;
    for (Eigen::Index column = 0; column < normalized.cols(); ++column)
        if (column_scales_[column] != 0.) normalized.col(column) /= column_scales_[column];
    // Factor the private scratch in place; normalized outlives the decomposition.
    Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(normalized);
    qr.setThreshold(tolerance);
    const Eigen::Index rank = qr.rank();
    if (rank == 0) throw std::invalid_argument("calibration regressor has no identifiable parameter directions");
    const auto& permutation = qr.colsPermutation().indices();
    selected_columns_.reserve(static_cast<std::size_t>(rank));
    full_to_base_ = Eigen::MatrixXd::Zero(rank, calibration.cols());
    for (Eigen::Index index = 0; index < rank; ++index) {
        selected_columns_.push_back(static_cast<std::size_t>(permutation[index]));
        full_to_base_(index, permutation[index]) = column_scales_[permutation[index]];
    }
    if (rank < calibration.cols()) {
        const Eigen::MatrixXd dependencies = qr.matrixR()
                                                 .topLeftCorner(rank, rank)
                                                 .template triangularView<Eigen::Upper>()
                                                 .solve(qr.matrixR().topRightCorner(rank, calibration.cols() - rank));
        for (Eigen::Index column = rank; column < calibration.cols(); ++column)
            for (Eigen::Index row = 0; row < rank; ++row)
                full_to_base_(row, permutation[column]) = CheckedConversion(
                    static_cast<long double>(dependencies(row, column - rank)) * column_scales_[permutation[column]]);
    }
    calibration_error_ = ReconstructionError(calibration);
}

std::size_t BaseParameterBasis::parameter_count() const noexcept { return selected_columns_.size(); }
std::size_t BaseParameterBasis::full_parameter_count() const noexcept {
    return static_cast<std::size_t>(column_scales_.size());
}
const std::vector<std::size_t>& BaseParameterBasis::selected_columns() const noexcept { return selected_columns_; }
const Eigen::VectorXd& BaseParameterBasis::column_scales() const noexcept { return column_scales_; }
const Eigen::MatrixXd& BaseParameterBasis::full_to_base() const noexcept { return full_to_base_; }
double BaseParameterBasis::calibration_error() const noexcept { return calibration_error_; }

double BaseParameterBasis::ReconstructionError(const Eigen::MatrixXd& values) const {
    ValidateRegressor(values, column_scales_.size());
    // Selected columns are exact identities in the learned mapping. Replaying
    // normalization and multiplication can introduce a spurious wide residual.
    if (parameter_count() == full_parameter_count()) return 0.;
    if ((column_scales_.array() != 0.).count() == full_to_base_.rows()) {
        // All nonzero calibration columns are independent. Only zero-column
        // dependencies remain, with exact error zero or one. Check nonzero
        // entries directly so subnormal values cannot vanish in a norm.
        for (Eigen::Index column = 0; column < values.cols(); ++column)
            if (column_scales_[column] == 0. && (values.col(column).array() != 0.).any()) return 1.;
        return 0.;
    }
    Eigen::VectorXd divisors = column_scales_;
    // Divisors are nonnegative. Reuse negative entries to mark selected
    // identities instead of allocating a separate per-call bitmap.
    for (const auto column : selected_columns_) divisors[static_cast<Eigen::Index>(column)] = -1.;
    for (Eigen::Index column = 0; column < values.cols(); ++column) {
        if (divisors[column] < 0.) continue;
        divisors[column] = std::max(divisors[column], values.col(column).cwiseAbs().maxCoeff());
        if (divisors[column] == 0.) divisors[column] = 1.;
    }
    // Bound extended-precision scratch independently of calibration size.
    // An odd row count avoids the 4 KiB column stride of 256 extended values
    // on common platforms, while keeping the same per-row accumulation order.
    const Eigen::Index block_rows = std::min<Eigen::Index>(values.rows(), 255);
    Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> normalized(block_rows, full_to_base_.rows());
    long double maximum_error = 0.L;
    for (Eigen::Index begin = 0; begin < values.rows();) {
        const Eigen::Index count = std::min(block_rows, values.rows() - begin);
        for (Eigen::Index base = 0; base < full_to_base_.rows(); ++base) {
            const auto selected = static_cast<Eigen::Index>(selected_columns_[static_cast<std::size_t>(base)]);
            for (Eigen::Index row = 0; row < count; ++row)
                normalized(row, base) =
                    static_cast<long double>(values(begin + row, selected)) / column_scales_[selected];
        }
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            if (divisors[column] < 0.) continue;
            const long double divisor = divisors[column];
            long double maximum_difference = 0.L;
            const auto observe = [&](Eigen::Index row, long double reconstructed) {
                const long double difference = std::abs(reconstructed - values(begin + row, column));
                if (difference > maximum_difference)
                    maximum_difference = difference;
                else if (std::isnan(difference))
                    return false;
                return true;
            };
            // Reuse each coefficient for two adjacent rows. Each row keeps
            // the original increasing-base multiplication/addition order.
            Eigen::Index row = 0;
            for (; row + 1 < count; row += 2) {
                long double first = 0.L, second = 0.L;
                for (Eigen::Index base = 0; base < full_to_base_.rows(); ++base) {
                    const long double coefficient = full_to_base_(base, column);
                    first += normalized(row, base) * coefficient;
                    second += normalized(row + 1, base) * coefficient;
                }
                if (!observe(row, first) || !observe(row + 1, second)) return std::numeric_limits<double>::infinity();
            }
            if (row < count) {
                long double reconstructed = 0.L;
                for (Eigen::Index base = 0; base < full_to_base_.rows(); ++base)
                    reconstructed += normalized(row, base) * full_to_base_(base, column);
                if (!observe(row, reconstructed)) return std::numeric_limits<double>::infinity();
            }
            // The divisor is fixed within this column/block. Division and
            // the one-wide-step boundary correction are monotone in a
            // nonnegative residual, so only its maximum needs conversion.
            const long double difference = maximum_difference;
            long double error = difference / divisor;
            if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
            if (error >= maximum_error && difference != 0.L) {
                // A wide quotient can land exactly on a double boundary
                // while its exact ratio is slightly larger. Only such
                // candidates need the fused division-residual check.
                if (error <= std::numeric_limits<double>::max() &&
                    static_cast<long double>(static_cast<double>(error)) == error &&
                    std::fma(-error, divisor, difference) > 0.L)
                    error = std::nextafter(error, std::numeric_limits<long double>::infinity());
                maximum_error = error;
            }
        }
        begin += count;
    }
    if (maximum_error > std::numeric_limits<double>::max()) return std::numeric_limits<double>::infinity();
    const double rounded = static_cast<double>(maximum_error);
    // Do not hide a positive residual at zero or round a value down to the
    // caller's tolerance when converting the wide maximum to double.
    return static_cast<long double>(rounded) < maximum_error
               ? std::nextafter(rounded, std::numeric_limits<double>::infinity())
               : rounded;
}

Eigen::MatrixXd BaseParameterBasis::ReduceRegressor(const Eigen::MatrixXd& values, double tolerance) const {
    ValidateErrorTolerance(tolerance);
    if (ReconstructionError(values) > tolerance)
        throw std::invalid_argument("regressor violates the learned base parameter dependencies");
    Eigen::MatrixXd reduced(values.rows(), static_cast<Eigen::Index>(parameter_count()));
    for (Eigen::Index column = 0; column < reduced.cols(); ++column) {
        const auto selected = static_cast<Eigen::Index>(selected_columns_[static_cast<std::size_t>(column)]);
        for (Eigen::Index row = 0; row < values.rows(); ++row)
            reduced(row, column) =
                CheckedConversion(static_cast<long double>(values(row, selected)) / column_scales_[selected]);
    }
    return reduced;
}

RegressionProblem BaseParameterBasis::ReduceProblem(const RegressionProblem& problem, double tolerance) const {
    std::string reason;
    if (!problem.Validate(&reason)) throw std::invalid_argument("invalid regression problem: " + reason);
    RegressionProblem reduced;
    reduced.regressor = ReduceRegressor(problem.regressor, tolerance);
    reduced.observation = problem.observation;
    reduced.row_weights = problem.row_weights;
    reduced.sample_count = problem.sample_count;
    reduced.dof = problem.dof;
    return reduced;
}

Eigen::VectorXd BaseParameterBasis::ToBaseParameters(const Eigen::VectorXd& parameters) const {
    if (parameters.size() != full_to_base_.cols() || !parameters.allFinite())
        throw std::invalid_argument("full parameter dimensions or values are invalid");
    Eigen::VectorXd result(full_to_base_.rows());
    const bool use_wide_screen = detail::kWideParameterDot && std::fegetround() == FE_TONEAREST;
    const bool independent_nonzero_columns =
        parameter_count() == full_parameter_count() || (column_scales_.array() != 0.).count() == full_to_base_.rows();
    for (Eigen::Index row = 0; row < result.size(); ++row) {
        if constexpr (detail::kBinary64ParameterDot) {
            if (independent_nonzero_columns) {
                const auto selected = static_cast<Eigen::Index>(selected_columns_[static_cast<std::size_t>(row)]);
                result[row] = detail::ParameterProduct(column_scales_[selected], parameters[selected], use_wide_screen);
            } else {
                result[row] = detail::ParameterDot(full_to_base_, row, parameters, use_wide_screen);
            }
        } else {
            long double value = 0.L;
            for (Eigen::Index column = 0; column < parameters.size(); ++column)
                value += static_cast<long double>(full_to_base_(row, column)) * parameters[column];
            result[row] = CheckedConversion(value);
        }
    }
    return result;
}

BaseParameterRegressor::BaseParameterRegressor(std::shared_ptr<const RegressorBackend> backend,
                                               BaseParameterBasis basis,
                                               double tolerance)
    : backend_(std::move(backend)), basis_(std::move(basis)), maximum_reconstruction_error_(tolerance) {
    ValidateErrorTolerance(tolerance);
    if (!backend_ || backend_->parameter_count() != basis_.full_parameter_count() ||
        basis_.calibration_error() > tolerance)
        throw std::invalid_argument("backend does not match the calibrated base parameter basis");
    const auto names = backend_->parameter_names();
    std::vector<std::string_view> sorted(names.begin(), names.end());
    std::sort(sorted.begin(), sorted.end());
    if (names.size() != backend_->parameter_count() || sorted.empty() || sorted.front().empty() ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument("base parameter backend names are invalid");
    for (const auto column : basis_.selected_columns()) parameter_names_.push_back("base/" + names[column]);
}

std::size_t BaseParameterRegressor::parameter_count() const noexcept { return basis_.parameter_count(); }
std::vector<std::string> BaseParameterRegressor::parameter_names() const { return parameter_names_; }
Eigen::MatrixXd BaseParameterRegressor::Compute(const Sample& sample) const {
    return basis_.ReduceRegressor(backend_->Compute(sample), maximum_reconstruction_error_);
}
const BaseParameterBasis& BaseParameterRegressor::basis() const noexcept { return basis_; }
}  // namespace robot_identification
