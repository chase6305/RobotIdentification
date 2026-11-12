#include "robot_identification/base_parameters.h"

#include <Eigen/QR>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "exact_parameter_dot.h"
#include "name_validation.h"

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
    Eigen::MatrixXd qr_storage = calibration;
    for (Eigen::Index column = 0; column < qr_storage.cols(); ++column)
        if (column_scales_[column] != 0.) qr_storage.col(column) /= column_scales_[column];
    // QR borrows this normalized copy; the caller's calibration is never modified.
    // Declare storage before the decomposition so it remains alive until QR is destroyed.
    Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(qr_storage);
    qr.setThreshold(tolerance);
    const Eigen::Index rank = qr.rank();
    if (rank == 0) throw std::invalid_argument("calibration regressor has no identifiable parameter directions");
    const auto& column_permutation = qr.colsPermutation().indices();
    selected_columns_.reserve(static_cast<std::size_t>(rank));
    full_to_base_ = Eigen::MatrixXd::Zero(rank, calibration.cols());
    for (Eigen::Index base_column = 0; base_column < rank; ++base_column) {
        const Eigen::Index full_column = column_permutation[base_column];
        selected_columns_.push_back(static_cast<std::size_t>(full_column));
        full_to_base_(base_column, full_column) = column_scales_[full_column];
    }
    // R11 * dependencies = R12 relates normalized dependent and selected columns.
    if (rank < calibration.cols()) {
        const Eigen::MatrixXd normalized_dependencies =
            qr.matrixR()
                .topLeftCorner(rank, rank)
                .template triangularView<Eigen::Upper>()
                .solve(qr.matrixR().topRightCorner(rank, calibration.cols() - rank));
        for (Eigen::Index column = rank; column < calibration.cols(); ++column)
            for (Eigen::Index row = 0; row < rank; ++row)
                full_to_base_(row, column_permutation[column]) =
                    CheckedConversion(static_cast<long double>(normalized_dependencies(row, column - rank)) *
                                      column_scales_[column_permutation[column]]);
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

double BaseParameterBasis::ReconstructionError(const Eigen::MatrixXd& regressor) const {
    ValidateRegressor(regressor, column_scales_.size());
    // Selected columns are exact identities in the learned mapping. Replaying
    // normalization and multiplication can introduce a spurious wide residual.
    if (parameter_count() == full_parameter_count()) return 0.;
    if ((column_scales_.array() != 0.).count() == full_to_base_.rows()) {
        // All nonzero calibration columns are independent. Only zero-column
        // dependencies remain, with exact relative error zero or one. Check nonzero
        // entries directly so subnormal values cannot vanish in a norm.
        for (Eigen::Index column = 0; column < regressor.cols(); ++column)
            if (column_scales_[column] == 0. && (regressor.col(column).array() != 0.).any()) return 1.;
        return 0.;
    }

    long double maximum_relative_error = 0.L;
    Eigen::VectorXd column_divisors = column_scales_;
    constexpr double kSkippedColumnMarker = -1.;
    // Divisors are nonnegative. Reuse negative entries to skip selected
    // identities and zero dependencies without a separate per-call bitmap.
    for (const auto column : selected_columns_)
        column_divisors[static_cast<Eigen::Index>(column)] = kSkippedColumnMarker;
    for (Eigen::Index column = 0; column < regressor.cols(); ++column) {
        if (column_divisors[column] < 0.) continue;
        if (column_scales_[column] == 0.) {
            // A zero calibration column has an exact zero mapping, even
            // alongside nonzero dependencies. Its relative error is 0 or 1.
            // Keep checking other columns: their errors can exceed one.
            if ((regressor.col(column).array() != 0.).any()) maximum_relative_error = 1.L;
            column_divisors[column] = kSkippedColumnMarker;
            continue;
        }
        column_divisors[column] = std::max(column_divisors[column], regressor.col(column).cwiseAbs().maxCoeff());
    }

    // Bound extended-precision scratch independently of calibration size.
    // An odd row count avoids the 4 KiB column stride of 256 long doubles
    // on common platforms, while keeping the same per-row accumulation order.
    const Eigen::Index rows_per_block = std::min<Eigen::Index>(regressor.rows(), 255);
    Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> normalized_basis(rows_per_block, full_to_base_.rows());

    for (Eigen::Index block_start = 0; block_start < regressor.rows();) {
        const Eigen::Index block_row_count = std::min(rows_per_block, regressor.rows() - block_start);
        for (Eigen::Index base_column = 0; base_column < full_to_base_.rows(); ++base_column) {
            const auto selected_column =
                static_cast<Eigen::Index>(selected_columns_[static_cast<std::size_t>(base_column)]);
            for (Eigen::Index row = 0; row < block_row_count; ++row)
                normalized_basis(row, base_column) =
                    static_cast<long double>(regressor(block_start + row, selected_column)) /
                    column_scales_[selected_column];
        }
        for (Eigen::Index column = 0; column < regressor.cols(); ++column) {
            if (column_divisors[column] < 0.) continue;
            const long double column_divisor = column_divisors[column];
            long double maximum_absolute_difference = 0.L;
            const auto accumulate_difference = [&](Eigen::Index row, long double reconstructed) {
                const long double difference = std::abs(reconstructed - regressor(block_start + row, column));
                if (difference > maximum_absolute_difference)
                    maximum_absolute_difference = difference;
                else if (std::isnan(difference))
                    return false;
                return true;
            };
            // Reuse each coefficient for two adjacent rows. Each row keeps
            // the same multiplication/addition order over increasing base columns.
            Eigen::Index row = 0;
            for (; row + 1 < block_row_count; row += 2) {
                long double first_reconstruction = 0.L, second_reconstruction = 0.L;
                for (Eigen::Index base_column = 0; base_column < full_to_base_.rows(); ++base_column) {
                    const long double coefficient = full_to_base_(base_column, column);
                    first_reconstruction += normalized_basis(row, base_column) * coefficient;
                    second_reconstruction += normalized_basis(row + 1, base_column) * coefficient;
                }
                if (!accumulate_difference(row, first_reconstruction) ||
                    !accumulate_difference(row + 1, second_reconstruction))
                    return std::numeric_limits<double>::infinity();
            }
            if (row < block_row_count) {
                long double reconstructed = 0.L;
                for (Eigen::Index base_column = 0; base_column < full_to_base_.rows(); ++base_column)
                    reconstructed += normalized_basis(row, base_column) * full_to_base_(base_column, column);
                if (!accumulate_difference(row, reconstructed)) return std::numeric_limits<double>::infinity();
            }
            // The divisor is fixed within this column/block. Division and
            // the one-wide-step boundary correction are monotone in a
            // nonnegative residual, so only its maximum needs conversion.
            const long double difference = maximum_absolute_difference;
            long double relative_error = difference / column_divisor;
            if (!std::isfinite(relative_error)) return std::numeric_limits<double>::infinity();
            if (relative_error >= maximum_relative_error && difference != 0.L) {
                // A wide quotient can land exactly on a double boundary
                // while its exact ratio is slightly larger. Only such
                // candidates need the fused division-residual check.
                if (relative_error <= std::numeric_limits<double>::max() &&
                    static_cast<long double>(static_cast<double>(relative_error)) == relative_error &&
                    std::fma(-relative_error, column_divisor, difference) > 0.L)
                    relative_error = std::nextafter(relative_error, std::numeric_limits<long double>::infinity());
                maximum_relative_error = relative_error;
            }
        }
        block_start += block_row_count;
    }

    if (maximum_relative_error > std::numeric_limits<double>::max()) return std::numeric_limits<double>::infinity();
    const double rounded = static_cast<double>(maximum_relative_error);
    // Do not hide a positive residual at zero or round a value down to the
    // caller's tolerance when converting the wide maximum to double.
    return static_cast<long double>(rounded) < maximum_relative_error
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
    const auto full_parameter_names = backend_->parameter_names();
    if (full_parameter_names.size() != backend_->parameter_count() ||
        !detail::HasUniqueNonemptyNames(full_parameter_names))
        throw std::invalid_argument("base parameter backend names are invalid");
    parameter_names_.reserve(basis_.parameter_count());
    for (const auto full_column : basis_.selected_columns())
        parameter_names_.push_back("base/" + full_parameter_names[full_column]);
}

std::size_t BaseParameterRegressor::parameter_count() const noexcept { return basis_.parameter_count(); }
std::vector<std::string> BaseParameterRegressor::parameter_names() const { return parameter_names_; }
Eigen::MatrixXd BaseParameterRegressor::Compute(const Sample& sample) const {
    return basis_.ReduceRegressor(backend_->Compute(sample), maximum_reconstruction_error_);
}
const BaseParameterBasis& BaseParameterRegressor::basis() const noexcept { return basis_; }
}  // namespace robot_identification
