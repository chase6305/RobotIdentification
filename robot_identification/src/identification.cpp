#include "robot_identification/identification.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "robot_identification/detail/regression_assembly.h"

namespace robot_identification {
namespace {

bool Fail(std::string* reason, const std::string& message) {
    if (reason) *reason = message;
    return false;
}

bool HasUniqueNonemptyNames(const std::vector<std::string>& names) {
    if (names.empty()) return false;
    std::vector<std::string_view> ordered(names.begin(), names.end());
    std::sort(ordered.begin(), ordered.end());
    return !ordered.front().empty() && std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end();
}

bool ValidateRawInputs(const std::vector<std::string>& joint_names,
                       const Eigen::VectorXd& time_seconds,
                       const Eigen::MatrixXd& position,
                       const Eigen::MatrixXd& torque,
                       std::string* reason) {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    if (!HasUniqueNonemptyNames(joint_names)) {
        return Fail(reason, "joint names must be non-empty and unique");
    }
    if (time_seconds.size() < 3 || position.rows() != time_seconds.size() || torque.rows() != time_seconds.size() ||
        position.cols() != static_cast<Eigen::Index>(joint_names.size()) || torque.cols() != position.cols() ||
        !time_seconds.allFinite() || !position.allFinite() || !torque.allFinite()) {
        return Fail(reason, "raw dataset dimensions are inconsistent or values are non-finite");
    }
    for (Eigen::Index index = 1; index < time_seconds.size(); ++index) {
        if (time_seconds[index] <= time_seconds[index - 1]) {
            return Fail(reason, "raw timestamps must be strictly increasing");
        }
    }
    return true;
}

struct ErrorMagnitudes {
    double rms{0.0};
    double mean_absolute{0.0};
    double maximum{0.0};
    double rms_fraction{0.0};
};

double RobustRowWeight(double base_weight, double delta, double residual_magnitude) {
    if (residual_magnitude <= delta) return base_weight;
    const double factor = delta / residual_magnitude;
    if (factor >= std::numeric_limits<double>::min()) return base_weight * factor;
    // A subnormal ratio can lose precision or become zero before a large
    // base weight restores a representable result. Combine mantissas first.
    int base_exponent, delta_exponent, residual_exponent;
    const double base_fraction = std::frexp(base_weight, &base_exponent);
    const double delta_fraction = std::frexp(delta, &delta_exponent);
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

ErrorMagnitudes MeasureErrors(const Eigen::VectorXd& values) {
    if (!values.allFinite()) throw std::overflow_error("error metrics require finite predictions and residuals");
    ErrorMagnitudes result;
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
    const long double rms_fraction = std::min(1.0L, std::sqrt(squared_sum / values.size()));
    result.rms_fraction = static_cast<double>(rms_fraction);
    result.rms = static_cast<double>(rms_fraction * result.maximum);
    result.mean_absolute = static_cast<double>(std::min(1.0L, absolute_sum / values.size()) * result.maximum);
    return result;
}

double NormalizeRms(const ErrorMagnitudes& error, const ErrorMagnitudes& reference) {
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
    return std::ldexp((error_fraction / reference_fraction) * (error.rms_fraction / reference.rms_fraction),
                      error_exponent - reference_exponent);
}

Eigen::MatrixXd SmoothSymmetric(const Eigen::MatrixXd& values, std::size_t window) {
    const Eigen::Index radius =
        static_cast<Eigen::Index>(std::min(window / 2, static_cast<std::size_t>(values.rows() - 1)));
    Eigen::MatrixXd output(values.rows(), values.cols());
    // Eigen's short reductions are faster for the usual small windows.
    if (radius <= 32) {
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
            output.row(row) = values.middleRows(begin, end - begin + 1).colwise().mean();
            for (Eigen::Index column = 0; column < values.cols(); ++column) {
                if (std::isfinite(output(row, column))) continue;
                // A finite mean can have an overflowing intermediate sum.
                // Normalize only exceptional windows, keeping the common path fast.
                const double scale = values.col(column).segment(begin, end - begin + 1).cwiseAbs().maxCoeff();
                long double normalized_sum = 0.0L;
                for (Eigen::Index index = begin; index <= end; ++index) {
                    normalized_sum += static_cast<long double>(values(index, column)) / scale;
                }
                const long double mean = normalized_sum / (end - begin + 1);
                output(row, column) = static_cast<double>(std::clamp(mean, -1.0L, 1.0L) * scale);
            }
        }
        return output;
    }
    const Eigen::Index rebase_interval =
        static_cast<Eigen::Index>(std::min(window, static_cast<std::size_t>(values.rows())));
    for (Eigen::Index column = 0; column < values.cols(); ++column) {
        long double sum = 0.0L;
        long double correction = 0.0L;
        const auto add = [&](long double value) {
            const long double next = sum + value;
            correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
            sum = next;
        };
        Eigen::Index previous_begin = 0;
        Eigen::Index previous_end = -1;
        Eigen::Index until_rebase = 0;
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = row + std::min(radius, values.rows() - 1 - row);
            if (until_rebase == 0) {
                // Rebuild once per window to bound roundoff accumulation;
                // the total work remains linear in the number of samples.
                sum = correction = 0.0L;
                for (Eigen::Index index = begin; index <= end; ++index) add(values(index, column));
                until_rebase = rebase_interval;
            } else {
                if (begin != previous_begin) add(-static_cast<long double>(values(previous_begin, column)));
                if (end != previous_end) add(values(end, column));
            }
            output(row, column) = static_cast<double>((sum + correction) / (end - begin + 1));
            previous_begin = begin;
            previous_end = end;
            --until_rebase;
        }
    }
    return output;
}

Eigen::MatrixXd Differentiate(const Eigen::VectorXd& time, const Eigen::MatrixXd& values) {
    // Internal preprocessing helper: dimensions and timestamps were checked by
    // ValidateRawInputs before smoothing. Keep arithmetic checks below.
    Eigen::MatrixXd derivative(values.rows(), values.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        const Eigen::Index first = row == 0 ? 0 : row + 1 == values.rows() ? row - 2 : row - 1;
        const Eigen::Index second = first + 1;
        const Eigen::Index third = first + 2;
        const double left_step = time[second] - time[first];
        const double right_step = time[third] - time[second];
        const double span = left_step + right_step;
        if (!std::isfinite(span)) {
            throw std::overflow_error("timestamp spacing is numerically unusable for differentiation");
        }
        // Differentiate the same three-point quadratic through its secant
        // slopes, so a constant position offset cancels before division.
        // At an interior point, start from the slope with the larger weight.
        // Otherwise a weight rounded to one can cancel the base slope and
        // erase the smaller contribution entirely on strongly unequal steps.
        const bool use_right_slope = row == third || (row != first && left_step > right_step);
        const double ratio = row == first      ? -left_step / span
                             : row == third    ? right_step / span
                             : use_right_slope ? -right_step / span
                                               : left_step / span;
        const bool ratio_underflow = std::abs(ratio) < std::numeric_limits<double>::min();
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            const double left_slope = (values(second, column) - values(first, column)) / left_step;
            const double right_slope = (values(third, column) - values(second, column)) / right_step;
            const double base = use_right_slope ? right_slope : left_slope;
            derivative(row, column) = base + ratio * (right_slope - left_slope);
            if (ratio_underflow || !std::isfinite(derivative(row, column))) {
                // Finite final derivatives can have overflowing differences or
                // individual secants, or an underflowed weight whose product
                // is representable. Retry these coefficients in extended range.
                const long double wide_left =
                    (static_cast<long double>(values(second, column)) - values(first, column)) / left_step;
                const long double wide_right =
                    (static_cast<long double>(values(third, column)) - values(second, column)) / right_step;
                const long double wide_span = static_cast<long double>(left_step) + right_step;
                const long double wide_ratio = row == first      ? -static_cast<long double>(left_step) / wide_span
                                               : row == third    ? static_cast<long double>(right_step) / wide_span
                                               : use_right_slope ? -static_cast<long double>(right_step) / wide_span
                                                                 : static_cast<long double>(left_step) / wide_span;
                const long double wide_base = use_right_slope ? wide_right : wide_left;
                const long double value = wide_base + wide_ratio * (wide_right - wide_left);
                if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) {
                    throw std::overflow_error("differentiation produced non-finite values");
                }
                derivative(row, column) = static_cast<double>(value);
            }
        }
    }
    return derivative;
}

bool ReadCsvHeader(std::istream& input, std::string& line, char& separator) {
    line.clear();
    separator = '\n';
    char character;
    while (input.get(character)) {
        if (character == '\n') return true;
        if (character == '\r') {
            if (input.peek() == '\n') {
                input.get();
            } else {
                separator = '\r';
            }
            return true;
        }
        line.push_back(character);
    }
    return !line.empty();
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    // ReadCsvHeader already consumed the record separator. Copy fields
    // directly, including an empty field after a final comma.
    if (line.empty()) return fields;
    for (std::size_t begin = 0;;) {
        const auto end = line.find(',', begin);
        fields.emplace_back(line, begin, end == std::string::npos ? end : end - begin);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return fields;
}

void ValidateCsvJointNames(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        if (name.empty() || name.find_first_of(",\r\n") != std::string::npos) {
            throw std::runtime_error("CSV joint names must be non-empty and contain no commas or newlines");
        }
    }
    if (!HasUniqueNonemptyNames(names)) {
        throw std::runtime_error("CSV joint names must be unique");
    }
}

std::ofstream OpenCsvOutput(const std::string& path,
                            const std::vector<std::string>& joint_names,
                            std::initializer_list<const char*> prefixes,
                            const char* kind) {
    if (path.empty()) throw std::invalid_argument(std::string(kind) + " output path must not be empty");
    for (const auto& name : joint_names)
        if (name.find_first_of(",\r\n") != std::string::npos)
            throw std::invalid_argument("joint names in CSV must not contain commas or newlines");
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open " + std::string(kind) + " output: " + path);
    output.imbue(std::locale::classic());
    output << "time_s";
    for (const char* prefix : prefixes)
        for (const auto& name : joint_names) output << ',' << prefix << name;
    output << '\n' << std::setprecision(17);
    return output;
}

bool ParseCsvRow(std::string& line, std::vector<double>& values, std::size_t line_number, double& previous_time) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const char* cursor = line.data();
    const char* end = cursor + line.size();
    if (cursor == end) return false;

    // from_chars bounds conversion to each field without changing the buffer
    // or consulting the process locale. Preserve leading ASCII whitespace,
    // an explicit plus sign, and the existing 0x-prefixed hexadecimal syntax.
    std::size_t column = 0;
    while (true) {
        if (column == values.size()) {
            throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
        }
        const char* field_end = std::find(cursor, end, ',');
        const char* number = cursor;
        while (number != field_end && (*number == ' ' || (*number >= '\t' && *number <= '\r'))) ++number;
        const bool negative = number != field_end && *number == '-';
        if (number != field_end && (*number == '+' || *number == '-')) ++number;
        auto format = std::chars_format::general;
        if (field_end - number >= 2 && number[0] == '0' && (number[1] == 'x' || number[1] == 'X')) {
            number += 2;
            format = std::chars_format::hex;
        }
        const auto location = [&]() {
            return " at line " + std::to_string(line_number) + ", column " + std::to_string(column + 1);
        };
        if (number == field_end || *number == '+' || *number == '-') {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        if (format == std::chars_format::hex) {
            auto exponent = std::find_if(number, field_end, [](char value) { return value == 'p' || value == 'P'; });
            if (exponent != field_end) {
                ++exponent;
                if (exponent != field_end && (*exponent == '+' || *exponent == '-')) ++exponent;
                // Some from_chars implementations accept a repeated exponent
                // sign such as p+-2. Require a digit after the optional sign.
                if (exponent == field_end || *exponent < '0' || *exponent > '9')
                    throw std::invalid_argument("CSV hexadecimal exponent is invalid" + location());
            }
        }
        double value = 0.0;
        const auto conversion = std::from_chars(number, field_end, value, format);
        if (conversion.ec == std::errc::invalid_argument) {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        if (conversion.ec == std::errc::result_out_of_range) {
            throw std::out_of_range("CSV numeric field is out of range" + location());
        }
        if (conversion.ptr != field_end || !std::isfinite(value)) {
            throw std::runtime_error("CSV numeric field is invalid" + location());
        }
        values[column++] = negative ? -value : value;
        if (field_end == end) break;
        cursor = field_end + 1;
    }
    if (column != values.size()) {
        throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
    }
    if (values[0] <= previous_time) {
        throw std::runtime_error("CSV timestamp must be strictly increasing at line " + std::to_string(line_number));
    }
    previous_time = values[0];
    return true;
}

class CsvInput {
public:
    CsvInput(const std::string& path, std::initializer_list<const char*> following_prefixes, const char* kind)
        : input_(path), path_(path), kind_(kind) {
        if (path.empty()) throw std::invalid_argument(std::string(kind) + " input path must not be empty");
        if (!input_) throw std::runtime_error("cannot open " + std::string(kind) + " input: " + path);
        if (!ReadCsvHeader(input_, line_, separator_)) throw std::runtime_error(std::string(kind) + " CSV is empty");
        const auto header = SplitCsvLine(line_);
        const auto groups = 1 + following_prefixes.size();
        if (header.size() <= groups || header[0] != "time_s" || (header.size() - 1) % groups != 0)
            throw std::runtime_error(std::string(kind) + " CSV header has an invalid shape");
        const auto dof = (header.size() - 1) / groups;
        const auto maximum_values = std::min(static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                                             std::numeric_limits<std::size_t>::max() / sizeof(double));
        if (header.size() > maximum_values) throw std::overflow_error(std::string(kind) + " CSV has too many columns");
        joint_names.reserve(dof);
        for (std::size_t joint = 0; joint < dof; ++joint) {
            const auto& field = header[1 + joint];
            if (field.rfind("q/", 0) != 0 || field.size() <= 2) throw std::runtime_error("invalid q column name");
            joint_names.push_back(field.substr(2));
        }
        std::size_t column = 1 + dof;
        for (const char* prefix : following_prefixes)
            for (const auto& joint : joint_names)
                if (header[column++] != prefix + joint)
                    throw std::runtime_error(std::string(kind) + " CSV columns do not share the same joint order");
        ValidateCsvJointNames(joint_names);
        values.resize(header.size());
    }

    bool ReadRow() {
        while (std::getline(input_, line_, separator_)) {
            ++line_number_;
            if (ParseCsvRow(line_, values, line_number_, previous_time_)) return true;
        }
        if (input_.bad() || !input_.eof())
            throw std::runtime_error("failed while reading " + std::string(kind_) + ": " + path_);
        return false;
    }

    std::vector<std::string> joint_names;
    std::vector<double> values;

private:
    std::ifstream input_;
    const std::string& path_;
    const char* kind_;
    std::string line_;
    char separator_;
    std::size_t line_number_{1};
    double previous_time_{-std::numeric_limits<double>::infinity()};
};

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
        if (diagnostic_svd_->info() != Eigen::Success || !diagnostic_svd_->singularValues().allFinite()) {
            throw std::runtime_error("regression diagnostic decomposition failed");
        }
        result.rank = static_cast<std::size_t>(diagnostic_svd_->rank());
        const auto& singular_values = diagnostic_svd_->singularValues();
        result.condition_number = result.rank < static_cast<std::size_t>(problem.regressor.cols())
                                      ? std::numeric_limits<double>::infinity()
                                      : singular_values[0] / singular_values[singular_values.size() - 1];
        if (ridge_rows_ == 0) {
            result.parameters = diagnostic_svd_->solve(weighted_observation_);
        } else {
            regularized_svd_->compute(weighted_regressor_);
            if (regularized_svd_->info() != Eigen::Success || !regularized_svd_->singularValues().allFinite()) {
                throw std::runtime_error("regularized regression decomposition failed");
            }
            result.parameters = regularized_svd_->solve(weighted_observation_);
        }
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
        const Eigen::Index rank = decomposition.rank();
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

bool Dataset::Validate(std::string* reason) const {
    if (joint_names.empty()) return Fail(reason, "joint_names must not be empty");
    if (samples.empty()) return Fail(reason, "dataset must contain samples");
    if (!HasUniqueNonemptyNames(joint_names)) {
        return Fail(reason, "joint names must be non-empty and unique");
    }
    double previous_time = -std::numeric_limits<double>::infinity();
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.time_seconds) || sample.time_seconds <= previous_time) {
            return Fail(reason, "sample timestamps must be finite and strictly increasing");
        }
        previous_time = sample.time_seconds;
        for (const auto* vector : {&sample.position, &sample.velocity, &sample.acceleration, &sample.torque}) {
            if (vector->size() != static_cast<Eigen::Index>(dof()) || !vector->allFinite()) {
                return Fail(reason, "sample vectors must match the joint count and be finite");
            }
        }
    }
    return true;
}

bool RawDataset::Validate(std::string* reason) const {
    return ValidateRawInputs(joint_names, time_seconds, position, torque, reason);
}

HarmonicTrajectory::HarmonicTrajectory(Eigen::VectorXd offset,
                                       Eigen::MatrixXd sin_coefficients,
                                       Eigen::MatrixXd cos_coefficients,
                                       double fundamental_frequency_hz)
    : offset_(std::move(offset)),
      sin_coefficients_(std::move(sin_coefficients)),
      cos_coefficients_(std::move(cos_coefficients)),
      angular_frequency_(2.0 * std::acos(-1.0) * fundamental_frequency_hz) {
    if (offset_.size() == 0 || sin_coefficients_.rows() != offset_.size() ||
        cos_coefficients_.rows() != offset_.size() || sin_coefficients_.cols() == 0 ||
        cos_coefficients_.cols() != sin_coefficients_.cols() || !offset_.allFinite() ||
        !sin_coefficients_.allFinite() || !cos_coefficients_.allFinite() || !std::isfinite(angular_frequency_) ||
        angular_frequency_ <= 0.0) {
        throw std::invalid_argument("harmonic trajectory configuration is invalid");
    }
}

std::size_t HarmonicTrajectory::dof() const noexcept { return static_cast<std::size_t>(offset_.size()); }

std::size_t HarmonicTrajectory::harmonic_count() const noexcept {
    return static_cast<std::size_t>(sin_coefficients_.cols());
}

double HarmonicTrajectory::period_seconds() const noexcept { return 2.0 * std::acos(-1.0) / angular_frequency_; }

ExcitationSample HarmonicTrajectory::Evaluate(double time_seconds) const {
    ExcitationSample sample;
    EvaluateInto(time_seconds, sample);
    return sample;
}

void HarmonicTrajectory::EvaluateInto(double time_seconds, ExcitationSample& sample) const {
    if (!std::isfinite(time_seconds)) throw std::invalid_argument("trajectory time must be finite");
    sample.position = offset_;
    sample.velocity.setZero(offset_.size());
    sample.acceleration.setZero(offset_.size());
    sample.jerk.setZero(offset_.size());
    bool retry_underflow = false;
    for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
        if (sin_coefficients_.col(harmonic).isZero(0.) && cos_coefficients_.col(harmonic).isZero(0.)) continue;
        const double frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
        const double phase = frequency * time_seconds;
        const double sine = std::sin(phase);
        const double cosine = std::cos(phase);
        retry_underflow |= time_seconds != 0. && std::abs(phase) < std::numeric_limits<double>::min();
        for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
            const double a = sin_coefficients_(joint, harmonic), b = cos_coefficients_(joint, harmonic);
            const double position_term = a * sine + b * cosine;
            const double derivative_term = a * cosine - b * sine;
            // Retry nonzero products that enter the subnormal range before
            // frequency factors or other harmonics can restore their scale.
            retry_underflow |= (std::abs(position_term) < std::numeric_limits<double>::min() &&
                                ((a != 0. && sine != 0.) || (b != 0. && cosine != 0.))) ||
                               (std::abs(derivative_term) < std::numeric_limits<double>::min() &&
                                ((a != 0. && cosine != 0.) || (b != 0. && sine != 0.)));
            sample.position[joint] += position_term;
            // Multiply the finite coefficient before successive frequency
            // factors: powers of frequency may overflow or underflow even
            // when every requested derivative remains representable.
            const double velocity_term = frequency * derivative_term;
            const double acceleration_term = frequency * (frequency * position_term);
            const double jerk_term = frequency * (frequency * velocity_term);
            // With a frequency below one, a later derivative can underflow
            // even when its coefficient combination is normal. Multiple such
            // harmonics can still have a representable combined derivative.
            retry_underflow |=
                frequency < 1. &&
                ((std::abs(velocity_term) < std::numeric_limits<double>::min() && derivative_term != 0.) ||
                 (std::abs(acceleration_term) < std::numeric_limits<double>::min() && position_term != 0.) ||
                 (std::abs(jerk_term) < std::numeric_limits<double>::min() && derivative_term != 0.));
            sample.velocity[joint] += velocity_term;
            sample.acceleration[joint] -= acceleration_term;
            sample.jerk[joint] -= jerk_term;
        }
    }
    if (!retry_underflow && sample.position.allFinite() && sample.velocity.allFinite() &&
        sample.acceleration.allFinite() && sample.jerk.allFinite())
        return;
    // Keep exceptional products and sums in extended range until all harmonic
    // contributions have been combined. Reuse the existing owning outputs.
    for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
        std::array<long double, 4> values{offset_[joint], 0.L, 0.L, 0.L};
        for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
            const long double a = sin_coefficients_(joint, harmonic), b = cos_coefficients_(joint, harmonic);
            if (a == 0.L && b == 0.L) continue;
            const double frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
            const double phase = frequency * time_seconds;
            if (!std::isfinite(phase)) throw std::overflow_error("harmonic trajectory evaluation overflowed");
            const bool small_phase = time_seconds != 0. && std::abs(phase) < std::numeric_limits<double>::min();
            const long double wide_phase = static_cast<long double>(frequency) * time_seconds;
            const long double sine = small_phase ? std::sin(wide_phase) : std::sin(phase);
            const long double cosine = small_phase ? std::cos(wide_phase) : std::cos(phase);
            const long double position = a * sine + b * cosine;
            const long double velocity = static_cast<long double>(frequency) * (a * cosine - b * sine);
            values[0] += position;
            values[1] += velocity;
            values[2] -= static_cast<long double>(frequency) * frequency * position;
            values[3] -= static_cast<long double>(frequency) * frequency * velocity;
        }
        const std::array<Eigen::VectorXd*, 4> outputs{&sample.position, &sample.velocity, &sample.acceleration,
                                                      &sample.jerk};
        for (std::size_t derivative = 0; derivative < outputs.size(); ++derivative) {
            if (!std::isfinite(values[derivative]) || std::abs(values[derivative]) > std::numeric_limits<double>::max())
                throw std::overflow_error("harmonic trajectory evaluation overflowed");
            (*outputs[derivative])[joint] = static_cast<double>(values[derivative]);
        }
    }
}

HarmonicBounds HarmonicTrajectory::ConservativeBounds() const {
    const auto upper_sum = [](long double left, long double right) {
        // FastTwoSum recovers the addition's rounding residual. Advance only
        // when the rounded sum is below the exact sum; exact sums stay exact.
        if (std::abs(left) < std::abs(right)) std::swap(left, right);
        const long double sum = left + right;
        const long double residual = right - (sum - left);
        return residual > 0.L ? std::nextafter(sum, std::numeric_limits<long double>::infinity()) : sum;
    };
    const auto accumulate = [&](long double& sum, long double& positive_error, long double term) {
        long double left = sum;
        if (std::abs(left) < std::abs(term)) std::swap(left, term);
        sum = left + term;
        const long double residual = term - (sum - left);
        // Track only downward rounding losses. Their small positive sum needs
        // far fewer nextafter calls than rounding every main addition upward.
        if (residual > 0.L) positive_error = upper_sum(positive_error, residual);
    };
    const auto upper_product = [](long double left, long double right) {
        const long double product = left * right;
        if (left == 0.L || right == 0.L || left == 1.L || right == 1.L) return product;
        // A fused residual gives the rounding direction unless it can itself
        // underflow. Round that tiny range conservatively on narrow platforms.
        const bool tiny = product < std::numeric_limits<long double>::min() /
                                        std::numeric_limits<long double>::epsilon() /
                                        std::numeric_limits<long double>::epsilon();
        return tiny || std::fma(left, right, -product) > 0.L
                   ? std::nextafter(product, std::numeric_limits<long double>::infinity())
                   : product;
    };
    const auto upper_root = [](long double value) {
        const long double root = std::sqrt(value);
        const bool tiny = value != 0.L && value < std::numeric_limits<long double>::min() /
                                                      std::numeric_limits<long double>::epsilon() /
                                                      std::numeric_limits<long double>::epsilon();
        return tiny || std::fma(-root, root, value) > 0.L
                   ? std::nextafter(root, std::numeric_limits<long double>::infinity())
                   : root;
    };
    const auto wide_amplitude = [&](double a, double b, bool precise) {
        const long double x = a, y = b;
        // Every finite double square, including denorm_min squared, is normal
        // in this wider exponent range. Direct squared sums are then safe and
        // avoid the scaling work needed by a general long-double hypot.
        if constexpr (std::numeric_limits<double>::radix == 2 && std::numeric_limits<long double>::radix == 2 &&
                      std::numeric_limits<long double>::max_exponent >
                          2 * std::numeric_limits<double>::max_exponent + 1 &&
                      std::numeric_limits<long double>::min_exponent - 1 <=
                          2 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits))
            return precise ? upper_root(upper_sum(upper_product(x, x), upper_product(y, y))) : std::sqrt(x * x + y * y);
        else {
            const long double large = std::max(std::abs(x), std::abs(y));
            const long double small = std::min(std::abs(x), std::abs(y));
            long double ratio = small / large;
            if (small < std::numeric_limits<long double>::min() / std::numeric_limits<long double>::epsilon() /
                            std::numeric_limits<long double>::epsilon() ||
                std::fma(-ratio, large, small) > 0.L)
                ratio = std::nextafter(ratio, std::numeric_limits<long double>::infinity());
            return upper_product(large, upper_root(upper_sum(1.L, upper_product(ratio, ratio))));
        }
    };
    const Eigen::Index joints = offset_.size();
    HarmonicBounds bounds{Eigen::VectorXd(joints), Eigen::VectorXd(joints), Eigen::VectorXd(joints),
                          Eigen::VectorXd(joints), Eigen::VectorXd(joints)};
    const auto upward = [](long double value) {
        const double maximum = std::numeric_limits<double>::max();
        if (value > maximum) return std::numeric_limits<double>::infinity();
        if (value < -maximum) return -maximum;
        const double rounded = static_cast<double>(value);
        return static_cast<long double>(rounded) < value
                   ? std::nextafter(rounded, std::numeric_limits<double>::infinity())
                   : rounded;
    };
    const auto significand_bits = [](double value) {
        if constexpr (sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
                      std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::max_exponent == 1024) {
            std::uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            std::uint64_t significand = bits & UINT64_C(0x000fffffffffffff);
            if ((bits & UINT64_C(0x7ff0000000000000)) != 0) significand |= UINT64_C(0x0010000000000000);
            if (significand == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
            return 64 - __builtin_clzll(significand) - __builtin_ctzll(significand);
#else
            while ((significand & 1) == 0) significand >>= 1;
            int width = 0;
            for (; significand != 0; significand >>= 1) ++width;
            return width;
#endif
        } else {
            return std::numeric_limits<double>::digits;
        }
    };
    struct EnvelopeSums {
        std::array<long double, 4> values;
        std::array<bool, 4> exact_terms;
    };
    const auto compute = [&](Eigen::Index joint, bool precise, bool prove_exact = false) {
        long double envelope = 0.L, velocity = 0.L, acceleration = 0.L, jerk = 0.L;
        long double envelope_error = 0.L, velocity_error = 0.L, acceleration_error = 0.L, jerk_error = 0.L;
        std::array<bool, 4> exact_terms{prove_exact, prove_exact, prove_exact, prove_exact};
        for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
            const double a = sin_coefficients_(joint, harmonic), b = cos_coefficients_(joint, harmonic);
            if (a == 0. && b == 0.) continue;
            // Keep amplitudes wide until the final envelope conversion: even
            // a normal double hypot can round inward before limits are tested.
            // A single active coefficient has an exact amplitude.
            const long double amplitude = a == 0.   ? std::abs(static_cast<long double>(b))
                                          : b == 0. ? std::abs(static_cast<long double>(a))
                                                    : wide_amplitude(a, b, precise);
            const double ordinary_frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
            const long double frequency =
                std::isfinite(ordinary_frequency)
                    ? static_cast<long double>(ordinary_frequency)
                    : upper_product(static_cast<long double>(harmonic + 1), angular_frequency_);
            if (prove_exact) {
                bool exact = a == 0. || b == 0.;
                int bits = exact ? significand_bits(a == 0. ? b : a) : 0;
                if (!exact && 2 * significand_bits(a) <= std::numeric_limits<long double>::digits &&
                    2 * significand_bits(b) <= std::numeric_limits<long double>::digits &&
                    amplitude <= std::numeric_limits<double>::max() &&
                    static_cast<long double>(static_cast<double>(amplitude)) == amplitude) {
                    bits = significand_bits(static_cast<double>(amplitude));
                    if (2 * bits <= std::numeric_limits<long double>::digits) {
                        long double left = static_cast<long double>(a) * a, right = static_cast<long double>(b) * b;
                        if (left < right) std::swap(left, right);
                        const long double sum = left + right;
                        exact = right - (sum - left) == 0.L && amplitude * amplitude == sum;
                    }
                }
                const int frequency_bits = significand_bits(ordinary_frequency);
                for (std::size_t derivative = 0; derivative < exact_terms.size(); ++derivative) {
                    if (derivative != 0) bits += frequency_bits;
                    exact &= bits <= std::numeric_limits<long double>::digits;
                    exact_terms[derivative] = exact_terms[derivative] && exact;
                }
            }
            const long double v = precise ? upper_product(amplitude, frequency) : amplitude * frequency;
            const long double acc = precise ? upper_product(v, frequency) : v * frequency;
            accumulate(envelope, envelope_error, amplitude);
            accumulate(velocity, velocity_error, v);
            accumulate(acceleration, acceleration_error, acc);
            accumulate(jerk, jerk_error, precise ? upper_product(acc, frequency) : acc * frequency);
        }
        return EnvelopeSums{{upper_sum(envelope, envelope_error), upper_sum(velocity, velocity_error),
                             upper_sum(acceleration, acceleration_error), upper_sum(jerk, jerk_error)},
                            exact_terms};
    };
    // Normal wide products and sqrt have a relative error bounded by fewer
    // than 8 unit roundoffs through the third derivative. Four epsilons give
    // that margin. Sum losses are already retained separately above.
    // The exponent checks also keep the margin itself normal, even for the
    // smallest double coefficient/frequency and the largest matrix index.
    constexpr bool wide_range =
        std::numeric_limits<double>::radix == 2 && std::numeric_limits<long double>::radix == 2 &&
        std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
        std::numeric_limits<long double>::max_exponent >
            4 * std::numeric_limits<double>::max_exponent + 4 * std::numeric_limits<Eigen::Index>::digits + 16 &&
        std::numeric_limits<long double>::min_exponent <
            4 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits) -
                std::numeric_limits<long double>::digits - 16;
    const bool screened = wide_range && static_cast<long double>(sin_coefficients_.cols()) * angular_frequency_ <=
                                            std::numeric_limits<double>::max();
    for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
        auto sums = compute(joint, !screened);
        const auto centers = [&](const std::array<long double, 4>& values) {
            return std::array<long double, 5>{upper_sum(-static_cast<long double>(offset_[joint]), values[0]),
                                              upper_sum(static_cast<long double>(offset_[joint]), values[0]), values[1],
                                              values[2], values[3]};
        };
        auto values = centers(sums.values);
        std::array<double, 5> rounded;
        const auto round_values = [&] {
            for (std::size_t index = 0; index < rounded.size(); ++index) rounded[index] = upward(values[index]);
        };
        round_values();
        const auto uncertain = [&] {
            for (std::size_t index = 0; index < rounded.size(); ++index) {
                const auto component = index < 2 ? 0 : index - 1;
                const long double margin =
                    sums.exact_terms[component]
                        ? 0.L
                        : sums.values[component] * (4.L * std::numeric_limits<long double>::epsilon());
                if (static_cast<long double>(rounded[index]) < upper_sum(values[index], margin)) return true;
            }
            return false;
        };
        if (screened && uncertain()) {
            // Small significands can prove every product exact without FMA.
            // This keeps dyadic coefficients and Pythagorean bounds cheap.
            sums = compute(joint, false, true);
            values = centers(sums.values);
            round_values();
            if (uncertain()) {
                sums = compute(joint, true);
                values = centers(sums.values);
                round_values();
            }
        }
        bounds.minimum_position[joint] = -rounded[0];
        bounds.maximum_position[joint] = rounded[1];
        bounds.maximum_absolute_velocity[joint] = rounded[2];
        bounds.maximum_absolute_acceleration[joint] = rounded[3];
        bounds.maximum_absolute_jerk[joint] = rounded[4];
    }
    return bounds;
}

ExcitationValidationResult ValidateExcitationTrajectory(const HarmonicTrajectory& trajectory,
                                                        const ExcitationLimits& limits,
                                                        double duration_seconds,
                                                        double sample_period_seconds) {
    const Eigen::Index dof = static_cast<Eigen::Index>(trajectory.dof());
    const bool dimensions_match = limits.minimum_position.size() == dof && limits.maximum_position.size() == dof &&
                                  limits.maximum_velocity.size() == dof && limits.maximum_acceleration.size() == dof &&
                                  limits.maximum_jerk.size() == dof;
    if (!dimensions_match || !limits.minimum_position.allFinite() || !limits.maximum_position.allFinite() ||
        !limits.maximum_velocity.allFinite() || !limits.maximum_acceleration.allFinite() ||
        !limits.maximum_jerk.allFinite() ||
        (limits.minimum_position.array() >= limits.maximum_position.array()).any() ||
        (limits.maximum_velocity.array() <= 0.0).any() || (limits.maximum_acceleration.array() <= 0.0).any() ||
        (limits.maximum_jerk.array() <= 0.0).any() || !std::isfinite(duration_seconds) || duration_seconds <= 0.0 ||
        !std::isfinite(sample_period_seconds) || sample_period_seconds <= 0.0) {
        throw std::invalid_argument("excitation validation configuration is invalid");
    }
    const double requested_steps = std::max(1., std::ceil(duration_seconds / sample_period_seconds));
    if (!std::isfinite(requested_steps) || requested_steps > 10000000.0) {
        throw std::invalid_argument("excitation validation requests too many samples");
    }
    const std::size_t steps = static_cast<std::size_t>(requested_steps);
    ExcitationValidationResult result;
    result.peak_absolute_velocity = Eigen::VectorXd::Zero(dof);
    result.peak_absolute_acceleration = Eigen::VectorXd::Zero(dof);
    result.peak_absolute_jerk = Eigen::VectorXd::Zero(dof);
    result.conservative_position_envelope.resize(dof);
    HarmonicBounds conservative = trajectory.ConservativeBounds();
    for (Eigen::Index joint = 0; joint < dof; ++joint) {
        const double width = conservative.maximum_position[joint] - conservative.minimum_position[joint];
        result.conservative_position_envelope[joint] =
            std::isfinite(width)
                ? width * 0.5
                : conservative.maximum_position[joint] * 0.5 - conservative.minimum_position[joint] * 0.5;
    }
    result.conservative_velocity_bound = std::move(conservative.maximum_absolute_velocity);
    result.conservative_acceleration_bound = std::move(conservative.maximum_absolute_acceleration);
    result.conservative_jerk_bound = std::move(conservative.maximum_absolute_jerk);
    if ((conservative.minimum_position.array() < limits.minimum_position.array()).any() ||
        (conservative.maximum_position.array() > limits.maximum_position.array()).any()) {
        result.reason = "conservative position bound exceeded";
        return result;
    }
    if ((result.conservative_velocity_bound.array() > limits.maximum_velocity.array()).any()) {
        result.reason = "conservative velocity bound exceeded";
        return result;
    }
    if ((result.conservative_acceleration_bound.array() > limits.maximum_acceleration.array()).any()) {
        result.reason = "conservative acceleration bound exceeded";
        return result;
    }
    if ((result.conservative_jerk_bound.array() > limits.maximum_jerk.array()).any()) {
        result.reason = "conservative jerk bound exceeded";
        return result;
    }
    ExcitationSample sample;
    for (std::size_t step = 0; step <= steps; ++step) {
        // Division can round an almost-integer quotient down to an integer.
        // Set the final time explicitly so multiplication cannot omit it.
        const double time = step == steps
                                ? duration_seconds
                                : std::min(duration_seconds, static_cast<double>(step) * sample_period_seconds);
        trajectory.EvaluateInto(time, sample);
        ++result.sample_count;
        result.peak_absolute_velocity = result.peak_absolute_velocity.cwiseMax(sample.velocity.cwiseAbs());
        result.peak_absolute_acceleration = result.peak_absolute_acceleration.cwiseMax(sample.acceleration.cwiseAbs());
        result.peak_absolute_jerk = result.peak_absolute_jerk.cwiseMax(sample.jerk.cwiseAbs());
        if ((sample.position.array() < limits.minimum_position.array()).any() ||
            (sample.position.array() > limits.maximum_position.array()).any()) {
            result.reason = "position limit exceeded";
            break;
        }
        if ((sample.velocity.cwiseAbs().array() > limits.maximum_velocity.array()).any()) {
            result.reason = "velocity limit exceeded";
            break;
        }
        if ((sample.acceleration.cwiseAbs().array() > limits.maximum_acceleration.array()).any()) {
            result.reason = "acceleration limit exceeded";
            break;
        }
        if ((sample.jerk.cwiseAbs().array() > limits.maximum_jerk.array()).any()) {
            result.reason = "jerk limit exceeded";
            break;
        }
    }
    result.passed = result.reason.empty();
    return result;
}

void WriteRawDatasetCsv(const RawDataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid raw dataset: " + reason);
    auto output = OpenCsvOutput(path, dataset.joint_names, {"q/", "tau/"}, "raw dataset");
    for (Eigen::Index row = 0; row < dataset.time_seconds.size(); ++row) {
        output << dataset.time_seconds[row];
        for (const auto* values : {&dataset.position, &dataset.torque})
            for (Eigen::Index joint = 0; joint < values->cols(); ++joint) output << ',' << (*values)(row, joint);
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing raw dataset: " + path);
}

RawDataset ReadRawDatasetCsv(const std::string& path) {
    CsvInput input(path, {"tau/"}, "raw dataset");
    RawDataset dataset;
    dataset.joint_names = std::move(input.joint_names);
    const std::size_t dof = dataset.joint_names.size();
    const auto maximum_values = std::min(static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                                         std::numeric_limits<std::size_t>::max() / sizeof(double));
    struct Chunk {
        Chunk(Eigen::Index capacity, Eigen::Index columns)
            : time(capacity), position(capacity, columns), torque(capacity, columns) {}
        Eigen::VectorXd time;
        Eigen::MatrixXd position, torque;
        Eigen::Index used{0};
    };
    std::vector<Chunk> chunks;
    // Small files start with eight rows. Larger chunks are capped by both row
    // count and coefficient count, bounding excess capacity without a file pass.
    const Eigen::Index maximum_chunk_rows =
        std::max<Eigen::Index>(1, std::min<Eigen::Index>(4096, 65536 / static_cast<Eigen::Index>(input.values.size())));
    const auto columns = static_cast<Eigen::Index>(dof);
    std::size_t row_count = 0;
    const auto& values = input.values;
    while (input.ReadRow()) {
        if (row_count >= maximum_values / values.size()) {
            throw std::overflow_error("raw dataset CSV contains too many values");
        }
        if (chunks.empty() || chunks.back().used == chunks.back().time.size()) {
            const Eigen::Index capacity =
                std::min<Eigen::Index>(maximum_chunk_rows, chunks.empty() ? 8 : 2 * chunks.back().time.size());
            chunks.emplace_back(capacity, columns);
        }
        auto& chunk = chunks.back();
        chunk.time[chunk.used] = values[0];
        chunk.position.row(chunk.used) = Eigen::Map<const Eigen::RowVectorXd>(values.data() + 1, columns);
        chunk.torque.row(chunk.used) = Eigen::Map<const Eigen::RowVectorXd>(values.data() + 1 + dof, columns);
        ++chunk.used;
        ++row_count;
    }
    // Header validation and ParseCsvRow already enforce names, dimensions,
    // finite fields and strict timestamps. Only the minimum row count remains.
    if (row_count < 3)
        throw std::runtime_error(
            "raw dataset CSV is invalid: raw dataset dimensions are inconsistent or values are non-finite");
    // Materialize one field at a time and release consumed chunks immediately.
    // This avoids retaining a complete flat parse buffer alongside all outputs.
    const auto samples = static_cast<Eigen::Index>(row_count);
    const auto materialize = [&](auto& output, auto field) {
        output.resize(samples, (chunks.front().*field).cols());
        Eigen::Index offset = 0;
        for (auto& chunk : chunks) {
            auto& values = chunk.*field;
            output.middleRows(offset, chunk.used) = values.topRows(chunk.used);
            offset += chunk.used;
            values.resize(0, values.cols());
        }
    };
    materialize(dataset.time_seconds, &Chunk::time);
    materialize(dataset.position, &Chunk::position);
    materialize(dataset.torque, &Chunk::torque);
    return dataset;
}

void WriteDatasetCsv(const Dataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid dataset: " + reason);
    auto output = OpenCsvOutput(path, dataset.joint_names, {"q/", "dq/", "ddq/", "tau/"}, "dataset");
    for (const auto& sample : dataset.samples) {
        output << sample.time_seconds;
        for (const auto* values : {&sample.position, &sample.velocity, &sample.acceleration, &sample.torque}) {
            for (Eigen::Index index = 0; index < values->size(); ++index) output << ',' << (*values)[index];
        }
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing dataset: " + path);
}

Dataset ReadDatasetCsv(const std::string& path) {
    CsvInput input(path, {"dq/", "ddq/", "tau/"}, "dataset");
    Dataset dataset;
    dataset.joint_names = std::move(input.joint_names);
    const std::size_t dof = dataset.dof();
    const auto& values = input.values;
    while (input.ReadRow()) {
        Sample sample;
        sample.time_seconds = values[0];
        std::array<Eigen::VectorXd*, 4> vectors{&sample.position, &sample.velocity, &sample.acceleration,
                                                &sample.torque};
        for (std::size_t group = 0; group < vectors.size(); ++group) {
            *vectors[group] =
                Eigen::Map<const Eigen::VectorXd>(values.data() + 1 + group * dof, static_cast<Eigen::Index>(dof));
        }
        dataset.samples.push_back(std::move(sample));
    }
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::runtime_error("dataset CSV is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                    const Eigen::VectorXd& time_seconds,
                                    const Eigen::MatrixXd& position,
                                    const Eigen::MatrixXd& torque,
                                    const PreprocessOptions& options) {
    std::string reason;
    if (!ValidateRawInputs(joint_names, time_seconds, position, torque, &reason)) {
        throw std::invalid_argument("invalid raw dataset: " + reason);
    }
    if (options.smoothing_window == 0 || options.smoothing_window % 2 == 0 || options.torque_smoothing_window == 0 ||
        options.torque_smoothing_window % 2 == 0) {
        throw std::invalid_argument("position and torque smoothing windows must be positive odd numbers");
    }
    const std::size_t sample_count = static_cast<std::size_t>(time_seconds.size());
    if (options.trim_front >= sample_count || options.trim_back >= sample_count - options.trim_front) {
        throw std::invalid_argument("trimming removes every sample");
    }
    // Keep caller-owned position/torque as views when smoothing is disabled.
    // Derivative matrices already own their storage and can be reused directly.
    const Eigen::MatrixXd position_storage =
        options.smoothing_window == 1 ? Eigen::MatrixXd{} : SmoothSymmetric(position, options.smoothing_window);
    const auto& filtered_position = options.smoothing_window == 1 ? position : position_storage;
    Eigen::MatrixXd filtered_velocity = Differentiate(time_seconds, filtered_position);
    if (options.smoothing_window != 1) {
        filtered_velocity = SmoothSymmetric(filtered_velocity, options.smoothing_window);
    }
    Eigen::MatrixXd filtered_acceleration = Differentiate(time_seconds, filtered_velocity);
    if (options.smoothing_window != 1) {
        filtered_acceleration = SmoothSymmetric(filtered_acceleration, options.smoothing_window);
    }
    const Eigen::MatrixXd torque_storage = options.torque_smoothing_window == 1
                                               ? Eigen::MatrixXd{}
                                               : SmoothSymmetric(torque, options.torque_smoothing_window);
    const auto& filtered_torque = options.torque_smoothing_window == 1 ? torque : torque_storage;

    Dataset dataset;
    dataset.joint_names = joint_names;
    const Eigen::Index begin = static_cast<Eigen::Index>(options.trim_front);
    const Eigen::Index end = time_seconds.size() - static_cast<Eigen::Index>(options.trim_back);
    dataset.samples.reserve(static_cast<std::size_t>(end - begin));
    for (Eigen::Index row = begin; row < end; ++row) {
        dataset.samples.push_back(
            Sample{time_seconds[row], filtered_position.row(row).transpose(), filtered_velocity.row(row).transpose(),
                   filtered_acceleration.row(row).transpose(), filtered_torque.row(row).transpose()});
    }
    if (!dataset.Validate(&reason)) throw std::invalid_argument("preprocessed dataset is invalid: " + reason);
    return dataset;
}

Dataset PreprocessPositionAndTorque(const RawDataset& dataset, const PreprocessOptions& options) {
    return PreprocessPositionAndTorque(dataset.joint_names, dataset.time_seconds, dataset.position, dataset.torque,
                                       options);
}

bool RegressionProblem::Validate(std::string* reason) const {
    if (sample_count == 0 || dof == 0 || regressor.rows() == 0 || regressor.cols() == 0) {
        return Fail(reason, "regression problem must not be empty");
    }
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) / dof ||
        regressor.rows() != observation.size() || row_weights.size() != observation.size() ||
        regressor.rows() != static_cast<Eigen::Index>(sample_count * dof)) {
        return Fail(reason, "regression problem dimensions are inconsistent");
    }
    if (!regressor.allFinite() || !observation.allFinite() || !row_weights.allFinite() ||
        (row_weights.array() <= 0.0).any()) {
        return Fail(reason, "regression values must be finite and row weights positive");
    }
    return true;
}

RegressionProblem BuildRegressionProblem(const Dataset& dataset, const RegressorBackend& backend) {
    return detail::AssembleRegressionProblem(dataset, backend,
                                             [&](const Sample& sample) { return backend.Compute(sample); });
}

JointActuatorRegressor::JointActuatorRegressor(std::shared_ptr<const RegressorBackend> base,
                                               std::vector<std::string> joint_names,
                                               JointActuatorRegressorOptions options)
    : base_(std::move(base)), joint_names_(std::move(joint_names)), options_(options) {
    if (!base_ || joint_names_.empty() ||
        std::any_of(joint_names_.begin(), joint_names_.end(), [](const std::string& name) { return name.empty(); }) ||
        !std::isfinite(options_.coulomb_velocity_deadband) || options_.coulomb_velocity_deadband < 0.0) {
        throw std::invalid_argument("joint actuator regressor configuration is invalid");
    }
    if (!HasUniqueNonemptyNames(joint_names_)) {
        throw std::invalid_argument("joint actuator regressor names must be unique");
    }
    terms_per_joint_ =
        static_cast<std::size_t>(options_.drive_inertia) + static_cast<std::size_t>(options_.viscous_friction) +
        static_cast<std::size_t>(options_.coulomb_friction) + static_cast<std::size_t>(options_.torque_bias);
    if (terms_per_joint_ == 0) throw std::invalid_argument("joint actuator regressor enables no terms");
}

std::size_t JointActuatorRegressor::parameter_count() const noexcept {
    return base_->parameter_count() + terms_per_joint_ * joint_names_.size();
}

std::vector<std::string> JointActuatorRegressor::parameter_names() const {
    auto names = base_->parameter_names();
    names.reserve(parameter_count());
    for (const auto& joint : joint_names_) {
        if (options_.drive_inertia) names.push_back("actuator/" + joint + "/drive_inertia");
        if (options_.viscous_friction) names.push_back("actuator/" + joint + "/viscous_friction");
        if (options_.coulomb_friction) names.push_back("actuator/" + joint + "/coulomb_friction");
        if (options_.torque_bias) names.push_back("actuator/" + joint + "/torque_bias");
    }
    return names;
}

Eigen::MatrixXd JointActuatorRegressor::Compute(const Sample& sample) const {
    const Eigen::Index dof = static_cast<Eigen::Index>(joint_names_.size());
    if (sample.position.size() != dof || sample.velocity.size() != dof || sample.acceleration.size() != dof ||
        !sample.position.allFinite() || !sample.velocity.allFinite() || !sample.acceleration.allFinite()) {
        throw std::invalid_argument("joint actuator regressor sample dimension is invalid");
    }
    const Eigen::MatrixXd base = base_->Compute(sample);
    if (base.rows() != dof || base.cols() != static_cast<Eigen::Index>(base_->parameter_count()) || !base.allFinite()) {
        throw std::runtime_error("base regressor returned an invalid block");
    }
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(dof, static_cast<Eigen::Index>(parameter_count()));
    output.leftCols(base.cols()) = base;
    Eigen::Index column = base.cols();
    for (Eigen::Index joint = 0; joint < dof; ++joint) {
        if (options_.drive_inertia) output(joint, column++) = sample.acceleration[joint];
        if (options_.viscous_friction) output(joint, column++) = sample.velocity[joint];
        if (options_.coulomb_friction) {
            output(joint, column++) = std::abs(sample.velocity[joint]) <= options_.coulomb_velocity_deadband
                                          ? 0.0
                                          : std::copysign(1.0, sample.velocity[joint]);
        }
        if (options_.torque_bias) output(joint, column++) = 1.0;
    }
    return output;
}

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
    Eigen::VectorXd previous;
    IdentificationResult result;
    WeightedSolveWorkspace workspace(problem, options);
    Eigen::VectorXd robust_residual;
    if (options.huber_delta > 0.0 && options.max_iterations > 1) robust_residual.resize(problem.regressor.rows());
    std::size_t iterations = 0;
    bool converged = options.huber_delta == 0.0;
    const std::size_t limit = options.huber_delta > 0.0 ? options.max_iterations : 1;
    for (; iterations < limit; ++iterations) {
        workspace.Compute(problem, weights, result);
        if (!result.parameters.allFinite()) throw std::runtime_error("parameter solution is non-finite");
        if (previous.size() == result.parameters.size()) {
            if (ParametersConverged(result.parameters, previous, options.convergence_tolerance)) {
                converged = true;
                ++iterations;
                break;
            }
        }
        // Only prepare another weighted solve when an iteration remains.
        if (options.huber_delta > 0.0 && iterations + 1 < limit) {
            previous = result.parameters;
            robust_residual.noalias() = problem.regressor * result.parameters;
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
    const auto errors = MeasureErrors(result.residual);
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
    const auto errors = MeasureErrors(residual);
    metrics.rmse = errors.rms;
    metrics.mean_absolute_error = errors.mean_absolute;
    metrics.max_absolute_error = errors.maximum;
    metrics.normalized_rmse = NormalizeRms(errors, MeasureErrors(problem.observation));
    return metrics;
}

PhysicalValidationResult ValidateMassMatrices(const Eigen::MatrixXd& joint_positions,
                                              const Eigen::VectorXd& parameters,
                                              const MassMatrixFunction& mass_matrix,
                                              double minimum_eigenvalue) {
    if (joint_positions.rows() == 0 || joint_positions.cols() == 0 || !joint_positions.allFinite() ||
        parameters.size() == 0 || !parameters.allFinite() || !mass_matrix || !std::isfinite(minimum_eigenvalue) ||
        minimum_eigenvalue <= 0.0) {
        throw std::invalid_argument("mass-matrix validation inputs are invalid");
    }
    PhysicalValidationResult result;
    result.minimum_eigenvalue = std::numeric_limits<double>::infinity();
    Eigen::VectorXd position(joint_positions.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(joint_positions.cols());
    for (Eigen::Index row = 0; row < joint_positions.rows(); ++row) {
        position = joint_positions.row(row).transpose();
        Eigen::MatrixXd mass = mass_matrix(position, parameters);
        ++result.tested;
        if (mass.rows() != joint_positions.cols() || mass.cols() != joint_positions.cols() || !mass.allFinite()) {
            ++result.failed;
            continue;
        }
        // Normalize the symmetry comparison so squared Frobenius norms cannot
        // hide asymmetry through overflow or underflow. The zero matrix keeps
        // its zero eigenvalue and fails the positive threshold below.
        const double scale = mass.cwiseAbs().maxCoeff();
        if (scale != 0.0) mass /= scale;
        if (!mass.isApprox(mass.transpose(), 1e-9)) {
            ++result.failed;
            continue;
        }
        solver.compute(mass, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success || !(solver.eigenvalues() * scale).allFinite()) {
            ++result.failed;
            continue;
        }
        const double value = solver.eigenvalues().minCoeff() * scale;
        result.minimum_eigenvalue = std::min(result.minimum_eigenvalue, value);
        if (value < minimum_eigenvalue) {
            ++result.failed;
        }
    }
    result.passed = result.failed == 0;
    return result;
}

}  // namespace robot_identification
