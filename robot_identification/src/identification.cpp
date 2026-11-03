#include "robot_identification/identification.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "data_validation.h"
#include "name_validation.h"
#include "robot_identification/detail/regression_assembly.h"

namespace robot_identification {
namespace {

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

}  // namespace

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

Dataset PreprocessPositionAndTorque(const std::vector<std::string>& joint_names,
                                    const Eigen::VectorXd& time_seconds,
                                    const Eigen::MatrixXd& position,
                                    const Eigen::MatrixXd& torque,
                                    const PreprocessOptions& options) {
    std::string reason;
    if (!detail::ValidateRawInputs(joint_names, time_seconds, position, torque, &reason)) {
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
    if (!detail::HasUniqueNonemptyNames(joint_names_)) {
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
    constexpr std::string_view kParameterPrefix = "actuator/";
    for (const auto& joint : joint_names_) {
        const auto append_name = [&](std::string_view suffix) {
            auto& name = names.emplace_back();
            name.reserve(kParameterPrefix.size() + joint.size() + suffix.size());
            name.append(kParameterPrefix).append(joint).append(suffix);
        };
        if (options_.drive_inertia) append_name("/drive_inertia");
        if (options_.viscous_friction) append_name("/viscous_friction");
        if (options_.coulomb_friction) append_name("/coulomb_friction");
        if (options_.torque_bias) append_name("/torque_bias");
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
