#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "robot_identification/identification.h"

namespace robot_identification {

namespace {

long double UpperSum(long double left, long double right) {
    // FastTwoSum recovers the addition's rounding residual. Advance only
    // when the rounded sum is below the exact sum; exact sums stay exact.
    if (std::abs(left) < std::abs(right)) std::swap(left, right);
    const long double sum = left + right;
    const long double residual = right - (sum - left);
    return residual > 0.L ? std::nextafter(sum, std::numeric_limits<long double>::infinity()) : sum;
}

void AccumulateEnvelopeTerm(long double& sum, long double& downward_rounding_loss, long double term) {
    long double left = sum;
    if (std::abs(left) < std::abs(term)) std::swap(left, term);
    sum = left + term;
    const long double residual = term - (sum - left);
    // Track only downward rounding losses. Their small positive sum needs
    // far fewer nextafter calls than rounding every main addition upward.
    if (residual > 0.L) downward_rounding_loss = UpperSum(downward_rounding_loss, residual);
}

long double UpperProduct(long double left, long double right) {
    const long double product = left * right;
    if (left == 0.L || right == 0.L || left == 1.L || right == 1.L) return product;
    // A fused residual gives the rounding direction unless it can itself
    // underflow. Round that tiny range conservatively on narrow platforms.
    const bool tiny = product < std::numeric_limits<long double>::min() / std::numeric_limits<long double>::epsilon() /
                                    std::numeric_limits<long double>::epsilon();
    return tiny || std::fma(left, right, -product) > 0.L
               ? std::nextafter(product, std::numeric_limits<long double>::infinity())
               : product;
}

long double UpperSquareRoot(long double value) {
    const long double root = std::sqrt(value);
    const bool tiny = value != 0.L && value < std::numeric_limits<long double>::min() /
                                                  std::numeric_limits<long double>::epsilon() /
                                                  std::numeric_limits<long double>::epsilon();
    return tiny || std::fma(-root, root, value) > 0.L
               ? std::nextafter(root, std::numeric_limits<long double>::infinity())
               : root;
}

inline long double WideAmplitude(double sine_coefficient, double cosine_coefficient, bool round_upward) {
    const long double wide_sine = sine_coefficient;
    const long double wide_cosine = cosine_coefficient;
    // Every finite double square, including denorm_min squared, is normal
    // in this wider exponent range. Direct squared sums are then safe and
    // avoid the scaling work needed by a general long-double hypot.
    if constexpr (std::numeric_limits<double>::radix == 2 && std::numeric_limits<long double>::radix == 2 &&
                  std::numeric_limits<long double>::max_exponent > 2 * std::numeric_limits<double>::max_exponent + 1 &&
                  std::numeric_limits<long double>::min_exponent - 1 <=
                      2 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits))
        return round_upward ? UpperSquareRoot(
                                  UpperSum(UpperProduct(wide_sine, wide_sine), UpperProduct(wide_cosine, wide_cosine)))
                            : std::sqrt(wide_sine * wide_sine + wide_cosine * wide_cosine);
    else {
        const long double large = std::max(std::abs(wide_sine), std::abs(wide_cosine));
        const long double small = std::min(std::abs(wide_sine), std::abs(wide_cosine));
        long double ratio = small / large;
        if (small < std::numeric_limits<long double>::min() / std::numeric_limits<long double>::epsilon() /
                        std::numeric_limits<long double>::epsilon() ||
            std::fma(-ratio, large, small) > 0.L)
            ratio = std::nextafter(ratio, std::numeric_limits<long double>::infinity());
        return UpperProduct(large, UpperSquareRoot(UpperSum(1.L, UpperProduct(ratio, ratio))));
    }
}

double RoundUpToDouble(long double value) {
    const double maximum = std::numeric_limits<double>::max();
    if (value > maximum) return std::numeric_limits<double>::infinity();
    if (value < -maximum) return -maximum;
    const double rounded = static_cast<double>(value);
    return static_cast<long double>(rounded) < value ? std::nextafter(rounded, std::numeric_limits<double>::infinity())
                                                     : rounded;
}

// Trailing zeros do not increase the width needed for an exact binary64 product.
int SignificandWidth(double value) {
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
}

inline void UpdateProductExactness(double sine_coefficient,
                                   double cosine_coefficient,
                                   long double amplitude,
                                   double evaluation_angular_frequency,
                                   std::array<bool, 4>& exact_products) {
    bool exact = sine_coefficient == 0. || cosine_coefficient == 0.;
    int bits = exact ? SignificandWidth(sine_coefficient == 0. ? cosine_coefficient : sine_coefficient) : 0;
    if (!exact && 2 * SignificandWidth(sine_coefficient) <= std::numeric_limits<long double>::digits &&
        2 * SignificandWidth(cosine_coefficient) <= std::numeric_limits<long double>::digits &&
        amplitude <= std::numeric_limits<double>::max() &&
        static_cast<long double>(static_cast<double>(amplitude)) == amplitude) {
        bits = SignificandWidth(static_cast<double>(amplitude));
        if (2 * bits <= std::numeric_limits<long double>::digits) {
            long double left = static_cast<long double>(sine_coefficient) * sine_coefficient;
            long double right = static_cast<long double>(cosine_coefficient) * cosine_coefficient;
            if (left < right) std::swap(left, right);
            const long double sum = left + right;
            exact = right - (sum - left) == 0.L && amplitude * amplitude == sum;
        }
    }
    const int frequency_bits = SignificandWidth(evaluation_angular_frequency);
    for (std::size_t derivative = 0; derivative < exact_products.size(); ++derivative) {
        if (derivative != 0) bits += frequency_bits;
        exact &= bits <= std::numeric_limits<long double>::digits;
        exact_products[derivative] = exact_products[derivative] && exact;
    }
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
    bool needs_wide_retry = false;
    for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
        if (sin_coefficients_.col(harmonic).isZero(0.) && cos_coefficients_.col(harmonic).isZero(0.)) continue;
        const double angular_frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
        const double phase_radians = angular_frequency * time_seconds;
        const double sine = std::sin(phase_radians);
        const double cosine = std::cos(phase_radians);
        needs_wide_retry |= time_seconds != 0. && std::abs(phase_radians) < std::numeric_limits<double>::min();
        for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
            const double sine_coefficient = sin_coefficients_(joint, harmonic);
            const double cosine_coefficient = cos_coefficients_(joint, harmonic);
            const double position_term = sine_coefficient * sine + cosine_coefficient * cosine;
            const double phase_derivative_term = sine_coefficient * cosine - cosine_coefficient * sine;
            // Retry nonzero products that enter the subnormal range before
            // frequency factors or other harmonics can restore their scale.
            needs_wide_retry |=
                (std::abs(position_term) < std::numeric_limits<double>::min() &&
                 ((sine_coefficient != 0. && sine != 0.) || (cosine_coefficient != 0. && cosine != 0.))) ||
                (std::abs(phase_derivative_term) < std::numeric_limits<double>::min() &&
                 ((sine_coefficient != 0. && cosine != 0.) || (cosine_coefficient != 0. && sine != 0.)));
            sample.position[joint] += position_term;
            // Multiply the finite coefficient before successive frequency
            // factors: powers of frequency may overflow or underflow even
            // when every requested derivative remains representable.
            const double velocity_term = angular_frequency * phase_derivative_term;
            const double acceleration_term = angular_frequency * (angular_frequency * position_term);
            const double jerk_term = angular_frequency * (angular_frequency * velocity_term);
            // With a frequency below one, a later derivative can underflow
            // even when its coefficient combination is normal. Multiple such
            // harmonics can still have a representable combined derivative.
            needs_wide_retry |=
                angular_frequency < 1. &&
                ((std::abs(velocity_term) < std::numeric_limits<double>::min() && phase_derivative_term != 0.) ||
                 (std::abs(acceleration_term) < std::numeric_limits<double>::min() && position_term != 0.) ||
                 (std::abs(jerk_term) < std::numeric_limits<double>::min() && phase_derivative_term != 0.));
            sample.velocity[joint] += velocity_term;
            sample.acceleration[joint] -= acceleration_term;
            sample.jerk[joint] -= jerk_term;
        }
    }
    if (!needs_wide_retry && sample.position.allFinite() && sample.velocity.allFinite() &&
        sample.acceleration.allFinite() && sample.jerk.allFinite())
        return;
    // Keep exceptional products and sums in extended range until all harmonic
    // contributions have been combined. Reuse the existing owning outputs.
    for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
        std::array<long double, 4> values{offset_[joint], 0.L, 0.L, 0.L};
        for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
            const long double sine_coefficient = sin_coefficients_(joint, harmonic);
            const long double cosine_coefficient = cos_coefficients_(joint, harmonic);
            if (sine_coefficient == 0.L && cosine_coefficient == 0.L) continue;
            const double angular_frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
            const double phase_radians = angular_frequency * time_seconds;
            if (!std::isfinite(phase_radians)) throw std::overflow_error("harmonic trajectory evaluation overflowed");
            const bool subnormal_phase =
                time_seconds != 0. && std::abs(phase_radians) < std::numeric_limits<double>::min();
            const long double wide_phase_radians = static_cast<long double>(angular_frequency) * time_seconds;
            const long double sine = subnormal_phase ? std::sin(wide_phase_radians) : std::sin(phase_radians);
            const long double cosine = subnormal_phase ? std::cos(wide_phase_radians) : std::cos(phase_radians);
            const long double position = sine_coefficient * sine + cosine_coefficient * cosine;
            const long double velocity =
                static_cast<long double>(angular_frequency) * (sine_coefficient * cosine - cosine_coefficient * sine);
            values[0] += position;
            values[1] += velocity;
            values[2] -= static_cast<long double>(angular_frequency) * angular_frequency * position;
            values[3] -= static_cast<long double>(angular_frequency) * angular_frequency * velocity;
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
    const Eigen::Index joints = offset_.size();
    HarmonicBounds bounds{Eigen::VectorXd(joints), Eigen::VectorXd(joints), Eigen::VectorXd(joints),
                          Eigen::VectorXd(joints), Eigen::VectorXd(joints)};
    struct EnvelopeSums {
        std::array<long double, 4> values;
        std::array<bool, 4> exact_products;
    };
    const auto compute_envelopes = [&](Eigen::Index joint, bool round_upward, bool prove_exact = false) {
        long double envelope = 0.L, velocity = 0.L, acceleration = 0.L, jerk = 0.L;
        long double envelope_error = 0.L, velocity_error = 0.L, acceleration_error = 0.L, jerk_error = 0.L;
        std::array<bool, 4> exact_products{prove_exact, prove_exact, prove_exact, prove_exact};
        for (Eigen::Index harmonic = 0; harmonic < sin_coefficients_.cols(); ++harmonic) {
            const double sine_coefficient = sin_coefficients_(joint, harmonic);
            const double cosine_coefficient = cos_coefficients_(joint, harmonic);
            if (sine_coefficient == 0. && cosine_coefficient == 0.) continue;
            // Keep amplitudes wide until the final envelope conversion: even
            // a normal double hypot can round inward before limits are tested.
            // A single active coefficient has an exact amplitude.
            const long double amplitude =
                sine_coefficient == 0.     ? std::abs(static_cast<long double>(cosine_coefficient))
                : cosine_coefficient == 0. ? std::abs(static_cast<long double>(sine_coefficient))
                                           : WideAmplitude(sine_coefficient, cosine_coefficient, round_upward);
            const double evaluation_angular_frequency = static_cast<double>(harmonic + 1) * angular_frequency_;
            const long double angular_frequency =
                std::isfinite(evaluation_angular_frequency)
                    ? static_cast<long double>(evaluation_angular_frequency)
                    : UpperProduct(static_cast<long double>(harmonic + 1), angular_frequency_);
            if (prove_exact)
                UpdateProductExactness(sine_coefficient, cosine_coefficient, amplitude, evaluation_angular_frequency,
                                       exact_products);
            const long double velocity_term =
                round_upward ? UpperProduct(amplitude, angular_frequency) : amplitude * angular_frequency;
            const long double acceleration_term =
                round_upward ? UpperProduct(velocity_term, angular_frequency) : velocity_term * angular_frequency;
            AccumulateEnvelopeTerm(envelope, envelope_error, amplitude);
            AccumulateEnvelopeTerm(velocity, velocity_error, velocity_term);
            AccumulateEnvelopeTerm(acceleration, acceleration_error, acceleration_term);
            AccumulateEnvelopeTerm(jerk, jerk_error,
                                   round_upward ? UpperProduct(acceleration_term, angular_frequency)
                                                : acceleration_term * angular_frequency);
        }
        return EnvelopeSums{{UpperSum(envelope, envelope_error), UpperSum(velocity, velocity_error),
                             UpperSum(acceleration, acceleration_error), UpperSum(jerk, jerk_error)},
                            exact_products};
    };
    // Normal wide products and sqrt have a relative error bounded by fewer
    // than 8 unit roundoffs through the third derivative. Four epsilons give
    // that margin. Sum losses are already retained separately above.
    // The exponent checks also keep the margin itself normal, even for the
    // smallest double coefficient/frequency and the largest matrix index.
    constexpr bool kWideRange =
        std::numeric_limits<double>::radix == 2 && std::numeric_limits<long double>::radix == 2 &&
        std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
        std::numeric_limits<long double>::max_exponent >
            4 * std::numeric_limits<double>::max_exponent + 4 * std::numeric_limits<Eigen::Index>::digits + 16 &&
        std::numeric_limits<long double>::min_exponent <
            4 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits) -
                std::numeric_limits<long double>::digits - 16;
    const bool use_wide_screen =
        kWideRange &&
        static_cast<long double>(sin_coefficients_.cols()) * angular_frequency_ <= std::numeric_limits<double>::max();
    for (Eigen::Index joint = 0; joint < offset_.size(); ++joint) {
        auto sums = compute_envelopes(joint, !use_wide_screen);
        const auto shifted_bounds = [&](const std::array<long double, 4>& values) {
            return std::array<long double, 5>{UpperSum(-static_cast<long double>(offset_[joint]), values[0]),
                                              UpperSum(static_cast<long double>(offset_[joint]), values[0]), values[1],
                                              values[2], values[3]};
        };
        // Round -minimum_position upward, then negate it to round the lower
        // position bound outward with the same conversion as the upper bounds.
        auto values = shifted_bounds(sums.values);
        std::array<double, 5> rounded;
        const auto round_values = [&] {
            for (std::size_t index = 0; index < rounded.size(); ++index)
                rounded[index] = RoundUpToDouble(values[index]);
        };
        round_values();
        const auto needs_outward_retry = [&] {
            for (std::size_t index = 0; index < rounded.size(); ++index) {
                const auto component = index < 2 ? 0 : index - 1;
                const long double margin =
                    sums.exact_products[component]
                        ? 0.L
                        : sums.values[component] * (4.L * std::numeric_limits<long double>::epsilon());
                if (static_cast<long double>(rounded[index]) < UpperSum(values[index], margin)) return true;
            }
            return false;
        };
        if (use_wide_screen && needs_outward_retry()) {
            // Small significands can prove every product exact without FMA.
            // This keeps dyadic coefficients and Pythagorean bounds cheap.
            sums = compute_envelopes(joint, false, true);
            values = shifted_bounds(sums.values);
            round_values();
            if (needs_outward_retry()) {
                sums = compute_envelopes(joint, true);
                values = shifted_bounds(sums.values);
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
        const double time_seconds = step == steps
                                        ? duration_seconds
                                        : std::min(duration_seconds, static_cast<double>(step) * sample_period_seconds);
        trajectory.EvaluateInto(time_seconds, sample);
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

}  // namespace robot_identification
