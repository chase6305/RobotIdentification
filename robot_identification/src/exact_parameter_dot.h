#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace robot_identification {
namespace detail {

constexpr bool kBinary64ParameterDot =
    sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53 &&
    std::numeric_limits<double>::min_exponent == -1021 && std::numeric_limits<double>::max_exponent == 1024;

// Binary64 products are integer multiples of 2^-2148 and have magnitude below
// 2^2048. The extra Index bits cover the sum of every representable column count.
// Separate positive/negative words avoid both overflow before cancellation and
// loss of a small residual. No allocation or nonstandard integer type is needed.
class ParameterDotAccumulator {
    static constexpr int kUnitExponent = -2148;
    static constexpr int kProductExponentLimit = 2048;
    // Cover the product exponent span, then allow carry bits for the column count.
    static constexpr std::size_t kWordCount =
        (kProductExponentLimit - kUnitExponent + std::numeric_limits<Eigen::Index>::digits + 63) / 64;
    using Words = std::array<std::uint64_t, kWordCount>;
    Words positive_{};
    Words negative_{};

    static void AddWord(Words& words, std::size_t index, std::uint64_t value) {
        while (value != 0) {
            const auto previous = words[index];
            words[index] += value;
            value = words[index] < previous;
            ++index;
        }
    }

    static std::uint64_t Bits(double value) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    static int TopBit(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
        if constexpr (std::numeric_limits<unsigned long long>::digits == 64) return 63 - __builtin_clzll(value);
#endif
        int result = 0;
        while (value >>= 1) ++result;
        return result;
    }

    static std::uint64_t Extract(const Words& words, int begin) {
        const auto word = static_cast<std::size_t>(begin / 64);
        const int offset = begin % 64;
        std::uint64_t value = words[word] >> offset;
        if (offset != 0 && word + 1 < words.size()) value |= words[word + 1] << (64 - offset);
        return value;
    }

    static bool AnyBelow(const Words& words, int end) {
        const auto whole = static_cast<std::size_t>(end / 64);
        for (std::size_t index = 0; index < whole; ++index)
            if (words[index] != 0) return true;
        const int remaining = end % 64;
        return remaining != 0 && (words[whole] & ((UINT64_C(1) << remaining) - 1)) != 0;
    }

public:
    void AddProduct(double left, double right) {
        const auto left_bits = Bits(left), right_bits = Bits(right);
        auto left_significand = left_bits & UINT64_C(0x000fffffffffffff);
        auto right_significand = right_bits & UINT64_C(0x000fffffffffffff);
        const int left_exponent = static_cast<int>((left_bits >> 52) & 0x7ff);
        const int right_exponent = static_cast<int>((right_bits >> 52) & 0x7ff);
        if (left_exponent != 0) left_significand |= UINT64_C(0x0010000000000000);
        if (right_exponent != 0) right_significand |= UINT64_C(0x0010000000000000);
        if (left_significand == 0 || right_significand == 0) return;
        const auto left_low = left_significand & UINT64_C(0xffffffff);
        const auto right_low = right_significand & UINT64_C(0xffffffff);
        const auto left_high = left_significand >> 32, right_high = right_significand >> 32;
        std::uint64_t low = left_low * right_low;
        const std::uint64_t middle = left_high * right_low + left_low * right_high;
        std::uint64_t high = left_high * right_high + (middle >> 32);
        const auto previous = low;
        low += middle << 32;
        high += low < previous;
        const int shift = (left_exponent == 0 ? -1074 : left_exponent - 1075) +
                          (right_exponent == 0 ? -1074 : right_exponent - 1075) - kUnitExponent;
        const auto word = static_cast<std::size_t>(shift / 64);
        const int offset = shift % 64;
        auto& words = ((left_bits ^ right_bits) >> 63) != 0 ? negative_ : positive_;
        AddWord(words, word, low << offset);
        if (offset == 0) {
            AddWord(words, word + 1, high);
        } else {
            AddWord(words, word + 1, low >> (64 - offset));
            AddWord(words, word + 1, high << offset);
            AddWord(words, word + 2, high >> (64 - offset));
        }
    }

    double Rounded() const {
        int comparison = 0;
        for (std::size_t index = kWordCount; index-- > 0;)
            if (positive_[index] != negative_[index]) {
                comparison = positive_[index] > negative_[index] ? 1 : -1;
                break;
            }
        if (comparison == 0) return 0.;
        const auto& larger = comparison > 0 ? positive_ : negative_;
        const auto& smaller = comparison > 0 ? negative_ : positive_;
        Words magnitude;
        std::uint64_t borrow = 0;
        for (std::size_t index = 0; index < kWordCount; ++index) {
            const auto subtrahend = smaller[index] + borrow;
            const bool wrapped = subtrahend < smaller[index];
            magnitude[index] = larger[index] - subtrahend;
            borrow = wrapped || larger[index] < subtrahend;
        }
        std::size_t highest_word = kWordCount - 1;
        while (magnitude[highest_word] == 0) --highest_word;
        const int highest = static_cast<int>(highest_word * 64) + TopBit(magnitude[highest_word]);
        constexpr int maximum_bit = 1023 - kUnitExponent;
        constexpr std::uint64_t maximum_significand = (UINT64_C(1) << 53) - 1;
        if (highest > maximum_bit ||
            (highest == maximum_bit && Extract(magnitude, maximum_bit - 52) == maximum_significand &&
             AnyBelow(magnitude, maximum_bit - 52)))
            throw std::overflow_error("base parameter coordinates exceed the finite double range");
        int shift = std::max(1074, highest - 52);
        std::uint64_t significand = Extract(magnitude, shift);
        const bool halfway = ((magnitude[static_cast<std::size_t>((shift - 1) / 64)] >> ((shift - 1) % 64)) & 1) != 0;
        if (halfway && (AnyBelow(magnitude, shift - 1) || (significand & 1) != 0)) ++significand;
        if (significand == 0) throw std::underflow_error("base parameter coordinates underflow the double range");
        if (significand == (UINT64_C(1) << 53)) {
            significand >>= 1;
            ++shift;
        }
        const double result = std::ldexp(static_cast<double>(significand), shift + kUnitExponent);
        return comparison > 0 ? result : -result;
    }
};

inline double ExactParameterDot(const Eigen::MatrixXd& mapping, Eigen::Index row, const Eigen::VectorXd& values) {
    ParameterDotAccumulator accumulator;
    for (Eigen::Index column = 0; column < values.size(); ++column)
        accumulator.AddProduct(mapping(row, column), values[column]);
    return accumulator.Rounded();
}

constexpr bool kWideParameterDot =
    std::numeric_limits<long double>::radix == 2 &&
    std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
    std::numeric_limits<long double>::digits >= std::numeric_limits<Eigen::Index>::digits &&
    std::numeric_limits<long double>::max_exponent > 2 * 1024 + std::numeric_limits<Eigen::Index>::digits + 16 &&
    std::numeric_limits<long double>::min_exponent < -2148 - std::numeric_limits<long double>::digits - 16;

inline bool ScreenParameterDot(long double sum, long double margin, double& result) {
    // The margin also covers endpoint-rounding error; no directed-rounding
    // mode changes or nextafter calls are required on the ordinary path.
    const long double lower = sum - margin, upper = sum + margin;
    const double maximum = std::numeric_limits<double>::max();
    if (lower < -maximum || upper > maximum) return false;
    const double left = static_cast<double>(lower), right = static_cast<double>(upper);
    if (left != right || left == 0.) return false;
    result = left;
    return true;
}

inline double ParameterProduct(double coefficient, double value, bool use_wide_screen) {
    if (coefficient == 0. || value == 0.) return 0.;
    if (std::abs(coefficient) == 1. || std::abs(value) == 1.) return coefficient * value;
    if (use_wide_screen) {
        const long double product = static_cast<long double>(coefficient) * value;
        double result;
        if (ScreenParameterDot(product, (4.L * std::numeric_limits<long double>::epsilon()) * std::abs(product),
                               result))
            return result;
    }
    ParameterDotAccumulator accumulator;
    accumulator.AddProduct(coefficient, value);
    return accumulator.Rounded();
}

inline double ParameterDot(const Eigen::MatrixXd& mapping,
                           Eigen::Index row,
                           const Eigen::VectorXd& values,
                           bool use_wide_screen) {
    if (!use_wide_screen) return ExactParameterDot(mapping, row, values);
    long double wide_sum = 0.L;
    long double absolute_product_sum = 0.L;
    long double summation_correction = 0.L;
    long double block_sum = 0.L;
    // Fixed small blocks reduce the serial compensation work without storing
    // extra vectors or changing the public object layout.
    constexpr unsigned kProductsPerBlock = 8;  // Fixed by the rounding-error bound below.
    unsigned block_product_count = 0;
    const auto merge_block = [&](long double value) {
        long double left = wide_sum, right = value;
        if (std::abs(left) < std::abs(right)) std::swap(left, right);
        wide_sum = left + right;
        summation_correction += right - (wide_sum - left);
    };
    Eigen::Index nonzero_product_count = 0;
    double last_coefficient = 0., last_value = 0.;

    for (Eigen::Index column = 0; column < values.size(); ++column) {
        const double coefficient = mapping(row, column), value = values[column];
        if (coefficient == 0. || value == 0.) continue;
        const long double product = static_cast<long double>(coefficient) * value;
        block_sum += product;
        if (++block_product_count == kProductsPerBlock) {
            merge_block(block_sum);
            block_sum = 0.L;
            block_product_count = 0;
        }
        absolute_product_sum += std::abs(product);
        ++nonzero_product_count;
        last_coefficient = coefficient;
        last_value = value;
    }

    if (block_product_count != 0) merge_block(block_sum);
    if (nonzero_product_count == 0) return 0.;
    if (nonzero_product_count == 1 && (std::abs(last_coefficient) == 1. || std::abs(last_value) == 1.))
        return last_coefficient * last_value;

    const long double wide_product_count = static_cast<long double>(nonzero_product_count);
    if (wide_product_count * wide_product_count * std::numeric_limits<long double>::epsilon() <= 1.L / 32.L) {
        // Each block contains at most eight nonzero products. Its seven
        // additions contribute at most gamma_7 times their absolute sum.
        // FastTwoSum compensates the block merges. Including product error,
        // the count guard bounds the total by 10.2 wide unit roundoffs times
        // absolute product sum; the sixteen-unit margin also covers inward-rounded
        // interval endpoints. Uncertain rows still use the exact accumulator.
        const long double margin = (8.L * std::numeric_limits<long double>::epsilon()) * absolute_product_sum;
        double result;
        if (ScreenParameterDot(wide_sum + summation_correction, margin, result)) return result;
    }
    return ExactParameterDot(mapping, row, values);
}

}  // namespace detail
}  // namespace robot_identification
