// Independent exact dyadic reference for the public ToBaseParameters API.
#include <robot_identification/base_parameters.h>

#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using boost::multiprecision::cpp_int;
struct Dyadic {
    cpp_int numerator;
    int exponent;
};

Dyadic Exact(double value) {
    if (value == 0.) return {0, 0};
    int exponent;
    const double fraction = std::frexp(std::abs(value), &exponent);
    cpp_int numerator = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
    if (std::signbit(value)) numerator = -numerator;
    return {numerator, exponent - 53};
}

Dyadic Add(const Dyadic& left, const Dyadic& right) {
    if (left.numerator == 0) return right;
    if (right.numerator == 0) return left;
    const int exponent = std::min(left.exponent, right.exponent);
    return {(left.numerator << (left.exponent - exponent)) + (right.numerator << (right.exponent - exponent)),
            exponent};
}

int Compare(const Dyadic& left, const Dyadic& right) {
    const int exponent = std::min(left.exponent, right.exponent);
    const cpp_int a = left.numerator << (left.exponent - exponent);
    const cpp_int b = right.numerator << (right.exponent - exponent);
    return a < b ? -1 : a > b ? 1 : 0;
}

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool Odd(double value) {
    if (value < std::numeric_limits<double>::min())
        return (static_cast<std::uint64_t>(value / std::numeric_limits<double>::denorm_min()) & 1) != 0;
    int exponent;
    return (static_cast<std::uint64_t>(std::ldexp(std::frexp(value, &exponent), 53)) & 1) != 0;
}

double RandomFinite(std::mt19937_64& random) {
    std::uint64_t bits = random();
    if (((bits >> 52) & 0x7ff) == 0x7ff) bits ^= UINT64_C(0x0010000000000000);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void Fixture(unsigned long long index, Eigen::MatrixXd& calibration, Eigen::VectorXd& parameters) {
    std::mt19937_64 random(index ^ UINT64_C(0x9eb2f34ac50d8167));
    const int columns = 1 + random() % 257;
    calibration.resize(1, columns);
    parameters.resize(columns);
    for (int column = 0; column < columns; ++column) {
        calibration(0, column) = RandomFinite(random);
        parameters[column] = RandomFinite(random);
    }
    switch (index % 16) {
        case 0:
        case 1: {
            calibration = Eigen::MatrixXd::Ones(1, 3);
            parameters.resize(3);
            const int exponent = static_cast<int>(random() % 1801) - 900;
            const double large = std::ldexp(1., exponent);
            parameters << large, std::ldexp(1., exponent - 100), -large;
            if ((random() & 1) != 0) parameters.reverseInPlace();
            break;
        }
        case 2:
        case 3: {
            calibration.resize(1, 2);
            parameters.resize(2);
            const double a = std::nextafter(1., 0.), c = 1. - std::ldexp(1., -52);
            const int exponent = static_cast<int>(random() % 2001) - 1000;
            calibration << std::ldexp(a, exponent), std::ldexp(c, exponent);
            parameters << a, -1.;
            break;
        }
        case 4:
            calibration = Eigen::MatrixXd::Constant(1, 2, std::numeric_limits<double>::max());
            parameters.resize(2);
            parameters << 2., -2.;
            break;
        case 5:
            calibration.resize(1, 2);
            calibration << std::numeric_limits<double>::max(), std::numeric_limits<double>::denorm_min();
            parameters = Eigen::VectorXd::Ones(2);
            if ((random() & 1) != 0) parameters = -parameters;
            break;
        case 6:
            calibration = Eigen::MatrixXd::Ones(1, 3);
            parameters.resize(3);
            parameters << 1., (random() & 1) == 0 ? 0. : std::ldexp(1., -52), std::ldexp(1., -53);
            break;
        case 7:
            calibration = Eigen::MatrixXd::Constant(1, 2, std::numeric_limits<double>::denorm_min());
            parameters.resize(2);
            parameters << static_cast<double>(random() % 8), .5;
            break;
        case 8:
            calibration.setZero();
            calibration(0, random() % columns) = 1.;
            for (int column = 0; column < columns; ++column)
                parameters[column] = std::ldexp(static_cast<double>(static_cast<int>(random() % 33) - 16),
                                                static_cast<int>(random() % 101) - 50);
            break;
        case 9:
            for (int column = 0; column < columns; ++column) {
                calibration(0, column) = static_cast<double>(static_cast<int>(random() % 2001) - 1000) / 1001.;
                parameters[column] = static_cast<double>(static_cast<int>(random() % 2001) - 1000) / 997.;
            }
            break;
        case 10:
            calibration.setOnes();
            parameters.setOnes();
            if (columns > 1) {
                parameters[0] = std::ldexp(1., 900);
                parameters[columns - 1] = -parameters[0];
            }
            break;
        default:
            break;
    }
    if (calibration.isZero(0.)) calibration(0, 0) = 1.;
}
}  // namespace

int main(int argc, char** argv) {
    using robot_identification::BaseParameterBasis;
    const double seconds = argc > 1 ? std::stod(argv[1]) : 60.;
    const unsigned long long first = argc > 2 ? std::stoull(argv[2]) : 0;
    if (!std::isfinite(seconds) || seconds <= 0.) return 2;
    const auto started = std::chrono::steady_clock::now();
    const int original_rounding = std::fegetround();
    unsigned long long cases = 0, finite = 0, overflow = 0, underflow = 0, rounding_samples = 0;
    double next_progress = 60.;
    Eigen::MatrixXd calibration;
    Eigen::VectorXd parameters;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            const auto index = first + cases;
            Fixture(index, calibration, parameters);
            const BaseParameterBasis basis(calibration);
            Require(basis.parameter_count() == 1, "one-row reference fixture changed its known rank");
            Dyadic exact{0, 0};
            for (Eigen::Index column = 0; column < parameters.size(); ++column) {
                const auto left = Exact(basis.full_to_base()(0, column)), right = Exact(parameters[column]);
                exact = Add(exact, {left.numerator * right.numerator, left.exponent + right.exponent});
            }
            const bool negative = exact.numerator < 0;
            if (negative) exact.numerator = -exact.numerator;
            const bool expected_overflow = Compare(exact, Exact(std::numeric_limits<double>::max())) > 0;
            const Dyadic twice{2 * exact.numerator, exact.exponent};
            const bool expected_underflow =
                exact.numerator != 0 && Compare(twice, Exact(std::numeric_limits<double>::denorm_min())) <= 0;
            int mode = FE_TONEAREST;
            if (index % 4093 == 0) {
                const int modes[]{FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
                mode = modes[(index / 4093) % 4];
                ++rounding_samples;
            }
            Require(std::fesetround(mode) == 0, "cannot set reference rounding mode");
            bool overflowed = false, underflowed = false;
            double actual = 0.;
            try {
                actual = basis.ToBaseParameters(parameters)[0];
            } catch (const std::overflow_error&) {
                overflowed = true;
            } catch (const std::underflow_error&) {
                underflowed = true;
            }
            (void)std::fesetround(original_rounding);
            Require(overflowed == expected_overflow, "exact coordinate overflow classification differs");
            Require(underflowed == expected_underflow, "exact coordinate underflow classification differs");
            if (overflowed) {
                ++overflow;
            } else if (underflowed) {
                ++underflow;
            } else {
                ++finite;
                Require(std::isfinite(actual), "finite exact coordinate became non-finite");
                if (exact.numerator == 0) {
                    Require(actual == 0., "exact coordinate cancellation became nonzero");
                } else {
                    Require(actual != 0. && std::signbit(actual) == negative,
                            "exact coordinate lost magnitude or sign");
                    const double magnitude = std::abs(actual);
                    const auto current = Exact(magnitude);
                    const auto lower = Add(current, Exact(std::nextafter(magnitude, 0.)));
                    const int lower_comparison = Compare(twice, lower);
                    Require(lower_comparison > 0 || (lower_comparison == 0 && !Odd(magnitude)),
                            "coordinate lies above its nearest-even rounding cell");
                    if (magnitude != std::numeric_limits<double>::max()) {
                        const auto upper =
                            Add(current, Exact(std::nextafter(magnitude, std::numeric_limits<double>::infinity())));
                        const int upper_comparison = Compare(twice, upper);
                        Require(upper_comparison < 0 || (upper_comparison == 0 && !Odd(magnitude)),
                                "coordinate lies below its nearest-even rounding cell");
                    }
                }
            }
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " finite=" << finite
                          << " overflow=" << overflow << " underflow=" << underflow
                          << " rounding_samples=" << rounding_samples << std::endl;
                next_progress = elapsed + 60.;
            }
        }
    } catch (const std::exception& error) {
        (void)std::fesetround(original_rounding);
        std::cerr << "FAILED case=" << first + cases << ' ' << error.what() << std::hexfloat
                  << "\ncalibration=" << calibration << "\nparameters=" << parameters.transpose() << '\n';
        return 1;
    }
    (void)std::fesetround(original_rounding);
    std::cout << "PASSED first_case=" << first << " cases=" << cases << " finite=" << finite << " overflow=" << overflow
              << " underflow=" << underflow << " rounding_samples=" << rounding_samples << std::endl;
}
