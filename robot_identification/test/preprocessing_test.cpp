#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "robot_identification/identification.h"

namespace {

void TestSymmetricMeans() {
    constexpr Eigen::Index count = 513;
    robot_identification::RawDataset raw;
    raw.joint_names = {"wave", "offset", "alternating", "impulse"};
    raw.time_seconds = Eigen::VectorXd::LinSpaced(count, 0.0, count - 1);
    raw.position.resize(count, 4);
    for (Eigen::Index row = 0; row < count; ++row) {
        raw.position(row, 0) = std::sin(0.17 * row);
        raw.position(row, 1) = 1e12 + (row % 9 - 4) * 0.25;
        raw.position(row, 2) = (row % 2 == 0 ? 1000.0 : -1000.0) + 0.001 * row;
        raw.position(row, 3) = row == 37 ? 1e100 : 1.0;
    }
    raw.torque = raw.position;
    for (std::size_t window :
         {std::size_t{1}, std::size_t{3}, std::size_t{31}, std::size_t{33}, std::size_t{35}, std::size_t{65},
          std::size_t{67}, std::size_t{101}, std::size_t{1001}, std::numeric_limits<std::size_t>::max()}) {
        robot_identification::PreprocessOptions options;
        options.smoothing_window = window;
        options.torque_smoothing_window = window;
        const auto result = robot_identification::PreprocessPositionAndTorque(raw, options);
        const auto radius = static_cast<Eigen::Index>(std::min(window / 2, static_cast<std::size_t>(count - 1)));
        for (Eigen::Index row = 0; row < count; ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = std::min(count - 1, row + radius);
            for (Eigen::Index column = 0; column < raw.position.cols(); ++column) {
                long double sum = 0;
                for (Eigen::Index index = begin; index <= end; ++index) sum += raw.position(index, column);
                const double expected = static_cast<double>(sum / (end - begin + 1));
                const auto& sample = result.samples[static_cast<std::size_t>(row)];
                const double tolerance = 1e-12 + 16 * std::numeric_limits<double>::epsilon() * std::abs(expected);
                if (std::abs(sample.position[column] - expected) > tolerance ||
                    std::abs(sample.torque[column] - expected) > tolerance) {
                    throw std::runtime_error("symmetric smoothing differs from an independent window mean");
                }
            }
        }
    }
}

void TestOffsetIndependentDerivatives() {
    robot_identification::RawDataset raw;
    raw.joint_names = {"positive", "negative", "stationary"};
    raw.time_seconds.resize(7);
    raw.time_seconds << 0.0, 0.125, 0.375, 0.75, 1.0, 1.5, 2.0;
    raw.position.resize(7, 3);
    raw.torque = Eigen::MatrixXd::Zero(7, 3);
    for (Eigen::Index row = 0; row < 7; ++row) {
        raw.position(row, 0) = 1e12 + 2 * raw.time_seconds[row];
        raw.position(row, 1) = -1e12 - 3 * raw.time_seconds[row];
        raw.position(row, 2) = 1e12;
    }
    const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
    for (const auto& sample : result.samples) {
        if (sample.velocity[0] != 2 || sample.velocity[1] != -3 || sample.velocity[2] != 0 ||
            !sample.acceleration.isZero(0)) {
            throw std::runtime_error("constant position offsets changed linear-motion derivatives");
        }
    }

    raw.time_seconds << 0.0, 0.011, 0.027, 0.041, 0.068, 0.083, 0.12;
    raw.position.setConstant(1e12);
    const auto stationary = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
    for (const auto& sample : stationary.samples) {
        if (!sample.velocity.isZero(0) || !sample.acceleration.isZero(0)) {
            throw std::runtime_error("stationary input produced spurious derivatives under timestamp jitter");
        }
    }
}

void TestExtremeSpacing() {
    robot_identification::RawDataset raw;
    raw.joint_names = {"joint"};
    raw.time_seconds = Eigen::VectorXd::LinSpaced(5, 0.0, 4e-170);
    raw.position = raw.time_seconds;
    raw.torque = Eigen::MatrixXd::Zero(5, 1);
    const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
    for (const auto& sample : result.samples) {
        if (sample.velocity[0] != 1 || sample.acceleration[0] != 0) {
            throw std::runtime_error("representable secant slopes were lost at small timestamp spacing");
        }
    }

    raw.time_seconds = Eigen::VectorXd::LinSpaced(5, 0.0, 4.0);
    raw.time_seconds[0] = -1e308;
    raw.time_seconds[1] = 1e308;
    raw.time_seconds[2] = 1.1e308;
    raw.time_seconds[3] = 1.2e308;
    raw.time_seconds[4] = 1.3e308;
    bool rejected = false;
    try {
        (void)robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("overflowed timestamp span was accepted");
}

void TestFiniteExtremeMeans() {
    robot_identification::RawDataset raw;
    raw.joint_names = {"positive", "negative", "alternating"};
    raw.time_seconds = Eigen::VectorXd::LinSpaced(101, 0.0, 100.0);
    raw.position = Eigen::MatrixXd::Zero(101, 3);
    raw.torque.resize(101, 3);
    const double maximum = std::numeric_limits<double>::max();
    for (Eigen::Index row = 0; row < 101; ++row) {
        raw.torque(row, 0) = maximum;
        raw.torque(row, 1) = -maximum;
        raw.torque(row, 2) = row % 2 == 0 ? maximum : -maximum;
    }
    for (std::size_t window : {std::size_t{1}, std::size_t{3}, std::size_t{65}, std::size_t{67}}) {
        robot_identification::PreprocessOptions options;
        options.smoothing_window = 1;
        options.torque_smoothing_window = window;
        const auto result = robot_identification::PreprocessPositionAndTorque(raw, options);
        const auto radius = static_cast<Eigen::Index>(window / 2);
        for (Eigen::Index row = 0; row < 101; ++row) {
            const Eigen::Index begin = std::max<Eigen::Index>(0, row - radius);
            const Eigen::Index end = std::min<Eigen::Index>(100, row + radius);
            const double expected =
                (end - begin + 1) % 2 == 0 ? 0.0 : (begin % 2 == 0 ? maximum : -maximum) / (end - begin + 1);
            const auto& sample = result.samples[static_cast<std::size_t>(row)];
            if (!sample.torque.allFinite() || sample.torque[0] != maximum || sample.torque[1] != -maximum ||
                std::abs(sample.torque[2] / maximum - expected / maximum) > 1e-15 || !sample.velocity.isZero(0) ||
                !sample.acceleration.isZero(0)) {
                throw std::runtime_error("finite extreme window means overflowed or changed stationary motion");
            }
        }
    }
}

void TestInvalidNamesBeforeNumericalWork() {
    const Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(3, 0.0, 2.0);
    Eigen::MatrixXd position(3, 2);
    position << -1e308, 0, 1e308, 0, -1e308, 0;
    const Eigen::MatrixXd torque = Eigen::MatrixXd::Zero(3, 2);
    for (const auto& names : {std::vector<std::string>{"same", "same"}, std::vector<std::string>{"", "joint"}}) {
        bool rejected = false;
        try {
            (void)robot_identification::PreprocessPositionAndTorque(names, time, position, torque, {1, 0, 0});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("invalid names reached numerical preprocessing");
    }
}

void TestPreprocessingEntryValidation() {
    robot_identification::RawDataset valid;
    valid.joint_names = {"joint"};
    valid.time_seconds = Eigen::Vector3d(0, 1, 2);
    valid.position = Eigen::Vector3d(0, 1, 4);
    valid.torque = Eigen::Vector3d(1, 2, 3);
    for (int fault = 0; fault < 6; ++fault) {
        auto raw = valid;
        if (fault == 0) raw.time_seconds[1] = 0;
        if (fault == 1) raw.time_seconds[2] = -1;
        if (fault == 2) raw.time_seconds[1] = std::numeric_limits<double>::quiet_NaN();
        if (fault == 3) raw.position.resize(2, 1);
        if (fault == 4) raw.torque(0, 0) = std::numeric_limits<double>::infinity();
        if (fault == 5) raw.joint_names[0].clear();
        std::string reason;
        if (raw.Validate(&reason)) throw std::runtime_error("invalid raw input passed validation");
        for (bool structured : {false, true}) {
            bool rejected = false;
            try {
                if (structured) {
                    (void)robot_identification::PreprocessPositionAndTorque(raw);
                } else {
                    (void)robot_identification::PreprocessPositionAndTorque(raw.joint_names, raw.time_seconds,
                                                                            raw.position, raw.torque);
                }
            } catch (const std::invalid_argument& error) {
                rejected = std::string(error.what()) == "invalid raw dataset: " + reason;
            }
            if (!rejected) throw std::runtime_error("preprocessing entry points disagree on invalid input");
        }
    }
}

void TestRepresentableDerivativesAcrossOverflowingDifferences() {
    for (bool alternating : {false, true}) {
        robot_identification::RawDataset raw;
        raw.joint_names = {"extreme"};
        const Eigen::Index count = alternating ? 7 : 3;
        const double step = alternating ? 1.5 : 2.;
        raw.time_seconds = Eigen::VectorXd::LinSpaced(count, 0., (count - 1) * step);
        raw.position.resize(count, 1);
        raw.torque = Eigen::MatrixXd::Zero(count, 1);
        if (alternating) {
            const double maximum = std::numeric_limits<double>::max();
            raw.position << 0., 0., maximum, -maximum, maximum, 0., 0.;
        } else {
            raw.position << -1e308, 1e308, 1.7e308;
        }
        const auto reference = [step](const Eigen::MatrixXd& values) {
            Eigen::MatrixXd derivative(values.rows(), 1);
            for (Eigen::Index row = 0; row < values.rows(); ++row) {
                // Independent equal-spacing Lagrange derivative weights.
                long double numerator;
                if (row == 0)
                    numerator = -3.L * values(0, 0) + 4.L * values(1, 0) - values(2, 0);
                else if (row + 1 == values.rows())
                    numerator = 3.L * values(row, 0) - 4.L * values(row - 1, 0) + values(row - 2, 0);
                else
                    numerator = static_cast<long double>(values(row + 1, 0)) - values(row - 1, 0);
                derivative(row, 0) = static_cast<double>(numerator / (2.L * step));
            }
            return derivative;
        };
        const auto expected_velocity = reference(raw.position);
        const auto expected_acceleration = reference(expected_velocity);
        if (!expected_velocity.allFinite() || !expected_acceleration.allFinite())
            throw std::runtime_error("invalid representable differentiation reference");
        const auto actual = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
        for (Eigen::Index row = 0; row < count; ++row) {
            const auto& sample = actual.samples[static_cast<std::size_t>(row)];
            for (const auto& pair : {std::pair{sample.velocity[0], expected_velocity(row, 0)},
                                     std::pair{sample.acceleration[0], expected_acceleration(row, 0)}}) {
                const long double error = std::abs(static_cast<long double>(pair.first) - pair.second);
                if (error > 32.L * std::numeric_limits<double>::epsilon() *
                                std::max(1.L, std::abs(static_cast<long double>(pair.second)))) {
                    throw std::runtime_error("representable derivative changed across an overflowing intermediate");
                }
            }
        }
    }
    robot_identification::RawDataset unusable;
    unusable.joint_names = {"extreme"};
    unusable.time_seconds = Eigen::Vector3d(0., 1., 2.);
    const double maximum = std::numeric_limits<double>::max();
    unusable.position = Eigen::Vector3d(0., maximum, -maximum);
    unusable.torque = Eigen::Vector3d::Zero();
    bool rejected = false;
    try {
        (void)robot_identification::PreprocessPositionAndTorque(unusable, {1, 0, 0});
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("unrepresentable derivative was accepted after extended-range retry");
}

void TestHighlyUnequalSpacing() {
    for (double short_step : {std::numeric_limits<double>::denorm_min(), 1e-308, 1e-200, 1e-30}) {
        for (bool reverse : {false, true}) {
            robot_identification::RawDataset raw;
            raw.joint_names = {"quadratic"};
            raw.time_seconds = reverse ? Eigen::Vector3d(-2., -short_step, 0.) : Eigen::Vector3d(0., short_step, 2.);
            raw.position = reverse ? Eigen::Vector3d(2e307, 0., 0.) : Eigen::Vector3d(0., 0., 2e307);
            raw.torque = Eigen::Vector3d::Zero();
            // q(t) = a*t*(t-short_step), mirrored in time when requested.
            // Its small derivatives remain finite even when short_step/2
            // underflows, or an interior slope weight rounds to one in double.
            const long double coefficient = 2e307L / (2.L * (2.L - short_step));
            const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
            for (std::size_t row = 0; row < result.samples.size(); ++row) {
                const long double time =
                    reverse ? -static_cast<long double>(raw.time_seconds[row]) : raw.time_seconds[row];
                const long double velocity = (reverse ? -1.L : 1.L) * coefficient * (2.L * time - short_step);
                const long double acceleration = 2.L * coefficient;
                const auto& sample = result.samples[row];
                for (const auto& pair :
                     {std::pair{sample.velocity[0], velocity}, std::pair{sample.acceleration[0], acceleration}}) {
                    if (std::abs(static_cast<long double>(pair.first) - pair.second) >
                        32.L * std::numeric_limits<double>::epsilon() * std::abs(pair.second)) {
                        throw std::runtime_error("unequal timestamp spacing lost a representable derivative");
                    }
                }
            }
        }
    }
}

void TestUnderflowingSecants() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (int exponent : {1, 500, 1000}) {
        for (double sign : {-1., 1.}) {
            const double step = std::ldexp(1., exponent);
            const double amplitude = sign * std::ldexp(tiny, exponent - 1);
            robot_identification::RawDataset raw;
            raw.joint_names = {"quadratic", "linear", "stationary"};
            raw.time_seconds = Eigen::Vector3d(0., step, 2. * step);
            raw.position.resize(3, 3);
            raw.position.col(0) = Eigen::Vector3d(0., amplitude, 0.);
            raw.position.col(1) = Eigen::Vector3d(0., amplitude, 2. * amplitude);
            raw.position.col(2).setConstant(amplitude);
            raw.torque = Eigen::MatrixXd::Zero(3, 3);
            const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
            for (Eigen::Index row = 0; row < 3; ++row) {
                const auto& sample = result.samples[static_cast<std::size_t>(row)];
                // q(t) = amplitude*t*(2*step-t)/(step*step) has endpoint
                // derivatives +/-sign*tiny, though its secants round to zero.
                const double expected = row == 0 ? sign * tiny : row == 2 ? -sign * tiny : 0.;
                if (sample.velocity[0] != expected || !sample.velocity.tail(2).isZero(0.) ||
                    !sample.acceleration.isZero(0.)) {
                    throw std::runtime_error("individually underflowing secants erased a representable derivative");
                }
            }
        }
    }
    // Neighboring subnormal slopes can round into the normal range before
    // endpoint extrapolation, losing one ulp even in a normal final derivative.
    for (double amplitude :
         {std::numeric_limits<double>::min() - tiny, 2. * std::numeric_limits<double>::min() - tiny}) {
        for (double final_time : {4., 6.}) {
            robot_identification::RawDataset raw;
            raw.joint_names = {"quadratic"};
            raw.time_seconds = Eigen::Vector3d(0., 2., final_time);
            raw.position = Eigen::Vector3d(0., amplitude, 0.);
            raw.torque = Eigen::Vector3d::Zero();
            const auto differentiate = [&](const Eigen::Vector3d& values) {
                Eigen::Vector3d derivative;
                for (Eigen::Index row = 0; row < 3; ++row) {
                    long double sum = 0.;
                    for (Eigen::Index term = 0; term < 3; ++term) {
                        const auto a = (term + 1) % 3, b = (term + 2) % 3;
                        const long double weight =
                            (2.L * raw.time_seconds[row] - raw.time_seconds[a] - raw.time_seconds[b]) /
                            ((static_cast<long double>(raw.time_seconds[term]) - raw.time_seconds[a]) *
                             (static_cast<long double>(raw.time_seconds[term]) - raw.time_seconds[b]));
                        sum += weight * values[term];
                    }
                    derivative[row] = static_cast<double>(sum);
                }
                return derivative;
            };
            const auto velocity = differentiate(raw.position);
            const auto acceleration = differentiate(velocity);
            const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
            for (Eigen::Index row = 0; row < 3; ++row) {
                const auto& sample = result.samples[static_cast<std::size_t>(row)];
                if (sample.velocity[0] != velocity[row] || sample.acceleration[0] != acceleration[row]) {
                    throw std::runtime_error("subnormal secant rounding changed a normal endpoint derivative");
                }
            }
        }
    }
}

void TestEqualStepCenterEliminatesMiddleSample() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (int fixture = 0; fixture < 3; ++fixture) {
        for (double sign : {-1., 1.}) {
            robot_identification::RawDataset raw;
            raw.joint_names = {"quadratic"};
            double step, expected_velocity;
            if (fixture == 0) {
                step = 1.;
                raw.position = Eigen::Vector3d(0., 0x1p-1016, 8. * tiny);
                expected_velocity = 4. * tiny;
            } else if (fixture == 1) {
                step = 0x1p+32;
                raw.position = Eigen::Vector3d(-0x1p-1007, -0x1.ffffffffffff9p-958, -0x1.2p-1007);
                expected_velocity = -0x0.000008p-1022;
            } else {
                step = 2.;
                raw.position = Eigen::Vector3d(4. * tiny, std::numeric_limits<double>::max(), 12. * tiny);
                expected_velocity = 2. * tiny;
            }
            raw.position *= sign;
            raw.time_seconds = Eigen::Vector3d(0., step, 2. * step);
            raw.torque = Eigen::Vector3d::Zero();
            const auto result = robot_identification::PreprocessPositionAndTorque(raw, {1, 0, 0});
            // The middle q coefficient is exactly zero in the centered stencil.
            // Even long-double adjacent secants lose these endpoint differences
            // when the middle q sample is close to the double maximum.
            if (result.samples[1].velocity[0] != sign * expected_velocity) {
                throw std::runtime_error("equal-step differentiation retained the cancelled middle sample");
            }
            // The acceleration pass must center the already rounded velocities,
            // including an endpoint difference that overflows in double.
            const double expected_acceleration = static_cast<double>(
                (static_cast<long double>(result.samples[2].velocity[0]) - result.samples[0].velocity[0]) /
                (2.L * step));
            if (result.samples[1].acceleration[0] != expected_acceleration) {
                throw std::runtime_error("equal-step acceleration lost the rounded velocity endpoint difference");
            }
        }
    }
}

}  // namespace

int main() {
    TestSymmetricMeans();
    TestOffsetIndependentDerivatives();
    TestExtremeSpacing();
    TestFiniteExtremeMeans();
    TestInvalidNamesBeforeNumericalWork();
    TestPreprocessingEntryValidation();
    TestRepresentableDerivativesAcrossOverflowingDifferences();
    TestHighlyUnequalSpacing();
    TestUnderflowingSecants();
    TestEqualStepCenterEliminatesMiddleSample();
}
