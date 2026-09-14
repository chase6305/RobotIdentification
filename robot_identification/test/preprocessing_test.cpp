#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
            const double expected = (end - begin + 1) % 2 == 0 ? 0.0 :
                                    (begin % 2 == 0 ? maximum : -maximum) / (end - begin + 1);
            const auto& sample = result.samples[static_cast<std::size_t>(row)];
            if (!sample.torque.allFinite() || sample.torque[0] != maximum || sample.torque[1] != -maximum ||
                std::abs(sample.torque[2] / maximum - expected / maximum) > 1e-15 ||
                !sample.velocity.isZero(0) || !sample.acceleration.isZero(0)) {
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
    for (const auto& names : {std::vector<std::string>{"same", "same"},
                              std::vector<std::string>{"", "joint"}}) {
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
                    (void)robot_identification::PreprocessPositionAndTorque(
                        raw.joint_names, raw.time_seconds, raw.position, raw.torque);
                }
            } catch (const std::invalid_argument& error) {
                rejected = std::string(error.what()) == "invalid raw dataset: " + reason;
            }
            if (!rejected) throw std::runtime_error("preprocessing entry points disagree on invalid input");
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
}
