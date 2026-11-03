// Closed-form extended-range harmonic samples and envelope checks.
#include <robot_identification/identification.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "stress_duration.h"

namespace {
using WideVector = Eigen::Matrix<long double, Eigen::Dynamic, 1>;
void Compare(double actual,
             long double reference,
             long double magnitude,
             const char* label,
             long double rounding_allowance = 128.L) {
    if (std::abs(reference) > std::numeric_limits<double>::max()) {
        if (std::isinf(actual) && std::signbit(actual) == std::signbit(reference)) return;
    } else {
        const long double tolerance = rounding_allowance * std::numeric_limits<double>::epsilon() * magnitude +
                                      4.L * std::numeric_limits<double>::denorm_min();
        if (std::isfinite(actual) && std::abs(static_cast<long double>(actual) - reference) <= tolerance) return;
    }
    std::cerr << std::setprecision(21) << label << " actual=" << actual << " reference=" << reference
              << " magnitude=" << magnitude << '\n';
    throw std::runtime_error("harmonic result differs from the closed-form reference");
}
}  // namespace

int main(int argc, char** argv) {
    const auto [seconds, first_case] = synthetic_validation::ReplayCampaign(argc, argv);
    if (std::numeric_limits<long double>::max_exponent <= 6500) {
        std::cerr << "This validation oracle requires extended-range long double.\n";
        return 2;
    }
    const double pi = std::acos(-1.);
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, accepted = 0, rejected = 0, extreme_cases = 0, reused_outputs = 0;
    double next_progress = 0, hz = 0, time = 0;
    Eigen::VectorXd offset;
    Eigen::MatrixXd a, b;
    robot_identification::ExcitationSample actual;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            const auto index = first_case + cases;
            std::mt19937_64 random(0x20261029a71cULL ^ (index * 0x9e3779b97f4a7c15ULL));
            std::uniform_real_distribution<double> unit(-1., 1.);
            if (index % 32 == 0) {
                offset = Eigen::VectorXd::Zero(1);
                a = b = Eigen::MatrixXd::Zero(1, 1);
                hz = 1e100 / (2. * pi);
                time = 1e-200;
                const auto variant = (index / 32) % 9;
                switch (variant) {
                    case 0:
                        a(0, 0) = 1e-250;
                        break;
                    case 1:
                        a(0, 0) = b(0, 0) = std::numeric_limits<double>::denorm_min();
                        time = pi / (4. * 1e100);
                        break;
                    case 2:
                        a(0, 0) = 1e308;
                        hz = 1e-100 / (2. * pi);
                        time = 1e-300;
                        break;
                    case 3:
                        offset[0] = 1e308;
                        a = b = Eigen::MatrixXd::Zero(1, 2);
                        b << 1e308, -1e308;
                        hz = 1e-110 / (2. * pi);
                        time = 0.;
                        break;
                    case 4:
                        offset = Eigen::VectorXd::Ones(2);
                        a = b = Eigen::MatrixXd::Zero(2, 8);
                        hz = std::numeric_limits<double>::max() / 8.;
                        time = std::numeric_limits<double>::max();
                        break;
                    case 5:
                        a(0, 0) = b(0, 0) = 1.4e308;
                        hz = 1e-110 / (2. * pi);
                        time = 0.;
                        break;
                    default: {
                        const int derivative = static_cast<int>(variant) - 5;
                        hz = 1e-100 / (2. * pi);
                        a = b = Eigen::MatrixXd::Zero(1, 32);
                        for (Eigen::Index harmonic = 0; harmonic < a.cols(); ++harmonic) {
                            const double frequency = static_cast<double>(harmonic + 1) * (2. * pi * hz);
                            long double power = 1.L;
                            for (int factor = 0; factor < derivative; ++factor) power *= frequency;
                            (derivative % 2 == 0 ? b : a)(0, harmonic) =
                                static_cast<double>(.25L * std::numeric_limits<double>::denorm_min() / power);
                        }
                        time = 0.;
                        break;
                    }
                }
                ++extreme_cases;
            } else {
                const int dof = 1 + random() % 14, harmonics = 1 + random() % 20;
                hz = std::ldexp(1., static_cast<int>(random() % 2001) - 1000) / (2. * pi);
                time = std::ldexp(unit(random), static_cast<int>(random() % 2098) - 1074);
                offset = Eigen::VectorXd::Zero(dof);
                a.resize(dof, harmonics);
                b.resize(dof, harmonics);
                for (int harmonic = 0; harmonic < harmonics; ++harmonic)
                    for (int joint = 0; joint < dof; ++joint) {
                        const int exponent = static_cast<int>(random() % 2075) - 1074;
                        a(joint, harmonic) = random() % 4 ? std::ldexp(unit(random), exponent) : 0.;
                        b(joint, harmonic) = random() % 4 ? std::ldexp(unit(random), exponent) : 0.;
                    }
            }
            const double angular = 2. * pi * hz;
            std::array<WideVector, 4> reference, magnitude;
            for (auto& values : reference) values = WideVector::Zero(offset.size());
            for (auto& values : magnitude) values = WideVector::Zero(offset.size());
            reference[0] = offset.cast<long double>();
            magnitude[0] = reference[0].cwiseAbs();
            std::array<WideVector, 4> envelopes;
            for (auto& values : envelopes) values = WideVector::Zero(offset.size());
            bool phase_valid = true;
            for (Eigen::Index harmonic = 0; harmonic < a.cols(); ++harmonic) {
                const double frequency = static_cast<double>(harmonic + 1) * angular, phase = frequency * time;
                const long double f = std::isfinite(frequency) ? static_cast<long double>(frequency)
                                                               : static_cast<long double>(harmonic + 1) * angular;
                for (Eigen::Index joint = 0; joint < offset.size(); ++joint) {
                    const long double x = a(joint, harmonic), y = b(joint, harmonic);
                    if (x == 0.L && y == 0.L) continue;
                    long double amplitude = std::hypot(x, y);
                    for (auto& envelope : envelopes) {
                        envelope[joint] += amplitude;
                        amplitude *= f;
                    }
                    if (!std::isfinite(phase)) {
                        phase_valid = false;
                        continue;
                    }
                    const bool small_phase = time != 0. && std::abs(phase) < std::numeric_limits<double>::min();
                    const long double sine = small_phase ? std::sin(f * time) : std::sin(phase);
                    const long double cosine = small_phase ? std::cos(f * time) : std::cos(phase);
                    const std::array<long double, 4> terms{x * sine + y * cosine, f * (x * cosine - y * sine),
                                                           -f * f * (x * sine + y * cosine),
                                                           -f * f * f * (x * cosine - y * sine)};
                    long double power = 1.L;
                    for (std::size_t derivative = 0; derivative < terms.size(); ++derivative) {
                        reference[derivative][joint] += terms[derivative];
                        magnitude[derivative][joint] +=
                            power * (derivative % 2 == 0 ? std::abs(x * sine) + std::abs(y * cosine)
                                                         : std::abs(x * cosine) + std::abs(y * sine));
                        power *= f;
                    }
                }
            }
            bool representable = phase_valid;
            for (const auto& values : reference)
                if ((values.cwiseAbs().array() > std::numeric_limits<double>::max()).any()) representable = false;
            const robot_identification::HarmonicTrajectory trajectory(offset, a, b, hz);
            bool failed = false;
            try {
                if (index % 2 == 0) {
                    trajectory.EvaluateInto(time, actual);
                    ++reused_outputs;
                } else {
                    actual = trajectory.Evaluate(time);
                }
            } catch (const std::overflow_error&) {
                failed = true;
            }
            if (failed == representable) throw std::runtime_error("trajectory accepted/rejected the wrong range");
            if (failed)
                ++rejected;
            else {
                const std::array<const Eigen::VectorXd*, 4> fields{&actual.position, &actual.velocity,
                                                                   &actual.acceleration, &actual.jerk};
                for (std::size_t derivative = 0; derivative < fields.size(); ++derivative)
                    for (Eigen::Index joint = 0; joint < offset.size(); ++joint)
                        Compare((*fields[derivative])[joint], reference[derivative][joint],
                                magnitude[derivative][joint], "sample");
                ++accepted;
            }
            const auto bounds = trajectory.ConservativeBounds();
            for (Eigen::Index joint = 0; joint < offset.size(); ++joint) {
                const long double lower = offset[joint] - envelopes[0][joint],
                                  upper = offset[joint] + envelopes[0][joint];
                const long double position_magnitude =
                    std::abs(static_cast<long double>(offset[joint])) + envelopes[0][joint];
                Compare(bounds.minimum_position[joint], lower, position_magnitude, "minimum_position", 8.L);
                Compare(bounds.maximum_position[joint], upper, position_magnitude, "maximum_position", 8.L);
                Compare(bounds.maximum_absolute_velocity[joint], envelopes[1][joint], envelopes[1][joint],
                        "velocity_bound", 8.L);
                Compare(bounds.maximum_absolute_acceleration[joint], envelopes[2][joint], envelopes[2][joint],
                        "acceleration_bound", 8.L);
                Compare(bounds.maximum_absolute_jerk[joint], envelopes[3][joint], envelopes[3][joint], "jerk_bound",
                        8.L);
            }
            ++cases;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " accepted=" << accepted
                          << " rejected=" << rejected << " extreme_cases=" << extreme_cases
                          << " reused_outputs=" << reused_outputs << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << first_case + cases << ": " << error.what() << '\n'
                  << std::setprecision(17) << "hz=" << hz << " time=" << time << "\noffset=" << offset.transpose()
                  << "\nsine=\n"
                  << a << "\ncosine=\n"
                  << b << std::endl;
        return 1;
    }
    std::cout << "PASSED first_case=" << first_case << " cases=" << cases << " accepted=" << accepted
              << " rejected=" << rejected << " extreme_cases=" << extreme_cases << " reused_outputs=" << reused_outputs
              << std::endl;
}
