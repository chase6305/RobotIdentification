#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "robot_identification/identification.h"

namespace {
void RequireClose(double actual, long double reference) {
    const long double tolerance = 64.L * std::numeric_limits<double>::epsilon() * std::abs(reference) +
                                  2.L * std::numeric_limits<double>::denorm_min();
    if (!std::isfinite(actual) || std::abs(static_cast<long double>(actual) - reference) > tolerance)
        throw std::runtime_error("a representable harmonic derivative was lost after an underflowing product");
}

void TestUnderflowingTrigonometricProducts() {
    const double pi = std::acos(-1.);
    for (const auto& input :
         {std::array<double, 3>{1e-250, 1e100 / (2. * pi), 1e-200},
          std::array<double, 3>{std::numeric_limits<double>::denorm_min(), 1e100 / (2. * pi), pi / (2. * 1e100)},
          std::array<double, 3>{1e308, 1e-100 / (2. * pi), 1e-300}}) {
        const auto [coefficient, hz, time] = input;
        const double frequency = 2. * pi * hz;
        const double phase = frequency * time;
        const bool small_phase = std::abs(phase) < std::numeric_limits<double>::min();
        const long double sine = small_phase ? std::sin(static_cast<long double>(frequency) * time) : std::sin(phase);
        const long double cosine = small_phase ? std::cos(static_cast<long double>(frequency) * time) : std::cos(phase);
        const long double amplitude = coefficient, wide_frequency = frequency;
        robot_identification::HarmonicTrajectory trajectory(
            Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, coefficient), Eigen::MatrixXd::Zero(1, 1), hz);
        const auto sample = trajectory.Evaluate(time);
        RequireClose(sample.position[0], amplitude * sine);
        RequireClose(sample.velocity[0], amplitude * wide_frequency * cosine);
        RequireClose(sample.acceleration[0], -amplitude * wide_frequency * wide_frequency * sine);
        RequireClose(sample.jerk[0], -amplitude * wide_frequency * wide_frequency * wide_frequency * cosine);
    }
}

void TestTinyDurationStillIncludesEndpoint() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
                                                        Eigen::MatrixXd::Zero(1, 1), 1. / (2. * std::acos(-1.)));
    const robot_identification::ExcitationLimits limits{
        Eigen::VectorXd::Constant(1, -2.), Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.),
        Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.)};
    const auto result = robot_identification::ValidateExcitationTrajectory(trajectory, limits, tiny,
                                                                           std::numeric_limits<double>::max());
    if (!result.passed || result.sample_count != 2 || result.peak_absolute_acceleration[0] != tiny)
        throw std::runtime_error("a positive trajectory duration omitted its final sample after step-count underflow");
}

void TestRoundedStepCountStillIncludesEndpoint() {
    // The quotient rounds to exactly 20 although the last period multiple
    // is one representable time before duration. The endpoint peak differs.
    const double duration = 0x1.f69b45f5dec85p-4;
    const double period = 0x1.9215d1917f06ap-8;
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
                                                        Eigen::MatrixXd::Zero(1, 1), 1. / (2. * std::acos(-1.)));
    const robot_identification::ExcitationLimits limits{
        Eigen::VectorXd::Constant(1, -2.), Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.),
        Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.)};
    const auto endpoint = trajectory.Evaluate(duration);
    const auto result = robot_identification::ValidateExcitationTrajectory(trajectory, limits, duration, period);
    if (!result.passed || result.sample_count != 21 ||
        result.peak_absolute_acceleration[0] != std::abs(endpoint.acceleration[0]))
        throw std::runtime_error("rounded step count omitted the exact trajectory endpoint");
}

void TestSubnormalDerivativeSums() {
    const double hz = 1e-100 / (2. * std::acos(-1.));
    const double angular = 2. * std::acos(-1.) * hz;
    for (int derivative = 1; derivative <= 3; ++derivative) {
        Eigen::MatrixXd sine = Eigen::MatrixXd::Zero(1, 32), cosine = sine;
        long double expected = 0.L;
        for (Eigen::Index harmonic = 0; harmonic < sine.cols(); ++harmonic) {
            const double frequency = static_cast<double>(harmonic + 1) * angular;
            long double power = 1.L;
            for (int factor = 0; factor < derivative; ++factor) power *= frequency;
            const double coefficient = static_cast<double>(.25L * std::numeric_limits<double>::denorm_min() / power);
            (derivative % 2 == 0 ? cosine : sine)(0, harmonic) = coefficient;
            expected += static_cast<long double>(coefficient) * power;
        }
        if (derivative > 1) expected = -expected;
        const robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(1), sine, cosine, hz);
        const auto sample = trajectory.Evaluate(0.);
        const double actual = derivative == 1   ? sample.velocity[0]
                              : derivative == 2 ? sample.acceleration[0]
                                                : sample.jerk[0];
        if (actual != static_cast<double>(expected))
            throw std::runtime_error("individually underflowing harmonic derivatives lost their representable sum");
    }
}

void TestSubnormalEnvelopeDoesNotAdmitExcessVelocity() {
    const double coefficient = std::numeric_limits<double>::denorm_min();
    const double pi = std::acos(-1.);
    const double hz = 1e100 / (2. * pi), frequency = 2. * pi * hz;
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(1),
                                                        Eigen::MatrixXd::Constant(1, 1, coefficient),
                                                        Eigen::MatrixXd::Constant(1, 1, coefficient), hz);
    const robot_identification::ExcitationLimits limits{Eigen::VectorXd::Constant(1, -2. * coefficient),
                                                        Eigen::VectorXd::Constant(1, 2. * coefficient),
                                                        Eigen::VectorXd::Constant(1, 1.2 * (coefficient * frequency)),
                                                        Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1)};
    const double duration = 2. * pi / frequency;
    const auto peak = trajectory.Evaluate(1.75 * pi / frequency);
    if (peak.velocity[0] <= limits.maximum_velocity[0])
        throw std::runtime_error("the subnormal envelope fixture does not exceed its velocity limit");
    const auto result = robot_identification::ValidateExcitationTrajectory(trajectory, limits, duration, duration);
    if (result.passed || result.reason != "conservative velocity bound exceeded")
        throw std::runtime_error("a rounded subnormal amplitude admitted a velocity-unsafe trajectory");
    const long double amplitude =
        std::hypot(static_cast<long double>(coefficient), static_cast<long double>(coefficient));
    const auto bounds = trajectory.ConservativeBounds();
    RequireClose(bounds.maximum_absolute_velocity[0], amplitude * frequency);
    RequireClose(bounds.maximum_absolute_acceleration[0], amplitude * frequency * frequency);
    RequireClose(bounds.maximum_absolute_jerk[0], amplitude * frequency * frequency * frequency);
}

void TestNormalEnvelopeRoundsOutward() {
    const double frequency = 1. / (2. * std::acos(-1.));
    const robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
                                                              Eigen::MatrixXd::Constant(1, 1, 5.), frequency);
    const double rounded_peak = std::hypot(1., 5.);
    const long double peak = std::hypot(1.L, 5.L);
    const auto bounds = trajectory.ConservativeBounds();
    if (static_cast<long double>(bounds.maximum_position[0]) < peak ||
        static_cast<long double>(bounds.maximum_absolute_velocity[0]) < peak ||
        static_cast<long double>(bounds.maximum_absolute_acceleration[0]) < peak ||
        static_cast<long double>(bounds.maximum_absolute_jerk[0]) < peak)
        throw std::runtime_error("a normal harmonic amplitude rounded its envelope inward");
    const robot_identification::ExcitationLimits limits{
        Eigen::VectorXd::Constant(1, -100.), Eigen::VectorXd::Constant(1, rounded_peak),
        Eigen::VectorXd::Constant(1, 100.), Eigen::VectorXd::Constant(1, 100.), Eigen::VectorXd::Constant(1, 100.)};
    const double duration = 2. * std::acos(-1.);
    const auto result = robot_identification::ValidateExcitationTrajectory(trajectory, limits, duration, duration);
    if (rounded_peak < peak && result.passed)
        throw std::runtime_error("a rounded normal amplitude admitted a position-unsafe trajectory");
    // Exact Pythagorean amplitudes should retain their exact boundary.
    const robot_identification::HarmonicTrajectory exact(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, 3.),
                                                         Eigen::MatrixXd::Constant(1, 1, 4.), frequency);
    if (exact.ConservativeBounds().maximum_position[0] != 5.)
        throw std::runtime_error("an exactly representable envelope was expanded unnecessarily");
}

void TestSmallHarmonicsSurviveEnvelopeAccumulation() {
    const double pi = std::acos(-1.);
    for (double small : {1e-20, 1e-100, std::numeric_limits<double>::denorm_min()}) {
        Eigen::MatrixXd sine = Eigen::MatrixXd::Zero(1, 2), cosine = sine;
        sine(0, 0) = 1.;
        cosine(0, 1) = small;
        const robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Ones(1), sine, cosine,
                                                                  1. / (2. * pi));
        const auto sample = trajectory.Evaluate(1.5 * pi);
        const auto bounds = trajectory.ConservativeBounds();
        if (sample.position[0] >= 0. || bounds.minimum_position[0] > sample.position[0] ||
            bounds.maximum_position[0] <= 2. || bounds.maximum_absolute_velocity[0] <= 1. ||
            bounds.maximum_absolute_acceleration[0] <= 1. || bounds.maximum_absolute_jerk[0] <= 1.)
            throw std::runtime_error("a small harmonic was lost from an outward envelope sum");
        const robot_identification::ExcitationLimits limits{
            Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 10.),
            Eigen::VectorXd::Constant(1, 10.), Eigen::VectorXd::Constant(1, 10.)};
        const auto validation = robot_identification::ValidateExcitationTrajectory(trajectory, limits, 2. * pi, pi);
        if (validation.passed || validation.reason != "conservative position bound exceeded")
            throw std::runtime_error("a lost small harmonic admitted an out-of-limit trajectory");
    }
}

void TestInactiveHarmonicsAtLargeTimes() {
    const double maximum = std::numeric_limits<double>::max();
    robot_identification::HarmonicTrajectory trajectory(Eigen::VectorXd::Constant(2, 1.), Eigen::MatrixXd::Zero(2, 8),
                                                        Eigen::MatrixXd::Zero(2, 8), maximum / 8.);
    const auto sample = trajectory.Evaluate(maximum);
    const auto bounds = trajectory.ConservativeBounds();
    if (!sample.position.isOnes() || !sample.velocity.isZero(0.) || !sample.acceleration.isZero(0.) ||
        !sample.jerk.isZero(0.) || !bounds.minimum_position.isOnes() || !bounds.maximum_position.isOnes() ||
        !bounds.maximum_absolute_velocity.isZero(0.) || !bounds.maximum_absolute_acceleration.isZero(0.) ||
        !bounds.maximum_absolute_jerk.isZero(0.))
        throw std::runtime_error("inactive harmonics changed a constant trajectory or its bounds");
}

void TestWideAmplitudeBoundaryRoundsOutward() {
    // Exact squared cross-products show that the amplitude is above this
    // double, although the old wide sqrt rounded exactly onto its boundary.
    const double a = 0x1.4b4f3c8061228p-1, b = 0x1.eb4b90d39696fp-1;
    const double inward = 0x1.2848b75fbc445p+0;
    const robot_identification::HarmonicTrajectory trajectory(
        Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, a), Eigen::MatrixXd::Constant(1, 1, b),
        1. / (2. * std::acos(-1.)));
    const auto bounds = trajectory.ConservativeBounds();
    if (bounds.maximum_position[0] <= inward || bounds.minimum_position[0] >= -inward)
        throw std::runtime_error("a wide amplitude rounded inward onto a double boundary");
    const robot_identification::ExcitationLimits limits{
        Eigen::VectorXd::Constant(1, -2.), Eigen::VectorXd::Constant(1, inward), Eigen::VectorXd::Constant(1, 100.),
        Eigen::VectorXd::Constant(1, 100.), Eigen::VectorXd::Constant(1, 100.)};
    const auto validation = robot_identification::ValidateExcitationTrajectory(trajectory, limits, 1., 1.);
    if (validation.passed || validation.reason != "conservative position bound exceeded")
        throw std::runtime_error("an inward wide amplitude admitted an insufficient position limit");
}

void TestWideDerivativeProductRoundsOutward() {
    const double a = 0x1.6417717c01db1p-1, hz = 0x1.bd482fd82cb5cp-4;
    const double inward = 0x1.e675a52e3f7adp-2;
    const robot_identification::HarmonicTrajectory trajectory(
        Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, a), Eigen::MatrixXd::Zero(1, 1), hz);
    if (trajectory.ConservativeBounds().maximum_absolute_velocity[0] <= inward)
        throw std::runtime_error("a wide derivative product rounded inward onto a double boundary");
    const robot_identification::ExcitationLimits limits{
        Eigen::VectorXd::Constant(1, -2.), Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, inward),
        Eigen::VectorXd::Constant(1, 100.), Eigen::VectorXd::Constant(1, 100.)};
    const auto validation = robot_identification::ValidateExcitationTrajectory(trajectory, limits, 1., 1.);
    if (validation.passed || validation.reason != "conservative velocity bound exceeded")
        throw std::runtime_error("an inward wide product admitted an insufficient velocity limit");
}

void TestReusableSamplingOutputs() {
    const double hz = 1. / (2. * std::acos(-1.));
    const Eigen::Vector2d offset(.25, -.5), a(.125, -.25), b(-.5, .375);
    const robot_identification::HarmonicTrajectory trajectory(offset, a, b, hz);
    robot_identification::ExcitationSample sample;
    sample.position = Eigen::VectorXd::Constant(7, std::numeric_limits<double>::quiet_NaN());
    for (const double time : {0., .1, .5, 1.}) {
        trajectory.EvaluateInto(time, sample);
        const std::array<const Eigen::VectorXd*, 4> fields{&sample.position, &sample.velocity, &sample.acceleration,
                                                           &sample.jerk};
        for (const auto field : fields)
            if (field->size() != 2 || !field->allFinite())
                throw std::runtime_error("reusable sampling did not replace or resize every output");
        for (Eigen::Index joint = 0; joint < 2; ++joint) {
            const long double sine = std::sin(static_cast<long double>(time));
            const long double cosine = std::cos(static_cast<long double>(time));
            RequireClose(sample.position[joint], offset[joint] + a[joint] * sine + b[joint] * cosine);
            RequireClose(sample.velocity[joint], a[joint] * cosine - b[joint] * sine);
            RequireClose(sample.acceleration[joint], -a[joint] * sine - b[joint] * cosine);
            RequireClose(sample.jerk[joint], -a[joint] * cosine + b[joint] * sine);
        }
    }
    const robot_identification::HarmonicTrajectory smaller(Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
                                                           Eigen::MatrixXd::Zero(1, 1), hz);
    smaller.EvaluateInto(0., sample);
    if (sample.position.size() != 1 || sample.position[0] != 0. || sample.velocity[0] != 1. ||
        sample.acceleration[0] != 0. || sample.jerk[0] != -1.)
        throw std::runtime_error("reusing outputs across trajectory dimensions retained stale values");
    bool rejected = false;
    try {
        trajectory.EvaluateInto(std::numeric_limits<double>::infinity(), sample);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("reusable sampling accepted a non-finite time");
}

void TestExactDyadicMultiHarmonicBounds() {
    const double hz = 1. / (2. * std::acos(-1.));
    const Eigen::Vector2d offset(.25, -.5);
    const robot_identification::HarmonicTrajectory trajectory(offset, Eigen::MatrixXd::Constant(2, 20, 3. / 1024.),
                                                              Eigen::MatrixXd::Constant(2, 20, 4. / 1024.), hz);
    const auto bounds = trajectory.ConservativeBounds();
    const double amplitude = 5. / 1024.;
    for (Eigen::Index joint = 0; joint < 2; ++joint)
        if (bounds.minimum_position[joint] != offset[joint] - 20. * amplitude ||
            bounds.maximum_position[joint] != offset[joint] + 20. * amplitude ||
            bounds.maximum_absolute_velocity[joint] != 210. * amplitude ||
            bounds.maximum_absolute_acceleration[joint] != 2870. * amplitude ||
            bounds.maximum_absolute_jerk[joint] != 44100. * amplitude)
            throw std::runtime_error("exact dyadic multi-harmonic bounds were enlarged");
}

void TestTinyCoefficientWithinAmplitude() {
    const robot_identification::HarmonicTrajectory trajectory(
        Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Ones(1, 1),
        Eigen::MatrixXd::Constant(1, 1, std::numeric_limits<double>::denorm_min()), 1. / (2. * std::acos(-1.)));
    const auto bounds = trajectory.ConservativeBounds();
    if (bounds.minimum_position[0] >= -1. || bounds.maximum_position[0] <= 1. ||
        bounds.maximum_absolute_velocity[0] <= 1. || bounds.maximum_absolute_acceleration[0] <= 1. ||
        bounds.maximum_absolute_jerk[0] <= 1.)
        throw std::runtime_error("a tiny coefficient vanished from the amplitude before outward conversion");
}
}  // namespace

int main() {
    int failed = 0;
    for (const auto test :
         {TestUnderflowingTrigonometricProducts, TestTinyDurationStillIncludesEndpoint,
          TestInactiveHarmonicsAtLargeTimes, TestSubnormalEnvelopeDoesNotAdmitExcessVelocity,
          TestRoundedStepCountStillIncludesEndpoint, TestNormalEnvelopeRoundsOutward, TestSubnormalDerivativeSums,
          TestSmallHarmonicsSurviveEnvelopeAccumulation, TestWideAmplitudeBoundaryRoundsOutward,
          TestWideDerivativeProductRoundsOutward, TestReusableSamplingOutputs, TestExactDyadicMultiHarmonicBounds,
          TestTinyCoefficientWithinAmplitude}) {
        try {
            test();
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ++failed;
        }
    }
    return failed == 0 ? 0 : 1;
}
