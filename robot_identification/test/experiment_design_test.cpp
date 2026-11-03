#include <robot_identification/experiment_design.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace robot_identification;
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Function>
void Reject(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "invalid excitation design was accepted");
}

class ActuatorBackend final : public RegressorBackend {
public:
    explicit ActuatorBackend(bool duplicate = false, double scale = 1.) : duplicate_(duplicate), scale_(scale) {}
    std::size_t parameter_count() const noexcept override { return 2; }
    std::vector<std::string> parameter_names() const override { return {"inertia", "viscous"}; }
    Eigen::MatrixXd Compute(const Sample& sample) const override {
        Eigen::MatrixXd block(1, 2);
        block << sample.acceleration[0], duplicate_ ? 2. * sample.acceleration[0] : sample.velocity[0];
        return block * scale_;
    }

private:
    bool duplicate_;
    double scale_;
};

HarmonicDesign Seed(double amplitude = .2) {
    return {Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Constant(1, 1, amplitude), Eigen::MatrixXd::Zero(1, 1),
            1. / (2. * std::acos(-1.))};
}
ExcitationDesignOptions Options() {
    ExcitationDesignOptions options;
    options.duration_seconds = 2. * std::acos(-1.);
    options.sample_count = 65;
    return options;
}
ExcitationLimits Limits() {
    return {Eigen::VectorXd::Constant(1, -2.), Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.),
            Eigen::VectorXd::Constant(1, 2.), Eigen::VectorXd::Constant(1, 2.)};
}

void TestAnalyticInformation() {
    ActuatorBackend backend;
    auto options = Options();
    const auto quality = EvaluateExcitationDesign(Seed().trajectory(), backend, options);
    const double expected = .5 * (std::log(.04 * 32. / 65.) + std::log(.04 * 33. / 65.));
    Require(quality.fully_identifiable() && quality.rank == 2, "analytic excitation rank is wrong");
    Require(std::abs(quality.mean_log_information - expected) < 1e-12,
            "information score differs from analytic Gram eigenvalues");
    Require(std::abs(quality.condition_number - std::sqrt(33. / 32.)) < 1e-12,
            "analytic excitation condition is wrong");
    const auto doubled = EvaluateExcitationDesign(Seed(.4).trajectory(), backend, options);
    Require(std::abs(doubled.mean_log_information - quality.mean_log_information - 2. * std::log(2.)) < 1e-12,
            "candidate normalization erased excitation amplitude");
    options.parameter_scales = Eigen::Vector2d(2., 3.);
    options.torque_noise_standard_deviation = Eigen::VectorXd::Constant(1, 4.);
    const auto scaled = EvaluateExcitationDesign(Seed().trajectory(), backend, options);
    Require(std::abs(scaled.mean_log_information - expected - std::log(6. / 16.)) < 1e-12,
            "fixed parameter/noise scales changed the wrong information units");
    Require(std::abs(scaled.condition_number - 1.5 * std::sqrt(33. / 32.)) < 1e-12,
            "parameter scales changed the wrong singular directions");
    const auto deficient = EvaluateExcitationDesign(Seed().trajectory(), ActuatorBackend(true), Options());
    Require(deficient.rank == 1 && !deficient.fully_identifiable() && std::isinf(deficient.condition_number),
            "dependent parameters were reported identifiable");
    if (std::numeric_limits<long double>::max_exponent > 4 * std::numeric_limits<double>::max_exponent) {
        options.parameter_scales = Eigen::Vector2d::Constant(1e200);
        options.torque_noise_standard_deviation = Eigen::VectorXd::Constant(1, 1e-200);
        const auto extreme = EvaluateExcitationDesign(Seed().trajectory(), ActuatorBackend(false, 1e200), options);
        Require(extreme.fully_identifiable() &&
                    std::abs(extreme.mean_log_information - expected - 1200. * std::log(10.)) < 1e-10,
                "a representable log information score was lost to scaling overflow");
    }
}

void TestOptimizationAndBudget() {
    const ActuatorBackend backend;
    auto options = Options();
    options.max_evaluations = 250;
    const auto first = OptimizeExcitationDesign(Seed(.05), Limits(), backend, options);
    const auto second = OptimizeExcitationDesign(Seed(.05), Limits(), backend, options);
    Require(first.validation.passed && first.quality.fully_identifiable(), "optimizer returned an invalid design");
    Require(first.accepted_steps > 0 &&
                first.quality.mean_log_information > first.initial_quality.mean_log_information + 1.,
            "optimizer did not improve analytic information");
    Require(first.evaluations <= options.max_evaluations && first.feasible_evaluations <= first.evaluations,
            "optimizer exceeded its evaluation budget");
    Require(first.design.sin_coefficients == second.design.sin_coefficients &&
                first.design.cos_coefficients == second.design.cos_coefficients &&
                first.evaluations == second.evaluations,
            "pattern search is not deterministic");
    const auto bounds = first.design.trajectory().ConservativeBounds();
    Require(bounds.maximum_position[0] <= 2. && bounds.maximum_absolute_jerk[0] <= 2.,
            "optimized envelope exceeds limits");
    options.max_evaluations = 1;
    const auto limited = OptimizeExcitationDesign(Seed(), Limits(), backend, options);
    Require(limited.evaluations == 1 && limited.accepted_steps == 0 && !limited.converged &&
                limited.design.sin_coefficients == Seed().sin_coefficients,
            "exhausting the budget changed the seed or reported convergence");
    options.max_evaluations = 2;
    options.minimum_step_fraction = options.initial_step_fraction;
    const auto interrupted_poll = OptimizeExcitationDesign(Seed(), Limits(), ActuatorBackend(false, 0.), options);
    Require(interrupted_poll.evaluations == 2 && !interrupted_poll.converged &&
                interrupted_poll.final_step_fraction == options.initial_step_fraction,
            "an incomplete coordinate poll consumed the budget but reported convergence");
    const auto deficient = OptimizeExcitationDesign(Seed(), Limits(), ActuatorBackend(true), Options());
    Require(!deficient.quality.fully_identifiable(), "optimization concealed structural rank deficiency");
    const auto unexcited = OptimizeExcitationDesign(Seed(0.), Limits(), backend, Options());
    Require(unexcited.initial_quality.rank == 0 && unexcited.quality.fully_identifiable() &&
                unexcited.accepted_steps != 0 && std::isfinite(unexcited.quality.mean_log_information),
            "pattern search failed to recover excitation from a constant seed");
}

class ConstantBackend final : public RegressorBackend {
public:
    explicit ConstantBackend(double value) : value_(value) {}
    std::size_t parameter_count() const noexcept override { return 1; }
    std::vector<std::string> parameter_names() const override { return {"constant"}; }
    Eigen::MatrixXd Compute(const Sample&) const override { return Eigen::MatrixXd::Constant(1, 1, value_); }

private:
    double value_;
};

void TestUnitScaleRange() {
    for (const double value : {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(), 1e-150,
                               -3.7, 1e150, std::numeric_limits<double>::max()}) {
        const ConstantBackend backend(value);
        auto options = Options();
        const auto ordinary = EvaluateExcitationDesign(Seed().trajectory(), backend, options);
        const double expected = 2. * std::log(std::abs(value));
        Require(ordinary.rank == 1 && ordinary.condition_number == 1. &&
                    std::abs(ordinary.mean_log_information - expected) < 1e-12,
                "unit-scale information lost a subnormal or large finite regressor");
        options.parameter_scales = options.torque_noise_standard_deviation = Eigen::VectorXd::Ones(1);
        const auto explicit_ones = EvaluateExcitationDesign(Seed().trajectory(), backend, options);
        Require(explicit_ones.mean_log_information == ordinary.mean_log_information,
                "explicit unit scales changed the information score");
        // Keep the equivalent weighted intermediate representable even when
        // long double has the same exponent range as double.
        const double neutral_scale = std::abs(value) > 1. ? .5 : 2.;
        options.parameter_scales *= neutral_scale;
        options.torque_noise_standard_deviation *= neutral_scale;
        const auto wide = EvaluateExcitationDesign(Seed().trajectory(), backend, options);
        Require(std::abs(wide.mean_log_information - ordinary.mean_log_information) < 1e-12,
                "equivalent weighted scaling changed the analytic information score");
    }
}

void TestInvalidInputs() {
    const ActuatorBackend backend;
    Reject([&] { (void)OptimizeExcitationDesign(Seed(3.), Limits(), backend, Options()); });
    for (int variant = 0; variant < 7; ++variant) {
        auto options = Options();
        if (variant == 0) options.sample_count = 2;
        if (variant == 1) options.max_evaluations = 0;
        if (variant == 2) options.step_contraction = 1.;
        if (variant == 3) options.minimum_step_fraction = 0.;
        if (variant == 4) options.parameter_scales = Eigen::VectorXd::Ones(3);
        if (variant == 5) options.torque_noise_standard_deviation = Eigen::VectorXd::Zero(1);
        if (variant == 6) options.duration_seconds = std::numeric_limits<double>::infinity();
        Reject([&] { (void)OptimizeExcitationDesign(Seed(), Limits(), backend, options); });
    }
}

void TestExtremeFixedScales() {
    if (std::numeric_limits<long double>::max_exponent <= 4 * std::numeric_limits<double>::max_exponent) return;
    const std::array<double, 7> values{std::numeric_limits<double>::denorm_min(), 1e-300, 1e-150, 1., 1e150, 1e300,
                                       std::numeric_limits<double>::max()};
    auto options = Options();
    options.sample_count = 3;
    for (const double value : values)
        for (const double parameter_scale : values)
            for (const double noise : values) {
                options.parameter_scales = Eigen::VectorXd::Constant(1, parameter_scale);
                options.torque_noise_standard_deviation = Eigen::VectorXd::Constant(1, noise);
                const auto quality = EvaluateExcitationDesign(Seed().trajectory(), ConstantBackend(value), options);
                const long double expected = 2.L * (std::log(static_cast<long double>(value)) +
                                                    std::log(static_cast<long double>(parameter_scale)) -
                                                    std::log(static_cast<long double>(noise)));
                if (!quality.fully_identifiable() || quality.condition_number != 1. ||
                    std::abs(quality.mean_log_information - expected) > 1e-10L)
                    throw std::runtime_error("fixed scale ratios lost an extreme analytic information score");
            }
}
}  // namespace

int main() {
    try {
        TestAnalyticInformation();
        TestOptimizationAndBudget();
        TestUnitScaleRange();
        TestInvalidInputs();
        TestExtremeFixedScales();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
