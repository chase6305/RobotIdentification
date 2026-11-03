// Synthetic scalar prototype: calibrated PD + learned bounded residual,
// positive inertia, nonnegative damping, and semi-implicit trajectory matching.
#include <robot_identification/identification.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr double kControlPeriodSeconds = 0.005;
constexpr double kPdPositionGain = 12.;  // Nm/rad: calibrated position feedback.
constexpr double kPdVelocityGain = .18;  // Nm s/rad: calibrated velocity feedback, separate from plant damping.
constexpr double kLoadTorque = .4;       // Nm: known opposing load.
constexpr int kInertiaIndex = 0, kViscousDampingIndex = 1;
constexpr int kPhysicalParameterCount = 2, kResidualFeatureCount = 2;
constexpr int kParameterCount = kPhysicalParameterCount + kResidualFeatureCount;
constexpr int kExactReferenceSubsteps = 0, kFittingSubsteps = 4;
using ModelParameters = Eigen::Vector4d;  // Inertia [kg m^2], viscous damping [Nm s/rad], residual weights [Nm].

struct State {
    double position;  // rad
    double velocity;  // rad/s
};
struct Sequence {
    std::vector<double> target_positions;  // rad, one command per control period.
    std::vector<State> observed_states;
};
struct ResidualFeatureHistory {
    std::array<double, 2> position_errors{}, velocities{};
    Eigen::Vector2d Observe(double position_error, double velocity) {
        // Inputs are normalized by 1 rad and 1 rad/s; hidden weights are fixed.
        const Eigen::Vector2d features(
            std::tanh(3. * position_error + .4 * position_errors[0] - .2 * position_errors[1] + .08 * velocity -
                      .04 * velocities[0] + .02 * velocities[1]),
            std::tanh(2. * (position_error - position_errors[1]) + .12 * (velocity - velocities[1])));
        // Shift only after computing both features from the same two previous periods.
        position_errors = {position_error, position_errors[0]};
        velocities = {velocity, velocities[0]};
        return features;
    }
};

std::vector<State> Rollout(const ModelParameters& parameters,
                           const std::vector<double>& target_positions,
                           State state,
                           int substeps) {
    const double inertia = parameters[kInertiaIndex];
    const double viscous_damping = parameters[kViscousDampingIndex];
    std::vector<State> predicted_states{state};
    predicted_states.reserve(target_positions.size() + 1);
    ResidualFeatureHistory history;
    for (double target_position : target_positions) {
        const double position_error = target_position - state.position;
        // Sample PD feedback and residual once per control period; hold this torque across all substeps.
        // Plant viscous damping is applied by the integrator, separately from the PD velocity gain.
        const double held_torque =
            kPdPositionGain * position_error - kPdVelocityGain * state.velocity +
            parameters.tail<kResidualFeatureCount>().dot(history.Observe(position_error, state.velocity)) - kLoadTorque;
        if (substeps == kExactReferenceSubsteps) {
            // Independent exact solution for the positive-damping synthetic reference, with held input torque.
            const double decay_fraction = -std::expm1(-viscous_damping * kControlPeriodSeconds / inertia);
            const double decay_integral_seconds = inertia * decay_fraction / viscous_damping;
            // Integrate position from the old velocity before updating velocity.
            state.position += decay_integral_seconds * state.velocity +
                              held_torque / viscous_damping * (kControlPeriodSeconds - decay_integral_seconds);
            state.velocity = (1. - decay_fraction) * state.velocity + held_torque * decay_fraction / viscous_damping;
        } else {
            const double substep_seconds = kControlPeriodSeconds / substeps;
            for (int step = 0; step < substeps; ++step) {
                // Backward-Euler velocity: J*(v_next-v)/h + B*v_next = held_torque.
                // Advance position with the updated velocity (semi-implicit position step).
                state.velocity = (inertia * state.velocity + substep_seconds * held_torque) /
                                 (inertia + substep_seconds * viscous_damping);
                state.position += substep_seconds * state.velocity;
            }
        }
        if (!std::isfinite(state.position) || !std::isfinite(state.velocity))
            throw std::runtime_error("non-finite rollout");
        predicted_states.push_back(state);
    }
    return predicted_states;
}

Sequence MakeSequence(const ModelParameters& true_parameters, int trajectory_variant, bool add_measurement_noise) {
    Sequence sequence;
    for (int sample_index = 0; sample_index < 1200; ++sample_index) {
        const double time_seconds = sample_index * kControlPeriodSeconds;
        sequence.target_positions.push_back(
            .18 * std::sin((1.7 + .7 * trajectory_variant) * time_seconds + .2 * trajectory_variant) +
            .12 * std::sin((5.3 + 1.1 * trajectory_variant) * time_seconds));
    }
    sequence.observed_states =
        Rollout(true_parameters, sequence.target_positions, {.1 * trajectory_variant, 0.}, kExactReferenceSubsteps);
    if (add_measurement_noise)
        for (std::size_t sample_index = 0; sample_index < sequence.observed_states.size(); ++sample_index) {
            sequence.observed_states[sample_index].position +=
                .0002 * std::sin(1.618 * sample_index + trajectory_variant);
            sequence.observed_states[sample_index].velocity +=
                .003 * std::cos(2.17 * sample_index + trajectory_variant);
        }
    return sequence;
}

bool Admissible(const ModelParameters& parameters) {
    return parameters.allFinite() && parameters[kInertiaIndex] >= .02 && parameters[kInertiaIndex] <= .3 &&
           parameters[kViscousDampingIndex] >= 0. && parameters[kViscousDampingIndex] <= 1. &&
           parameters.tail<kResidualFeatureCount>().cwiseAbs().maxCoeff() <= 2.;
}

double TrajectoryLoss(const ModelParameters& parameters, const std::vector<Sequence>& sequences, int substeps) {
    if (!Admissible(parameters)) return std::numeric_limits<double>::infinity();
    double squared_error = 0.;
    std::size_t sample_count = 0;
    for (const auto& sequence : sequences) {
        const auto predicted_states =
            Rollout(parameters, sequence.target_positions, sequence.observed_states.front(), substeps);
        for (std::size_t sample_index = 1; sample_index < predicted_states.size(); ++sample_index) {
            const auto& predicted = predicted_states[sample_index];
            const auto& observed = sequence.observed_states[sample_index];
            // Normalize position by 1 rad and velocity by 10 rad/s before combining errors.
            squared_error += std::pow(predicted.position - observed.position, 2) +
                             .01 * std::pow(predicted.velocity - observed.velocity, 2);
            ++sample_count;
        }
    }
    return squared_error / sample_count;
}

ModelParameters InitializeFromMomentumBalance(const std::vector<Sequence>& sequences) {
    std::size_t row_count = 0;
    for (const auto& sequence : sequences) row_count += sequence.target_positions.size();
    robot_identification::RegressionProblem problem;
    problem.sample_count = row_count;
    problem.dof = 1;
    problem.regressor.resize(row_count, kParameterCount);
    problem.observation.resize(row_count);
    problem.row_weights = Eigen::VectorXd::Ones(row_count);
    Eigen::Index row_index = 0;
    for (const auto& sequence : sequences) {
        ResidualFeatureHistory history;
        for (std::size_t sample_index = 0; sample_index < sequence.target_positions.size();
             ++sample_index, ++row_index) {
            const auto& state = sequence.observed_states[sample_index];
            const auto& next_state = sequence.observed_states[sample_index + 1];
            const double position_error = sequence.target_positions[sample_index] - state.position;
            const auto features = history.Observe(position_error, state.velocity);
            // One-period backward-Euler momentum balance [Nm s]; no second derivative of position.
            // Damping uses the observed endpoint velocity, so this initializer approximates the exact reference.
            // Columns follow [inertia, viscous damping, residual weight 1, residual weight 2].
            problem.regressor.row(row_index) << next_state.velocity - state.velocity,
                kControlPeriodSeconds * next_state.velocity, -kControlPeriodSeconds * features[0],
                -kControlPeriodSeconds * features[1];
            problem.observation[row_index] = kControlPeriodSeconds * (kPdPositionGain * position_error -
                                                                      kPdVelocityGain * state.velocity - kLoadTorque);
        }
    }
    const auto fit = robot_identification::Solve(problem);
    if (fit.rank != kParameterCount) throw std::runtime_error("insufficient excitation");
    std::cout << "linear_rank=" << fit.rank << " condition=" << fit.condition_number
              << " unconstrained=" << fit.parameters.transpose() << '\n';
    return fit.parameters.cwiseMax(ModelParameters(.02, 0., -2., -2.)).cwiseMin(ModelParameters(.3, 1., 2., 2.));
}

ModelParameters RefineWithCoordinateSearch(ModelParameters parameters,
                                           const std::vector<Sequence>& training_sequences,
                                           int active_parameter_count = kParameterCount) {
    ModelParameters parameter_steps(.005, .01, .05, .05);
    double best_loss = TrajectoryLoss(parameters, training_sequences, kFittingSubsteps);
    for (int iteration = 0; iteration < 400 && parameter_steps.maxCoeff() > 1e-7; ++iteration) {
        bool improved = false;
        for (int parameter_index = 0; parameter_index < active_parameter_count; ++parameter_index)
            for (double direction : {-1., 1.}) {
                ModelParameters trial_parameters = parameters;
                trial_parameters[parameter_index] += direction * parameter_steps[parameter_index];
                const double trial_loss = TrajectoryLoss(trial_parameters, training_sequences, kFittingSubsteps);
                if (trial_loss < best_loss) {
                    parameters = trial_parameters;
                    best_loss = trial_loss;
                    improved = true;
                }
            }
        if (!improved) parameter_steps *= .5;
    }
    return parameters;
}
}  // namespace

int main() {
    const ModelParameters true_parameters(.09, .11, .7, -.25);
    const std::vector<Sequence> training_sequences{MakeSequence(true_parameters, 0, true),
                                                   MakeSequence(true_parameters, 1, true)};
    const std::vector<Sequence> holdout_sequences{MakeSequence(true_parameters, 3, false)};
    const ModelParameters physics_only_parameters =
        RefineWithCoordinateSearch(ModelParameters(.13, .2, 0., 0.), training_sequences, kPhysicalParameterCount);
    const ModelParameters initial_parameters = InitializeFromMomentumBalance(training_sequences);
    const ModelParameters refined_parameters = RefineWithCoordinateSearch(initial_parameters, training_sequences);
    const double training_loss_before = TrajectoryLoss(initial_parameters, training_sequences, kFittingSubsteps);
    const double training_loss_after = TrajectoryLoss(refined_parameters, training_sequences, kFittingSubsteps);
    const double holdout_loss_before = TrajectoryLoss(physics_only_parameters, holdout_sequences, kFittingSubsteps);
    const double holdout_loss_after = TrajectoryLoss(refined_parameters, holdout_sequences, kFittingSubsteps);
    const double coarse_integrator_loss = TrajectoryLoss(true_parameters, holdout_sequences, 1);
    const double fine_integrator_loss = TrajectoryLoss(true_parameters, holdout_sequences, kFittingSubsteps);
    std::cout << "truth=" << true_parameters.transpose() << "\nphysics_only=" << physics_only_parameters.transpose()
              << "\ninitial=" << initial_parameters.transpose() << "\nrefined=" << refined_parameters.transpose()
              << "\ntraining_loss=" << training_loss_before << " -> " << training_loss_after
              << "\nholdout_loss=" << holdout_loss_before << " -> " << holdout_loss_after
              << "\nintegrator_loss=" << coarse_integrator_loss << " -> " << fine_integrator_loss << '\n';
    if (!(training_loss_after < training_loss_before && holdout_loss_after < holdout_loss_before &&
          fine_integrator_loss < .2 * coarse_integrator_loss))
        throw std::runtime_error("hybrid trajectory validation failed");
}
