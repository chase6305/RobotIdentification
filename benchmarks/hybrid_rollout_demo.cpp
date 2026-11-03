// Synthetic scalar prototype: calibrated PD + learned bounded residual,
// positive inertia/damping, and semi-implicit trajectory matching.
#include <robot_identification/identification.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr double kPeriod = 0.005, kP = 12., kD = .18, kLoad = .4;
using Parameters = Eigen::Vector4d;  // J [kg m^2], D [Nm s/rad], residual weights [Nm].
struct State {
    double q, v;
};
struct Sequence {
    std::vector<double> command;
    std::vector<State> observed;
};
struct History {
    std::array<double, 2> error{}, velocity{};
    Eigen::Vector2d Observe(double e, double v) {
        // Inputs are normalized by 1 rad and 1 rad/s; hidden weights are fixed.
        const Eigen::Vector2d features(
            std::tanh(3. * e + .4 * error[0] - .2 * error[1] + .08 * v - .04 * velocity[0] + .02 * velocity[1]),
            std::tanh(2. * (e - error[1]) + .12 * (v - velocity[1])));
        error = {e, error[0]};
        velocity = {v, velocity[0]};
        return features;
    }
};

std::vector<State> Rollout(const Parameters& p, const std::vector<double>& command, State state, int substeps) {
    std::vector<State> states{state};
    states.reserve(command.size() + 1);
    History history;
    for (double target : command) {
        const double error = target - state.q;
        const double force = kP * error - kD * state.v + p.tail<2>().dot(history.Observe(error, state.v)) - kLoad;
        if (substeps == 0) {
            // Independent exact solution with constant torque held for one controller period.
            const double factor = -std::expm1(-p[1] * kPeriod / p[0]);
            const double integral = p[0] * factor / p[1];
            state.q += integral * state.v + force / p[1] * (kPeriod - integral);
            state.v = (1. - factor) * state.v + force * factor / p[1];
        } else {
            const double h = kPeriod / substeps;
            for (int step = 0; step < substeps; ++step) {
                state.v = (p[0] * state.v + h * force) / (p[0] + h * p[1]);
                state.q += h * state.v;
            }
        }
        if (!std::isfinite(state.q) || !std::isfinite(state.v)) throw std::runtime_error("non-finite rollout");
        states.push_back(state);
    }
    return states;
}

Sequence MakeSequence(const Parameters& truth, int variant, bool noisy) {
    Sequence data;
    for (int i = 0; i < 1200; ++i) {
        const double t = i * kPeriod;
        data.command.push_back(.18 * std::sin((1.7 + .7 * variant) * t + .2 * variant) +
                               .12 * std::sin((5.3 + 1.1 * variant) * t));
    }
    data.observed = Rollout(truth, data.command, {.1 * variant, 0.}, 0);
    if (noisy)
        for (std::size_t i = 0; i < data.observed.size(); ++i) {
            data.observed[i].q += .0002 * std::sin(1.618 * i + variant);
            data.observed[i].v += .003 * std::cos(2.17 * i + variant);
        }
    return data;
}

bool Admissible(const Parameters& p) {
    return p.allFinite() && p[0] >= .02 && p[0] <= .3 && p[1] >= 0. && p[1] <= 1. &&
           p.tail<2>().cwiseAbs().maxCoeff() <= 2.;
}

double Loss(const Parameters& p, const std::vector<Sequence>& sequences, int substeps) {
    if (!Admissible(p)) return std::numeric_limits<double>::infinity();
    double error = 0.;
    std::size_t count = 0;
    for (const auto& data : sequences) {
        const auto states = Rollout(p, data.command, data.observed.front(), substeps);
        for (std::size_t i = 1; i < states.size(); ++i) {
            // Normalize q by 1 rad and v by 10 rad/s before combining errors.
            error +=
                std::pow(states[i].q - data.observed[i].q, 2) + .01 * std::pow(states[i].v - data.observed[i].v, 2);
            ++count;
        }
    }
    return error / count;
}

Parameters Initialize(const std::vector<Sequence>& sequences) {
    std::size_t rows = 0;
    for (const auto& data : sequences) rows += data.command.size();
    robot_identification::RegressionProblem problem;
    problem.sample_count = rows;
    problem.dof = 1;
    problem.regressor.resize(rows, 4);
    problem.observation.resize(rows);
    problem.row_weights = Eigen::VectorXd::Ones(rows);
    Eigen::Index row = 0;
    for (const auto& data : sequences) {
        History history;
        for (std::size_t i = 0; i < data.command.size(); ++i, ++row) {
            const auto& state = data.observed[i];
            const auto& next = data.observed[i + 1];
            const double error = data.command[i] - state.q;
            const auto features = history.Observe(error, state.v);
            // Integrated balance, in angular-momentum units; no q double differentiation.
            problem.regressor.row(row) << next.v - state.v, kPeriod * next.v, -kPeriod * features[0],
                -kPeriod * features[1];
            problem.observation[row] = kPeriod * (kP * error - kD * state.v - kLoad);
        }
    }
    const auto fit = robot_identification::Solve(problem);
    if (fit.rank != 4) throw std::runtime_error("insufficient excitation");
    std::cout << "linear_rank=" << fit.rank << " condition=" << fit.condition_number
              << " unconstrained=" << fit.parameters.transpose() << '\n';
    return fit.parameters.cwiseMax(Parameters(.02, 0., -2., -2.)).cwiseMin(Parameters(.3, 1., 2., 2.));
}

Parameters Refine(Parameters p, const std::vector<Sequence>& training, int columns = 4) {
    Parameters step(.005, .01, .05, .05);
    double best = Loss(p, training, 4);
    for (int iteration = 0; iteration < 400 && step.maxCoeff() > 1e-7; ++iteration) {
        bool improved = false;
        for (int column = 0; column < columns; ++column)
            for (double sign : {-1., 1.}) {
                Parameters trial = p;
                trial[column] += sign * step[column];
                const double value = Loss(trial, training, 4);
                if (value < best) {
                    p = trial;
                    best = value;
                    improved = true;
                }
            }
        if (!improved) step *= .5;
    }
    return p;
}
}  // namespace

int main() {
    const Parameters truth(.09, .11, .7, -.25);
    const std::vector<Sequence> training{MakeSequence(truth, 0, true), MakeSequence(truth, 1, true)};
    const std::vector<Sequence> holdout{MakeSequence(truth, 3, false)};
    const Parameters baseline = Refine(Parameters(.13, .2, 0., 0.), training, 2);
    const Parameters initial = Initialize(training), refined = Refine(initial, training);
    const double before = Loss(initial, training, 4), after = Loss(refined, training, 4);
    const double holdout_before = Loss(baseline, holdout, 4), holdout_after = Loss(refined, holdout, 4);
    const double coarse = Loss(truth, holdout, 1), fine = Loss(truth, holdout, 4);
    std::cout << "truth=" << truth.transpose() << "\nphysics_only=" << baseline.transpose()
              << "\ninitial=" << initial.transpose() << "\nrefined=" << refined.transpose()
              << "\ntraining_loss=" << before << " -> " << after << "\nholdout_loss=" << holdout_before << " -> "
              << holdout_after << "\nintegrator_loss=" << coarse << " -> " << fine << '\n';
    if (!(after < before && holdout_after < holdout_before && fine < .2 * coarse))
        throw std::runtime_error("hybrid trajectory validation failed");
}
