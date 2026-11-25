#include "raw_preprocessing.h"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "robust_scale.h"

namespace arm_identification {

robot_identification::Dataset EstimateLocalCubic(const robot_identification::RawDataset& raw, std::size_t window) {
    using Eigen::Index;
    const Index rows = raw.time_seconds.size();
    if (window < 5 || window % 2 == 0 || window > static_cast<std::size_t>(rows - 2))
        throw std::invalid_argument("local cubic requires an odd window of at least five and three interior samples");

    const Index radius = static_cast<Index>(window / 2);
    Eigen::Matrix<double, Eigen::Dynamic, 4> polynomial(window, 4);
    polynomial.col(0).setOnes();
    robot_identification::Dataset output{raw.joint_names, {}};
    output.samples.reserve(rows - 2 * radius);
    for (Index center = radius; center < rows - radius; ++center) {
        const Index first = center - radius;
        const double time = raw.time_seconds[center];
        const double scale = std::max(time - raw.time_seconds[first], raw.time_seconds[center + radius] - time);
        if (!std::isfinite(scale) || scale <= 0.)
            throw std::invalid_argument("local cubic requires a finite positive time span");

        // Center the position and scale time before building powers of the offsets.
        polynomial.col(1) = (raw.time_seconds.segment(first, window).array() - time) / scale;
        polynomial.col(2) = polynomial.col(1).array().square();
        polynomial.col(3) = polynomial.col(1).array() * polynomial.col(2).array();
        const auto qr = polynomial.colPivHouseholderQr();
        if (qr.rank() != 4) throw std::invalid_argument("local cubic has a rank-deficient timestamp window");
        const Eigen::MatrixXd displacement =
            raw.position.middleRows(first, window).rowwise() - raw.position.row(center);
        const Eigen::Matrix<double, 4, Eigen::Dynamic> coefficients = qr.solve(displacement);
        output.samples.push_back({time, (raw.position.row(center) + coefficients.row(0)).transpose(),
                                  coefficients.row(1).transpose() / scale,
                                  (2. / scale) * coefficients.row(2).transpose() / scale,
                                  raw.torque.row(center).transpose()});
    }
    return output;
}

namespace {
const double kCycleAngle = 2. * std::acos(-1.);

Eigen::MatrixXd FourierBasis(
    const Eigen::VectorXd& times, double origin, double period, Eigen::Index modes, double delay) {
    Eigen::MatrixXd basis(times.size(), 2 * modes + 1);
    basis.col(0).setOnes();
    for (Eigen::Index row = 0; row < times.size(); ++row) {
        const double angle = kCycleAngle * ((times[row] - origin - delay) / period);
        for (Eigen::Index mode = 1; mode <= modes; ++mode) {
            basis(row, 2 * mode - 1) = std::sin(mode * angle);
            basis(row, 2 * mode) = std::cos(mode * angle);
        }
    }
    return basis;
}
}  // namespace

PeriodicFourierModel FitPeriodicFourier(const robot_identification::RawDataset& raw,
                                        std::size_t harmonics,
                                        bool robust) {
    constexpr std::size_t maximum_harmonics = 32;
    const auto rows = raw.time_seconds.size();
    if (harmonics == 0 || harmonics > maximum_harmonics || static_cast<std::size_t>(rows) < 2 * harmonics + 2)
        throw std::invalid_argument("periodic Fourier requires 1 to 32 harmonics and at least 2h+2 samples");

    const double origin = raw.time_seconds[0];
    const double period = raw.time_seconds[rows - 1] - origin;
    const double fundamental = kCycleAngle / period;
    if (!std::isfinite(period) || !std::isfinite(fundamental) || fundamental <= 0.)
        throw std::invalid_argument("periodic Fourier requires a representable sampling period");
    auto basis = FourierBasis(raw.time_seconds, origin, period, static_cast<Eigen::Index>(harmonics), 0.);
    const auto qr = basis.colPivHouseholderQr();
    if (qr.rank() != basis.cols())
        throw std::invalid_argument("periodic Fourier harmonics are rank deficient at these timestamps");
    Eigen::RowVectorXd offset = raw.position.row(0);
    const Eigen::MatrixXd centered = raw.position.rowwise() - offset;
    Eigen::MatrixXd coefficients = qr.solve(centered);
    const auto dof = raw.position.cols();
    Eigen::VectorXd huber_delta(robust ? dof : 0);
    if (robust) {
        robot_identification::RegressionProblem position_fit{
            basis, {}, Eigen::VectorXd::Ones(rows), static_cast<std::size_t>(rows), 1};
        for (Eigen::Index joint = 0; joint < dof; ++joint) {
            position_fit.observation = centered.col(joint);
            const Eigen::VectorXd residual = position_fit.observation - basis * coefficients.col(joint);
            huber_delta[joint] = HuberDeltaFromResiduals(residual, raw.position.col(joint).cwiseAbs().maxCoeff());
            robot_identification::SolverOptions options;
            options.huber_delta = huber_delta[joint];
            coefficients.col(joint) = robot_identification::Solve(position_fit, options).parameters;
        }
    }
    return {origin, period, std::move(offset), std::move(basis), std::move(coefficients), std::move(huber_delta)};
}

robot_identification::Dataset ReconstructPeriodicFourier(const robot_identification::RawDataset& raw,
                                                         const PeriodicFourierModel& model,
                                                         double torque_delay_seconds) {
    if (!std::isfinite(torque_delay_seconds) || std::abs(torque_delay_seconds) >= .5 * model.period_seconds)
        throw std::invalid_argument("torque delay must be smaller than half a capture period");
    const auto modes = (model.coefficients.rows() - 1) / 2;
    const auto dof = raw.position.cols();
    const double fundamental = kCycleAngle / model.period_seconds;
    // A torque sample delayed by d at time t describes the state at t - d.
    const Eigen::MatrixXd shifted =
        torque_delay_seconds == 0.
            ? Eigen::MatrixXd{}
            : FourierBasis(raw.time_seconds, model.origin_seconds, model.period_seconds, modes, torque_delay_seconds);
    const auto& basis = torque_delay_seconds == 0. ? model.sampled_basis : shifted;
    robot_identification::Dataset output{raw.joint_names, {}};
    output.samples.reserve(raw.time_seconds.size());
    for (Eigen::Index row = 0; row < raw.time_seconds.size(); ++row) {
        Eigen::RowVectorXd velocity = Eigen::RowVectorXd::Zero(dof);
        Eigen::RowVectorXd acceleration = Eigen::RowVectorXd::Zero(dof);
        for (Eigen::Index mode = 1; mode <= modes; ++mode) {
            const double omega = fundamental * mode;
            const double sine = basis(row, 2 * mode - 1);
            const double cosine = basis(row, 2 * mode);
            const auto sine_coefficient = model.coefficients.row(2 * mode - 1);
            const auto cosine_coefficient = model.coefficients.row(2 * mode);
            velocity += omega * (cosine * sine_coefficient - sine * cosine_coefficient);
            acceleration -= (omega * omega) * (sine * sine_coefficient + cosine * cosine_coefficient);
        }
        output.samples.push_back({raw.time_seconds[row],
                                  (model.position_offset + basis.row(row) * model.coefficients).transpose(),
                                  velocity.transpose(), acceleration.transpose(), raw.torque.row(row).transpose()});
    }
    return output;
}

}  // namespace arm_identification
