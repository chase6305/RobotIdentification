// Synthetic actuator composition timing. Fixture construction and full output
// verification are outside the measured API batch; no hardware timing claim.
#include <robot_identification/identification.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace robot_identification;

class StoredBackend final : public RegressorBackend {
public:
    StoredBackend(Eigen::Index dof, Eigen::Index columns) : block_(dof, columns) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            names_.push_back("base/" + std::to_string(column));
            for (Eigen::Index row = 0; row < dof; ++row)
                block_(row, column) = (row + column) % 17 == 0 ? -0. : double((row + 3 * column) % 31 - 15) / 16.;
        }
    }
    std::size_t parameter_count() const noexcept override { return static_cast<std::size_t>(block_.cols()); }
    std::vector<std::string> parameter_names() const override { return names_; }
    Eigen::MatrixXd Compute(const Sample&) const override { return block_; }
    const Eigen::MatrixXd& block() const { return block_; }

private:
    Eigen::MatrixXd block_;
    std::vector<std::string> names_;
};

std::uint64_t Bits(double value) {
    std::uint64_t result;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(value));
    return result;
}

std::uint64_t Verify(const Eigen::MatrixXd& actual,
                     const StoredBackend& backend,
                     const Sample& sample,
                     const JointActuatorRegressorOptions& options) {
    const Eigen::Index dof = sample.position.size(), base_columns = backend.block().cols();
    const Eigen::Index terms =
        options.drive_inertia + options.viscous_friction + options.coulomb_friction + options.torque_bias;
    if (actual.rows() != dof || actual.cols() != base_columns + dof * terms || !actual.allFinite())
        throw std::runtime_error("invalid composed output");
    std::uint64_t signature = UINT64_C(1469598103934665603);
    for (Eigen::Index column = 0; column < actual.cols(); ++column) {
        for (Eigen::Index row = 0; row < dof; ++row) {
            double expected = 0.;
            if (column < base_columns) {
                expected = backend.block()(row, column);
            } else if ((column - base_columns) / terms == row) {
                const Eigen::Index local = (column - base_columns) % terms;
                Eigen::Index term = 0;
                if (options.drive_inertia && local == term++) expected = sample.acceleration[row];
                if (options.viscous_friction && local == term++) expected = sample.velocity[row];
                if (options.coulomb_friction && local == term++)
                    expected = std::abs(sample.velocity[row]) <= options.coulomb_velocity_deadband
                                   ? 0.
                                   : std::copysign(1., sample.velocity[row]);
                if (options.torque_bias && local == term++) expected = 1.;
            }
            if (Bits(actual(row, column)) != Bits(expected))
                throw std::runtime_error("composition changed a coefficient");
            signature = (signature ^ Bits(actual(row, column))) * UINT64_C(1099511628211);
        }
    }
    return signature;
}

std::size_t ParseCount(const std::string& value, bool allow_zero = false) {
    std::size_t consumed;
    const auto count = std::stoull(value, &consumed);
    if (value.empty() || value.front() == '-' || consumed != value.size() || (!allow_zero && count == 0) ||
        count > 1000000)
        throw std::invalid_argument("count out of range");
    return static_cast<std::size_t>(count);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("expected DOF BASE_COLUMNS full|friction|bias ITERATIONS");
        const auto dof = static_cast<Eigen::Index>(ParseCount(argv[1]));
        const auto columns = static_cast<Eigen::Index>(ParseCount(argv[2], true));
        const std::string mode = argv[3];
        const auto iterations = ParseCount(argv[4]);
        if (dof > 128 || columns > 4096) throw std::invalid_argument("fixture dimensions out of range");
        auto backend = std::make_shared<StoredBackend>(dof, columns);
        JointActuatorRegressorOptions options;
        if (mode == "friction") {
            options.drive_inertia = options.viscous_friction = options.torque_bias = false;
        } else if (mode == "bias") {
            options.drive_inertia = options.viscous_friction = options.coulomb_friction = false;
        } else if (mode != "full") {
            throw std::invalid_argument("unknown term mode");
        }
        options.coulomb_velocity_deadband = .125;
        std::vector<std::string> names;
        for (Eigen::Index row = 0; row < dof; ++row) names.push_back("joint" + std::to_string(row));
        JointActuatorRegressor actuator(backend, names, options);
        Sample sample;
        sample.position = sample.torque = Eigen::VectorXd::Zero(dof);
        sample.velocity.resize(dof);
        sample.acceleration.resize(dof);
        for (Eigen::Index row = 0; row < dof; ++row) {
            sample.velocity[row] = double(row % 5 - 2) / 8.;
            sample.acceleration[row] = double(row % 7 - 3) / 16.;
        }
        const auto signature = Verify(actuator.Compute(sample), *backend, sample, options);
        for (int warmup = 0; warmup < 64; ++warmup)
            if (Verify(actuator.Compute(sample), *backend, sample, options) != signature)
                throw std::runtime_error("unstable warmup output");
        double checksum = 0.;
        const auto start = std::chrono::steady_clock::now();
        const auto cpu_start = std::clock();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const auto block = actuator.Compute(sample);
            checksum += block(0, 0) + block(dof - 1, block.cols() - 1);
        }
        const auto cpu_end = std::clock();
        const auto stop = std::chrono::steady_clock::now();
        if (Verify(actuator.Compute(sample), *backend, sample, options) != signature || !std::isfinite(checksum))
            throw std::runtime_error("unstable measured output");
        if (cpu_start == static_cast<std::clock_t>(-1) || cpu_end == static_cast<std::clock_t>(-1))
            throw std::runtime_error("CPU timer unavailable");
        std::cout << std::setprecision(17) << "{\"dof\":" << dof << ",\"base_columns\":" << columns << ",\"mode\":\""
                  << mode << "\",\"iterations\":" << iterations
                  << ",\"wall_us\":" << std::chrono::duration<double, std::micro>(stop - start).count() / iterations
                  << ",\"cpu_us\":" << 1e6 * double(cpu_end - cpu_start) / CLOCKS_PER_SEC / iterations
                  << ",\"checksum\":" << checksum << ",\"signature\":\"" << signature << "\"}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
