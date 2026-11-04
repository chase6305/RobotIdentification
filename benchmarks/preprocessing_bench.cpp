// Synthetic preprocessing benchmark; fixture generation is outside the timer.
#include <robot_identification/identification.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::size_t PositiveArgument(const char* argument) {
    const std::string text(argument);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("arguments must be positive integers");
    const auto value = std::stoull(text);
    if (value == 0) throw std::invalid_argument("arguments must be positive integers");
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 5) throw std::invalid_argument("usage: preprocessing_bench [SAMPLES] [DOF] [WINDOW] [REPETITIONS]");
        const auto sample_count = argc > 1 ? PositiveArgument(argv[1]) : 10000;
        const auto joint_count = argc > 2 ? PositiveArgument(argv[2]) : 7;
        const auto window = argc > 3 ? PositiveArgument(argv[3]) : 5;
        const auto repetitions = argc > 4 ? PositiveArgument(argv[4]) : 20;
        // Keep accidental benchmark invocations within a practical fixture size.
        if (sample_count < 3 || sample_count > 1000000 || joint_count > 128 || window % 2 == 0)
            throw std::invalid_argument("require 3..1000000 samples, 1..128 joints, and an odd window");

        robot_identification::RawDataset raw;
        for (std::size_t joint = 0; joint < joint_count; ++joint)
            raw.joint_names.push_back("joint_" + std::to_string(joint));
        raw.time_seconds = Eigen::VectorXd::LinSpaced(sample_count, 0., .001 * (sample_count - 1));
        raw.position.resize(sample_count, joint_count);
        raw.torque.resize(sample_count, joint_count);
        for (Eigen::Index joint = 0; joint < raw.position.cols(); ++joint)
            for (Eigen::Index row = 0; row < raw.position.rows(); ++row) {
                const double phase = (.7 + .03 * joint) * raw.time_seconds[row];
                raw.position(row, joint) = .2 * std::sin(phase) + .01 * std::cos(3. * phase);
                raw.torque(row, joint) = .4 * std::cos(phase) + .1 * joint;
            }
        robot_identification::PreprocessOptions options;
        options.smoothing_window = window;
        options.torque_smoothing_window = window;
        const auto run = [&] {
            const auto dataset = robot_identification::PreprocessPositionAndTorque(raw, options);
            const auto& sample = dataset.samples[dataset.samples.size() / 2];
            return sample.position.sum() + sample.velocity.sum() + sample.acceleration.sum() + sample.torque.sum();
        };
        (void)run();
        double checksum = 0.;
        const auto wall_start = std::chrono::steady_clock::now();
        const auto cpu_start = std::clock();
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) checksum += run();
        const auto cpu_end = std::clock();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        if (cpu_start == static_cast<std::clock_t>(-1) || cpu_end == static_cast<std::clock_t>(-1) ||
            !std::isfinite(checksum))
            throw std::runtime_error("invalid benchmark measurement");
        const double cpu_seconds = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
        std::cout << std::setprecision(17) << "{\"samples\":" << sample_count << ",\"dof\":" << joint_count
                  << ",\"window\":" << window << ",\"repetitions\":" << repetitions
                  << ",\"microseconds_per_call\":" << 1e6 * seconds / repetitions
                  << ",\"cpu_microseconds_per_call\":" << 1e6 * cpu_seconds / repetitions
                  << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
