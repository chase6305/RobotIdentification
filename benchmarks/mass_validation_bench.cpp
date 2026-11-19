// Synthetic/offline mass validation; model construction and fixture checks are untimed.
#include <robot_identification/identification.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "allocation_hooks.h"

namespace {
std::size_t Count(const char* value, std::size_t maximum) {
    const std::string text(value);
    std::size_t consumed;
    const auto result = std::stoull(text, &consumed);
    if (text.empty() || text.front() == '-' || consumed != text.size() || result == 0 || result > maximum)
        throw std::invalid_argument("invalid benchmark count");
    return result;
}
double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " DOF SAMPLES ITERATIONS positive|mixed|diagonal\n";
        return 0;
    }
    try {
        if (argc != 5) throw std::invalid_argument("expected DOF SAMPLES ITERATIONS positive|mixed|diagonal");
        const auto dof = Count(argv[1], 64), rows = Count(argv[2], 100000), iterations = Count(argv[3], 1000000);
        const std::string kind(argv[4]);
        if ((kind != "positive" && kind != "mixed" && kind != "diagonal") || rows * dof > 2000000)
            throw std::invalid_argument("invalid benchmark fixture");
        Eigen::MatrixXd positions(rows, dof);
        for (Eigen::Index row = 0; row < positions.rows(); ++row)
            for (Eigen::Index joint = 0; joint < positions.cols(); ++joint)
                positions(row, joint) = static_cast<double>((row + joint) % 7) / 64.;
        const Eigen::VectorXd parameters = Eigen::VectorXd::Constant(1, 2.);
        constexpr double threshold = .125;
        const auto model = [&](const Eigen::VectorXd& q, const Eigen::VectorXd& p) -> Eigen::MatrixXd {
            const auto phase = static_cast<int>(q[0] * 64.);
            if (kind == "mixed" && phase == 2) return Eigen::MatrixXd::Identity(dof + 1, dof + 1);
            Eigen::MatrixXd mass = Eigen::MatrixXd::Constant(dof, dof, .125);
            mass.diagonal().array() += p[0] + q[0];
            if (kind == "diagonal") {
                mass.setZero();
                for (Eigen::Index joint = 0; joint < mass.rows(); ++joint)
                    mass(joint, joint) = p[0] + q[0] + joint * .125;
            }
            if (kind == "mixed") {
                if (phase == 1) mass = -Eigen::MatrixXd::Identity(dof, dof);
                if (phase == 3) mass(0, 0) = std::numeric_limits<double>::quiet_NaN();
                if (phase == 4) mass.setZero();
                if (phase == 5) mass = threshold * Eigen::MatrixXd::Identity(dof, dof);
                if (phase == 6) {
                    if (dof == 1)
                        mass(0, 0) = -.5;
                    else
                        mass(0, 1) += .5;
                }
            }
            return mass;
        };
        const auto validate = [&] {
            return robot_identification::ValidateMassMatrices(positions, parameters, model, threshold);
        };
        const auto expected = validate();
        std::size_t expected_failed = 0;
        for (std::size_t row = 0; row < rows; ++row)
            if (kind == "mixed" && row % 7 != 0 && row % 7 != 5) ++expected_failed;
        const double expected_minimum = kind == "mixed" && rows > 1 ? -1. : dof == 1 && kind != "diagonal" ? 2.125 : 2.;
        if (expected.tested != rows || expected.failed != expected_failed ||
            expected.passed != (expected_failed == 0) ||
            std::abs(expected.minimum_eigenvalue - expected_minimum) > 2e-12)
            throw std::runtime_error("mass benchmark failed its analytic fixture");
        std::vector<double> cpu, wall;
        cpu.reserve(5);
        wall.reserve(5);
        std::size_t total_new = 0, total_malloc = 0;
        double checksum = 0;
        for (int repeat = 0; repeat < 5; ++repeat) {
            allocations = new_allocations = 0;
            counting = true;
            const auto wall_start = std::chrono::steady_clock::now();
            const auto cpu_start = std::clock();
            for (std::size_t index = 0; index < iterations; ++index) {
                const auto result = validate();
                if (result.tested != expected.tested || result.failed != expected.failed ||
                    result.passed != expected.passed || result.minimum_eigenvalue != expected.minimum_eigenvalue)
                    throw std::runtime_error("mass benchmark output changed");
                checksum += result.minimum_eigenvalue;
            }
            const auto cpu_finish = std::clock();
            const auto wall_finish = std::chrono::steady_clock::now();
            counting = false;
            if (cpu_start == static_cast<std::clock_t>(-1) || cpu_finish == static_cast<std::clock_t>(-1))
                throw std::runtime_error("CPU timer is unavailable");
            cpu.push_back(1e6 * static_cast<double>(cpu_finish - cpu_start) / CLOCKS_PER_SEC / iterations);
            wall.push_back(std::chrono::duration<double, std::micro>(wall_finish - wall_start).count() / iterations);
            total_new += new_allocations;
            total_malloc += allocations;
        }
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        std::uint64_t minimum_bits;
        std::memcpy(&minimum_bits, &expected.minimum_eigenvalue, sizeof(minimum_bits));
        std::cout << std::setprecision(17) << "{\"dof\":" << dof << ",\"samples\":" << rows
                  << ",\"iterations\":" << iterations << ",\"repetitions\":5,\"warmup_runs\":1,\"kind\":\"" << kind
                  << "\",\"cpu_median_us\":" << Median(cpu) << ",\"wall_median_us\":" << Median(wall)
                  << ",\"new_per_call\":" << static_cast<double>(total_new) / (5 * iterations)
                  << ",\"malloc_calloc_per_call\":" << static_cast<double>(total_malloc) / (5 * iterations)
                  << ",\"tested\":" << expected.tested << ",\"failed\":" << expected.failed
                  << ",\"passed\":" << (expected.passed ? "true" : "false") << ",\"minimum_bits\":\"" << std::hex
                  << minimum_bits << std::dec << "\",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& error) {
        counting = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
