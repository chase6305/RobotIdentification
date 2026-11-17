// Synthetic/offline export measurement; fixture generation and verification are untimed.
#include <robot_identification/identification.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allocation_hooks.h"

namespace {
struct Directory {
    Directory() {
        path = std::filesystem::temp_directory_path() / ("identification-csv-write-" + std::to_string(getpid()));
        if (!std::filesystem::create_directory(path))
            throw std::runtime_error("CSV benchmark directory already exists");
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

std::size_t Count(const char* argument, std::size_t maximum) {
    const std::string text(argument);
    std::size_t consumed;
    const auto value = std::stoull(text, &consumed);
    if (text.empty() || text.front() == '-' || consumed != text.size() || value == 0 || value > maximum)
        throw std::invalid_argument("invalid benchmark count");
    return static_cast<std::size_t>(value);
}

template <typename Matrix>
bool SameBits(const Matrix& left, const Matrix& right) {
    return left.rows() == right.rows() && left.cols() == right.cols() &&
           std::memcmp(left.data(), right.data(), static_cast<std::size_t>(left.size()) * sizeof(double)) == 0;
}

double Median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : .5 * values[middle - 1] + .5 * values[middle];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " raw|processed SAMPLES DOF REPETITIONS [long|short]\n";
        return 0;
    }
    try {
        if (argc < 5 || argc > 6)
            throw std::invalid_argument("expected raw|processed SAMPLES DOF REPETITIONS [long|short]");
        const std::string format(argv[1]), distribution = argc == 6 ? argv[5] : "long";
        if ((format != "raw" && format != "processed") || (distribution != "long" && distribution != "short"))
            throw std::invalid_argument("invalid format or numeric distribution");
        const auto rows = Count(argv[2], 1000000), dof = Count(argv[3], 256), repeats = Count(argv[4], 100);
        if (rows < 3 || rows * dof > 10000000)
            throw std::invalid_argument("fixture needs at least 3 rows and at most 10000000 joint samples");
        robot_identification::RawDataset raw;
        robot_identification::Dataset processed;
        for (std::size_t joint = 0; joint < dof; ++joint) raw.joint_names.push_back("j" + std::to_string(joint));
        processed.joint_names = raw.joint_names;
        if (format == "raw") {
            raw.time_seconds.resize(static_cast<Eigen::Index>(rows));
            raw.position.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(dof));
            raw.torque.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(dof));
        } else {
            processed.samples.reserve(rows);
        }
        for (std::size_t row = 0; row < rows; ++row) {
            const double time = static_cast<double>(row) * .0009765625;
            Eigen::VectorXd position(static_cast<Eigen::Index>(dof));
            for (std::size_t joint = 0; joint < dof; ++joint)
                position[static_cast<Eigen::Index>(joint)] =
                    distribution == "short" ? ((row + joint) % 7 == 0 ? -0. : .25 * ((row + joint) % 5))
                                            : std::sin(row * .003 + joint * .2);
            if (format == "raw") {
                raw.time_seconds[static_cast<Eigen::Index>(row)] = time;
                raw.position.row(static_cast<Eigen::Index>(row)) = position.transpose();
                raw.torque.row(static_cast<Eigen::Index>(row)) = -position.transpose();
            } else {
                processed.samples.push_back({time, position, -position, position, -position});
            }
        }
        const Directory directory;
        const std::string path = (directory.path / "output.csv").string();
        const auto write = [&] {
            if (format == "raw")
                robot_identification::WriteRawDatasetCsv(raw, path);
            else
                robot_identification::WriteDatasetCsv(processed, path);
        };
        if (repeats > 1) write();
        std::vector<double> wall_ms, cpu_ms;
        std::size_t total_allocations = 0, total_new = 0;
        wall_ms.reserve(repeats);
        cpu_ms.reserve(repeats);
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            allocations = new_allocations = 0;
            counting = true;
            const auto started = std::chrono::steady_clock::now();
            const auto cpu_started = std::clock();
            write();
            const auto cpu_finished = std::clock();
            const auto finished = std::chrono::steady_clock::now();
            counting = false;
            if (cpu_started == static_cast<std::clock_t>(-1) || cpu_finished == static_cast<std::clock_t>(-1))
                throw std::runtime_error("process CPU timer is unavailable");
            total_allocations += allocations;
            total_new += new_allocations;
            wall_ms.push_back(std::chrono::duration<double, std::milli>(finished - started).count());
            cpu_ms.push_back(1000. * static_cast<double>(cpu_finished - cpu_started) / CLOCKS_PER_SEC);
        }
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
        if (format == "raw") {
            const auto restored = robot_identification::ReadRawDatasetCsv(path);
            if (restored.joint_names != raw.joint_names || !SameBits(restored.time_seconds, raw.time_seconds) ||
                !SameBits(restored.position, raw.position) || !SameBits(restored.torque, raw.torque))
                throw std::runtime_error("raw CSV export did not preserve the complete fixture");
        } else {
            const auto restored = robot_identification::ReadDatasetCsv(path);
            if (restored.joint_names != processed.joint_names || restored.samples.size() != processed.samples.size())
                throw std::runtime_error("processed CSV export changed dimensions or joint order");
            for (std::size_t row = 0; row < rows; ++row) {
                const auto& actual = restored.samples[row];
                const auto& expected = processed.samples[row];
                if (actual.time_seconds != expected.time_seconds || !SameBits(actual.position, expected.position) ||
                    !SameBits(actual.velocity, expected.velocity) ||
                    !SameBits(actual.acceleration, expected.acceleration) || !SameBits(actual.torque, expected.torque))
                    throw std::runtime_error("processed CSV export did not preserve the complete fixture");
            }
        }
        std::ifstream input(path, std::ios::binary);
        std::uint64_t signature = 14695981039346656037ULL;
        std::uintmax_t bytes = 0;
        char buffer[8192];
        while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
            bytes += static_cast<std::uintmax_t>(input.gcount());
            for (std::streamsize index = 0; index < input.gcount(); ++index) {
                signature ^= static_cast<unsigned char>(buffer[index]);
                signature *= 1099511628211ULL;
            }
        }
        if (!input.eof()) throw std::runtime_error("failed to checksum CSV export");
        std::cout << std::setprecision(17) << "{\"format\":\"" << format << "\",\"distribution\":\"" << distribution
                  << "\",\"samples\":" << rows << ",\"dof\":" << dof << ",\"repetitions\":" << repeats
                  << ",\"warmup_runs\":" << (repeats > 1 ? 1 : 0) << ",\"cpu_median_ms\":" << Median(cpu_ms)
                  << ",\"wall_median_ms\":" << Median(wall_ms)
                  << ",\"malloc_calloc_per_write\":" << static_cast<double>(total_allocations) / repeats
                  << ",\"new_per_write\":" << static_cast<double>(total_new) / repeats
                  << ",\"peak_rss_before_verification_kib\":" << usage.ru_maxrss << ",\"output_bytes\":" << bytes
                  << ",\"output_signature\":\"" << std::hex << signature << "\"}\n";
    } catch (const std::exception& error) {
        counting = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
