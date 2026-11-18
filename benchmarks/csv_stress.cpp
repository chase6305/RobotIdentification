// Deterministic CSV storage/error campaign; no hardware validation.
#include <robot_identification/identification.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "stress_duration.h"

namespace {
struct Directory {
    Directory() {
        path = std::filesystem::temp_directory_path() / ("identification-csv-stress-" + std::to_string(getpid()));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("CSV stress directory already exists");
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};
template <typename Matrix>
bool SameBits(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) == 0;
}

void RewriteNewlines(const std::filesystem::path& path, const std::string& separator) {
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const char character : bytes) {
        if (character == '\n')
            output << separator;
        else
            output.put(character);
    }
    output.close();
    if (!output) throw std::runtime_error("failed to write CSV stress fixture");
}

void AppendInvalidRow(const std::filesystem::path& path, int signal_count, const std::string& separator) {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output << "100000,nan";
    for (int field = 1; field < signal_count; ++field) output << ",0";
    output << separator;
    output.close();
    if (!output) throw std::runtime_error("failed to append CSV stress fixture");
}

bool SameProcessed(const robot_identification::Dataset& left, const robot_identification::Dataset& right) {
    if (left.joint_names != right.joint_names || left.samples.size() != right.samples.size()) return false;
    for (std::size_t row = 0; row < left.samples.size(); ++row) {
        const auto& a = left.samples[row];
        const auto& b = right.samples[row];
        if (std::memcmp(&a.time_seconds, &b.time_seconds, sizeof(double)) != 0 || !SameBits(a.position, b.position) ||
            !SameBits(a.velocity, b.velocity) || !SameBits(a.acceleration, b.acceleration) ||
            !SameBits(a.torque, b.torque))
            return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const double seconds = synthetic_validation::DurationSeconds(argc, argv);
    const Directory directory;
    const auto path = directory.path / "raw.csv";
    const auto processed_path = directory.path / "processed.csv";
    const auto started = std::chrono::steady_clock::now();
    std::mt19937_64 random(0x202610282c64b3dULL);
    unsigned long long cases = 0, rows_checked = 0, rejected = 0;
    double next_progress = 0;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < seconds) {
            const int dof = 1 + random() % 64;
            const int rows = cases % 16 == 0 ? 10003 : 3 + random() % 1000;
            robot_identification::RawDataset original;
            for (int joint = 0; joint < dof; ++joint) original.joint_names.push_back("axis_" + std::to_string(joint));
            original.time_seconds.resize(rows);
            original.position.resize(rows, dof);
            original.torque.resize(rows, dof);
            const double step = std::ldexp(1., cases % 2 ? -10 : -100);
            for (int row = 0; row < rows; ++row) {
                original.time_seconds[row] = (row - 2) * step;
                for (int joint = 0; joint < dof; ++joint) {
                    original.position(row, joint) = std::sin(row * .13 + joint * .3);
                    original.torque(row, joint) = std::cos(row * .17 - joint * .2);
                }
            }
            original.position(0, 0) = -0.;
            if (cases % 7 == 0) {
                original.torque(1, 0) = std::numeric_limits<double>::denorm_min();
                original.position(2, 0) = std::numeric_limits<double>::max();
            }
            robot_identification::WriteRawDatasetCsv(original, path.string());
            const std::string separator = cases % 3 == 0 ? "\n" : cases % 3 == 1 ? "\r\n" : "\r";
            RewriteNewlines(path, separator);
            const auto actual = robot_identification::ReadRawDatasetCsv(path.string());
            if (actual.joint_names != original.joint_names || !SameBits(actual.time_seconds, original.time_seconds) ||
                !SameBits(actual.position, original.position) || !SameBits(actual.torque, original.torque)) {
                throw std::runtime_error("raw CSV changed shape, order, numeric bits or signed zero");
            }
            AppendInvalidRow(path, 2 * dof, separator);
            bool failed = false;
            try {
                (void)robot_identification::ReadRawDatasetCsv(path.string());
            } catch (const std::runtime_error&) {
                failed = true;
            }
            if (!failed || !SameBits(actual.position, original.position))
                throw std::runtime_error("late error changed an owning result or was accepted");

            // Reuse the raw RNG stream and values so previous raw cases remain
            // reproducible while all four processed signal groups are checked.
            robot_identification::Dataset processed;
            processed.joint_names = original.joint_names;
            processed.samples.reserve(static_cast<std::size_t>(rows));
            for (int row = 0; row < rows; ++row) {
                if (cases % 5 == 0) {
                    const double zero = row % 2 ? -0. : 0.;
                    processed.samples.push_back({original.time_seconds[row], Eigen::VectorXd::Constant(dof, zero),
                                                 Eigen::VectorXd::Constant(dof, -zero),
                                                 Eigen::VectorXd::Constant(dof, zero),
                                                 Eigen::VectorXd::Constant(dof, -zero)});
                } else {
                    processed.samples.push_back({original.time_seconds[row], original.position.row(row).transpose(),
                                                 -original.position.row(row).transpose(),
                                                 original.torque.row(row).transpose(),
                                                 -original.torque.row(row).transpose()});
                }
            }
            robot_identification::WriteDatasetCsv(processed, processed_path.string());
            RewriteNewlines(processed_path, separator);
            const auto actual_processed = robot_identification::ReadDatasetCsv(processed_path.string());
            if (!SameProcessed(actual_processed, processed))
                throw std::runtime_error("processed CSV changed shape, order, numeric bits or signed zero");
            AppendInvalidRow(processed_path, 4 * dof, separator);
            failed = false;
            try {
                (void)robot_identification::ReadDatasetCsv(processed_path.string());
            } catch (const std::runtime_error&) {
                failed = true;
            }
            if (!failed || !SameProcessed(actual_processed, processed))
                throw std::runtime_error("late error changed a processed owning result or was accepted");
            ++cases;
            rejected += 2;
            rows_checked += rows;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " raw_rows=" << rows_checked
                          << " processed_rows=" << rows_checked << std::endl;
                next_progress = elapsed + 60;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << cases << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED cases=" << cases << " raw_rows=" << rows_checked << " processed_rows=" << rows_checked
              << " rejected=" << rejected << std::endl;
}
