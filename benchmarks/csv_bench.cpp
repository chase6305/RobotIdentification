// Synthetic/offline file measurement; no hardware timing guarantee.
#include <robot_identification/identification.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allocation_hooks.h"

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " RAW_CSV [REPETITIONS: 1..100]\n";
        return 0;
    }
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " RAW_CSV [REPETITIONS: 1..100]\n";
        return 2;
    }
    try {
        int repeats = 5;
        if (argc == 3) {
            std::size_t consumed;
            repeats = std::stoi(argv[2], &consumed);
            if (argv[2][consumed] != '\0' || repeats < 1 || repeats > 100) {
                throw std::invalid_argument("repetitions must be 1..100");
            }
        }
        if (repeats > 1) (void)robot_identification::ReadRawDatasetCsv(argv[1]);
        std::vector<double> elapsed;
        std::size_t total_allocations = 0, total_new = 0, rows = 0, dof = 0;
        double checksum = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            allocations = new_allocations = 0;
            counting = true;
            const auto started = std::chrono::steady_clock::now();
            {
                const auto dataset = robot_identification::ReadRawDatasetCsv(argv[1]);
                rows = static_cast<std::size_t>(dataset.time_seconds.size());
                dof = dataset.joint_names.size();
                checksum += dataset.time_seconds.sum() + dataset.position.sum() + dataset.torque.sum();
            }
            const auto finished = std::chrono::steady_clock::now();
            counting = false;
            total_allocations += allocations;
            total_new += new_allocations;
            elapsed.push_back(std::chrono::duration<double, std::milli>(finished - started).count());
        }
        std::sort(elapsed.begin(), elapsed.end());
        const auto middle = elapsed.size() / 2;
        const double median = elapsed.size() % 2 ? elapsed[middle] : .5 * elapsed[middle - 1] + .5 * elapsed[middle];
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
        std::cout << std::setprecision(12) << "raw_csv samples=" << rows << " dof=" << dof << " repeats=" << repeats
                  << " warmup_runs=" << (repeats > 1 ? 1 : 0) << " median_ms=" << median
                  << " malloc_calloc_per_read=" << double(total_allocations) / repeats
                  << " new_per_read=" << double(total_new) / repeats << " peak_rss_kib=" << usage.ru_maxrss
                  << " checksum=" << checksum << '\n';
    } catch (const std::exception& error) {
        counting = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
