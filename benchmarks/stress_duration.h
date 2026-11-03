#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace synthetic_validation {
inline double DurationSeconds(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("too many arguments");
        double seconds = 60.0;
        if (argc == 2) {
            std::size_t consumed = 0;
            seconds = std::stod(argv[1], &consumed);
            if (consumed != std::strlen(argv[1])) throw std::invalid_argument("invalid duration");
        }
        if (!std::isfinite(seconds) || seconds < 1.0 || seconds > 86400.0) {
            throw std::invalid_argument("duration out of range");
        }
        return seconds;
    } catch (const std::exception&) {
        std::cerr << "Usage: " << argv[0] << " [DURATION_SECONDS: 1..86400]\n";
        std::exit(2);
    }
}

struct ReplayOptions {
    double seconds;
    unsigned long long first_case;
};

inline ReplayOptions ReplayCampaign(int argc, char** argv, unsigned long long maximum_case = 1000000000000ULL) {
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0] << " [DURATION_SECONDS] [FIRST_CASE]\n";
        std::exit(2);
    }
    ReplayOptions options{DurationSeconds(argc > 2 ? 2 : argc, argv), 0};
    if (argc == 3) {
        try {
            std::size_t consumed;
            const std::string argument(argv[2]);
            options.first_case = std::stoull(argument, &consumed);
            if (argument.empty() || argument.front() == '-' || consumed != argument.size() ||
                options.first_case > maximum_case)
                throw std::invalid_argument("case");
        } catch (const std::exception&) {
            std::cerr << "FIRST_CASE must be an integer between 0 and " << maximum_case << ".\n";
            std::exit(2);
        }
    }
    return options;
}
}  // namespace synthetic_validation
