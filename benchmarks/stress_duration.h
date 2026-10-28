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
}  // namespace synthetic_validation
