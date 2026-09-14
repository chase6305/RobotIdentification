#include <iostream>
#include <string>

#include "robot_identification/identification.h"

// Test-only bridge used by the Python/C++ interoperability regression.
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        const std::string kind = argv[1];
        if (kind == "raw") {
            robot_identification::WriteRawDatasetCsv(robot_identification::ReadRawDatasetCsv(argv[2]), argv[3]);
        } else if (kind == "processed") {
            robot_identification::WriteDatasetCsv(robot_identification::ReadDatasetCsv(argv[2]), argv[3]);
        } else {
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
