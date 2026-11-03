#include <robot_identification/base_parameters.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "allocation_hooks.h"

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::invalid_argument("expected CASE [ITERATIONS]");
        const std::string name(argv[1]);
        const int iterations = argc == 3 ? std::stoi(argv[2]) : 1000;
        if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
        Eigen::MatrixXd calibration;
        if (name == "map-scalar") {
            calibration = Eigen::MatrixXd::Constant(1, 1, 383.03580173451326);
        } else if (name == "map-unit") {
            calibration = Eigen::MatrixXd::Identity(80, 80);
        } else if (name == "map-sparse") {
            calibration = Eigen::MatrixXd::Zero(80, 160);
            for (int index = 0; index < 80; ++index) calibration(index, index) = .7 + .013 * index;
        } else if (name == "map-dense" || name == "map-dense-small" || name == "map-dense-wide" ||
                   name == "map-dense-large" || name == "map-dense-signed") {
            const int rank = name == "map-dense-small"                                ? 8
                             : name == "map-dense-large"                              ? 160
                             : name == "map-dense-wide" || name == "map-dense-signed" ? 24
                                                                                      : 80;
            const int dependent = name == "map-dense-wide" ? 168 : name == "map-dense-signed" ? 72 : rank;
            const int rows = rank + rank / 2;
            std::mt19937_64 random(0x237dac918fULL);
            std::uniform_real_distribution<double> unit(-1., 1.);
            Eigen::MatrixXd independent(rows, rank), dependencies(rank, dependent);
            for (Eigen::Index column = 0; column < independent.cols(); ++column)
                for (Eigen::Index row = 0; row < independent.rows(); ++row) independent(row, column) = unit(random);
            for (Eigen::Index column = 0; column < dependencies.cols(); ++column)
                for (Eigen::Index row = 0; row < dependencies.rows(); ++row)
                    dependencies(row, column) = .0125 * unit(random);
            calibration.resize(rows, rank + dependent);
            calibration.leftCols(rank) = independent;
            calibration.rightCols(dependent).noalias() = independent * dependencies;
        } else {
            throw std::invalid_argument("unknown parameter mapping case");
        }
        const robot_identification::BaseParameterBasis basis(calibration);
        Eigen::VectorXd parameters = Eigen::VectorXd::LinSpaced(calibration.cols(), .53, 1.41);
        if (name == "map-dense-signed")
            for (Eigen::Index column = 1; column < parameters.size(); column += 2)
                parameters[column] = -parameters[column];
        (void)basis.ToBaseParameters(parameters);
        allocations = new_allocations = 0;
        counting = true;
        double checksum = 0.;
        const auto wall_started = std::chrono::steady_clock::now();
        const auto cpu_started = std::clock();
        for (int iteration = 0; iteration < iterations; ++iteration)
            checksum += basis.ToBaseParameters(parameters).sum();
        const double cpu = double(std::clock() - cpu_started) / CLOCKS_PER_SEC;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_started).count();
        counting = false;
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite mapping checksum");
        std::cout << std::setprecision(17) << "{\"case\":\"" << name << "\",\"iterations\":" << iterations
                  << ",\"microseconds_per_call\":" << wall * 1e6 / iterations
                  << ",\"cpu_microseconds_per_call\":" << cpu * 1e6 / iterations
                  << ",\"allocations_per_call\":" << double(allocations + new_allocations) / iterations
                  << ",\"rank\":" << basis.parameter_count() << ",\"columns\":" << basis.full_parameter_count()
                  << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
