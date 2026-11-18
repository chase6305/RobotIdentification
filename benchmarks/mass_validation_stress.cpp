// Deterministic mass-spectrum and callback campaign; no hardware validation.
#include <robot_identification/identification.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stress_duration.h"

namespace {
double PositiveFinite(std::mt19937_64& random) {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53);
    std::uint64_t bits = (random() & UINT64_C(0x000fffffffffffff)) | ((random() % 2047) << 52);
    if (bits == 0) bits = 1;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void CheckMinimum(double actual, double expected, bool coupled) {
    if (actual == expected) return;
    if (!coupled || !std::isfinite(actual) || !std::isfinite(expected))
        throw std::runtime_error("minimum mass eigenvalue differs from its exact diagonal oracle");
    const long double scale =
        std::max(std::abs(static_cast<long double>(actual)), std::abs(static_cast<long double>(expected)));
    if (scale == 0.L || std::abs((static_cast<long double>(actual) - expected) / scale) > 1e-11L)
        throw std::runtime_error("minimum mass eigenvalue differs from its closed-form coupled oracle");
}
}  // namespace

int main(int argc, char** argv) {
    const auto options = synthetic_validation::ReplayCampaign(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    unsigned long long cases = 0, matrices = 0, failures = 0, case_index = options.first_case;
    std::bitset<2047> exponents;
    double next_progress = 0.;
    try {
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < options.seconds) {
            // Seed each case separately so FIRST_CASE replays it without a prefix.
            std::mt19937_64 random(UINT64_C(0x202611183b8db731) ^ (case_index * UINT64_C(0x9e3779b97f4a7c15)));
            const int dof = case_index % 16 == 0 ? 2 : 1 + random() % 64;
            const int rows = 1 + random() % 7;
            const double threshold = case_index % 3 == 0   ? std::numeric_limits<double>::denorm_min()
                                     : case_index % 3 == 1 ? std::numeric_limits<double>::min()
                                                           : std::ldexp(1., static_cast<int>(random() % 1801) - 900);
            Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(rows, dof);
            const Eigen::VectorXd parameters = Eigen::VectorXd::Ones(1);
            std::vector<Eigen::MatrixXd> models;
            models.reserve(rows);
            std::size_t expected_failed = 0;
            double expected_minimum = std::numeric_limits<double>::infinity();
            bool has_coupled_spectrum = false;
            for (int row = 0; row < rows; ++row) {
                positions(row, 0) = row;
                const unsigned kind = (case_index + row) % 7;
                Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(dof, dof);
                double eigenvalue = std::numeric_limits<double>::infinity();
                bool valid_spectrum = true;
                if (kind <= 2) {
                    if (kind == 2) {
                        mass.diagonal().setOnes();
                        mass(0, 0) = 1e-200;
                        if (dof > 1) mass(dof - 1, dof - 1) = 1e200;
                    } else {
                        for (int joint = 0; joint < dof; ++joint) mass(joint, joint) = PositiveFinite(random);
                        if (kind == 1) {
                            const int joint = random() % dof;
                            mass(joint, joint) = case_index % 2 ? -mass(joint, joint) : (row % 2 ? -0. : 0.);
                        }
                    }
                    eigenvalue = mass(0, 0);
                    for (int joint = 1; joint < dof; ++joint) eigenvalue = std::min(eigenvalue, mass(joint, joint));
                } else if (kind <= 5) {
                    const int exponent = static_cast<int>(random() % 1801) - 900;
                    const double scale = std::ldexp(1., exponent);
                    mass.diagonal().setConstant(kind == 4 ? 2. * scale : scale);
                    if (kind == 3 && dof == 1) {
                        mass.resize(2, 2);
                        mass.setIdentity();
                        valid_spectrum = false;
                    } else if (dof == 1) {
                        if (kind == 5) mass(0, 0) = -scale;
                        eigenvalue = mass(0, 0);
                    } else {
                        const int first = random() % dof;
                        const int second = (first + 1 + random() % (dof - 1)) % dof;
                        if (kind == 3) {
                            mass(first, second) = 2. * scale;
                            valid_spectrum = false;
                        } else {
                            mass(first, second) = mass(second, first) = kind == 4 ? .5 * scale : 2. * scale;
                            eigenvalue = kind == 4 ? 1.5 * scale : -scale;
                            has_coupled_spectrum = true;
                        }
                    }
                } else {
                    if (row % 2) {
                        mass.resize(dof + 1, dof + 1);
                        mass.setIdentity();
                    } else {
                        mass.setIdentity();
                        mass(0, 0) = std::numeric_limits<double>::quiet_NaN();
                    }
                    valid_spectrum = false;
                }
                if (valid_spectrum) expected_minimum = std::min(expected_minimum, eigenvalue);
                if (kind <= 2) {
                    for (int joint = 0; joint < dof; ++joint) {
                        std::uint64_t bits;
                        const double value = mass(joint, joint);
                        std::memcpy(&bits, &value, sizeof(bits));
                        exponents.set((bits >> 52) & 0x7ff);
                    }
                }
                if (!valid_spectrum || eigenvalue < threshold) ++expected_failed;
                models.push_back(std::move(mass));
            }
            int calls = 0;
            const auto result = robot_identification::ValidateMassMatrices(
                positions, parameters,
                [&](const Eigen::VectorXd& q, const Eigen::VectorXd& p) -> Eigen::MatrixXd {
                    if (calls >= rows || q.size() != dof || q[0] != calls || p.size() != 1 || p[0] != 1.)
                        throw std::runtime_error("mass callback inputs or order changed");
                    for (int joint = 1; joint < dof; ++joint)
                        if (q[joint] != 0.) throw std::runtime_error("mass callback joint coordinates changed");
                    return models[calls++];
                },
                threshold);
            if (calls != rows || result.tested != static_cast<std::size_t>(rows) || result.failed != expected_failed ||
                result.passed != (expected_failed == 0))
                throw std::runtime_error("mass validity differs from the independent spectrum oracle");
            CheckMinimum(result.minimum_eigenvalue, expected_minimum, has_coupled_spectrum);
            ++cases;
            matrices += rows;
            failures += expected_failed;
            ++case_index;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (elapsed >= next_progress) {
                std::cout << "elapsed_s=" << elapsed << " cases=" << cases << " matrices=" << matrices
                          << " next_case=" << case_index << std::endl;
                next_progress = elapsed + 30.;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED case=" << case_index << ": " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASSED cases=" << cases << " matrices=" << matrices << " failed_matrices=" << failures
              << " diagonal_exponents=" << exponents.count() << " next_case=" << case_index << std::endl;
}
