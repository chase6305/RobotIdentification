#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "robot_identification/identification.h"

namespace {
robot_identification::PhysicalValidationResult Check(const Eigen::MatrixXd& mass, double threshold) {
    return robot_identification::ValidateMassMatrices(
        Eigen::MatrixXd::Zero(1, mass.cols()), Eigen::VectorXd::Ones(1),
        [&](const Eigen::VectorXd&, const Eigen::VectorXd&) -> Eigen::MatrixXd { return mass; }, threshold);
}

void RequireMinimum(const robot_identification::PhysicalValidationResult& result, double value, bool passed) {
    if (result.tested != 1 || result.failed != static_cast<std::size_t>(!passed) || result.passed != passed ||
        std::memcmp(&result.minimum_eigenvalue, &value, sizeof(double)) != 0)
        throw std::runtime_error("diagonal mass validation changed its exact eigenvalue or threshold decision");
}

void TestDiagonalScaleSeparation() {
    struct RestoreRounding {
        int previous = std::fegetround();
        ~RestoreRounding() {
            if (previous != -1) std::fesetround(previous);
        }
    } restore;
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        if (std::fesetround(mode) != 0) continue;
        for (const int dof : {1, 2, 7, 14}) {
            for (const double smallest :
                 {std::numeric_limits<double>::denorm_min(), std::nextafter(std::numeric_limits<double>::min(), 0.),
                  std::numeric_limits<double>::min(), 1e-200, 1e-100, 1.}) {
                for (const double largest : {1., 1e200, std::numeric_limits<double>::max()}) {
                    Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(dof, dof);
                    mass.diagonal().setConstant(largest);
                    mass(0, 0) = smallest;
                    RequireMinimum(Check(mass, smallest), smallest, true);
                    const double above = std::nextafter(smallest, std::numeric_limits<double>::infinity());
                    RequireMinimum(Check(mass, above), smallest, false);
                    if (std::fegetround() != mode)
                        throw std::runtime_error("mass validation changed the caller's rounding mode");
                }
            }
        }
    }
}

void TestDiagonalSignsAndInvalidEntries() {
    const double threshold = std::numeric_limits<double>::denorm_min();
    for (const int dof : {1, 3}) {
        for (const double value : {-std::numeric_limits<double>::max(), -1., -threshold, -0., 0., threshold,
                                   std::numeric_limits<double>::min(), 1., std::numeric_limits<double>::max()}) {
            Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(dof, dof);
            mass.diagonal().setConstant(std::numeric_limits<double>::max());
            mass(0, 0) = value;
            RequireMinimum(Check(mass, threshold), value, value >= threshold);
        }
        for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity()}) {
            Eigen::MatrixXd mass = Eigen::MatrixXd::Identity(dof, dof);
            mass(0, 0) = invalid;
            const auto result = Check(mass, threshold);
            if (result.passed || result.tested != 1 || result.failed != 1 ||
                result.minimum_eigenvalue != std::numeric_limits<double>::infinity())
                throw std::runtime_error("non-finite diagonal mass entries were accepted or diagnosed as finite");
        }
    }
}

void TestEveryNonfinitePosition() {
    for (const int dof : {1, 2, 3, 7, 14}) {
        for (int row = 0; row < dof; ++row) {
            for (int column = 0; column < dof; ++column) {
                for (const double invalid :
                     {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()}) {
                    Eigen::MatrixXd mass = Eigen::MatrixXd::Identity(dof, dof);
                    mass(row, column) = invalid;
                    const auto result = Check(mass, 1e-9);
                    if (result.passed || result.tested != 1 || result.failed != 1 ||
                        result.minimum_eigenvalue != std::numeric_limits<double>::infinity())
                        throw std::runtime_error("a nonfinite mass entry bypassed physical validation");
                }
            }
        }
    }
}

void TestNonzeroOffDiagonalEntries() {
    for (const double scale : {std::numeric_limits<double>::denorm_min(), 1e-300, 1e-200, 1.}) {
        Eigen::Matrix2d mass;
        mass << scale, 2. * scale, 2. * scale, scale;
        const auto indefinite = Check(mass, std::numeric_limits<double>::denorm_min());
        if (indefinite.passed || indefinite.tested != 1 || indefinite.failed != 1 || indefinite.minimum_eigenvalue > 0.)
            throw std::runtime_error("a small nonzero coupling was treated as a diagonal mass matrix");
        for (const bool upper : {false, true}) {
            mass = scale * Eigen::Matrix2d::Identity();
            mass(upper ? 0 : 1, upper ? 1 : 0) = 2. * scale;
            const auto asymmetric = Check(mass, std::numeric_limits<double>::denorm_min());
            if (asymmetric.passed || asymmetric.tested != 1 || asymmetric.failed != 1 ||
                asymmetric.minimum_eigenvalue != std::numeric_limits<double>::infinity())
                throw std::runtime_error("a small asymmetric coupling bypassed mass symmetry validation");
        }
    }
}

void TestEveryOffDiagonalPosition() {
    for (const int dof : {2, 3, 4, 5, 7, 8, 14, 16}) {
        for (const double scale : {std::numeric_limits<double>::denorm_min(), 1e-200, 1.}) {
            for (int row = 0; row < dof; ++row) {
                for (int column = 0; column < dof; ++column) {
                    if (row == column) continue;
                    Eigen::MatrixXd mass = scale * Eigen::MatrixXd::Identity(dof, dof);
                    mass(row, column) = 2. * scale;
                    const auto result = Check(mass, scale);
                    if (result.passed || result.tested != 1 || result.failed != 1 ||
                        result.minimum_eigenvalue != std::numeric_limits<double>::infinity())
                        throw std::runtime_error("an off-diagonal position bypassed mass symmetry validation");
                }
            }
        }
    }
}

void TestBatchTransitions() {
    Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(8, 2);
    for (Eigen::Index row = 0; row < positions.rows(); ++row) positions(row, 0) = row;
    const Eigen::VectorXd parameters = Eigen::VectorXd::Constant(1, 7.);
    int calls = 0;
    const auto result = robot_identification::ValidateMassMatrices(
        positions, parameters,
        [&](const Eigen::VectorXd& q, const Eigen::VectorXd& p) -> Eigen::MatrixXd {
            if (q[0] != calls || q[1] != 0. || p[0] != 7.)
                throw std::runtime_error("mass callback inputs or order changed");
            const int row = calls++;
            if (row == 3) return Eigen::MatrixXd::Identity(1, 1);
            Eigen::Matrix2d mass = Eigen::Matrix2d::Identity();
            if (row == 0) mass.diagonal() << 1e-200, 1e200;
            if (row == 1) mass << 2., .25, .25, 2.;
            if (row == 2) mass.diagonal() << 3., 4.;
            if (row == 4) mass(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (row == 5) mass << 1e-200, 2e-200, 0., 1e-200;
            if (row == 6) mass << 1., 2., 2., 1.;
            if (row == 7) mass.diagonal() << 5., 6.;
            return mass;
        },
        1e-210);
    if (calls != 8 || result.tested != 8 || result.failed != 4 || result.passed ||
        std::abs(result.minimum_eigenvalue + 1.) > 1e-12)
        throw std::runtime_error("mass validation lost a diagonal/dense transition or a batch failure");
}

void TestCallbackException() {
    int calls = 0;
    bool propagated = false;
    try {
        (void)robot_identification::ValidateMassMatrices(
            Eigen::MatrixXd::Zero(3, 1), Eigen::VectorXd::Ones(1),
            [&](const Eigen::VectorXd&, const Eigen::VectorXd&) -> Eigen::MatrixXd {
                if (++calls == 2) throw std::runtime_error("mass callback failure");
                return Eigen::MatrixXd::Identity(1, 1);
            });
    } catch (const std::runtime_error& error) {
        propagated = std::string(error.what()) == "mass callback failure";
    }
    if (!propagated || calls != 2) throw std::runtime_error("mass callback exception was hidden or evaluated past");
}
}  // namespace

int main() {
    try {
        TestDiagonalScaleSeparation();
        TestDiagonalSignsAndInvalidEntries();
        TestEveryNonfinitePosition();
        TestNonzeroOffDiagonalEntries();
        TestEveryOffDiagonalPosition();
        TestBatchTransitions();
        TestCallbackException();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
