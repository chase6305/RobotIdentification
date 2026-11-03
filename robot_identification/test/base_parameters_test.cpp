#include <robot_identification/base_parameters.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace robot_identification;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void Reject(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "invalid base parameter input was accepted");
}

Eigen::MatrixXd Calibration() {
    Eigen::MatrixXd matrix(8, 4);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        const double u = std::sin(.7 * row), v = std::cos(.3 * row);
        matrix.row(row) << u, v, 2. * u - 3. * v, 0.;
    }
    return matrix;
}

void TestMappingAndHoldout() {
    const auto matrix = Calibration();
    const BaseParameterBasis basis(matrix);
    Require(basis.parameter_count() == 2 && basis.full_parameter_count() == 4, "wrong base parameter dimension");
    Require(basis.calibration_error() < 1e-13, "base parameter reconstruction is inaccurate");
    const Eigen::Vector4d truth(.7, -.3, .2, 9.);
    const auto reduced = basis.ReduceRegressor(matrix);
    Require((reduced * basis.ToBaseParameters(truth)).isApprox(matrix * truth, 1e-12),
            "base mapping changed predictions");
    Require((basis.full_to_base() * truth).isApprox(basis.ToBaseParameters(truth), 1e-13),
            "reported mapping is inconsistent");
    for (Eigen::Index column = 0; column < reduced.cols(); ++column) {
        const auto selected = static_cast<Eigen::Index>(basis.selected_columns()[static_cast<std::size_t>(column)]);
        Require(reduced.col(column).isApprox(matrix.col(selected) / basis.column_scales()[selected], 1e-14),
                "basis column does not use the fixed calibration scale");
    }
    Eigen::MatrixXd holdout(2, 4);
    holdout << .8, .2, 1., 0., -.4, .7, -2.9, 0.;
    Require(basis.ReconstructionError(holdout) < 1e-13, "valid holdout dependencies failed");
    Require((basis.ReduceRegressor(holdout) * basis.ToBaseParameters(truth)).isApprox(holdout * truth, 1e-12),
            "holdout prediction changed");
    holdout(0, 3) = 1e-150;
    Require(basis.ReconstructionError(holdout) == 1., "a newly excited zero column was hidden by its scale");
    Reject([&] { (void)basis.ReduceRegressor(holdout); });

    Eigen::MatrixXd weak(3, 2);
    weak << 1., 1., 2., 2., 3., 3.;
    const BaseParameterBasis insufficient(weak);
    weak(0, 1) += .1;
    Reject([&] { (void)insufficient.ReduceRegressor(weak); });

    Eigen::MatrixXd near(3, 2);
    near << 1., 1., 0., 1e-7, -1., -1.;
    const BaseParameterBasis coarse(near, 1e-6), fine(near, 1e-10);
    Require(coarse.parameter_count() == 1 && fine.parameter_count() == 2,
            "relative rank tolerance did not distinguish a weak direction");
    Require(coarse.calibration_error() > 1e-8, "approximate reduction concealed its reconstruction error");
    Reject([&] { (void)coarse.ReduceRegressor(near); });
    Require(coarse.ReduceRegressor(near, 1e-5).cols() == 1, "explicit approximation tolerance was ignored");
}

void TestLargeCalibrationAndLastRow() {
    Eigen::MatrixXd matrix(600, 4);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        const double u = std::sin(.7 * row), v = std::cos(.3 * row);
        matrix.row(row) << u, v, 2. * u - 3. * v, 0.;
    }
    const BaseParameterBasis basis(matrix);
    const Eigen::Vector4d truth(.7, -.3, .2, 9.);
    Require((basis.ReduceRegressor(matrix) * basis.ToBaseParameters(truth)).isApprox(matrix * truth, 1e-12),
            "large calibration changed its parameter mapping");
    matrix(599, 3) = 1e-150;
    Reject([&] { (void)basis.ReduceRegressor(matrix); });
}

void TestPairedReconstructionRows() {
    const BaseParameterBasis basis(Eigen::MatrixXd::Ones(1, 2));
    const auto selected = static_cast<Eigen::Index>(basis.selected_columns().front());
    const Eigen::Index dependent = 1 - selected;
    struct RoundingGuard {
        int previous{std::fegetround()};
        ~RoundingGuard() { (void)std::fesetround(previous); }
    } guard;
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        Require(std::fesetround(mode) == 0, "cannot configure reconstruction rounding test");
        for (const Eigen::Index rows : {1, 2, 3, 4, 5, 7, 14, 254, 255, 256, 257, 509, 510, 511, 512, 765, 766, 1021}) {
            Eigen::MatrixXd holdout = Eigen::MatrixXd::Zero(rows, 2);
            for (const Eigen::Index worst : {Eigen::Index{0}, Eigen::Index{1}, Eigen::Index{254}, Eigen::Index{255},
                                             Eigen::Index{256}, rows - 1}) {
                if (worst >= rows) continue;
                holdout.setZero();
                holdout(worst, dependent) = .5;
                Require(basis.ReconstructionError(holdout) == .5,
                        "a paired row or block tail lost its largest dependency error");
                Require(basis.ReduceRegressor(holdout, .5).rows() == rows,
                        "an exact dependency error failed its inclusive tolerance");
                Reject([&] { (void)basis.ReduceRegressor(holdout, std::nextafter(.5, 0.)); });
            }
            holdout.setZero();
            holdout(rows - 1, dependent) = std::numeric_limits<double>::denorm_min();
            Require(basis.ReconstructionError(holdout) == std::numeric_limits<double>::denorm_min(),
                    "a paired row or tail erased a subnormal dependency error");
            Require(std::fegetround() == mode, "reconstruction changed the caller rounding mode");
        }
    }
}

void TestColumnErrorMaxima() {
    struct RoundingGuard {
        int previous{std::fegetround()};
        ~RoundingGuard() { (void)std::fesetround(previous); }
    } guard;
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        Require(std::fesetround(mode) == 0, "cannot configure column-error rounding test");
        for (const double scale : {2., 3., 1e200}) {
            const BaseParameterBasis basis(Eigen::MatrixXd::Constant(1, 2, scale));
            const Eigen::Index dependent = basis.selected_columns().front() == 0 ? 1 : 0;
            const double value = scale == 1e200 ? 1e-200 : 1.;
            const double expected = scale == 2.   ? .5
                                    : scale == 3. ? 0x1.5555555555556p-2
                                                  : std::numeric_limits<double>::denorm_min();
            for (const Eigen::Index rows : {1, 2, 3, 254, 255, 256, 511, 512, 766}) {
                for (const Eigen::Index worst : {Eigen::Index{0}, rows / 2, rows - 1}) {
                    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(rows, 2);
                    values.col(dependent).setConstant(value / 2.);
                    values(worst, dependent) = value;
                    Require(basis.ReconstructionError(values) == expected,
                            "column maximum changed an outward dependency error");
                    Require(basis.ReduceRegressor(values, expected).rows() == rows,
                            "column maximum rejected its inclusive tolerance");
                    Reject([&] { (void)basis.ReduceRegressor(values, std::nextafter(expected, 0.)); });
                }
            }
        }
        Eigen::MatrixXd calibration(1, 4);
        calibration << 1., 2., 4., 8.;
        const BaseParameterBasis scaled(calibration);
        const auto selected = static_cast<Eigen::Index>(scaled.selected_columns().front());
        Eigen::MatrixXd values = Eigen::MatrixXd::Zero(511, 4);
        Eigen::Index index = 0;
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            if (column == selected) continue;
            const std::array<double, 3> errors{{.125, .25, .1875}};
            values(510, column) = scaled.column_scales()[column] * errors[static_cast<std::size_t>(index++)];
        }
        Require(scaled.ReconstructionError(values) == .25,
                "raw residuals from different column divisors were combined");
        Reject([&] { (void)scaled.ReduceRegressor(values, std::nextafter(.25, 0.)); });
        values(510, selected == 0 ? 1 : 0) = std::numeric_limits<double>::quiet_NaN();
        Reject([&] { (void)scaled.ReconstructionError(values); });

        const BaseParameterBasis boundary(Eigen::MatrixXd::Constant(1, 2, 0x1.7ce4f635ef8ffp+951));
        const Eigen::Index dependent = boundary.selected_columns().front() == 0 ? 1 : 0;
        const double rounded = 0x1.3a81a8289839ep-338;
        for (const Eigen::Index rows : {2, 255, 256, 511}) {
            Eigen::MatrixXd held = Eigen::MatrixXd::Zero(rows, 2);
            held(0, dependent) = 0x1.d3f1bf90e0dc3p+612;
            held(rows - 1, dependent) = 0x1.d3f1bf90e0dc3p+613;
            Require(boundary.ReconstructionError(held) == std::nextafter(rounded, 1.),
                    "column maximum lost the fused division residual at its largest row");
            Reject([&] { (void)boundary.ReduceRegressor(held, rounded); });
        }
        Require(std::fegetround() == mode, "column-error reduction changed the caller rounding mode");
    }
}

void TestExactIndependentColumns() {
    Eigen::MatrixXd calibration(1, 1), holdout(1, 1);
    calibration << 383.03580173451326;
    holdout << -52.208600076785956;
    const BaseParameterBasis full_rank(calibration);
    Require(full_rank.ReconstructionError(holdout) == 0., "an independent column acquired a rounding residual");
    Require(full_rank.ReduceRegressor(holdout, 0.).rows() == 1, "exact independent data failed zero tolerance");
    auto invalid = holdout;
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    Reject([&] { (void)full_rank.ReconstructionError(invalid); });
    Reject([&] { (void)full_rank.ReconstructionError(Eigen::MatrixXd::Zero(1, 2)); });

    calibration.conservativeResize(1, 2);
    holdout.conservativeResize(1, 2);
    calibration(0, 1) = holdout(0, 1) = 0.;
    const BaseParameterBasis with_zero_column(calibration);
    Require(with_zero_column.ReconstructionError(holdout) == 0., "independent residual hid an exact zero dependency");
    Require(with_zero_column.ReduceRegressor(holdout, 0.).cols() == 1, "exact zero dependency failed zero tolerance");
    holdout(0, 1) = 1e-150;
    Require(with_zero_column.ReconstructionError(holdout) == 1., "independent identity concealed a new direction");
    Reject([&] { (void)with_zero_column.ReduceRegressor(holdout, 0.); });
}

void TestReconstructionErrorRoundsOutward() {
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        const BaseParameterBasis basis(Eigen::MatrixXd::Constant(1, 2, 3.));
        Eigen::MatrixXd sample = Eigen::MatrixXd::Zero(1, 2);
        sample(0, basis.selected_columns()[0] == 0 ? 1 : 0) = 1.;
        const double tolerance = 1. / 3.;
        Require(static_cast<long double>(tolerance) < 1.L / 3.L, "boundary fixture did not round downward");
        Require(basis.ReconstructionError(sample) > tolerance, "a dependency error rounded down to its tolerance");
        Reject([&] { (void)basis.ReduceRegressor(sample, tolerance); });
        Require(basis.ReduceRegressor(sample, std::nextafter(tolerance, 1.)).cols() == 1,
                "outward conversion rejected an adequate tolerance");
        // An exact cross-product oracle finds this ratio just above the given
        // double even though the extended-precision division rounds to it.
        const BaseParameterBasis boundary(Eigen::MatrixXd::Constant(1, 2, 0x1.7ce4f635ef8ffp+951));
        sample.setZero();
        sample(0, boundary.selected_columns()[0] == 0 ? 1 : 0) = 0x1.d3f1bf90e0dc3p+613;
        const double rounded_ratio = 0x1.3a81a8289839ep-338;
        Require(boundary.ReconstructionError(sample) == std::nextafter(rounded_ratio, 1.),
                "a wide quotient hid the positive residual at a double boundary");
        Reject([&] { (void)boundary.ReduceRegressor(sample, rounded_ratio); });
        Require(boundary.ReduceRegressor(sample, std::nextafter(rounded_ratio, 1.)).cols() == 1,
                "division residual guard rejected an adequate tolerance");
        const BaseParameterBasis default_boundary(Eigen::MatrixXd::Constant(1, 2, 0x1.3cd33a5e8c8d2p-1));
        sample.setZero();
        sample(0, default_boundary.selected_columns()[0] == 0 ? 1 : 0) = 0x1.a93c475712722p-28;
        Require(default_boundary.ReconstructionError(sample) == std::nextafter(1e-8, 1.),
                "default dependency tolerance lost a positive division residual");
        Reject([&] { (void)default_boundary.ReduceRegressor(sample); });
        Require(default_boundary.ReduceRegressor(sample, std::nextafter(1e-8, 1.)).cols() == 1,
                "default-boundary residual guard rejected an adequate tolerance");
    }
    if (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent) {
        const BaseParameterBasis basis(Eigen::MatrixXd::Constant(1, 2, 1e200));
        Eigen::MatrixXd sample = Eigen::MatrixXd::Zero(1, 2);
        sample(0, basis.selected_columns()[0] == 0 ? 1 : 0) = 1e-200;
        Require(basis.ReconstructionError(sample) == std::numeric_limits<double>::denorm_min(),
                "a positive dependency error underflowed to zero");
        Reject([&] { (void)basis.ReduceRegressor(sample, 0.); });
        Require(basis.ReduceRegressor(sample, std::numeric_limits<double>::denorm_min()).cols() == 1,
                "subnormal tolerance rejected a smaller dependency error");
    }
    const BaseParameterBasis exact(Eigen::MatrixXd::Constant(1, 2, 2.));
    Eigen::MatrixXd exact_sample = Eigen::MatrixXd::Zero(1, 2);
    exact_sample(0, exact.selected_columns()[0] == 0 ? 1 : 0) = 1.;
    Require(exact.ReconstructionError(exact_sample) == .5 && exact.ReduceRegressor(exact_sample, .5).cols() == 1,
            "an exactly representable dependency boundary was widened");
}

void TestZeroColumnsWithIndependentStates() {
    Eigen::MatrixXd calibration = Eigen::MatrixXd::Zero(2, 4);
    calibration(0, 1) = 1e-200;
    calibration(1, 3) = 1e200;
    const BaseParameterBasis basis(calibration);
    Eigen::MatrixXd holdout = Eigen::MatrixXd::Zero(257, 4);
    for (Eigen::Index row = 0; row < holdout.rows(); ++row) {
        holdout(row, 1) = std::sin(.2 * row) * 1e-200;
        holdout(row, 3) = std::cos(.3 * row) * 1e200;
    }
    const Eigen::Vector4d truth(7., 2e200, -8., 3e-200);
    Require(basis.ReconstructionError(holdout) == 0. &&
                (basis.ReduceRegressor(holdout, 0.) * basis.ToBaseParameters(truth)).isApprox(holdout * truth, 1e-12),
            "zero dependencies changed scaled independent-state predictions");
    holdout(256, 0) = -std::numeric_limits<double>::denorm_min();
    Require(basis.ReconstructionError(holdout) == 1., "a subnormal new zero-column direction was ignored");
    Reject([&] { (void)basis.ReduceRegressor(holdout, std::nextafter(1., 0.)); });
    holdout(0, 2) = std::numeric_limits<double>::quiet_NaN();
    Reject([&] { (void)basis.ReconstructionError(holdout); });
    Reject([&] { (void)basis.ReconstructionError(Eigen::MatrixXd::Zero(0, 4)); });
    Reject([&] { (void)basis.ReconstructionError(Eigen::MatrixXd::Zero(1, 3)); });
}

void TestSolveAndScales() {
    auto matrix = Calibration();
    const Eigen::Vector4d scales(1e-120, 1e120, 1e80, 1.);
    matrix = matrix * scales.asDiagonal();
    const BaseParameterBasis basis(matrix);
    Require(basis.parameter_count() == 2, "column units changed the learned rank");
    RegressionProblem problem;
    problem.regressor = matrix;
    const Eigen::Vector4d truth(1e120, -1e-120, 1e-80, 0.);
    problem.observation = matrix * truth;
    problem.row_weights = Eigen::VectorXd::LinSpaced(matrix.rows(), .5, 2.);
    problem.sample_count = static_cast<std::size_t>(matrix.rows());
    problem.dof = 1;
    const auto reduced = basis.ReduceProblem(problem);
    const auto fit = Solve(reduced);
    Require(fit.rank == 2 && fit.predicted.isApprox(problem.observation, 1e-12),
            "reduced weighted solve is inaccurate");
    Require(fit.parameters.isApprox(basis.ToBaseParameters(truth), 1e-12), "reduced solve lost the parameter mapping");
    Require(
        reduced.row_weights == problem.row_weights && reduced.sample_count == problem.sample_count && reduced.dof == 1,
        "reduction changed weights or row metadata");
    Require(problem.regressor == matrix, "reduction modified its source");
}

void TestAccurateParameterCoordinates() {
    if (!std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::digits != 53) return;
    const BaseParameterBasis sum_basis(Eigen::MatrixXd::Ones(1, 3));
    std::array<double, 3> terms{-1e300, 1., 1e300};
    do {
        Eigen::Vector3d parameters(terms[0], terms[1], terms[2]);
        Require(sum_basis.ToBaseParameters(parameters)[0] == 1., "parameter cancellation lost a finite coordinate");
    } while (std::next_permutation(terms.begin(), terms.end()));
    const double a = std::nextafter(1., 0.), c = 1. - std::ldexp(1., -52);
    Eigen::MatrixXd calibration(1, 2);
    calibration << a, c;
    const BaseParameterBasis product_basis(calibration);
    Eigen::Vector2d parameters(a, -1.);
    Require(product_basis.ToBaseParameters(parameters)[0] == std::ldexp(1., -106),
            "rounded intermediate products erased a representable coordinate");
    calibration *= std::ldexp(1., -1000);
    const BaseParameterBasis tiny_basis(calibration);
    bool underflow = false;
    try {
        (void)tiny_basis.ToBaseParameters(parameters);
    } catch (const std::underflow_error&) {
        underflow = true;
    }
    Require(underflow, "a nonzero parameter coordinate silently underflowed after cancellation");
    calibration << std::numeric_limits<double>::max(), std::numeric_limits<double>::denorm_min();
    const BaseParameterBasis overflow_basis(calibration);
    bool overflow = false;
    try {
        (void)overflow_basis.ToBaseParameters(Eigen::Vector2d::Ones());
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    Require(overflow, "a parameter coordinate just beyond the finite range was accepted");
    const BaseParameterBasis large_basis(Eigen::MatrixXd::Constant(1, 2, std::numeric_limits<double>::max()));
    Require(large_basis.ToBaseParameters(Eigen::Vector2d(2., -2.))[0] == 0.,
            "representable cancellation was rejected because separate products exceed the double range");
    const BaseParameterBasis pair_basis(Eigen::MatrixXd::Ones(1, 2));
    Require(pair_basis.ToBaseParameters(Eigen::Vector2d(1., std::ldexp(1., -53)))[0] == 1.,
            "an even halfway coordinate rounded upward");
    Require(sum_basis.ToBaseParameters(Eigen::Vector3d(1., std::ldexp(1., -52), std::ldexp(1., -53)))[0] ==
                1. + std::ldexp(1., -51),
            "an odd halfway coordinate rounded downward");
    const BaseParameterBasis subnormal_basis(
        Eigen::MatrixXd::Constant(1, 2, std::numeric_limits<double>::denorm_min()));
    Require(
        subnormal_basis.ToBaseParameters(Eigen::Vector2d(1., .5))[0] == 2. * std::numeric_limits<double>::denorm_min(),
        "subnormal halfway rounding ignored the result parity");
    struct RoundingGuard {
        int previous{std::fegetround()};
        ~RoundingGuard() { (void)std::fesetround(previous); }
    } guard;
    for (const int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        Require(std::fesetround(mode) == 0, "cannot configure rounding-mode test");
        Require(product_basis.ToBaseParameters(parameters)[0] == std::ldexp(1., -106),
                "parameter coordinates changed with the caller rounding mode");
        Require(pair_basis.ToBaseParameters(Eigen::Vector2d(1., std::ldexp(1., -53)))[0] == 1.,
                "parameter halfway rounding changed with the caller rounding mode");
    }
}

void TestBlockedParameterCoordinates() {
    struct RoundingGuard {
        int previous{std::fegetround()};
        ~RoundingGuard() { (void)std::fesetround(previous); }
    } guard;
    for (const int columns : {7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257}) {
        const BaseParameterBasis dense(Eigen::MatrixXd::Ones(1, columns));
        Eigen::MatrixXd sparse = Eigen::MatrixXd::Zero(1, 2 * columns);
        for (int column = 0; column < columns; ++column) sparse(0, 2 * column) = 1.;
        const BaseParameterBasis with_zeros(sparse);
        Eigen::VectorXd values = Eigen::VectorXd::Ones(columns);
        values[0] = std::ldexp(1., 900);
        values[columns - 1] = -values[0];
        Eigen::VectorXd expanded = Eigen::VectorXd::Constant(2 * columns, std::numeric_limits<double>::max());
        for (int column = 0; column < columns; ++column) expanded[2 * column] = values[column];
        for (const int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
            Require(std::fesetround(mode) == 0, "cannot configure blocked coordinate rounding test");
            Require(dense.ToBaseParameters(values)[0] == columns - 2., "block boundary erased cancellation residuals");
            Require(with_zeros.ToBaseParameters(expanded)[0] == columns - 2.,
                    "zero columns changed the nonzero-product block boundary");
        }
        Require(std::fesetround(guard.previous) == 0, "cannot restore blocked coordinate rounding test");
        long long exact_sixteenths = 0;
        for (int column = 0; column < columns; ++column) {
            const int numerator = column % 17 - 8;
            values[column] = std::ldexp(static_cast<double>(numerator), -4);
            exact_sixteenths += numerator;
        }
        Require(dense.ToBaseParameters(values)[0] == std::ldexp(static_cast<double>(exact_sixteenths), -4),
                "ordinary blocked coordinate differs from an exactly representable sum");
    }
}

class DependentBackend final : public RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return 3; }
    std::vector<std::string> parameter_names() const override { return {"q", "constant", "mixed"}; }
    Eigen::MatrixXd Compute(const Sample& sample) const override {
        Eigen::MatrixXd result(1, 3);
        result << sample.position[0], 1., 2. * sample.position[0] + 3.;
        return result;
    }
};

void TestBackend() {
    auto source = std::make_shared<DependentBackend>();
    Eigen::MatrixXd matrix(5, 3);
    Dataset data;
    data.joint_names = {"joint"};
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        Sample sample;
        sample.time_seconds = .1 * row;
        sample.position = Eigen::VectorXd::Constant(1, -.5 + .2 * row);
        sample.velocity = sample.acceleration = Eigen::VectorXd::Zero(1);
        matrix.row(row) = source->Compute(sample);
        sample.torque = matrix.row(row) * Eigen::Vector3d(2., 3., .5);
        data.samples.push_back(sample);
    }
    BaseParameterRegressor reduced(source, BaseParameterBasis(matrix));
    const auto problem = BuildRegressionProblem(data, reduced);
    Require(reduced.parameter_count() == 2 && reduced.parameter_names()[0].rfind("base/", 0) == 0,
            "base backend names or dimensions are invalid");
    Require(Solve(problem).predicted.isApprox(problem.observation, 1e-12), "generic base backend assembly failed");
    Sample sample = data.samples.front();
    const auto owned = reduced.Compute(sample);
    sample.position[0] = 3.;
    (void)reduced.Compute(sample);
    Require(owned.isApprox(reduced.Compute(data.samples.front()), 0.), "base backend returned aliased scratch");
}

void TestInvalidInputs() {
    const auto valid = Calibration();
    for (double tolerance : {0., -1., 1., std::numeric_limits<double>::quiet_NaN()})
        Reject([&] { (void)BaseParameterBasis(valid, tolerance); });
    Reject([] { (void)BaseParameterBasis(Eigen::MatrixXd::Zero(3, 2)); });
    Reject([] { (void)BaseParameterBasis(Eigen::MatrixXd(0, 2)); });
    auto invalid = valid;
    invalid(0, 0) = std::numeric_limits<double>::infinity();
    Reject([&] { (void)BaseParameterBasis(invalid); });
    const BaseParameterBasis basis(valid);
    Reject([&] { (void)basis.ReduceRegressor(valid, -1.); });
    Reject([&] { (void)basis.ToBaseParameters(Eigen::VectorXd::Zero(3)); });
    Reject([&] { (void)basis.ReduceProblem(RegressionProblem{}); });
    Reject([&] { (void)BaseParameterRegressor(nullptr, basis); });
}
}  // namespace

int main() {
    try {
        TestMappingAndHoldout();
        TestSolveAndScales();
        TestAccurateParameterCoordinates();
        TestBlockedParameterCoordinates();
        TestLargeCalibrationAndLastRow();
        TestPairedReconstructionRows();
        TestColumnErrorMaxima();
        TestExactIndependentColumns();
        TestReconstructionErrorRoundsOutward();
        TestZeroColumnsWithIndependentStates();
        TestBackend();
        TestInvalidInputs();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
