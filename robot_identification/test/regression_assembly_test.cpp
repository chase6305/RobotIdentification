#include <robot_identification/detail/regression_assembly.h>
#include <robot_identification/identification.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace robot_identification;

class Backend final : public RegressorBackend {
public:
    std::size_t parameter_count() const noexcept override { return count; }
    std::vector<std::string> parameter_names() const override { return names; }
    Eigen::MatrixXd Compute(const Sample& sample) const override {
        ++calls;
        Eigen::MatrixXd block(3, 2);
        block.col(0) = sample.position;
        block.col(1) = sample.velocity;
        if (wrong_shape) block.conservativeResize(2, 2);
        if (nonfinite) block(0, 0) = std::numeric_limits<double>::quiet_NaN();
        return block;
    }
    std::size_t count{2};
    std::vector<std::string> names{"z_velocity_parameter", "a_position_parameter"};
    bool wrong_shape{false}, nonfinite{false};
    mutable int calls{0};
};

Dataset MakeDataset() {
    Dataset dataset;
    dataset.joint_names = {"hip", "knee", "ankle"};
    for (int index = 0; index < 9; ++index) {
        Sample sample;
        sample.time_seconds = index * .01;
        sample.position = Eigen::Vector3d(.1 * index, -.2 * index, .3 * index);
        sample.velocity = Eigen::Vector3d(1. + index, 2. - index, 3. + index);
        sample.acceleration = Eigen::Vector3d::Zero();
        sample.torque = 2. * sample.position - 3. * sample.velocity;
        dataset.samples.push_back(std::move(sample));
    }
    return dataset;
}

template <typename Exception, typename Function>
void RequireFailure(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("regression assembly accepted an invalid contract");
}

void Verify(const Dataset& dataset, const RegressionProblem& result) {
    if (!result.Validate() || result.sample_count != dataset.samples.size() || result.dof != 3) {
        throw std::runtime_error("assembled problem has invalid dimensions");
    }
    for (std::size_t sample = 0; sample < dataset.samples.size(); ++sample) {
        const Eigen::Index first = static_cast<Eigen::Index>(sample * 3);
        if (!result.regressor.block(first, 0, 3, 1).isApprox(dataset.samples[sample].position, 0) ||
            !result.regressor.block(first, 1, 3, 1).isApprox(dataset.samples[sample].velocity, 0) ||
            !result.observation.segment(first, 3).isApprox(dataset.samples[sample].torque, 0)) {
            throw std::runtime_error("assembly changed sample/parameter ordering or retained aliased scratch");
        }
    }
    if (!result.row_weights.isOnes()) throw std::runtime_error("default row weights changed");
}
}  // namespace

int main() {
    const auto dataset = MakeDataset();
    Backend backend;
    const auto original_names = backend.names;
    Verify(dataset, BuildRegressionProblem(dataset, backend));
    if (backend.names != original_names) throw std::runtime_error("metadata validation mutated parameter order");

    Eigen::MatrixXd scratch(3, 2);
    auto referenced = detail::AssembleRegressionProblem(dataset, backend, [&](const Sample& sample) -> const auto& {
        scratch.col(0) = sample.position;
        scratch.col(1) = sample.velocity;
        return scratch;
    });
    scratch.setConstant(123.);
    Verify(dataset, referenced);

    for (auto names : {std::vector<std::string>{}, std::vector<std::string>{"one"}, std::vector<std::string>{"", "two"},
                       std::vector<std::string>{"same", "same"}}) {
        Backend invalid;
        invalid.names = std::move(names);
        RequireFailure<std::invalid_argument>([&] { (void)BuildRegressionProblem(dataset, invalid); });
        if (invalid.calls != 0) throw std::runtime_error("invalid metadata reached sample computation");
    }
    for (bool wrong_shape : {false, true}) {
        Backend invalid;
        invalid.wrong_shape = wrong_shape;
        invalid.nonfinite = !wrong_shape;
        RequireFailure<std::runtime_error>([&] { (void)BuildRegressionProblem(dataset, invalid); });
    }
    auto invalid_dataset = dataset;
    invalid_dataset.samples.back().torque[0] = std::numeric_limits<double>::infinity();
    Backend untouched;
    RequireFailure<std::invalid_argument>([&] { (void)BuildRegressionProblem(invalid_dataset, untouched); });
    if (untouched.calls != 0) throw std::runtime_error("invalid dataset reached sample computation");
}
