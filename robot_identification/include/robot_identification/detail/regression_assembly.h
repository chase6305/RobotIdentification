#pragma once

#include <robot_identification/identification.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace robot_identification::detail {

// Shared implementation detail, not a stable public extension interface.
// A returned block may own storage or reference call-local scratch. It is
// validated and consumed before computing the following sample's block.
template <typename ComputeBlock>
RegressionProblem AssembleRegressionProblem(const Dataset& dataset,
                                            const RegressorBackend& backend,
                                            ComputeBlock&& compute_block) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid dataset: " + reason);
    const std::size_t parameter_count = backend.parameter_count();
    auto parameter_names = backend.parameter_names();
    std::sort(parameter_names.begin(), parameter_names.end());
    if (parameter_count == 0 || parameter_names.size() != parameter_count || parameter_names.front().empty() ||
        std::adjacent_find(parameter_names.begin(), parameter_names.end()) != parameter_names.end()) {
        throw std::invalid_argument("regressor backend has an invalid parameter contract");
    }
    const auto maximum_index = static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
    const auto maximum_coefficients = std::min(maximum_index, std::numeric_limits<std::size_t>::max() / sizeof(double));
    if (dataset.samples.size() > maximum_index / dataset.dof() || parameter_count > maximum_index ||
        dataset.samples.size() * dataset.dof() > maximum_coefficients / parameter_count) {
        throw std::overflow_error("regression problem dimensions overflowed");
    }
    RegressionProblem problem;
    problem.sample_count = dataset.samples.size();
    problem.dof = dataset.dof();
    problem.regressor.resize(static_cast<Eigen::Index>(problem.sample_count * problem.dof),
                             static_cast<Eigen::Index>(parameter_count));
    problem.observation.resize(problem.regressor.rows());
    problem.row_weights = Eigen::VectorXd::Ones(problem.regressor.rows());
    for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
        const auto& block = compute_block(dataset.samples[index]);
        if (block.rows() != static_cast<Eigen::Index>(problem.dof) ||
            block.cols() != static_cast<Eigen::Index>(parameter_count) || !block.allFinite()) {
            throw std::runtime_error("regressor backend returned an invalid block");
        }
        const Eigen::Index row = static_cast<Eigen::Index>(index * problem.dof);
        problem.regressor.middleRows(row, static_cast<Eigen::Index>(problem.dof)) = block;
        problem.observation.segment(row, static_cast<Eigen::Index>(problem.dof)) = dataset.samples[index].torque;
    }
    return problem;
}

}  // namespace robot_identification::detail
