# robot_identification

C++17/Eigen building blocks for offline robot parameter identification. The core
supports arbitrary DoF and named joints and is independent of ROS, controllers,
robot brands and model generators. It builds with CMake or colcon.

任意自由度的离线辨识核心；仅依赖 C++17/Eigen，不依赖 ROS、控制器或具体机器人。
[中文算法、数据契约和参考项目](../docs/parameter-identification.md)提供公式与数值假设；
版本变化统一记录在 [CHANGELOG](../CHANGELOG.md)。

The model is linear in the backend's parameters:

```text
tau[k] = Y(q[k], dq[k], ddq[k], context[k]) * beta + error[k]
```

`RegressorBackend` constructs `Y` and declares parameter names and count. It can
wrap Pinocchio, generated code or a vendor model. `JointActuatorRegressor` adds
optional drive inertia, viscous/Coulomb friction and torque bias columns.

## Build and run / 构建与运行

Requires Eigen >=3.3, CMake >=3.15 and a C++17 library with floating-point
`std::from_chars` support, verified with GCC 11/libstdc++.

```bash
cmake -S robot_identification -B /tmp/robot-identification \
  -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build /tmp/robot-identification --parallel 2
cmake -E chdir /tmp/robot-identification ctest --output-on-failure
/tmp/robot-identification/generic_regressor_example
/tmp/robot-identification/identification_design_example
```

Installed consumers use `find_package(robot_identification REQUIRED)` and
`robot_identification::robot_identification`. When Python and the sibling tools
are present, CTest also checks Python/C++ CSV interchange without ROS.
Examples use synthetic data and do not supply hardware parameters.

## Data and CSV / 数据与 CSV

```text
raw:       time_s,q/<joint>...,tau/<joint>...
processed: time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
```

Names must be unique, nonempty and contain no commas or newlines. Every signal
group uses the same joint order; values are finite and timestamps strictly
increase. Quoting and escaping are not interpreted. Readers accept LF/CRLF/CR,
blank lines and a final row without a newline; writers use LF and the classic
locale. Leading ASCII whitespace, explicit signs and `0x` numbers are accepted;
trailing junk, overflow and nonzero values that underflow to zero are rejected.
Representable subnormals and signed zero round-trip exactly.

Raw imports use growing column chunks and release each consumed field while
assembling owning outputs. Final storage still scales with sample/joint count.
Both writers check buffered output when closing. Keep raw recordings immutable
and create separate processed artifacts. Installed JSON Schemas under
`share/robot_identification/schema` define session, candidate, review and rollback
contracts; result status remains `candidate`.

## Numerical behavior / 数值行为

- Preprocessing uses symmetric means and nonuniform three-point secant
  derivatives. Large windows use compensated rolling sums with periodic rebasing;
  short windows use direct reductions. Exceptional means and derivatives retry
  with scaled or extended-range arithmetic. Unrepresentable results fail.
- `Solve` supports WLS, ridge and Huber IRLS and reuses call-local weighted
  matrices and SVD storage. Reported rank and conditioning describe measured
  data, including with ridge. Returned weights belong to the last completed solve.
- Covariance uses the solve's retained singular directions, including augmented
  ridge rows. Scaled root errors preserve representable uncertainty at extreme
  scales. Rank-deficient nullspace zeros follow the minimum-norm pseudoinverse
  convention and do not imply certainty; nonfinite covariance fails.
- Residual metrics use scaled magnitudes. Normalized RMSE is dimensionless;
  zero reference/zero error gives zero and zero reference/nonzero error gives
  infinity. Nonfinite predictions or residuals fail.
- `HarmonicTrajectory` provides owning `Evaluate` and reusable `EvaluateInto`.
  Matching output dimensions need no vector allocation; failed reusable calls
  may leave partial output. Conservative outward envelopes precede sampled
  position, velocity, acceleration and jerk limits, including both endpoints.
  Active phase overflow fails; exceptional products retry in extended range.
- `BaseParameterBasis` learns a normalized column-pivoted QR basis and checks
  dependencies on new regressors. Selected columns are identities; zero-column
  dependencies are checked directly. Generic reconstruction preserves per-row
  accumulation order with bounded scratch and an outward error result.
  `ToBaseParameters` rounds the exact stored-map dot product to nearest/ties-even
  on IEEE binary64 with gradual underflow, using a wide screen and bounded integer
  fallback. Exact overflow and nonzero results rounding to zero throw. QR learning
  itself remains numerical; keep the basis with its model and validate holdout states.
- Fourier experiment design performs deterministic coordinate search over
  coefficients with fixed offsets, frequency and parameter/noise scales. It
  improves rank, then SVD log information, under conservative limits. Exhausted
  budgets and rank deficiency remain explicit; local convergence is not global optimality.

The [algorithm notes](../docs/parameter-identification.md) specify equations,
units, rounding assumptions and limits. Numerical recovery depends on the
platform's long-double range; strict floating arithmetic is required.
[Benchmarks](../benchmarks/README.md) describe measurement and reference campaigns.

## Offline boundary / 离线边界

A numerical solve produces a candidate. Applying parameters requires independent
quality review, physical checks, holdout validation, staged hardware validation
and rollback preparation. The library does not update a running controller.
