# Changelog

## Unreleased

- Detect CSV record boundaries from the header in both Python and C++ readers.
  Retain Python's existing CR-only support and extend C++ readers to match it.
  Preserve leading carriage-return whitespace inside LF/CRLF numeric fields and
  original bytes in digest-enabled inspection.

- Preserve covariance across uniform row-weight scaling when a residual variance
  intermediate would overflow or underflow. Keep a scaled root error through the
  singular-value ratio, and continue rejecting non-finite final covariance.

- Declare recorder pytest requirements in setuptools as well as the ROS manifest,
  so colcon executes its unit tests instead of accepting an empty unittest run.

- Add `scripts/test_ros2.sh` to build, test and exercise installed recording with
  the same root-package discovery. Exclude nested generated CMake projects from
  colcon tests, support external output roots, and propagate build/test failures.
  Use the wrapper in ROS CI and quick-start instructions.

- Compute covariance from the final consumed solve's SVD and its retained
  directions. Avoid squaring the condition number in a separately decomposed
  information matrix, which could report zero variance for weak observable
  parameters. Ridge uses the augmented SVD; measured-data rank is unchanged.
  Non-finite final covariance still fails explicitly.

- Pinocchio backend 0.2.0 adds `BuildProblem(dataset)`, reusing a call-local
  workspace across samples while retaining core dataset validation and ordering.
  Concurrent calls keep independent scratch. The API is additive with no object
  layout or vtable change; consumers using it must link version 0.2.0.

- Preserve representable Huber row weights when the residual ratio is subnormal
  or underflows before multiplication by a large base weight. Keep ordinary
  ratios on the existing fast path and cover rank/covariance at extreme scales.

- Compute harmonic envelopes with scaled hypot and form derivatives through
  successive coefficient/frequency products. Reject tiny trajectories whose
  true envelope exceeds limits even when sparse samples miss every peak, and
  keep representable wide position half-spans finite.

- Run installed core/Pinocchio consumers as part of CTest. Forward the selected
  compiler, build type, and instrumentation flags to consumer configurations so
  static sanitizer builds link and execute against their required runtimes.
- Reuse weighted data matrices, singular-value decompositions, and residual
  scratch across Huber IRLS iterations. Keep measured-data rank, regularization,
  final weights, and covariance contracts unchanged. Avoid dynamic per-harmonic
  temporaries when evaluating excitation trajectories.

- Extract the core, optional Pinocchio regressor, ROS 2 recorder, schemas, tests,
  and offline artifact tools from RobotServer `a6c70ec`, retaining relevant Git history.
- Preserve the existing public C++ targets, dataset formats, and artifact schemas.
  Core remains at 0.1.0. Recorder 0.1.1 explicitly declares
  string-array joint names so ROS Humble accepts configured joint ordering, and
  closes its context idempotently after SIGINT.
- Publish offline inspection and artifact commands under the new
  `robot_identification_tools` 0.1.0 package. Python imports and `ros2 run` package
  selection move from `robot_server_tools`; console-script names are unchanged.
- Add independent validation, Python packaging, formatting, schema checks, and
  standalone/ROS 2 CI definitions, including an installed-recorder integration
  check with synthetic standard JointState messages.
