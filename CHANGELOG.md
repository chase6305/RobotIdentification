# Changelog

## Unreleased

- Core 0.2.16 preserves representable preprocessing derivatives when individual secants
  underflow or round at the normal/subnormal boundary; retry only exceptional
  arithmetic in extended range and test both differentiation passes. Exact-midpoint
  interior differences eliminate the middle coordinate before arithmetic,
  preserving tiny endpoint changes even beside extreme middle samples. Check
  exact interval equality through subtraction residuals under nearest rounding;
  rounded-equal nonuniform intervals and other modes retain the generic stencil.
- Skip zero calibration columns during mixed base-parameter dependency
  reconstruction, preserving newly excited zero-column rejection and errors
  larger than one from actual dependencies.
- Tools 0.1.2 rejects occupied artifact destinations before scanning input recordings or
  configurations, while retaining the publication-time check and atomic
  no-overwrite semantics.

- Core 0.2.15 uses project-specific schema identifiers and package metadata.
  Remove obsolete migration text and tracked process notes. Artifact fields
  and schema version 1 remain unchanged.
- Separate preprocessing and harmonic trajectories into focused source files;
  name numerical quantities and recorder state explicitly, and correct smoothing,
  Coulomb-deadband, numerical-rank and CSV-underflow comments. Reuse shared
  backend-name checks.
- Compute ridge observability diagnostics once at the final IRLS weights, and
  traverse short smoothing windows by contiguous joint columns. Keep public APIs
  and numerical policies compatible. Extend independent preprocessing and Huber
  campaigns to cover smoothing, trimming and ridge; add a preprocessing benchmark.
- Initialize every synthetic torque row before preprocessing in the generic
  example. Its full raw input is validated before trimming; uninitialized edge
  rows could previously cause an intermittent failure. Check the fitted values
  against the known synthetic parameters, and initialize Eigen storage with NaNs
  in standalone CI to catch this class of regression.
- Retain only the ridge directions needed for covariance before releasing the
  augmented decomposition workspace ahead of final measured-data diagnostics.
  Preserve its initial allocation order and rank policies; add analytic tests
  for different measured/augmented ranks, tiny ridge and zero retained rank.

- Core 0.2.14 separates weighted/robust solving, diagnostics and covariance into
  `solver.cpp`, and CSV parsing/writing into `dataset_csv.cpp`. Put shared data
  contracts in `data_validation.cpp` so I/O and solvers avoid unrelated trajectory
  code. Make base/full column coordinates, trial coefficients and step
  fractions explicit; iterate sine/cosine matrices directly and reserve the
  base-parameter name cache once. Numerical expressions and public APIs remain
  compatible.
- Recorder 0.2.3 separates CSV encoding from durable writes and state commit,
  with explicit ordering and failure-state comments. The scalar hybrid demo
  names physical quantities, units, parameter indices and integration modes.

- Core 0.2.13 factors its private calibration scratch in place, avoiding a
  second full matrix in QR, and reserves final actuator parameter names instead
  of constructing temporary prefixes. Public headers and numerical behavior
  remain compatible.
- Recorder 0.2.2 shares one dictionary for name uniqueness and reordering,
  and removes an unused subscription alias; the ROS node continues to own and
  destroy its subscription. ROS 2 CI relies on its existing colcon coverage
  instead of repeating the same Python suites after a successful run.

- Core 0.2.12 borrows QR permutation indices and changes one trial coefficient
  instead of copying an entire excitation design. Restore rejected coefficients
  exactly, including signed zero. Rotate IRLS parameter buffers to avoid copying
  a parameter vector before every subsequent solve. Numerical policies and
  public interfaces are unchanged.
- Pinocchio 0.2.2 reserves each final parameter name once. Recorder 0.2.1 removes
  a duplicate joint-set cache; ordering and failure/recovery behavior are unchanged.
- Tools 0.1.1 avoids discarded metadata digests and redundant JSON name scans,
  uses the JSON scanner's numeric grammar for zero checks, and rejects missing
  fields before allocating an unknown-field set. CSV error locations are
  formatted only on rejected numeric fields. Review and rollback continue to
  bind digests to the same validated byte snapshots.

- Add an opt-in scalar hybrid-rollout demonstration: causal fixed tanh features,
  learned residual readout, positive inertia, nonnegative damping and
  semi-implicit stepping.
  Compare a fitted physics baseline against independent exact synthetic motion;
  document force-scale anchors, integration error and identifiability limits.
  The demo uses the existing Eigen dependency; core interfaces and artifact
  schemas are unchanged.

- Core 0.2.11 removes redundant IRLS weight/singular-value copies and intermediate
  results, moves trajectory bounds into validation output, and stores raw CSV
  chunks directly. Share CSV input and joint-name checks; derive validation
  success from its existing failure state. Remove a duplicate derivative scan
  after each coefficient has already been checked. Public interfaces and
  numerical policies are unchanged.
  Read header bytes through the existing file buffer and validate backend names
  through views, removing per-byte stream setup and redundant string copies.
  Header fields and column-order checks use views during validation; output
  joint names retain their owning storage.
  Compute rank/conditioning only for the final solve, avoiding unnecessary
  intermediate diagnostic overflow; reuse the measured rank in covariance.
- Correct the minimum Eigen version to 3.3.1, which exports `Eigen3::Eigen`.
  Remove JacobiSVD status queries unavailable in early Eigen 3.3 releases;
  inputs and singular values remain checked for finiteness. Installed consumers
  use the same config-package dependency as the core build.
  Optional solver campaigns compute stable matrix norms through vector views
  supported by the same minimum version.
- Consolidate formatting and replay-argument runners, remove inactive header
  checks, skip generated directories at every depth, and shorten the benchmark
  guide around current commands. Processed CSV retains its final validation
  after paired measurements showed a regression when it was omitted.
  Python inspection constructs only the selected CSV layout and reuses position
  headers instead of allocating both expected layouts.
  Scan header delimiters in buffered blocks while preserving original-byte
  digests, newline handling and numeric-row boundaries.

- Core 0.2.10 reads CSV header fields directly without a string stream or
  whole-header copies. Share the raw chunk materialization loop while releasing
  each field before allocating the next output. Preserve header grammar, numeric
  parsing, row order and bounded spare capacity. Benchmarks inherit the core's
  exported C++17 requirement and share identical instrumented-target setup.
  Public headers and numerical algorithms are unchanged.

- Core 0.2.9 stores solver decompositions in place, removing heap ownership
  indirection while preserving the allocation-saving diagnostic-input branch.
  Share raw/processed CSV
  writer setup while preserving validation order, headers, numeric formatting
  and close-time errors. Consolidate repeated CMake test declarations and keep
  the core README focused on current contracts, with version history here.
  Public headers and numerical policies are unchanged.

- Core 0.2.8 reduces dependency-check division work to once per column/row block.
  The largest nonnegative residual gives the same corrected quotient by
  monotonicity. Preserve row accumulation order, outward conversion, tolerance
  policy, NaN/infinite handling and bounded scratch. Add column-scale, block-tail,
  rounding-mode and fused-residual maximum regressions. Public headers are unchanged.
- Core 0.2.7 shares each dependency coefficient between two adjacent rows in
  `ReconstructionError`, retaining each row's increasing-base accumulation order,
  the 255-row scratch bound and outward error conversion. Public headers, object
  layouts, tolerance policy and heap workspace are unchanged. Add analytical
  pair/block-tail regressions and isolated reconstruction benchmarks.
- Core 0.2.6 reduces dense coordinate mapping overhead with eight-product wide
  blocks and compensated partial-sum merging. A derived16-unit-roundoff screen
  preserves exact nearest-even output and range checks through the existing
  integer fallback. Public layouts, independent/zero-column shortcuts and core
  dependencies are unchanged. Add block-tail/cancellation tests, longer dyadic
  reference rows and benchmarks for several dense matrix shapes.
- Core 0.2.5 preserves cancellation in `ToBaseParameters`. IEEE binary64 stored
  mapping products are accumulated exactly when a compensated wide screen cannot
  certify nearest/ties-even rounding. Range checks use the exact coordinate,
  including positive values lost below zero and values just beyond max double.
  The integer fallback uses bounded stack storage and no additional dependency.
  Independent nonzero columns bypass redundant zero entries. Add rounding-mode,
  product-residual and range regressions, mapping benchmarks and an optional
  independent Boost integer reference campaign.

- Core 0.2.4 adds `HarmonicTrajectory::EvaluateInto` for owning output-buffer
  reuse. Sampled limit validation and excitation scoring reuse buffers across
  samples while the existing owning `Evaluate` API remains available.
  Guard wide amplitude square roots and derivative products before outward
  conversion: a wide result can itself land below the exact value on a double
  boundary. Screen ordinary wide calculations with an error margin and recompute
  uncertain joints with fused residuals, preserving exact Pythagorean bounds.
  Add exact-dyadic boundary validation and sampling benchmarks.
  Forward configured dependency prefixes to the core and Pinocchio CTest
  install consumers, including builds configured without environment prefixes.

- Core 0.2.3 rounds the extended-precision dependency error upward on return.
  A value above the caller's tolerance cannot pass merely because conversion
  rounded it down, and a positive error cannot become zero during conversion.
  Check the fused division residual when a wide quotient lands on a double
  boundary; exactly representable error ratios retain their exact boundary.
  When only zero calibration columns are dependent, check those columns directly
  without allocating reconstruction scratch. Newly excited subnormal columns
  still fail, and dimensions/finite values remain validated first.

- Core 0.2.2 preserves small contributions in harmonic envelope accumulation
  and offset addition by accumulating positive addition residuals and rounding
  their final combination outward. Fix a
  sparse-validation false pass after cancellation at a position boundary; exact
  sums retain their exact bounds. Reuse the dependency divisor workspace for
  selected-column markers, removing a per-call bitmap allocation. Use an odd
  bounded row-block size to avoid power-of-two cache strides in reconstruction.

- Core 0.2.1 treats selected base columns as exact identities, fixing spurious
  zero-tolerance holdout rejection and avoiding their redundant reconstruction.
  Unit parameter/noise scales use direct normalization without an overflowing
  reciprocal; non-unit scales retain extended-range arithmetic. Add synthetic
  design benchmarks and independent-column/default-scale stress coverage.
- Standalone and ROS validation support unpacked source archives without Git
  metadata; Git checkout roots retain the final whitespace check.

- Core 0.2.0 adds normalized, column-pivoted QR base parameter identification,
  an explicit full-to-base transform, holdout dependency checks and a composable
  backend wrapper. Bound reconstruction scratch by row blocks and avoid copying
  the full source regressor during problem reduction. Existing public object
  layouts and signatures remain unchanged.
- Add constrained Fourier experiment design using deterministic coordinate
  pattern search and SVD log information. Keep parameter/noise scales fixed,
  rank deficiency visible, evaluation budgets strict and limit checks mandatory.
  A coordinate poll interrupted by the budget cannot shrink the search step or
  report convergence.
  Include an end-to-end example and native Pinocchio integration validation.

- Run CTest from the selected build directory using CMake 3.15-compatible commands.
  Avoid silently passing with zero tests when older CTest ignores `--test-dir`.

- Recover representable harmonic samples after phase/product underflow or
  overflowing partial sums. Preserve sums of individually underflowing
  derivatives, skip inactive harmonics, and include the final sample for every
  positive validation duration. Add independent trajectory sample/envelope checks.

- Preserve derivative envelopes from subnormal amplitudes and prevent normal
  amplitudes from rounding inward before a limit check. Keep amplitudes and
  envelope sums in extended range and round the final conversion outward, removing
  one temporary vector. Retain owning outputs and unchanged public object layouts.

- Exclude the documented `.venv` directory from Python and C++ formatting scans,
  keeping installed dependency sources outside style checks and format edits.

- Reject repeated hexadecimal exponent signs such as `0x1p+-2` independently
  of the standard library conversion behavior. Preserve valid signed exponents
  and keep raw/processed C++ and Python numeric acceptance aligned.

- Preserve representable normalized RMSE when individual RMS values are
  subnormal or underflow to zero. Retain the scaled roots and combine mantissas
  before restoring their exponent difference. Add sparse analytic cases and an
  independent extended-range, directly replayable metric validation campaign.

- Preserve finite velocity and acceleration on strongly unequal adjacent sample
  intervals. Anchor interior differentiation at the larger-weight secant and
  retry underflowed weights in extended range. Add mirrored analytic quadratic
  cases and include them in the replayable preprocessing campaign.

- Avoid rescanning parsed raw CSV datasets after validating every header and
  numeric row. Preserve minimum sample counts and test invalid values in every
  signal column. Extend file benchmarks and fixture generation to processed CSV
  with analytic velocity and acceleration; retain its final validation after
  paired measurements found slower reads when it was omitted.

- Read raw CSV once into growing column-oriented chunks, then release each
  consumed field while assembling the owning output matrices. Avoid retaining
  the complete flat parse buffer beside all outputs. Preserve numeric grammar,
  row/joint order and validation, with large bitwise round-trip/late-error tests.
  Add synthetic file generation and read/RSS benchmarks; bounded chunks trade
  additional allocation calls for reduced peak memory.

- Reuse bounded thread-local Pinocchio Data in the individual-sample regressor,
  preserving independent concurrent calls and owning returned matrices. Track
  shared ownership as well as model address so a reused address rebuilds scratch.
  Batch calls keep their independent local workspace and validate samples/blocks
  once through the shared assembler.

- Core 0.1.1 and Pinocchio backend 0.2.1 share an internal regression assembly
  helper. Consume the backend's current regressor directly into the owning
  problem before reusing scratch, removing one temporary matrix per batch sample.
  Validate parameter names without an extra copy, preserve column/sample order,
  and check allocation dimensions before resizing. Public object layouts and
  interfaces are unchanged; the backend build requires core >=0.1.1.

- Include a package-local recorder license so a standalone source distribution
  can build a wheel without files from its parent repository. Update extracted
  recorder maintainer metadata to RobotIdentification contributors.

- Add opt-in analytic solver campaigns and an independent installed-recorder load
  harness with explicit delivery counts, row/digest checks and retained logs.

- Pin typing_extensions alongside jsonschema in developer and ROS CI setup.
  Document isolated Python environments and validate the pinned versions instead
  of relying on an unrelated user-site installation.

- Add standalone Release benchmarks for robust/ridge solves, harmonic evaluation,
  and optional Pinocchio dataset assembly. Record scoped C/C++ allocation counts
  and five-run medians, with explicit synthetic and shared-library limitations.

- Validate named Pinocchio batch coordinates before constructing workspace data.
  Require scalar joints and names in model q/v order; reject wrong or permuted
  names. Retain the positional single-sample interface for fixed-base multi-DoF
  joint models.

- Recorder 0.2.0 adds a positive `qos_depth` setting, retaining default depth 10.
  Validate it before creating files, exercise a configured depth in installed
  recording smoke tests, and declare Python >=3.10 in package metadata.

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

- Provide the standalone core, optional Pinocchio regressor, ROS 2 recorder,
  schemas, tests, and offline artifact tools.
- Preserve the existing public C++ targets, dataset formats, and artifact schemas.
  Core remains at 0.1.0. Recorder 0.1.1 explicitly declares
  string-array joint names so ROS Humble accepts configured joint ordering, and
  closes its context idempotently after SIGINT.
- Publish offline inspection and artifact commands under the new
  `robot_identification_tools` 0.1.0 package with stable Python imports,
  `ros2 run` package selection, and console-script names.
- Add independent validation, Python packaging, formatting, schema checks, and
  standalone/ROS 2 CI definitions, including an installed-recorder integration
  check with synthetic standard JointState messages.
