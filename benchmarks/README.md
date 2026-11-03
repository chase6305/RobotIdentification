# Synthetic identification benchmarks / 合成辨识基准

These Linux/GNU-linker programs measure software behavior on synthetic data.
They do not establish hardware timing, trajectory safety or sufficient excitation.
合成基准用于复现软件性能与数值行为；不替代硬件验证，也不设置 CI 时序门槛。

## Build

Build both the library and consumers in Release mode with matching compiler,
headers and Eigen allocation ABI:

```bash
cmake -S robot_identification -B build/benchmark-core -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$PWD/build/benchmark-install"
cmake --build build/benchmark-core --parallel 2
cmake --install build/benchmark-core
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/benchmark-install"
cmake --build build/benchmarks --parallel 2
build/benchmarks/core_bench
```

Optional CMake switches:

| Switch | Enables | Additional requirement |
|---|---|---|
| `ROBOT_IDENTIFICATION_BENCH_STRESS_TARGETS=ON` | Analytic numerical/storage campaigns | Extended-range `long double` for wide-scale oracles |
| `ROBOT_IDENTIFICATION_BENCH_EXACT_DOT_REFERENCE=ON` | `parameter_dot_stress` | Boost headers; core >=0.2.0 |
| `ROBOT_IDENTIFICATION_BENCH_PINOCCHIO=ON` | `pinocchio_bench`, and `pinocchio_stress` when stress targets are enabled | Installed native Pinocchio backend |

For Pinocchio, first build and install the backend with the same Release and
allocation settings, adding native dependency prefixes to `CMAKE_PREFIX_PATH`:

```bash
cmake -S robot_identification_pinocchio -B build/benchmark-pin -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH="$PWD/build/benchmark-install" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/benchmark-install"
cmake --build build/benchmark-pin --parallel 2
cmake --install build/benchmark-pin
```

`COLCON_IGNORE` excludes these consumers from ROS workspace discovery. The core
and default benchmarks require neither ROS nor Pinocchio.

## Timing and allocation measurements

| Program | Timed operation | Reporting |
|---|---|---|
| `core_bench` | Huber/ridge solves on 1,500 rows and 12 parameters; seven-axis/20-harmonic trajectories | Warmed median of five repetitions; allocations per call |
| `csv_bench CSV [REPETITIONS] [raw\|processed]` | Read, checksum and destroy a complete dataset | Median milliseconds, allocations per read, process peak RSS |
| `design_bench CASE [ITERATIONS]` | One API batch; fixture construction excluded | JSON wall/process CPU microseconds per call and checksum |
| `parameter_bench CASE [ITERATIONS]` | Warmed `ToBaseParameters`; construction excluded | JSON wall/process CPU time, allocations and checksum |
| `pinocchio_bench` | Generic and named batch assembly of 1,000 samples on two/14-axis chains | Warmed median milliseconds and allocations per batch |

GNU linker wrappers count executable/static-library `malloc` and `calloc`;
C++ overrides count ordinary, array and aligned `new`. These disjoint counters
apply to the benchmark thread; direct C allocation inside shared libraries can
escape instrumentation. `design_bench` has no allocation instrumentation.

For comparisons, use identical harness sources, matched installed headers and
libraries, alternating fresh processes and an allowed CPU (`taskset -c CPU` if
needed). Preserve raw observations, compiler/build flags and dependency versions.
CPU and wall medians are separate statistics; shared-host scheduling, frequency,
filesystem cache and allocator state still affect results. Do not compare an
older API emulation with a current native API as though only internals changed.

### API cases

Use `parameter_bench` for mapping and `design_bench` for the other groups.

| Program/group | Cases |
|---|---|
| Parameter mapping | `map-scalar`, `map-unit`, `map-sparse`, `map-dense`, `map-dense-small`, `map-dense-wide`, `map-dense-large`, `map-dense-signed` |
| Basis reduction | `basis-scalar`, `basis-small`, `basis-batch`, `basis-full-rank`, `basis-zero-scalar`, `basis-zero-small`, `basis-zero-batch` |
| Dependency error | `reconstruction-scalar`, `reconstruction-pair`, `reconstruction-small`, `reconstruction-wide`, `reconstruction-large`, `reconstruction-254`, `reconstruction-255`, `reconstruction-256`, `reconstruction-511`, `reconstruction-batch` |
| Envelopes | `bounds-dense`, `bounds-sine`, `bounds-exact` |
| Sampling | `trajectory-owning`, `trajectory-reused`, `trajectory-validation` |
| Information/search | `quality-unit`, `quality-scaled`, `optimize`, `optimize-scaled` |

The mapping cases include one owning result-vector allocation. `map-unit` has
80 independent columns; `map-sparse` adds 80 zero columns. Dense shapes are
rank/columns 8/16 (small), 24/192 (wide), 80/160 (default), 160/320 (large) and
24/96 with alternating parameter signs (signed).

Basis cases check dependencies and construct reduced output; reconstruction
cases time `ReconstructionError` alone. The usual basis has 80 directions and
140 columns, with 14 rows (small) or 65,537 (batch); the full-rank case has 80
independent columns. Zero cases retain only zero-column dependencies. Scalar
cases isolate one-row/two-column inputs; pair and 254/255/256/511-row cases
exercise reconstruction block tails.

Envelope cases use both coefficient families, only sine, or exact dyadic 3-4-5
amplitudes. Exact boundaries can require more work than ordinary coefficients.
Sampling compares owning output with buffer reuse; validation checks 201 samples.
Search uses seven independent actuators and a 15-evaluation budget; scaled cases
fix parameter/noise scales. Core >=0.2.0 enables these APIs; `trajectory-reused`
uses `EvaluateInto` at >=0.2.4 and owning evaluation/assignment on older cores.
The Pinocchio batch comparison requires backend >=0.2.0.

```bash
build/benchmarks/parameter_bench map-dense 10000
build/benchmarks/design_bench reconstruction-255 2000
build/benchmarks/design_bench basis-batch 1
build/benchmarks/design_bench trajectory-reused 100000
build/benchmarks/design_bench trajectory-validation 1000
build/benchmarks/design_bench optimize-scaled 20
```

### CSV comparisons

```bash
python3 benchmarks/generate_raw_csv.py --samples 200000 --dof 14 \
  --output build/raw-fixture.csv
python3 benchmarks/generate_raw_csv.py --format processed --samples 200000 --dof 14 \
  --output build/processed-fixture.csv
build/benchmarks/csv_bench build/raw-fixture.csv 1
build/benchmarks/csv_bench build/processed-fixture.csv 5 processed
python3 benchmarks/compare_csv.py build/before/csv_bench build/after/csv_bench \
  build/raw-fixture.csv --rounds 4 --repetitions 5 --output build/csv-comparison.json
```

The generator refuses replacement and reports size/SHA-256. Processed fixtures
include analytic velocity and acceleration of the sine position. One repetition
skips warmup; multiple repetitions warm once. `peak_rss_kib` is Linux `getrusage`
for the whole process, including warmup/launcher history; compare RSS in fresh
processes. Chunked imports still retain storage proportional to final dataset size.

The comparison helper snapshots both executables, runs fresh copies at one path
in alternating order, checks dimensions/checksums, and hashes the fixture before
and after with bounded memory. Dependencies must remain discoverable after copying.
Add `--format processed` or `--cpu CPU` as needed. Reports must use new paths;
positive `reduction_percent` describes this sample, not a general speedup guarantee.

## Correctness campaigns

With stress targets enabled, `build/benchmarks/hybrid_rollout_demo` runs a scalar
offline prototype combining positive inertia, nonnegative damping, two fixed
tanh features of six causal history inputs, learned residual readout weights
and semi-implicit rollouts. It initializes with a backward-Euler momentum
balance, projects into fixture bounds, then refines a normalized trajectory
loss by bounded coordinate search.
The physics-only baseline is fitted on the same training sequences. An exact
held-torque solution generates the independent synthetic reference; holdout and
integration-substep checks must pass. Known PD gains/load anchor the torque scale.
Finite search budget, noise and discretization leave parameter bias: this is a
workflow demonstration, with no real-robot accuracy or parameter-uniqueness claim.

Enable stress targets, then run a program with an optional duration in seconds
(default 60; allowed 1–86,400). Fixed seeds produce reproducible cases; duration
and scheduling determine counts. Programs stop on a mismatch and report counts.

| Program | Reference and coverage |
|---|---|
| `solver_stress` | Orthogonal analytic systems; dimensions, rank, conditioning, weights and ridge |
| `solver_scale_stress` | Common weights through `1e±300`, design scales through `1e±130`, extended-range covariance |
| `csv_stress` | Bitwise raw round trips across chunk boundaries, signed zero, finite extremes, LF/CRLF/CR, late-error rejection and retained owning results |
| `preprocessing_stress` | Independent long-double Lagrange derivatives, nonuniform intervals, extreme scales, unusable-result rejection |
| `metrics_stress` | Independent extended-range RMS/absolute errors and normalized ratios, sparse subnormals, zero references, residual overflow |
| `robust_stress` | Scalar Huber score roots by bisection, weighted covariance, outliers, weight/design exponents ±1000/±450, one/two-iteration stops |
| `trajectory_stress` | Closed-form position through jerk and envelopes; phase/product underflow, cancellation, overflow, inactive harmonics |
| `design_stress` | Known-rank cosine bases, column units at binary exponents -300..300, weighted fits, independent holdout, analytic Gram eigenvalues and constrained search |
| `pinocchio_stress` | Four workers, shared backends, model-owner churn, scalar/translation joints, actuator composition and native RNEA |

```bash
build/benchmarks/solver_scale_stress 3600
build/benchmarks/csv_stress 3600
build/benchmarks/robust_stress 3600
build/benchmarks/design_stress 3600
```

Solver noise is bounded relative to the weakest nonzero singular direction.
Parameter error is `norm(actual-reference) / max(1, norm(reference))`; covariance
error uses stable Frobenius norms and denominator `max(1e-300, norm(reference))`.
Pinocchio uses the parameter-error convention, requires scalar named batches to
match generic assembly exactly, and rejects named multi-coordinate batches.

Preprocessing allowance is
`64 * epsilon(double) * sum(abs(w * input)) + sum(abs(w) * input_error) + 32 * denorm_min(double)`.
Acceleration propagates velocity allowances through the second derivative; this
handles nearly linear inputs and double quantization without a relative-only
acceleration test. Trajectory sample allowance is
`128 * epsilon(double) * sum(abs(contributions)) + 4 * denorm_min(double)`;
envelopes use `8 * epsilon(double)` times their magnitude plus the same subnormal
allowance. Ordinary phases use the API's double phase; subnormal phases use the
extended product. Nonfinite active phases and unrepresentable final samples reject.

Metrics form normalized RMS ratios before rounding the individual RMS values.
Huber fits must solve their reported weight set and retain/reduce the initial WLS
objective; only converged fits are compared with the Huber root. Fits consuming
the 200-iteration cap without convergence are counted separately. Design holdout
checks must reject a newly excited zero calibration column.

For replay, append a case index: `PROGRAM 10 INDEX`. Preprocessing accepts up to
100000000 and generates the RNG prefix before starting the timed campaign.
Metrics, trajectory and design accept up to 1000000000000 and seed each case
directly. Failure output identifies the case and relevant input.

### Exact parameter-dot reference

```bash
build/benchmarks/parameter_dot_stress 60 0
```

This separately enabled program uses Boost `cpp_int` dyadics built with `frexp`
and compares public API outputs with exact adjacent-double midpoints/tie parity.
It covers up to 257 columns, product/block cancellation, subnormal ties, exact
overflow/nonzero underflow, zero-column mappings and sampled caller rounding modes.
`rounding_samples` includes nearest and the three directed modes. Counts include
repeated boundary fixtures, not globally unique inputs. Keep corrected-cancellation
cases in correctness checks; an incorrect zero is not a useful timing baseline.
The core does not depend on Boost or this reference implementation.

## Recorder load test

After sourcing installed ROS packages:

```bash
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=78 python3 benchmarks/recorder_soak.py \
  --duration 3600 --rate 1000 --qos-depth 1000 --output build/soaks/recorder.json
```

The 14-axis producer periodically reverses joint order and sends NaN,
missing-effort and duplicate-name messages. Validation checks every joint/torque
row, strict ordering, metadata counts and raw-file SHA-256. Reports distinguish
missing valid/invalid deliveries; the default minimum is 95% for each class.
Use `--minimum-delivery-fraction 1` to require all sent messages. Queue depth
trades memory/latency against burst tolerance; timestamps remain process-monotonic.
Outputs must use a new `.json` path; CSV/logs remain beside it. At 1000 Hz, one hour
writes about 1.8 GB. Passing the default threshold can include transport losses.
