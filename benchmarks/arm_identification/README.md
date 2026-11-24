# MuJoCo arm identification

This experiment uses the left arm of EngineAI PM01 (5 joints, fixed torso, fixed elbow end retained) and the left arm of OpenArm (7 joints, fixed base, detachable gripper omitted). The runner reads the two URDFs from a local [HumanoidAssets checkout](https://github.com/chase6305/HumanoidAssets/tree/067393d2b63b7ffa5b022e847ef2f4436799ad70) and generates mesh-free URDFs in the output directory. The extracted models preserve the joint frames, axes, limits, and moving-link inertias. The default plant uses rigid-body dynamics in free space.

## Run

The launcher builds the C++ tools and creates an isolated Python environment using the pinned wheels for Pinocchio 4.1.0 and MuJoCo 3.13.0. On Ubuntu it needs a C++17 compiler, CMake, Eigen 3.4 (`libeigen3-dev`), and Python 3.10 with `venv`.

```bash
./scripts/run_arm_identification.sh --demo --output build/arms-demo
```

The demo prepares both arms, generates reference points, captures raw encoder and torque data in MuJoCo, fits with periodic Fourier derivatives, replays 16 validation windows, and exports the identified payload models. With the default build directory, native build output is saved to `build/arm-identification/build.log`. The runner prints the results and the path to `report.json`.

To run forward acquisition using the simulated derivative states, use the simpler rollout command.

```bash
./scripts/run_arm_identification.sh --assets-root ../HumanoidAssets \
  --acquisition rollout --output build/arm-rollout
```

Use `--acquisition inverse` for prescribed-state inverse dynamics, `--torque-noise-nm 0 --outlier-nm 0` for the clean control, or `--robot pm01` / `--robot openarm` for a single arm. The default asset location is the sibling `HumanoidAssets` checkout. The launcher reuses its build directory on later runs. Each experiment uses a new output directory. See the [arm identification workflow](../../.github/workflows/arm-identification-ci.yml).

Export automatic points and their analytic trajectory coefficients before running the simulator.

```bash
./scripts/run_arm_identification.sh --design-only --sample-count 401 --output build/arm-points
```

Use `--training-period-s` and `--holdout-period-s` to set each cycle's duration. The design rescales the initial amplitude when a shorter period would exceed the configured velocity, acceleration, or jerk bounds. The generated CSVs and Fourier coefficients use the requested period and the report records the kinematic bounds.

Run the forward simulation and fit using the recorder-format position and torque measurements.

```bash
./scripts/run_arm_identification.sh --acquisition rollout --fit-input raw \
  --raw-estimator local-cubic --raw-window 9 --timestamp-jitter-ms 2 --output build/arm-raw
```

The local cubic estimator fits position against centered, scaled timestamps in each window and evaluates its first and second derivatives at the center. A nine-point window discards four points at either end and preserves each retained timestamp and torque. `--raw-estimator moving-average` selects the core's centered smoothing and finite differences. The independent holdout uses clean simulated derivatives in both cases.

`--raw-estimator periodic-fourier --raw-harmonics 7` fits the measured joint positions over one full capture period, then evaluates the derivatives analytically at the actual timestamps. It retains all position and torque samples and uses the first and last timestamps to determine the period. Use this estimator with steady periodic captures.

`--timestamp-jitter-ms` shifts the capture times within the requested bound on the MuJoCo integration grid. The CSV records the actual times. The two training trajectories share the same sample offsets, and the holdout uses an independent seed.

`--capture-dropout` sets the probability of losing each interior training sample, preserving the period endpoints. `--encoder-resolution-rad` rounds the raw positions to encoder increments. The training trajectories share the seeded dropout mask. These controls keep positions and torques synchronized at every retained timestamp.

Add `--actuator-scale 1` to the command to test synthetic reflected rotor inertia and linear viscous friction. The plant uses per-joint armature `0.04 r kg·m²` and damping `(0.02 + 0.05 r) N·m·s/rad`, where `r` is each joint's URDF effort limit divided by the largest effort limit on that arm. The scale multiplies both terms. In this mode, `truth.mjb` contains the compiled MuJoCo plant. The fit adds columns for each joint's acceleration and velocity, and the report compares against a rigid-only model fitted to the same samples. The base-parameter rank determines which combinations can be identified. See the [MuJoCo joint model reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint).

The actuator report recovers individually observable coefficients by solving for each full coordinate as a linear combination of base coordinates. All viscous coefficients are observable on both arms, and the first two rotor inertias appear as `null` individual estimates. The projected candidate uses the final Huber weighted least-squares metric to make the observable passive coefficients nonnegative within numerical tolerance. Its holdout and mass-matrix metrics are reported alongside the unconstrained fit.

For a complete example with drive dynamics, raw encoder measurements, and forward replay:

```bash
./scripts/run_arm_identification.sh --acquisition rollout --actuator-scale 1 --replay \
  --fit-input raw --raw-estimator periodic-fourier --timestamp-jitter-ms 2 \
  --output build/arm-drive-replay
```

Add `--estimate-payload` when the arm and drive dynamics are known. The solver subtracts that prior model's torque and identifies the 10 inertial coefficients of the terminal payload. It preserves the fitted mass and center of mass while projecting the centered second-moment matrix onto positive eigenvalues when needed. The resulting `identified_payload.urdf` and `identified_payload.mjb` can be loaded again, and the runner checks the binary model's holdout torque against the native regression prediction.

An existing core and Pinocchio installation can also build the benchmark with `cmake -S benchmarks/arm_identification -B build/arm-bench` and `cmake --build build/arm-bench --parallel 2`. In that case, run `run_experiment.py` with `--executable build/arm-bench/arm_identification` and a Python environment containing MuJoCo and NumPy. See the [standalone installation guide](../../README_EN.md#standalone-build).

## Protocol

1. The Pinocchio rigid-body regressor builds the base-parameter basis from 192 deterministic calibration states. These states are used only to learn linear dependencies in the regressor. The existing Fourier optimizer improves the mean log information score in that fixed basis under position, velocity, acceleration, and jerk limits. The position interval has a 5% margin at each end and the speed cap is the smaller of 80% of the URDF limit and 4 rad/s. The acceleration and jerk caps are 25 rad/s² and 160 rad/s³.
2. The benchmark automatically generates 401 joint references for each 8 s training trajectory and an independent 401-point holdout over 6.25 s. The Fourier coefficients let the rollout evaluate the same continuous reference at every integration step. With `--acquisition rollout`, a computed-torque controller based on the unloaded model tracks the reference while the loaded MuJoCo plant integrates with steps of at most 1 ms. It tracks the periodic reference for 3 s before recording, limits applied torque to each joint's URDF effort, and records position, velocity, acceleration, and applied torque at the same instant. The acceleration feedback gains are 10,000 s⁻² for position error and 200 s⁻¹ for velocity error. The plant and replay both use semi-implicit Euler with the viscous force evaluated explicitly. With `--acquisition inverse`, MuJoCo evaluates the torque at each analytical reference state and rejects torques above the URDF effort. Both modes reject active constraints and use gravity `(0, 0, -9.81) m/s²`. See the [MuJoCo stepping and inverse-dynamics documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html).
3. The truth model includes a fixed 0.35 kg box with its center displaced from the last moving link. For the default noisy experiment, each training torque gets Gaussian noise with 0.02 N·m standard deviation and a 1% chance of an additional signed 0.75 N·m impulse. The baseline and optimized trajectories share the same noise realization by sample index and joint. Holdout labels remain clean. OLS and Huber (threshold 0.06 N·m) fit the loaded-arm inertial combinations using the nominal arm's kinematics. The known payload is used to construct the simulator and the oracle check, and never enters the fitted regression problem.
4. The fit solves `tau = Y_base(q, dq, ddq) beta`, where `beta = full_to_base * theta` and `theta` stacks the Pinocchio rigid-body coordinates. The oracle limits the maximum MuJoCo versus full Pinocchio torque error to `1e-7 N·m`, allowing for the URDF importer's conversion of full inertia tensors to principal axes. It independently limits the change introduced by reducing the Pinocchio regressor to `1e-8 N·m`. The reported coordinates describe observable combinations that include the payload. Individual masses, centers of mass, and inertia tensors cannot be recovered uniquely from these data.

`report.json` records source URDF hashes, tool versions, joint order, design scores, ranks, condition numbers, effort fractions, and holdout errors in N·m. It also includes the zero-based selected regressor columns, normalization scales, full-to-base mapping, and fitted and oracle base coordinates. Full coordinates repeat the Pinocchio order `mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz` for each joint, with inertia about the corresponding joint-frame origin. The per-arm directories contain temporary models and CSVs matching the core contracts. The `*_states.csv` files contain the automatic references with placeholder zero torque. The `*_raw.csv` files contain timestamps, joint positions, and torque. The processed `train_baseline.csv`, `train_optimized.csv`, and `holdout.csv` files also contain velocity and acceleration. Raw training positions include paired encoder noise with a default standard deviation of `5e-5 rad`.

`--huber-auto` sets each training trajectory's threshold to `1.345 × 1.4826 × MAD` of its OLS torque residuals, with a numerical floor for clean inputs. The fitted thresholds appear in the report.

The forward check reconstructs the joint-space mass matrix from the acceleration-linear part of the regressor and checks its symmetry and smallest eigenvalue at up to 81 calibration poses and 81 holdout poses. It predicts acceleration by solving `M(q) ddq = tau - bias(q, dq)` at the holdout poses. The acceleration RMSE is `null` when any tested holdout mass matrix fails validation. `holdout_predicted_optimized_huber.csv` contains the optimized Huber torque predictions on the exact states recorded in `holdout.csv`.

With `--replay`, `holdout_dense.csv` records the states and torques at every integration step. The nominal and fitted models each start at the recorded state of 16 independent 100 ms windows and integrate under the same applied torques. The report counts completed windows and gives position errors when all windows complete.

For paired preprocessing comparisons on an existing `--fit-input simulator` run, replay the same raw samples through each window.

```bash
benchmark_dir=build/arm-identification
"$benchmark_dir/venv/bin/python" benchmarks/arm_identification/compare_raw.py \
  --executable "$benchmark_dir/benchmark/arm_identification" \
  --output build/arm-window-comparison.csv build/arm-rollout
```

## Reference results

Prescribed-state inverse dynamics holdout torque RMSE in N·m, with fitted results reported as medians over the CI noise seeds `20261124, 11, 29, 47, 71`.

| Left arm | Base / full coordinates | Unloaded nominal | Baseline OLS | Baseline Huber | Optimized OLS | Optimized Huber |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PM01 | 33 / 50 | 0.41340 | 0.04383 | 0.01110 | 0.01216 | 0.00390 |
| OpenArm | 43 / 70 | 0.23947 | 0.03789 | 0.01160 | 0.01109 | 0.00337 |

In the clean control, the relative error against the known base coordinates is about `1.04e-8` for PM01 and `6.54e-9` for OpenArm. These figures characterize rigid-body inverse dynamics along prescribed trajectories in simulation.
