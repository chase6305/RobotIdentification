# MuJoCo arm identification

This experiment uses the left arm of EngineAI PM01 (5 joints, fixed torso, fixed elbow end retained) and the left arm of OpenArm (7 joints, fixed base, detachable gripper omitted). The runner reads the two URDFs from a local [HumanoidAssets checkout](https://github.com/chase6305/HumanoidAssets/tree/067393d2b63b7ffa5b022e847ef2f4436799ad70) and generates mesh-free URDFs in the output directory. The extracted models preserve the joint frames, axes, limits, and moving-link inertias. There are no contacts, passive joint forces, or drive dynamics in the experiment.

## Run

First build and install the core and Pinocchio backend. See the [standalone validation and installation guide](../../README_EN.md#standalone-build). Add that installation and the native Pinocchio dependencies to `CMAKE_PREFIX_PATH`.

```bash
cmake -S benchmarks/arm_identification -B build/arm-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build/arm-bench --parallel 2
python3 -m pip install 'mujoco==3.13.0' 'numpy==1.26.4'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 benchmarks/arm_identification/run_experiment.py \
  --assets-root ../HumanoidAssets --executable build/arm-bench/arm_identification \
  --output build/arm-experiment
```

Run with `--torque-noise-nm 0 --outlier-nm 0` and a new output directory for the clean control. Use `--payload-kg 0` for a matched-model control, or `--robot pm01` / `--robot openarm` for a single case. The GitHub Actions workflow builds against Pinocchio 2.7.0 and checks both pinned assets with and without torque contamination. It uploads both reports.

## Protocol

1. The Pinocchio rigid-body regressor builds the base-parameter basis from 192 deterministic calibration states. These states are used only to learn linear dependencies in the regressor. The existing Fourier optimizer improves the mean log information score in that fixed basis under position, velocity, acceleration, and jerk limits. The position interval has a 5% margin at each end and the speed cap is the smaller of 80% of the URDF limit and 4 rad/s. The acceleration and jerk caps are 25 rad/s² and 160 rad/s³.
2. Each training trajectory lasts 8 s and has 401 samples with analytical position, velocity, and acceleration. The holdout has 401 samples with different offsets, phases, amplitudes, and frequency over 6.25 s. MuJoCo evaluates inverse dynamics at the prescribed states with gravity `(0, 0, -9.81) m/s²`. The experiment rejects active constraints and sampled rigid-body torques outside URDF effort limits. It evaluates no actuator saturation and applies no torque clipping. See the [MuJoCo inverse-dynamics documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#inverse-dynamics).
3. The truth model includes a fixed 0.35 kg box with its center displaced from the last moving link. For the default noisy experiment, each training torque gets Gaussian noise with 0.02 N·m standard deviation and a 1% chance of an additional signed 0.75 N·m impulse. The baseline and optimized trajectories share the same noise realization by sample index and joint. Holdout labels remain clean. OLS and Huber (threshold 0.06 N·m) fit the loaded-arm inertial combinations using the nominal arm's kinematics. The known payload is used to construct the simulator and the oracle check, and never enters the fitted regression problem.
4. The fit solves `tau = Y_base(q, dq, ddq) beta`, where `beta = full_to_base * theta` and `theta` stacks the Pinocchio rigid-body coordinates. The oracle limits the maximum MuJoCo versus full Pinocchio torque error to `1e-7 N·m`, allowing for the URDF importer's conversion of full inertia tensors to principal axes. It independently limits the change introduced by reducing the Pinocchio regressor to `1e-8 N·m`. The reported coordinates describe observable combinations that include the payload. Individual masses, centers of mass, and inertia tensors cannot be recovered uniquely from these data.

`report.json` records source URDF hashes, tool versions, joint order, design scores, ranks, condition numbers, effort fractions, and holdout errors in N·m. It also includes the zero-based selected regressor columns, normalization scales, full-to-base mapping, and fitted and oracle base coordinates. Full coordinates repeat the Pinocchio order `mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz` for each joint, with inertia about the corresponding joint-frame origin. The per-arm directories contain both temporary models and processed CSVs matching the core dataset contract. The `*_states.csv` files are kinematic inputs with placeholder zero torque. The `train_*.csv` and `holdout.csv` files carry labels.

These results characterize rigid-body inverse dynamics along prescribed trajectories in simulation. They do not establish hardware identification accuracy or torque-controlled tracking performance.
