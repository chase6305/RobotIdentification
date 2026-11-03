# RobotIdentification

[简体中文](README.md)

Robot dynamics parameter identification: named joint datasets, preprocessing, regression, diagnostics, physical validation, and immutable candidate artifacts. The C++17/Eigen core builds independently; Pinocchio and ROS 2 are optional adapters.

| Package | Responsibility | Additional dependency |
|---|---|---|
| `robot_identification` | Arbitrary-DoF datasets, CSV, preprocessing, weighted/ridge/Huber solving, observability, and physical checks | Eigen |
| `robot_identification_pinocchio` | Fixed-base URDF rigid-body regressor | Pinocchio |
| `robot_identification_ros2` | Raw `JointState.position/effort` recording and session metadata | ROS 2 |
| `robot_identification_tools` | Streaming CSV inspection and candidate/review/rollback artifacts | Python standard library |

Recorders and model adapters depend on the identification core. No RobotServer, controller, or vendor SDK is required.

For complete Pinocchio datasets, `backend.BuildProblem(dataset)` reuses call-local scratch. Individual samples and generic backend composition retain `Compute`; see the [Pinocchio backend](robot_identification_pinocchio/README.md).

Core 0.2.14 provides QR base parameters, a full-to-base mapping and constrained Fourier excitation design that compose with the Pinocchio backend. The [end-to-end example](robot_identification/examples/identification_design_example.cpp) covers design, fitting and independent validation; [algorithm and reference notes](docs/parameter-identification.md#参考项目评估与本次实现) describe equations, numerical assumptions and limits. Public interfaces remain compatible; the Pinocchio backend is 0.2.2.

The [actuator-network assessment](docs/parameter-identification.md#执行器动态网络方向) evaluates `sunzhon/actuator_net`; [semi-implicit hybrid identification](docs/parameter-identification.md#半隐式步进与物理学习混合辨识) adds an optional synthetic prototype and documents command/state histories, label units and independent trajectory validation.

## Standalone build

The Ubuntu venv recipe requires `python3-venv`.

Requires CMake 3.15+, C++17, and Eigen 3.3.1+. Full validation also requires Python 3.10+ and `jsonschema==4.23.0`.

```bash
cmake -S robot_identification -B build/core -DBUILD_TESTING=ON \
  -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build build/core --parallel 2
cmake -E chdir build/core ctest --output-on-failure
cmake --install build/core --prefix "$PWD/build/install"
./build/core/generic_regressor_example
./build/core/identification_design_example
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
./scripts/test.sh
./scripts/run_identification_example.sh
```

Installed C++ consumers use `find_package(robot_identification REQUIRED)` and `robot_identification::robot_identification`.

With a native Pinocchio development environment, run `./scripts/test.sh build/with-pinocchio --with-pinocchio` to validate the optional backend and its installed consumers. Add dependency prefixes to `CMAKE_PREFIX_PATH` and shared-library directories to the loader path.

Reproduce synthetic solver and Pinocchio assembly measurements with the [standalone benchmarks](benchmarks/README.md).

## Offline tools

```bash
python3 -m pip install ./robot_identification_tools
dataset_inspect session.csv --json
identification_artifact --help
```

See the [identification contract](docs/parameter-identification.md) and [tool reference](robot_identification_tools/README.md) for formats, review, and rollback semantics.

## Optional ROS 2 workspace

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src --rosdistro humble -y
./scripts/test_ros2.sh
source build/ros2/install/setup.bash
ros2 run robot_identification_ros2 joint_state_recorder --ros-args \
  -p input_topic:=/joint_states -p output_path:=/tmp/session-001.csv
```

The recorder consumes standard `sensor_msgs/JointState` messages from RobotServer or another robot system. See the [recorder reference](robot_identification_ros2/README.md) for sample and file-recovery contracts.

Run `python3 utils/ros2_recording_smoke.py` to exercise the installed recorder with synthetic messages, checking joint order, dataset hashes, and shutdown cleanup.

## Migration from RobotServer

Extracted from RobotServer commit `a6c70ec`, preserving relevant Git history and the MIT license. The three existing identification package names, C++ namespaces, CMake targets, and JSON/CSV contracts remain compatible. Python imports move from `robot_server_tools` to `robot_identification_tools`; the corresponding `ros2 run` package also changes. The `dataset_inspect` and `identification_artifact` command names remain available. Runtime state monitoring, control, and device commissioning stay in RobotServer.

Identification produces candidate artifacts. Applying parameters requires a separate review, validation, and rollback process. Synthetic examples provide no hardware parameter sets or timing guarantees.

See the [2026-10-28 optimization record](docs/optimization-2026-10-28.md) for numeric fixes, synthetic comparisons and sustained recording evidence, with reproducible programs in the [benchmark guide](benchmarks/README.md).
See the [second session](docs/optimization-2026-10-28-round-2.md) for workspace reuse, CSV peak storage and subsequent validation.
