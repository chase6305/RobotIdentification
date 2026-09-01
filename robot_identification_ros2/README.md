# robot_identification_ros2

ROS 2 recording/session utilities for the middleware-independent `robot_identification` library.

The first recorder consumes a generic `sensor_msgs/msg/JointState`, treats `position` as `q` and `effort` as the measured torque selected by the hardware adapter, reorders by stable joint name, timestamps each callback with the process monotonic clock, and writes the raw CSV contract accepted by `ReadRawDatasetCsv()`.

```bash
ros2 run robot_identification_ros2 joint_state_recorder --ros-args \
  -p input_topic:=/robot_a/feedback/joint \
  -p output_path:=/var/lib/robot-identification/session-001.csv \
  -p joint_names:="[joint_1,joint_2]" \
  -p session_id:=session-001 \
  -p model_digest:=sha256-of-model \
  -p torque_source:=calibrated_joint_torque
```

Stop with Ctrl-C. The recorder writes data and metadata through separate
`.partial` files, fsyncs each file, and only then atomically promotes it. It
refuses to overwrite any existing dataset, partial file, or metadata. A
zero-sample run retains the data partial artifact and does not claim success.

This node never commands motion and cannot establish that a trajectory is safe or sufficiently exciting. `JointState.effort` provenance is adapter-specific and must be documented in the session record.
