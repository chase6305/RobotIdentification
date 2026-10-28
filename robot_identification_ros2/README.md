# robot_identification_ros2

ROS 2 recording/session utilities for the middleware-independent `robot_identification` library.

Version 0.1.1 declares `joint_names` as a string array and handles an already
stopped ROS context during shutdown. Omit joint names to learn the ordering
from the first valid sample, or set an explicit list of names.

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
`.partial` files and fsyncs both before publishing either one. Publication uses
exclusive hard links in the output directory, so the filesystem must support
hard links and directory fsync. Each file is published atomically without
overwriting an existing file or symlink, including artifacts created while the
recorder is running. Both partials are removed after the final links are synced.

The CSV and metadata are separate publications, not an atomic pair. If the
metadata link fails after the CSV link succeeds, the CSV and both partials remain
for inspection and the run reports an error. Consume a session only when both
final files exist and the metadata digest matches the CSV. A zero-sample run or
a failed/interrupted write retains its data partial and does not publish a
dataset. Flush, close, or publication failures still release the node and ROS
context.

Only the first valid sample determines an unconfigured joint order. Rejected
samples do not write headers or change that order, and serialized timestamps
must remain strictly increasing. Repeated input orders reuse their joint-index
mapping. SHA-256 is updated from the exact UTF-8 bytes written during recording,
so finalization does not reread the dataset.

Direct callers of the validation core receive rejection reasons for malformed
joint-name elements, a string used as the name sequence, and invalid or
out-of-range numeric fields. These rejections increment only the rejected-sample
count; subsequent valid samples can still be recorded. Actual storage failures
remain fatal to the session and retain partial artifacts.

The validation/storage core and adapter cleanup paths can be tested without ROS:

```bash
PYTHONPATH=robot_identification_ros2 \
  python3 -m unittest discover -s robot_identification_ros2/test
```

Run this from the repository root, or use `./scripts/test.sh`. Adapter
tests use ROS stubs and do not replace integration testing on a ROS graph.

This node never commands motion and cannot establish that a trajectory is safe or sufficiently exciting. `JointState.effort` provenance is adapter-specific and must be documented in the session record.

Version 0.2.0 adds `qos_depth`, a positive integer with the existing default of 10.
For example, pass `-p qos_depth:=1000` to absorb a longer publication burst.
Larger queues use more memory and can increase callback delay. Receipt timestamps
remain process-monotonic callback times; queue depth does not establish source
sampling times or loss-free delivery. Validate acquisition timing separately for
identification. Python packaging requires version 3.10 or newer.

0.2.0 新增正整数参数 `qos_depth`，默认仍为 10。高频录制可按实测情况配置，例如
`-p qos_depth:=1000`；更深的队列增加内存和回调延迟。时间戳继续使用回调时的进程
单调时钟，队列深度不保证原始采样时序或无丢包，辨识采集时应另行验证。
