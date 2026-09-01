# robot_server_tools

机器人无关的只读诊断工具。工具通过命令行参数接收文件或 topic，不包含机器人型号、固定关节数或硬件写操作。

```bash
ros2 run robot_server_tools state_monitor --topic /feedback/robot_server_state
ros2 run robot_server_tools dataset_inspect session.csv --json
```

`state_monitor` validates the basic JSON state envelope before printing status.
`dataset_inspect` works without a ROS graph and strictly validates raw or
processed identification CSV headers, finite values, monotonic timestamps,
sample rate, timing spread, and peak absolute torque.

Motion commands and commissioning writes intentionally live outside this
package so installing diagnostics cannot grant a tool an accidental actuation
path.

Candidate parameters use an explicit, non-mutating workflow:

```bash
ros2 run robot_server_tools identification_artifact rollback \
  --configuration robot_config.yaml --profile default --output rollback.json
ros2 run robot_server_tools identification_artifact candidate \
  --dataset session.csv --model-digest MODEL_SHA --joints joint_1 joint_2 \
  --parameters parameters.json --diagnostics diagnostics.json --output candidate.json
ros2 run robot_server_tools identification_artifact review \
  --candidate candidate.json --decision approved --reviewer NAME \
  --reason "offline and low-speed validation passed" \
  --rollback rollback.json --output review.json
```

Approval requires full numerical rank, passed mass-matrix validation, and a
hash-bound rollback manifest. This command records a decision; it never applies
parameters to a live configuration.
