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

`dataset_inspect` 逐行累计统计，额外内存只随关节数增长，不随录制时长增长。即使各时间戳有限，溢出的采样间隔、总时长或平均采样率也会被拒绝。

CSV summaries accumulate one row at a time, using additional memory proportional to joint count, independent of recording length. Non-finite intervals, duration, or mean sample rate are rejected even when the individual timestamps are finite.

CSV 字段按逗号直接分隔，不解释引号或转义。数字支持十进制与 `0x` 十六进制格式，以及行内的前导 ASCII 空白；下划线、非 ASCII 数字、带引号的数字、尾随空白、非有限数和下溢成零的非零值都会被拒绝。可表示的次正规数仍然有效。空行被跳过，但原始字节仍全部参与摘要计算，错误位置按实际文件行列报告。`dataset_inspect` 命令行遇到校验或文件读取失败时返回状态码 2，并输出简短错误。

CSV fields are literal comma-separated text without quoting or escaping.
Numbers accept decimal and `0x` hexadecimal syntax plus leading ASCII whitespace
within a line. Underscores, non-ASCII digits, quoted numbers, trailing whitespace,
non-finite values, and nonzero values that underflow to zero are rejected;
representable subnormal values remain valid. Blank lines are skipped while their
bytes still contribute to the digest. Diagnostics use physical file lines and
columns. The `dataset_inspect` CLI reports validation/read failures with a concise
error and exit status 2.

候选结果创建通过 `inspect_dataset_with_digest()` 在一次流式读取中同时校验 CSV 和计算 SHA-256，摘要保留原始换行与 UTF-8 字节，不会在校验后重新打开数据集。参数和诊断元数据无效时，会在扫描数据集前报错。

Candidate creation uses `inspect_dataset_with_digest()` to validate the CSV and hash the same stream in one pass, preserving original newline and UTF-8 bytes without reopening the dataset after validation. Invalid parameters or diagnostic metadata are rejected before scanning the recording.

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

JSON 输入拒绝重复字段（包括嵌套字段和转义后重名）、`NaN`、`Infinity` 及溢出的浮点数。创建候选与批准评审共用诊断校验；未声明的诊断字段会报错，不再静默丢弃。JSON 整数形式的 `row_count`、`rank` 保留原始精度，参数值仍按有限浮点数处理。

批准前还会检查候选的数据摘要、模型标识、字段完整性、参数名称与值的对应关系，以及 Schema 中的 100000 个参数上限。回滚清单必须包含配置路径及摘要、profile 和带时区的有效创建时间。缺失字段、未知字段和用布尔值充当数字均会被拒绝。`rejected` 仍可记录字段不完整或诊断不合格的候选，但输入必须是无重复字段的有效 JSON，且保留正确的 `kind` 与 `status`。

JSON inputs reject duplicate keys (including nested or escaped duplicates),
nonstandard `NaN`/`Infinity` values, and floating-point overflow. Candidate
creation and approval share diagnostic validation; unknown diagnostic fields
now fail instead of being silently discarded. Integer JSON tokens for
`row_count` and `rank` retain their exact values, while parameter values must
fit finite floating-point numbers.

Approval checks required and optional candidate fields, dataset digest, model
identity, matching parameter names/values, and the schema's 100000-parameter
limit. Rollback manifests must include their configuration path/digest, profile,
and a valid creation timestamp with a timezone. Missing or unknown fields and
booleans used as numbers are rejected. A `rejected` review can still document
an incomplete or numerically unqualified candidate without a rollback manifest;
the input must remain unambiguous JSON with the candidate `kind` and `status`.

命令行遇到无效输入或文件发布失败时，输出简短错误并返回状态码 2；Python API 仍抛出对应异常。

The artifact CLI reports validation or file-publication failures with a concise
error and exit status 2. The Python API continues to raise the underlying exception.

离线 artifact 在同一目录写入并同步临时文件后，通过不覆盖目标的原子硬链接发布；即使其他进程抢先创建目标，本次写入也会失败并保留对方文件。失败时仅清理本次独占创建的临时文件，已有 `.partial` 文件仍会阻止重试。输出位置的悬空符号链接会被拒绝。评审摘要直接绑定实际校验的 candidate/rollback 字节；文件随后变化时，摘要不再匹配该文件。

Offline artifacts publish a flushed temporary file through an atomic, non-overwriting hard link in the same directory. A concurrent destination is preserved and causes publication to fail. Failed attempts clean up their own exclusively created temporary files; existing `.partial` files still block retries. Dangling output symlinks are rejected. Review digests bind the exact candidate/rollback bytes that were validated, so subsequent file changes no longer match the recorded digest.
