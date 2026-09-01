# 参数辨识 / Parameter Identification

## 中文

### 范围

仓库将原有的动力学参数辨识能力迁移为机器人无关的分层设计：

```text
设备适配器或日志
        │ 原始 q、tau、时间戳及上下文
        ▼
ROS 2 采集/会话层（非实时）
        │ 不可变原始数据集
        ▼
robot_identification（不依赖中间件）
        │ 预处理、Y 矩阵、求解、验证
        ▼
候选参数 artifact
        │ 人工评审、低速验证、回滚检查
        ▼
机器人 profile、URDF 或模型后端
```

当前交付包括中间件无关核心和最小 ROS 2 recorder。核心提供数据集、原始/处理后 CSV、预处理、回归器接口、任意 DoF 驱动惯量/黏性摩擦/库仑摩擦/偏置列扩展、加权/ridge/Huber 求解、可观测性诊断、协方差、独立集指标和质量矩阵正定性验证。`robot_identification_ros2` 从具名 `JointState.position/effort` 采集原始 CSV，按稳定关节名重排并使用进程单调时钟；它不发送运动命令。ROS 2 仅承担采集和会话编排，不进入算法库，也不要求算法测试启动 ROS graph。

可选的 `robot_identification_pinocchio` 使用 Pinocchio 官方 joint-torque regressor 生成刚体参数列，并保持 Pinocchio 依赖不进入核心包。当前明确支持 `nq == nv` 的固定基模型；浮动基模型需要版本化扩展数据契约。

`HarmonicTrajectory` 使用任意 DoF、任意谐波数的正余弦系数解析生成位置、速度、加速度和 jerk。`ValidateExcitationTrajectory()` 先以各谐波幅值之和计算保守解析包络，再依据所选机器人 profile 的逐关节限制采样完整轨迹。验证通过不授权节点发送运动；实际执行层仍需碰撞检查、watchdog 和状态联锁。

### 与旧实现的对应关系

| 旧能力 | 新设计 |
|---|---|
| 固定两个 7DoF 手臂 | `Dataset` 和 `RegressorBackend` 支持任意 DoF/运动组 |
| 固定 71 维最小参数 | 后端声明参数数量与名称 |
| GUI 内部构造矩阵和求解 | 核心库提供可测试的 `BuildRegressionProblem()` 与 `Solve()` |
| 普通采样直接按固定 10 ms 处理 | 数据集要求真实、严格递增时间戳 |
| 三阶 Butterworth + 差分 | 提供独立的位置/力矩对称零相位平滑窗口和非均匀时间三点差分；力矩默认不平滑，后续可增加版本化 Butterworth 策略 |
| LMI/DSDP 物理约束 | 核心提供通用质量矩阵验证；约束求解器作为可选后端，不成为基础依赖 |
| 固定型号符号代码注册表 | Pinocchio、生成代码或厂商模型通过回归器后端接入 |
| 正则表达式覆盖运行 YAML | 输出候选 artifact，禁止自动覆盖运行配置 |

原实现中的 DH 转换、激励轨迹优化、完整刚体参数 LMI 和符号代码生成具有较强模型假设，将在回归器/激励器扩展层迁移，不能重新引入具体机器人名称、左右臂或固定自由度。

### 数据契约

处理后 CSV 表头固定为：

```text
time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
```

要求：

- 关节名称非空、唯一且顺序稳定；
- 时间戳有限并严格递增；
- `q/dq/ddq/tau` 维度完全相同且全部有限；
- 力矩必须标明来源，例如驱动电流估算、关节传感器或外部传感器映射；
- 原始数据、处理参数和处理后数据分别保存；
- 使用原始 CSV 的 SHA-256 将 session metadata、候选结果和实际输入绑定；
- 接触、饱和、watchdog、错误和丢帧区间不得作为正常辨识样本。

### 数值诊断

求解结果必须至少检查：

- `rank == parameter_count`，否则存在不可辨识方向；
- 条件数不能只记录而不评审；
- 训练残差、独立验证残差和逐关节峰值误差；
- 协方差或等价的不确定性指标；
- 鲁棒权重是否大量下降，大量 outlier 通常意味着数据或模型错误；
- 恢复出的质量矩阵在覆盖工作空间内对称正定；
- 参数量纲、符号和物理范围。

ridge 可以改善数值稳定性，但不能把秩亏伪装成充分激励。Huber IRLS 用于降低少量异常点的影响，不能代替接触/通信异常剔除。

### 推荐迁移顺序

1. 工具质量、质心和力矩 bias；
2. 单关节黏性/库仑摩擦；
3. 使用 Pinocchio 回归器并进行基参数/可观测性分析；
4. 在已有限位预检查的谐波轨迹上增加可观测性优化和独立验证轨迹；
5. 可选物理一致性约束求解器；
6. 扩展现有 ROS 2 recorder，增加显式 session action、激励轨迹联锁和可选 GUI；
7. 候选 artifact 的签名、审批和导入工具。

### 安全边界

辨识只生成候选结果，不能在线热更新力控模型。导入前必须保存旧参数、数据摘要、代码版本、机器人 profile、工具状态、拟合指标和验证结论。任何 URDF、传动、固件、工具、采样频率或力矩来源变化都要求重新评审。真机验证从位置模式和低速度开始，再逐步验证重力补偿、力矩模式、导纳与阻抗。

安装空间中的 session、result、review 和 rollback Schema 分别约束采集元数据、候选结果、人工决定和回滚清单。候选结果 `status` 固定为 `candidate`；批准只能存在于独立 review artifact，且必须绑定满秩、质量矩阵检查通过的候选文件和回滚清单 SHA-256。工具不会写入运行配置。

## English

The migration separates robot-specific regressor construction from reusable preprocessing, solving, and validation. `robot_identification` is a plain CMake package and therefore works both standalone and in a colcon workspace. It assumes neither left/right arms, seven joints, nor a fixed base-parameter dimension.

Processed CSV uses `time_s`, followed by ordered `q/`, `dq/`, `ddq/`, and `tau/` columns for every joint. Names must be unique, timestamps strictly increasing, and all values finite. Keep raw recordings immutable and record torque provenance, preprocessing settings, model digest, software revision, exclusions, and operating context.

Every accepted result needs full numerical rank, reviewed conditioning, uncertainty, per-joint training and holdout errors, robust-weight diagnostics, and physical validation such as symmetric positive-definite mass matrices over the intended workspace. Ridge and robust regression improve numerical behavior but do not repair insufficient excitation or a wrong model.

The `robot_identification_ros2` recorder consumes named `JointState.position/effort`, reorders a stable joint set, uses process-monotonic callback timestamps, and atomically finalizes the raw CSV without commanding motion. ROS 2 remains an orchestration layer and must never apply an identified model automatically. Produce a candidate artifact, review it offline, retain rollback parameters, and perform staged low-speed hardware validation before changing a runtime profile or URDF.
