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

记录器在写入时增量计算 CSV 摘要，仅用有效样本确定关节顺序和表头。CSV 与元数据先分别写入并同步 `.partial`，再以不覆盖已有文件的硬链接逐个发布；两份文件不是一次原子事务。写入失败时保留临时数据，发布冲突时保留恢复文件并报错。使用会话前应确认两份正式文件齐全且摘要匹配，具体恢复语义见 [recorder README](../robot_identification_ros2/README.md)。

可选的 `robot_identification_pinocchio` 使用 Pinocchio 官方 joint-torque regressor 生成刚体参数列，并保持 Pinocchio 依赖不进入核心包。当前明确支持 `nq == nv` 的固定基模型；浮动基模型需要版本化扩展数据契约。

`HarmonicTrajectory` 使用任意 DoF、任意谐波数的正余弦系数解析生成位置、速度、加速度和 jerk。`ValidateExcitationTrajectory()` 先以各谐波幅值之和计算保守解析包络，再依据所选机器人 profile 的逐关节限制采样完整轨迹。验证通过不授权节点发送运动；实际执行层仍需碰撞检查、watchdog 和状态联锁。

对关节 j，系数使用与 q 相同的单位，基本频率 f 的单位为 Hz，时间 t 为秒：

```text
omega_h = 2*pi*f*h
q_j(t) = offset_j + sum_h(a_jh*sin(omega_h*t) + b_jh*cos(omega_h*t))
v_j(t) = sum_h(omega_h*(a_jh*cos(omega_h*t) - b_jh*sin(omega_h*t)))
a_j(t) = -sum_h(omega_h^2*(a_jh*sin(omega_h*t) + b_jh*cos(omega_h*t)))
jerk_j(t) = -sum_h(omega_h^3*(a_jh*cos(omega_h*t) - b_jh*sin(omega_h*t)))
```

导数单位依次为 q 单位/s、/s²、/s³；第 d 阶导数的解析包络为
`sum_h(hypot(a_jh, b_jh) * omega_h^d)`，位置包络以 offset 为中心。
配置和采样时间必须有限，基本频率为正。求值不钳位输出；有效谐波的 double 相位
溢出或最终采样无法表示时抛出溢出异常，无法表示的包络返回无穷大，使有限限位检查失败。

double `hypot` 可能把普通幅值向下舍入，极小幅值的量化还可能被频率放大，导致包络偏小。因此振幅和跨谐波累加均保持扩展精度，最后向外转换到 double；单一非零系数使用其精确绝对值。采样检查对任何正时长都包含起点和终点。全零谐波不会因无关相位溢出而改变恒定轨迹；最终结果溢出仍报错，极小结果按 double 舍入。扩展范围恢复依赖平台的 long double 能力。平台指数范围能够覆盖所有 double 输入的平方和时，振幅使用扩展精度平方根；范围较窄的平台先缩放再求范数。

0.2.4 还检查宽精度幅值开方与导数乘积本身的向内舍入：它们也可能恰好落在偏小的 double 边界，单修最终转换不足以保证保守包络。在代码的二进制指数范围条件内，先用 `4 * epsilon(long double)` 的正包络相对余量筛选；若候选 double 上界不能覆盖余量，只重算对应关节，通过融合残差和向外运算补齐上界。其他平台直接走逐项向外运算。默认二进制舍入、正确的 `sqrt`/FMA 行为是数值假设；本轮验证覆盖 Linux x86_64。加法与乘法残差的基础方法见 [Ogita、Rump、Oishi（2005），第2至3节](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)；本实现的筛选余量推导与整数精确对照保存在验证证据中。

`HarmonicTrajectory::EvaluateInto(time, output)` 可复用 `ExcitationSample` 的四个拥有式向量；成功调用且尺寸不变时不重新分配向量。`Evaluate(time)` 继续返回独立拥有的样本。无效时间和结果溢出会抛出异常；复用输出在异常后可能只被部分更新。限位采样及实验设计评分内部使用复用缓冲，调用者无需改变原有入口。

核心 0.2.2 进一步保留累加的正舍入残差。例如 `offset=1`、第一正弦系数为
1、第二余弦系数为 `1e-100` 时，普通扩展精度累加仍会把半宽舍入成 1，
令下界变成 0；实际 `t=3*pi/2` 的位置为 `-1e-100`。累加器使用
FastTwoSum 恢复每次加法的残差，仅累计使结果偏小的正残差，并对误差总和、
最终 offset 组合和 double 转换向外舍入。负残差可忽略，因为这只会放宽上界。
精确可表示的加法保持精确，不统一扩大所有边界。

FastTwoSum 的依据与舍入假设见
[Ogita、Rump、Oishi：Accurate Sum and Dot Product，Algorithm 1.1](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)。
该推导使用二进制、舍入到最近值以及按绝对值排序的操作数；构建不得启用
`fast-math` 或更改浮点舍入模式。这里修复的是加法丢项及 offset 抵消，
没有为振幅平方根、频率乘积或三角函数提供完整区间误差证明。

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
- 候选结果创建在同一次读取中校验并计算 CSV 摘要，避免校验与摘要阶段读取到不同内容；
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

对误差 `e_i = prediction_i - observation_i`，
`RMSE = sqrt(sum(e_i²)/N)`，`MAE = sum(abs(e_i))/N`，
`normalized_RMSE = sqrt(sum(e_i²)/sum(observation_i²))`。
前两项使用观测值的单位，归一化指标无量纲、不钳位；零观测和零误差的结果为 0，
零观测与非零误差的结果为正无穷大。

RMSE 和平均绝对误差先按最大绝对值缩放再统计，避免平方或求和的中间溢出/下溢。归一化 RMSE 在两个 RMS 为次正规数或舍入为零时保留尺度信息，避免把非零误差比值误报为零或无穷大。无法表示的预测残差会报错；参考观测全为零而误差非零时，归一化 RMSE 仍为无穷大。求解器仍会拒绝不可表示的加权数据、预测残差或最终协方差。

协方差复用最后一次实际求解的 SVD，使用同一秩阈值和保留方向，避免重新分解信息矩阵时平方条件数并丢失弱可观测方向。ridge 协方差使用增广系统，对外报告的秩仍仅来自测量数据。秩亏时采用最小范数伪逆约定：零空间中的零协方差不代表确定性，必须同时检查秩。加权残差平方和超出 double 范围时，保留根误差的尾数和指数，先与奇异值做比例再平方；一致缩放所有行权重不会抹去可表示的协方差。最终协方差出现非有限值时抛出溢出异常，避免返回不可用的不确定性估计。鲁棒迭代达到次数上限时省去未使用的下一轮权重计算，返回的权重仍对应实际完成的最后一次求解。

鲁棒迭代的收敛判定使用缩放后的参数变化范数与相对比值，保留原有绝对/相对容差语义，避免大参数的平方范数或容差乘积溢出后提前报告收敛。

质量矩阵物理验证先归一化再检查对称性和特征值，避免极端尺度掩盖非对称项。特征值恢复至原始尺度后必须有限，再与配置的最小特征值阈值比较；零矩阵及非正定矩阵仍判定失败。

批量验证复用位置缓冲区和特征值求解器，并原地归一化回调返回的矩阵，减少逐姿态内存分配。某个矩阵验证失败后仍继续检查后续姿态，累计失败数和可计算的最小特征值；回调本身抛出的异常仍向调用者传播。

预处理的大窗口平滑使用带补偿和定期重算的滚动求和，保持对称窗口与边界均值规则，总计算量随样本数线性增长。非均匀三点差分以相邻割线斜率求值，使恒定位置偏置先相消，减少虚假速度和加速度。

小窗口求和发生溢出时使用归一化回退，避免把仍可表示的有限均值误判为无效数据。窗口为 1 时直接读取原始位置/力矩矩阵，减少整份数据复制。JSON 工件导入拒绝非零数下溢为零，保留真实零和可表示的次正规数。

矩阵形式和 `RawDataset` 形式的预处理入口共用输入校验，在数值计算前检查关节名、维度、有限值和时间戳严格递增。内部差分复用这项前置保证，避免重复扫描；差分的算术溢出检查和最终数据集校验仍保留。

原始和预处理 CSV 均在逐行解析时检查时间戳顺序，遇到重复或倒退立即停止读取并报告实际行号，空行也计入行号。负起始时间仍然有效，只要所有时间戳有限且严格递增。

两个 CSV 读取入口也会在解析数据行前拒绝重复关节名或包含回车等 CSV 分隔字符的名称，与写入端的名称限制保持一致，避免读取整份无效数据后才报错。

### 参考项目评估与本次实现

2026-10-29 按原始仓库文档与相关实现评估。以下“采用/保留”是针对本仓库现有
Eigen 核心、可选 Pinocchio 后端和离线候选结果流程的工程选择，不是统一性能排名。

| 项目 | 值得参考的部分 | 本次选择 |
|---|---|---|
| [BIRDy](https://github.com/TUM-ICS/BIRDy) | 覆盖 OLS/WLS/IRLS、IV、物理约束方法和 Fourier 实验设计；主要为 MATLAB，SDP 路径需要 CVX | 采用受限激励与回归矩阵质量评估思路；直接对 SVD 的对数信息评分 |
| [dynamic_calibration](https://github.com/shamilmamedov/dynamic_calibration) | 基参数、Fourier/多项式轨迹、pattern search、独立轨迹验证及 YALMIP/SDPT3 物理约束 | 参考流程与 pattern search；物理约束求解仍是独立扩展 |
| [lbr-stack](https://github.com/lbr-stack/dynamic_parameter_identification) | ROS 2 采集、URDF、摩擦、几何约束和具体机器人实验 | 参考集成流程；代码含固定七轴顺序与显式正规方程求逆，当前仅作为参考 |
| [SymPyBotics](https://github.com/cdsousa/SymPyBotics) | 符号回归器与 QR 基参数依赖分析；仓库标注不再维护 | 采用基参数思路；独立实现带列尺度与主元的 QR、三角求解和新数据依赖检查 |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | 原生动力学回归器及 RNEA/ABA 解析导数 | 继续作为已有数值后端，两项新功能通过通用接口直接组合 |
| [Drake](https://underactuated.csail.mit.edu/sysid.html) | URDF/符号 lumped 参数、可辨识子空间与伪惯量物理约束 | 参考可辨识性定义；完整仿真/SDP 框架暂不进入核心依赖 |
| [RBDL](https://github.com/rbdl/rbdl) | RNEA、ABA、CRBA 等 C++ 动力学算法 | 可作为后续交叉验证/替代后端，目前与已有 Pinocchio 的功能重叠 |
| [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) | 模型、动力学和 Python 原型验证 | 适合离线模型对照；当前实现继续通过 C++ 通用后端组合 |
| [iDynTree](https://github.com/gbionics/idyntree) | 浮动基、多坐标表示、URDF 工具与力矩/外力估计 | 浮动基或外力辨识扩展时优先重评；需要明确扩展当前状态/力矩契约 |
| [actuator_net](https://github.com/sunzhon/actuator_net/tree/ec8d395d54b255e283df6bc26b2c29637a839d3d) | 位置误差与速度历史驱动的执行器网络、训练及 TorchScript 推理 | 作为执行器动态建模的重要方向；先明确命令、时序和标签单位，在可选工具层开展独立轨迹验证 |

参考入口还包括 BIRDy 的[实际轨迹目标函数](https://github.com/TUM-ICS/BIRDy/blob/4d9802f1d692815792dfb5652a3d19f1fcdaec02/Benchmark/Robot_Data_Generation/Trajectory_Data_Generation/Trajectory_Optimizer/trajectoryCriterion.m)、
SymPyBotics 的[依赖分析](https://github.com/cdsousa/SymPyBotics/blob/master/sympybotics/dynamics/dyn_parm_dep.py)、
lbr-stack 的[回归实现](https://github.com/lbr-stack/dynamic_parameter_identification/blob/29e761356e09323c240378216a21179c0fff4031/src/regression.py)。
本次代码独立实现，未移植这些项目的源代码。BIRDy 为 MIT，SymPyBotics 为 BSD 三条款；
Drake 主体也是 [BSD 三条款](https://github.com/RobotLocomotion/drake/blob/master/LICENSE.TXT)。
所查 lbr-stack 快照的文件树没有 LICENSE 文件，因此未将其代码引入本仓库。

#### 执行器动态网络方向

2026-11-03 补充评估 `sunzhon/actuator_net`，固定快照为
`ec8d395d54b255e283df6bc26b2c29637a839d3d`。其
[预处理](https://github.com/sunzhon/actuator_net/blob/ec8d395d54b255e283df6bc26b2c29637a839d3d/process_dataset.py)
将命令位置减去实测位置，构成当前及前两帧的位置误差和速度，共六维输入；
[训练](https://github.com/sunzhon/actuator_net/blob/ec8d395d54b255e283df6bc26b2c29637a839d3d/train.py)
使用 MLP 并导出 TorchScript，
[推理](https://github.com/sunzhon/actuator_net/blob/ec8d395d54b255e283df6bc26b2c29637a839d3d/test.py)
逐步更新历史。该方向可学习控制器、传动与电机的综合动态，适合仿真中的执行器映射。
现有 `ActuatorRegressor` 的转子惯量、黏性/库仑摩擦和 bias 列仍用于线性参数辨识；
两类模型应分别验证，不能把网络权重解释为刚体惯量或基参数。

六维输入相同也不代表权重兼容：其引用的
[Walk These Ways](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/base/legged_robot.py)
使用 `q-q_target+motor_offset`，并在每个仿真子步更新历史；零偏置时误差符号与
`actuator_net` 相反。接入必须同时核对符号、偏置及历史更新周期；该实现按仿真子步
更新历史，不能改用策略周期，也不能直接交换已有权重。

随附 [CSV](https://github.com/sunzhon/actuator_net/blob/ec8d395d54b255e283df6bc26b2c29637a839d3d/app/resources/actuator_data.csv)
的表头和首段预览含两组 `index/time/diff`；同一行命令与反馈的时间值并不总相等。
处理脚本没有使用这些时间列对齐，应先确认时钟域、单位和命令实际生效时间，
再构造因果历史，避免把尚未生效的命令用于当前状态。

接入顺序以数据和验证为先，暂不增加训练框架或 GUI 到 Eigen 核心：

1. 采集同步的命令位置、实测位置、实测速度和输出标签，保留关节名称、会话、
   时间戳、采样周期、控制模式与增益。当前 raw CSV 只有 `time/q/tau`，
   缺少命令和速度来源，不能从实测位置反推控制目标。
   若命令来自归一化策略 action，需先按实际控制器还原目标位置单位及默认姿态偏置。
2. 固定特征顺序为 `[e[k],e[k-1],e[k-2],dq[k],dq[k-1],dq[k-2]]`，其中
   `e=q_command-q`；明确周期、命令/反馈延迟和历史初始化策略。
   每个关节、会话、控制模式切换或采样中断都重新建立历史。
   现有对称平滑和中心差分用于离线辨识，含未来样本，不能直接用于实时特征。
3. 明确标签是电流还是输出轴力矩，并记录单位、正方向、减速比和标定来源。
   参考默认处理直接取 `motorStateCur`，没有执行电流到 Nm 的换算；应核对采集端，
   不能仅凭字段名或 README 判断单位，也不能采用注释中的特定电机换算常数。
4. 先按完整会话/轨迹划分训练、验证和测试集，再在各分区内逐关节生成窗口；
   scaler 只在训练集拟合。参考处理先拼接电机再移位，训练脚本按窗口随机划分，
   因而接入时需避免跨电机历史和相邻窗口泄漏。参考默认关闭 scaling；开启时必须
   让训练与部署使用同一变换。
5. 在未参与训练的连续轨迹上，与静态 PD 及使用同组历史特征的线性预测器比较，
   报告逐关节 MAE/RMSE、峰值误差、时延及饱和区间误差，保留真实采样间隔。
   模型、scaler、特征顺序、周期、单位和控制配置应共同绑定摘要。

完整网络训练与真机验证仍待实现；本次采用建模思路并提供独立合成原型，未移植参考
代码、权重或数据，也未改变现有 CSV、结果 Schema 和运行参数导入流程。

#### 半隐式步进与物理/学习混合辨识

该方向结合两层选择：保留可解释的刚体参数，以因果历史残差补充执行器动态；
通过半隐式前向步进拟合连续状态轨迹。令 B 为非负黏性阻尼，h 为步长，
`H[k]` 为当前及过去的命令/状态历史，一个可选离散模型是：

```text
u[k] = tau_PD[k] + r_phi(H[k]) - c(q[k],v[k];theta) - g(q[k];theta)
(M(q[k];theta) + h*B) * v[k+1] = M(q[k];theta)*v[k] + h*u[k]
q[k+1] = integrate(q[k], h*v[k+1])
```

每步解线性系统，位置用更新后的速度推进；B 在速度更新中隐式处理。
完整执行器网络输出 `tau_act` 时，右端直接使用该输出；学习残差时才与 PD 项相加。
这是建模选择，不能同时重复计算同一控制力矩。
[MuJoCo 的积分说明](https://mujoco.readthedocs.io/en/3.6.0/computation/#numerical-integration)
区分半隐式位置更新与速度隐式处理；对高增益控制或学习项仍需验证步长收敛。

辨识目标改为同一命令下预测的 q/v 轨迹误差，各分量按单位和噪声尺度归一化。
训练窗口只在开始时初始化实测状态；随后滚动预测，并使用预测状态更新网络历史。
会话切分先于窗口生成，测试使用完整独立轨迹。该路径可减少对 q 二阶数值微分的
依赖，但仍依赖速度来源、初始状态和命令/反馈延迟的准确性。

现有 `Solve()` 可初始化物理参数及固定非线性特征的线性读出；多步轨迹损失通常
需要单独的非线性优化层。当前 Pinocchio 包只暴露逆动力学回归器，完整多体 rollout
还需前向动力学适配。已有 QR 基参数 beta 不能唯一恢复完整惯性；rollout 必须先
选定物理可行的完整模型，并保持质量矩阵正定、B 非负。

优先固定已标定的电机增益、负载或部分惯性，先拟合物理模型，再学习小残差，
最后有选择地联合校正。若惯量、阻尼、输入增益和残差幅值都自由变化，可能存在
保持同一运动的尺度变换；灵活残差也可能吸收物理项。数值满秩、低轨迹误差或
正则化均不能单独证明物理参数唯一，需检查激励、参数敏感性及独立负载/轨迹。

相关研究入口包括[解析动力学与神经网络混合](https://arxiv.org/abs/2007.06045)
和[通过可微仿真做轨迹辨识](https://arxiv.org/abs/2604.10351v2)。这里采用上述
最小模型独立验证流程，不复现这些论文的真机结果。

可选的 [`hybrid_rollout_demo`](../benchmarks/hybrid_rollout_demo.cpp) 用单轴合成系统
验证这条流程：已知 PD/负载、正惯量和阻尼、六维历史的两个固定 tanh 特征及可学习
读出。数据由恒定控制力矩下的解析解生成，初始化使用离散动量平衡，再以有界坐标
搜索改善多步轨迹损失；测试轨迹不参与优化，同时比较不同积分子步数。
历史每 5 ms 控制周期更新一次，物理积分子步内保持控制力矩，细分积分不改变历史周期。
原型不训练完整 MLP，不输出既有力矩候选 artifact；其中线性初始化残差单位为角动量。

#### QR 基参数

核心 0.2.0 的 `base_parameters.h` 提供 `BaseParameterBasis` 和
`BaseParameterRegressor`。对校准回归矩阵 Y，以各列最大绝对值 s 做固定归一化，
使用带列主元的 QR 在相对阈值下确定数值秩 r。零列保持为零，不作为独立列：

```text
Z = Y * diag(1/s)
Z * P = Q * [ R11  R12 ; 0  R22 ]
K = solve_upper_triangular(R11, R12)
Y_base = Y[:, independent] * diag(1/s_independent)
beta = diag(s_independent)*theta_independent + K*diag(s_dependent)*theta_dependent
Y*theta ~= Y_base*beta
```

实现保留 `selected_columns()`、`column_scales()`、`full_to_base()` 和校准重构误差。
`ToBaseParameters()` 在 IEEE binary64 下对存储映射与输入参数的精确点积作一次
最近舍入，半格取偶；精确结果超出有限 double 范围、或非零结果舍入为零时明确报错。
其他浮点格式保留扩展范围累加。映射本身来自数值 QR，其误差不会因此消失。
`ReduceProblem()` 保留观测、行权重、样本数与 DoF，结果可直接交给已有 `Solve()`。
依赖检查按固定行块复用扩展精度工作区，临时缓存不随总行数增长；输入、输出和
QR 分解仍按各自矩阵尺寸占用内存。
后端包装器可组合原生 Pinocchio 与执行器列，名称使用 `base/` 前缀。

核心 0.2.5 修复了仅增加精度仍可能丢失的坐标。例如映射 `[1,1,1]` 与参数
`[1e300,1,-1e300]` 的精确坐标是 1；取 `a=1-2^-53`、`c=1-2^-52`，
映射 `[a,c]` 与参数 `[a,-1]` 的精确坐标是 `2^-106`。先将乘积或部分和舍入到
long double 都可能得到零；将第二例的映射再乘 `2^-1000`，则应报告下溢。

核心0.2.6的普通路径先跳过零项，每8个非零乘积作一块扩展精度顺序求和，
再用 FastTwoSum/Neumaier 补偿合并块结果。对满足计数和指数范围条件的平台，
乘积、块内与块间累计误差小于 `10.2*u*A`，其中 `u` 是扩展精度单位舍入误差，
`A` 是正的绝对乘积和。筛查使用 `16*u*A` 余量，包含区间端点可能向内舍入的误差。
只有两端均在有限 double 范围内且舍入为同一非零值时采用该结果；否则使用原有
固定整数累加器，检查精确范围后作一次最近舍入，半格取偶。

该筛查要求严格二进制运算、渐进下溢且未启用 fast-math。非默认舍入模式或不满足
宽精度条件时走整数回退。独立列与零列捷径不变，也不增加核心依赖或堆工作区。
按非零乘积计数分块，零系数、零参数和不足8项的尾块都保留原有语义。
映射本身仍来自数值QR；此改进不会消除依赖学习误差。

分块后再准确合并的参考为
[Blanchard、Higham、Mary，2020](https://doi.org/10.1137/19M1257780)。
这里的8项块、筛查常数与端点证明是本实现的本地推导；没有复制论文代码。
原有固定累加器方法参考仍为
[Ogita、Rump、Oishi，2005，§1.1与§2](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)。
Boost大整数只用于可选的独立参考程序，以相邻double的精确中点检验结果。

校准输入必须覆盖足够丰富的状态。样本矩阵的零空间也可能来自激励不足，不能自动
证明结构不可辨识。约简新数据时逐列检查 `Y ~= Y_base*T`；误差按该列校准/当前
最大值中的较大者归一化，零校准列在新数据中被激励时会拒绝。默认误差阈值为
1e-8，默认 QR 相对秩阈值为 1e-10。输入维度、有限值和阈值均校验。
核心 0.2.1 按精确恒等关系处理已选择的独立列，它们对依赖重构误差贡献为零；
满列秩时仍先检查输入维度与有限值。避免对这些列重复回代，也修复零误差容差下
由归一化再乘回尺度产生的微小舍入残差误拒绝。非独立列仍逐列检查。
核心 0.2.2 复用除数向量中的负标记表示独立列；有效列尺度始终非负，
数值计算仍从原始尺度读取，因此无需另外分配逐列位图。
校验缓存的行块上限为 255，避开常见 16 字节 long double 在 256 行时产生的
4 KiB 列跨度；每行仍按原有基参数顺序累加。该选择的收益依赖矩阵尺寸和缓存架构，
不改变依赖误差定义或输出坐标。

核心 0.2.7 在同一列内同时重建相邻两行，每个系数只读取一次；两行分别保留按基参数
索引递增的乘加顺序，并按原行顺序检查误差。每块的奇数尾行继续单独累加，
255 行缓存上限不变，不新增堆工作区或公开对象成员。向上转换、融合除法残差检查、
输入校验和容差策略均沿用原实现；该改动针对依赖校验，不改变 QR 学习算法。

核心 0.2.8 对每列、每行块先取最大绝对残差，再执行一次除法和融合除法残差检查。
同一列的除数固定且为正，舍入除法及其至多一格的扩展精度修正均对残差单调，
因此得到与逐行检查相同的最大误差。不同列仍各用自己的除数；NaN 残差会拒绝，
无穷残差保留到最终非有限检查。各行的乘加顺序、255 行缓存上限和向上转换不变，
不新增堆工作区、公开成员或核心依赖；QR 学习仍是数值过程。

核心 0.2.3 对返回的依赖误差向上转换，避免真实误差略大于 double 容差时，
因舍入成同一数而误通过。例如单行校准矩阵 `[3,3]` 与新数据 `[0,1]` 的误差为
`1/3`，严格大于向下舍入的 double `1/3`；约 `1e-400` 的正误差也不能在返回时变成零。
后者返回最小正 double，使零容差拒绝，而足够大的次正规数容差仍可接受。

扩展精度除法也可能恰好舍入到某个 double 边界。对可能成为最大值、且落在这种
边界上的商 `q`，使用 `fma(-q, divisor, difference)` 检查除法残差；正残差使该商
向上进一格，再进行最终转换。精确的 `1/2` 等边界保持原值。
FMA 的一次最终舍入定义见
[WG14 N1570 委员会草案 §7.12.13.1](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf#page=276)。
这里检查的是已算得的扩展精度重构差值的归一化和返回转换，使用二进制默认舍入、
可表示的残差及正确 FMA 行为；没有对 QR、列归一化和全部乘加提供完整区间证明。

若非零校准列的数量恰好等于基参数秩，这些列都已被选中，剩余依赖仅为零列。
其误差可精确判定：新数据中所有这些列仍为零时是 0，否则是 1。
实现直接比较元素是否为零，跳过除数向量和重构缓存；不使用可能平方下溢的范数判断。
维度和有限值仍先校验，输出坐标的范围检查也保留。该优化不适用于含非零依赖的矩阵，
基准分别提供两类用例。

beta 是可辨识组合，与原始惯性参数通常具有不同的尺度与含义。它不能直接写回
URDF，也不能据此推断原始参数或质量矩阵物理可行。当前接口不提供唯一的完整
惯性向量恢复；应用应保留校准回归器、映射、尺度和版本，作为模型的一部分。
既有 JSON/CSV Schema 不变，`model_digest` 应绑定应用实际采用的模型与基参数定义。

#### 受限 Fourier 实验设计

`experiment_design.h` 提供 `EvaluateExcitationDesign()` 和
`OptimizeExcitationDesign()`。`HarmonicDesign` 使用与现有谐波类相同的位置单位、
Hz 频率和秒时间；本次优化正余弦系数，保持 offset 和频率固定。
初值必须已满足逐关节位置、速度、加速度、jerk 限位，每个接受的候选都通过既有
解析包络和采样限位检查。系数步长按关节半位置范围和谐波数归一化。

给定固定参数尺度 D 与逐关节观测噪声标准差 sigma，将所有采样的回归块堆叠：

```text
A = diag(1/sigma) * Y * D
F = A.transpose() * A / N
score = mean(log(nonzero eigenvalues(F)))
      = mean(2*log(nonzero singular_values(A)) - log(N))
```

N 是时间样本数；噪声尺度使用各关节广义力/力矩的单位，参数尺度使用相应参数
单位，使 A 的列可在统一参考尺度下比较。两个尺度向量为空时视为全一，且在整个
搜索中固定。实现直接对缩放后的 A 做 SVD 并恢复对数尺度，避免显式构造 F 后
平方条件数，也不会因对每个候选重新归一化列而抹去振幅带来的信息增长。
核心 0.2.1 在两个尺度均为一时直接除以矩阵最大绝对值，支持次正规数最大值，
避免生成可能溢出的倒数；非单位尺度保留扩展精度缩放路径。

确定性坐标 pattern search 先提升数值秩，再提升信息分数；只有满列秩时，该分数
才是完整参数集的 D-optimal 指标。`fully_identifiable()`、秩和条件数始终单独报告；
秩亏时条件数为无穷大。搜索预算包含被限位拒绝的候选，预算用尽不标记收敛。
`converged` 仅表示局部步长已缩小到阈值，不代表全局最优。默认最多评估 250 次，
参数及噪声尺度必须有限为正，数值范围或后端错误明确传播。

`identification_design_example` 展示三列冗余回归器约简为两列、受限激励设计、
拟合和不同频率的独立轨迹验证。可选 Pinocchio 测试使用原生 RNEA 力矩作参考。
这些软件结果不包含自碰撞、环境碰撞或硬件运动授权，保持本仓库既有执行边界。

### 推荐迁移顺序

1. 工具质量、质心和力矩 bias；
2. 单关节黏性/库仑摩擦；
3. 使用 Pinocchio 回归器并进行基参数/可观测性分析；
4. 使用已有基参数与受限谐波优化接口设计实验，并完成独立验证轨迹；
5. 根据目标选择物理一致性约束，或按上述数据契约验证执行器动态网络；
6. 扩展现有 ROS 2 recorder，增加显式 session action、激励轨迹联锁和可选 GUI；
7. 候选 artifact 的签名、审批和导入工具。

谐波振幅通过 `hypot` 求值，导数逐次乘频率，避免中间平方、立方的上溢/下溢隐藏真实包络或丢失可表示导数。位置区间的半宽也在全宽溢出时保留有限值；极小振幅仍必须满足保守位置限位。

### 安全边界

辨识只生成候选结果，不能在线热更新力控模型。导入前必须保存旧参数、数据摘要、代码版本、机器人 profile、工具状态、拟合指标和验证结论。任何 URDF、传动、固件、工具、采样频率或力矩来源变化都要求重新评审。真机验证从位置模式和低速度开始，再逐步验证重力补偿、力矩模式、导纳与阻抗。

安装空间中的 session、result、review 和 rollback Schema 分别约束采集元数据、候选结果、人工决定和回滚清单。候选结果 `status` 固定为 `candidate`；批准只能存在于独立 review artifact，且必须绑定满秩、质量矩阵检查通过的候选文件和回滚清单 SHA-256。工具不会写入运行配置。

## English

Core 0.2.0 adds two independent implementations inspired by the reference projects:
column-pivoted QR base parameters and constrained Fourier experiment design.
The detailed comparison and primary-source links appear in the Chinese section
above. The core still depends only on Eigen; its existing layouts, signatures
and JSON/CSV contracts remain compatible.

The actuator-network reference above adds a separate direction: learn a causal
mapping from command-position error and velocity history to actuator output.
Adoption first requires synchronized commands/states, explicit label units,
per-joint/session history resets and trajectory-level holdouts before windowing.
Fit scaling on training data only and bind the deployed model to its feature
order, sign, period and control configuration. Existing symmetric filtering and
central differences are offline operations. Reconcile distinct command/state
timestamps causally before building histories.

The opt-in `hybrid_rollout_demo` combines positive inertia/damping, fixed tanh
history features and learned readout weights with semi-implicit trajectory
matching. It uses an exact scalar reference, a fitted physics-only baseline and
independent holdout motion. Known controller gains/load anchor the force scale;
low rollout error does not establish unique physical parameters. Full MLP
training and multibody forward-dynamics integration remain future adapter work.

For column maxima s, factor `Z = Y*diag(1/s)` with pivoted QR, solve
`K = R11\R12`, and retain the mapping
`beta = diag(s_I)*theta_I + K*diag(s_D)*theta_D`. The reduced regressor is
`Y_base = Y[:,I]*diag(1/s_I)`. Zero calibration columns remain zero. The wrapper
checks reconstruction on new states, so insufficient calibration cannot silently
remove a newly excited direction. Base coordinates are parameter combinations,
not a unique full inertia vector or a physical-feasibility certificate; retain the
basis and its calibration definition as part of the model. Dependency checking
reuses bounded row-block scratch; input, output and QR storage still scale with
their matrix dimensions.

Experiment design stacks `A = diag(1/noise_std)*Y*diag(parameter_scales)` with
scales fixed across candidates, then maximizes the mean log nonzero eigenvalue
of `A.transpose()*A/N` using SVD without forming the normal matrix. Rank is the
first search criterion; the score is D-optimal for the complete parameter set
only at full column rank. Coefficient pattern search preserves fixed offsets and
frequency, requires a feasible seed, and applies conservative and sampled
position/velocity/acceleration/jerk limits to every accepted candidate. Noise
scales use joint generalized force/torque units, parameter scales use parameter
units, and omitted scales are ones. Invalid dimensions, scales and numerical
ranges fail explicitly. A consumed evaluation budget is not convergence, and
local step convergence is not a global-optimality guarantee.

The end-to-end example covers reduction, design, fitting and independent holdout
prediction. Native Pinocchio tests use RNEA as an independent torque reference.
Existing execution and artifact-review boundaries still apply.


Harmonic envelopes use scaled hypot, successive coefficient/frequency products, and overflow-resistant position half-spans. Tiny amplitudes must still satisfy conservative position bounds when sparse sampling misses their peaks.

For harmonic h, `omega_h = 2*pi*f*h`, with f in Hz and time in seconds.
The position is `offset + sum(a_h*sin(omega_h*t) + b_h*cos(omega_h*t))`;
its analytic derivatives multiply each term by successive angular frequencies
and rotate the sine/cosine signs. Coefficients use position units; velocity,
acceleration and jerk use those units per second, second squared and second cubed.
The order-d envelope is `sum(hypot(a_h,b_h)*omega_h^d)`. Amplitudes, envelope
sums and exceptional sample intermediates use extended precision; final envelope conversion
rounds outward. A direct wide squared norm is used only when exponent-range
checks cover every finite double square; narrower platforms retain `hypot`.
Core 0.2.2 keeps positive FastTwoSum addition residuals in a separate upper sum,
then combines that error with the ordinary sum and offset using outward rounding.
This preserves tiny contributions across cancellation without widening exact sums.
The addition argument assumes binary round-to-nearest arithmetic and no fast-math;
it does not certify all errors in square roots, products or trigonometric functions.

Core 0.2.3 rounds reported dependency errors upward and checks the fused division
residual when a candidate maximum quotient lies exactly on a double boundary.
Positive errors cannot disappear during conversion; exactly representable ratios
keep their boundary. This guards normalization/conversion of the computed wide
reconstruction difference, rather than certifying every QR and multiply-add error.
When every nonzero calibration column is independent, remaining zero columns are
checked directly for exact zero. Any new direction, including a subnormal entry,
has error one; input validation still precedes the shortcut.
Core 0.2.4 adds reusable `EvaluateInto` outputs and uses them inside sampled
validation and design scoring. Successful calls with matching output dimensions
need no new vector storage. Owning `Evaluate` calls remain available.
Envelope amplitude roots and derivative products also guard inward rounding
before the final conversion. A relative error screen and exact-significand
checks avoid most wide fused operations; uncertain joints are recomputed with
outward arithmetic. These checks assume default binary rounding and correct
sqrt/FMA behavior; this round validates Linux x86_64.

Samples are not clamped: overflowing active double phases or overflowing
results throw; tiny final results round to double. An unrepresentable envelope is infinite and fails finite limits. Configurations and
times must be finite, with positive fundamental frequency.

Normalized RMSE is `sqrt(sum(error²)/sum(observation²))` and is dimensionless.
Scaled roots preserve representable ratios even when individual RMS values round
to zero. Zero reference and zero error produce zero; zero reference with nonzero
error produces positive infinity. Metrics are not clamped, and nonfinite
predictions or residuals fail explicitly.


The migration separates robot-specific regressor construction from reusable preprocessing, solving, and validation. `robot_identification` is a plain CMake package and therefore works both standalone and in a colcon workspace. It assumes neither left/right arms, seven joints, nor a fixed base-parameter dimension.

Processed CSV uses `time_s`, followed by ordered `q/`, `dq/`, `ddq/`, and `tau/` columns for every joint. Names must be unique, timestamps strictly increasing, and all values finite. Keep raw recordings immutable and record torque provenance, preprocessing settings, model digest, software revision, exclusions, and operating context.

Candidate creation validates and hashes the CSV in a single read so its digest always describes the byte stream that passed validation.

Every accepted result needs full numerical rank, reviewed conditioning, uncertainty, per-joint training and holdout errors, robust-weight diagnostics, and physical validation such as symmetric positive-definite mass matrices over the intended workspace. Ridge and robust regression improve numerical behavior but do not repair insufficient excitation or a wrong model.

Large preprocessing windows use compensated rolling sums with periodic recomputation, preserving symmetric windows and boundary means with linear work in sample count. Nonuniform three-point derivatives are evaluated using adjacent secant slopes so constant position offsets cancel before time division.

The `robot_identification_ros2` recorder consumes named `JointState.position/effort`, reorders a stable joint set, uses process-monotonic callback timestamps, and atomically finalizes the raw CSV without commanding motion. ROS 2 remains an orchestration layer and must never apply an identified model automatically. Produce a candidate artifact, review it offline, retain rollback parameters, and perform staged low-speed hardware validation before changing a runtime profile or URDF.
