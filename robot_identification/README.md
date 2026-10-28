# robot_identification

Middleware-independent building blocks for robot parameter identification. The package is intentionally independent of ROS 2, robot brands, joint count, and generated symbolic models. It can be built directly with CMake or discovered as a plain-CMake package by colcon.

主要能力 / Capabilities:

- 任意自由度、具名关节和严格时间戳的数据集 / arbitrary-DoF named-joint datasets with strict timestamps;
- 便携 CSV 读写 / portable processed-dataset CSV I/O;
- 对称零相位平滑、非均匀时间三点速度与加速度 / symmetric zero-phase smoothing and three-point nonuniform-time derivatives;
- 可插拔 `RegressorBackend` / pluggable regressor backend;
- 任意 DoF 驱动惯量、黏性/库仑摩擦和力矩偏置回归器装饰层 / arbitrary-DoF drive-inertia, viscous/Coulomb-friction, and torque-bias augmentation;
- 加权最小二乘、ridge 和 Huber IRLS / weighted least squares, ridge, and Huber IRLS;
- 秩、条件数、协方差与残差指标 / rank, conditioning, covariance, and residual metrics;
- 独立验证集评估和质量矩阵正定性检查 / holdout evaluation and mass-matrix positive-definiteness checks.
- 任意 DoF 谐波激励轨迹及位置/速度/加速度/jerk 解析包络和采样预验证 / arbitrary-DoF harmonic excitation with analytic envelopes and sampled position, velocity, acceleration, and jerk preflight checks.

The library solves a linear-in-parameters model:

```text
tau[k] = Y(q[k], dq[k], ddq[k], context[k]) beta + error[k]
```

`RegressorBackend` owns the model-specific construction of `Y`. It may wrap Pinocchio, generated symbolic code, a payload-only model, or a vendor model. The solver never assumes primary/secondary means left/right, seven joints, or a fixed parameter count.

未启用 ridge 时，每次加权求解复用同一次 SVD 完成参数估计与可观测性诊断；启用 ridge 时预分配正则化行，秩和条件数仍仅由测量数据决定。预处理会拒绝溢出的裁剪数量，并将平滑窗口限制在可用样本范围内；矩阵维度和加权数值溢出会在分解前报错。

Without ridge, each weighted solve reuses one SVD for parameter estimation and observability diagnostics. Ridge rows are preallocated, while rank and conditioning still describe measured data alone. Preprocessing rejects overflowing trim counts and bounds smoothing windows by the available samples; invalid dimension arithmetic and non-finite weighted values are rejected before factorization.

大窗口对称平滑使用补偿滚动求和，并定期从当前窗口重新求和以限制累计舍入误差；总计算量随样本数线性增长。短窗口保留 Eigen 直接求均值的路径，两者都在边界使用实际可用样本。三点导数由相邻割线斜率计算，在相减后再除以采样间隔，避免较大恒定位置偏置造成虚假速度与加速度。

Large symmetric windows use compensated rolling sums with periodic recomputation to limit accumulated rounding error, giving linear work in sample count. Short windows retain Eigen reductions; both paths average the available samples at boundaries. Three-point derivatives use adjacent secant slopes, subtracting positions before dividing by time intervals so large constant offsets do not create spurious velocity or acceleration.

Huber IRLS 在一次 `Solve` 内复用加权矩阵、SVD 和残差工作区。谐波采样直接累加各关节的解析项，不再为每阶谐波构造动态向量；公开数据与结果接口保持兼容。

Huber IRLS reuses weighted matrices, SVD decompositions, and residual storage within one `Solve` call. Harmonic sampling accumulates analytical joint terms without dynamic vectors for each harmonic. Public dataset and result interfaces remain compatible.

谐波包络使用 `hypot` 计算振幅，避免平方时上溢或下溢。导数逐次乘频率，避免先计算频率幂导致可表示结果丢失；宽位置区间的半宽也使用防溢出计算。保守包络仍在采样前检查，稀疏采样不能替代连续轨迹限位。

Harmonic amplitudes use `hypot` to avoid square overflow or underflow. Derivatives multiply coefficients by successive frequency factors so representable results survive extreme scales. Position half-spans avoid overflowing the full width. Conservative envelopes are checked before sampling; sparse samples cannot replace continuous-trajectory limits.

Huber 比例进入次正规数范围时，通过尾数和指数组合计算最终权重，避免先计算极小比例而丢失可表示的加权结果。正常比例保留直接乘法路径。

When a Huber ratio is subnormal, mantissa/exponent arithmetic preserves representable final weights before a large base weight restores their scale. Ordinary ratios retain the direct multiplication path.

## Build and run / 构建与运行

Requires Eigen 3.3 or newer and a C++17 standard library with floating-point
`std::from_chars` support (verified with GCC 11/libstdc++).

```bash
cmake -S robot_identification -B /tmp/robot-identification \
  -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build /tmp/robot-identification --parallel 2
ctest --test-dir /tmp/robot-identification --output-on-failure
/tmp/robot-identification/generic_regressor_example
```

The example provides a small synthetic backend and performs preprocessing, problem assembly, identification, and diagnostics. It is an API example, not a hardware parameter set.

When Python and the sibling recording/inspection packages are present, CTest
also exercises Python-recorded raw CSV and processed CSV through the C++ reader
and writer. These interoperability tests use the recording core without ROS.

## CSV contract / CSV 契约

Processed datasets use one exact header order:

```text
time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
```

All rows must be finite, timestamps strictly increasing, and every signal group must use the same unique joint order. Raw device records should remain immutable; preprocessing should create a separate processed artifact.

Fields are literal comma-separated text. CSV quoting and escaping are not
interpreted; quotes inside joint names remain literal characters. Python/C++
interoperability tests cover accepted numeric syntax, rejected extensions,
subnormal boundaries, and blank rows in addition to exact signal values.

CSV import parses numeric fields in the reusable line buffer; raw recordings accumulate in contiguous storage before conversion to the output matrices. LF/CRLF, blank lines, and a final row without a newline remain supported. Readers report I/O failures, and writers check the final buffered write when closing the file.

CSV 导入直接复用行缓冲区解析数值，原始记录先连续存储，再转换为输出矩阵，减少逐字段字符串复制和逐行向量分配。继续支持 LF/CRLF、空行及末行无换行符；读取发生 I/O 错误或关闭文件时最后一次缓冲写入失败，均会抛出异常。

C++ 数值解析使用有界的 `std::from_chars`，不修改输入缓冲区，也不读取进程语言环境。写出时固定使用小数点 `.` 且不插入数字分组符。可表示的极小非零数（次正规数）和正负零能够精确往返；数值溢出或非零值过小而只能表示为零时仍会报错，并给出行列位置。保留前导 ASCII 空白、显式正负号及 `0x` 十六进制格式；尾随垃圾字符仍被拒绝。

Numeric conversion uses bounded `std::from_chars` calls without modifying the
line buffer or consulting the process locale. Writers always use `.` as the
decimal separator and disable digit grouping. Representable subnormal values
and signed zero survive exact round trips; overflow and nonzero values that
underflow to zero still report their source line and column. Leading ASCII
whitespace, explicit signs, and `0x` hexadecimal syntax remain supported;
trailing garbage remains invalid.

Installed Draft 2020-12 Schemas define recorder metadata and candidate-result artifacts under `share/robot_identification/schema`. Result status is deliberately fixed to `candidate`; promotion into a reviewed runtime configuration is a separate operation.

## Safety boundary / 安全边界

Identification is offline. A successful numerical solve does not authorize writing parameters into a running controller. Accept a result only after excitation-quality review, physical checks, holdout validation, low-speed hardware validation, and an explicit rollback artifact. See [Parameter Identification](../docs/parameter-identification.md).

协方差复用最后一次实际求解的 SVD，按保留方向计算
`variance * V_r * diag(1 / s_r²) * V_rᵀ`，避免信息矩阵平方条件数后误丢弱方向。
ridge 使用增广系统的分解；对外报告的秩仍只来自加权测量数据。秩亏时沿用
最小范数伪逆约定，零空间中的零协方差不能解释为参数已经确定。

Covariance reuses the last consumed solve's SVD and retained singular directions,
using the augmented decomposition for ridge. Reported observability still uses
weighted measured data. A rank-deficient pseudoinverse has zero covariance in
its nullspace; this is a minimum-norm convention, not evidence of certainty.

当加权残差平方和发生范围损失时，协方差保留根误差的尾数/指数，先除以奇异值
再平方，支持一致行权重缩放后的可表示结果。不可表示的最终协方差仍明确报错。

If the weighted residual squared norm loses range, covariance retains a scaled
root error until division by singular values. Uniform weight scaling preserves
representable uncertainty; an unrepresentable final covariance still fails.
