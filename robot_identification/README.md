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

## Build and run / 构建与运行

```bash
cmake -S robot_identification -B /tmp/robot-identification \
  -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build /tmp/robot-identification --parallel 2
ctest --test-dir /tmp/robot-identification --output-on-failure
/tmp/robot-identification/generic_regressor_example
```

The example provides a small synthetic backend and performs preprocessing, problem assembly, identification, and diagnostics. It is an API example, not a hardware parameter set.

## CSV contract / CSV 契约

Processed datasets use one exact header order:

```text
time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>...
```

All rows must be finite, timestamps strictly increasing, and every signal group must use the same unique joint order. Raw device records should remain immutable; preprocessing should create a separate processed artifact.

Installed Draft 2020-12 Schemas define recorder metadata and candidate-result artifacts under `share/robot_identification/schema`. Result status is deliberately fixed to `candidate`; promotion into a reviewed runtime configuration is a separate operation.

## Safety boundary / 安全边界

Identification is offline. A successful numerical solve does not authorize writing parameters into a running controller. Accept a result only after excitation-quality review, physical checks, holdout validation, low-speed hardware validation, and an explicit rollback artifact. See [Parameter Identification](../docs/parameter-identification.md).
