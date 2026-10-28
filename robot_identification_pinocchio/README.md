# robot_identification_pinocchio

Optional fixed-base rigid-body regressor for `robot_identification`. It wraps
Pinocchio's joint-torque regressor without adding Pinocchio to the lightweight
core package.

参数顺序严格跟随 Pinocchio `Inertia::toDynamicParameters()`：每个非 universe
joint 依次为 `mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz`。参数名称包含
模型关节名，因此结果可以审计，不能仅凭数组位置写回模型。

The positional backend accepts fixed-base models with `nq == nv`; sample vectors
follow model q/v order. Named batch assembly additionally requires one coordinate
per joint and matching dataset names. Floating-base identification requires a
separately versioned dataset contract rather than dropping base coordinates.

```bash
colcon build --packages-up-to robot_identification_pinocchio
colcon test --packages-select robot_identification_pinocchio
```

Use `JointActuatorRegressor` around this backend when drive inertia, viscous or
Coulomb friction, and torque bias should be estimated together with rigid-body
parameters. Full rank and physical validation remain mandatory.

For a complete dataset, use the 0.2.0 batch API:

```cpp
robot_identification_pinocchio::PinocchioRegressor backend(model);
auto problem = backend.BuildProblem(dataset);
```

`BuildProblem` reuses one call-local `pinocchio::Data` across samples, while the
core assembler retains dataset validation, row ordering and weights. Independent
calls may run concurrently on the same backend, provided the shared model is not
modified. `Compute(sample)` remains available for individual samples and generic
backend composition; it uses independent scratch for each invocation. The batch
API is additive, with no object layout or vtable change. Consumers calling it
must require and link the 0.2.0 package.

整批数据建议使用 0.2.0 新增的 `backend.BuildProblem(dataset)`。每次调用独占一个
Pinocchio 工作区并在样本间复用；数据校验、行顺序和权重仍由核心组装器处理。
在共享模型不被修改的前提下，同一后端允许并发批量调用。单样本 `Compute`
继续使用独立工作区，适合通用后端组合。新接口不改变对象布局或虚函数表；
调用新接口的消费者需要链接 0.2.0。

The named batch API requires one q/v coordinate per model joint and dataset names
in the model's q/v order. Wrong names, permutations, and ambiguous multi-DoF joint
labels fail before workspace construction. Configure the recorder's `joint_names`
accordingly. Positional `Compute(sample)` supports the existing fixed-base
`nq == nv` interface; generic assembly callers remain responsible for coordinate
ordering.

具名批量入口要求每个模型关节只有一个 q/v 坐标，数据集名称顺序必须与模型一致。
名称错误、顺序颠倒和多自由度关节的歧义标签会提前报错；采集时应相应配置
`joint_names`。位置式 `Compute(sample)` 保留原有固定基座 `nq == nv` 接口，通用
组装调用方需要自行保证坐标顺序。
