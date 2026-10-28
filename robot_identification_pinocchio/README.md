# robot_identification_pinocchio

Optional fixed-base rigid-body regressor for `robot_identification`. It wraps
Pinocchio's joint-torque regressor without adding Pinocchio to the lightweight
core package.

参数顺序严格跟随 Pinocchio `Inertia::toDynamicParameters()`：每个非 universe
joint 依次为 `mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz`。参数名称包含
模型关节名，因此结果可以审计，不能仅凭数组位置写回模型。

The backend currently accepts fixed-base models with `nq == nv`, matching the
core dataset's one-position/velocity/acceleration value per named joint
contract. Floating-base identification requires a separately versioned dataset
contract rather than silently dropping base coordinates.

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
