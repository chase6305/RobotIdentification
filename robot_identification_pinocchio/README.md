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
