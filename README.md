# RobotIdentification

[English](README_EN.md)

机器人动力学参数辨识工具集。从具名关节数据出发，完成预处理、回归、诊断、物理验证和候选参数工件管理。核心使用 C++17 和 Eigen；ROS 2 与 Pinocchio 分别由可选包提供。

| 包 | 职责 | 额外依赖 |
|---|---|---|
| `robot_identification` | 任意自由度数据、CSV、预处理、加权/ridge/Huber 求解、可观测性及物理验证 | Eigen |
| `robot_identification_pinocchio` | 固定基 URDF 刚体动力学回归器 | Pinocchio |
| `robot_identification_ros2` | `JointState.position/effort` 原始数据采集和会话元数据 | ROS 2 |
| `robot_identification_tools` | 流式 CSV 检查、候选、评审和回滚工件 | Python 标准库 |

依赖由采集、模型后端指向辨识核心。核心库独立于控制器和厂商 SDK。

整批 Pinocchio 数据可用 `backend.BuildProblem(dataset)` 复用调用内工作区；单样本和通用后端组合继续使用 `Compute`。详见 [Pinocchio 后端](robot_identification_pinocchio/README.md)。

核心 0.2.16 提供 QR 基参数约简、完整参数映射与受限 Fourier 激励优化，可与已有 Pinocchio 后端组合。[端到端示例](robot_identification/examples/identification_design_example.cpp)演示设计、拟合与独立验证；[算法说明与参考评估](docs/parameter-identification.md#参考项目评估与本次实现)记录公式、数值假设和限制。公开接口保持兼容，Pinocchio 后端为 0.2.2。

[执行器动态网络方向](docs/parameter-identification.md#执行器动态网络方向)评估 `sunzhon/actuator_net`；[半隐式混合辨识](docs/parameter-identification.md#半隐式步进与物理学习混合辨识)提供可选合成原型，记录命令/反馈历史、标签单位和独立轨迹验证条件。

预处理保留极小割线斜率下可表示的端点导数；基参数重构跳过混合依赖中的零列；工件目标已存在时在扫描输入前失败。

## 独立构建

Ubuntu 上的 venv 示例需要 `python3-venv`。

需要 CMake 3.15+、C++17、Eigen 3.3.1+；完整测试还需要 Python 3.10+ 和 `jsonschema==4.23.0`。

```bash
cmake -S robot_identification -B build/core -DBUILD_TESTING=ON \
  -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build build/core --parallel 2
cmake -E chdir build/core ctest --output-on-failure
cmake --install build/core --prefix "$PWD/build/install"
./build/core/generic_regressor_example
./build/core/identification_design_example
```

安装后的 C++ 消费者使用 `find_package(robot_identification REQUIRED)` 和 `robot_identification::robot_identification`。

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
./scripts/test.sh
./scripts/run_identification_example.sh
```

如已配置 Pinocchio 的原生开发环境，可运行 `./scripts/test.sh build/with-pinocchio --with-pinocchio`，同时验证后端和安装包消费者。将其安装前缀加入 `CMAKE_PREFIX_PATH`，共享库目录加入动态加载器路径。

合成求解与 Pinocchio 批量组装性能可用[独立基准程序](benchmarks/README.md)复现。

## 离线工具

```bash
python3 -m pip install ./robot_identification_tools
dataset_inspect session.csv --json
identification_artifact --help
```

完整数据格式、评审与回滚流程见[参数辨识契约](docs/parameter-identification.md)和[工具说明](robot_identification_tools/README.md)。

## 可选 ROS 2 工作区

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src --rosdistro humble -y
./scripts/test_ros2.sh
source build/ros2/install/setup.bash
ros2 run robot_identification_ros2 joint_state_recorder --ros-args \
  -p input_topic:=/joint_states -p output_path:=/tmp/session-001.csv
```

采集器消费标准 `sensor_msgs/JointState`。采样契约与文件恢复语义见[采集器说明](robot_identification_ros2/README.md)。

`python3 utils/ros2_recording_smoke.py` 使用安装后的采集器录制合成消息，验证关节重排、文件摘要与退出清理。

## 结果使用

结果仅作为候选工件生成。实际参数应用仍需独立评审、验证和回滚流程；合成示例不提供硬件参数或时序保证。

采用 [MIT 许可证](LICENSE)。
