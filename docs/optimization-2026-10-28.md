# 2026-10-28 优化记录 / Optimization record

## 范围与环境 / Scope and environment

本轮为本地五小时优化，窗口为 2026-10-28 12:23–17:23（Asia/Shanghai，
04:23–09:23 UTC）。本仓库已从 `RobotServer` 拆出核心、Pinocchio 后端、ROS
采集器和离线工件工具，保留相关历史与 MIT 许可证。本轮只提交本地，没有创建
远程仓库、推送或发布。

对照基线 `dcf42f4`，核心数值修复最终快照 `b310f0e`，具名 Pinocchio 批处理为
`95d43dd`，采集器配置为 `8de01fc`。后续补充可复现基准和开发依赖。环境为
Ubuntu 22.04.5、AMD Ryzen 9 9950X、GCC 11、Eigen 3.4.0、Python 3.10、ROS 2
Humble、Pinocchio 4.0.0。轻量核心只依赖 C++17/Eigen，可独立构建。

This local session optimizes the extracted repository independently. Core,
Pinocchio, recorder and artifact tools retain relevant history and the MIT
license. The baseline is `dcf42f4`; snapshots above identify the measured fixes.
No remote repository was created and nothing was published or pushed.

## 正确性变化 / Correctness changes

- 协方差复用最终求解所消费的 SVD，避免再次分解正规矩阵而丢失较弱但可观测的
  方向。`A=diag(1,1e-8)` 的旧结果报告 rank 2，却把第二方向方差算成 0；
  修复后对应 `diag(1,1e16)`。ridge 使用增广 SVD，数据秩仍按数据矩阵定义。
- 将残差尺度保留到除奇异值后，避免中间方差上溢/下溢。设计尺度 `1e±100`、
  公共权重 `1e±300` 的可表示参考协方差 `1/3`，旧实现可能为 0 或抛出异常，
  修复后四种组合均保持参考值。
- Huber 权重采用分解尺度组合，保留可表示的极小有效权重；避免先算 `delta/r`
  下溢为零。极端回归中原 rank 0/零协方差修复为 rank 1/有限非零协方差。
- 谐波包络使用 hypot，导数尺度逐次相乘；覆盖极小/极大频率下中间值溢出、
  被稀疏采样掩盖的包络超限和有限位置区间跨度。
- 具名批处理在创建数据问题前检查模型标量关节、名称与 q/v 坐标顺序，拒绝
  维度相同但顺序错误的数据；单点位置接口仍允许固定基座 `nq==nv` 多自由度关节。
- Python/C++ CSV 读取器统一支持 LF、CRLF 和 CR 行分隔，并保持数值空白与错误
  尾部检查。SHA-256 针对原始字节，包含头部与行分隔。
- ROS 采集器新增正整数 `qos_depth`，默认仍为 10；验证发生在工件创建之前。
  安装冒烟通过参数服务确认实际声明值，并检查重排、摘要和退出。
- 修复采集器源码分发包的父目录许可证依赖。原分发包转 wheel 时因找不到
  `../LICENSE` 失败，现在包内自带同一 MIT 许可证；维护者元数据同步到独立仓库。

The solver retains observable covariance directions and representable uncertainty
across extreme scales. Huber weights and harmonic bounds avoid destructive
intermediate arithmetic. Named Pinocchio batches validate model coordinates;
CSV readers agree on line endings. The recorder validates configurable queue
depth before creating files.

协方差在不可辨识零空间中的零值仍是最小范数计算约定，不能解释为已知且没有
不确定性；应同时检查 rank、条件数与可观测性。辨识结果仍只是候选工件。

Zero covariance in a nullspace remains a minimum-norm convention, not evidence
of certainty. Review rank, conditioning and observability alongside covariance.

## 合成性能 / Synthetic performance

使用提交的 [`benchmarks`](../benchmarks/README.md) 对匹配 Release 安装包预热，
报告五轮中位数。核心比较固定 CPU 30，Pinocchio 固定 CPU 31；其他验证并行运行，
因此不是空闲隔离主机的时序实验，也没有硬件时序或轨迹激励充分性保证。

| 核心调用 | 基线 µs/次 | 优化后 µs/次 | 捕获分配数，前→后 |
|---|---:|---:|---:|
| 1500×12 Huber IRLS | 2431.594 | 1852.762 | 147 → 39 |
| 同系统 + ridge | 2781.007 | 2502.747 | 202 → 55 |
| 七轴、20 谐波轨迹求值 | 1.051 | 0.565 | 44 → 4 |

| 1000 样本 Pinocchio 问题组装 | 通用逐样本 ms/批 | 新批处理 ms/批 | 捕获分配数，前→后 |
|---|---:|---:|---:|
| 二自由度 | 4.597 | 0.647 | 69046 → 1114 |
| 十四自由度 | 49.609 | 38.345 | 93286 → 1378 |

IRLS 复用加权工作区与分解对象，协方差复用最终 SVD；批处理在一次调用内复用
Pinocchio Data，互不共享可变工作区。并发批处理与通用入口矩阵对照均有回归。
批处理保留每样本拥有的回归块，仍捕获 1003 次 C 分配，并未实现整个组装零分配。

计数针对基准线程，覆盖可执行文件/静态库中包装的 malloc/calloc 与覆盖函数捕获的
普通、数组、对齐 C++ new；二者相加，预编译共享库内部的直接 C 分配可能漏计。
两边核心计数口径相同。校验和一致有单独的精度回归补充。

These are warmed synthetic medians under concurrent validation. Counters cover
wrapped executable/static-library malloc/calloc and overridden C++ allocation
on the benchmark thread; direct C allocation in shared libraries may escape.
Batch Data reuse is call-local, preserving independent concurrent calls. The
batch API still owns its per-sample blocks and is not allocation-free.

## 验证与持续采集 / Validation and sustained recording

| 项目 | 结果与实现快照 |
|---|---|
| 完整 ROS 工作区 | 4 包、87 测试，无失败/跳过；安装采集器 20 样本验证通过；`95d43dd` 及后续工具提交 |
| 轻量独立构建 | 核心 10 测试、Python 工具 49、采集器 23、runner 4，以及 schema 检查；固定独立 Python 环境 |
| 消毒器 | 核心 10、Pinocchio 5 测试通过；核心独立 LSan 10 测试通过 |
| Conda Pinocchio 2.7 | GCC 11 全新后端构建，5/5 后端与安装消费者测试通过；核心复用独立安装 |
| 干净源码 | 从纯 Git 提交导出源码，核心/Pinocchio 独立构建、安装消费者与 Python 验证通过；`b09d766` |
| Python 分发 | 两 Python 包单独复制后制作 sdist，再解包构建 wheel；验证许可证、入口点和仓库外安装工具 CLI；setuptools 59.6.0 / wheel 0.37.1 |
| 原始/最终解对照 | 2000 个固定种子系统，参数、预测、残差、权重、秩、迭代、误差与条件数逐位一致；协方差为有意修正 |
| 标准解析系统一小时 | ASan，2,275,696 用例；最大参数归一化误差 `4.33e-8`、协方差相对误差 `3.82e-8`；`6ffec92` |
| 宽尺度解析系统一小时 | ASan，2,304,589 用例；设计尺度 `1e±130`、公共权重 `1e±300`；最大参数误差 `5.13e-8`、协方差误差 `4.06e-8`；`bb02c40` |
| 最终核心：标准解析系统一小时 | ASan/UBSan/LSan，1,956,325 用例；最大参数归一化误差 `4.33e-8`、协方差相对误差 `3.82e-8`；`b09d766` 程序与最终核心实现，退出 0 |
| 最终核心：宽尺度解析系统一小时 | ASan/UBSan/LSan，2,027,444 用例；最大参数误差 `5.13e-8`、协方差误差 `4.06e-8`；同一提交程序与最终核心实现，退出 0 |
| 常规控制状态采集一小时 | 35587 有效行、0 拒绝，行数和摘要一致；早期采集器快照 |

解析压力测试中，参数误差为 `norm(actual-reference) / max(1, norm(reference))`，
协方差误差为差矩阵 Frobenius 范数除以 `max(1e-300, norm(reference_covariance))`，
使用稳定范数计算。小参数参考向量使用绝对尺度，不能将该参数指标解释为纯相对误差。

高频采集使用同机合成发布器，十四关节、目标 1000 Hz，周期性反转关节顺序并注入
NaN、缺 effort、重复名称。每轮一小时发送 3,600,000 条，其中 3567 条无效：

| 订阅深度 | 有效行 | 拒绝数 | 未送达有效/无效 | 每行与摘要校验 |
|---|---:|---:|---:|---|
| 10 | 3,588,555 | 3557 | 7878 / 10 | 通过 |
| 1000 | 3,596,433 | 3567 | 0 / 0 | 通过 |

默认深度一轮观察到约 0.2191% 总消息未送达。深度 1000 一轮未观察到遗漏，但最长
回调接收时间间隔约 0.557 秒，说明积压容忍与低延迟不能混为一谈。两轮负载与调度
并非严格受控的单变量实验，结果不能证明任意环境无丢包。文件时间仍为进程单调
接收时间，不是传感器采样时间。深度参数增加内存与潜在排队延迟，默认值保留 10。

持续数据逐行检查具名位置、力矩、严格顺序，最终行数与元数据及原始文件摘要
一致。单轮约保留 1.8 GB CSV。进程采样用于观察，不能单凭 RSS 断言无泄漏。

The one-hour 1000 Hz runs explicitly distinguish delivery from rejection.
Depth 10 missed 7,888 messages; depth 1000 missed none in this run, with a maximum
callback receipt gap of about 0.557 seconds. Scheduling/load differed between
runs. This is not a loss-free acquisition guarantee or source-time measurement.
Earlier numeric campaigns apply to the listed snapshots and supplement final
targeted regressions. Native Pinocchio sanitizer runs disable leak detection;
the standalone core also passed with LSan enabled.

Fresh one-hour runs of the committed programs against the final core completed
1,956,325 standard and 2,027,444 wide-scale analytic cases. Both exited 0 with
ASan/UBSan and leak detection enabled; maximum scaled parameter and normalized
covariance errors were respectively `4.33e-8`/`3.82e-8` and `5.13e-8`/`4.06e-8`.
Parameter normalization divides by `max(1, norm(reference))`; covariance uses
`max(1e-300, norm(reference_covariance))`. Small parameter vectors therefore use
an absolute scale rather than a pure relative error.

## 复现与兼容性 / Reproduction and compatibility

```bash
python3 -m pip install -r requirements-dev.txt
./scripts/test.sh build/validation --with-pinocchio  # optional native dependency
./scripts/test_ros2.sh build/ros2-validation         # sourced ROS environment
```

无 Pinocchio 时去掉 `--with-pinocchio`。性能、解析压力测试与独立 ROS 采集负载
程序见 [`benchmarks/README.md`](../benchmarks/README.md)。核心与 Python 工具
不依赖 RobotServer。原始日志、计时输出、长测 JSON/CSV 保存在本地
`build/optimization-20261028/`，不进入版本库。

Python 分发复现时，从各包目录运行 `python3 setup.py sdist`，将得到的源码包
解压到一个没有父仓库文件的新目录，再运行 `python3 setup.py bdist_wheel`；需要
setuptools 和 wheel 构建工具。检查两包的 `share/<package>/LICENSE` 与原 MIT
许可证一致。这里验证打包流程，没有上传分发平台。

核心维持 0.1.0，公开布局不变；Pinocchio 后端为 0.2.0，新增非虚批处理入口，
保留原入口；ROS 采集器为 0.2.0，要求 Python >=3.10。安装脚本与包版本一致。
新后端调用方需链接匹配新库，原生 Pinocchio 调用方包含所用算法头文件。
参数应用仍需要独立评审、物理验证、回滚与真机验收。

The core retains version 0.1.0 and its public layout. Pinocchio 0.2.0 adds a
nonvirtual batch entry without removing existing APIs. Recorder 0.2.0 exposes
validated QoS depth and requires Python >=3.10. Link new callers against the
matching backend. Hardware commissioning and applying candidate parameters
remain outside this validation.
