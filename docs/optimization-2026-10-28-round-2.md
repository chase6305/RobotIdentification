# 第二轮优化记录 / Second optimization session

## 范围 / Scope

本轮原定窗口为 2026-10-28 14:35:40–19:35:40 UTC，即北京时间 10 月 28 日
22:35:40 至 10 月 29 日 03:35:40。代码与长测在 10 月 28 日完成，10 月 29 日
02:02 UTC 恢复会话后汇总交付；计划窗口不等同连续活跃时长。仓库保持本地独立，
没有创建远程仓库、推送或发布；[第一轮报告](optimization-2026-10-28.md) 保留。

本轮基线 `0858f06`；生产变更为 `2ebb839`（共享组装实现）、`ef2dcaf`（单样本
Data 缓存）、`2c64b3d`（原始 CSV 分块读取），持续验证程序为 `5bce2d0`。
后续 `44ad750` 修复可表示导数的中间值溢出，并加入独立预处理压力验证。
验证程序 `e5bbe11` 按两次差分传播舍入余量，并支持指定用例编号重放。
环境为 Ubuntu 22.04.5、Ryzen 9 9950X、GCC 11、Eigen 3.4.0、Python 3.10、
ROS 2 Humble；分别验证原生 Pinocchio 4.0.0 和 Conda Pinocchio 2.7.0。

This second local session starts from `0858f06`. Core 0.1.1 and Pinocchio backend
0.2.1 retain public object layouts and signatures; the backend now requires core
>=0.1.1. The installed `detail` assembly header is an implementation detail,
not a stable extension interface. The lightweight core still needs only C++17
and Eigen.

## 实现 / Implementation

- 通用组装和具名 Pinocchio 批量入口使用同一套数据、参数名称、维度及有限值
  校验。批量入口在原生 Data 中生成回归块后直接写入最终问题，移除逐样本拥有
  矩阵的临时副本。回调返回值在下一次计算前被消费，最终矩阵独立拥有存储。
- 通用单样本 `Compute` 使用每线程至多一个 Data 缓存，以弱所有权和地址为键，
  不保留源模型的强所有权，也不在线程间共享可变 Data。单样本返回值仍拥有存储；
  具名批量入口继续使用单次调用私有的 Data。
- 原始 CSV 读取改用逐步增长、可逐列释放的块，避免保留整份行缓冲并同时分配
  全部最终数组。保持单次顺序读取和原有数值/换行合同；无需寻址文件。最终
  返回值仍需与行数和关节数成正比的内存。此变化用更多块分配换取较低峰值。
- 导数的普通路径保留相邻割线算法；中间值溢出时仅对该系数以 long double 重算，
  最后检查 double 范围。已复现并修复“位置差溢出、最终速度/加速度有限”及
  “单个割线溢出、组合导数有限”两种情况。不可表示的最终结果仍报错，恢复能力
  取决于平台的 long double 范围/精度。

Both assembly paths consume each validated block before the next computation.
Thread-local single-sample storage supports generic backend composition without
sharing mutable Data. Raw CSV chunks can be released as each final array is
filled, reducing peak storage while preserving owning outputs and one-pass I/O.
Exceptional differentiation coefficients are retried in long double; ordinary
data retains the offset-cancelling secant path. Final representability is checked.

## 合成测量 / Synthetic measurements

使用提交的 [基准程序](../benchmarks/README.md)，Release、固定 CPU 31。除 CSV
单轮外，计时为预热后的五轮中位数。主机同时运行其他工作，未隔离；这些结果
不构成硬件时序、采样无丢失或轨迹激励充分性的保证。各配对结果校验和一致。

| 1000 样本问题组装 | 基线 ms/批 | 本轮 ms/批 | 捕获分配数，前→后 |
|---|---:|---:|---:|
| 二自由度，通用 | 4.024 | 0.567 | 69046 → 1025 |
| 二自由度，具名批量 | 0.586 | 0.549 | 1114 → 93 |
| 十四自由度，通用 | 43.955 | 34.927 | 93286 → 1145 |
| 十四自由度，具名批量 | 35.481 | 34.912 | 1378 → 237 |

分配数是当前线程中包装的 malloc/calloc 与覆盖的普通/数组/对齐 C++ new 之和。
预编译共享库内部的直接 C 分配可能漏计。通用接口仍为每个单样本返回矩阵分配；
整批组装也没有实现零分配。十四自由度具名批量的时间改善较小。

| 原始 CSV，独立新进程单轮 | 基线 ms | 本轮 ms | 基线峰值 KiB | 本轮峰值 KiB |
|---|---:|---:|---:|---:|
| 200000 行 × 14 关节，117884086 字节 | 189.969 | 152.267 | 94048 | 51200 |
| 3596433 行 × 14 关节，1787470095 字节 | 3088.321 | 2124.934 | 1633120 | 850176 |

上述范围包含读取、校验和与返回值销毁；RSS 为 Linux `getrusage` 的进程峰值，
不只是临时工作区。200000 行文件的五轮预热中位数反而为 128.750→131.319 ms，
因此不能宣称读取总会更快；该实验峰值为 123868→94232 KiB。单轮和预热的进程
分配器状态不同，不能混用。大文件的分配增加到 4803 次 malloc/calloc 和 1632
次 new，原实现分别为 3/43 次。

The CSV change primarily reduces peak storage. Smaller warmed reads were about
2% slower in this run. Fresh-process and warmed results measure different
allocator histories; filesystem cache and host activity also affect timings.
The deterministic 200000-row fixture SHA-256 is
`8d3521bcca84e47cf975737a48b521bea59183efc2f8de5c9c7a23d26db6059e`.
Use the committed generator with a new path and retain the digest for comparisons.
The retained 1787470095-byte recording has SHA-256
`20165438c1812d5111b3559c7e6910f64ceba4a809c384f28c41bdee61808ed6`,
rechecked against its original recorder metadata.

## 验证 / Validation

- 原生与 Conda 独立构建均通过核心 11 项、Pinocchio 5 项 CTest，以及工具/采集器/
  运行脚本的 Python 与 Schema 检查。核心 CTest 包含安装后消费者与 CSV 跨语言往返。
- ASan/UBSan/浮点转整数溢出检查的核心 11 项、Pinocchio 5 项通过；纯核心另以
  LeakSanitizer 打开方式通过 11 项。Pinocchio 插桩使用匹配 GCC/Eigen ABI，关闭
  该进程的泄漏检测，不将其描述为泄漏验证。
- 最终四包 ROS 2 colcon 验证：88 项、零失败；安装录制器通过 20 条采样测试。
- 新的并发回归/组合压力程序在 Pinocchio 4.0 与 2.7 的短时运行均通过；CSV 位级
  往返/延迟错误短测通过。
- 原生 Pinocchio 四线程 ASan/UBSan 一小时：14423447 组、72123305 样本、
  408504274 执行器组合行通过，2523636 次不支持的具名多坐标批量按预期拒绝；
  最大归一化误差 `4.86915e-15`。此进程关闭泄漏检测。
- CSV ASan/UBSan/LeakSanitizer 一小时：54693 个文件用例、60041935 行位级
  往返及 54693 次末尾错误拒绝全部通过，未报告泄漏。
- 最终导数修复在原生/Conda 独立构建、ASan/UBSan、纯核心 LeakSanitizer 11 项
  及完整 ROS 2 88 项中均通过。独立预处理短测约 96085 组，其中 80952 组有限结果
  通过参考比较，15133 组超范围结果按预期拒绝。
- 更正误差传播余量后，预处理 ASan/UBSan/LeakSanitizer 一小时通过 31670975
  组：26687026 组有限结果对照通过、4983949 组超范围按预期拒绝，其中 989718
  组覆盖双精度割线溢出但组合导数有限的固定场景。未报告插桩错误或泄漏。
- `44ad750` 完整 Git bundle 含 HEAD/主分支信息，已在新的独立目录还原。清除
  ROS/Pinocchio 前缀、库和 Python 路径后，用系统 GCC 11、系统 Eigen 3.4.0
  与系统 C++ 运行库通过轻量核心 11 项；未依赖 RobotServer 源码或安装空间。

`pinocchio_stress` 用四个固定种子线程，检查共享不可变后端、所有权切换、标量与
平移关节、独立 RNEA、具名/通用矩阵、执行器组合及返回值拥有性。误差定义为
`norm(actual-reference) / max(1, norm(reference))`。`csv_stress` 覆盖分块边界、
负零、最小非正规数、最大有限值、LF/CRLF/CR 和文件末尾非法行；结果按位比较。
`preprocessing_stress` 则用独立的扩展范围拉格朗日权重，在非均匀时间及宽位置/
时间尺度上检查速度、加速度和超范围拒绝。验证余量按 `64*eps*sum(abs(w*input))`
加前一阶段误差传播及次正规量化项计算。初始相对容差在第 7342120 号用例失败；
重放确认速度差 1 ULP，经近线性二次差分相消放大后，超出纯加速度相对容差。
按差分权重传播的余量内则通过，没有为此修改生产算法。保留原失败日志、输入、
Decimal 分析和更正后的重放结果。一小时新验证已通过。性能表分别测于组装/TLS 和
CSV 实现快照，导数修复没有改动这些被测函数。

The initial relative-only preprocessing oracle failed on case 7342120. Replay
showed a one-ULP velocity difference amplified by cancellation in the second
derivative. The corrected oracle propagates Lagrange-weighted rounding allowances;
production code was unchanged. The captured data, Decimal analysis and passing
replay are retained alongside the original failed log.

Local raw evidence is under `build/optimization-20261028-r2/`: `pin-before.txt`,
`pin-after-tls.txt`, `csv-*-memory.txt`, `final-colcon.log`, standalone and
sanitizer logs, plus the new stress logs. Case counts and source snapshots for
sustained runs are recorded separately from unit-test totals. Dataset acquisition
timing, torque provenance, physical parameter validity and robot commissioning
remain deployment responsibilities; these are synthetic software checks.

The local handoff directory also contains `RobotIdentification-round2-final.bundle`,
`RobotIdentification-round2-evidence.tar.gz`, `round2-final-manifest.json` and
`round2-final-SHA256SUMS`. The final bundle includes branch/HEAD information and
the corrected validation harness. Large CSV fixtures and build trees stay outside
the evidence archive; hashes and source snapshots identify the retained inputs.
