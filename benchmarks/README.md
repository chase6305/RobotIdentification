# Synthetic identification benchmarks / 合成辨识基准

These opt-in Linux/GNU-linker executables measure warmed calls and report the
median of five repetitions. They are software measurements, not hardware timing
guarantees or evidence that a robot trajectory is safe or sufficiently exciting.
Compile both libraries and consumers in Release mode with matching compiler and
Eigen allocation ABI.

The lightweight core requires C++17 and Eigen, with no ROS or Pinocchio:

```bash
cmake -S robot_identification -B build/benchmark-core -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$PWD/build/benchmark-install"
cmake --build build/benchmark-core --parallel 2
cmake --install build/benchmark-core
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/benchmark-install"
cmake --build build/benchmarks --parallel 2
build/benchmarks/core_bench
```

With a configured native Pinocchio environment, install the optional backend in
Release mode and enable its benchmark:

```bash
cmake -S robot_identification_pinocchio -B build/benchmark-pin -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH="$PWD/build/benchmark-install" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/benchmark-install"
cmake --build build/benchmark-pin --parallel 2
cmake --install build/benchmark-pin
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/benchmark-install" -DROBOT_IDENTIFICATION_BENCH_PINOCCHIO=ON
cmake --build build/benchmarks --parallel 2
build/benchmarks/pinocchio_bench
```

Core cases use a 1,500-row/12-parameter system with Huber IRLS, optional ridge,
and a seven-axis/20-harmonic trajectory. The Pinocchio cases assemble 1,000
samples on two/14-axis scalar-joint chains. Backend versions >=0.2.0 additionally
measure the named batch API with matching model coordinate names.

Allocation counters apply to the benchmark thread. GNU linker wrappers count
malloc/calloc calls in the executable and static libraries; executable overrides
count ordinary/array/aligned C++ new. The counters are disjoint and their sum is
printed as the tracked total. Direct C allocation inside prebuilt shared
libraries can escape those wrappers. Instrumentation is compiled into these
standalone executables. Pinocchio reports per-batch counts; core cases report
per-call counts.

Use matching installed builds, identical harness sources, the same allowed CPU
(optional `taskset -c CPU ...`) and a quiet machine for comparisons. Record raw
output, CPU/compiler/dependency versions and build flags. `COLCON_IGNORE` keeps
these CMake consumers outside workspace package discovery.

基准只测量预热后的合成调用，报告五轮中位数；不构成硬件时序或轨迹验收保证。
核心用例覆盖 Huber IRLS、ridge 和多谐波轨迹；可选 Pinocchio 用例比较逐样本通用
组装与具名批量入口。库和程序均应以匹配的 Release、编译器和 Eigen 配置构建。

计数针对执行基准的线程，覆盖可执行文件/静态库中被包装的 malloc/calloc，以及
被覆盖函数捕获的 C++ new/new[]/对齐 new；共享库内部直接 C 分配可能不被捕获。
保存原始输出和运行环境，区分每次调用与每批数据的统计单位。

Measure raw CSV read/checksum/destruction as a complete file operation:

```bash
python3 benchmarks/generate_raw_csv.py --samples 200000 --dof 14 --output build/raw-fixture.csv
build/benchmarks/csv_bench build/raw-fixture.csv 1
build/benchmarks/csv_bench build/raw-fixture.csv 5
```

The generator refuses to overwrite and prints the file size and SHA-256. One
repetition skips warmup; use a fresh process for peak-RSS comparisons. Multiple
repetitions warm once and report the median. `peak_rss_kib` comes from Linux
`getrusage` for the entire process, including any warmup. Allocation counts are
per read, with the same scope limitations above. Compare the same fixture and
retain its digest. Filesystem cache, allocator and host activity affect results.

CSV 基准包含读取、校验和计算及返回值销毁；与控制热调用的范围不同。单轮不预热，
峰值 RSS 比较使用独立新进程；多轮先预热一次，报告中位数。生成器拒绝覆盖已有
文件，输出大小与摘要。分块读取仍须为最终数据集保留与行数/关节数成正比的存储，
并非恒定内存接口；减少峰值的代价可能是更多块分配。

Enable `-DROBOT_IDENTIFICATION_BENCH_STRESS_TARGETS=ON` for deterministic analytic
solver campaigns:

```bash
build/benchmarks/solver_stress 3600
build/benchmarks/solver_scale_stress 3600
```

The optional duration is 1–86,400 seconds (default 60). Orthogonal synthetic
systems vary dimensions, conditioning, rank and weights; the first campaign also
covers ridge. The second varies common weights through `1e±300` and design scales
through `1e±130`, using extended-range long-double covariance oracles. Noise stays
bounded relative to the weakest nonzero singular direction so the constructed
reference remains meaningful under floating-point perturbations. These opt-in
correctness programs report case counts and stop on a mismatch. They are separate
from timing benchmarks and do not add CTest timing gates.

The reported parameter error is `norm(actual-reference) / max(1, norm(reference))`;
covariance error is the Frobenius norm of its difference divided by
`max(1e-300, norm(reference_covariance))`, evaluated with stable norms. Parameter
error therefore uses an absolute scale for small reference vectors rather than
being a pure relative error.

可选随机验证使用固定种子和解析正交系统，检查不同维度、秩、条件数、ridge 和
权重范围。宽尺度协方差参考使用扩展范围的 long double。时间参数默认 60 秒，
可显式运行一小时；程序遇到不一致即失败，不将合成性能设为 CI 时序门槛。

After sourcing the installed ROS packages, run an independent 14-axis recorder
load test on Linux:

```bash
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=78 python3 benchmarks/recorder_soak.py \
  --duration 3600 --rate 1000 --qos-depth 1000 \
  --output build/soaks/recorder.json
```

The producer reverses joint order periodically and injects NaN, missing-effort
and duplicate-name messages. Validation checks every recorded joint/torque row,
strict sample order, metadata counts and the raw-file SHA-256. It reports missing
valid/invalid deliveries separately. The default minimum delivery fraction is
0.95 for both classes; use `--minimum-delivery-fraction 1` to require all sent
messages. A passing default run can therefore include transport losses. QoS
depth trades queue memory and callback latency against burst tolerance; these
measurements do not establish loss-free or source-time acquisition. Receipt
timestamps remain process-monotonic. Output must be a new `.json` path; CSV and
logs are retained beside it. A 1,000 Hz, one-hour run writes about 1.8 GB of CSV.

`recorder_soak.py` 自带十四关节合成发布器，独立于控制仓库；检查重排、非法消息、
每行数据、计数与摘要，明确记录未送达数量。默认要求至少 95% 的有效/无效消息
分别送达，可用参数设为 100%。队列深度不改变时间戳来源，也不构成无丢包承诺；
一小时 1000 Hz 测试会保留约 1.8 GB 数据，请使用新的输出路径。
