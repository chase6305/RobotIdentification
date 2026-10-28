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
