#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="$ROOT_DIR/benchmarks/arm_identification"
BUILD_DIR="${ARM_IDENTIFICATION_BUILD_DIR:-$ROOT_DIR/build/arm-identification}"
PYTHON_BIN="${PYTHON:-python3}"
WORKERS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

if [[ ${1:-} == -h || ${1:-} == --help || $# == 0 ]]; then
  cat <<'EOF'
Usage: ./scripts/run_arm_identification.sh --output DIRECTORY [experiment options]
       ./scripts/run_arm_identification.sh --build-only

This builds the optional C++ benchmark and runs the two arms in HumanoidAssets.
The asset checkout defaults to ../HumanoidAssets relative to the repository.
Pass --assets-root to choose a different location or --robot pm01 / openarm
to run one arm. Pass --acquisition rollout for torque-driven simulation
or --design-only to export automatic reference points.
Use a new output directory for each experiment.

PYTHON selects the base interpreter. ARM_IDENTIFICATION_BUILD_DIR selects the
directory for the virtual environment and native build products.
EOF
  exit 0
fi
if [[ "$WORKERS" != 1 && "$WORKERS" != 2 ]]; then
  echo "CMAKE_BUILD_PARALLEL_LEVEL must be 1 or 2" >&2
  exit 2
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL="$WORKERS" CTEST_PARALLEL_LEVEL=1 MAX_JOBS="$WORKERS" RAYON_NUM_THREADS="$WORKERS"

mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
VENV="$BUILD_DIR/venv"
PREFIX="$BUILD_DIR/install"

# Guard the shared build directory during the dependency and native builds.
exec {build_lock}>"$BUILD_DIR/build.lock"
flock "$build_lock"
if [[ ! -x "$VENV/bin/pip" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
if ! cmp -s "$BENCHMARK_DIR/requirements.txt" "$VENV/installed-arm-requirements.txt"; then
  "$VENV/bin/python" -m pip --disable-pip-version-check install --no-input \
    --only-binary=:all: -r "$BENCHMARK_DIR/requirements.txt" >&2
  cp "$BENCHMARK_DIR/requirements.txt" "$VENV/installed-arm-requirements.txt"
fi
PIN_PREFIX="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/cmeel.prefix")')"
export CMAKE_PREFIX_PATH="$PREFIX:$PIN_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$PIN_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for package in robot_identification robot_identification_pinocchio; do
  cmake -S "$ROOT_DIR/$package" -B "$BUILD_DIR/$package" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF >&2
  cmake --build "$BUILD_DIR/$package" --parallel "$WORKERS" >&2
  cmake --install "$BUILD_DIR/$package" --prefix "$PREFIX" >&2
done
cmake -S "$BENCHMARK_DIR" -B "$BUILD_DIR/benchmark" -DCMAKE_BUILD_TYPE=Release >&2
cmake --build "$BUILD_DIR/benchmark" --parallel "$WORKERS" >&2
exec {build_lock}>&-

if [[ $# == 1 && "$1" == --build-only ]]; then
  printf '%s\n' "$BUILD_DIR/benchmark/arm_identification"
  exit 0
fi
exec "$VENV/bin/python" "$BENCHMARK_DIR/run_experiment.py" \
  --assets-root "$ROOT_DIR/../HumanoidAssets" --executable "$BUILD_DIR/benchmark/arm_identification" "$@"
