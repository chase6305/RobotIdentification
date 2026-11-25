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
       ./scripts/run_arm_identification.sh --demo [--output DIRECTORY]
       ./scripts/run_arm_identification.sh --refit CAPTURE_DIRECTORY --output DIRECTORY [fit options]
       ./scripts/run_arm_identification.sh --build-only

This builds the optional C++ benchmark and runs the two arms in HumanoidAssets.
The asset checkout defaults to ../HumanoidAssets relative to the repository.
Pass --assets-root to choose a different location or --robot pm01 / openarm
to run one arm. Pass --acquisition rollout for torque-driven simulation
or --design-only to export automatic reference points.
The demo runs capture, raw-data fitting, forward replay, and model export.
Use a new output directory for each experiment. The demo creates one by default.
The refit reads a saved capture and reruns preprocessing and fitting.

PYTHON selects the base interpreter. ARM_IDENTIFICATION_BUILD_DIR selects the
directory for the virtual environment and native build products.
EOF
  exit 0
fi
ENTRY="run_experiment.py"
if [[ "$1" == --demo ]]; then
  shift
  demo_output="$ROOT_DIR/build/arm-demo-$(date +%Y%m%d-%H%M%S)-$$"
  set -- --acquisition rollout --fit-input raw --raw-estimator periodic-fourier \
    --timestamp-jitter-ms auto --actuator-scale 1 --replay --estimate-payload --estimate-arm \
    --summary --output "$demo_output" "$@"
elif [[ "$1" == --refit ]]; then
  shift
  ENTRY="refit_capture.py"
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
REQUIREMENTS="$BENCHMARK_DIR/requirements.txt"
BUILD_LOG="$BUILD_DIR/build.log"

# Guard the shared build directory during the dependency and native builds.
exec {build_lock}>"$BUILD_DIR/build.lock"
flock "$build_lock"

set +e
(
  set -e
  if [[ ! -x "$VENV/bin/pip" ]]; then
    "$PYTHON_BIN" -m venv "$VENV"
  fi
  if ! cmp -s "$REQUIREMENTS" "$VENV/installed-arm-requirements.txt"; then
    PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-45}" "$VENV/bin/python" -m pip \
      --disable-pip-version-check install --no-input --only-binary=:all: -r "$REQUIREMENTS"
    cp "$REQUIREMENTS" "$VENV/installed-arm-requirements.txt"
  fi
  pin_prefix="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/cmeel.prefix")')"
  export CMAKE_PREFIX_PATH="$PREFIX:$pin_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$pin_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  native_stamp="$BUILD_DIR/native-arm-requirements.txt"
  configure_build() {
    local source="$1" target="$2" cached_source=""
    shift 2
    if [[ -f "$target/CMakeCache.txt" ]]; then
      cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$target/CMakeCache.txt")"
    fi
    if [[ "$cached_source" != "$source" ]] || ! cmp -s "$REQUIREMENTS" "$native_stamp"; then
      cmake -S "$source" -B "$target" -DCMAKE_BUILD_TYPE=Release "$@"
    fi
    cmake --build "$target" --parallel "$WORKERS"
  }
  for package in robot_identification robot_identification_pinocchio; do
    configure_build "$ROOT_DIR/$package" "$BUILD_DIR/$package" -DBUILD_TESTING=OFF
    cmake --install "$BUILD_DIR/$package" --prefix "$PREFIX"
  done
  configure_build "$BENCHMARK_DIR" "$BUILD_DIR/benchmark"
  cp "$REQUIREMENTS" "$native_stamp"
) > "$BUILD_LOG" 2>&1
build_status=$?
set -e
if [[ "$build_status" != 0 ]]; then
  tail -n 40 "$BUILD_LOG" >&2 || true
  printf 'Arm benchmark build log: %s\n' "$BUILD_LOG" >&2
  exit "$build_status"
fi
exec {build_lock}>&-

if [[ $# == 1 && "$1" == --build-only ]]; then
  printf '%s\n' "$BUILD_DIR/benchmark/arm_identification"
  exit 0
fi
PIN_PREFIX="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/cmeel.prefix")')"
export LD_LIBRARY_PATH="$PIN_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$ENTRY" == run_experiment.py ]]; then
  set -- --assets-root "$ROOT_DIR/../HumanoidAssets" "$@"
fi
exec "$VENV/bin/python" "$BENCHMARK_DIR/$ENTRY" \
  --executable "$BUILD_DIR/benchmark/arm_identification" "$@"
