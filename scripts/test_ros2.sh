#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# > 1)); then
  echo "Usage: $0 [BUILD_ROOT]" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [BUILD_ROOT]"
  exit 0
fi
BUILD_ROOT="${1:-$ROOT_DIR/build/ros2}"
WORKERS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
PYTHON_BIN="${PYTHON:-python3}"
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer" >&2
  exit 2
fi
if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is required; source a ROS 2 environment first" >&2
  exit 1
fi
# Build and test exactly the same source packages. Generated CMake consumers and
# dependency checkouts must not become accidental colcon test inputs.
mapfile -d '' -t PACKAGE_PATHS < <(find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -type f -name package.xml -printf '%h\0' | sort -z)
if ((${#PACKAGE_PATHS[@]} == 0)); then
  echo "No root-level ROS/colcon packages found under $ROOT_DIR" >&2
  exit 1
fi
mkdir -p -- "$BUILD_ROOT"
BUILD_ROOT="$(realpath -- "$BUILD_ROOT")"
cd "$ROOT_DIR"
export CMAKE_BUILD_PARALLEL_LEVEL="$WORKERS"
export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j$WORKERS"
colcon --log-base "$BUILD_ROOT/log" build --paths "${PACKAGE_PATHS[@]}" \
  --build-base "$BUILD_ROOT/build" --install-base "$BUILD_ROOT/install" \
  --parallel-workers "$WORKERS" \
  --cmake-args -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
set +u
source "$BUILD_ROOT/install/setup.bash"
set -u
colcon --log-base "$BUILD_ROOT/test-log" test --paths "${PACKAGE_PATHS[@]}" \
  --build-base "$BUILD_ROOT/build" --install-base "$BUILD_ROOT/install" \
  --return-code-on-test-failure
colcon test-result --test-result-base "$BUILD_ROOT/build" --verbose
"$PYTHON_BIN" utils/ros2_recording_smoke.py
if [[ -e "$ROOT_DIR/.git" ]]; then
  git diff --check
else
  echo "Skipping git whitespace check: source tree has no .git metadata."
fi
echo "RobotIdentification ROS 2 validation passed."
