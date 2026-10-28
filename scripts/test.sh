#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${1:-$ROOT_DIR/build/validation}"
WITH_PINOCCHIO="${2:-}"
WORKERS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
PYTHON_BIN="${PYTHON:-python3}"
if (($# > 2)) || [[ -n "$WITH_PINOCCHIO" && "$WITH_PINOCCHIO" != "--with-pinocchio" ]]; then
  echo "Usage: $0 [BUILD_ROOT] [--with-pinocchio]" >&2
  exit 2
fi
mkdir -p "$BUILD_ROOT"
BUILD_ROOT="$(realpath "$BUILD_ROOT")"
cd "$ROOT_DIR"
cmake -S robot_identification -B "$BUILD_ROOT/core" \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON \
  "-DPython3_EXECUTABLE=$(command -v "$PYTHON_BIN")"
cmake --build "$BUILD_ROOT/core" --parallel "$WORKERS"
ctest --test-dir "$BUILD_ROOT/core" --output-on-failure
cmake --install "$BUILD_ROOT/core" --prefix "$BUILD_ROOT/install"
"$BUILD_ROOT/core/generic_regressor_example"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR/robot_identification_tools:$ROOT_DIR/robot_identification_ros2${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m unittest discover -s robot_identification_tools/test
"$PYTHON_BIN" -m unittest discover -s robot_identification_ros2/test
"$PYTHON_BIN" -m unittest discover -s utils/tests
"$PYTHON_BIN" utils/check_schemas.py

if [[ "$WITH_PINOCCHIO" == "--with-pinocchio" ]]; then
  export CMAKE_PREFIX_PATH="$BUILD_ROOT/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  cmake -S robot_identification_pinocchio -B "$BUILD_ROOT/pinocchio" \
    -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
    "-DPython3_EXECUTABLE=$(command -v "$PYTHON_BIN")"
  cmake --build "$BUILD_ROOT/pinocchio" --parallel "$WORKERS"
  ctest --test-dir "$BUILD_ROOT/pinocchio" --output-on-failure
  cmake --install "$BUILD_ROOT/pinocchio" --prefix "$BUILD_ROOT/install"
fi
git diff --check
echo "RobotIdentification validation passed."
