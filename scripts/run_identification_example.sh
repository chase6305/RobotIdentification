#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build/identification-example}"
WORKERS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

cmake -S "$ROOT_DIR/robot_identification" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DROBOT_IDENTIFICATION_BUILD_EXAMPLES=ON
cmake --build "$BUILD_DIR" --parallel "$WORKERS"
cmake -E chdir "$BUILD_DIR" ctest --output-on-failure
"$BUILD_DIR/generic_regressor_example"
