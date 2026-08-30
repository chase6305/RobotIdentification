#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"

if ! command -v clang-format-18 >/dev/null 2>&1 && ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is required for C/C++ style checks" >&2
  exit 1
fi

case "$MODE" in
  check) exec python3 "$ROOT_DIR/utils/check_style.py" ;;
  --apply) exec python3 "$ROOT_DIR/utils/check_style.py" --apply ;;
  *) echo "Usage: $0 [--apply]" >&2; exit 2 ;;
esac
