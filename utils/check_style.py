# ----------------------------------------------------------------------------
# Copyright (c) 2026 RobotServer contributors.
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Check or apply the repository's C++ and Python formatting."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

EXCLUDE_DIRS = {
    ".venv", "build", ".git", "third_party", "deb_build", "install", "log",
    ".cache", "__pycache__"
}


def _require_yapf():
    try:
        import yapf
    except ImportError as error:
        raise ImportError(
            "yapf not found. Install with `pip install yapf==0.43.0`."
        ) from error
    if yapf.__version__ != "0.43.0":
        raise RuntimeError(
            "yapf 0.43.0 required. Install with `pip install yapf==0.43.0`.")
    print(f"Using yapf version {yapf.__version__}")


def _glob_files(exclude_directories, extensions, root_dir=None):
    """Find source files while pruning generated directories at every depth."""
    root_dir = Path(root_dir) if root_dir is not None else Path(
        __file__).resolve().parent.parent
    suffixes = tuple(f".{extension}" for extension in extensions)
    files = []

    def on_error(error):
        raise error

    for directory, children, names in os.walk(root_dir, onerror=on_error):
        parent = Path(directory)
        children[:] = sorted(
            name for name in children if name not in exclude_directories)
        files.extend(
            str(parent / name)
            for name in names
            if name.endswith(suffixes) and (parent / name).is_file())
    return sorted(files)


def _positive_jobs(value):
    jobs = int(value)
    if jobs <= 0:
        raise argparse.ArgumentTypeError("jobs must be a positive integer")
    return jobs


def _worker_count(file_count, jobs=None):
    limit = jobs if jobs is not None else min(multiprocessing.cpu_count(), 16)
    return max(1, min(file_count, limit))


def _find_clang_format():
    binary = shutil.which("clang-format-18") or shutil.which("clang-format")
    if binary is None:
        raise RuntimeError("clang-format not found.")
    version = subprocess.check_output([binary, "--version"], text=True).strip()
    match = re.search(r"clang-format version (\d+)(?:\.\d+)*", version)
    if match is None or int(match[1]) != 18:
        print(f"Warning: clang-format-18 required, but got {version}.")
    print(f"Using {version}.")
    return binary


def _format_cpp(file_path, *, binary, apply):
    result = subprocess.check_output(
        [binary, "-style=file", "-output-replacements-xml", file_path],
        text=True)
    changed = "<replacement " in result
    if changed and apply:
        subprocess.check_call([binary, "-style=file", "-i", file_path])
    return changed


def _format_python(file_path, *, style_config, apply):
    from yapf.yapflib.yapf_api import FormatFile

    _, _, changed = FormatFile(file_path,
                               style_config=style_config,
                               in_place=apply)
    return changed


def _run_formatter(files, formatter, pool_type, args):
    if args.verbose:
        for file_path in files:
            print(f"> {file_path}")
    start = time.monotonic()
    if args.no_parallel or len(files) <= 1:
        results = list(map(formatter, files))
    else:
        # clang-format does its work in child processes; YAPF runs Python code.
        with pool_type(_worker_count(len(files), args.jobs)) as pool:
            results = list(pool.map(formatter, files))
    print(f"Formatting takes {time.monotonic() - start:.2f}s")
    return [path for path, changed in zip(files, results) if changed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply",
                        action="store_true",
                        help="Apply style to files in-place.")
    parser.add_argument("--no_parallel",
                        action="store_true",
                        help="Disable parallel execution.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print file names while formatting.")
    parser.add_argument(
        "--jobs",
        type=_positive_jobs,
        help="Maximum parallel checks (default: min(CPU count, 16)).")
    args = parser.parse_args()
    _require_yapf()
    binary = _find_clang_format()
    root = Path(__file__).resolve().parent.parent
    changed_files = []
    for label, extensions, formatter, pool_type in (
        ("C++/CUDA", ["h", "hpp", "cpp", "cuh", "cu", "isph", "ispc", "h.in"],
         partial(_format_cpp, binary=binary,
                 apply=args.apply), ThreadPoolExecutor),
        ("Python", ["py"],
         partial(_format_python,
                 style_config=str(root / ".style.yapf"),
                 apply=args.apply), multiprocessing.Pool),
    ):
        print(f"{'Applying' if args.apply else 'Checking'} {label} style...")
        changed_files.extend(
            _run_formatter(_glob_files(EXCLUDE_DIRS, extensions), formatter,
                           pool_type, args))
    if not changed_files:
        print("All files passed style check.")
        return 0
    print("Style applied to the following files:" if args.
          apply else "Style error found in the following files:")
    print("\n".join(changed_files))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
