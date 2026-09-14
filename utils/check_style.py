# ----------------------------------------------------------------------------
# Copyright (c) 2026 RobotServer contributors.
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import subprocess
import re
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import multiprocessing
from functools import partial
import time
import sys

# Yapf requires python 3.6+
if sys.version_info < (3, 6):
    raise RuntimeError(
        "Requires Python 3.6+, currently using Python {}.{}.".format(
            sys.version_info.major, sys.version_info.minor))


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
    print("Using yapf version {}".format(yapf.__version__))


PYTHON_EXCLUDE_DIRS = [
    "build",
    ".git",
    "third_party",
    "script",
    "deb_build",
    "install",
    "log",
    ".cache",
]

PYTHON_EXCLUDE_FILES = [
    "commit_format.py",
]

CPP_EXCLUDE_DIRS = [
    "build", "install", "log", ".git", "third_party", "deb_build", ".cache"
]


def _glob_files(exclude_directories,
                extensions,
                exclude_files=None,
                root_dir=None):
    """
    Find files with certain extensions in the root directory recursively,
    excluding specified directories and files.

    Args:
        exclude_directories: list of directories to exclude, relative to the root repo directory.
        extensions: list of extensions, e.g. ["cpp", "h"].
        exclude_files: list of specific filenames to exclude.

    Return:
        List of file paths.
    """
    root_dir = Path(root_dir) if root_dir is not None else Path(
        __file__).resolve().parent.parent
    exclude_paths = {root_dir / directory for directory in exclude_directories}
    excluded_names = set(exclude_files or [])
    suffixes = tuple(f".{extension}" for extension in extensions)
    files = []

    def on_error(error):
        raise error

    for directory, children, names in os.walk(root_dir, onerror=on_error):
        parent = Path(directory)
        children[:] = sorted(
            name for name in children
            if parent / name not in exclude_paths and name != "__pycache__")
        for name in names:
            path = parent / name
            if name not in excluded_names and name.endswith(
                    suffixes) and path.is_file():
                files.append(str(path))
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
    """
    Find clang-format:
      - not found: throw exception
      - version mismatch: print warning
    """
    preferred_clang_format_name = "clang-format-18"
    preferred_version_major = 18
    clang_format_bin = shutil.which(preferred_clang_format_name)
    if clang_format_bin is None:
        clang_format_bin = shutil.which("clang-format")
    if clang_format_bin is None:
        raise RuntimeError("clang-format not found.")
    version_str = subprocess.check_output([clang_format_bin, "--version"
                                          ]).decode("utf-8").strip()
    try:
        m = re.match("^clang-format version ([0-9.]*).*$", version_str)
        if m:
            version_str = m.group(1)
            version_str_token = version_str.split(".")
            major = int(version_str_token[0])
            if major != preferred_version_major:
                print("Warning: {} required, but got {}.".format(
                    preferred_clang_format_name, version_str))
        else:
            raise
    except:
        print("Warning: failed to parse clang-format version {}, "
              "please ensure {} is used.".format(version_str,
                                                 preferred_clang_format_name))
    print("Using clang-format version {}.".format(version_str))

    return clang_format_bin


class CppFormatter:

    standard_header = """// ----------------------------------------------------------------------------
// Copyright (c) 2026 RobotServer contributors.
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
"""

    def __init__(self, file_paths, clang_format_bin):
        self.file_paths = file_paths
        self.clang_format_bin = clang_format_bin

    @staticmethod
    def _check_style(file_path, clang_format_bin):
        """
        Returns (true, true) if (style, header) is valid.
        """

        cmd = [
            clang_format_bin,
            "-style=file",
            "-output-replacements-xml",
            file_path,
        ]
        result = subprocess.check_output(cmd).decode("utf-8")
        if "<replacement " in result:
            is_valid_style = False
        else:
            is_valid_style = True
        return (is_valid_style, True)

    @staticmethod
    def _apply_style(file_path, clang_format_bin):
        cmd = [
            clang_format_bin,
            "-style=file",
            "-i",
            file_path,
        ]
        subprocess.check_output(cmd)

    def run(self, apply, no_parallel, verbose, jobs=None):
        if apply:
            print("Applying C++/CUDA style...")
        else:
            print("Checking C++/CUDA style...")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print("> {}".format(file_path))

        start_time = time.time()
        if no_parallel or len(self.file_paths) <= 1:
            is_valid_files = map(
                partial(self._check_style,
                        clang_format_bin=self.clang_format_bin),
                self.file_paths)
        else:
            # clang-format does its CPU work in child processes; threads
            # avoid an additional Python process for every concurrent check.
            with ThreadPoolExecutor(max_workers=_worker_count(
                    len(self.file_paths), jobs)) as pool:
                is_valid_files = list(
                    pool.map(
                        partial(self._check_style,
                                clang_format_bin=self.clang_format_bin),
                        self.file_paths))

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.clang_format_bin)
            if not is_valid_header:
                wrong_header_files.append(file_path)
        print("Formatting takes {:.2f}s".format(time.time() - start_time))

        return (changed_files, wrong_header_files)


class PythonFormatter:

    standard_header = """# ----------------------------------------------------------------------------
# Copyright (c) 2026 RobotServer contributors.
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_style(file_path, style_config):
        """
        Returns (true, true) if (style, header) is valid.
        """

        from yapf.yapflib.yapf_api import FormatFile

        _, _, changed = FormatFile(file_path,
                                   style_config=style_config,
                                   in_place=False)
        return (not changed, True)

    @staticmethod
    def _apply_style(file_path, style_config):
        from yapf.yapflib.yapf_api import FormatFile

        _, _, _ = FormatFile(file_path,
                             style_config=style_config,
                             in_place=True)

    def run(self, apply, no_parallel, verbose, jobs=None):
        if apply:
            print("Applying Python style...")
        else:
            print("Checking Python style...")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print("> {}".format(file_path))

        start_time = time.time()
        if no_parallel or len(self.file_paths) <= 1:
            is_valid_files = map(
                partial(self._check_style, style_config=self.style_config),
                self.file_paths)
        else:
            with multiprocessing.Pool(_worker_count(len(self.file_paths),
                                                    jobs)) as pool:
                is_valid_files = pool.map(
                    partial(self._check_style, style_config=self.style_config),
                    self.file_paths)

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.style_config)
            if not is_valid_header:
                wrong_header_files.append(file_path)

        print("Formatting takes {:.2f}s".format(time.time() - start_time))
        return (changed_files, wrong_header_files)


def _filter_files(files, ignored_patterns):
    return [
        file for file in files if not any(
            [ignored_pattern in file for ignored_pattern in ignored_patterns])
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=False,
        help="Apply style to files in-place.",
    )
    parser.add_argument(
        "--no_parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel execution.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="If true, prints file names while formatting.",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_jobs,
        help="Maximum parallel checks (default: min(CPU count, 16)).")
    args = parser.parse_args()

    # Check formatting libs
    _require_yapf()
    clang_format_bin = _find_clang_format()
    pwd = Path(os.path.dirname(os.path.abspath(__file__)))
    python_style_config = str(pwd.parent / ".style.yapf")

    # Check or apply style
    cpp_formatter = CppFormatter(
        _glob_files(CPP_EXCLUDE_DIRS,
                    ["h", "hpp", "cpp", "cuh", "cu", "isph", "ispc", "h.in"]),
        clang_format_bin=clang_format_bin)
    python_formatter = PythonFormatter(_glob_files(
        PYTHON_EXCLUDE_DIRS, ["py"], exclude_files=PYTHON_EXCLUDE_FILES),
                                       style_config=python_style_config)

    changed_files = []
    wrong_header_files = []
    changed_files_cpp, wrong_header_files_cpp = cpp_formatter.run(
        apply=args.apply,
        no_parallel=args.no_parallel,
        verbose=args.verbose,
        jobs=args.jobs)
    changed_files.extend(changed_files_cpp)
    wrong_header_files.extend(wrong_header_files_cpp)
    print(f"apply cpp file num is {len(changed_files_cpp)}")

    changed_files_python, wrong_header_files_python = python_formatter.run(
        apply=args.apply,
        no_parallel=args.no_parallel,
        verbose=args.verbose,
        jobs=args.jobs)
    changed_files.extend(changed_files_python)
    wrong_header_files.extend(wrong_header_files_python)
    print(f"apply python file num is {len(changed_files_python)}")

    if len(changed_files) == 0 and len(wrong_header_files) == 0:
        print("All files passed style check.")
        exit(0)

    if args.apply:
        if len(changed_files) != 0:
            print("Style applied to the following files:")
            print("\n".join(changed_files))
            exit(1)
        if len(wrong_header_files) != 0:
            print("Please correct license header *manually* in the following "
                  "files (see util/check_style.py for the standard header):")
            print("\n".join(wrong_header_files))
            exit(1)
    else:
        error_files_no_duplicates = list(set(changed_files +
                                             wrong_header_files))
        if len(error_files_no_duplicates) != 0:
            print("Style error found in the following files:")
            print("\n".join(error_files_no_duplicates))
            exit(1)
