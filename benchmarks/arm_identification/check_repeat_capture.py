"""Verify a median refit and its standalone refit against the original captures."""

import json
from pathlib import Path
import sys

import numpy as np


def read_report(capture):
    return json.loads((capture / "report.json").read_text(encoding="utf-8"))


def check(median_dir, refit_dir, captures):
    primary_dir = captures[0]
    primary_report = read_report(primary_dir)
    arm_names = list(primary_report["robots"])
    aggregation = {
        "method": "median",
        "captures": [str(capture.resolve()) for capture in captures],
    }
    for output in (median_dir, refit_dir):
        report = read_report(output)
        if report["training_capture_aggregation"] != aggregation:
            raise ValueError(
                f"{output} has different training capture provenance")
        if list(report["robots"]) != arm_names:
            raise ValueError(
                f"{output} has different arms from the primary capture")

    primary_files = ["calibration.csv", "holdout.csv"]
    if primary_report["replay"]:
        primary_files.append("holdout_dense.csv")
    for arm in arm_names:
        for filename in primary_files:
            original = (primary_dir / arm / filename).read_bytes()
            for output in (median_dir, refit_dir):
                path = output / arm / filename
                if path.read_bytes() != original:
                    raise ValueError(f"{path} differs from the primary capture")
        for filename in ("train_baseline_raw.csv", "train_optimized_raw.csv"):
            median_csv = median_dir / arm / filename
            refit_csv = refit_dir / arm / filename
            inputs = [
                np.loadtxt(capture / arm / filename, delimiter=",", skiprows=1)
                for capture in captures
            ]
            observed = np.loadtxt(median_csv, delimiter=",", skiprows=1)
            np.testing.assert_array_equal(observed,
                                          np.median(inputs, axis=0),
                                          err_msg=str(median_csv))
            if refit_csv.read_bytes() != median_csv.read_bytes():
                raise ValueError(f"{refit_csv} differs from the median capture")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(
            "usage: check_repeat_capture.py MEDIAN REFIT CAPTURE REPEAT [REPEAT ...]"
        )
    median_dir, refit_dir, *captures = map(Path, sys.argv[1:])
    check(median_dir, refit_dir, captures)
