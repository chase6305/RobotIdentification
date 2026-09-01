#!/usr/bin/env python3
"""Validate and summarize a raw or processed identification CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def inspect_dataset(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("dataset is empty") from error
        if not header or header[0] != "time_s":
            raise ValueError("first column must be time_s")
        q_columns = [name for name in header if name.startswith("q/")]
        if not q_columns:
            raise ValueError("dataset has no position columns")
        joint_names = [name[2:] for name in q_columns]
        if (any(not name or any(character in name for character in ",\r\n")
                for name in joint_names) or
                len(set(joint_names)) != len(joint_names)):
            raise ValueError(
                "position joint names must be unique, non-empty, and CSV-safe")
        raw_header = (["time_s"] + [f"q/{name}" for name in joint_names] +
                      [f"tau/{name}" for name in joint_names])
        processed_header = (
            ["time_s"] + [f"q/{name}" for name in joint_names] +
            [f"dq/{name}" for name in joint_names] +
            [f"ddq/{name}" for name in joint_names] +
            [f"tau/{name}" for name in joint_names])
        if header == raw_header:
            kind = "raw"
        elif header == processed_header:
            kind = "processed"
        else:
            raise ValueError("columns do not match the identification CSV contract")

        times: list[float] = []
        maximum_torque = [0.0] * len(joint_names)
        torque_offset = 1 + len(joint_names) * (1 if kind == "raw" else 3)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"line {line_number} has the wrong field count")
            try:
                values = [float(field) for field in row]
            except ValueError as error:
                raise ValueError(f"line {line_number} contains a non-number") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"line {line_number} contains a non-finite value")
            if times and values[0] <= times[-1]:
                raise ValueError(f"line {line_number} timestamp is not strictly increasing")
            times.append(values[0])
            for joint, value in enumerate(values[torque_offset:]):
                maximum_torque[joint] = max(maximum_torque[joint], abs(value))
    if len(times) < 3:
        raise ValueError("dataset needs at least three samples")
    intervals = [after - before for before, after in zip(times, times[1:])]
    duration = times[-1] - times[0]
    return {
        "kind": kind,
        "joint_names": joint_names,
        "sample_count": len(times),
        "duration_s": duration,
        "mean_rate_hz": (len(times) - 1) / duration,
        "minimum_interval_s": min(intervals),
        "maximum_interval_s": max(intervals),
        "maximum_absolute_torque": dict(zip(joint_names, maximum_torque)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    summary = inspect_dataset(args.dataset)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"{summary['kind']} dataset: {summary['sample_count']} samples, "
              f"{summary['mean_rate_hz']:.3f} Hz, joints={','.join(summary['joint_names'])}")


if __name__ == "__main__":
    main()
