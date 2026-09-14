#!/usr/bin/env python3
"""Validate and summarize a raw or processed identification CSV."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import BinaryIO, TextIO

# Numeric rows are unquoted, with optional leading ASCII whitespace per field.
# float() completes decimal grammar validation after this inexpensive alphabet
# check, rejecting malformed exponents and repeated signs/decimal points.
_DECIMAL_FIELD = r"[ \t\v\f\r]*[+\-.0-9eE]+"
_DECIMAL_ROW = re.compile(r"[+\-.0-9eE,]+")
_DECIMAL_NUMBER = re.compile(_DECIMAL_FIELD)
_HEX_NUMBER = re.compile(
    r"[ \t\v\f\r]*[+-]?0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)"
    r"(?:[pP][+-]?[0-9]+)?")
_NONZERO_DECIMAL = re.compile(r"[^eE]*[1-9]")
_NONZERO_HEX = re.compile(r"[^pP]*[1-9a-fA-F]")
# A fixed decimal shorter than this cannot reach even the smallest normal
# magnitude. This conservative bound lets ordinary zero rows avoid rescanning.
_MIN_FIXED_RANGE_LENGTH = -sys.float_info.min_10_exp


def _parse_number(field: str, line_number: int, column: int) -> float:
    location = f"line {line_number}, column {column}"
    hexadecimal = _HEX_NUMBER.fullmatch(field) is not None
    decimal = _DECIMAL_NUMBER.fullmatch(field) is not None
    try:
        value = float.fromhex(field) if hexadecimal else float(field)
    except (ValueError, OverflowError) as error:
        raise ValueError(
            f"{location} contains an invalid or out-of-range number") from error
    if not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite value")
    if not hexadecimal and not decimal:
        raise ValueError(f"{location} contains invalid numeric syntax")
    nonzero = _NONZERO_HEX if hexadecimal else _NONZERO_DECIMAL
    if value == 0.0 and nonzero.match(field):
        raise ValueError(
            f"{location} contains a nonzero number that underflows to zero")
    return value


def _parse_numeric_row(line: str, fields: list[str],
                       line_number: int) -> list[float]:
    if _DECIMAL_ROW.fullmatch(line):
        try:
            values = list(map(float, fields))
        except ValueError:
            pass  # Resolve the exact invalid column through the diagnostic path.
        else:
            if not all(map(math.isfinite, values)):
                for column, value in enumerate(values, start=1):
                    if not math.isfinite(value):
                        raise ValueError(
                            f"line {line_number}, column {column} contains a non-finite value"
                        )
            if ("e" in line or "E" in line or
                    len(line) >= _MIN_FIXED_RANGE_LENGTH) and 0.0 in values:
                for column, (field, value) in enumerate(zip(fields, values),
                                                        start=1):
                    if value == 0.0 and _NONZERO_DECIMAL.match(field):
                        raise ValueError(
                            f"line {line_number}, column {column} contains a nonzero number that underflows to zero"
                        )
            return values
    return [
        _parse_number(field, line_number, column)
        for column, field in enumerate(fields, start=1)
    ]


class _HashingReader(io.RawIOBase):
    """Hash the bytes consumed by the text decoder without retaining the file."""

    def __init__(self, source: BinaryIO) -> None:
        super().__init__()
        self._source = source
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        count = self._source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def inspect_dataset(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        return _inspect_stream(stream)


def inspect_dataset_with_digest(path: Path) -> tuple[dict[str, object], str]:
    """Validate and hash the same byte stream with one bounded-memory read."""
    with path.open("rb", buffering=0) as source:
        reader = _HashingReader(source)
        with io.TextIOWrapper(io.BufferedReader(reader),
                              encoding="utf-8",
                              newline="") as stream:
            summary = _inspect_stream(stream)
        return summary, reader.digest.hexdigest()


def _inspect_stream(stream: TextIO) -> dict[str, object]:
    try:
        header = next(stream).removesuffix("\n").removesuffix("\r").split(",")
    except StopIteration as error:
        raise ValueError("dataset is empty") from error
    if not header or header[0] != "time_s":
        raise ValueError("first column must be time_s")
    q_columns = [name for name in header if name.startswith("q/")]
    if not q_columns:
        raise ValueError("dataset has no position columns")
    joint_names = [name[2:] for name in q_columns]
    if (any(not name or any(character in name
                            for character in ",\r\n")
            for name in joint_names) or
            len(set(joint_names)) != len(joint_names)):
        raise ValueError(
            "position joint names must be unique, non-empty, and CSV-safe")
    raw_header = (["time_s"] + [f"q/{name}" for name in joint_names] +
                  [f"tau/{name}" for name in joint_names])
    processed_header = (["time_s"] + [f"q/{name}" for name in joint_names] +
                        [f"dq/{name}" for name in joint_names] +
                        [f"ddq/{name}" for name in joint_names] +
                        [f"tau/{name}" for name in joint_names])
    if header == raw_header:
        kind = "raw"
    elif header == processed_header:
        kind = "processed"
    else:
        raise ValueError("columns do not match the identification CSV contract")

    sample_count = 0
    first_time = 0.0
    previous_time = 0.0
    minimum_interval = math.inf
    maximum_interval = 0.0
    maximum_torque = [0.0] * len(joint_names)
    torque_offset = 1 + len(joint_names) * (1 if kind == "raw" else 3)
    for line_number, line in enumerate(stream, start=2):
        line = line.removesuffix("\n").removesuffix("\r")
        if not line:
            continue
        row = line.split(",")
        if len(row) != len(header):
            raise ValueError(f"line {line_number} has the wrong field count")
        values = _parse_numeric_row(line, row, line_number)
        current_time = values[0]
        if sample_count:
            if current_time <= previous_time:
                raise ValueError(
                    f"line {line_number} timestamp is not strictly increasing")
            interval = current_time - previous_time
            if not math.isfinite(interval):
                raise ValueError(
                    f"line {line_number} timestamp interval is non-finite")
            if interval < minimum_interval:
                minimum_interval = interval
            if interval > maximum_interval:
                maximum_interval = interval
        else:
            first_time = current_time
        previous_time = current_time
        sample_count += 1
        for joint, value in enumerate(values[torque_offset:]):
            magnitude = abs(value)
            if magnitude > maximum_torque[joint]:
                maximum_torque[joint] = magnitude
    if sample_count < 3:
        raise ValueError("dataset needs at least three samples")
    duration = previous_time - first_time
    if not math.isfinite(duration):
        raise ValueError("dataset duration is non-finite")
    mean_rate = (sample_count - 1) / duration
    if not math.isfinite(mean_rate):
        raise ValueError("dataset mean sample rate is non-finite")
    return {
        "kind": kind,
        "joint_names": joint_names,
        "sample_count": sample_count,
        "duration_s": duration,
        "mean_rate_hz": mean_rate,
        "minimum_interval_s": minimum_interval,
        "maximum_interval_s": maximum_interval,
        "maximum_absolute_torque": dict(zip(joint_names, maximum_torque)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        summary = inspect_dataset(args.dataset)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"{summary['kind']} dataset: {summary['sample_count']} samples, "
            f"{summary['mean_rate_hz']:.3f} Hz, joints={','.join(summary['joint_names'])}"
        )


if __name__ == "__main__":
    main()
