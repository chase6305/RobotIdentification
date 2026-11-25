"""Read and write the processed and recorder-format arm CSV files."""

import csv
from itertools import zip_longest
import warnings

import numpy as np

_MEDIAN_BLOCK_ROWS = 256
_CSV_FLOAT_FORMAT = "%.17g"


def dataset_header(joints, fields=("q", "dq", "ddq", "tau")):
    return ["time_s"
           ] + [f"{name}/{joint}" for name in fields for joint in joints]


def _check_header(stream, path, expected):
    rows = csv.reader(iter(stream.readline, ""))
    if next(rows, None) != expected:
        raise ValueError(f"unexpected dataset header in {path}")


def read_motion(path, joints, *, max_rows=None):
    header = dataset_header(joints)
    with path.open(newline="", encoding="utf-8") as stream:
        _check_header(stream, path, header)
        motion = np.loadtxt(stream, delimiter=",", ndmin=2, max_rows=max_rows)
    if (motion.shape[0] < 3 or motion.shape[1] != len(header) or
            not np.isfinite(motion).all() or
            not np.all(np.diff(motion[:, 0]) > 0)):
        raise ValueError(f"invalid motion samples in {path}")
    return motion


def write_dataset(path, labels, header):
    np.savetxt(path,
               labels,
               fmt=_CSV_FLOAT_FORMAT,
               delimiter=",",
               header=",".join(header),
               comments="")


def _read_csv_block(stream):
    # NumPy skips blank rows and comments and reports an empty block at EOF.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                r"loadtxt: input contained no data:",
                                category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            r"Input line [0-9]+ contained no data and will not be counted",
            category=UserWarning)
        return np.loadtxt(iter(stream.readline, ""),
                          delimiter=",",
                          ndmin=2,
                          max_rows=_MEDIAN_BLOCK_ROWS)


def _raw_blocks(path, header):
    position = None
    while True:
        # Close each file between blocks so the descriptor count stays fixed.
        with path.open(newline="", encoding="utf-8") as stream:
            if position is None:
                _check_header(stream, path, header)
            else:
                stream.seek(position)
            samples = _read_csv_block(stream)
            position = stream.tell()
        if not len(samples):
            return
        if samples.shape[1] != len(header) or not np.isfinite(samples).all():
            raise ValueError(f"invalid motion samples in {path}")
        yield samples


def write_median_raw(path, inputs, joints):
    """Write median joint positions and torques at matching timestamps."""
    primary_path, *repeat_paths = inputs
    header = dataset_header(joints, ("q", "tau"))
    row_format = ",".join([_CSV_FLOAT_FORMAT] * len(header)) + "\n"
    capture_blocks = [
        _raw_blocks(capture_path, header) for capture_path in inputs
    ]
    rows_written = 0
    previous_time = -np.inf
    with path.open("w", newline="", encoding="utf-8") as output:
        output.write(",".join(header) + "\n")
        for primary_rows, *repeat_rows in zip_longest(*capture_blocks):
            if primary_rows is None:
                raise ValueError(
                    f"repeat capture has more samples than {primary_path}")
            timestamps = primary_rows[:, 0]
            if timestamps[0] <= previous_time or np.any(
                    timestamps[1:] <= timestamps[:-1]):
                raise ValueError(f"invalid motion samples in {primary_path}")
            for repeat_path, samples in zip(repeat_paths, repeat_rows):
                if samples is None or not np.array_equal(
                        samples[:, 0], timestamps):
                    raise ValueError(
                        f"repeat capture timestamps differ in {repeat_path}")

            channels = np.stack(
                [rows[:, 1:] for rows in (primary_rows, *repeat_rows)])
            primary_rows[:, 1:] = np.median(channels,
                                            axis=0,
                                            overwrite_input=True)
            output.write("".join(
                row_format % tuple(row) for row in primary_rows.tolist()))
            previous_time = timestamps[-1]
            rows_written += len(primary_rows)
        if rows_written < 3:
            raise ValueError(f"invalid motion samples in {primary_path}")


def write_raw(path, labels, joints, encoder_noise, encoder_resolution=0.0):
    dof = len(joints)
    positions = labels[:, 1:1 + dof] + encoder_noise
    if encoder_resolution:
        positions = encoder_resolution * np.rint(positions / encoder_resolution)
    observations = np.column_stack((labels[:, 0], positions, labels[:, -dof:]))
    write_dataset(path, observations, dataset_header(joints, ("q", "tau")))
