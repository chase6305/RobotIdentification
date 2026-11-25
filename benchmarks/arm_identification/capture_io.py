"""Read and write the processed and recorder-format arm CSV files."""

import csv
from itertools import islice, zip_longest

import numpy as np

_MEDIAN_BLOCK_ROWS = 256


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
               fmt="%.17g",
               delimiter=",",
               header=",".join(header),
               comments="")


def _raw_blocks(path, header):
    position = None
    while True:
        # Close each file between blocks so the descriptor count stays fixed.
        with path.open(newline="", encoding="utf-8") as stream:
            if position is None:
                _check_header(stream, path, header)
            else:
                stream.seek(position)
            rows = (line for line in iter(stream.readline, "")
                    if line not in ("\n", "\r",
                                    "\r\n") and not line.startswith("#"))
            lines = list(islice(rows, _MEDIAN_BLOCK_ROWS))
            position = stream.tell()
        if not lines:
            return
        samples = np.loadtxt(lines, delimiter=",", ndmin=2)
        if samples.shape[1] != len(header) or not np.isfinite(samples).all():
            raise ValueError(f"invalid motion samples in {path}")
        yield samples


def write_median_raw(path, inputs, joints):
    header = dataset_header(joints, ("q", "tau"))
    captures = [_raw_blocks(filename, header) for filename in inputs]
    count = 0
    last_time = -np.inf
    with path.open("w", newline="", encoding="utf-8") as output:
        output.write(",".join(header) + "\n")
        for reference, *repeats in zip_longest(*captures):
            if reference is None:
                raise ValueError(
                    f"repeat capture has more samples than {inputs[0]}")
            times = reference[:, 0]
            if times[0] <= last_time or np.any(times[1:] <= times[:-1]):
                raise ValueError(f"invalid motion samples in {inputs[0]}")
            for filename, samples in zip(inputs[1:], repeats):
                if samples is None or not np.array_equal(samples[:, 0], times):
                    raise ValueError(
                        f"repeat capture timestamps differ in {filename}")
            values = np.stack([block[:, 1:] for block in (reference, *repeats)])
            reference[:, 1:] = np.median(values, axis=0, overwrite_input=True)
            write_dataset(output, reference, ())
            last_time = times[-1]
            count += len(reference)
        if count < 3:
            raise ValueError(f"invalid motion samples in {inputs[0]}")


def write_raw(path, labels, joints, encoder_noise, encoder_resolution=0.0):
    dof = len(joints)
    positions = labels[:, 1:1 + dof] + encoder_noise
    if encoder_resolution:
        positions = encoder_resolution * np.rint(positions / encoder_resolution)
    observations = np.column_stack((labels[:, 0], positions, labels[:, -dof:]))
    write_dataset(path, observations, dataset_header(joints, ("q", "tau")))
