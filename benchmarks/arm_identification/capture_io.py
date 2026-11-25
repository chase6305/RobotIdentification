"""Read and write the processed and recorder-format arm CSV files."""

import csv

import numpy as np


def dataset_header(joints, fields=("q", "dq", "ddq", "tau")):
    return ["time_s"
           ] + [f"{name}/{joint}" for name in fields for joint in joints]


def read_motion(path,
                joints,
                *,
                max_rows=None,
                fields=("q", "dq", "ddq", "tau")):
    header = dataset_header(joints, fields)
    with path.open(newline="", encoding="utf-8") as stream:
        if next(csv.reader(stream), None) != header:
            raise ValueError(f"unexpected dataset header in {path}")
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


def write_median_raw(path, inputs, joints):
    fields = ("q", "tau")
    reference = read_motion(inputs[0], joints, fields=fields)
    values = np.empty((len(inputs), len(reference), reference.shape[1] - 1))
    values[0] = reference[:, 1:]
    for index, filename in enumerate(inputs[1:], 1):
        samples = read_motion(filename, joints, fields=fields)
        if not np.array_equal(samples[:, 0], reference[:, 0]):
            raise ValueError(f"repeat capture timestamps differ in {filename}")
        values[index] = samples[:, 1:]
    reference[:, 1:] = np.median(values, axis=0, overwrite_input=True)
    write_dataset(path, reference, dataset_header(joints, fields))


def write_raw(path, labels, joints, encoder_noise, encoder_resolution=0.0):
    dof = len(joints)
    positions = labels[:, 1:1 + dof] + encoder_noise
    if encoder_resolution:
        positions = encoder_resolution * np.rint(positions / encoder_resolution)
    observations = np.column_stack((labels[:, 0], positions, labels[:, -dof:]))
    write_dataset(path, observations, dataset_header(joints, ("q", "tau")))
