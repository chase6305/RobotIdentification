"""Read and write the processed and recorder-format arm CSV files."""

import csv

import numpy as np


def dataset_header(joints, fields=("q", "dq", "ddq", "tau")):
    return ["time_s"
           ] + [f"{name}/{joint}" for name in fields for joint in joints]


def read_motion(path, joints):
    with path.open(newline="", encoding="utf-8") as stream:
        if next(csv.reader(stream), None) != dataset_header(joints):
            raise ValueError(f"unexpected dataset header in {path}")
        motion = np.loadtxt(stream, delimiter=",", ndmin=2)
    if (motion.shape[0] < 3 or motion.shape[1] != 1 + 4 * len(joints) or
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


def write_raw(path, labels, joints, encoder_noise, encoder_resolution=0.0):
    dof = len(joints)
    positions = labels[:, 1:1 + dof] + encoder_noise
    if encoder_resolution:
        positions = encoder_resolution * np.rint(positions / encoder_resolution)
    observations = np.column_stack((labels[:, 0], positions, labels[:, -dof:]))
    write_dataset(path, observations, dataset_header(joints, ("q", "tau")))
