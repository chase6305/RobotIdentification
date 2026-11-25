"""Verify packet loss and encoder resolution without misaligning torque labels."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

from capture_io import dataset_header, read_motion


def check(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if (report["fit_input"] != "raw" or
            report["raw_estimator"] != "periodic-fourier" or
            not report["capture_dropout"] > 0.0 or
            not report["encoder_resolution_rad"] > 0.0):
        raise ValueError("expected the imperfect-capture fixture")
    for name, robot in report["robots"].items():
        joints = robot["joint_names"]
        acquired = []
        for split in ("baseline", "optimized"):
            raw_path = directory / name / f"train_{split}_raw.csv"
            with raw_path.open(newline="", encoding="utf-8") as stream:
                if next(csv.reader(stream),
                        None) != dataset_header(joints, ("q", "tau")):
                    raise ValueError(f"{name}/{split}: unexpected raw columns")
                raw = np.loadtxt(stream, delimiter=",", ndmin=2)
            processed = read_motion(directory / name / f"train_{split}.csv",
                                    joints)
            if not 3 < len(
                    raw) == robot["samples"][split] < report["sample_count"]:
                raise ValueError(
                    f"{name}/{split}: dropout did not remove interior samples")
            dof = len(joints)
            ticks = raw[:, 1:1 + dof] / report["encoder_resolution_rad"]
            np.testing.assert_allclose(ticks,
                                       np.rint(ticks),
                                       rtol=0.0,
                                       atol=1e-10)
            np.testing.assert_array_equal(raw[:, 0], processed[:, 0])
            np.testing.assert_array_equal(raw[:, -dof:], processed[:, -dof:])
            np.testing.assert_allclose(raw[[0, -1], 0],
                                       [0.0, report["training_period_s"]],
                                       rtol=1e-14)
            acquired.append(raw[:, 0])
        np.testing.assert_array_equal(*acquired)
        fit = robot["fit"]["optimized"]["huber"]
        if not fit["holdout_rmse_nm"] < 0.03 or not fit["forward_validation"][
                "mass_check_passed"]:
            raise ValueError(
                f"{name}: imperfect capture did not retain predictable dynamics"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_capture.py RUN_DIRECTORY")
    check(Path(sys.argv[1]))
