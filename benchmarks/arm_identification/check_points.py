"""Verify the exported automatic joint points against their Fourier coefficients."""

import json
from pathlib import Path
import sys

import numpy as np

from arm_models import ARMS
from capture_io import read_motion
from simulation import harmonic_states


def check(path):
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if not report["design_only"] or report["sample_count"] != 161:
        raise ValueError("expected the 161-point design-only experiment")
    for name, arm in ARMS.items():
        data = report["robots"][name]
        limits = data["design"]["kinematic_limits"]
        for split, coefficients in data["design"]["references"].items():
            points = read_motion(path / name / f"{split}_states.csv",
                                 arm.joints)
            if len(points) != report["sample_count"]:
                raise ValueError(
                    f"{name}/{split}: wrong number of generated points")
            states = np.hstack(harmonic_states(coefficients, points[:, 0]))
            np.testing.assert_allclose(points[:, 1:1 + states.shape[1]],
                                       states,
                                       rtol=0.0,
                                       atol=1e-12)
            expected_duration = report["holdout_period_s" if split ==
                                       "holdout" else "training_period_s"]
            np.testing.assert_allclose(points[-1, 0],
                                       expected_duration,
                                       rtol=1e-14)
            dense_time = np.linspace(0.0, expected_duration, 1001)
            q, dq, ddq = harmonic_states(coefficients, dense_time)
            if (np.any(q < limits["minimum_position_rad"]) or
                    np.any(q > limits["maximum_position_rad"]) or
                    np.any(np.abs(dq) > limits["maximum_velocity_rad_s"]) or
                    np.any(
                        np.abs(ddq) > limits["maximum_acceleration_rad_s2"])):
                raise ValueError(
                    f"{name}/{split}: exported trajectory violated its limits")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_points.py DESIGN_OUTPUT_DIRECTORY")
    check(Path(sys.argv[1]))
