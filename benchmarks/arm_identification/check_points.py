"""Verify the exported automatic joint points against their Fourier coefficients."""

import json
from pathlib import Path
import sys

import numpy as np

from run_experiment import ARMS, read_motion
from simulation import harmonic_states


def check(path):
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if not report["design_only"] or report["sample_count"] != 161:
        raise ValueError("expected the 161-point design-only experiment")
    for name, arm in ARMS.items():
        data = report["robots"][name]
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_points.py DESIGN_OUTPUT_DIRECTORY")
    check(Path(sys.argv[1]))
