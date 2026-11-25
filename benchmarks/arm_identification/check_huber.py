"""Check automatic Huber scaling and the high-noise passive-drive projection."""

import json
from pathlib import Path
import sys


def check(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not report["huber_auto"]:
        raise ValueError("expected an automatic-threshold experiment")
    for name, robot in report["robots"].items():
        fit = robot["fit"]
        huber = fit["optimized"]["huber"]
        if not huber["converged"] or not 0.04 < huber["huber_delta_nm"] < 0.25:
            raise ValueError(
                f"{name}: invalid automatic Huber threshold or convergence")
        projection = fit["nonnegative_actuator_projection"]
        candidate = projection["candidate"]
        if candidate is None or not candidate["forward_validation"][
                "mass_check_passed"]:
            raise ValueError(f"{name}: invalid projected model")
        for term in candidate["actuator_parameters"]:
            if term["estimate"] is not None and term["estimate"] < -1e-9:
                raise ValueError(
                    f"{name}: projected drive coefficient was negative")
        if name == "openarm" and (
                not projection["negative_terms_before"] or
                not candidate["holdout_rmse_nm"] < huber["holdout_rmse_nm"]):
            raise ValueError(
                "OpenArm high-noise fixture did not exercise an improving projection"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_huber.py HIGH_NOISE_REPORT")
    check(sys.argv[1])
