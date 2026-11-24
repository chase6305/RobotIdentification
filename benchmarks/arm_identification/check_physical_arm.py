"""Check clean and noisy whole-arm identification and exported MuJoCo replay."""

import json
from pathlib import Path
import sys


def check_case(directory, maximum_torque_rmse, maximum_position_rmse):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for name, arm in report["robots"].items():
        fit = arm["fit"]["physical_arm_fit"]
        if (not fit["converged"] or
                not fit["forward_validation"]["mass_check_passed"] or
                fit["holdout_rmse_nm"] >= maximum_torque_rmse or
                fit["reloaded_max_difference_nm"] > 1e-7 or
                fit["mjcf_binary_max_difference_nm"] > 1e-10):
            raise ValueError(f"{name}: physical arm fit exceeded tolerance")
        if arm["fit"]["full_parameter_count"] != len(fit["full_parameters"]):
            raise ValueError(f"{name}: full arm parameters are incomplete")
        replay = fit["reloaded_mujoco_replay"]
        if (replay["windows"] != 16 or replay["complete_windows"] != 16 or
                replay["position_rmse_rad"] >= maximum_position_rmse):
            raise ValueError(f"{name}: physical arm replay exceeded tolerance")
        workspace = fit["workspace_validation"]
        if (workspace["requested_samples"] != 1024 or
                not 768 <= workspace["effort_feasible_samples"] <= 1024 or
                workspace["minimum_mass_eigenvalue_kg_m2"] <= 0.0 or
                workspace["identified_torque_rmse_nm"]
                >= 0.4 * workspace["nominal_torque_rmse_nm"]):
            raise ValueError(f"{name}: physical arm failed the workspace check")
        for filename in ("identified_arm.urdf", "identified_arm.mjb",
                         "identified_arm.xml",
                         "holdout_predicted_physical_arm.csv"):
            if (directory / name / filename).stat().st_size == 0:
                raise ValueError(f"{name}: {filename} is empty")
    return report


def main(clean, demo, noisy):
    check_case(clean, 2e-5, 2e-6)
    check_case(demo, 0.008, 0.002)
    report = check_case(noisy, 0.045, 0.008)
    openarm = report["robots"]["openarm"]["fit"]
    unconstrained = openarm["optimized"]["huber"]
    if (unconstrained["forward_validation"]["mass_check_passed"] or
            openarm["physical_arm_fit"]["holdout_rmse_nm"]
            >= 0.65 * unconstrained["holdout_rmse_nm"]):
        raise ValueError(
            "OpenArm high-noise fixture must exercise the inertia constraint")


if __name__ == "__main__":
    try:
        main(*map(Path, sys.argv[1:]))
    except (KeyError, OSError, ValueError) as error:
        print(f"physical arm verification: {error}", file=sys.stderr)
        sys.exit(1)
