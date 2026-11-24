"""Verify the actuator experiment and its reopenable MuJoCo plant model."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from run_experiment import read_motion


def check(directory, clean):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for name, dof, rank in (("pm01", 5, 41), ("openarm", 7, 55)):
        robot = report["robots"][name]
        fit = robot["fit"]
        estimated = fit["optimized"]["huber"]
        if (fit["base_parameter_count"] != rank or
                fit["full_parameter_count"] != 12 * dof or
                fit["oracle_holdout_max_absolute_error_nm"] > 1e-7 or
                fit["reduced_oracle_max_difference_nm"] > 1e-8):
            raise ValueError(
                f"{name}: unexpected actuator observability or oracle error")
        if clean and not estimated["relative_base_error"] < 1e-6:
            raise ValueError(
                f"{name}: clean actuator base parameters did not recover")
        rigid_error = fit["rigid_only_optimized_huber_holdout_rmse_nm"]
        if not estimated["holdout_rmse_nm"] < 0.25 * rigid_error:
            raise ValueError(
                f"{name}: joint dynamics did not improve forward-data prediction"
            )
        if not clean and report["fit_input"] != "raw":
            raise ValueError(f"{name}: expected a raw actuator experiment")
        forward = estimated["forward_validation"]
        if (not forward["mass_check_passed"] or
                not forward["minimum_mass_eigenvalue_kg_m2"] > 0 or
                not forward["holdout_acceleration_rmse_rad_s2"]
                < (1e-8 if clean else 1.0)):
            raise ValueError(f"{name}: invalid fitted forward predictions")
        replay = fit["replay"]["optimized_huber"]
        if (not report["replay"] or replay["windows"] != 16 or
                replay["complete_windows"] != 16 or
                not replay["position_rmse_rad"] < (1e-9 if clean else 0.004)):
            raise ValueError(
                f"{name}: fitted forward replay did not match the recorded plant"
            )
        model = mujoco.MjModel.from_binary_path(
            str(directory / name / "truth.mjb"))
        joints = robot["joint_names"]
        dense = read_motion(directory / name / "holdout_dense.csv", joints)
        if len(dense
              ) != robot["acquisition"]["holdout"]["integration_steps"] + 1:
            raise ValueError(
                f"{name}: replay trace is missing integration steps")
        observed = read_motion(directory / name / "holdout.csv", joints)
        predicted = read_motion(
            directory / name / "holdout_predicted_optimized_huber.csv", joints)
        np.testing.assert_array_equal(predicted[:, :-dof], observed[:, :-dof])
        rmse = np.sqrt(np.mean((predicted[:, -dof:] - observed[:, -dof:])**2))
        np.testing.assert_allclose(rmse,
                                   estimated["holdout_rmse_nm"],
                                   rtol=1e-10,
                                   atol=1e-12)
        addresses = model.jnt_dofadr[[
            model.joint(joint).id for joint in joints
        ]]
        for values, field in ((model.dof_armature, "armature_kg_m2"),
                              (model.dof_damping, "viscous_nm_s_per_rad")):
            expected = [robot["actuators"][joint][field] for joint in joints]
            np.testing.assert_array_equal(values[addresses], expected)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_drive.py CLEAN_RUN_DIRECTORY RAW_RUN_DIRECTORY")
    check(Path(sys.argv[1]), clean=True)
    check(Path(sys.argv[2]), clean=False)
