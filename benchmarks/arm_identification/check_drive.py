"""Verify the actuator experiment and its reopenable MuJoCo plant model."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from capture_io import read_motion


def check_payload_mount(directory, name, robot, report):
    center = report["payload_center_xyz_m"]
    if center is None:
        # Fixed joint origins from the two URDFs pinned in arm-identification-ci.
        flange, position, sign = {
            "pm01": ("J_FIXED_ELBOW_END_L", (0.03, -0.02, -0.14), -1),
            "openarm": ("LEFT_HAND_CONNECTOR", (0.0, 0.0, 0.1001), 1),
        }[name]
        if robot["payload_flange_joint"] != flange:
            raise ValueError(f"{name}: wrong payload flange")
        center = np.array(position)
        center[2] += sign * report["payload_box_size_m"][2] / 2.0

    resolved_center = robot.get("payload_center_xyz_m", center)
    np.testing.assert_allclose(resolved_center, center, rtol=0.0, atol=1e-15)
    truth = ET.parse(directory / "truth.urdf")
    mount = truth.find("joint[@name='experiment_payload_mount']")
    if mount.find("parent").get("link") != robot["payload_parent"]:
        raise ValueError(f"{name}: wrong payload parent")
    mounted_center = [
        float(value) for value in mount.find("origin").get("xyz").split()
    ]
    np.testing.assert_allclose(mounted_center, center, rtol=0.0, atol=1e-15)
    return center


def check(directory, clean):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for name, dof, rank in (("pm01", 5, 39), ("openarm", 7, 55)):
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
        terms = fit["optimized_huber_actuator_parameters"]
        for index, term in enumerate(terms):
            ambiguous = index in (0, 2)
            if (term["estimate"] is None) != ambiguous:
                raise ValueError(
                    f"{name}: incorrect individual observability for {term['name']}"
                )
            if clean and not ambiguous and abs(term["estimate"] -
                                               term["configured"]) > 1e-8:
                raise ValueError(f"{name}: clean drive term was not recovered")
        for values, field in ((model.dof_armature, "armature_kg_m2"),
                              (model.dof_damping, "viscous_nm_s_per_rad")):
            expected = [robot["actuators"][joint][field] for joint in joints]
            np.testing.assert_array_equal(values[addresses], expected)
        payload = fit["payload_fit"]["physical_candidate"]
        if not report["estimate_payload"] or payload is None:
            raise ValueError(f"{name}: expected an exported physical payload")
        if abs(payload["mass_kg"] -
               report["payload_kg"]) > (1e-7 if clean else 0.01):
            raise ValueError(f"{name}: inaccurate estimated payload mass")
        center = check_payload_mount(directory / name, name, robot, report)
        np.testing.assert_allclose(payload["center_m"],
                                   center,
                                   rtol=0.0,
                                   atol=(1e-7 if clean else 0.005))
        inertia = np.asarray(payload["inertia_at_center_kg_m2"])
        reference = fit["payload_fit"]["reference_errors"]
        if clean and reference["inertia_frobenius_error_kg_m2"] > 1e-7:
            raise ValueError(f"{name}: clean payload inertia was not recovered")
        central_moment = 0.5 * np.trace(inertia) * np.eye(3) - inertia
        if np.linalg.eigvalsh(central_moment).min() <= 0:
            raise ValueError(
                f"{name}: exported payload inertia violates physical constraints"
            )
        binary = mujoco.MjModel.from_binary_path(
            str(directory / name / "identified_payload.mjb"))
        nominal = mujoco.MjModel.from_xml_path(
            str(directory / name / "nominal.urdf"))
        np.testing.assert_allclose(binary.body_mass.sum() -
                                   nominal.body_mass.sum(),
                                   payload["mass_kg"],
                                   rtol=0.0,
                                   atol=1e-12)
        if abs(payload["reloaded_mujoco_holdout_rmse_nm"] -
               payload["holdout_rmse_nm"]) > 1e-7:
            raise ValueError(
                f"{name}: exported payload torque changed on reload")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_drive.py CLEAN_RUN_DIRECTORY RAW_RUN_DIRECTORY")
    check(Path(sys.argv[1]), clean=True)
    check(Path(sys.argv[2]), clean=False)
