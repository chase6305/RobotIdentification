"""Extract and export the mesh-free arm models used by the benchmark."""

import copy
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from capture_io import read_motion
from simulation import (ArmDynamics, configure_integrator, inverse_dynamics,
                        replay_windows, save_contact_free_mjcf)


@dataclass(frozen=True)
class Arm:
    urdf: str
    base: str
    joints: tuple[str, ...]
    fixed_joints: tuple[str, ...] = ()


ARMS = {
    "pm01":
        Arm("Engineai_PM01/urdf/robot.urdf", "LINK_TORSO_YAW",
            ("J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
             "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L"),
            ("J_FIXED_ELBOW_END_L",)),
    "openarm":
        Arm("OpenArm/left_arm.urdf", "left_arm_base",
            ("SHOULDER_PITCH_L_J1", "SHOULDER_ROLL_L_J2", "ELBOW_PITCH_L_J3",
             "ELBOW_YAW_L_J4", "WRIST_PITCH_L_J5", "WRIST_YAW_L_J6",
             "WRIST_ROLL_L_J7")),
}
PAYLOAD_SIZE_METERS = (0.06, 0.05, 0.04)
PAYLOAD_CENTER_METERS = (0.05, 0.0, -0.07)


def _required(element, path, label):
    result = element.find(path)
    if result is None:
        raise ValueError(f"{label} is missing {path}")
    return result


def extract_arm(source, arm, name):
    """Copy the specified serial chain, keeping joint frames and inertias."""
    links = {link.attrib["name"]: link for link in source.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in source.findall("joint")}
    robot = ET.Element("robot", name=name)
    ET.SubElement(ET.SubElement(robot, "mujoco"),
                  "compiler",
                  fusestatic="true",
                  inertiafromgeom="false",
                  balanceinertia="false")
    parent = arm.base
    if parent not in links:
        raise ValueError(f"source URDF has no base link {parent}")
    ET.SubElement(robot, "link", name=parent)
    effort_limits = []
    terminal = parent
    for joint_name in arm.joints + arm.fixed_joints:
        kind = "revolute" if joint_name in arm.joints else "fixed"
        joint = joints.get(joint_name)
        if joint is None or joint.get("type") != kind:
            raise ValueError(f"source URDF has no {kind} joint {joint_name}")
        if _required(joint, "parent", joint_name).get("link") != parent:
            raise ValueError(f"{joint_name} must be attached to {parent}")
        child = _required(joint, "child", joint_name).attrib["link"]
        if child not in links:
            raise ValueError(f"{joint_name} has unknown child {child}")
        link = ET.SubElement(robot, "link", name=child)
        link.append(copy.deepcopy(_required(links[child], "inertial", child)))
        kept_joint = ET.SubElement(robot, "joint", name=joint_name, type=kind)
        for tag in ("parent", "child", "origin"):
            kept_joint.append(copy.deepcopy(_required(joint, tag, joint_name)))
        if kind == "revolute":
            kept_joint.append(
                copy.deepcopy(_required(joint, "axis", joint_name)))
            limit = _required(joint, "limit", joint_name)
            lower, upper, velocity, effort = (float(limit.attrib[key])
                                              for key in ("lower", "upper",
                                                          "velocity", "effort"))
            if not all(map(math.isfinite,
                           (lower, upper, velocity, effort))) or not (
                               lower < upper and velocity > 0 and effort > 0):
                raise ValueError(f"invalid limits for {joint_name}")
            effort_limits.append(effort)
            kept_joint.append(copy.deepcopy(limit))
            terminal = child
        parent = child
    return robot, terminal, np.asarray(effort_limits)


def add_payload(robot, terminal, mass, size, center):
    if mass == 0.0:
        return
    x, y, z = size
    diagonal = [
        mass * (y * y + z * z) / 12.0, mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in diagonal):
        raise ValueError("payload box inertia must be finite and positive")
    add_payload_inertia(robot, terminal, mass, center, np.diag(diagonal))


def add_payload_inertia(robot, terminal, mass, center, matrix):
    box = ET.SubElement(robot, "link", name="experiment_payload")
    set_link_inertia(box, mass, matrix)
    joint = ET.SubElement(robot,
                          "joint",
                          name="experiment_payload_mount",
                          type="fixed")
    ET.SubElement(joint, "parent", link=terminal)
    ET.SubElement(joint, "child", link="experiment_payload")
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")


def set_link_inertia(link, mass, matrix, center=None):
    inertia = ET.SubElement(link, "inertial")
    if center is not None:
        ET.SubElement(inertia,
                      "origin",
                      xyz=" ".join(map(str, center)),
                      rpy="0 0 0")
    ET.SubElement(inertia, "mass", value=str(mass))
    ET.SubElement(inertia,
                  "inertia",
                  ixx=str(matrix[0, 0]),
                  iyy=str(matrix[1, 1]),
                  izz=str(matrix[2, 2]),
                  ixy=str(matrix[0, 1]),
                  ixz=str(matrix[0, 2]),
                  iyz=str(matrix[1, 2]))


def write_urdf(robot, path):
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def export_mujoco_model(urdf,
                        joints,
                        drive,
                        holdout,
                        dense_holdout,
                        candidate,
                        effort=None):
    binary = urdf.with_suffix(".mjb")
    text = urdf.with_suffix(".xml")
    model = ArmDynamics.from_urdf(urdf, joints, drive)
    if dense_holdout is not None:
        configure_integrator(model.model,
                             dense_holdout[1, 0] - dense_holdout[0, 0])
    mujoco.mj_saveModel(model.model, str(binary))
    save_contact_free_mjcf(model.model, text)
    reloaded = ArmDynamics.from_mjb(binary, joints)
    text_model = ArmDynamics.from_mjcf(text, joints)
    motions = {"holdout": holdout}
    binary_torque = inverse_dynamics(reloaded, motions,
                                     effort)["holdout"][:, -len(joints):]
    text_torque = inverse_dynamics(text_model, motions,
                                   effort)["holdout"][:, -len(joints):]
    difference = float(np.max(np.abs(binary_torque - text_torque)))
    if difference > 1e-10:
        raise ValueError(f"{text} and {binary} disagree on holdout torque")
    candidate["mjcf_binary_max_difference_nm"] = difference
    return reloaded, binary_torque


def export_payload(candidate, nominal, terminal, directory, joints,
                   drive_parameters, holdout, dense_holdout, effort):
    estimated = copy.deepcopy(nominal)
    add_payload_inertia(estimated, terminal, candidate["mass_kg"],
                        candidate["center_m"],
                        np.asarray(candidate["inertia_at_center_kg_m2"]))
    urdf = directory / "identified_payload.urdf"
    write_urdf(estimated, urdf)
    _, predicted = export_mujoco_model(urdf, joints, drive_parameters, holdout,
                                       dense_holdout, candidate, effort)
    torque_error = predicted - holdout[:, -len(joints):]
    candidate["reloaded_mujoco_holdout_rmse_nm"] = float(
        np.sqrt(np.mean(torque_error**2)))
    if abs(candidate["reloaded_mujoco_holdout_rmse_nm"] -
           candidate["holdout_rmse_nm"]) > 1e-7:
        raise ValueError(
            "exported payload disagrees with the native holdout prediction")


def payload_reference_errors(fit, truth_path):
    model = ET.parse(truth_path).getroot()
    payload = model.find("./link[@name='experiment_payload']/inertial")
    mass = float(
        payload.find("mass").attrib["value"]) if payload is not None else 0.0
    errors = {"mass_absolute_error_kg": abs(fit["estimated_mass_kg"] - mass)}
    candidate = fit["physical_candidate"]
    if payload is not None and candidate is not None:
        mount = model.find("./joint[@name='experiment_payload_mount']/origin")
        center = np.asarray(
            [float(value) for value in mount.attrib["xyz"].split()])
        tensor = payload.find("inertia").attrib
        ixx, ixy, ixz, iyy, iyz, izz = (float(tensor[key])
                                        for key in ("ixx", "ixy", "ixz", "iyy",
                                                    "iyz", "izz"))
        inertia = np.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz,
                                                                 izz]])
        errors["center_distance_error_m"] = float(
            np.linalg.norm(np.asarray(candidate["center_m"]) - center))
        errors["inertia_frobenius_error_kg_m2"] = float(
            np.linalg.norm(
                np.asarray(candidate["inertia_at_center_kg_m2"]) - inertia))
    return errors


def export_identified_arm(fit, nominal, directory, joints, holdout,
                          dense_holdout):
    candidate = fit["physical_arm_fit"]
    estimated = copy.deepcopy(nominal)
    links = {link.attrib["name"]: link for link in estimated.findall("link")}
    moving_links = {
        joint.attrib["name"]: joint.find("child").attrib["link"]
        for joint in estimated.findall("joint")
        if joint.get("type") == "revolute"
    }
    for link in links.values():
        inertia = link.find("inertial")
        if inertia is not None:
            link.remove(inertia)
    dof = len(joints)
    parameters = np.asarray(candidate["full_parameters"])
    for name, body in zip(joints,
                          parameters[:10 * dof].reshape(dof, 10),
                          strict=True):
        mass, hx, hy, hz, ixx, ixy, iyy, ixz, iyz, izz = body
        center = np.array([hx, hy, hz]) / mass
        origin = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        centered = origin - mass * (np.dot(center, center) * np.eye(3) -
                                    np.outer(center, center))
        # The estimated joint-frame inertia already includes fixed descendants.
        set_link_inertia(links[moving_links[name]], mass, centered, center)

    drive = parameters[10 * dof:]
    drive = drive.reshape(dof, 2) if drive.size else None
    urdf = directory / "identified_arm.urdf"
    write_urdf(estimated, urdf)
    reloaded, predicted = export_mujoco_model(urdf, joints, drive, holdout,
                                              dense_holdout, candidate)
    native = read_motion(directory / "holdout_predicted_physical_arm.csv",
                         joints)[:, -dof:]
    candidate["reloaded_mujoco_holdout_rmse_nm"] = float(
        np.sqrt(np.mean((predicted - holdout[:, -dof:])**2)))
    candidate["reloaded_max_difference_nm"] = float(
        np.max(np.abs(predicted - native)))
    if candidate["reloaded_max_difference_nm"] > 1e-7:
        raise ValueError(
            "exported arm disagrees with the native holdout prediction")
    if dense_holdout is not None:
        native_replay = fit["replay"]["physical_arm"]
        replay = replay_windows(reloaded, dense_holdout,
                                native_replay["windows"],
                                native_replay["window_seconds"])
        candidate["reloaded_mujoco_replay"] = replay
        if replay["complete_windows"] != native_replay["complete_windows"]:
            raise ValueError(
                "exported arm disagrees with the native replay completion")
        if (replay["position_rmse_rad"] is not None and
                native_replay["position_rmse_rad"] is not None and
                abs(replay["position_rmse_rad"] -
                    native_replay["position_rmse_rad"]) > 1e-7):
            raise ValueError(
                "exported arm disagrees with the native replay position error")
    return reloaded
