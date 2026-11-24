"""Identify fixed-base arm inertial combinations against MuJoCo torque labels."""

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


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
PAYLOAD_CENTER_XYZ = "0.05 0 -0.07"
OUTLIER_PROBABILITY = 0.01


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


def add_payload(robot, terminal, mass):
    if mass == 0.0:
        return
    x, y, z = PAYLOAD_SIZE_METERS
    box = ET.SubElement(robot, "link", name="experiment_payload")
    inertia = ET.SubElement(box, "inertial")
    ET.SubElement(inertia, "mass", value=str(mass))
    ET.SubElement(inertia,
                  "inertia",
                  ixx=str(mass * (y * y + z * z) / 12.0),
                  iyy=str(mass * (x * x + z * z) / 12.0),
                  izz=str(mass * (x * x + y * y) / 12.0),
                  ixy="0",
                  ixz="0",
                  iyz="0")
    joint = ET.SubElement(robot,
                          "joint",
                          name="experiment_payload_mount",
                          type="fixed")
    ET.SubElement(joint, "parent", link=terminal)
    ET.SubElement(joint, "child", link="experiment_payload")
    ET.SubElement(joint, "origin", xyz=PAYLOAD_CENTER_XYZ, rpy="0 0 0")


def write_urdf(robot, path):
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def dataset_header(joints):
    return ["time_s"] + [
        f"{name}/{joint}" for name in ("q", "dq", "ddq", "tau")
        for joint in joints
    ]


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


def inverse_dynamics(urdf, joints, motions, effort_limits):
    model = mujoco.MjModel.from_xml_path(str(urdf))
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joints
    ]
    dof = len(joints)
    if (model.nq != dof or model.nv != dof or
            any(index < 0 for index in joint_ids) or
            np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError(
            f"expected exactly the requested hinge joints in {urdf}")
    qpos = model.jnt_qposadr[joint_ids]
    qvel = model.jnt_dofadr[joint_ids]
    data = mujoco.MjData(model)
    result = {}
    for name, motion in motions.items():
        labels = motion.copy()
        for row in labels:
            data.qpos[qpos] = row[1:1 + dof]
            data.qvel[qvel] = row[1 + dof:1 + 2 * dof]
            data.qacc[qvel] = row[1 + 2 * dof:1 + 3 * dof]
            mujoco.mj_inverse(model, data)
            if data.nefc:
                raise ValueError(
                    f"active MuJoCo constraints in {name} at t={row[0]:.6g} s")
            row[-dof:] = data.qfrc_inverse[qvel]
        torque_ratio = np.abs(labels[:, -dof:]) / effort_limits
        if not np.isfinite(torque_ratio).all() or np.any(torque_ratio > 1.0):
            raise ValueError(
                f"{name} inverse dynamics exceeded a URDF joint effort limit")
        result[name] = labels
    return result


def make_noise(shape, seed, standard_deviation, outlier_amplitude):
    random = np.random.default_rng(seed)
    gaussian = random.normal(0.0, standard_deviation, size=shape)
    mask = random.random(shape) < OUTLIER_PROBABILITY
    signs = random.choice([-1.0, 1.0], size=shape)
    return gaussian + outlier_amplitude * mask * signs, int(
        mask.sum()) if outlier_amplitude else 0


def run_cpp(executable, *args):
    try:
        result = subprocess.run([str(executable), *map(str, args)],
                                capture_output=True,
                                text=True,
                                check=True)
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip(
        ) or f"arm_identification exited {error.returncode}"
        raise ValueError(f"{args[0]} {args[1]}: {message}") from error
    return json.loads(result.stdout)


def write_dataset(path, labels, joints):
    np.savetxt(path,
               labels,
               fmt="%.17g",
               delimiter=",",
               header=",".join(dataset_header(joints)),
               comments="")


def run_arm(name, args):
    arm = ARMS[name]
    source_bytes = (args.assets_root / arm.urdf).read_bytes()
    nominal, terminal, effort = extract_arm(ET.fromstring(source_bytes), arm,
                                            name)
    directory = args.output / name
    directory.mkdir()
    nominal_path = directory / "nominal.urdf"
    truth_path = directory / "truth.urdf"
    write_urdf(nominal, nominal_path)
    truth = copy.deepcopy(nominal)
    add_payload(truth, terminal, args.payload_kg)
    write_urdf(truth, truth_path)

    design = run_cpp(args.executable, "design", nominal_path, directory)
    motions = {
        split: read_motion(directory / f"{split}_states.csv", arm.joints)
        for split in ("baseline", "optimized", "holdout")
    }
    labels = inverse_dynamics(truth_path, arm.joints, motions, effort)
    dof = len(arm.joints)
    if labels["baseline"].shape != labels["optimized"].shape:
        raise ValueError(
            "baseline and optimized trajectories must have the same sample count"
        )
    noise, outlier_count = make_noise(labels["baseline"][:, -dof:].shape,
                                      args.seed, args.torque_noise_nm,
                                      args.outlier_nm)
    for split in ("baseline", "optimized"):
        training = labels[split].copy()
        training[:, -dof:] += noise
        write_dataset(directory / f"train_{split}.csv", training, arm.joints)
    write_dataset(directory / "holdout.csv", labels["holdout"], arm.joints)
    fit = run_cpp(args.executable, "fit", nominal_path, directory,
                  args.huber_delta_nm)

    return {
        "urdf": arm.urdf,
        "urdf_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "fixed_base": arm.base,
        "payload_parent": terminal,
        "joint_names": arm.joints,
        "joint_effort_limits_nm": effort.tolist(),
        "samples": {
            split: len(dataset) for split, dataset in labels.items()
        },
        "peak_effort_fraction": {
            split: float(np.max(np.abs(dataset[:, -dof:]) / effort))
            for split, dataset in labels.items()
        },
        "outlier_count_per_training_trajectory": outlier_count,
        "design": design,
        "fit": fit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root",
                        required=True,
                        type=Path,
                        help="path to HumanoidAssets")
    parser.add_argument("--executable",
                        required=True,
                        type=Path,
                        help="built arm_identification executable")
    parser.add_argument("--output",
                        required=True,
                        type=Path,
                        help="new directory for datasets and report.json")
    parser.add_argument("--robot", choices=("both", *ARMS), default="both")
    parser.add_argument("--seed",
                        type=int,
                        default=20261124,
                        help="training noise seed")
    parser.add_argument("--payload-kg",
                        type=float,
                        default=0.35,
                        help="added terminal payload in truth model")
    parser.add_argument("--torque-noise-nm",
                        type=float,
                        default=0.02,
                        help="training Gaussian torque noise sigma")
    parser.add_argument("--outlier-nm",
                        type=float,
                        default=0.75,
                        help="training outlier magnitude")
    parser.add_argument("--huber-delta-nm",
                        type=float,
                        default=0.06,
                        help="Huber threshold")
    args = parser.parse_args()
    for field in ("payload_kg", "torque_noise_nm", "outlier_nm",
                  "huber_delta_nm"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0.0 or (field == "huber_delta_nm"
                                                       and value == 0.0):
            parser.error(f"invalid --{field.replace('_', '-')} value")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.executable = args.executable.resolve()
    robots = ARMS if args.robot == "both" else (args.robot,)
    try:
        for name in robots:
            if not (args.assets_root / ARMS[name].urdf).is_file():
                raise ValueError(
                    f"cannot find {ARMS[name].urdf} in {args.assets_root}")
        if not args.executable.is_file():
            raise ValueError(f"cannot find executable {args.executable}")
        args.output.mkdir(parents=True, exist_ok=False)
        report = {
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "seed": args.seed,
            "payload_kg": args.payload_kg,
            "payload_box_size_m": PAYLOAD_SIZE_METERS,
            "payload_center_xyz_m": [
                float(value) for value in PAYLOAD_CENTER_XYZ.split()
            ],
            "torque_noise_nm": args.torque_noise_nm,
            "outlier_nm": args.outlier_nm,
            "outlier_probability": OUTLIER_PROBABILITY,
            "huber_delta_nm": args.huber_delta_nm,
            "robots": {
                name: run_arm(name, args) for name in robots
            },
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) +
                               "\n",
                               encoding="utf-8")
    except (OSError, KeyError, ET.ParseError, ValueError) as error:
        print(f"arm experiment: {error}", file=sys.stderr)
        return 1
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
