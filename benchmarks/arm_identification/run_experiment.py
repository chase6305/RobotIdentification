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

from simulation import ArmDynamics, inverse_dynamics, rollout


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


def write_dataset(path, labels, header):
    np.savetxt(path,
               labels,
               fmt="%.17g",
               delimiter=",",
               header=",".join(header),
               comments="")


def write_raw(path, labels, joints, encoder_noise):
    dof = len(joints)
    observations = np.column_stack(
        (labels[:, 0], labels[:, 1:1 + dof] + encoder_noise, labels[:, -dof:]))
    write_dataset(path, observations, dataset_header(joints, ("q", "tau")))


def differentiation_metrics(estimated, measured, dof):
    indices = np.searchsorted(measured[:, 0], estimated[:, 0])
    if np.any(indices == len(measured)) or not np.array_equal(
            estimated[:, 0], measured[indices, 0]):
        raise ValueError("raw and processed dataset timestamps do not align")
    error = estimated[:, 1:1 + 3 * dof] - measured[indices, 1:1 + 3 * dof]
    return {
        name: float(np.sqrt(np.mean(group**2))) for name, group in zip((
            "position_rmse_rad", "velocity_rmse_rad_s",
            "acceleration_rmse_rad_s2"), np.split(error, 3, axis=1))
    }


def run_arm(name, args):
    arm = ARMS[name]
    source_bytes = (args.assets_root / arm.urdf).read_bytes()
    nominal, terminal, effort = extract_arm(ET.fromstring(source_bytes), arm,
                                            name)
    directory = args.output / name
    directory.mkdir()
    nominal_path = directory / "nominal.urdf"
    write_urdf(nominal, nominal_path)
    drive_parameters = None
    design_options = []
    if args.actuator_scale > 0.0:
        effort_fraction = effort / np.max(effort)
        drive_parameters = args.actuator_scale * np.column_stack(
            (0.04 * effort_fraction, 0.02 + 0.05 * effort_fraction))
        design_options.append("actuators")
    design = run_cpp(args.executable, "design", nominal_path, directory,
                     args.sample_count, args.design_evaluations,
                     *design_options)
    result = {
        "urdf": arm.urdf,
        "urdf_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "fixed_base": arm.base,
        "joint_names": arm.joints,
        "joint_effort_limits_nm": effort.tolist(),
        "design": design,
    }
    if args.design_only:
        return result

    truth_path = directory / "truth.urdf"
    truth = copy.deepcopy(nominal)
    add_payload(truth, terminal, args.payload_kg)
    write_urdf(truth, truth_path)

    motions = {
        split: read_motion(directory / f"{split}_states.csv", arm.joints)
        for split in ("baseline", "optimized", "holdout")
    }
    plant = ArmDynamics.from_urdf(truth_path, arm.joints, drive_parameters)
    drive_options = []
    if drive_parameters is not None:
        drive_options = drive_parameters.ravel().tolist()
        result["actuators"] = {
            name: {
                "armature_kg_m2": float(armature),
                "viscous_nm_s_per_rad": float(viscous)
            } for name, (armature, viscous) in zip(arm.joints, drive_parameters)
        }
    acquisition = {}
    dense_holdout = None
    if args.acquisition == "rollout":
        controller = ArmDynamics.from_urdf(nominal_path, arm.joints)
        labels, acquisition, dense_holdout = rollout(
            plant, controller, motions, design["references"], effort,
            args.timestamp_jitter_ms / 1000, args.seed, args.replay)
    else:
        labels = inverse_dynamics(plant, motions, effort)
    if drive_parameters is not None or args.replay:
        mujoco.mj_saveModel(plant.model, str(directory / "truth.mjb"))
    dof = len(arm.joints)
    if labels["baseline"].shape != labels["optimized"].shape:
        raise ValueError(
            "baseline and optimized trajectories must have the same sample count"
        )
    noise, outlier_count = make_noise(labels["baseline"][:, -dof:].shape,
                                      args.seed, args.torque_noise_nm,
                                      args.outlier_nm)
    encoder_random = np.random.default_rng(
        np.random.SeedSequence([args.seed, 1]))
    encoder_noise = encoder_random.normal(0.0,
                                          args.encoder_noise_rad,
                                          size=noise.shape)
    processed = {}
    estimator_size = args.raw_harmonics if args.raw_estimator == "periodic-fourier" else args.raw_window
    for split in ("baseline", "optimized"):
        training = labels[split].copy()
        training[:, -dof:] += noise
        raw_path = directory / f"train_{split}_raw.csv"
        processed_path = directory / f"train_{split}.csv"
        write_raw(raw_path, training, arm.joints, encoder_noise)
        if args.fit_input == "raw":
            processed[split] = run_cpp(args.executable, "preprocess", raw_path,
                                       processed_path, estimator_size,
                                       args.raw_estimator)
            estimated = read_motion(processed_path, arm.joints)
            processed[split].update(
                differentiation_metrics(estimated, labels[split], dof))
        else:
            write_dataset(processed_path, training, dataset_header(arm.joints))
    write_raw(directory / "holdout_raw.csv", labels["holdout"], arm.joints, 0.0)
    write_dataset(directory / "holdout.csv", labels["holdout"],
                  dataset_header(arm.joints))
    if dense_holdout is not None:
        write_dataset(directory / "holdout_dense.csv", dense_holdout,
                      dataset_header(arm.joints))
    fit = run_cpp(args.executable, "fit", nominal_path, directory,
                  args.huber_delta_nm, *drive_options)

    result.update({
        "payload_parent": terminal,
        "samples": {
            split: len(dataset) for split, dataset in labels.items()
        },
        "peak_effort_fraction": {
            split: float(np.max(np.abs(dataset[:, -dof:]) / effort))
            for split, dataset in labels.items()
        },
        "outlier_count_per_training_trajectory": outlier_count,
        "acquisition": acquisition,
        "preprocessing": processed,
        "fit": fit,
    })
    return result


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
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="export automatic reference points without simulating or fitting")
    parser.add_argument("--sample-count",
                        type=int,
                        default=401,
                        help="number of reference points per trajectory")
    parser.add_argument("--design-evaluations",
                        type=int,
                        default=240,
                        help="Fourier design search evaluation budget")
    parser.add_argument(
        "--acquisition",
        choices=("inverse", "rollout"),
        default="inverse",
        help=
        "prescribed-state inverse dynamics or torque-driven forward simulation")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="replay short measured-torque windows with the fitted model")
    parser.add_argument(
        "--fit-input",
        choices=("simulator", "raw"),
        default="simulator",
        help=
        "fit with simulated derivatives or estimate them from recorder-format CSV"
    )
    parser.add_argument(
        "--timestamp-jitter-ms",
        type=float,
        default=0.0,
        help="maximum forward-simulation acquisition jitter in milliseconds")
    parser.add_argument(
        "--encoder-noise-rad",
        type=float,
        default=5e-5,
        help="training encoder noise standard deviation in recorded positions")
    parser.add_argument("--raw-window",
                        type=int,
                        default=5,
                        help="positive odd smoothing window for raw input")
    parser.add_argument(
        "--raw-estimator",
        choices=("moving-average", "local-cubic", "periodic-fourier"),
        default="moving-average",
        help="position derivative estimator for recorder-format input")
    parser.add_argument(
        "--raw-harmonics",
        type=int,
        default=7,
        help="harmonics fitted over one capture period for periodic-fourier")
    parser.add_argument("--seed",
                        type=int,
                        default=20261124,
                        help="training noise seed")
    parser.add_argument("--payload-kg",
                        type=float,
                        default=0.35,
                        help="added terminal payload in truth model")
    parser.add_argument(
        "--actuator-scale",
        type=float,
        default=0.0,
        help="scale of synthetic rotor inertia and viscous joint friction")
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
    for field in ("payload_kg", "actuator_scale", "torque_noise_nm",
                  "outlier_nm", "encoder_noise_rad", "timestamp_jitter_ms",
                  "huber_delta_nm"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0.0 or (field == "huber_delta_nm"
                                                       and value == 0.0):
            parser.error(f"invalid --{field.replace('_', '-')} value")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.sample_count < 3:
        parser.error("--sample-count must be at least three")
    if args.design_evaluations < 1:
        parser.error("--design-evaluations must be positive")
    if args.raw_window <= 0 or args.raw_window % 2 == 0:
        parser.error("--raw-window must be positive and odd")
    if args.raw_estimator == "local-cubic" and args.raw_window < 5:
        parser.error("local-cubic requires --raw-window of at least five")
    if not 1 <= args.raw_harmonics <= 32:
        parser.error("--raw-harmonics must be between 1 and 32")
    if (args.fit_input == "raw" and args.raw_estimator == "periodic-fourier" and
            args.sample_count < 2 * args.raw_harmonics + 2):
        parser.error("periodic-fourier requires at least 2h+2 samples")
    if args.timestamp_jitter_ms > 0.0 and args.acquisition != "rollout":
        parser.error("--timestamp-jitter-ms requires --acquisition rollout")
    if args.replay and (args.acquisition != "rollout" or args.design_only):
        parser.error(
            "--replay requires --acquisition rollout with simulation and fitting"
        )
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
            "acquisition": args.acquisition,
            "replay": args.replay,
            "design_only": args.design_only,
            "sample_count": args.sample_count,
            "design_evaluations": args.design_evaluations,
            "fit_input": args.fit_input,
            "seed": args.seed,
            "payload_kg": args.payload_kg,
            "actuator_scale": args.actuator_scale,
            "payload_box_size_m": PAYLOAD_SIZE_METERS,
            "payload_center_xyz_m": [
                float(value) for value in PAYLOAD_CENTER_XYZ.split()
            ],
            "torque_noise_nm": args.torque_noise_nm,
            "encoder_noise_rad": args.encoder_noise_rad,
            "timestamp_jitter_ms": args.timestamp_jitter_ms,
            "raw_window": args.raw_window,
            "raw_estimator": args.raw_estimator,
            "raw_harmonics": args.raw_harmonics,
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
