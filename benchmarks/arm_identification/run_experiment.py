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

from simulation import (ArmDynamics, configure_integrator, inverse_dynamics,
                        replay_windows, rollout, save_contact_free_mjcf)


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
OUTLIER_PROBABILITY = 0.01
FOURIER_ESTIMATORS = ("periodic-fourier", "periodic-fourier-huber")
RAW_ESTIMATORS = ("moving-average", "local-cubic", *FOURIER_ESTIMATORS)


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


def write_raw(path, labels, joints, encoder_noise, encoder_resolution=0.0):
    dof = len(joints)
    positions = labels[:, 1:1 + dof] + encoder_noise
    if encoder_resolution:
        positions = encoder_resolution * np.rint(positions / encoder_resolution)
    observations = np.column_stack((labels[:, 0], positions, labels[:, -dof:]))
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


def fit_captured_arm(args, directory, metadata, holdout, dense_holdout):
    nominal_path = directory / "nominal.urdf"
    joints = metadata["joint_names"]
    actuators = metadata.get("actuators")
    drive = None
    drive_options = []
    if actuators:
        drive = np.asarray([[
            actuators[name]["armature_kg_m2"],
            actuators[name]["viscous_nm_s_per_rad"]
        ] for name in joints])
        drive_options = drive.ravel().tolist()
    options = []
    if args.estimate_payload:
        options.append("payload")
    if args.estimate_arm:
        options.append("physical")
    huber = "auto" if args.huber_auto else args.huber_delta_nm
    fit = run_cpp(args.executable, "fit", nominal_path, directory, huber,
                  *options, *drive_options)
    if args.estimate_payload or args.estimate_arm:
        nominal = ET.parse(nominal_path).getroot()
        effort = np.asarray(metadata["joint_effort_limits_nm"])
        if args.estimate_payload:
            payload = fit["payload_fit"]
            payload["reference_errors"] = payload_reference_errors(
                payload, directory / "truth.urdf")
            candidate = payload["physical_candidate"]
            if candidate is not None:
                export_payload(candidate, nominal, metadata["payload_parent"],
                               directory, joints, drive, holdout, dense_holdout,
                               effort)
        if args.estimate_arm:
            export_identified_arm(fit, nominal, directory, joints, holdout,
                                  dense_holdout)
    return fit


def add_torque_alignment_arguments(parser):
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument(
        "--align-torque-ms",
        type=float,
        help="correct a known signed torque delay during raw Fourier fitting")
    alignment.add_argument(
        "--search-torque-delay-ms",
        type=float,
        metavar="BOUND",
        help=
        "estimate a signed torque delay within plus or minus BOUND milliseconds"
    )


def select_torque_alignment(args, directory, metadata):
    if args.search_torque_delay_ms is not None:
        bound = args.search_torque_delay_ms
        if not math.isfinite(
                bound
        ) or bound <= 0.0 or args.raw_estimator not in FOURIER_ESTIMATORS:
            raise ValueError(
                "--search-torque-delay-ms requires a finite positive bound and periodic Fourier fitting"
            )
        actuator_options = ["actuators"] if metadata.get("actuators") else []
        search_delta = 0.06 if args.huber_auto else args.huber_delta_nm
        return run_cpp(args.executable, "align", directory / "nominal.urdf",
                       directory, bound, args.raw_harmonics, args.raw_estimator,
                       search_delta, *actuator_options)
    if args.align_torque_ms is not None:
        alignment = {"delay_ms": args.align_torque_ms}
    else:
        alignment = metadata.get("torque_alignment", {"delay_ms": 0.0})
    delay = alignment["delay_ms"]
    if not math.isfinite(delay) or (delay and args.raw_estimator
                                    not in FOURIER_ESTIMATORS):
        raise ValueError(
            "torque alignment requires a finite delay and periodic Fourier fitting"
        )
    return alignment


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
                     args.training_period_s, args.holdout_period_s,
                     *design_options)
    result = {
        "urdf": arm.urdf,
        "urdf_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "fixed_base": arm.base,
        "joint_names": arm.joints,
        "joint_effort_limits_nm": effort.tolist(),
        "payload_parent": terminal,
        "design": design,
    }
    if args.design_only:
        return result

    truth_path = directory / "truth.urdf"
    truth = copy.deepcopy(nominal)
    add_payload(truth, terminal, args.payload_kg, args.payload_size_m,
                args.payload_center_m)
    write_urdf(truth, truth_path)

    motions = {
        split: read_motion(directory / f"{split}_states.csv", arm.joints)
        for split in ("baseline", "optimized", "holdout")
    }
    plant = ArmDynamics.from_urdf(truth_path, arm.joints, drive_parameters)
    if drive_parameters is not None:
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
            args.timestamp_jitter_ms / 1000, args.seed, args.replay,
            args.torque_delay_ms / 1000)
    else:
        labels = inverse_dynamics(plant, motions, effort)
    if drive_parameters is not None or args.replay:
        mujoco.mj_saveModel(plant.model, str(directory / "truth.mjb"))
    dof = len(arm.joints)
    if labels["baseline"].shape != labels["optimized"].shape:
        raise ValueError(
            "baseline and optimized trajectories must have the same sample count"
        )
    if args.capture_dropout:
        capture_random = np.random.default_rng(
            np.random.SeedSequence([args.seed, 3]))
        keep = capture_random.random(len(
            labels["baseline"])) >= args.capture_dropout
        keep[[0, -1]] = True
        if np.count_nonzero(keep) < 3:
            raise ValueError(
                "capture dropout left fewer than three training samples")
        for split in ("baseline", "optimized"):
            labels[split] = labels[split][keep]
    noise, outlier_count = make_noise(labels["baseline"][:, -dof:].shape,
                                      args.seed, args.torque_noise_nm,
                                      args.outlier_nm)
    encoder_noise, encoder_outlier_count = make_noise(
        noise.shape, np.random.SeedSequence([args.seed, 1]),
        args.encoder_noise_rad, args.encoder_outlier_rad)
    processed = {}
    estimator_size = args.raw_harmonics if args.raw_estimator in FOURIER_ESTIMATORS else args.raw_window
    for split in ("baseline", "optimized"):
        training = labels[split].copy()
        training[:, -dof:] += noise
        raw_path = directory / f"train_{split}_raw.csv"
        processed_path = directory / f"train_{split}.csv"
        write_raw(raw_path, training, arm.joints, encoder_noise,
                  args.encoder_resolution_rad)
        if args.fit_input == "simulator":
            write_dataset(processed_path, training, dataset_header(arm.joints))
    if args.fit_input == "raw":
        alignment = select_torque_alignment(args, directory, result)
        result["torque_alignment"] = alignment
        delay = alignment["delay_ms"]
        for split in ("baseline", "optimized"):
            raw_path = directory / f"train_{split}_raw.csv"
            processed_path = directory / f"train_{split}.csv"
            processed[split] = run_cpp(args.executable, "preprocess", raw_path,
                                       processed_path, estimator_size,
                                       args.raw_estimator, delay)
            if delay == 0.0:
                estimated = read_motion(processed_path, arm.joints)
                processed[split].update(
                    differentiation_metrics(estimated, labels[split], dof))
    write_raw(directory / "holdout_raw.csv", labels["holdout"], arm.joints, 0.0)
    write_dataset(directory / "holdout.csv", labels["holdout"],
                  dataset_header(arm.joints))
    if dense_holdout is not None:
        write_dataset(directory / "holdout_dense.csv", dense_holdout,
                      dataset_header(arm.joints))
    result.update({
        "samples": {
            split: len(dataset) for split, dataset in labels.items()
        },
        "peak_effort_fraction": {
            split: float(np.max(np.abs(dataset[:, -dof:]) / effort))
            for split, dataset in labels.items()
        },
        "outlier_count_per_training_trajectory":
            outlier_count,
        "encoder_outlier_count_per_training_trajectory":
            encoder_outlier_count,
        "acquisition":
            acquisition,
        "preprocessing":
            processed,
        "fit":
            fit_captured_arm(args, directory, result, labels["holdout"],
                             dense_holdout),
    })
    return result


def print_summary(report):
    if report["design_only"]:
        print(
            f"Exported {report['sample_count']} points per trajectory for {', '.join(report['robots'])}.",
            file=sys.stderr)
        return
    if report["estimate_arm"]:
        print("Physically constrained arm fit", file=sys.stderr)
    print(
        f"{'Arm':<10}{'Base rank':>10}{'Torque RMSE [N m]':>20}"
        f"{'Replay RMSE [rad]':>20}{'Payload [kg]':>16}",
        file=sys.stderr)
    for name, robot in report["robots"].items():
        fit = robot["fit"]
        physical = fit.get("physical_arm_fit")
        if physical is None:
            torque = fit["optimized"]["huber"]["holdout_rmse_nm"]
            replay = fit.get("replay", {}).get("optimized_huber",
                                               {}).get("position_rmse_rad")
        else:
            torque = physical["holdout_rmse_nm"]
            replay = physical.get("reloaded_mujoco_replay",
                                  {}).get("position_rmse_rad")
            if not physical["converged"]:
                print(f"{name}: physical fit reached its solver limit",
                      file=sys.stderr)
        payload = fit.get("payload_fit", {}).get("physical_candidate")
        replay_value = f"{replay:.5g}" if replay is not None else "—"
        payload_value = f"{payload['mass_kg']:.6g}" if payload is not None else "—"
        print(
            f"{name:<10}{fit['base_parameter_count']:>10}"
            f"{torque:>20.5g}"
            f"{replay_value:>20}{payload_value:>16}",
            file=sys.stderr)


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
    parser.add_argument("--summary",
                        action="store_true",
                        help="print concise per-arm results to stderr")
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
        "--training-period-s",
        type=float,
        default=8.0,
        help="duration of each generated training cycle in seconds")
    parser.add_argument(
        "--holdout-period-s",
        type=float,
        default=6.25,
        help="duration of the independent validation cycle in seconds")
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
        "--torque-delay-ms",
        type=float,
        default=0.0,
        help="signed simulated offset of the training torque channel")
    add_torque_alignment_arguments(parser)
    parser.add_argument(
        "--encoder-noise-rad",
        type=float,
        default=5e-5,
        help="training encoder noise standard deviation in recorded positions")
    parser.add_argument("--encoder-outlier-rad",
                        type=float,
                        default=0.0,
                        help="amplitude of randomly signed encoder outliers")
    parser.add_argument(
        "--encoder-resolution-rad",
        type=float,
        default=0.0,
        help="quantization step for recorded training positions")
    parser.add_argument(
        "--capture-dropout",
        type=float,
        default=0.0,
        help="probability of losing an interior training sample")
    parser.add_argument("--raw-window",
                        type=int,
                        default=5,
                        help="positive odd smoothing window for raw input")
    parser.add_argument(
        "--raw-estimator",
        choices=RAW_ESTIMATORS,
        default="moving-average",
        help="position derivative estimator for recorder-format input")
    parser.add_argument("--raw-harmonics",
                        type=int,
                        default=7,
                        help="harmonics fitted over one capture period")
    parser.add_argument("--seed",
                        type=int,
                        default=20261124,
                        help="training noise seed")
    parser.add_argument("--payload-kg",
                        type=float,
                        default=0.35,
                        help="added terminal payload in truth model")
    parser.add_argument("--payload-size-m",
                        nargs=3,
                        type=float,
                        metavar=("X", "Y", "Z"),
                        default=PAYLOAD_SIZE_METERS,
                        help="simulated box side lengths in meters")
    parser.add_argument("--payload-center-m",
                        nargs=3,
                        type=float,
                        metavar=("X", "Y", "Z"),
                        default=PAYLOAD_CENTER_METERS,
                        help="simulated box center in the terminal joint frame")
    parser.add_argument(
        "--actuator-scale",
        type=float,
        default=0.0,
        help="scale of synthetic rotor inertia and viscous joint friction")
    parser.add_argument(
        "--estimate-payload",
        action="store_true",
        help=
        "calibrate and export payload inertia using the known arm and drive model"
    )
    parser.add_argument(
        "--estimate-arm",
        action="store_true",
        help="fit and export a physically consistent arm and drive model")
    parser.add_argument("--torque-noise-nm",
                        type=float,
                        default=0.02,
                        help="training Gaussian torque noise sigma")
    parser.add_argument("--outlier-nm",
                        type=float,
                        default=0.75,
                        help="training outlier magnitude")
    huber_group = parser.add_mutually_exclusive_group()
    huber_group.add_argument("--huber-delta-nm",
                             type=float,
                             default=0.06,
                             help="fixed Huber threshold")
    huber_group.add_argument(
        "--huber-auto",
        action="store_true",
        help="calibrate Huber threshold from training residuals")
    args = parser.parse_args()
    for field in ("payload_kg", "actuator_scale", "torque_noise_nm",
                  "outlier_nm", "encoder_noise_rad", "timestamp_jitter_ms",
                  "huber_delta_nm", "training_period_s", "holdout_period_s",
                  "encoder_resolution_rad", "encoder_outlier_rad",
                  "capture_dropout"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"invalid --{field.replace('_', '-')} value")
    if min(args.training_period_s, args.holdout_period_s,
           args.huber_delta_nm) == 0.0:
        parser.error(
            "trajectory periods and the fixed Huber threshold must be positive")
    if not all(
            math.isfinite(value) and value > 0
            for value in args.payload_size_m):
        parser.error("--payload-size-m requires three finite positive lengths")
    if not all(map(math.isfinite, args.payload_center_m)):
        parser.error("--payload-center-m requires three finite coordinates")
    if args.capture_dropout >= 1.0:
        parser.error("--capture-dropout must be less than one")
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
    if (args.fit_input == "raw" and args.raw_estimator in FOURIER_ESTIMATORS and
            args.sample_count < 2 * args.raw_harmonics + 2):
        parser.error(
            "periodic Fourier estimators require at least 2h+2 samples")
    if args.timestamp_jitter_ms > 0.0 and args.acquisition != "rollout":
        parser.error("--timestamp-jitter-ms requires --acquisition rollout")
    if not math.isfinite(
            args.torque_delay_ms) or (args.align_torque_ms is not None and
                                      not math.isfinite(args.align_torque_ms)):
        parser.error("torque delay and alignment must be finite")
    if args.torque_delay_ms and (args.acquisition != "rollout" or
                                 args.design_only):
        parser.error("--torque-delay-ms requires simulated rollout capture")
    if args.search_torque_delay_ms is not None and (not math.isfinite(
            args.search_torque_delay_ms) or args.search_torque_delay_ms <= 0.0):
        parser.error(
            "--search-torque-delay-ms requires a finite positive bound")
    align_requested = args.align_torque_ms or args.search_torque_delay_ms is not None
    if align_requested and (args.fit_input != "raw" or args.raw_estimator
                            not in FOURIER_ESTIMATORS or args.design_only):
        parser.error(
            "torque alignment requires fitting raw data with a periodic Fourier estimator"
        )
    if args.replay and (args.acquisition != "rollout" or args.design_only):
        parser.error(
            "--replay requires --acquisition rollout with simulation and fitting"
        )
    args.executable = args.executable.resolve()
    if args.estimate_payload and args.design_only:
        parser.error("--estimate-payload requires simulated torque")
    if args.estimate_arm and args.design_only:
        parser.error("--estimate-arm requires simulated torque")
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
            "training_period_s": args.training_period_s,
            "holdout_period_s": args.holdout_period_s,
            "fit_input": args.fit_input,
            "seed": args.seed,
            "payload_kg": args.payload_kg,
            "estimate_payload": args.estimate_payload,
            "estimate_arm": args.estimate_arm,
            "actuator_scale": args.actuator_scale,
            "payload_box_size_m": args.payload_size_m,
            "payload_center_xyz_m": args.payload_center_m,
            "torque_noise_nm": args.torque_noise_nm,
            "encoder_noise_rad": args.encoder_noise_rad,
            "encoder_outlier_rad": args.encoder_outlier_rad,
            "encoder_resolution_rad": args.encoder_resolution_rad,
            "capture_dropout": args.capture_dropout,
            "timestamp_jitter_ms": args.timestamp_jitter_ms,
            "torque_delay_ms": args.torque_delay_ms,
            "align_torque_ms": args.align_torque_ms or 0.0,
            "search_torque_delay_ms": args.search_torque_delay_ms,
            "raw_window": args.raw_window,
            "raw_estimator": args.raw_estimator,
            "raw_harmonics": args.raw_harmonics,
            "outlier_nm": args.outlier_nm,
            "outlier_probability": OUTLIER_PROBABILITY,
            "huber_delta_nm": None if args.huber_auto else args.huber_delta_nm,
            "huber_auto": args.huber_auto,
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
    if args.summary:
        print_summary(report)
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
