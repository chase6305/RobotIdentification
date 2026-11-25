"""Identify fixed-base arm inertial combinations against MuJoCo torque labels."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from arm_models import (ARMS, PAYLOAD_SIZE_METERS, add_payload,
                        export_identified_arm, export_payload, extract_arm,
                        flange_box_pose, payload_reference_errors, write_urdf)
from capture_io import dataset_header, read_motion, write_dataset, write_raw
from simulation import (ArmDynamics, harmonic_states, inverse_dynamics, rollout,
                        validate_workspace)

OUTLIER_PROBABILITY = 0.01
FOURIER_ESTIMATORS = ("periodic-fourier", "periodic-fourier-huber")
RAW_ESTIMATORS = ("moving-average", "local-cubic", *FOURIER_ESTIMATORS)
REFERENCE_DEFAULTS = {
    "sample_count": 401,
    "design_evaluations": 240,
    "training_period_s": 8.0,
    "holdout_period_s": 6.25,
}


def configure_reference_plan(args):
    plan = None
    if args.reference_plan is not None:
        args.reference_plan = args.reference_plan.resolve(strict=True)
        plan = json.loads(
            (args.reference_plan / "report.json").read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or not isinstance(plan.get("robots"),
                                                        dict):
            raise ValueError("saved reference plan has no arm metadata")
    for name, default in REFERENCE_DEFAULTS.items():
        requested = getattr(args, name)
        if plan is None:
            if requested is None:
                setattr(args, name, default)
        else:
            planned = plan.get(name, default)
            valid_types = (int,) if isinstance(default, int) else (int, float)
            if type(planned) not in valid_types or (requested is not None and
                                                    planned != requested):
                raise ValueError(
                    f"--{name.replace('_', '-')} must agree with the saved reference plan"
                )
            setattr(args, name, type(default)(planned))
    planned_scale = plan.get("actuator_scale", 0.0) if plan is not None else 0.0
    if (type(planned_scale) not in (int, float) or
            not math.isfinite(planned_scale) or planned_scale < 0.0):
        raise ValueError("saved reference plan has an invalid actuator scale")
    if args.actuator_scale is None:
        args.actuator_scale = planned_scale
    elif plan is not None and (args.actuator_scale > 0.0) != (planned_scale
                                                              > 0.0):
        raise ValueError(
            "the reference plan must use the requested regressor structure")
    return plan


def copy_arm_reference_plan(args, name, directory, joints, source_hash):
    record = args.reference_plan_report["robots"][name]
    if record["urdf_sha256"] != source_hash or tuple(
            record["joint_names"]) != joints:
        raise ValueError(
            f"{name}: saved reference plan uses a different URDF or joint order"
        )
    design = record["design"]
    source = args.reference_plan / name
    for split in ("baseline", "optimized", "holdout"):
        filename = f"{split}_states.csv"
        destination = directory / filename
        shutil.copyfile(source / filename, destination)
        points = read_motion(destination, joints)
        period = args.holdout_period_s if split == "holdout" else args.training_period_s
        if len(points) != args.sample_count or not np.allclose(
                points[:, 0],
                np.linspace(0.0, period, args.sample_count),
                rtol=0.0,
                atol=1e-12):
            raise ValueError(
                f"{name}/{split}: saved timestamps disagree with the plan")
        state = np.hstack(
            harmonic_states(design["references"][split], points[:, 0]))
        recorded = points[:, 1:-len(joints)]
        if state.shape != recorded.shape or not np.allclose(
                state, recorded, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{name}/{split}: saved states disagree with the Fourier coefficients"
            )
    shutil.copyfile(source / "calibration.csv", directory / "calibration.csv")
    read_motion(directory / "calibration.csv", joints)
    return design


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
            identified = export_identified_arm(fit, nominal, directory, joints,
                                               holdout, dense_holdout)
            limits = metadata["design"].get("kinematic_limits")
            if limits is not None:
                truth = ArmDynamics.from_urdf(directory / "truth.urdf", joints,
                                              drive)
                nominal_arm = ArmDynamics.from_urdf(nominal_path, joints)
                fit["physical_arm_fit"][
                    "workspace_validation"] = validate_workspace(
                        truth, nominal_arm, identified, limits, effort)
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
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = ET.fromstring(source_bytes)
    nominal, terminal, effort = extract_arm(source, arm, name)
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
    if args.reference_plan_report is None:
        design = run_cpp(args.executable, "design", nominal_path, directory,
                         args.sample_count, args.design_evaluations,
                         args.training_period_s, args.holdout_period_s,
                         *design_options)
    else:
        design = copy_arm_reference_plan(args, name, directory, arm.joints,
                                         source_hash)
    result = {
        "urdf": arm.urdf,
        "urdf_sha256": source_hash,
        "fixed_base": arm.base,
        "joint_names": arm.joints,
        "joint_effort_limits_nm": effort.tolist(),
        "payload_parent": terminal,
        "design": design,
    }
    if args.design_only:
        return result

    center = args.payload_center_m
    rotation = None
    if center is None:
        center, rotation = flange_box_pose(source, arm, terminal,
                                           args.payload_size_m[2])
        result["payload_flange_joint"] = arm.flange_joint
    result["payload_center_xyz_m"] = np.asarray(center).tolist()

    truth_path = directory / "truth.urdf"
    truth = copy.deepcopy(nominal)
    add_payload(truth, terminal, args.payload_kg, args.payload_size_m, center,
                rotation)
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


def parse_jitter(value):
    return value if value == "auto" else float(value)


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
    parser.add_argument(
        "--reference-plan",
        type=Path,
        help=
        "reuse previously exported reference points and Fourier coefficients")
    parser.add_argument("--sample-count",
                        type=int,
                        help="number of reference points per trajectory")
    parser.add_argument("--design-evaluations",
                        type=int,
                        help="Fourier design search evaluation budget")
    parser.add_argument(
        "--training-period-s",
        type=float,
        help="duration of each generated training cycle in seconds")
    parser.add_argument(
        "--holdout-period-s",
        type=float,
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
        type=parse_jitter,
        default=0.0,
        help=
        "capture jitter in milliseconds, or auto for up to 2 ms within the sampling interval"
    )
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
                        help="override box center in the terminal link frame")
    parser.add_argument(
        "--actuator-scale",
        type=float,
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
    try:
        args.reference_plan_report = configure_reference_plan(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for field in ("payload_kg", "actuator_scale", "torque_noise_nm",
                  "outlier_nm", "encoder_noise_rad", "huber_delta_nm",
                  "training_period_s", "holdout_period_s",
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
    if args.payload_center_m is not None and not all(
            map(math.isfinite, args.payload_center_m)):
        parser.error("--payload-center-m requires three finite coordinates")
    if args.capture_dropout >= 1.0:
        parser.error("--capture-dropout must be less than one")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.sample_count < 3:
        parser.error("--sample-count must be at least three")
    jitter_limit = 500 * (min(args.training_period_s, args.holdout_period_s) /
                          (args.sample_count - 1))
    if args.timestamp_jitter_ms == "auto":
        args.timestamp_jitter_ms = min(2.0, 0.8 * jitter_limit)
    if not math.isfinite(
            args.timestamp_jitter_ms) or args.timestamp_jitter_ms < 0.0:
        parser.error(
            "--timestamp-jitter-ms must be nonnegative and finite or auto")
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
    if not args.design_only and args.timestamp_jitter_ms >= jitter_limit:
        parser.error(
            f"--timestamp-jitter-ms must be below {jitter_limit:.6g} ms at this sample spacing"
        )
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
        if args.reference_plan_report is not None:
            planned_robots = args.reference_plan_report["robots"]
            if args.robot == "both":
                robots = tuple(name for name in ARMS if name in planned_robots)
            if not robots or any(name not in planned_robots for name in robots):
                raise ValueError(
                    "the saved reference plan does not cover the requested arm")
        for name in robots:
            if not (args.assets_root / ARMS[name].urdf).is_file():
                raise ValueError(
                    f"cannot find {ARMS[name].urdf} in {args.assets_root}")
        if not args.executable.is_file():
            raise ValueError(f"cannot find executable {args.executable}")
        args.output.mkdir(parents=True, exist_ok=False)
        plan_path = str(
            args.reference_plan) if args.reference_plan is not None else None
        report = {
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "acquisition": args.acquisition,
            "replay": args.replay,
            "design_only": args.design_only,
            "reference_plan": plan_path,
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
