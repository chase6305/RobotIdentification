"""Refit a saved MuJoCo capture from its recorded encoder and torque CSVs."""

import argparse
from contextlib import suppress
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from arm_models import ARMS
from capture_io import read_motion, write_median_raw
from run_experiment import (RAW_ESTIMATORS, add_torque_alignment_arguments,
                            differentiation_metrics, fit_captured_arm,
                            preprocess_training_split, print_summary,
                            select_torque_alignment)

_ARM_MODEL_FILES = ("nominal.urdf", "truth.urdf")
_TRAINING_SPLITS = ("baseline", "optimized")


def collect_capture_dirs(primary_dir, repeat_paths, primary_report, arm_names):
    """Return the primary capture followed by the validated repeats."""
    capture_dirs = [primary_dir]
    for repeat_path in repeat_paths:
        repeat_dir = repeat_path.resolve(strict=True)
        if repeat_dir in capture_dirs:
            raise ValueError(f"duplicate repeat capture {repeat_dir}")
        repeat_report = json.loads(
            (repeat_dir / "report.json").read_text(encoding="utf-8"))
        if repeat_report["design_only"]:
            raise ValueError(
                f"repeat capture {repeat_dir} has reference points only")
        primary_delay = primary_report.get("torque_delay_ms", 0.0)
        repeat_delay = repeat_report.get("torque_delay_ms", 0.0)
        if (repeat_report["acquisition"] != primary_report["acquisition"] or
                repeat_delay != primary_delay):
            raise ValueError(
                f"repeat capture {repeat_dir} has different acquisition settings"
            )
        for arm_name in arm_names:
            primary_arm = primary_report["robots"][arm_name]
            repeat_arm = repeat_report["robots"].get(arm_name)
            if repeat_arm is None:
                raise ValueError(
                    f"{arm_name} is missing from repeat capture {repeat_dir}")
            if (repeat_arm["joint_names"] != primary_arm["joint_names"] or
                    repeat_arm.get("actuators") != primary_arm.get("actuators")
                    or repeat_arm["design"]["references"]
                    != primary_arm["design"]["references"]):
                raise ValueError(
                    f"{arm_name} uses different joints, drive or references in {repeat_dir}"
                )
            for filename in _ARM_MODEL_FILES:
                primary_model = primary_dir / arm_name / filename
                repeat_model = repeat_dir / arm_name / filename
                if repeat_model.read_bytes() != primary_model.read_bytes():
                    raise ValueError(
                        f"{arm_name} uses a different {filename} in {repeat_dir}"
                    )
        capture_dirs.append(repeat_dir)
    return capture_dirs


def refit_arm(args, source_report, arm_name, output, capture_dirs):
    arm_dirs = [capture_dir / arm_name for capture_dir in capture_dirs]
    primary_arm_dir = arm_dirs[0]
    arm_metadata = source_report["robots"][arm_name]
    arm_report = dict(arm_metadata)
    output_dir = output / arm_name
    output_dir.mkdir()

    holdout_files = ("holdout.csv",)
    if source_report["replay"]:
        holdout_files += ("holdout_dense.csv",)
    for filename in (*_ARM_MODEL_FILES, "calibration.csv", *holdout_files):
        shutil.copyfile(primary_arm_dir / filename, output_dir / filename)

    joints = arm_metadata["joint_names"]
    for split in _TRAINING_SPLITS:
        filename = f"train_{split}_raw.csv"
        raw_inputs = [arm_dir / filename for arm_dir in arm_dirs]
        if len(raw_inputs) == 1:
            shutil.copyfile(raw_inputs[0], output_dir / filename)
        else:
            write_median_raw(output_dir / filename, raw_inputs, joints)
    # The native fit checks the remaining rows after this capture-specific prefix.
    for filename in holdout_files:
        read_motion(output_dir / filename, joints, max_rows=3)

    arm_report.setdefault(
        "torque_alignment",
        {"delay_ms": source_report.get("align_torque_ms", 0.0)})
    alignment = select_torque_alignment(args, output_dir, arm_report)
    arm_report["torque_alignment"] = alignment
    delay_ms = alignment["delay_ms"]

    preprocessing = {}
    for split in _TRAINING_SPLITS:
        split_metrics = preprocess_training_split(args, output_dir, split,
                                                  delay_ms)
        if source_report["fit_input"] == "simulator" and delay_ms == 0.0:
            filename = f"train_{split}.csv"
            measured = read_motion(primary_arm_dir / filename, joints)
            estimated = read_motion(output_dir / filename, joints)
            split_metrics.update(
                differentiation_metrics(estimated, measured, len(joints)))
        preprocessing[split] = split_metrics

    arm_report["preprocessing"] = preprocessing
    arm_report["fit"] = fit_captured_arm(args, output_dir, arm_metadata)
    return arm_report


def run(args):
    source_report = json.loads(
        (args.capture / "report.json").read_text(encoding="utf-8"))
    if source_report["design_only"]:
        raise ValueError("the input directory has reference points only")
    available_arms = source_report["robots"]
    arm_names = tuple(available_arms) if args.robot == "both" else (args.robot,)
    if not arm_names or any(name not in available_arms or name not in ARMS
                            for name in arm_names):
        raise ValueError("the requested arm is missing from the capture")
    capture_dirs = collect_capture_dirs(args.capture, args.repeat_capture,
                                        source_report, arm_names)
    for name, default in (("raw_estimator", "periodic-fourier"),
                          ("raw_harmonics", 7), ("raw_window", 5),
                          ("estimate_arm", False), ("estimate_payload", False)):
        if getattr(args, name) is None:
            setattr(args, name, source_report.get(name, default))
    if not args.huber_auto and args.huber_delta_nm is None:
        args.huber_auto = source_report.get("huber_auto", False)
        args.huber_delta_nm = source_report.get("huber_delta_nm", .06)
    args.executable = args.executable.resolve(strict=True)

    # Reserve the final path before work. Publish both arms and the report together.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix=".arm-refit-",
                                         dir=args.output.parent) as temporary:
            staged = Path(temporary) / "capture"
            staged.mkdir()
            report = write_refit(args, source_report, arm_names, staged,
                                 capture_dirs)
            staged.rename(args.output)
    except BaseException:
        with suppress(OSError):
            args.output.rmdir()
        raise
    if args.summary:
        print_summary(report)
    print(args.output / "report.json")


def write_refit(args, source_report, arm_names, output, capture_dirs):
    manual_delay_ms = args.align_torque_ms
    if manual_delay_ms is None:
        recorded_delay_ms = source_report.get("align_torque_ms", 0.0)
        manual_delay_ms = (0.0 if args.search_torque_delay_ms is not None else
                           recorded_delay_ms)
    capture_version = source_report.get("capture_mujoco_version",
                                        source_report["mujoco_version"])
    report = dict(
        source_report,
        refit_source=str(capture_dirs[0]),
        capture_mujoco_version=capture_version,
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        fit_input="raw",
        raw_estimator=args.raw_estimator,
        raw_harmonics=args.raw_harmonics,
        raw_window=args.raw_window,
        align_torque_ms=manual_delay_ms,
        search_torque_delay_ms=args.search_torque_delay_ms,
        huber_auto=args.huber_auto,
        huber_delta_nm=None if args.huber_auto else args.huber_delta_nm,
        estimate_arm=args.estimate_arm,
        estimate_payload=args.estimate_payload)
    report["robots"] = {
        name: refit_arm(args, source_report, name, output, capture_dirs)
        for name in arm_names
    }
    if len(capture_dirs) > 1:
        report["training_capture_aggregation"] = {
            "method": "median",
            "captures": [str(path) for path in capture_dirs],
        }
    destination = output / "report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",
                        type=Path,
                        help="existing arm experiment directory")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output",
                        type=Path,
                        required=True,
                        help="new refit directory")
    parser.add_argument("--robot", choices=("both", *ARMS), default="both")
    parser.add_argument(
        "--repeat-capture",
        type=Path,
        action="append",
        default=[],
        help="add a synchronized capture to the sample-wise median")
    parser.add_argument("--raw-estimator", choices=RAW_ESTIMATORS)
    parser.add_argument("--raw-window", type=int)
    parser.add_argument("--raw-harmonics", type=int)
    add_torque_alignment_arguments(parser)
    parser.add_argument("--estimate-arm", action=argparse.BooleanOptionalAction)
    parser.add_argument("--estimate-payload",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--summary", action="store_true")
    huber = parser.add_mutually_exclusive_group()
    huber.add_argument("--huber-auto", action="store_true")
    huber.add_argument("--huber-delta-nm", type=float)
    args = parser.parse_args()
    try:
        args.capture = args.capture.resolve(strict=True)
        run(args)
    except (KeyError, OSError, ET.ParseError, ValueError) as error:
        print(f"arm refit: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
