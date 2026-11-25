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
from capture_io import read_motion
from run_experiment import (RAW_ESTIMATORS, add_torque_alignment_arguments,
                            differentiation_metrics, fit_captured_arm,
                            preprocess_training_split, print_summary,
                            select_torque_alignment)


def refit_arm(args, source_report, name, output):
    source = args.capture / name
    metadata = source_report["robots"][name]
    result = dict(metadata)
    directory = output / name
    directory.mkdir()

    holdout_files = ("holdout.csv",)
    if source_report["replay"]:
        holdout_files += ("holdout_dense.csv",)
    files = ("nominal.urdf", "truth.urdf", "calibration.csv",
             "train_baseline_raw.csv", "train_optimized_raw.csv",
             *holdout_files)
    for filename in files:
        shutil.copyfile(source / filename, directory / filename)

    joints = metadata["joint_names"]
    # The native fit checks the remaining rows after this capture-specific prefix.
    for filename in holdout_files:
        read_motion(directory / filename, joints, max_rows=3)

    result.setdefault("torque_alignment",
                      {"delay_ms": source_report.get("align_torque_ms", 0.0)})
    result["torque_alignment"] = select_torque_alignment(
        args, directory, result)
    delay = result["torque_alignment"]["delay_ms"]
    processed = {}
    for split in ("baseline", "optimized"):
        processed[split] = preprocess_training_split(args, directory, split,
                                                     delay)
        if source_report["fit_input"] == "simulator" and delay == 0.0:
            measured = read_motion(source / f"train_{split}.csv", joints)
            estimated = read_motion(directory / f"train_{split}.csv", joints)
            processed[split].update(
                differentiation_metrics(estimated, measured, len(joints)))

    result["preprocessing"] = processed
    result["fit"] = fit_captured_arm(args, directory, metadata)
    return result


def run(args):
    source = json.loads(
        (args.capture / "report.json").read_text(encoding="utf-8"))
    if source["design_only"]:
        raise ValueError("the input directory has reference points only")
    robots = tuple(source["robots"]) if args.robot == "both" else (args.robot,)
    if not robots or any(name not in source["robots"] or name not in ARMS
                         for name in robots):
        raise ValueError("the requested arm is missing from the capture")
    for name, default in (("raw_estimator", "periodic-fourier"),
                          ("raw_harmonics", 7), ("raw_window", 5),
                          ("estimate_arm", False), ("estimate_payload", False)):
        if getattr(args, name) is None:
            setattr(args, name, source.get(name, default))
    if not args.huber_auto and args.huber_delta_nm is None:
        args.huber_auto = source.get("huber_auto", False)
        args.huber_delta_nm = source.get("huber_delta_nm", .06)
    args.executable = args.executable.resolve(strict=True)

    # Reserve the final path before work. Publish both arms and the report together.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix=".arm-refit-",
                                         dir=args.output.parent) as temporary:
            staged = Path(temporary) / "capture"
            staged.mkdir()
            report = write_refit(args, source, robots, staged)
            staged.rename(args.output)
    except BaseException:
        with suppress(OSError):
            args.output.rmdir()
        raise
    if args.summary:
        print_summary(report)
    print(args.output / "report.json")


def write_refit(args, source, robots, output):
    manual_delay = args.align_torque_ms
    if manual_delay is None:
        manual_delay = 0.0 if args.search_torque_delay_ms is not None else source.get(
            "align_torque_ms", 0.0)
    report = dict(
        source,
        refit_source=str(args.capture),
        capture_mujoco_version=source.get("capture_mujoco_version",
                                          source["mujoco_version"]),
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        fit_input="raw",
        raw_estimator=args.raw_estimator,
        raw_harmonics=args.raw_harmonics,
        raw_window=args.raw_window,
        align_torque_ms=manual_delay,
        search_torque_delay_ms=args.search_torque_delay_ms,
        huber_auto=args.huber_auto,
        huber_delta_nm=None if args.huber_auto else args.huber_delta_nm,
        estimate_arm=args.estimate_arm,
        estimate_payload=args.estimate_payload,
        robots={name: refit_arm(args, source, name, output) for name in robots})
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
