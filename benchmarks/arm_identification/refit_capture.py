"""Refit a saved MuJoCo capture from its recorded encoder and torque CSVs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from run_experiment import (ARMS, FOURIER_ESTIMATORS, RAW_ESTIMATORS,
                            differentiation_metrics, fit_captured_arm,
                            print_summary, read_motion, run_cpp)


def refit_arm(args, source_report, name):
    source = args.capture / name
    metadata = source_report["robots"][name]
    result = dict(metadata)
    directory = args.output / name
    directory.mkdir()

    files = ("nominal.urdf", "truth.urdf", "calibration.csv", "holdout.csv",
             "train_baseline_raw.csv", "train_optimized_raw.csv")
    if source_report["replay"]:
        files += ("holdout_dense.csv",)
    for filename in files:
        shutil.copyfile(source / filename, directory / filename)

    joints = metadata["joint_names"]
    estimator_size = (args.raw_harmonics if args.raw_estimator
                      in FOURIER_ESTIMATORS else args.raw_window)
    processed = {}
    for split in ("baseline", "optimized"):
        processed_path = directory / f"train_{split}.csv"
        processed[split] = run_cpp(args.executable, "preprocess",
                                   directory / f"train_{split}_raw.csv",
                                   processed_path, estimator_size,
                                   args.raw_estimator)
        if source_report["fit_input"] == "simulator":
            measured = read_motion(source / f"train_{split}.csv", joints)
            estimated = read_motion(processed_path, joints)
            processed[split].update(
                differentiation_metrics(estimated, measured, len(joints)))

    holdout = read_motion(directory / "holdout.csv", joints)
    dense = read_motion(directory / "holdout_dense.csv",
                        joints) if source_report["replay"] else None
    result["preprocessing"] = processed
    result["fit"] = fit_captured_arm(args, directory, metadata, holdout, dense)
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
    args.output.mkdir(parents=True, exist_ok=False)
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
        huber_auto=args.huber_auto,
        huber_delta_nm=None if args.huber_auto else args.huber_delta_nm,
        estimate_arm=args.estimate_arm,
        estimate_payload=args.estimate_payload,
        robots={name: refit_arm(args, source, name) for name in robots})
    destination = args.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    if args.summary:
        print_summary(report)
    print(destination)


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
