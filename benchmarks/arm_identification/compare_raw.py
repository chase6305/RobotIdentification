"""Compare raw preprocessing windows on existing simulated arm captures."""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

from run_experiment import differentiation_metrics, read_motion, run_cpp

ESTIMATOR_SIZES = {
    "moving-average": (1, 3, 5, 7, 9, 13, 17),
    "local-cubic": (5, 7, 9, 13, 17),
    "periodic-fourier": (3, 5, 7, 9, 13, 17),
}


def compare_arm(executable, case, report, name, source, estimator_sizes):
    joints = source["joint_names"]
    directory = case / name
    splits = ("baseline", "optimized")
    measured = {
        split: read_motion(directory / f"train_{split}.csv", joints)
        for split in splits
    }
    actuators = source.get("actuators", {})
    drive_options = [
        actuators[joint][field]
        for joint in joints
        for field in ("armature_kg_m2", "viscous_nm_s_per_rad")
    ] if actuators else []
    huber = "auto" if report.get("huber_auto",
                                 False) else report["huber_delta_nm"]
    with tempfile.TemporaryDirectory() as temporary:
        replay = Path(temporary)
        for filename in ("nominal.urdf", "truth.urdf", "calibration.csv",
                         "holdout.csv"):
            (replay / filename).symlink_to(
                (directory / filename).resolve(strict=True))
        for method, lengths in estimator_sizes.items():
            for length in lengths:
                errors = {}
                for split in splits:
                    raw = directory / f"train_{split}_raw.csv"
                    processed = replay / f"train_{split}.csv"
                    run_cpp(executable, "preprocess", raw, processed, length,
                            method)
                    errors[split] = differentiation_metrics(
                        read_motion(processed, joints), measured[split],
                        len(joints))
                fit = run_cpp(executable, "fit", replay / "nominal.urdf",
                              replay, huber, *drive_options)
                fitted = fit["optimized"]["huber"]
                yield {
                    "case":
                        str(case),
                    "robot":
                        name,
                    "seed":
                        report["seed"],
                    "samples":
                        report["sample_count"],
                    "payload_kg":
                        report["payload_kg"],
                    "actuator_scale":
                        report["actuator_scale"],
                    "encoder_noise_rad":
                        report["encoder_noise_rad"],
                    "timestamp_jitter_ms":
                        report["timestamp_jitter_ms"],
                    "method":
                        method,
                    "window_or_harmonics":
                        length,
                    "baseline_acceleration_rmse_rad_s2":
                        errors["baseline"]["acceleration_rmse_rad_s2"],
                    "optimized_acceleration_rmse_rad_s2":
                        errors["optimized"]["acceleration_rmse_rad_s2"],
                    "optimized_holdout_rmse_nm":
                        fitted["holdout_rmse_nm"],
                    "optimized_relative_base_error":
                        fitted["relative_base_error"],
                    "optimized_mass_check_passed":
                        fitted["forward_validation"]["mass_check_passed"],
                    "optimized_holdout_acceleration_rmse_rad_s2":
                        fitted["forward_validation"]
                        ["holdout_acceleration_rmse_rad_s2"],
                }


def run_cases(executable, cases, estimator_sizes):
    for case in cases:
        report = json.loads((case / "report.json").read_text(encoding="utf-8"))
        if report["design_only"] or report["fit_input"] != "simulator":
            raise ValueError(
                f"{case} must contain raw samples and simulated reference derivatives"
            )
        for name, source in report["robots"].items():
            yield from compare_arm(executable, case, report, name, source,
                                   estimator_sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output",
                        required=True,
                        type=Path,
                        help="new CSV for the comparison results")
    parser.add_argument(
        "cases",
        nargs="+",
        type=Path,
        help="experiment directories generated with --fit-input simulator")
    args = parser.parse_args()
    try:
        rows = run_cases(args.executable.resolve(), args.cases, ESTIMATOR_SIZES)
        first = next(rows)
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,
                                    fieldnames=list(first),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)
    except (KeyError, OSError, ValueError) as error:
        print(f"raw preprocessing comparison: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
