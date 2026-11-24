"""Verify raw preprocessing on exact cubic and periodic trajectories."""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

TIMESTAMPS = (0.0, 0.025, 0.069, 0.102, 0.16, 0.192, 0.244, 0.287, 0.301)


def maximum_state_error(executable, rows, expected, method, size):
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "raw.csv"
        output = Path(directory) / "processed.csv"
        with raw.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("time_s", "q/joint", "tau/joint"))
            writer.writerows(rows)
        subprocess.run([
            executable, "preprocess",
            str(raw),
            str(output),
            str(size), method
        ],
                       check=True,
                       capture_output=True)
        with output.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if next(reader) != [
                    "time_s", "q/joint", "dq/joint", "ddq/joint", "tau/joint"
            ]:
                raise ValueError("unexpected processed CSV columns")
            actual = [tuple(map(float, row)) for row in reader]
    if len(actual) != len(expected):
        raise ValueError(f"{method}: incorrect endpoint handling")
    for measured, oracle in zip(actual, expected):
        if not all(map(math.isfinite, measured)):
            raise ValueError("preprocessing returned nonfinite samples")
        if measured[0] != oracle[0] or measured[-1] != oracle[-1]:
            raise ValueError(
                "preprocessing changed timestamps or torque labels")
    return max(
        abs(measured[column] - oracle[column])
        for measured, oracle in zip(actual, expected)
        for column in (1, 2, 3))


def check_analytic(executable, rows, expected, method, size):
    if maximum_state_error(executable, rows, expected, method, size) > 1e-10:
        raise ValueError(
            f"{method} did not recover analytic state and derivatives")


def periodic_fixture(timestamps):
    omega = 2 * math.pi / timestamps[-1]
    raw = []
    expected = []
    for time in timestamps:
        angle = omega * time
        position = 2 + 0.3 * math.sin(angle) + 0.1 * math.cos(2 * angle)
        velocity = omega * (0.3 * math.cos(angle) - 0.2 * math.sin(2 * angle))
        acceleration = -omega**2 * (0.3 * math.sin(angle) +
                                    0.4 * math.cos(2 * angle))
        torque = 0.5 - time
        raw.append((time, position, torque))
        expected.append((time, position, velocity, acceleration, torque))
    return raw, expected


def check(executable):
    rows = [(time, 2.0 + 0.4 * time - 0.7 * time**2 + 0.9 * time**3, 0.5 - time)
            for time in TIMESTAMPS]
    expected = [(time, position, 0.4 - 1.4 * time + 2.7 * time**2,
                 -1.4 + 5.4 * time, torque)
                for time, position, torque in rows[2:-2]]
    check_analytic(executable, rows, expected, "local-cubic", 5)

    raw, expected = periodic_fixture(TIMESTAMPS)
    check_analytic(executable, raw, expected, "periodic-fourier", 2)
    check_analytic(executable, raw, expected, "periodic-fourier-huber", 2)

    timestamps = [4 * (index / 128)**1.1 for index in range(129)]
    raw, expected = periodic_fixture(timestamps)
    for index, outlier in ((11, .15), (62, -.13), (114, .20)):
        time, position, torque = raw[index]
        raw[index] = time, position + outlier, torque
    ordinary = maximum_state_error(executable, raw, expected,
                                   "periodic-fourier", 2)
    robust = maximum_state_error(executable, raw, expected,
                                 "periodic-fourier-huber", 2)
    if ordinary < .005 or robust >= .2 * ordinary:
        raise ValueError("robust Fourier failed to reject encoder outliers")


def check_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report["fit_input"] != "raw" or report["raw_estimator"] != "local-cubic":
        raise ValueError("expected a raw local-cubic arm experiment")
    for arm, rank in (("pm01", 33), ("openarm", 43)):
        data = report["robots"][arm]
        for motion, acquisition in data["acquisition"].items():
            if not 0 < acquisition["max_timestamp_jitter_s"] <= report[
                    "timestamp_jitter_ms"] / 1000 + 1e-12:
                raise ValueError(
                    f"{arm}/{motion}: capture jitter did not match the requested range"
                )
        for name, metrics in data["preprocessing"].items():
            if metrics["used_samples"] != metrics["input_samples"] - report[
                    "raw_window"] + 1:
                raise ValueError(
                    f"{arm}/{name}: unexpected local-cubic window trimming")
            if not 0 < metrics["acceleration_rmse_rad_s2"] < 0.5:
                raise ValueError(
                    f"{arm}/{name}: inaccurate estimated acceleration")
        fit = data["fit"]["optimized"]["huber"]
        if fit["rank"] != rank or not fit["holdout_rmse_nm"] < 0.02:
            raise ValueError(
                f"{arm}: raw data did not recover the expected model")


def check_encoder_outliers(ordinary_path, robust_path):
    ordinary = json.loads(Path(ordinary_path).read_text(encoding="utf-8"))
    robust = json.loads(Path(robust_path).read_text(encoding="utf-8"))
    if robust["raw_estimator"] != "periodic-fourier-huber":
        raise ValueError("expected a robust periodic encoder refit")
    for name, arm in robust["robots"].items():
        thresholds = arm["preprocessing"]["optimized"]["huber_delta_rad"]
        if (arm["encoder_outlier_count_per_training_trajectory"] < 1 or
                len(thresholds) != len(arm["joint_names"]) or
                min(thresholds) <= 0):
            raise ValueError(
                f"{name}: expected spikes and positive encoder Huber thresholds"
            )
        fit = arm["fit"]["physical_arm_fit"]
        baseline = ordinary["robots"][name]["fit"]["physical_arm_fit"]
        if (fit["holdout_rmse_nm"] >= .85 * baseline["holdout_rmse_nm"] or
                fit["reloaded_mujoco_replay"]["complete_windows"] != 16):
            raise ValueError(
                f"{name}: robust encoder fit did not recover the holdout")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(
            "usage: check_raw_preprocessing.py ARM_IDENTIFICATION_EXECUTABLE RAW_REPORT "
            "[SPIKED_REPORT ROBUST_REFIT_REPORT]")
    check(sys.argv[1])
    check_report(sys.argv[2])
    if len(sys.argv) == 5:
        check_encoder_outliers(*sys.argv[3:])
