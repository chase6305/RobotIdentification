"""Verify delay compensation on the same simulated torque and encoder capture."""

import json
from pathlib import Path
import sys


def check(capture, aligned):
    original = json.loads((capture / "report.json").read_text(encoding="utf-8"))
    corrected = json.loads(
        (aligned / "report.json").read_text(encoding="utf-8"))
    delay = original["torque_delay_ms"]
    if not delay or original["align_torque_ms"] or corrected[
            "align_torque_ms"] != delay:
        raise ValueError(
            "expected one delayed capture and a matching alignment")
    for name, result in corrected["robots"].items():
        before, after = capture / name, aligned / name
        for filename in ("train_baseline_raw.csv", "train_optimized_raw.csv",
                         "holdout.csv", "holdout_dense.csv"):
            if (before / filename).read_bytes() != (after /
                                                    filename).read_bytes():
                raise ValueError(
                    f"{name}: alignment changed a recorded dataset")
        observed = result["acquisition"]
        if (observed["optimized"]["torque_channel_delay_s"] != delay / 1000 or
                observed["holdout"]["torque_channel_delay_s"] != 0.0):
            raise ValueError(f"{name}: unexpected delay in the acquisition")
        base = original["robots"][name]["fit"]["physical_arm_fit"]
        fit = result["fit"]["physical_arm_fit"]
        if (fit["holdout_rmse_nm"] >= 0.4 * base["holdout_rmse_nm"] or
                fit["reloaded_mujoco_replay"]["complete_windows"] != 16):
            raise ValueError(f"{name}: alignment did not recover the holdout")


def check_search(capture, searched, undelayed):
    source = json.loads((capture / "report.json").read_text(encoding="utf-8"))
    estimate = json.loads(
        (searched / "report.json").read_text(encoding="utf-8"))
    synchronous = json.loads(
        (undelayed / "report.json").read_text(encoding="utf-8"))
    for name, result in estimate["robots"].items():
        timing = result["torque_alignment"]
        error = abs(timing["delay_ms"] - source["torque_delay_ms"])
        rmse = result["fit"]["physical_arm_fit"]["holdout_rmse_nm"]
        initial = source["robots"][name]["fit"]["physical_arm_fit"][
            "holdout_rmse_nm"]
        if error > 2.1 or timing["tested_delays"] > 17 or rmse >= 0.4 * initial:
            raise ValueError(
                f"{name}: search did not recover the delayed torque channel")
        if synchronous["robots"][name]["torque_alignment"]["delay_ms"] != 0.0:
            raise ValueError(
                f"{name}: search added a shift to synchronized data")


if __name__ == "__main__":
    args = tuple(map(Path, sys.argv[1:]))
    if len(args) not in (2, 4):
        sys.exit("usage: check_delay.py CAPTURE ALIGNED [SEARCHED SYNCHRONOUS]")
    check(*args[:2])
    if len(args) == 4:
        check_search(args[0], args[2], args[3])
