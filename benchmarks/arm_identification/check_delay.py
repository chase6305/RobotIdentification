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


if __name__ == "__main__":
    check(*map(Path, sys.argv[1:]))
