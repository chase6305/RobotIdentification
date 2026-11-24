"""Check the clean reference and summarize paired noise-seed runs."""

import json
from pathlib import Path
from statistics import median
import sys


def check_arm(name, dof, expected_rank, experiment, clean):
    fit, design = experiment["fit"], experiment["design"]
    rank = fit["base_parameter_count"]
    if (len(experiment["joint_names"]) != dof or
            fit["full_parameter_count"] != 10 * dof or rank != expected_rank or
            not fit["oracle_holdout_max_absolute_error_nm"] <= 1e-7 or
            not fit["reduced_oracle_max_difference_nm"] <= 1e-8 or
            not fit["nominal_holdout_rmse_nm"] > 0.1 or
            not design["optimized"]["mean_log_information"]
            > design["baseline"]["mean_log_information"] or
            not all(0 < peak < 1
                    for peak in experiment["peak_effort_fraction"].values())):
        raise ValueError(
            f"{name}: joint layout, oracle, excitation or effort check failed")
    for split, acquisition in experiment.get("acquisition", {}).items():
        if (acquisition["integration_steps"] <= experiment["samples"][split] or
                acquisition["saturation_steps"] != 0 or
                not acquisition["position_tracking_rmse_rad"] < 0.02):
            raise ValueError(
                f"{name}/{split}: invalid forward simulation or tracking accuracy"
            )
    for trajectory in ("baseline", "optimized"):
        if design[trajectory]["rank"] != rank:
            raise ValueError(f"{name}/{trajectory}: design rank mismatch")
        for solver in ("ols", "huber"):
            estimate = fit[trajectory][solver]
            if estimate["rank"] != rank or len(
                    estimate["holdout_joint_rmse_nm"]) != dof:
                raise ValueError(
                    f"{name}/{trajectory}/{solver}: fit rank or joint metrics mismatch"
                )
            if clean and (not estimate["holdout_rmse_nm"] < 1e-7 or
                          not estimate["relative_base_error"] < 1e-6):
                raise ValueError(
                    f"{name}/{trajectory}/{solver}: clean torque or base parameter recovery failed"
                )
            if clean:
                forward = estimate["forward_validation"]
                if (not forward["mass_check_passed"] or
                        forward["mass_failures"] or
                        not forward["holdout_acceleration_rmse_rad_s2"] < 1e-8):
                    raise ValueError(
                        f"{name}/{trajectory}/{solver}: clean forward prediction failed"
                    )
    if not clean and not experiment["outlier_count_per_training_trajectory"]:
        raise ValueError(f"{name}: missing training outliers")


def check(clean_report, noisy_reports):
    if len({report["seed"] for report in noisy_reports}) != len(noisy_reports):
        raise ValueError("noise seeds must be distinct")
    for name, dof, expected_rank in (("pm01", 5, 33), ("openarm", 7, 43)):
        clean = clean_report["robots"][name]
        check_arm(name, dof, expected_rank, clean, clean=True)
        noisy_arms = [report["robots"][name] for report in noisy_reports]
        for experiment in noisy_arms:
            check_arm(name, dof, expected_rank, experiment, clean=False)
        medians = {
            (trajectory, solver):
                median(experiment["fit"][trajectory][solver]["holdout_rmse_nm"]
                       for experiment in noisy_arms)
            for trajectory in ("baseline", "optimized")
            for solver in ("ols", "huber")
        }
        nominal = clean["fit"]["nominal_holdout_rmse_nm"]
        if (not medians["optimized", "huber"] < medians["optimized", "ols"] or
                not medians["optimized", "huber"] < medians["baseline", "huber"]
                or not medians["optimized", "huber"] < 0.1 * nominal):
            raise ValueError(
                f"{name}: optimized Huber did not improve on the comparison fits"
            )
        print(
            f"{name}: rank {clean['fit']['base_parameter_count']}, nominal {nominal:.6g} N m"
        )
        for (trajectory, solver), value in medians.items():
            print(
                f"  {trajectory}/{solver}: median holdout RMSE {value:.6g} N m")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(
            "usage: check_experiment.py CLEAN_REPORT NOISY_REPORT [NOISY_REPORT ...]"
        )
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in sys.argv[1:]
    ]
    check(reports[0], reports[1:])
