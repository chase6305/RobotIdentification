"""Check the clean and noisy reference runs on the two pinned assets."""

import json
from pathlib import Path
import sys


def check(clean, noisy):
    for name, dof in (("pm01", 5), ("openarm", 7)):
        exact, contaminated = clean["robots"][name], noisy["robots"][name]
        for experiment in (exact, contaminated):
            fit, design = experiment["fit"], experiment["design"]
            rank = fit["base_parameter_count"]
            if (len(experiment["joint_names"]) != dof or
                    fit["full_parameter_count"] != 10 * dof or
                    not 0 < rank < fit["full_parameter_count"] or
                    not fit["oracle_holdout_max_absolute_error_nm"] < 1e-8 or
                    not fit["nominal_holdout_rmse_nm"] > 0.1 or
                    not design["optimized"]["mean_log_information"]
                    > design["baseline"]["mean_log_information"] or
                    not all(0 < peak < 1 for peak in
                            experiment["peak_effort_fraction"].values())):
                raise ValueError(
                    f"{name}: joint layout, oracle, excitation or effort check failed"
                )
            for trajectory in ("baseline", "optimized"):
                if design[trajectory]["rank"] != rank:
                    raise ValueError(
                        f"{name}/{trajectory}: design rank mismatch")
                for solver in ("ols", "huber"):
                    estimate = fit[trajectory][solver]
                    if estimate["rank"] != rank or len(
                            estimate["holdout_joint_rmse_nm"]) != dof:
                        raise ValueError(
                            f"{name}/{trajectory}/{solver}: fit rank or joint metrics mismatch"
                        )
                    if experiment is exact and not estimate[
                            "holdout_rmse_nm"] < 1e-8:
                        raise ValueError(
                            f"{name}/{trajectory}/{solver}: clean torque recovery failed"
                        )
        fit = contaminated["fit"]
        optimized = fit["optimized"]
        if (not contaminated["outlier_count_per_training_trajectory"] or
                not optimized["huber"]["holdout_rmse_nm"]
                < optimized["ols"]["holdout_rmse_nm"] or
                not optimized["huber"]["holdout_rmse_nm"]
                < 0.1 * fit["nominal_holdout_rmse_nm"]):
            raise ValueError(
                f"{name}: optimized Huber did not reject outliers or improve on nominal"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_experiment.py CLEAN_REPORT NOISY_REPORT")
    check(*(json.loads(Path(path).read_text(encoding="utf-8"))
            for path in sys.argv[1:]))
