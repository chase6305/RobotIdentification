#!/usr/bin/env python3
"""Create immutable candidate, review, and rollback identification artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from robot_server_tools.dataset_inspect import inspect_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_create(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    temporary = Path(str(path) + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _finite_number(value: object, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def create_candidate(dataset: Path,
                     model_digest: str,
                     joint_names: list[str],
                     parameters_file: Path,
                     diagnostics_file: Path,
                     output: Path,
                     software_revision: str = "") -> None:
    parameters = _read_object(parameters_file)
    diagnostics = _read_object(diagnostics_file)
    if not dataset.is_file() or not isinstance(model_digest, str) or not model_digest:
        raise ValueError("dataset and model_digest are required")
    if (not joint_names or any(not isinstance(name, str) or not name for name in joint_names) or
            len(set(joint_names)) != len(joint_names)):
        raise ValueError("joint names must be unique and non-empty")
    dataset_summary = inspect_dataset(dataset)
    if dataset_summary["joint_names"] != joint_names:
        raise ValueError("candidate joint names do not match the dataset column order")
    if not parameters or any(not name for name in parameters):
        raise ValueError("parameters must be a non-empty named object")
    parameter_values = [_finite_number(value, f"parameter {name}") for name, value in parameters.items()]
    required_metrics = {
        "row_count": 1,
        "rank": 1,
        "condition_number": 1.0,
        "train_rmse": 0.0,
        "validation_rmse": 0.0,
    }
    normalized_diagnostics: dict[str, object] = {}
    for name, minimum in required_metrics.items():
        value = _finite_number(diagnostics.get(name), name, minimum)
        if name in {"row_count", "rank"}:
            if not value.is_integer():
                raise ValueError(f"{name} must be an integer")
            normalized_diagnostics[name] = int(value)
        else:
            normalized_diagnostics[name] = value
    if not isinstance(diagnostics.get("mass_matrix_passed"), bool):
        raise ValueError("mass_matrix_passed must be boolean")
    normalized_diagnostics["mass_matrix_passed"] = diagnostics["mass_matrix_passed"]
    if "max_absolute_error" in diagnostics:
        normalized_diagnostics["max_absolute_error"] = _finite_number(
            diagnostics["max_absolute_error"], "max_absolute_error", 0.0)
    candidate: dict[str, object] = {
        "schema_version": 1,
        "kind": "robot_identification_result",
        "status": "candidate",
        "dataset_sha256": _sha256(dataset),
        "model_digest": model_digest,
        "joint_names": joint_names,
        "parameter_names": list(parameters),
        "parameters": parameter_values,
        "diagnostics": normalized_diagnostics,
    }
    if not isinstance(software_revision, str):
        raise ValueError("software_revision must be a string")
    if software_revision:
        candidate["software_revision"] = software_revision
    _atomic_create(output, candidate)


def create_review(candidate_path: Path,
                  decision: str,
                  reviewer: str,
                  reason: str,
                  output: Path,
                  rollback_path: Path | None = None) -> None:
    candidate = _read_object(candidate_path)
    if candidate.get("kind") != "robot_identification_result" or candidate.get("status") != "candidate":
        raise ValueError("review input is not a candidate identification result")
    if (decision not in {"approved", "rejected"} or not isinstance(reviewer, str) or not reviewer or
            not isinstance(reason, str) or not reason):
        raise ValueError("decision, reviewer, and reason are required")
    rollback_digest = ""
    if decision == "approved":
        if candidate.get("schema_version") != 1:
            raise ValueError("candidate schema version is unsupported")
        diagnostics = candidate.get("diagnostics")
        joint_names = candidate.get("joint_names")
        parameter_names = candidate.get("parameter_names")
        parameters = candidate.get("parameters")
        if (not isinstance(joint_names, list) or not joint_names or
                any(not isinstance(name, str) or not name for name in joint_names) or
                len(set(joint_names)) != len(joint_names) or
                not isinstance(parameter_names, list) or not parameter_names or
                any(not isinstance(name, str) or not name for name in parameter_names) or
                len(set(parameter_names)) != len(parameter_names) or
                not isinstance(parameters, list) or len(parameters) != len(parameter_names) or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) for value in parameters) or
                not isinstance(diagnostics, dict)):
            raise ValueError("candidate parameter contract is invalid")
        rank = diagnostics.get("rank")
        row_count = diagnostics.get("row_count")
        if (isinstance(rank, bool) or not isinstance(rank, int) or
                isinstance(row_count, bool) or not isinstance(row_count, int) or
                rank != len(parameter_names) or row_count < rank or
                diagnostics.get("mass_matrix_passed") is not True):
            raise ValueError("approval requires full rank and passed mass-matrix validation")
        for metric, minimum in (("condition_number", 1.0), ("train_rmse", 0.0),
                                ("validation_rmse", 0.0)):
            _finite_number(diagnostics.get(metric), metric, minimum)
        if rollback_path is None:
            raise ValueError("approval requires a rollback manifest")
        rollback = _read_object(rollback_path)
        if rollback.get("schema_version") != 1 or rollback.get("kind") != "robot_identification_rollback":
            raise ValueError("rollback input is not a rollback manifest")
        configuration_value = rollback.get("configuration_path")
        recorded_digest = rollback.get("configuration_sha256")
        if not isinstance(configuration_value, str) or not configuration_value:
            raise ValueError("rollback configuration path is invalid")
        configuration = Path(configuration_value)
        if (not configuration.is_absolute() or not configuration.is_file() or
                not isinstance(recorded_digest, str) or _sha256(configuration) != recorded_digest):
            raise ValueError("rollback configuration no longer matches its manifest")
        rollback_digest = _sha256(rollback_path)
    review = {
        "schema_version": 1,
        "kind": "robot_identification_review",
        "decision": decision,
        "candidate_sha256": _sha256(candidate_path),
        "reviewer": reviewer,
        "reason": reason,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if rollback_digest:
        review["rollback_sha256"] = rollback_digest
    _atomic_create(output, review)


def create_rollback(configuration: Path, profile: str, output: Path) -> None:
    if not configuration.is_file() or not isinstance(profile, str) or not profile:
        raise ValueError("existing configuration and profile are required")
    rollback = {
        "schema_version": 1,
        "kind": "robot_identification_rollback",
        "configuration_path": str(configuration.resolve()),
        "configuration_sha256": _sha256(configuration),
        "profile": profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_create(output, rollback)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--dataset", type=Path, required=True)
    candidate.add_argument("--model-digest", required=True)
    candidate.add_argument("--joints", nargs="+", required=True)
    candidate.add_argument("--parameters", type=Path, required=True)
    candidate.add_argument("--diagnostics", type=Path, required=True)
    candidate.add_argument("--software-revision", default="")
    candidate.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--candidate", type=Path, required=True)
    review.add_argument("--decision", choices=["approved", "rejected"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--rollback", type=Path)
    review.add_argument("--output", type=Path, required=True)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--configuration", type=Path, required=True)
    rollback.add_argument("--profile", required=True)
    rollback.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "candidate":
        create_candidate(args.dataset, args.model_digest, args.joints, args.parameters,
                         args.diagnostics, args.output, args.software_revision)
    elif args.command == "review":
        create_review(args.candidate, args.decision, args.reviewer, args.reason,
                      args.output, args.rollback)
    else:
        create_rollback(args.configuration, args.profile, args.output)


if __name__ == "__main__":
    main()
