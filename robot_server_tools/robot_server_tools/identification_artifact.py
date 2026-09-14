#!/usr/bin/env python3
"""Create immutable candidate, review, and rollback identification artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from robot_server_tools.dataset_inspect import inspect_dataset_with_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    return _read_object_with_digest(path)[0]


def _unique_object(pairs) -> dict[str, object]:
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field: {name}")
        value[name] = item
    return value


def _invalid_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"JSON number exceeds the finite floating-point range: {value}")
    if result == 0.0 and re.search(r"[1-9]", value.lower().partition("e")[0]):
        raise ValueError(f"nonzero JSON number underflows to zero: {value}")
    return result


def _read_object_with_digest(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"),
                           object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant,
                           parse_float=_json_float)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _atomic_create(path: Path, value: dict[str, object]) -> None:
    # Resolve the directory without following a possibly dangling final link.
    path = path.parent.resolve() / path.name
    temporary = Path(str(path) + ".partial")
    if (path.exists() or path.is_symlink() or temporary.exists() or
            temporary.is_symlink()):
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking fails atomically if another publisher created the target
        # after the preflight check. Rename/replace would overwrite it.
        os.link(temporary, path)
    finally:
        # Only this invocation's exclusively created temporary is removed.
        temporary.unlink()
    directory_fd = os.open(path.parent,
                           os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _finite_number(value: object,
                   name: str,
                   minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    # Integer JSON counts must never pass through a lossy float conversion.
    return value


def _fields(value: object, required: set[str], optional: set[str],
            name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(
            f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _names(value: object, name: str) -> None:
    if (not isinstance(value, list) or not value or
            any(not isinstance(item, str) or not item for item in value) or
            len(set(value)) != len(value)):
        raise ValueError(
            f"{name} must be a non-empty list of unique non-empty strings")


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _digest(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}",
                                                  value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _version(value: object, name: str) -> None:
    if type(value) not in (int, float) or value != 1:
        raise ValueError(f"{name} schema version is unsupported")


def _normalize_diagnostics(value: object) -> dict[str, object]:
    _fields(
        value, {
            "row_count", "rank", "condition_number", "train_rmse",
            "validation_rmse", "mass_matrix_passed"
        }, {"max_absolute_error"}, "diagnostics")
    result = {
        name: _positive_integer(value[name], name)
        for name in ("row_count", "rank")
    }
    for name, minimum in (("condition_number", 1.0), ("train_rmse", 0.0),
                          ("validation_rmse", 0.0)):
        result[name] = _finite_number(value[name], name, minimum)
    if not isinstance(value["mass_matrix_passed"], bool):
        raise ValueError("mass_matrix_passed must be boolean")
    result["mass_matrix_passed"] = value["mass_matrix_passed"]
    if "max_absolute_error" in value:
        result["max_absolute_error"] = _finite_number(
            value["max_absolute_error"], "max_absolute_error", 0.0)
    return result


def _validate_candidate(candidate: dict[str, object]) -> dict[str, object]:
    _fields(
        candidate, {
            "schema_version", "kind", "status", "dataset_sha256",
            "model_digest", "joint_names", "parameter_names", "parameters",
            "diagnostics"
        }, {"software_revision"}, "candidate")
    _version(candidate["schema_version"], "candidate")
    _digest(candidate["dataset_sha256"], "dataset_sha256")
    _nonempty_string(candidate["model_digest"], "model_digest")
    _names(candidate["joint_names"], "joint_names")
    _names(candidate["parameter_names"], "parameter_names")
    names = candidate["parameter_names"]
    parameters = candidate["parameters"]
    if len(names) > 100000:
        raise ValueError("candidate must have at most 100000 parameters")
    if not isinstance(parameters, list) or len(parameters) != len(names):
        raise ValueError(
            "candidate parameter names and values must have matching lengths")
    for name, value in zip(names, parameters):
        _finite_number(value, f"parameter {name}")
    if "software_revision" in candidate and not isinstance(
            candidate["software_revision"], str):
        raise ValueError("software_revision must be a string")
    return _normalize_diagnostics(candidate["diagnostics"])


def _validate_rollback(rollback: dict[str, object]) -> None:
    _fields(
        rollback, {
            "schema_version", "kind", "configuration_path",
            "configuration_sha256", "profile", "created_at"
        }, set(), "rollback")
    _version(rollback["schema_version"], "rollback")
    if rollback["kind"] != "robot_identification_rollback":
        raise ValueError("rollback input is not a rollback manifest")
    _nonempty_string(rollback["configuration_path"], "configuration_path")
    _digest(rollback["configuration_sha256"], "configuration_sha256")
    _nonempty_string(rollback["profile"], "profile")
    timestamp = rollback["created_at"]
    if (not isinstance(timestamp, str) or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
            timestamp) is None):
        raise ValueError("created_at must be a date-time with a timezone")
    try:
        datetime.fromisoformat(timestamp[:-1] +
                               "+00:00" if timestamp[-1] in "Zz" else timestamp)
    except ValueError as error:
        raise ValueError("created_at is not a valid date-time") from error


def create_candidate(dataset: Path,
                     model_digest: str,
                     joint_names: list[str],
                     parameters_file: Path,
                     diagnostics_file: Path,
                     output: Path,
                     software_revision: str = "") -> None:
    parameters = _read_object(parameters_file)
    diagnostics = _read_object(diagnostics_file)
    if not dataset.is_file() or not isinstance(model_digest,
                                               str) or not model_digest:
        raise ValueError("dataset and model_digest are required")
    _names(joint_names, "joint names")
    if not parameters or any(not name for name in parameters):
        raise ValueError("parameters must be a non-empty named object")
    if len(parameters) > 100000:
        raise ValueError("candidate must have at most 100000 parameters")
    parameter_values = [
        _finite_number(value, f"parameter {name}")
        for name, value in parameters.items()
    ]
    normalized_diagnostics = _normalize_diagnostics(diagnostics)
    if not isinstance(software_revision, str):
        raise ValueError("software_revision must be a string")
    dataset_summary, dataset_digest = inspect_dataset_with_digest(dataset)
    if dataset_summary["joint_names"] != joint_names:
        raise ValueError(
            "candidate joint names do not match the dataset column order")
    candidate: dict[str, object] = {
        "schema_version": 1,
        "kind": "robot_identification_result",
        "status": "candidate",
        "dataset_sha256": dataset_digest,
        "model_digest": model_digest,
        "joint_names": joint_names,
        "parameter_names": list(parameters),
        "parameters": parameter_values,
        "diagnostics": normalized_diagnostics,
    }
    if software_revision:
        candidate["software_revision"] = software_revision
    _atomic_create(output, candidate)


def create_review(candidate_path: Path,
                  decision: str,
                  reviewer: str,
                  reason: str,
                  output: Path,
                  rollback_path: Path | None = None) -> None:
    candidate, candidate_digest = _read_object_with_digest(candidate_path)
    if candidate.get("kind") != "robot_identification_result" or candidate.get(
            "status") != "candidate":
        raise ValueError(
            "review input is not a candidate identification result")
    if (decision not in {"approved", "rejected"} or
            not isinstance(reviewer, str) or not reviewer or
            not isinstance(reason, str) or not reason):
        raise ValueError("decision, reviewer, and reason are required")
    rollback_digest = ""
    if decision == "approved":
        diagnostics = _validate_candidate(candidate)
        rank = diagnostics["rank"]
        if (rank != len(candidate["parameter_names"]) or
                diagnostics["row_count"] < rank or
                diagnostics["mass_matrix_passed"] is not True):
            raise ValueError(
                "approval requires full rank and passed mass-matrix validation")
        if rollback_path is None:
            raise ValueError("approval requires a rollback manifest")
        rollback, rollback_digest = _read_object_with_digest(rollback_path)
        _validate_rollback(rollback)
        configuration = Path(rollback["configuration_path"])
        if (not configuration.is_absolute() or not configuration.is_file() or
                _sha256(configuration) != rollback["configuration_sha256"]):
            raise ValueError(
                "rollback configuration no longer matches its manifest")
    review = {
        "schema_version": 1,
        "kind": "robot_identification_review",
        "decision": decision,
        "candidate_sha256": candidate_digest,
        "reviewer": reviewer,
        "reason": reason,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if rollback_digest:
        review["rollback_sha256"] = rollback_digest
    _atomic_create(output, review)


def create_rollback(configuration: Path, profile: str, output: Path) -> None:
    if not configuration.is_file() or not isinstance(profile,
                                                     str) or not profile:
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
    review.add_argument("--decision",
                        choices=["approved", "rejected"],
                        required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--rollback", type=Path)
    review.add_argument("--output", type=Path, required=True)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--configuration", type=Path, required=True)
    rollback.add_argument("--profile", required=True)
    rollback.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "candidate":
            create_candidate(args.dataset, args.model_digest, args.joints,
                             args.parameters, args.diagnostics, args.output,
                             args.software_revision)
        elif args.command == "review":
            create_review(args.candidate, args.decision, args.reviewer,
                          args.reason, args.output, args.rollback)
        else:
            create_rollback(args.configuration, args.profile, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
