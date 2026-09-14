"""ROS-independent validation and durable storage for a raw JointState session."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Sequence


def _valid_names(names: Sequence[str]) -> bool:
    return (all(
        isinstance(name, str) and name and not any(character in name
                                                   for character in ",\r\n")
        for name in names) and len(set(names)) == len(names))


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RawJointStateSession:
    """Single-writer session; failed recordings retain their partial artifacts."""

    def __init__(self,
                 output_path: str,
                 *,
                 input_topic: str = "/feedback/joint",
                 joint_names: Sequence[str] = (),
                 flush_interval: int = 100,
                 session_id: str = "",
                 model_digest: str = "",
                 torque_source: str = "sensor_msgs/JointState.effort") -> None:
        if not isinstance(input_topic, str) or not input_topic:
            raise ValueError("input_topic must be a non-empty string")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("output_path must be explicitly configured")
        if type(flush_interval) is not int or flush_interval <= 0:
            raise ValueError("flush_interval must be a positive integer")
        if not isinstance(session_id, str) or not isinstance(model_digest, str):
            raise ValueError("session_id and model_digest must be strings")
        if not isinstance(torque_source, str) or not torque_source:
            raise ValueError("torque_source must be a non-empty string")
        if isinstance(joint_names,
                      (str, bytes)) or not _valid_names(joint_names):
            raise ValueError(
                "joint_names must be unique non-empty CSV-safe names")

        requested = Path(output_path).expanduser()
        # Resolve directories only: even a dangling final symlink is a conflict.
        self.output_path = requested.parent.resolve() / requested.name
        self.partial_path = Path(str(self.output_path) + ".partial")
        self._metadata_path = Path(str(self.output_path) + ".metadata.json")
        self._metadata_partial_path = Path(
            str(self._metadata_path) + ".partial")
        if any(
                _exists(path)
                for path in (self.output_path, self.partial_path,
                             self._metadata_path, self._metadata_partial_path)):
            raise FileExistsError(
                "refusing to overwrite an existing dataset or partial artifact")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._input_topic = input_topic
        self._flush_interval = flush_interval
        self._session_id = session_id or self.output_path.stem
        self._model_digest = model_digest
        self._torque_source = torque_source
        self._joint_names = tuple(joint_names)
        self._joint_name_set = frozenset(joint_names)
        self._incoming_names = ()
        self._order = ()
        self.sample_count = 0
        self.rejected_count = 0
        self._start_ns = time.monotonic_ns()
        self._last_sample_ns = self._start_ns - 1
        self._last_elapsed = -1.0
        self._digest = hashlib.sha256()
        self._failed = False
        self._closed = False
        self._finalized = False
        self._stream = self.partial_path.open("xb")

    def _reject(self, reason: str) -> str:
        self.rejected_count += 1
        return reason

    def append(self, names: Sequence[str], position: Sequence[float],
               effort: Sequence[float]) -> str | None:
        """Write a valid sample, or return its rejection reason without writing."""
        if self._closed or self._failed:
            raise RuntimeError("recording session is closed or has failed")
        if isinstance(names, (str, bytes)):
            return self._reject("joint names must be a sequence of names")
        names = tuple(names)
        if not names or len(position) != len(names) or len(effort) != len(
                names):
            return self._reject(
                "name/position/effort dimensions are inconsistent")
        if names == self._incoming_names:
            order = self._order
        else:
            if not _valid_names(names):
                return self._reject(
                    "joint names must be unique, non-empty and CSV-safe")
            index = {name: offset for offset, name in enumerate(names)}
            # Matching the already-validated joint set also validates each name.
            if self._joint_names and (len(names) != len(self._joint_names) or
                                      index.keys() != self._joint_name_set):
                return self._reject("joint set changed during the session")
            order = tuple(index[name] for name in (self._joint_names or names))
        try:
            finite = (all(math.isfinite(value) for value in position) and
                      all(math.isfinite(value) for value in effort))
        except (TypeError, ValueError, OverflowError):
            return self._reject("position or effort contains an invalid number")
        if not finite:
            return self._reject(
                "position or effort contains a non-finite value")
        sample_ns = time.monotonic_ns()
        elapsed = (sample_ns - self._start_ns) * 1e-9
        if sample_ns <= self._last_sample_ns or elapsed <= self._last_elapsed:
            return self._reject("monotonic timestamp did not advance")
        fields = [format(elapsed, ".17g")]
        fields.extend(format(position[offset], ".17g") for offset in order)
        fields.extend(format(effort[offset], ".17g") for offset in order)
        row = ",".join(fields) + "\n"
        if self.sample_count == 0:
            joints = self._joint_names or names
            header = (["time_s"] + [f"q/{name}" for name in joints] +
                      [f"tau/{name}" for name in joints])
            row = ",".join(header) + "\n" + row
        payload = row.encode("utf-8")
        try:
            if self._stream.write(payload) != len(payload):
                raise OSError("incomplete dataset write")
            if (self.sample_count + 1) % self._flush_interval == 0:
                self._stream.flush()
            self._digest.update(payload)
            self._joint_names = self._joint_names or names
            self._joint_name_set = self._joint_name_set or frozenset(names)
            self._incoming_names = names
            self._order = order
            self._last_sample_ns = sample_ns
            self._last_elapsed = elapsed
            self.sample_count += 1
        except BaseException:
            # A failed or interrupted write may have persisted only part of a row.
            self._failed = True
            raise
        return None

    def close(self) -> bool:
        """Close once; return whether both artifacts were successfully finalized."""
        if self._closed:
            return self._finalized
        self._closed = True
        try:
            if not self._failed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
        finally:
            self._stream.close()
        if self._failed or self.sample_count == 0:
            return False
        if _exists(self.output_path) or _exists(self._metadata_path):
            raise FileExistsError(
                "refusing to overwrite an existing dataset or metadata")
        metadata = {
            "schema_version": 1,
            "kind": "robot_identification_raw_joint_state",
            "session_id": self._session_id,
            "input_topic": self._input_topic,
            "time_source": "process_monotonic",
            "joint_names": self._joint_names,
            "sample_count": self.sample_count,
            "rejected_count": self.rejected_count,
            "torque_source": self._torque_source,
            "dataset_sha256": self._digest.hexdigest(),
        }
        if self._model_digest:
            metadata["model_digest"] = self._model_digest
        # Prepare and sync both files before publishing either of them.
        with self._metadata_partial_path.open("xb") as stream:
            payload = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
            if stream.write(payload) != len(payload):
                raise OSError("incomplete metadata write")
            stream.flush()
            os.fsync(stream.fileno())
        # Each link creates a complete artifact without replacing a racing writer.
        # The pair is not a transaction; retain both partials if either link fails.
        os.link(self.partial_path, self.output_path)
        os.link(self._metadata_partial_path, self._metadata_path)
        _sync_directory(self.output_path.parent)
        self.partial_path.unlink()
        self._metadata_partial_path.unlink()
        _sync_directory(self.output_path.parent)
        self._finalized = True
        return True
