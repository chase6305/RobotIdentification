#!/usr/bin/env python3
"""Record finite, ordered JointState position/effort samples using monotonic time."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import TextIO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class JointStateRecorder(Node):
    """Non-real-time recorder that never changes robot control state."""

    def __init__(self) -> None:
        super().__init__("robot_identification_recorder")
        self.declare_parameter("input_topic", "/feedback/joint")
        self.declare_parameter("output_path", "")
        self.declare_parameter("joint_names", [])
        self.declare_parameter("flush_interval", 100)
        self.declare_parameter("session_id", "")
        self.declare_parameter("model_digest", "")
        self.declare_parameter("torque_source", "sensor_msgs/JointState.effort")

        self._input_topic = self.get_parameter("input_topic").value
        output_value = self.get_parameter("output_path").value
        configured_names = list(self.get_parameter("joint_names").value)
        self._flush_interval = self.get_parameter("flush_interval").value
        self._session_id = self.get_parameter("session_id").value
        self._model_digest = self.get_parameter("model_digest").value
        self._torque_source = self.get_parameter("torque_source").value
        if not isinstance(self._input_topic, str) or not self._input_topic:
            raise ValueError("input_topic must be a non-empty string")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("output_path must be explicitly configured")
        if (not isinstance(self._flush_interval, int) or
                self._flush_interval <= 0):
            raise ValueError("flush_interval must be a positive integer")
        if not isinstance(self._session_id, str) or not isinstance(
                self._model_digest, str):
            raise ValueError("session_id and model_digest must be strings")
        if not isinstance(self._torque_source,
                          str) or not self._torque_source:
            raise ValueError("torque_source must be a non-empty string")
        if (any(not isinstance(name, str) or not name or "," in name or
                "\n" in name or "\r" in name for name in configured_names) or
                len(set(configured_names)) != len(configured_names)):
            raise ValueError(
                "joint_names must be unique non-empty CSV-safe names")

        self._output_path = Path(output_value).expanduser().resolve()
        self._partial_path = Path(str(self._output_path) + ".partial")
        self._metadata_path = Path(str(self._output_path) + ".metadata.json")
        self._metadata_partial_path = Path(
            str(self._metadata_path) + ".partial")
        if (self._output_path.exists() or self._partial_path.exists() or
                self._metadata_path.exists() or
                self._metadata_partial_path.exists()):
            raise FileExistsError(
                "refusing to overwrite an existing dataset or partial artifact")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._session_id:
            self._session_id = self._output_path.stem
        self._stream: TextIO = self._partial_path.open("x", encoding="utf-8")
        self._joint_names = configured_names
        self._sample_count = 0
        self._rejected_count = 0
        self._start_ns = time.monotonic_ns()
        self._last_sample_ns = self._start_ns - 1
        self._closed = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._subscription = self.create_subscription(JointState,
                                                      self._input_topic,
                                                      self._on_sample, qos)
        self.get_logger().info(
            f"recording {self._input_topic} to {self._partial_path}")

    def _reject(self, reason: str) -> None:
        self._rejected_count += 1
        if self._rejected_count == 1 or self._rejected_count % 100 == 0:
            self.get_logger().warning(
                f"rejected JointState sample: {reason}; rejected={self._rejected_count}"
            )

    def _on_sample(self, message: JointState) -> None:
        if self._closed:
            return
        names = list(message.name)
        if (not names or len(set(names)) != len(names) or
                len(message.position) != len(names) or
                len(message.effort) != len(names)):
            self._reject("name/position/effort dimensions are inconsistent")
            return
        if not self._joint_names:
            if any(not name or "," in name or "\n" in name or "\r" in name
                   for name in names):
                self._reject("joint names are empty or not CSV-safe")
                return
            self._joint_names = names
        index = {name: offset for offset, name in enumerate(names)}
        if set(index) != set(self._joint_names):
            self._reject("joint set changed during the session")
            return
        position = [message.position[index[name]] for name in self._joint_names]
        torque = [message.effort[index[name]] for name in self._joint_names]
        if not all(math.isfinite(value) for value in position + torque):
            self._reject("position or effort contains a non-finite value")
            return
        if self._sample_count == 0:
            header = (["time_s"] + [f"q/{name}" for name in self._joint_names] +
                      [f"tau/{name}" for name in self._joint_names])
            self._stream.write(",".join(header) + "\n")
        sample_ns = time.monotonic_ns()
        if sample_ns <= self._last_sample_ns:
            self._reject("monotonic timestamp did not advance")
            return
        self._last_sample_ns = sample_ns
        elapsed = (sample_ns - self._start_ns) * 1e-9
        fields = [format(elapsed, ".17g")]
        fields.extend(format(value, ".17g") for value in position)
        fields.extend(format(value, ".17g") for value in torque)
        self._stream.write(",".join(fields) + "\n")
        self._sample_count += 1
        if self._sample_count % self._flush_interval == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        if self._sample_count == 0:
            self.get_logger().error(
                f"no valid samples recorded; partial file retained: {self._partial_path}"
            )
            return
        os.replace(self._partial_path, self._output_path)
        digest = hashlib.sha256()
        with self._output_path.open("rb") as dataset_stream:
            for chunk in iter(lambda: dataset_stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = {
            "schema_version": 1,
            "kind": "robot_identification_raw_joint_state",
            "session_id": self._session_id,
            "input_topic": self._input_topic,
            "time_source": "process_monotonic",
            "joint_names": self._joint_names,
            "sample_count": self._sample_count,
            "rejected_count": self._rejected_count,
            "torque_source": self._torque_source,
            "dataset_sha256": digest.hexdigest(),
        }
        if self._model_digest:
            metadata["model_digest"] = self._model_digest
        with self._metadata_partial_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self._metadata_partial_path, self._metadata_path)
        self.get_logger().info(
            f"finalized {self._sample_count} samples at {self._output_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    recorder = None
    try:
        recorder = JointStateRecorder()
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
            recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
