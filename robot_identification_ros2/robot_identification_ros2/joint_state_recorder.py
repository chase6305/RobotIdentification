#!/usr/bin/env python3
"""Record finite, ordered JointState position/effort samples using monotonic time."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from robot_identification_ros2.raw_joint_state_session import RawJointStateSession


class JointStateRecorder(Node):
    """Non-real-time recorder that never changes robot control state."""

    def __init__(self) -> None:
        super().__init__("robot_identification_recorder")
        self._session = None
        self._closed = False
        try:
            self._configure()
        except BaseException:
            try:
                if self._session is not None:
                    self._session.close()
            finally:
                self.destroy_node()
            raise

    def _configure(self) -> None:
        self.declare_parameter("input_topic", "/feedback/joint")
        self.declare_parameter("output_path", "")
        self.declare_parameter("joint_names", [])
        self.declare_parameter("flush_interval", 100)
        self.declare_parameter("session_id", "")
        self.declare_parameter("model_digest", "")
        self.declare_parameter("torque_source", "sensor_msgs/JointState.effort")

        input_topic = self.get_parameter("input_topic").value
        self._session = RawJointStateSession(
            self.get_parameter("output_path").value,
            input_topic=input_topic,
            joint_names=self.get_parameter("joint_names").value,
            flush_interval=self.get_parameter("flush_interval").value,
            session_id=self.get_parameter("session_id").value,
            model_digest=self.get_parameter("model_digest").value,
            torque_source=self.get_parameter("torque_source").value)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._subscription = self.create_subscription(JointState, input_topic,
                                                      self._on_sample, qos)
        self.get_logger().info(
            f"recording {input_topic} to {self._session.partial_path}")

    def _on_sample(self, message: JointState) -> None:
        if self._closed:
            return
        reason = self._session.append(message.name, message.position,
                                      message.effort)
        if reason is not None:
            count = self._session.rejected_count
            if count == 1 or count % 100 == 0:
                self.get_logger().warning(
                    f"rejected JointState sample: {reason}; rejected={count}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session.close():
            self.get_logger().info(
                f"finalized {self._session.sample_count} samples at {self._session.output_path}"
            )
        else:
            self.get_logger().error(
                f"recording not finalized; partial file retained: {self._session.partial_path}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    recorder = None
    try:
        recorder = JointStateRecorder()
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if recorder is not None:
                try:
                    recorder.close()
                finally:
                    recorder.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
