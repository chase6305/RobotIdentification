"""Exercise the installed recorder with a synthetic standard ROS JointState."""

import csv
import json
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix
from sensor_msgs.msg import JointState

from robot_identification_tools.dataset_inspect import inspect_dataset_with_digest


def main():
    tag = uuid.uuid4().hex[:12]
    topic = f"/identification_smoke_{tag}/joint_states"
    executable = (Path(get_package_prefix("robot_identification_ros2")) /
                  "lib" / "robot_identification_ros2" / "joint_state_recorder")
    with tempfile.TemporaryDirectory(
            prefix="identification-smoke-") as directory:
        root = Path(directory)
        output = root / "session.csv"
        with (root / "recorder.log").open("w+") as log:
            process = subprocess.Popen([
                str(executable), "--ros-args", "-p", f"input_topic:={topic}",
                "-p", f"output_path:={output}", "-p", "flush_interval:=1", "-p",
                "joint_names:=[joint_a,joint_b]"
            ],
                                       stdout=log,
                                       stderr=subprocess.STDOUT)
            node = None
            try:
                rclpy.init()
                node = rclpy.create_node(f"identification_publisher_{tag}")
                publisher = node.create_publisher(JointState, topic, 10)
                deadline = time.monotonic() + 15
                while publisher.get_subscription_count() == 0:
                    if process.poll() is not None:
                        raise RuntimeError("installed recorder exited early")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("recorder discovery timed out")
                    rclpy.spin_once(node, timeout_sec=0.05)
                for index in range(20):
                    sample = JointState()
                    if index % 2:
                        sample.name = ["joint_a", "joint_b"]
                        sample.position = [1.0, 2.0]
                        sample.effort = [3.0, 4.0]
                    else:
                        sample.name = ["joint_b", "joint_a"]
                        sample.position = [2.0, 1.0]
                        sample.effort = [4.0, 3.0]
                    publisher.publish(sample)
                    rclpy.spin_once(node, timeout_sec=0.05)
                rclpy.spin_once(node, timeout_sec=0.2)
                process.send_signal(signal.SIGINT)
                if process.wait(timeout=10) != 0:
                    raise RuntimeError("recorder did not shut down cleanly")
            except Exception as error:
                log.flush()
                log.seek(0)
                raise RuntimeError(f"{error}\n{log.read()}") from error
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if node is not None:
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
        summary, digest = inspect_dataset_with_digest(output)
        if metadata["sample_count"] < 5 or metadata["rejected_count"] != 0:
            raise AssertionError(f"unexpected recording: {metadata}")
        if metadata["dataset_sha256"] != digest:
            raise AssertionError("recording digest does not match its metadata")
        if summary["sample_count"] != metadata["sample_count"]:
            raise AssertionError("recording counts differ")
        with output.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if rows[0] != [
                "time_s", "q/joint_a", "q/joint_b", "tau/joint_a", "tau/joint_b"
        ]:
            raise AssertionError(f"unexpected CSV header: {rows[0]}")
        if any([float(value)
                for value in row[1:]] != [1, 2, 3, 4]
               for row in rows[1:]):
            raise AssertionError("named-joint samples were not reordered")
        print(
            f"Installed ROS recorder passed: {metadata['sample_count']} samples."
        )


if __name__ == "__main__":
    main()
