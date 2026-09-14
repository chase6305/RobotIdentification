"""Exercise the ROS adapter lifecycle with stubs; no ROS graph is started."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robot_identification_ros2 import raw_joint_state_session as recording


class JointStateRecorderTest(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "session.csv"
        self.options = {"output_path": str(self.path)}
        self.nodes = []
        self.subscribe = mock.Mock()
        options, nodes, subscribe = self.options, self.nodes, self.subscribe

        class StubNode:

            def __init__(self, name):
                self.parameters = {}
                self.logger = mock.Mock()
                self.destroy = mock.Mock()
                nodes.append(self)

            def declare_parameter(self, name, default):
                self.parameters[name] = options.get(name, default)

            def get_parameter(self, name):
                return types.SimpleNamespace(value=self.parameters[name])

            def create_subscription(self, *args):
                return subscribe(*args)

            def get_logger(self):
                return self.logger

            def destroy_node(self):
                return self.destroy()

        self.ros = types.ModuleType("rclpy")
        self.ros.init = mock.Mock()
        self.ros.spin = mock.Mock()
        self.ros.shutdown = mock.Mock()
        node_module = types.ModuleType("rclpy.node")
        node_module.Node = StubNode
        qos_module = types.ModuleType("rclpy.qos")
        qos_module.QoSProfile = mock.Mock()
        qos_module.ReliabilityPolicy = types.SimpleNamespace(
            RELIABLE="reliable")
        message_module = types.ModuleType("sensor_msgs.msg")
        message_module.JointState = types.SimpleNamespace
        source = Path(recording.__file__).with_name("joint_state_recorder.py")
        spec = importlib.util.spec_from_file_location(
            "_recorder_lifecycle_test", source)
        self.recorder = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
                sys.modules, {
                    "rclpy": self.ros,
                    "rclpy.node": node_module,
                    "rclpy.qos": qos_module,
                    "sensor_msgs": types.ModuleType("sensor_msgs"),
                    "sensor_msgs.msg": message_module
                }):
            spec.loader.exec_module(self.recorder)

    @staticmethod
    def sample(names=("a",), position=(1.0,), effort=(2.0,)):
        return types.SimpleNamespace(name=names,
                                     position=position,
                                     effort=effort)

    def assert_released(self):
        self.nodes[0].destroy.assert_called_once_with()
        self.ros.shutdown.assert_called_once_with()

    def test_callback_validation_and_normal_interrupt_finalize(self):
        self.options.update(input_topic="/robot/joints",
                            joint_names=["b", "a"],
                            session_id="configured-session",
                            flush_interval=1)

        def spin(node):
            for _ in range(3):
                node._on_sample(self.sample(["a", "b"], [1, 2], [3, 4]))
            for _ in range(101):
                node._on_sample(self.sample())
            raise KeyboardInterrupt()

        self.ros.spin.side_effect = spin
        self.recorder.main(args=["--ros-args"])
        self.ros.init.assert_called_once_with(args=["--ros-args"])
        self.assert_released()
        self.assertEqual(self.subscribe.call_args.args[1], "/robot/joints")
        self.nodes[0].logger.warning.assert_has_calls([
            mock.call(
                "rejected JointState sample: joint set changed during the session; rejected=1"
            ),
            mock.call(
                "rejected JointState sample: joint set changed during the session; rejected=100"
            )
        ])
        metadata = json.loads(
            Path(str(self.path) + ".metadata.json").read_text())
        self.assertEqual(metadata["sample_count"], 3)
        self.assertEqual(metadata["rejected_count"], 101)
        self.assertEqual(metadata["session_id"], "configured-session")
        self.assertEqual(metadata["joint_names"], ["b", "a"])
        self.nodes[0].close()
        self.nodes[0]._on_sample(self.sample())
        self.assertEqual(self.nodes[0]._session.sample_count, 3)

    def test_configuration_failure_destroys_node_and_shuts_down(self):
        self.options["flush_interval"] = 0
        with self.assertRaises(ValueError):
            self.recorder.main()
        self.assert_released()
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_subscription_failure_closes_partial_and_destroys_node(self):
        self.subscribe.side_effect = RuntimeError("subscription failed")
        with self.assertRaisesRegex(RuntimeError, "subscription failed"):
            self.recorder.main()
        self.assert_released()
        session = self.nodes[0]._session
        self.assertTrue(session._stream.closed)
        self.assertTrue(session.partial_path.exists())
        self.assertFalse(self.path.exists())

    def test_final_sync_failure_still_destroys_node_and_shuts_down(self):
        self.ros.spin.side_effect = lambda node: node._on_sample(self.sample())
        with mock.patch.object(recording.os,
                               "fsync",
                               side_effect=OSError("sync failed")):
            with self.assertRaisesRegex(OSError, "sync failed"):
                self.recorder.main()
        self.assert_released()
        self.assertTrue(self.nodes[0]._session._stream.closed)
        self.assertFalse(self.path.exists())

    def test_callback_write_failure_preserves_partial_and_releases_node(self):

        def spin(node):
            node._on_sample(self.sample())
            stream = node._session._stream
            node._session._stream = mock.Mock(wraps=stream)
            node._session._stream.write.side_effect = OSError("disk full")
            node._on_sample(self.sample())

        self.ros.spin.side_effect = spin
        with self.assertRaisesRegex(OSError, "disk full"):
            self.recorder.main()
        self.assert_released()
        self.assertFalse(self.path.exists())
        self.assertTrue(self.nodes[0]._session.partial_path.exists())

    def test_node_destruction_failure_still_shuts_down(self):

        def spin(node):
            node.destroy.side_effect = RuntimeError("destroy failed")

        self.ros.spin.side_effect = spin
        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            self.recorder.main()
        self.assert_released()
        self.assertTrue(self.nodes[0]._session._stream.closed)


if __name__ == "__main__":
    unittest.main()
