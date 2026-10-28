"""Check finite numeric CSV interchange between Python and the C++ library."""

import csv
import hashlib
import itertools
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

repository = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(repository / "robot_identification_tools"),
    str(repository / "robot_identification_ros2")
]

from robot_identification_ros2 import raw_joint_state_session as recording
from robot_identification_tools.dataset_inspect import inspect_dataset_with_digest


class CsvInteropTest(unittest.TestCase):
    executable = ""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.values = [
            float.fromhex("0x1p-1074"), sys.float_info.min, sys.float_info.max,
            0.0, -0.0, 1e-310
        ]
        self.names = ["关节一", "joint_2", "j3", "j4", "j5", "j6"]

    def assert_roundtrip(self, source, kind):
        restored = self.root / f"{kind}-restored.csv"
        result = subprocess.run(
            [self.executable, kind,
             str(source), str(restored)],
            capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before, digest = inspect_dataset_with_digest(source)
        after, _ = inspect_dataset_with_digest(restored)
        self.assertEqual(before, after)
        self.assertEqual(before["kind"], kind)
        self.assertEqual(before["joint_names"], self.names)
        self.assertEqual(before["sample_count"], 3)
        with source.open(encoding="utf-8", newline="") as left, restored.open(
                encoding="utf-8", newline="") as right:
            expected_rows = [
                row for row in csv.reader(left, quoting=csv.QUOTE_NONE) if row
            ]
            actual_rows = [
                row for row in csv.reader(right, quoting=csv.QUOTE_NONE) if row
            ]
        self.assertEqual(expected_rows[0], actual_rows[0])
        self.assertEqual(len(expected_rows), len(actual_rows))
        for expected, actual in zip(expected_rows[1:], actual_rows[1:]):
            self.assertEqual(
                [struct.pack("!d", float(field)) for field in expected],
                [struct.pack("!d", float(field)) for field in actual])
        return digest

    def test_recorded_raw_extremes_preserve_values_and_metadata(self):
        source = self.root / "recorded.csv"
        with mock.patch.object(recording.time,
                               "monotonic_ns",
                               side_effect=itertools.count(0, 1000000)):
            session = recording.RawJointStateSession(str(source),
                                                     joint_names=self.names)
            self.addCleanup(session.close)
            for index in range(3):
                names = self.names[::-1] if index % 2 else self.names
                position = self.values[::-1] if index % 2 else self.values
                effort = [-value for value in position]
                self.assertIsNone(session.append(names, position, effort))
            self.assertTrue(session.close())
        digest = self.assert_roundtrip(source, "raw")
        metadata = json.loads(Path(str(source) + ".metadata.json").read_text())
        self.assertEqual(metadata["dataset_sha256"], digest)
        self.assertEqual(digest,
                         hashlib.sha256(source.read_bytes()).hexdigest())

    def test_processed_extremes_preserve_every_signal_group(self):
        source = self.root / "processed.csv"
        header = ["time_s"] + [
            f"{group}/{name}" for group in ("q", "dq", "ddq", "tau")
            for name in self.names
        ]
        with source.open("w", encoding="utf-8", newline="") as stream:
            stream.write(",".join(header) + "\r\n")
            for index in range(3):
                values = [index * 0.125]
                for group in range(4):
                    values.extend((-1 if group % 2 else 1) * value
                                  for value in self.values)
                stream.write(",".join(
                    format(value, ".17g") for value in values) + "\r\n")
        self.assert_roundtrip(source, "processed")

    def test_numeric_acceptance_matches_cpp_for_raw_and_processed(self):
        valid = [
            "0", "-0", " +1.25", "\t-2.5e+2", "\v\f1.25", "0x1.8p+2", "-0x.8",
            " +0X1.8P+2", "0x1e2", "1.", ".5", "0e-9999", "-0x0p-9999",
            "1.7976931348623157e308", "2.2250738585072014e-308", "1e-310",
            "4.9406564584124654e-324", "0x1p-1074"
        ]
        invalid = [
            "", " ", "1 ", "1junk", "1e", "nan", "inf", "-infinity", "1e9999",
            "1e-9999", '"1"', "--1", "+-1", "-+1", "++1", "0x-1", "0x+1",
            "-0x-1", "0x", "0x1p", "0x1e+2", "0x1p+1024", "0x1p-1075", "1_000",
            "１２", "\u00a01", "1\0junk"
        ]
        source = self.root / "syntax.csv"
        restored = self.root / "syntax-restored.csv"
        for kind in ("raw", "processed"):
            header = "time_s,q/a,tau/a" if kind == "raw" else "time_s,q/a,dq/a,ddq/a,tau/a"
            for token in valid + invalid:
                with self.subTest(kind=kind, token=token):
                    expected = token in valid
                    rows = [
                        f"{index},{token},{token}"
                        if kind == "raw" else f"{index},{token},0,0,{token}"
                        for index in range(3)
                    ]
                    source.write_text(header + "\n" + "\n".join(rows) + "\n",
                                      encoding="utf-8")
                    result = subprocess.run(
                        [self.executable, kind,
                         str(source),
                         str(restored)],
                        capture_output=True,
                        text=True)
                    self.assertEqual(result.returncode == 0, expected,
                                     result.stderr)
                    if expected:
                        summary, _ = inspect_dataset_with_digest(source)
                        self.assertEqual(
                            summary,
                            inspect_dataset_with_digest(restored)[0])
                        value = float.fromhex(
                            token) if "x" in token.lower() else float(token)
                        self.assertEqual(
                            summary["maximum_absolute_torque"]["a"], abs(value))
                    else:
                        with self.assertRaises(ValueError):
                            inspect_dataset_with_digest(source)

    def test_blank_lines_and_literal_headers_match_cpp(self):
        source = self.root / "blank-lines.csv"
        source.write_bytes(
            b'time_s,q/a"b,tau/a"b\r\n\r\n0,0,1\r\n\r\n1,0,2\r\n2,0,3')
        self.names = ['a"b']
        self.assert_roundtrip(source, "raw")


if __name__ == "__main__":
    CsvInteropTest.executable = sys.argv.pop(1)
    unittest.main()
