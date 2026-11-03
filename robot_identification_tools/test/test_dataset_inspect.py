import contextlib
import hashlib
import io
import json
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from robot_identification_tools import dataset_inspect as inspection
from robot_identification_tools.dataset_inspect import inspect_dataset
from robot_identification_tools.dataset_inspect import inspect_dataset_with_digest


class DatasetInspectTest(unittest.TestCase):

    def test_numeric_syntax_and_representable_extremes(self):
        tokens = [
            "0", "-0", "+0", "0e-9999", "-0.0e+9999", " \t+1.25", "\v\f-2.5e2",
            ".5", "1.", "+0X1.8P+2", "-0x.8", "0x1e2", "0x0p+9999",
            "-0x0p-9999", "1.7976931348623157e308", "1e-310",
            "4.9406564584124654e-324", "0x1p-1074", "." + "0" * 323 + "5"
        ]
        for token in tokens:
            with self.subTest(token=token):
                expected = float.fromhex(
                    token) if "x" in token.lower() else float(token)
                summary = self.inspect_text("time_s,q/a,tau/a\n" + "\n".join(
                    f"{index},{token},{token}" for index in range(3)))
                self.assertEqual(summary["maximum_absolute_torque"]["a"],
                                 abs(expected))

    def test_rejects_numeric_extensions_not_supported_by_cpp(self):
        for token in ("1_000", "１２", "١", '"1"', "1 ", "1\t", "\u00a01",
                      "1\u00a0", "1e", "++1", "--1", "+-1", "0x+1", "-0x-1",
                      "0x1e+2", "0x", ".", "1\0x", "0x1p+-2", "-0X0P+-999",
                      "0x1p--2", "0x1p-+2", "0x1p++2"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "line 2, column 2"):
                    self.inspect_text(
                        f"time_s,q/a,tau/a\n0,{token},1\n1,0,2\n2,0,3\n")

    def test_rejects_nonzero_underflow_without_rejecting_zero_mantissas(self):
        for token in ("1e-9999", "-1e-9999", "0x1p-1075", "-0x1p-9999",
                      "0x0e0p-9999", "." + "0" * 323 + "1",
                      "-." + "0" * 323 + "1", "0." + "0" * 400 + "1"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "underflows to zero"):
                    self.inspect_text(
                        f"time_s,q/a,tau/a\n0,0,{token}\n1,0,0\n2,0,0\n")
        for token in ("0e9999", "-0e-9999", "0.000e9999", "0x0p-9999",
                      "-0x0.0p+9999"):
            with self.subTest(token=token):
                summary = self.inspect_text(
                    f"time_s,q/a,tau/a\n0,0,{token}\n1,0,0\n2,0,0\n")
                self.assertEqual(summary["maximum_absolute_torque"]["a"], 0)

    def test_blank_lines_keep_original_digest_and_physical_error_locations(
            self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            payload = b"time_s,q/a,tau/a\r\n\r\n0,0,1\r\n\r\n1,0,2\r\n2,0,3\r\n\r\n"
            path.write_bytes(payload)
            summary, digest = inspect_dataset_with_digest(path)
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(summary, inspect_dataset(path))
            path.write_bytes(payload.replace(b"1,0,2", b"1,1e-9999,2"))
            with self.assertRaisesRegex(ValueError, "line 5, column 2"):
                inspect_dataset_with_digest(path)
            path.write_text("time_s,q/a,tau/a\n \n0,0,1\n1,0,2\n2,0,3\n")
            with self.assertRaisesRegex(ValueError, "line 2.*field count"):
                inspect_dataset(path)

    def test_headers_use_literal_unquoted_fields(self):
        for header in ('"time_s",q/a,tau/a', 'time_s,"q/a","tau/a"'):
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    self.inspect_text(header + "\n0,0,1\n1,0,2\n2,0,3\n")
        summary = self.inspect_text(
            'time_s,q/a"b,tau/a"b\n0,0,1\n1,0,2\n2,0,3\n')
        self.assertEqual(summary["joint_names"], ['a"b'])

    def test_processed_syntax_validation_includes_every_signal_group(self):
        header = "time_s,q/a,dq/a,ddq/a,tau/a\n"
        for column in range(5):
            with self.subTest(column=column):
                row = ["0"] * 5
                row[column] = "1_0"
                with self.assertRaisesRegex(ValueError,
                                            f"line 2, column {column + 1}"):
                    self.inspect_text(header + ",".join(row) +
                                      "\n1,0,0,0,0\n2,0,0,0,0\n")

    def test_cli_reports_invalid_data_and_missing_files_without_tracebacks(
            self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            for exists in (False, True):
                with self.subTest(exists=exists):
                    if exists:
                        path.write_text(
                            "time_s,q/a,tau/a\n0,1_0,1\n1,0,2\n2,0,3\n")
                    errors = io.StringIO()
                    with mock.patch.object(
                            sys, "argv",
                        ["dataset_inspect",
                         str(path), "--json"
                        ]), contextlib.redirect_stderr(errors):
                        with self.assertRaises(SystemExit) as raised:
                            inspection.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertNotIn("Traceback", errors.getvalue())
                    self.assertIn("error:", errors.getvalue())
                    if exists:
                        self.assertIn("line 2, column 2", errors.getvalue())

    def test_cli_json_summary_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("time_s,q/a,tau/a\n0,0,1\n1,0,2\n2,0,3\n")
            output = io.StringIO()
            with mock.patch.object(
                    sys, "argv",
                ["dataset_inspect", str(path), "--json"
                ]), contextlib.redirect_stdout(output):
                inspection.main()
            self.assertEqual(json.loads(output.getvalue()),
                             inspect_dataset(path))

    def test_digest_preserves_original_line_endings_and_unicode(self):
        for newline in ("\n", "\r\n", "\r"):
            for final_newline in (False, True):
                with self.subTest(newline=newline, final_newline=final_newline):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "dataset.csv"
                        text = newline.join(
                            ["time_s,q/关节,tau/关节", "0,0,1", "1,0,2", "2,0,3"])
                        payload = (
                            text +
                            (newline if final_newline else "")).encode("utf-8")
                        path.write_bytes(payload)
                        summary, digest = inspect_dataset_with_digest(path)
                        self.assertEqual(summary, inspect_dataset(path))
                        self.assertEqual(digest,
                                         hashlib.sha256(payload).hexdigest())

    def test_hashing_rejects_invalid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_text("time_s,q/a,tau/a\n0,0,1\n1,0,nan\n2,0,1\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                inspect_dataset_with_digest(path)

    def inspect_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_text(text, encoding="utf-8")
            return inspect_dataset(path)

    def test_raw_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text(
                "time_s,q/a,tau/a\n0,0,1\n0.01,0.1,-2\n0.02,0.2,1.5\n",
                encoding="utf-8")
            result = inspect_dataset(path)
            self.assertEqual(result["kind"], "raw")
            self.assertEqual(result["sample_count"], 3)
            self.assertAlmostEqual(result["mean_rate_hz"], 100.0)
            self.assertEqual(result["maximum_absolute_torque"]["a"], 2.0)

    def test_rejects_non_monotonic_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("time_s,q/a,tau/a\n0,0,1\n0.01,0.1,1\n0.01,0.2,1\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                inspect_dataset(path)

    def test_processed_summary_with_irregular_timing(self):
        result = self.inspect_text(
            "time_s,q/a,q/b,dq/a,dq/b,ddq/a,ddq/b,tau/a,tau/b\r\n"
            "-0.5,0,1,2,3,4,5,1,-7\r\n"
            "0,1,2,3,4,5,6,-2,6\r\n"
            "0.25,2,3,4,5,6,7,1.5,5\r\n"
            "1,3,4,5,6,7,8,0,4\r\n")
        self.assertEqual(result["kind"], "processed")
        self.assertEqual(result["joint_names"], ["a", "b"])
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["duration_s"], 1.5)
        self.assertEqual(result["mean_rate_hz"], 2.0)
        self.assertEqual(result["minimum_interval_s"], 0.25)
        self.assertEqual(result["maximum_interval_s"], 0.75)
        self.assertEqual(result["maximum_absolute_torque"], {
            "a": 2.0,
            "b": 7.0
        })

    def test_rejects_non_finite_derived_timing(self):
        for times, message in ((["-1e308", "1e308", "1.1e308"], "interval"),
                               (["-1e308", "0", "1e308"], "duration"),
                               (["0", "1e-320", "2e-320"], "sample rate")):
            with self.subTest(times=times):
                with self.assertRaisesRegex(ValueError, message):
                    self.inspect_text("time_s,q/a,tau/a\n" + "".join(
                        f"{time},0,1\n" for time in times))

    def test_rejects_incomplete_or_non_finite_data(self):
        for rows, message in (("0,0,1\n1,0,1\n", "at least three"),
                              ("0,0,1\n1,0\n2,0,1\n", "field count"),
                              ("0,0,1\n1,0,nan\n2,0,1\n", "non-finite")):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, message):
                    self.inspect_text("time_s,q/a,tau/a\n" + rows)

    def test_memory_does_not_grow_with_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            for inspect in (inspect_dataset, inspect_dataset_with_digest):
                with self.subTest(inspector=inspect.__name__):
                    peaks = []
                    for count in (100, 20000):
                        with path.open("w", encoding="utf-8") as stream:
                            stream.write("time_s,q/a,tau/a\n")
                            stream.writelines(
                                f"{index},0,1\n" for index in range(count))
                        tracemalloc.start()
                        try:
                            result = inspect(path)
                            peaks.append(tracemalloc.get_traced_memory()[1])
                        finally:
                            tracemalloc.stop()
                        summary = result[0] if isinstance(result,
                                                          tuple) else result
                        self.assertEqual(summary["sample_count"], count)
                    self.assertLess(peaks[1], peaks[0] + 128 * 1024)


if __name__ == "__main__":
    unittest.main()
