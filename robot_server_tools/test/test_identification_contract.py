"""Validate artifact boundaries through the public creation/review workflow."""

import copy
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robot_server_tools import identification_artifact as artifact


class IdentificationContractTest(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = self.root / "data.csv"
        self.dataset.write_text("time_s,q/a,tau/a\n0,0,1\n1,0,2\n2,0,3\n")
        self.parameters = self.root / "parameters.json"
        self.parameters.write_text('{"mass": 1.0}')
        self.diagnostics = self.root / "diagnostics.json"
        self.diagnostics.write_text(
            json.dumps({
                "row_count": 3,
                "rank": 1,
                "condition_number": 1.0,
                "train_rmse": 0.0,
                "validation_rmse": 0.0,
                "mass_matrix_passed": True,
            }))
        self.candidate_path = self.root / "candidate.json"
        self.create_candidate(self.candidate_path)
        self.candidate = json.loads(self.candidate_path.read_text())
        self.configuration = self.root / "robot.yaml"
        self.configuration.write_text("config_version: 1\n")
        self.rollback_path = self.root / "rollback.json"
        artifact.create_rollback(self.configuration, "default",
                                 self.rollback_path)
        self.rollback = json.loads(self.rollback_path.read_text())
        self.output = self.root / "review.json"

    def create_candidate(self, output):
        artifact.create_candidate(self.dataset, "model", ["a"], self.parameters,
                                  self.diagnostics, output)

    def approve(self):
        artifact.create_review(self.candidate_path, "approved", "reviewer",
                               "offline validation", self.output,
                               self.rollback_path)

    def assert_no_output(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + ".partial").exists())

    def assert_candidate_rejected(self, candidate):
        self.candidate_path.write_text(json.dumps(candidate))
        with mock.patch.object(artifact, "_sha256") as hash_configuration:
            with self.assertRaises(ValueError):
                self.approve()
            hash_configuration.assert_not_called()
        self.assert_no_output()

    def test_every_required_candidate_field_is_checked_before_approval(self):
        for name in self.candidate:
            with self.subTest(missing=name):
                candidate = copy.deepcopy(self.candidate)
                del candidate[name]
                self.assert_candidate_rejected(candidate)

    def test_approval_rejects_invalid_candidate_identity_and_arrays(self):
        cases = [
            ("schema_version", True),
            ("schema_version", "1"),
            ("schema_version", 2),
            ("dataset_sha256", "a" * 63),
            ("dataset_sha256", "G" * 64),
            ("dataset_sha256", "a" * 64 + "\n"),
            ("dataset_sha256", None),
            ("model_digest", ""),
            ("model_digest", False),
            ("software_revision", 12),
            ("joint_names", "a"),
            ("joint_names", []),
            ("joint_names", ["a", "a"]),
            ("joint_names", [None]),
            ("joint_names", [""]),
            ("parameter_names", []),
            ("parameter_names", ["mass", "mass"]),
            ("parameter_names", [False]),
            ("parameters", []),
            ("parameters", [1, 2]),
            ("parameters", {
                "mass": 1
            }),
            ("parameters", [True]),
            ("parameters", [10**400]),
            ("parameters", [float("inf")]),
            ("diagnostics", []),
            ("approved", True),
        ]
        for name, value in cases:
            with self.subTest(field=name, value=value):
                candidate = copy.deepcopy(self.candidate)
                candidate[name] = value
                self.assert_candidate_rejected(candidate)

    def test_approval_checks_required_optional_and_unknown_diagnostics(self):
        for name in self.candidate["diagnostics"]:
            with self.subTest(missing=name):
                candidate = copy.deepcopy(self.candidate)
                del candidate["diagnostics"][name]
                self.assert_candidate_rejected(candidate)
        for name, value in (("rank", True), ("rank", 0), ("rank", 1.5),
                            ("row_count", False), ("row_count", 0),
                            ("row_count", 2.5), ("condition_number", 0.5),
                            ("condition_number", 10**400), ("train_rmse", -1),
                            ("validation_rmse",
                             False), ("max_absolute_error",
                                      -1), ("max_absolute_error",
                                            "0"), ("max_absolute_error", True),
                            ("mass_matrix_passed",
                             1), ("mass_matrix_passed",
                                  False), ("validation_rms", 0)):
            with self.subTest(field=name, value=value):
                candidate = copy.deepcopy(self.candidate)
                candidate["diagnostics"][name] = value
                self.assert_candidate_rejected(candidate)

    def test_rank_and_row_count_must_support_all_parameters(self):
        for rank, rows in ((1, 10), (2, 1), (3, 10)):
            with self.subTest(rank=rank, rows=rows):
                candidate = copy.deepcopy(self.candidate)
                candidate["parameter_names"] = ["mass", "friction"]
                candidate["parameters"] = [1, 0.5]
                candidate["diagnostics"].update(rank=rank, row_count=rows)
                self.assert_candidate_rejected(candidate)

    def test_approval_rejects_duplicate_fields_including_escaped_names(self):
        original = self.candidate_path.read_text()
        for payload in (original.replace(
                '"schema_version": 1',
                '"schema_version": 2, "schema_version": 1'),
                        original.replace('"rank": 1',
                                         '"rank": 0, "r\\u0061nk": 1')):
            with self.subTest(payload=payload):
                self.candidate_path.write_text(payload)
                with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                    self.approve()
                self.assert_no_output()

    def test_creation_rejects_ambiguous_json_before_reading_dataset(self):
        valid_diagnostics = self.diagnostics.read_text()
        cases = ((self.parameters, '{"mass": 10, "mass": 1}'),
                 (self.parameters, '{"mass": 10, "m\\u0061ss": 1}'),
                 (self.diagnostics,
                  valid_diagnostics.replace('"rank": 1',
                                            '"rank": 0, "rank": 1')))
        for path, payload in cases:
            with self.subTest(path=path, payload=payload):
                original = path.read_text()
                path.write_text(payload)
                try:
                    with mock.patch.object(
                            artifact, "inspect_dataset_with_digest") as inspect:
                        with self.assertRaisesRegex(ValueError,
                                                    "duplicate JSON field"):
                            self.create_candidate(self.output)
                        inspect.assert_not_called()
                    self.assert_no_output()
                finally:
                    path.write_text(original)

    def test_creation_rejects_nonfinite_and_overflowing_json_numbers(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999",
                      str(10**400)):
            with self.subTest(token=token):
                self.parameters.write_text('{"mass": ' + token + '}')
                with mock.patch.object(
                        artifact, "inspect_dataset_with_digest") as inspect:
                    with self.assertRaises(ValueError):
                        self.create_candidate(self.output)
                    inspect.assert_not_called()
                self.assert_no_output()

    def test_creation_rejects_nonzero_json_underflow(self):
        for index, token in enumerate(
            ("1e-999", "-1E-999", "0." + "0" * 400 + "1")):
            with self.subTest(token=token):
                self.parameters.write_text('{"mass": ' + token + '}')
                output = self.root / f"underflow-{index}.json"
                with self.assertRaisesRegex(ValueError, "underflows to zero"):
                    self.create_candidate(output)
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output) + ".partial").exists())

    def test_json_zero_and_representable_subnormals_are_preserved(self):
        for index, token in enumerate(
            ("0e-999", "-0.0E-999", "5e-324", "-5e-324")):
            with self.subTest(token=token):
                self.parameters.write_text('{"mass": ' + token + '}')
                output = self.root / f"small-{index}.json"
                self.create_candidate(output)
                value = json.loads(output.read_text())["parameters"][0]
                self.assertEqual(value, float(token))

    def test_creation_rejects_unknown_diagnostics_before_reading_dataset(self):
        diagnostics = json.loads(self.diagnostics.read_text())
        diagnostics["validation_rms"] = 0
        self.diagnostics.write_text(json.dumps(diagnostics))
        with mock.patch.object(artifact,
                               "inspect_dataset_with_digest") as inspect:
            with self.assertRaisesRegex(ValueError,
                                        "unknown fields: validation_rms"):
                self.create_candidate(self.output)
            inspect.assert_not_called()
        self.assert_no_output()

    def test_integer_counts_preserve_all_digits(self):
        diagnostics = json.loads(self.diagnostics.read_text())
        for index, count in enumerate((2**53 + 1, 2**63 + 1, 10**400)):
            with self.subTest(count=count):
                diagnostics["row_count"] = count
                self.diagnostics.write_text(json.dumps(diagnostics))
                path = self.root / f"large-count-{index}.json"
                self.create_candidate(path)
                candidate = json.loads(path.read_text())
                self.assertEqual(candidate["diagnostics"]["row_count"], count)
                self.assertIsInstance(candidate["diagnostics"]["row_count"],
                                      int)
                artifact.create_review(path, "approved", "reviewer",
                                       "validated",
                                       self.root / f"large-review-{index}.json",
                                       self.rollback_path)

    def test_integral_json_floats_are_accepted_without_treating_booleans_as_numbers(
            self):
        self.candidate["schema_version"] = 1.0
        self.candidate["diagnostics"].update(row_count=3.0,
                                             rank=1.0,
                                             max_absolute_error=0)
        self.candidate["software_revision"] = ""
        self.candidate_path.write_text(json.dumps(self.candidate))
        self.approve()
        self.assertEqual(
            json.loads(self.output.read_text())["candidate_sha256"],
            hashlib.sha256(self.candidate_path.read_bytes()).hexdigest())

    def test_creation_normalizes_integral_float_counts(self):
        diagnostics = json.loads(self.diagnostics.read_text())
        diagnostics.update(row_count=3.0, rank=1.0)
        self.diagnostics.write_text(json.dumps(diagnostics))
        self.create_candidate(self.output)
        normalized = json.loads(self.output.read_text())["diagnostics"]
        self.assertIs(type(normalized["rank"]), int)
        self.assertIs(type(normalized["row_count"]), int)

    def test_parameter_limit_matches_the_published_schema(self):
        schema_path = Path(__file__).resolve().parents[
            2] / "robot_identification/schema/identification-result.schema.json"
        schema = json.loads(schema_path.read_text())
        limit = schema["allOf"][0]["properties"]["parameters"]["maxItems"]
        parameters = {f"p{index}": 1 for index in range(limit)}
        self.parameters.write_text(json.dumps(parameters))
        diagnostics = json.loads(self.diagnostics.read_text())
        diagnostics.update(rank=limit, row_count=limit)
        self.diagnostics.write_text(json.dumps(diagnostics))
        valid_path = self.root / "parameter-limit.json"
        self.create_candidate(valid_path)
        artifact.create_review(valid_path, "approved", "reviewer", "at limit",
                               self.output, self.rollback_path)
        self.assertEqual(len(json.loads(valid_path.read_text())["parameters"]),
                         limit)

        parameters["too_many"] = 1
        self.parameters.write_text(json.dumps(parameters))
        invalid_path = self.root / "too-many.json"
        with mock.patch.object(artifact,
                               "inspect_dataset_with_digest") as inspect:
            with self.assertRaisesRegex(ValueError, "at most"):
                self.create_candidate(invalid_path)
            inspect.assert_not_called()
        self.assertFalse(invalid_path.exists())
        candidate = json.loads(valid_path.read_text())
        candidate["parameter_names"].append("too_many")
        candidate["parameters"].append(1)
        self.candidate_path.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, "at most"):
            artifact.create_review(self.candidate_path, "approved", "reviewer",
                                   "over limit", invalid_path,
                                   self.rollback_path)
        self.assertFalse(invalid_path.exists())

    def test_every_required_rollback_field_is_checked_before_configuration_read(
            self):
        for name in self.rollback:
            with self.subTest(missing=name):
                rollback = copy.deepcopy(self.rollback)
                del rollback[name]
                self.rollback_path.write_text(json.dumps(rollback))
                with mock.patch.object(artifact,
                                       "_sha256") as hash_configuration:
                    with self.assertRaises(ValueError):
                        self.approve()
                    hash_configuration.assert_not_called()
                self.assert_no_output()

    def test_approval_rejects_invalid_rollback_contracts(self):
        for name, value in (("schema_version", True), ("schema_version", 2),
                            ("kind", "other"), ("configuration_path", ""),
                            ("configuration_path",
                             "robot.yaml"), ("configuration_sha256", "a" * 63),
                            ("configuration_sha256",
                             "A" * 64), ("configuration_sha256",
                                         "a" * 64 + "\n"), ("profile", ""),
                            ("profile", False), ("created_at", None),
                            ("created_at", ""), ("created_at",
                                                 "2026-09-05T12:00:00"),
                            ("created_at",
                             "2026-02-30T12:00:00Z"), ("created_at",
                                                       "2026-09-05T24:00:00Z"),
                            ("created_at", "2026-09-05T12:00:00+00:60"),
                            ("created_at",
                             "2026-09-05 12:00:00+08:00"), ("confirmed", True)):
            with self.subTest(field=name, value=value):
                rollback = copy.deepcopy(self.rollback)
                rollback[name] = value
                self.rollback_path.write_text(json.dumps(rollback))
                with mock.patch.object(artifact,
                                       "_sha256") as hash_configuration:
                    with self.assertRaises(ValueError):
                        self.approve()
                    hash_configuration.assert_not_called()
                self.assert_no_output()

    def test_approval_accepts_timezone_timestamps_and_binds_original_bytes(
            self):
        for index, timestamp in enumerate(
            ("2026-09-05T12:00:00Z", "2026-09-05t12:00:00z",
             "2026-09-05T12:00:00.123456+08:00", "2024-02-29T23:59:59-05:30")):
            with self.subTest(timestamp=timestamp):
                self.rollback["created_at"] = timestamp
                self.rollback_path.write_text(
                    json.dumps(self.rollback, ensure_ascii=True))
                output = self.root / f"timestamp-{index}.json"
                artifact.create_review(self.candidate_path, "approved",
                                       "reviewer", "validated", output,
                                       self.rollback_path)
                review = json.loads(output.read_text())
                self.assertEqual(
                    review["rollback_sha256"],
                    hashlib.sha256(self.rollback_path.read_bytes()).hexdigest())

    def test_approval_rejects_ambiguous_rollback(self):
        payload = self.rollback_path.read_text().replace(
            '"profile": "default"', '"profile": "other", "profile": "default"')
        self.rollback_path.write_text(payload)
        with self.assertRaisesRegex(ValueError,
                                    "duplicate JSON field: profile"):
            self.approve()
        self.assert_no_output()

    def test_rejection_can_document_incomplete_candidate_without_rollback(self):
        self.candidate_path.write_text(
            json.dumps({
                "kind": "robot_identification_result",
                "status": "candidate",
                "diagnostics": {
                    "mass_matrix_passed": False
                },
            }))
        with mock.patch.object(artifact, "_sha256") as hash_configuration:
            artifact.create_review(self.candidate_path, "rejected", "reviewer",
                                   "missing provenance", self.output)
            hash_configuration.assert_not_called()
        review = json.loads(self.output.read_text())
        self.assertEqual(review["decision"], "rejected")
        self.assertNotIn("rollback_sha256", review)
        self.assertEqual(
            review["candidate_sha256"],
            hashlib.sha256(self.candidate_path.read_bytes()).hexdigest())

    def test_rejection_still_requires_unambiguous_json(self):
        self.candidate_path.write_text(
            '{"kind": "robot_identification_result",'
            '"status": "other", "status": "candidate"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            artifact.create_review(self.candidate_path, "rejected", "reviewer",
                                   "ambiguous", self.output)
        self.assert_no_output()

    def test_cli_validation_errors_are_concise_and_do_not_publish(self):
        del self.candidate["dataset_sha256"]
        self.candidate_path.write_text(json.dumps(self.candidate))
        args = [
            "identification_artifact", "review", "--candidate",
            str(self.candidate_path), "--decision", "approved", "--reviewer",
            "tester", "--reason", "invalid", "--rollback",
            str(self.rollback_path), "--output",
            str(self.output)
        ]
        errors = io.StringIO()
        with mock.patch.object(sys, "argv",
                               args), contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                artifact.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("missing fields: dataset_sha256", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.assert_no_output()

    def test_cli_publication_errors_preserve_existing_artifacts(self):
        original = self.rollback_path.read_bytes()
        args = [
            "identification_artifact", "rollback", "--configuration",
            str(self.configuration), "--profile", "new-profile", "--output",
            str(self.rollback_path)
        ]
        errors = io.StringIO()
        with mock.patch.object(sys, "argv",
                               args), contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                artifact.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("refusing to overwrite", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertEqual(self.rollback_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
