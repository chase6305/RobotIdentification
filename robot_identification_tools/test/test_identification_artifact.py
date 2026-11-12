import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robot_identification_tools import identification_artifact as artifact
from robot_identification_tools.identification_artifact import create_candidate
from robot_identification_tools.identification_artifact import create_review
from robot_identification_tools.identification_artifact import create_rollback


class IdentificationArtifactTest(unittest.TestCase):

    @staticmethod
    def candidate_inputs(root):
        dataset = root / "dataset.csv"
        dataset.write_text("time_s,q/a,tau/a\n0,0,1\n1,0,2\n2,0,3\n",
                           encoding="utf-8")
        parameters = root / "parameters.json"
        parameters.write_text('{"mass": 1.0}', encoding="utf-8")
        diagnostics = root / "diagnostics.json"
        diagnostics.write_text(json.dumps({
            "row_count": 3,
            "rank": 1,
            "condition_number": 1.0,
            "train_rmse": 0.0,
            "validation_rmse": 0.0,
            "mass_matrix_passed": True,
        }),
                               encoding="utf-8")
        return dataset, parameters, diagnostics

    def test_candidate_reads_and_hashes_the_validated_dataset_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, parameters, diagnostics = self.candidate_inputs(root)
            expected_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            inspect = artifact.inspect_dataset_with_digest
            open_path = Path.open
            reads = []

            def tracked_open(path, *args, **kwargs):
                mode = args[0] if args else kwargs.get("mode", "r")
                if path == dataset and "r" in mode:
                    reads.append(mode)
                return open_path(path, *args, **kwargs)

            def change_after_inspection(path):
                result = inspect(path)
                path.write_text("unvalidated replacement", encoding="utf-8")
                return result

            output = root / "candidate.json"
            with mock.patch.object(Path, "open",
                                   new=tracked_open), mock.patch.object(
                                       artifact,
                                       "inspect_dataset_with_digest",
                                       side_effect=change_after_inspection):
                create_candidate(dataset, "model", ["a"], parameters,
                                 diagnostics, output)
            self.assertEqual(reads, ["rb"])
            candidate = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(candidate["dataset_sha256"], expected_digest)
            self.assertNotEqual(
                candidate["dataset_sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest())

    def test_existing_outputs_fail_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact.json"
            partial = Path(str(output) + ".partial")
            missing = root / "absent-input"
            operations = (
                lambda: create_candidate(missing, "model", ["a"], missing,
                                         missing, output),
                lambda: create_review(missing, "approved", "reviewer",
                                      "checked", output, missing),
                lambda: create_rollback(missing, "default", output),
            )
            for blocker in (output, partial):
                for kind in ("file", "directory", "dangling symlink"):
                    if kind == "file":
                        blocker.write_bytes(b"existing artifact")
                    elif kind == "directory":
                        blocker.mkdir()
                    else:
                        blocker.symlink_to(missing)
                    for operation in operations:
                        with self.subTest(blocker=blocker.name,
                                          kind=kind,
                                          operation=operation):
                            # Every input is absent: an attempted read would
                            # raise FileNotFoundError rather than this conflict.
                            with self.assertRaisesRegex(
                                    FileExistsError, "refusing to overwrite"):
                                operation()
                            if kind == "file":
                                self.assertEqual(blocker.read_bytes(),
                                                 b"existing artifact")
                            elif kind == "directory":
                                self.assertTrue(blocker.is_dir())
                            else:
                                self.assertEqual(blocker.readlink(), missing)
                                self.assertFalse(missing.exists())
                    if kind == "directory":
                        blocker.rmdir()
                    else:
                        blocker.unlink()

    def test_blockers_published_during_validation_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, parameters, diagnostics = self.candidate_inputs(root)
            output = root / "artifact.json"
            partial = Path(str(output) + ".partial")
            inspect = artifact.inspect_dataset_with_digest
            for blocker in (output, partial):
                with self.subTest(blocker=blocker.name):

                    def publish_after_inspection(path):
                        summary = inspect(path)
                        blocker.write_bytes(b"another publisher")
                        return summary

                    with mock.patch.object(
                            artifact,
                            "inspect_dataset_with_digest",
                            side_effect=publish_after_inspection):
                        with self.assertRaisesRegex(FileExistsError,
                                                    "refusing to overwrite"):
                            create_candidate(dataset, "model", ["a"],
                                             parameters, diagnostics, output)
                    self.assertEqual(blocker.read_bytes(), b"another publisher")
                    if blocker == output:
                        self.assertFalse(partial.exists())
                    else:
                        self.assertFalse(output.exists())
                    blocker.unlink()

    def test_invalid_metadata_fails_before_scanning_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, parameters, diagnostics = self.candidate_inputs(root)
            with mock.patch.object(artifact,
                                   "inspect_dataset_with_digest") as inspect:
                with self.assertRaisesRegex(ValueError, "software_revision"):
                    create_candidate(dataset, "model", ["a"], parameters,
                                     diagnostics, root / "candidate.json", 42)
                parameters.write_text('{"mass": true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "finite number"):
                    create_candidate(dataset, "model", ["a"], parameters,
                                     diagnostics, root / "candidate.json")
                inspect.assert_not_called()

    def test_candidate_numeric_validation_matches_cpp_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, parameters, diagnostics = self.candidate_inputs(root)
            output = root / "candidate.json"
            for token in ("1_000", '"1"', "１２", "1 ", "1e-9999", "0x1p-1075"):
                with self.subTest(token=token):
                    dataset.write_text(
                        f"time_s,q/a,tau/a\n0,{token},1\n1,0,2\n2,0,3\n",
                        encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "line 2, column 2"):
                        create_candidate(dataset, "model", ["a"], parameters,
                                         diagnostics, output)
                    self.assertFalse(output.exists())
                    self.assertFalse(Path(str(output) + ".partial").exists())
            dataset.write_text(
                "time_s,q/a,tau/a\n\n0,0x1p-1074,1\n1,-0x.8,2\n2,0,3\n",
                encoding="utf-8")
            create_candidate(dataset, "model", ["a"], parameters, diagnostics,
                             output)
            candidate = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(candidate["dataset_sha256"],
                             hashlib.sha256(dataset.read_bytes()).hexdigest())

    def test_concurrent_publication_does_not_overwrite_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "robot.yaml"
            configuration.write_text("config_version: 1\n", encoding="utf-8")
            output = root / "rollback.json"
            fsync = os.fsync

            def publish_competing_file(descriptor):
                output.write_text("another publisher", encoding="utf-8")
                fsync(descriptor)

            with mock.patch.object(artifact.os,
                                   "fsync",
                                   side_effect=publish_competing_file):
                with self.assertRaises(FileExistsError):
                    create_rollback(configuration, "default", output)
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "another publisher")
            self.assertFalse(Path(str(output) + ".partial").exists())

    def test_failed_write_can_be_retried_without_removing_foreign_partials(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "robot.yaml"
            configuration.write_text("config_version: 1\n", encoding="utf-8")
            output = root / "rollback.json"
            partial = Path(str(output) + ".partial")
            with mock.patch.object(artifact.os,
                                   "fsync",
                                   side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    create_rollback(configuration, "default", output)
            self.assertFalse(output.exists())
            self.assertFalse(partial.exists())

            partial.write_text("another writer owns this", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_rollback(configuration, "default", output)
            self.assertEqual(partial.read_text(encoding="utf-8"),
                             "another writer owns this")
            partial.unlink()
            create_rollback(configuration, "default", output)
            self.assertEqual(
                json.loads(output.read_text())["kind"],
                "robot_identification_rollback")

    def test_dangling_output_link_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "robot.yaml"
            configuration.write_text("config_version: 1\n", encoding="utf-8")
            target = root / "absent.json"
            output = root / "rollback.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                create_rollback(configuration, "default", output)
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())

    def test_review_hashes_the_exact_validated_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({
                "schema_version": 1,
                "kind": "robot_identification_result",
                "status": "candidate",
                "dataset_sha256": "0" * 64,
                "model_digest": "model",
                "joint_names": ["a"],
                "parameter_names": ["mass"],
                "parameters": [1.0],
                "diagnostics": {
                    "rank": 1,
                    "row_count": 10,
                    "mass_matrix_passed": True,
                    "condition_number": 1.0,
                    "train_rmse": 0,
                    "validation_rmse": 0,
                },
            }),
                                 encoding="utf-8")
            configuration = root / "robot.yaml"
            configuration.write_text("config_version: 1\n", encoding="utf-8")
            rollback = root / "rollback.json"
            create_rollback(configuration, "default", rollback)
            candidate_digest = hashlib.sha256(
                candidate.read_bytes()).hexdigest()
            rollback_digest = hashlib.sha256(rollback.read_bytes()).hexdigest()
            sha256 = artifact._sha256

            def change_files_after_validation(path):
                if path == configuration:
                    candidate.write_text('{"parameters": [999]}',
                                         encoding="utf-8")
                    rollback.write_text('{"profile": "changed"}',
                                        encoding="utf-8")
                return sha256(path)

            output = root / "review.json"
            with mock.patch.object(artifact,
                                   "_sha256",
                                   side_effect=change_files_after_validation):
                create_review(candidate, "approved", "reviewer", "validated",
                              output, rollback)
            review = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(review["candidate_sha256"], candidate_digest)
            self.assertEqual(review["rollback_sha256"], rollback_digest)
            self.assertNotEqual(review["candidate_sha256"], sha256(candidate))
            self.assertNotEqual(review["rollback_sha256"], sha256(rollback))

    def test_candidate_review_and_rollback_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "time_s,q/a,tau/a\n0,0,0\n0.01,0.1,0.2\n0.02,0.2,0.4\n",
                encoding="utf-8")
            parameters = root / "parameters.json"
            parameters.write_text(json.dumps({"rigid_body/a/mass": 1.2}),
                                  encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps({
                "row_count": 100,
                "rank": 1,
                "condition_number": 2.0,
                "train_rmse": 0.1,
                "validation_rmse": 0.2,
                "mass_matrix_passed": True,
            }),
                                   encoding="utf-8")
            candidate = root / "candidate.json"
            create_candidate(dataset, "model-sha256", ["a"], parameters,
                             diagnostics, candidate, "revision")
            candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertEqual(candidate_value["status"], "candidate")
            self.assertEqual(candidate_value["parameters"], [1.2])

            configuration = root / "robot.yaml"
            configuration.write_text("config_version: 1\n", encoding="utf-8")
            rollback = root / "rollback.json"
            create_rollback(configuration, "default", rollback)
            rollback_value = json.loads(rollback.read_text(encoding="utf-8"))
            self.assertEqual(len(rollback_value["configuration_sha256"]), 64)

            review = root / "review.json"
            create_review(candidate, "approved", "reviewer",
                          "validated offline", review, rollback)
            review_value = json.loads(review.read_text(encoding="utf-8"))
            self.assertEqual(review_value["decision"], "approved")
            self.assertEqual(len(review_value["candidate_sha256"]), 64)
            self.assertEqual(len(review_value["rollback_sha256"]), 64)

            with self.assertRaises(FileExistsError):
                create_review(candidate, "approved", "reviewer", "again",
                              review, rollback)

            configuration.write_text("config_version: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                create_review(candidate, "approved", "reviewer",
                              "stale rollback", root / "second-review.json",
                              rollback)

    def test_rejects_fractional_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "time_s,q/a,tau/a\n0,0,0\n0.01,0.1,0.2\n0.02,0.2,0.4\n",
                encoding="utf-8")
            parameters = root / "parameters.json"
            parameters.write_text('{"p": 1}', encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps({
                "row_count": 10,
                "rank": 1.5,
                "condition_number": 2,
                "train_rmse": 0,
                "validation_rmse": 0,
                "mass_matrix_passed": True,
            }),
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rank must be an integer"):
                create_candidate(dataset, "model", ["a"], parameters,
                                 diagnostics, root / "candidate.json")

    def test_rejects_boolean_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "time_s,q/a,tau/a\n0,0,0\n0.01,0.1,0.2\n0.02,0.2,0.4\n",
                encoding="utf-8")
            parameters = root / "parameters.json"
            parameters.write_text('{"p": true}', encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps({
                "row_count": 10,
                "rank": 1,
                "condition_number": 2,
                "train_rmse": 0,
                "validation_rmse": 0,
                "mass_matrix_passed": True,
            }),
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite number"):
                create_candidate(dataset, "model", ["a"], parameters,
                                 diagnostics, root / "candidate.json")

    def test_rejects_dataset_joint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "time_s,q/a,tau/a\n0,0,0\n0.01,0.1,0.2\n0.02,0.2,0.4\n",
                encoding="utf-8")
            parameters = root / "parameters.json"
            parameters.write_text('{"p": 1}', encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps({
                "row_count": 10,
                "rank": 1,
                "condition_number": 2,
                "train_rmse": 0,
                "validation_rmse": 0,
                "mass_matrix_passed": True,
            }),
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                create_candidate(dataset, "model", ["different_joint"],
                                 parameters, diagnostics,
                                 root / "candidate.json")


if __name__ == "__main__":
    unittest.main()
