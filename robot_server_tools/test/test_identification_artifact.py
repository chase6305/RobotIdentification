import json
import tempfile
import unittest
from pathlib import Path

from robot_server_tools.identification_artifact import create_candidate
from robot_server_tools.identification_artifact import create_review
from robot_server_tools.identification_artifact import create_rollback


class IdentificationArtifactTest(unittest.TestCase):
    def test_candidate_review_and_rollback_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "time_s,q/a,tau/a\n0,0,0\n0.01,0.1,0.2\n0.02,0.2,0.4\n",
                encoding="utf-8")
            parameters = root / "parameters.json"
            parameters.write_text(json.dumps({"rigid_body/a/mass": 1.2}), encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps({
                "row_count": 100,
                "rank": 1,
                "condition_number": 2.0,
                "train_rmse": 0.1,
                "validation_rmse": 0.2,
                "mass_matrix_passed": True,
            }), encoding="utf-8")
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
            create_review(candidate, "approved", "reviewer", "validated offline",
                          review, rollback)
            review_value = json.loads(review.read_text(encoding="utf-8"))
            self.assertEqual(review_value["decision"], "approved")
            self.assertEqual(len(review_value["candidate_sha256"]), 64)
            self.assertEqual(len(review_value["rollback_sha256"]), 64)

            with self.assertRaises(FileExistsError):
                create_review(candidate, "approved", "reviewer", "again", review,
                              rollback)

            configuration.write_text("config_version: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                create_review(candidate, "approved", "reviewer", "stale rollback",
                              root / "second-review.json", rollback)

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
            }), encoding="utf-8")
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
            }), encoding="utf-8")
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
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                create_candidate(dataset, "model", ["different_joint"],
                                 parameters, diagnostics,
                                 root / "candidate.json")


if __name__ == "__main__":
    unittest.main()
