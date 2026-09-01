import tempfile
import unittest
from pathlib import Path

from robot_server_tools.dataset_inspect import inspect_dataset


class DatasetInspectTest(unittest.TestCase):
    def test_raw_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("time_s,q/a,tau/a\n0,0,1\n0.01,0.1,-2\n0.02,0.2,1.5\n",
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


if __name__ == "__main__":
    unittest.main()
