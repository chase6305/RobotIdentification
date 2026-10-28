"""Check workspace discovery and failure propagation without requiring ROS."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class Ros2RunnerTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "workspace with spaces"
        (self.root / "scripts").mkdir(parents=True)
        self.script = self.root / "scripts/test_ros2.sh"
        shutil.copy2(
            Path(__file__).resolve().parents[2] / "scripts/test_ros2.sh",
            self.script)
        (self.root / "utils").mkdir()
        (self.root / "utils/ros2_recording_smoke.py"
        ).write_text("from pathlib import Path\nPath('smoke-ran').touch()\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        colcon = self.bin / "colcon"
        colcon.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "p = pathlib.Path(os.environ['CALLS'])\n"
            "calls = json.loads(p.read_text()) if p.exists() else []\n"
            "args = sys.argv[1:]\n"
            "p.write_text(json.dumps(calls + [args]))\n"
            "phase = next(a for a in args if a in ('build', 'test', 'test-result'))\n"
            "if phase == os.environ.get('FAIL_PHASE'): sys.exit(7)\n"
            "if phase == 'build':\n"
            "    install = pathlib.Path(args[args.index('--install-base') + 1])\n"
            "    install.mkdir(parents=True, exist_ok=True)\n"
            "    (install / 'setup.bash').write_text('')\n")
        colcon.chmod(0o755)
        git = self.bin / "git"
        git.write_text("#!/usr/bin/env bash\nexit 0\n")
        git.chmod(0o755)
        self.calls = self.root / "calls.json"

    def package(self, name):
        path = self.root / name
        path.mkdir(parents=True)
        (path / "package.xml").write_text("<package/>")
        return str(path)

    def run_script(self, fail_phase=""):
        return subprocess.run(
            [str(self.script),
             str(self.root.parent / "external output")],
            env={
                **os.environ, "PATH":
                    f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "PYTHON":
                    sys.executable,
                "CALLS":
                    str(self.calls),
                "FAIL_PHASE":
                    fail_phase
            },
            capture_output=True,
            text=True,
            timeout=15)

    def test_build_and_test_share_only_root_packages(self):
        expected = [self.package("adapter with spaces"), self.package("core")]
        self.package("build/dependencies/core")
        self.package("docs/consumer")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = json.loads(self.calls.read_text())
        for phase in ("build", "test"):
            args = next(call for call in calls if phase in call)
            self.assertEqual(
                args[args.index("--paths") + 1:args.index("--build-base")],
                expected)
        self.assertIn("--return-code-on-test-failure", calls[1])
        self.assertTrue((self.root / "smoke-ran").exists())

    def test_build_failure_stops_before_testing(self):
        self.package("core")
        result = self.run_script("build")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(json.loads(self.calls.read_text())), 1)
        self.assertFalse((self.root / "smoke-ran").exists())

    def test_test_failure_stops_before_smoke(self):
        self.package("core")
        result = self.run_script("test")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(json.loads(self.calls.read_text())), 2)
        self.assertFalse((self.root / "smoke-ran").exists())

    def test_empty_workspace_fails_before_building(self):
        self.package("build/dependency")
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("No root-level", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
