import csv
import hashlib
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robot_identification_ros2 import raw_joint_state_session as recording


class RawJointStateSessionTest(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "session.csv"
        clock_patch = mock.patch.object(recording.time,
                                        "monotonic_ns",
                                        side_effect=itertools.count(0, 1000000))
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def session(self, **options):
        session = recording.RawJointStateSession(str(self.path), **options)
        self.addCleanup(session._stream.close)
        return session

    def metadata(self):
        return json.loads(Path(str(self.path) + ".metadata.json").read_text())

    def test_records_ordered_unicode_samples_and_exact_digest_without_reread(
            self):
        session = self.session(joint_names=["关节", "b"],
                               flush_interval=2,
                               session_id="run-1",
                               input_topic="/arm/joints",
                               model_digest="model-hash",
                               torque_source="calibrated")
        for names, position, effort in ((["b", "关节"], [2.0, 1.0], [4.0, 3.0]),
                                        (["关节", "b"], [5.0, 6.0], [7.0, 8.0]),
                                        (["关节", "b"], [9.0, 10.0], [11.0,
                                                                    12.0])):
            self.assertIsNone(session.append(names, position, effort))
        self.assertIsNotNone(
            session.append(["关节", "b"], [0.0, 1.0], [float("nan"), 2.0]))
        original_open = Path.open

        def no_read(path, mode="r", *args, **kwargs):
            self.assertNotIn("r", mode)
            self.assertNotIn("+", mode)
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, "open", no_read):
            self.assertTrue(session.close())
            self.assertTrue(session.close())
        payload = self.path.read_bytes()
        rows = list(csv.reader(payload.decode("utf-8").splitlines()))
        self.assertEqual(rows[0], ["time_s", "q/关节", "q/b", "tau/关节", "tau/b"])
        self.assertEqual(
            [[float(value) for value in row[1:]] for row in rows[1:]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        times = [float(row[0]) for row in rows[1:]]
        self.assertTrue(
            all(left < right for left, right in zip(times, times[1:])))
        metadata = self.metadata()
        self.assertEqual(metadata["dataset_sha256"],
                         hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata["sample_count"], 3)
        self.assertEqual(metadata["rejected_count"], 1)
        self.assertEqual(metadata["joint_names"], ["关节", "b"])
        self.assertEqual(metadata["session_id"], "run-1")
        self.assertEqual(metadata["input_topic"], "/arm/joints")
        self.assertEqual(metadata["model_digest"], "model-hash")
        self.assertEqual(metadata["torque_source"], "calibrated")
        self.assertEqual(list(self.path.parent.glob("*.partial")), [])
        with self.assertRaises(RuntimeError):
            session.append(["关节", "b"], [0, 0], [0, 0])

    def test_rejected_first_samples_do_not_commit_header_or_joint_set(self):
        session = self.session()
        self.assertIsNotNone(session.append(["bad"], [float("inf")], [1]))
        self.clock.side_effect = iter([-1, 1000000, 2000000])
        self.assertIsNotNone(session.append(["old"], [1], [2]))
        self.assertIsNone(session.append(["new"], [3], [4]))
        self.assertIsNone(session.append(["new"], [5], [6]))
        self.assertTrue(session.close())
        self.assertEqual(self.path.read_text().count("time_s"), 1)
        self.assertTrue(
            self.path.read_text().startswith("time_s,q/new,tau/new\n"))
        self.assertEqual(self.metadata()["rejected_count"], 2)

    def test_rejects_invalid_and_changed_names_after_cached_order(self):
        session = self.session()
        self.assertIsNone(session.append(["a", "b"], [1, 2], [3, 4]))
        for names, position, effort in (([], [], []), (["a", "b"], [1], [1, 2]),
                                        (["a", "b"], [1, 2],
                                         [1]), (["a", "a"], [1, 2], [3, 4]),
                                        (["a", "c"], [1, 2],
                                         [3, 4]), (["a", ""], [1, 2], [3, 4]),
                                        (["a", "b\n"], [1, 2],
                                         [3,
                                          4]), (["a",
                                                 "b"], [1,
                                                        float("nan")], [3, 4])):
            with self.subTest(names=names, position=position):
                self.assertIsNotNone(session.append(names, position, effort))
        self.assertIsNone(session.append(["b", "a"], [5, 6], [7, 8]))
        self.assertTrue(session.close())
        self.assertEqual(self.metadata()["sample_count"], 2)
        self.assertEqual(self.metadata()["rejected_count"], 8)

    def test_rejects_duplicate_regressing_and_rounded_timestamps(self):
        session = self.session()
        # Beyond 2**53 ns, different clock ticks can format as the same double.
        self.clock.side_effect = iter(
            [100, 100, 99, 2**54, 2**54 + 1, 2**54 + 16])
        accepted = [session.append(["a"], [1], [2]) is None for _ in range(6)]
        self.assertEqual(accepted, [True, False, False, True, False, True])
        self.assertTrue(session.close())
        self.assertEqual(self.metadata()["sample_count"], 3)

    def test_malformed_sample_fields_are_rejected_without_losing_session(self):
        session = self.session(joint_names=["a", "b"])
        self.assertIsNone(session.append(["a", "b"], [1, 2], [3, 4]))
        for names, position, effort in ((["a", []], [1, 2], [3, 4]), ([
                "a", None
        ], [1, 2], [3, 4]), ("ab", [1, 2], [3, 4]), (["a", "b"], ["invalid",
                                                                  2], [3, 4]),
                                        (["a", "b"], [1, 2], [10**400, 4]),
                                        (["a", "b"], [1, 2], [None, 4])):
            with self.subTest(names=names, position=position, effort=effort):
                self.assertIsNotNone(session.append(names, position, effort))
                self.assertEqual(session.sample_count, 1)
        self.assertIsNone(session.append(["b", "a"], [5, 6], [7, 8]))
        self.assertTrue(session.close())
        self.assertEqual(self.metadata()["sample_count"], 2)
        self.assertEqual(self.metadata()["rejected_count"], 6)
        rows = list(csv.reader(self.path.read_text().splitlines()))
        self.assertEqual([float(value) for value in rows[-1][1:]], [6, 5, 8, 7])

    def test_zero_samples_retain_only_empty_partial(self):
        session = self.session()
        self.assertFalse(session.close())
        self.assertFalse(session.close())
        self.assertTrue(session._stream.closed)
        self.assertEqual(session.partial_path.read_bytes(), b"")
        self.assertFalse(self.path.exists())
        self.assertFalse(Path(str(self.path) + ".metadata.json").exists())

    def test_invalid_configuration_creates_no_artifacts(self):
        for options in ({
                "flush_interval": True
        }, {
                "flush_interval": 0
        }, {
                "flush_interval": 1.5
        }, {
                "input_topic": ""
        }, {
                "session_id": None
        }, {
                "model_digest": 12
        }, {
                "torque_source": ""
        }, {
                "joint_names": "abc"
        }, {
                "joint_names": ["a", "a"]
        }, {
                "joint_names": ["a,b"]
        }, {
                "joint_names": [None]
        }, {
                "joint_names": [""]
        }):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.session(**options)
                self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_refuses_existing_files_and_dangling_symlinks(self):
        for suffix in ("", ".partial", ".metadata.json",
                       ".metadata.json.partial"):
            for symlink in (False, True):
                with self.subTest(suffix=suffix, symlink=symlink):
                    artifact = Path(str(self.path) + suffix)
                    target = self.path.parent / "missing-target"
                    if symlink:
                        artifact.symlink_to(target)
                    else:
                        artifact.write_bytes(b"existing")
                    try:
                        with self.assertRaises(FileExistsError):
                            self.session()
                        if symlink:
                            self.assertTrue(artifact.is_symlink())
                            self.assertFalse(target.exists())
                        else:
                            self.assertEqual(artifact.read_bytes(), b"existing")
                    finally:
                        artifact.unlink()

    def test_second_writer_cannot_open_active_session(self):
        session = self.session()
        with self.assertRaises(FileExistsError):
            self.session()
        self.assertIsNone(session.append(["a"], [1], [2]))
        self.assertTrue(session.close())

    def test_artifacts_created_during_recording_are_never_overwritten(self):
        for suffix in ("", ".metadata.json", ".metadata.json.partial"):
            with self.subTest(
                    suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "session.csv"
                session = recording.RawJointStateSession(str(path))
                session.append(["a"], [1], [2])
                artifact = Path(str(path) + suffix)
                artifact.write_bytes(b"other writer")
                with self.assertRaises(FileExistsError):
                    session.close()
                self.assertEqual(artifact.read_bytes(), b"other writer")
                self.assertTrue(session.partial_path.exists())
                self.assertTrue(session._stream.closed)
                self.assertFalse(session.close())

    def test_publication_races_do_not_replace_files_or_symlinks(self):
        original_link = os.link
        for suffix in ("", ".metadata.json"):
            for symlink in (False, True):
                with self.subTest(suffix=suffix,
                                  symlink=symlink), tempfile.TemporaryDirectory(
                                  ) as directory:
                    path = Path(directory) / "session.csv"
                    session = recording.RawJointStateSession(str(path))
                    session.append(["a"], [1], [2])
                    artifact = Path(str(path) + suffix)
                    target = Path(directory) / "missing-target"

                    def racing_link(source, destination):
                        if destination == artifact:
                            if symlink:
                                artifact.symlink_to(target)
                            else:
                                artifact.write_bytes(b"racing writer")
                        return original_link(source, destination)

                    with mock.patch.object(recording.os, "link", racing_link):
                        with self.assertRaises(FileExistsError):
                            session.close()
                    if symlink:
                        self.assertTrue(artifact.is_symlink())
                        self.assertFalse(target.exists())
                    else:
                        self.assertEqual(artifact.read_bytes(),
                                         b"racing writer")
                    self.assertTrue(session.partial_path.exists())
                    self.assertTrue(
                        Path(str(path) + ".metadata.json.partial").exists())
                    self.assertFalse(session.close())

    def test_partial_and_interrupted_writes_never_finalize_previous_samples(
            self):
        for failure in (OSError("disk full"), KeyboardInterrupt(), None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(
            ) as directory:
                path = Path(directory) / "session.csv"
                session = recording.RawJointStateSession(str(path))
                session.append(["a"], [1], [2])
                stream = session._stream
                session._stream = mock.Mock(wraps=stream)

                def broken_write(payload):
                    stream.write(payload[:3])
                    if failure is not None:
                        raise failure
                    return 3

                session._stream.write.side_effect = broken_write
                with self.assertRaises(
                        type(failure) if failure is not None else OSError):
                    session.append(["a"], [3], [4])
                with self.assertRaises(RuntimeError):
                    session.append(["a"], [5], [6])
                self.assertFalse(session.close())
                self.assertTrue(stream.closed)
                self.assertTrue(session.partial_path.exists())
                self.assertFalse(path.exists())
                self.assertFalse(Path(str(path) + ".metadata.json").exists())

    def test_flush_and_data_fsync_failures_close_stream_without_publishing(
            self):
        for failure_at in ("append_flush", "close_flush", "fsync", "close"):
            with self.subTest(failure_at=failure_at
                             ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "session.csv"
                session = recording.RawJointStateSession(str(path),
                                                         flush_interval=1)
                session.append(["a"], [1], [2])
                stream = session._stream
                session._stream = mock.Mock(wraps=stream)
                if failure_at in ("append_flush", "close_flush"):
                    session._stream.flush.side_effect = OSError("flush failed")
                if failure_at == "append_flush":
                    with self.assertRaisesRegex(OSError, "flush failed"):
                        session.append(["a"], [3], [4])
                    self.assertFalse(session.close())
                elif failure_at == "fsync":
                    with mock.patch.object(recording.os,
                                           "fsync",
                                           side_effect=OSError("sync failed")):
                        with self.assertRaisesRegex(OSError, "sync failed"):
                            session.close()
                else:
                    if failure_at == "close":

                        def broken_close():
                            stream.close()
                            raise OSError("close failed")

                        session._stream.close.side_effect = broken_close
                    with self.assertRaises(OSError):
                        session.close()
                self.assertTrue(stream.closed)
                self.assertTrue(session.partial_path.exists())
                self.assertFalse(path.exists())
                self.assertFalse(session.close())

    def test_metadata_sync_failure_publishes_neither_artifact(self):
        session = self.session()
        session.append(["a"], [1], [2])
        with mock.patch.object(
                recording.os,
                "fsync",
                side_effect=[None, OSError("metadata sync failed")]):
            with self.assertRaisesRegex(OSError, "metadata sync failed"):
                session.close()
        self.assertTrue(session._stream.closed)
        self.assertTrue(session.partial_path.exists())
        self.assertTrue(
            Path(str(self.path) + ".metadata.json.partial").exists())
        self.assertFalse(self.path.exists())
        self.assertFalse(Path(str(self.path) + ".metadata.json").exists())

    def test_directory_sync_failure_retains_recoverable_artifacts(self):
        session = self.session()
        session.append(["a"], [1], [2])
        with mock.patch.object(
                recording.os,
                "fsync",
                side_effect=[None, None,
                             OSError("directory sync failed")]):
            with self.assertRaisesRegex(OSError, "directory sync failed"):
                session.close()
        self.assertTrue(session.partial_path.exists())
        self.assertTrue(
            Path(str(self.path) + ".metadata.json.partial").exists())
        self.assertEqual(self.path.read_bytes(),
                         session.partial_path.read_bytes())
        self.assertFalse(session.close())


if __name__ == "__main__":
    unittest.main()
