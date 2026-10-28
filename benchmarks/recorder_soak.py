"""Exercise the installed recorder with reordered and deliberately invalid data."""
import argparse
import csv
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def memory(pid):
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values


def run(args):
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from sensor_msgs.msg import JointState
    from robot_identification_tools.dataset_inspect import inspect_dataset_with_digest

    tag = uuid.uuid4().hex[:12]
    topic = f"/identification_load_{tag}/joint_states"
    names = [f"joint_{i:02}" for i in range(14)]
    executable = (Path(get_package_prefix("robot_identification_ros2")) /
                  "lib/robot_identification_ros2/joint_state_recorder")
    dataset = args.output.with_suffix(".csv")
    report = dict(status="running",
                  requested_duration_s=args.duration,
                  requested_rate_hz=args.rate,
                  qos_depth=args.qos_depth,
                  dof=len(names),
                  minimum_delivery_fraction=args.minimum_delivery_fraction,
                  samples=[],
                  published=0,
                  invalid_published=0)
    rclpy.init()
    node = None
    process = None
    try:
        node = rclpy.create_node("identification_load_" + tag)
        publisher = node.create_publisher(JointState, topic, 100)
        with args.output.with_suffix(".recorder.log").open("w") as log:
            process = subprocess.Popen([
                str(executable), "--ros-args", "-p", f"input_topic:={topic}",
                "-p", f"output_path:={dataset}", "-p", "flush_interval:=1000",
                "-p", f"qos_depth:={args.qos_depth}", "-p",
                "joint_names:=[" + ",".join(names) + "]"
            ],
                                       stdout=log,
                                       stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            while publisher.get_subscription_count() == 0:
                require(process.poll() is None,
                        "recorder exited before discovery")
                if time.monotonic() > deadline:
                    raise TimeoutError("recorder discovery timeout")
                rclpy.spin_once(node, timeout_sec=.05)
            started = time.monotonic()
            next_heartbeat = 0.
            sent = invalid = 0
            while time.monotonic() - started < args.duration:
                require(process.poll() is None,
                        "recorder exited during publication")
                order = list(range(14)) if (sent // 64) % 2 == 0 else list(
                    reversed(range(14)))
                message = JointState()
                message.name = [names[i] for i in order]
                message.position = [sent * .001 + i * .25 for i in order]
                message.effort = [-2 * value for value in message.position]
                if sent and sent % 1009 == 0:
                    kind = (sent // 1009) % 3
                    if kind == 0:
                        message.position[0] = math.nan
                    elif kind == 1:
                        message.effort.pop()
                    else:
                        message.name[0] = message.name[1]
                    invalid += 1
                publisher.publish(message)
                sent += 1
                elapsed = time.monotonic() - started
                if elapsed >= next_heartbeat:
                    sample = memory(process.pid)
                    partial = Path(str(dataset) + ".partial")
                    sample.update(elapsed_s=elapsed,
                                  published=sent,
                                  invalid_published=invalid,
                                  csv_bytes=partial.stat().st_size
                                  if partial.exists() else 0)
                    report["samples"].append(sample)
                    report.update(published=sent, invalid_published=invalid)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(sample), flush=True)
                    next_heartbeat = elapsed + 60
                remaining = started + sent / args.rate - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            publication_elapsed = time.monotonic() - started
            time.sleep(2)
            process.send_signal(signal.SIGINT)
            require(process.wait(timeout=20) == 0, "recorder shutdown failed")
        report.update(published=sent,
                      invalid_published=invalid,
                      publication_elapsed_s=publication_elapsed)
        summary, digest = inspect_dataset_with_digest(dataset)
        metadata = json.loads(Path(str(dataset) + ".metadata.json").read_text())
        require(metadata["dataset_sha256"] == digest, "dataset digest mismatch")
        require(metadata["sample_count"] == summary["sample_count"],
                "metadata count mismatch")
        accepted, rejected = metadata["sample_count"], metadata[
            "rejected_count"]
        require(accepted <= sent - invalid and rejected <= invalid,
                "unexpected duplicate counts")
        report.update(undelivered_valid=sent - invalid - accepted,
                      undelivered_invalid=invalid - rejected)
        require(accepted >= args.minimum_delivery_fraction * (sent - invalid),
                "valid-message delivery below requested fraction")
        require(rejected >= args.minimum_delivery_fraction * invalid,
                "invalid-message delivery below requested fraction")
        with dataset.open(newline="") as stream:
            reader = csv.reader(stream)
            require(
                next(reader) == ["time_s"] + ["q/" + name for name in names] +
                ["tau/" + name for name in names], "CSV joint order mismatch")
            previous = -1.
            verified = 0
            for row in reader:
                values = list(map(float, row))
                base = values[1]
                require(base > previous, "out-of-order or duplicate sample")
                previous = base
                for joint in range(14):
                    require(
                        math.isclose(values[1 + joint],
                                     base + joint * .25,
                                     rel_tol=0,
                                     abs_tol=1e-10),
                        "position reordering mismatch")
                    require(values[15 + joint] == -2 * values[1 + joint],
                            "torque mismatch")
                verified += 1
        require(verified == accepted, "verified row count mismatch")
        report.update(status="passed",
                      metadata=metadata,
                      summary=summary,
                      verified_rows=verified,
                      exit_code=process.returncode)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--rate", type=int, default=1000)
    parser.add_argument("--qos-depth", type=int, default=10)
    parser.add_argument("--minimum-delivery-fraction", type=float, default=.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 86400:
        parser.error(
            "--duration must be finite and between 1 and 86400 seconds")
    if not 1 <= args.rate <= 1000000 or not 1 <= args.qos_depth <= 1000000:
        parser.error("--rate and --qos-depth must be between 1 and 1000000")
    if not math.isfinite(args.minimum_delivery_fraction
                        ) or not 0 < args.minimum_delivery_fraction <= 1:
        parser.error("--minimum-delivery-fraction must be in (0, 1]")
    args.output = args.output.resolve()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    dataset = args.output.with_suffix(".csv")
    artifacts = [
        args.output,
        args.output.with_suffix(".recorder.log"), dataset,
        Path(str(dataset) + ".partial"),
        Path(str(dataset) + ".metadata.json")
    ]
    if any(path.exists() for path in artifacts):
        parser.error(
            "choose a new output path; validation artifacts already exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
