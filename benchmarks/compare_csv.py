"""Compare two CSV readers using fresh executable copies and paired runs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_result(output, expected_format):
    fields = output.split()
    if not fields or fields[0] != expected_format + "_csv":
        raise ValueError("benchmark did not report the requested CSV format")
    values = dict(field.split("=", 1) for field in fields[1:])
    elapsed = float(values["median_ms"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("benchmark reported an invalid elapsed time")
    identity = (int(values["samples"]), int(values["dof"]), values["checksum"])
    if min(identity[:2]) < 1 or not math.isfinite(float(identity[2])):
        raise ValueError("benchmark reported invalid dimensions or checksum")
    return values, elapsed, identity


def compare(args):
    fixture_digest = digest(args.dataset)
    records = []
    elapsed = {"before": [], "after": []}
    identities = set()
    binaries = {}
    with tempfile.TemporaryDirectory(
            prefix="identification-csv-compare-") as directory:
        temporary = Path(directory)
        for label, source in (("before", args.before), ("after", args.after)):
            snapshot = temporary / label
            shutil.copyfile(source, snapshot)
            snapshot.chmod(0o700)
            binaries[label] = dict(path=str(source.resolve()),
                                   sha256=digest(snapshot))
        executable = temporary / "csv_bench"
        for round_index in range(args.rounds):
            order = ("before", "after") if round_index % 2 == 0 else ("after",
                                                                      "before")
            for label in order:
                # Copy bytes rather than hardlinking an existing executable;
                # both variants run from the same fresh file and argv path.
                shutil.copyfile(temporary / label, executable)
                executable.chmod(0o700)
                command = (["taskset", "-c", str(args.cpu)]
                           if args.cpu is not None else [])
                command += [
                    str(executable),
                    str(args.dataset.resolve()),
                    str(args.repetitions)
                ]
                if args.format == "processed":
                    command.append("processed")
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(
                        f"{label} benchmark failed: {result.stderr.strip()}")
                values, milliseconds, identity = parse_result(
                    result.stdout, args.format)
                identities.add(identity)
                elapsed[label].append(milliseconds)
                records.append(
                    dict(variant=label, round=round_index + 1, values=values))
                print(
                    f"round={round_index + 1} variant={label} median_ms={milliseconds}",
                    file=sys.stderr,
                    flush=True)
    if len(identities) != 1:
        raise ValueError(
            "benchmark variants changed row count, joint count or checksum")
    if digest(args.dataset) != fixture_digest:
        raise ValueError("CSV fixture changed during the comparison")
    medians = {
        label: statistics.median(values) for label, values in elapsed.items()
    }
    reduction = (100. * (1. - medians["after"] / medians["before"])
                 if medians["before"] else None)
    return dict(binaries=binaries,
                dataset=dict(path=str(args.dataset.resolve()),
                             sha256=fixture_digest,
                             bytes=args.dataset.stat().st_size,
                             format=args.format),
                cpu=args.cpu,
                rounds=args.rounds,
                repetitions=args.repetitions,
                warmup_runs_per_process=int(args.repetitions > 1),
                median_ms=medians,
                reduction_percent=reduction,
                runs=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--format", choices=("raw", "processed"), default="raw")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 100 or not 1 <= args.repetitions <= 100:
        parser.error("--rounds and --repetitions must be between 1 and 100")
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        parser.error("--cpu must be in this process's allowed CPU set")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("--output must be a new path")
    try:
        payload = json.dumps(compare(args), indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
