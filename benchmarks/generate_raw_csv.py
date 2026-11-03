"""Generate a deterministic raw or processed CSV for reader measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--dof", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("raw", "processed"), default="raw")
    args = parser.parse_args()
    if args.samples < 3 or args.dof < 1:
        parser.error("--samples must be at least 3 and --dof must be positive")
    digest = hashlib.sha256()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:

            def write(line):
                payload = line.encode("utf-8")
                if output.write(payload) != len(payload):
                    raise OSError("incomplete synthetic fixture write")
                digest.update(payload)

            prefixes = ("q", "tau") if args.format == "raw" else ("q", "dq",
                                                                  "ddq", "tau")
            header = ["time_s"] + [
                f"{prefix}/j{joint}" for prefix in prefixes
                for joint in range(args.dof)
            ]
            write(",".join(header) + "\n")
            for row in range(args.samples):
                position = [
                    math.sin(row * .003 + joint * .2)
                    for joint in range(args.dof)
                ]
                torque = [
                    .5 * math.cos(row * .007 - joint * .3)
                    for joint in range(args.dof)
                ]
                if args.format == "raw":
                    values = [row * .001] + position + torque
                else:
                    velocity = [
                        3. * math.cos(row * .003 + joint * .2)
                        for joint in range(args.dof)
                    ]
                    acceleration = [-9. * value for value in position]
                    values = [row * .001
                             ] + position + velocity + acceleration + torque
                write(",".join(format(value, ".17g") for value in values) +
                      "\n")
        print(
            json.dumps(dict(format=args.format,
                            samples=args.samples,
                            dof=args.dof,
                            bytes=args.output.stat().st_size,
                            sha256=digest.hexdigest()),
                       sort_keys=True))
    except OSError as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
