"""Generate a deterministic synthetic raw CSV for offline reader measurements."""
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

            header = (["time_s"] +
                      [f"q/j{joint}" for joint in range(args.dof)] +
                      [f"tau/j{joint}" for joint in range(args.dof)])
            write(",".join(header) + "\n")
            for row in range(args.samples):
                values = ([row * .001] + [
                    math.sin(row * .003 + joint * .2)
                    for joint in range(args.dof)
                ] + [
                    .5 * math.cos(row * .007 - joint * .3)
                    for joint in range(args.dof)
                ])
                write(",".join(format(value, ".17g") for value in values) +
                      "\n")
        print(
            json.dumps(dict(samples=args.samples,
                            dof=args.dof,
                            bytes=args.output.stat().st_size,
                            sha256=digest.hexdigest()),
                       sort_keys=True))
    except OSError as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
