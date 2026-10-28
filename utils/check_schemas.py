"""Validate identification schemas, fixtures, and review digest bindings."""

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def main():
    root = Path(__file__).resolve().parents[1]
    schema_dir = root / "robot_identification/schema"
    for path in sorted(schema_dir.glob("*.schema.json")):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    examples = schema_dir / "examples"
    fixtures = {}
    for name in ("session", "result", "review", "rollback"):
        instance = json.loads((examples / f"{name}.json").read_text())
        validator = Draft202012Validator(
            json.loads((schema_dir /
                        f"identification-{name}.schema.json").read_text()))
        validator.validate(instance)
        fixtures[name] = instance
    result = fixtures["result"]
    if (len(result["parameter_names"]) != len(result["parameters"]) or
            result["diagnostics"]["rank"] != len(result["parameters"])):
        raise ValueError("candidate fixture must match names, values, and rank")
    for field, artifact in (("candidate_sha256", "result"), ("rollback_sha256",
                                                             "rollback")):
        digest = hashlib.sha256(
            (examples / f"{artifact}.json").read_bytes()).hexdigest()
        if fixtures["review"][field] != digest:
            raise ValueError(f"review {field} does not bind {artifact}.json")
    print("Identification schemas and artifact bindings passed.")


if __name__ == "__main__":
    main()
