# Contributing

- Keep the C++ core independent of ROS, Pinocchio, controllers, and vendor SDKs.
- Preserve named joint ordering, arbitrary DoF, units, finite-input checks, and immutable artifact contracts.
- Change model integration through `RegressorBackend`; recording uses standard ROS messages.
- Update package versions, migration notes, schemas, and installed consumers when a public contract changes.
- Never commit build outputs, private robot recordings, credentials, or hardware configuration.
- Run `./scripts/test.sh`, `./check_style.sh`, and `git diff --check`; use `--with-pinocchio` for backend changes.
- Formatting uses clang-format 18.1.8 and YAPF 0.43.0, pinned in `requirements-dev.txt`.
- Maintain both root READMEs. Algorithm changes must describe equations, units, saturation, and invalid-input behavior.
