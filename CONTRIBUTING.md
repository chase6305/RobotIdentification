# Contributing

- Keep the C++ core independent of ROS, Pinocchio, controllers, and vendor SDKs.
- Preserve named joint ordering, arbitrary DoF, units, finite-input checks, and immutable artifact contracts.
- Change model integration through `RegressorBackend`; recording uses standard ROS messages.
- Update package versions, migration notes, schemas, and installed consumers when a public contract changes.
- Never commit build outputs, private robot recordings, credentials, or hardware configuration.
- Run `./scripts/test.sh`, `./check_style.sh`, and `git diff --check`; use `--with-pinocchio` for backend changes.
- Validation builds and style checks default to at most two workers; ROS packages build and
  test sequentially so package and compiler pools do not multiply. Worker
  counts must be positive integers; zero must not enable an unbounded build.
- Formatting uses clang-format 18.1.8 and YAPF 0.43.0, pinned in `requirements-dev.txt`.
- Standalone CI initializes Eigen storage with NaNs to expose unfilled inputs;
  reproduce with `CXXFLAGS=-DEIGEN_INITIALIZE_MATRICES_BY_NAN ./scripts/test.sh <fresh-build-root>`.
- Maintain both root READMEs. Algorithm changes must describe equations, units, saturation, and invalid-input behavior.
