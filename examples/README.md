# cuNLS Examples

This folder contains standalone examples that link against a prebuilt and
installed `cuNLS` library.

Each example lives in its own subdirectory and provides:
- source code
- a `README.md` with a code walkthrough

Shared host-side utilities (SE(3) math, camera projection, validation metrics)
live in `utils/` and are included by all examples as header-only helpers.

Build orchestration is centralized:
- `examples/CMakeLists.txt`: single CMake entrypoint that defines all example
  binaries.
- `examples/build_in_docker.sh`: single Docker script that builds and exports
  all binaries in one run.

The Docker build script now:
- builds `cuNLS` fully inside the container
- installs `cuNLS` directly into the mounted output folder (`include/` + `lib/`)
- builds all examples against that installation
- writes example binaries into the same output folder
- generates `run_all_examples.sh` to execute all examples sequentially

Available examples:
- `sparse_bundle_adjustment`: Uses `ReprojectionFactorBatch` to optimize 3D
  points from synthetic camera observations while keeping camera poses fixed.
- `pose_graph_optimization`: Uses `SE3BetweenFactorBatch` to optimize a chain
  of SE(3) poses from consecutive relative-transform measurements, with the
  first pose fixed as a gauge anchor.
- `custom_factor`: Implements a simple user-defined scalar difference factor
  and combines it with `PriorVectorFactorBatch<1>` to anchor the solution.

## Build all examples locally

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

## Build all examples in Docker and export binaries

```bash
./examples/build_in_docker.sh Release ./artifacts/examples
```

After completion, run:

```bash
./artifacts/examples/run_all_examples.sh
```
