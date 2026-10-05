# cuNLS Examples

This folder contains standalone examples that link against a prebuilt and
installed `cuNLS` library.

Each example lives in its own subdirectory and provides:
- source code
- a `README.md` with a code walkthrough

Each `main` reads top to bottom as the cuNLS workflow: get data, upload it,
build state and factor batches, assemble the `Problem`, solve, read back, and
report. Everything else lives in header-only helpers in `utils/`:
- `datasets.h`: synthetic scenes (PnP with optional outliers, bundle
  adjustment, pose chain, constant-velocity trajectory, scalar chain);
- `validation.h`: error metrics (MSE, rotation / translation error,
  inlier-mask statistics);
- `report.h`: printing helpers and the final quality verdict;
- `cli.h`: `--key value` command-line parsing;
- `se3_utils.h`, `camera_utils.h`: host SE(3) math and camera projection.

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
- `sparse_bundle_adjustment`: Uses `ReprojectionFactorBatch` to jointly optimize
  camera poses and 3D landmarks from synthetic observations (first two poses
  fixed as gauge anchors for frame and scale; remaining poses and all points
  optimized).
- `pose_graph_optimization`: Uses `BetweenFactorBatch` (manifold deduced via
  CTAD, here SE(3)) to optimize a chain of poses from consecutive
  relative-transform measurements, with the first pose fixed as a gauge
  anchor.
- `custom_factor`: Implements a simple user-defined scalar difference factor
  and combines it with `PriorFactorBatch<manifold::Vector<1>>` to anchor the
  solution.
- `motion_prior`: Uses `ConstantVelocityInformationSE3FactorBatch` (the
  constant-velocity motion-prior factor with the closed-form process-noise
  covariance fused in) to optimize a chain of SE(3) poses and body-velocity
  states, with the first pose and velocity fixed as gauge anchors.
- `pnp`: Uses `PnPFactorBatch` to recover a camera pose from 3D-2D
  correspondences, with analytic and numeric Jacobians.
- `ransac_pnp`: The same PnP problem with 50% gross outliers, solved with
  `RansacLevenbergMarquardtMinimizer`; compares against plain
  Levenberg-Marquardt and reports the inlier mask.

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
