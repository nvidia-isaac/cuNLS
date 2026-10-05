# Sparse Bundle Adjustment Example

This example demonstrates synthetic sparse bundle adjustment with:
- `ReprojectionFactorBatch` for reprojection residuals
- `SE3StateBatch` for camera poses (first two fixed, rest optimized)
- `VectorStateBatch<3>` for 3D points (all optimized)
- `LevenbergMarquardtMinimizer` for nonlinear optimization

The setup is intentionally compact but complete:
1. Generate synthetic camera poses in SE(3).
2. Generate random 3D points visible from every camera.
3. Project points into each camera to produce normalized 2D observations.
4. Perturb poses (except the first two, the gauge anchors) and points.
5. Jointly optimize poses and points to recover the original geometry.

## Files

- `main.cpp`: the cuNLS workflow, step by step (data, upload, state and factor
  batches, problem, solve, read back, report).
- `../utils/`: header-only helpers kept out of `main` so it reads as the cuNLS
  workflow: `datasets.h` (synthetic scenes), `validation.h` (error metrics),
  `report.h` (printing and the quality verdict), `cli.h` (command-line flags),
  `se3_utils.h` / `camera_utils.h` (host SE(3) math and projection).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## How the code is structured

### Data generation

`examples::MakeBundleAdjustmentScene()` (`utils/datasets.h`) creates cameras
looking at the origin, points visible from every camera, a perturbed initial
guess (cameras 0 and 1 kept exact as the gauge anchors), and the observations.

`examples::ProjectNormalized()` (from `utils/camera_utils.h`) computes
observations in normalized camera coordinates:
- transform `P_world` to camera frame with the inverse of the camera pose
  state `T_world_from_cam`
- divide by depth to get `(x/z, y/z)`

`ReprojectionFactorBatch` expects these normalized coordinates.

### Problem construction

- Poses are stored in `SE3StateBatch` as the cameras' world poses
  (world_from_camera, the pose convention of cuNLS). The first two poses are
  marked constant via `const_pose_ids`: camera 0 fixes the frame and camera 1
  the scale, which reprojections alone do not observe; all other poses are
  optimized.
- Points are stored in `VectorStateBatch<3>` and are optimized.
- State pointers are created with layout:
  `[pose_0, point_0, pose_0, point_1, ..., pose_M, point_N]`.

This layout matches `ReprojectionFactorBatch` requirements: each factor consumes
two states `(pose, point)`.

Each batch is constructed with its **capacity** (how many states /
observations its bound device buffers hold, fixed for the batch's lifetime)
and starts with 0 active; `SetNumActiveStates` / `SetNumActiveFactors` set the
**active count** the next solve uses (host-only: no allocation, no device
work; a solve without it throws). Size the capacity once for the largest
problem you expect; the active count may change between solves up to it, so
a real-time application allocates once and reuses the same buffers every
frame while the problem size changes. The example keeps the two in separate
variables (`*_capacity` vs. `num_*`); it solves every slot once, so each
active count equals its capacity.

`LevenbergMarquardtMinimizer::Minimize` runs `Initialize`, which allocates
minimizer working buffers and prepares state/factor batches for the solve.

### Optimization and checks

The example runs LM and prints:
- initial/final cost
- iteration count
- point MSE before and after optimization
- pose MSE before and after optimization

It returns a non-zero exit code if the final quality checks fail.

## Build locally (all examples)

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

## Build inside Docker and export binaries

```bash
./examples/build_in_docker.sh Release ./artifacts/examples
```

Output binary:
- `./artifacts/examples/sparse_bundle_adjustment_example`
