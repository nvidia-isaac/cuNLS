# Sparse Bundle Adjustment Example

This example demonstrates synthetic sparse bundle adjustment with:
- `ReprojectionFactorBatch` for reprojection residuals
- `SE3StateBatch` for camera poses (first pose fixed, rest optimized)
- `VectorStateBatch<3>` for 3D points (all optimized)
- `LevenbergMarquardtMinimizer` for nonlinear optimization

The setup is intentionally compact but complete:
1. Generate synthetic camera poses in SE(3).
2. Generate random 3D points visible from every camera.
3. Project points into each camera to produce normalized 2D observations.
4. Perturb poses (except the first, which serves as the gauge anchor) and points.
5. Jointly optimize poses and points to recover the original geometry.

## Files

- `main.cpp`: full end-to-end example.
- `../utils/`: shared host-side utilities (SE(3) math, projection, validation).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## How the code is structured

### Data generation

`GenerateRandomPoses()` creates random twists with a positive tz bias and
converts them into SE(3) transforms via `examples::TwistsToSE3`.

`examples::ProjectNormalized()` (from `utils/camera_utils.h`) computes
observations in normalized camera coordinates:
- transform `P_world` to camera frame with `T_cam_from_world`
- divide by depth to get `(x/z, y/z)`

`ReprojectionFactorBatch` expects these normalized coordinates.

### Problem construction

- Poses are stored in `SE3StateBatch`. Only the first pose is marked constant
  via `const_pose_ids` (gauge anchor); all other poses are optimized.
- Points are stored in `VectorStateBatch<3>` and are optimized.
- State pointers are created with layout:
  `[pose_0, point_0, pose_0, point_1, ..., pose_M, point_N]`.

This layout matches `ReprojectionFactorBatch` requirements: each factor consumes
two state blocks `(pose, point)`.

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
