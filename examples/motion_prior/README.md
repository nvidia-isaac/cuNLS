# Motion Prior Example

This example demonstrates a constant-velocity motion-prior chain with:
- `ConstantVelocitySE3FactorBatch` for consecutive pose/velocity constraints
- `SE3StateBatch` for pose variables and `VectorStateBatch<6>` for body-twist
  velocity variables
- `LevenbergMarquardtMinimizer` for solving the nonlinear least-squares system

It builds a synthetic **pose + velocity chain**:
- Poses `T_0, T_1, ..., T_{N-1}` and body velocities `v_0, v_1, ..., v_{N-1}`
- One `ConstantVelocitySE3FactorBatch` factor per consecutive pair
  `((T_i, T_{i+1}), (v_i, v_{i+1}))`
- Both `T_0` and `v_0` are held fixed as gauge anchors
- All remaining poses and velocities are optimized

See `docs/design/motion_prior_factors.md` for the full derivation and
`docs/sphinx/api/factor.rst` (Motion prior factors section) for the API
reference.

## Files

- `main.cpp`: complete synthetic motion-prior pipeline.
- `../utils/`: shared host-side utilities (SE(3) math, validation).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## How the factor is used

`ConstantVelocitySE3FactorBatch` computes residuals with the convention
(`twist := Log(T_i^{-1} * T_{i+1})`, `Jl_inv := J_l^{-1}(twist)`):

```
r_pose = twist - dt * v_i
r_vel  = Jl_inv * v_{i+1} - v_i
```

For zero residual, consecutive states should satisfy:

```
T_{i+1} = T_i * Exp(dt * v_i)
v_{i+1} = J_l(dt * v_i) * v_i
```

The example constructs the ground-truth chain exactly this way (integrating
forward from a random anchor pose and velocity), then disturbs every pose
and velocity except the fixed anchor to create the initial estimate.

## Problem setup

1. Generate a random anchor pose `T_0` and body velocity `v_0`.
2. Integrate the constant-velocity model forward to build the ground-truth
   chain (`T_{i+1}`, `v_{i+1}` from `T_i`, `v_i`, and `dt`).
3. Disturb all poses and velocities except the anchor.
4. Add an `SE3StateBatch` (poses) and a `VectorStateBatch<6>` (velocities),
   both with index `0` marked constant.
5. Build one `ConstantVelocitySE3FactorBatch` factor per consecutive pair.

Fixing both `T_0` and `v_0` (rather than just `T_0`, as in
`pose_graph_optimization`) is required here: with only relative
pose-velocity constraints, one full gauge degree of freedom (12 DOF) would
otherwise remain unconstrained.

## Validation

After optimization, the example prints:
- initial/final solver cost
- iteration count
- pose MSE and velocity MSE against the ground-truth chain

The binary returns non-zero if convergence quality is poor.

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
- `./artifacts/examples/motion_prior_example`
