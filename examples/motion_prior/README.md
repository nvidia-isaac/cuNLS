# Motion Prior Example

This example demonstrates a constant-velocity motion-prior chain with:
- `ConstantVelocityInformationSE3FactorBatch` for consecutive pose/velocity
  constraints, with the paper's closed-form process-noise covariance
  `Q(dt)^-1` fused directly into the residual/Jacobian
- `SE3StateBatch` for pose variables and `VectorStateBatch<6>` for body-twist
  velocity variables
- `LevenbergMarquardtMinimizer` for solving the nonlinear least-squares system

It builds a synthetic **pose + velocity chain**:
- Poses `T_0, T_1, ..., T_{N-1}` and body velocities `v_0, v_1, ..., v_{N-1}`
- One `ConstantVelocityInformationSE3FactorBatch` factor per consecutive pair
  `((T_i, T_{i+1}), (v_i, v_{i+1}))`, weighted by a continuous-time
  process-noise PSD `Qc` (one entry per SE(3) tangent DOF)
- Both `T_0` and `v_0` are held fixed as gauge anchors
- All remaining poses and velocities are optimized

See `docs/sphinx/api/factor.rst` (Motion prior factors / Motion prior
covariance weighting sections) for the API reference.

## Files

- `main.cpp`: the cuNLS workflow, step by step; the trajectory comes from
  `examples::MakeConstantVelocityScene`.
- `../utils/`: header-only helpers kept out of `main` so it reads as the cuNLS
  workflow: `datasets.h` (synthetic scenes), `validation.h` (error metrics),
  `report.h` (printing and the quality verdict), `cli.h` (command-line flags),
  `se3_utils.h` / `camera_utils.h` (host SE(3) math and projection).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## How the factor is used

The underlying `ConstantVelocitySE3FactorBatch` computes residuals with the
convention (`twist := Log(T_i^{-1} * T_{i+1})`, `Jl_inv := J_l^{-1}(twist)`):

```
r_pose = twist - dt * v_i
r_vel  = Jl_inv * v_{i+1} - v_i
```

For zero residual, consecutive states should satisfy:

```
T_{i+1} = T_i * Exp(dt * v_i)
v_{i+1} = J_l(dt * v_i) * v_i
```

`ConstantVelocityInformationSE3FactorBatch` wraps this residual/Jacobian with
the closed-form sqrt-information `S(dt)` derived from `Qc`, so the effective
weighted residual is `S(dt) * [r_pose; r_vel]` — callers only provide `dt`
and `Qc`, never a sqrt-information matrix directly.

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
5. Build one `ConstantVelocityInformationSE3FactorBatch` factor per
   consecutive pair, from per-factor `dt` and a shared `Qc` diagonal.

Each batch is constructed with its **capacity** (how many blocks / factors
its bound device buffers hold, fixed for the batch's lifetime) and starts
with 0 active; `SetNumStateBlocks` / `SetNumFactors` set the **active
count** the next solve uses (host-only: no allocation, no device work; a
solve without it throws). Size the capacity once for the largest problem you
expect; the active count may change between solves up to it, so a real-time
application allocates once and reuses the same buffers every frame while the
problem size changes. The example keeps the two in separate variables
(`*_capacity` vs. `num_*`); it solves every slot once, so each active count
equals its capacity.

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
