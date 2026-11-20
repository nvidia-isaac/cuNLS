# Module: factor/motion

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Motion priors between consecutive trajectory steps: constant velocity or constant acceleration.

## Files

- `constant_velocity_<group>_factor_batch.h` and `constant_acceleration_<group>_factor_batch.h`
  for so2, so3, se2, se3; all eight are implemented in `motion_prior_factor_batches.cu` (one
  kernel `MotionPriorKernel<Group, Order>`, coalesced writes, no scratch state).
- Facades `constant_velocity_factor_batch.h` / `constant_acceleration_factor_batch.h`:
  `ConstantVelocityFactorBatch<Manifold>` / `ConstantAccelerationFactorBatch<Manifold>`. The
  manifold is an **explicit** template argument: the constructor takes only time steps
  (`dt_ptr`), which carry nothing to deduce from (a deliberate choice).

## Conventions

- States per factor: pose and body velocity (and acceleration) at step k and k+1; the velocity
  is a vector state in the group's tangent layout (SE3: `[angular; linear]`).
- Residuals use `twist = Log(pose_k^-1 * pose_{k+1})`; per-factor `dt`.
- Weighting by the process-noise model: `cunls/factor/information/AGENTS.md`
  (`MotionPriorInformationFactorBatch`).
