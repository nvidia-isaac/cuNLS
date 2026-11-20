# Module: factor

Batched residual / Jacobian models. Each subfolder has its own `AGENTS.md`: `between/` (relative
poses, `BetweenFactorBatch<Manifold>`), `prior/` (`PriorFactorBatch<Manifold>`), `motion/`
(constant-velocity / acceleration priors), `information/` (sqrt-information weighting),
`clearance/` (obstacle constraints), `dynamics/` (vehicle / rigid-body shooting defects),
`kinematics/` (articulated robots, `LinkFrameFactorBatch`).

## Contracts every factor follows

- `FactorBatch` (`factor_batch.h`), `SizedFactorBatch` (`sized_factor_batch.h`). Inputs are device
  pointers that stay valid for the batch's lifetime.
- Capacity is fixed at construction; `NumActiveFactors()` starts at 0 and `SetNumActiveFactors(n)`
  (host) sets it before a solve, up to `Capacity()`. Custom factors pass the capacity to
  `SizedFactorBatch(capacity)`; wrappers forward the size calls.
- `Evaluate(residuals, jacobians, state_pointers, stream, factor_ids = nullptr,
  num_factor_ids = 0)` evaluates n items (n = num_factor_ids, or the active count): item t is
  factor `factor_ids[t]` (or t % active count) with its own pointer row
  `state_pointers[t * slots ..]`; outputs are item-major and bitwise equal to a plain evaluation
  (device helper `FactorMeasurementIndex` in each `.cu`; RANSAC relies on it).
- Poses (SE2 / SE3 / SO2 / SO3) are world_from_rig T, perturbed on the right (T Exp(δ)).
  Reprojection / PnP see a world point at E T^-1 P.
- Robust losses come through `ResidualBatch` (`robustifier/`).
- Vector factors (prior, between, bound, halfspace) are `Dynamic*` classes with the size set at
  run time (Python binds these); the `<Dim>` templates derive from them and add
  `sized_layout`, which `WeightedFactorBatch` / `InformationFactorBatch` read through the
  `HasSizedLayout` trait (`common/type_traits.h`) for a compile-time residual size.

## Factors in this directory

- Vision: `reprojection_factor_batch.h`, `pnp_factor_batch.h` (fixed points, pose-only Jacobian;
  one kernel for both in `reprojection_factor_batch.cu`).
- Registration: `point_to_point_`, `point_to_plane_`, `symmetric_point_to_plane_factor_batch.h`.
- `weighted_factor_batch.h`: generic weighting wrapper; `EvaluateWeighted` for any inner batch.
- `se3_joint_axis_factor_batch.h`: `SE3AxisAlignmentFactorBatch` (`R_1 a - R_0 a`, hinge) and
  `SE3AxisOffsetFactorBatch` (`(R_0 b_k) . (p_1 - p_0)`, slider) between any two SE(3) poses;
  with `LinkFrameFactorBatch` views they close revolute / prismatic loops.
- `imu_factor_batch.h`: keyframe pair (T, v, bias each), raw samples in a ragged CSR device
  array; the Euler chain is marginalized every evaluation (covariance-form Riccati sweep, split
  over 1-32 lanes); 15 rows (whitened defect 9 + bias walk 6). Guide: `docs/sphinx/imu.rst`.
- Constraints (solved by `AugmentedLagrangianMinimizer`): `constraint_factor_batch.h`
  (`ConstraintFactorBatchBase` holds multipliers and penalties; `ConstraintFactorBatch` wraps any
  batch's rows as `s r = 0` / `s r <= 0`), `bound_factor_batch.h`, `halfspace_factor_batch.h`. A
  constraint must be the outermost wrapper: the AL finds constraints by type, so one wrapped
  inside another batch would be solved as a plain residual.
