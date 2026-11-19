# Module: factor

> Notes for agents and contributors working in this directory (loaded automatically by coding agents that read nested `AGENTS.md`). Users of the library: see
> `python/pycunls/AGENT_GUIDE.md` and `docs/sphinx/`.

## Purpose

Defines residual/Jacobian models evaluated in GPU batches.

## Core interfaces

- `factor_batch.h`: abstract `FactorBatch`
- `sized_factor_batch.h`: compile-time dimension helper `SizedFactorBatch`

## Pose convention

Every factor reads pose states (SE2/SE3/SO2/SO3) as world_from_rig T (the
rig's pose in the world), perturbed on the right (T Exp(δ), δ in the rig
frame). Reprojection / PnP see a world point at E T^-1 P.

## Subfolders

Each has its own `AGENTS.md` with its files, conventions and invariants; read it before
changing that folder.

- `between/`: relative-pose factors, one per manifold, and the `BetweenFactorBatch<Manifold>`
  facade (`cunls/factor/between/AGENTS.md`).
- `prior/`: prior factors and the `PriorFactorBatch<Manifold>` facade
  (`cunls/factor/prior/AGENTS.md`).
- `motion/`: constant-velocity / constant-acceleration motion priors
  (`cunls/factor/motion/AGENTS.md`).
- `information/`: sqrt-information weighting, including the closed-form motion-prior noise
  (`cunls/factor/information/AGENTS.md`).
- `clearance/`: disk / sphere obstacle clearance as constraint functions
  (`cunls/factor/clearance/AGENTS.md`).
- `dynamics/`: vehicle and rigid-body models as shooting defects
  (`cunls/factor/dynamics/AGENTS.md`).
- `kinematics/`: articulated robots (`ArticulationStateBatch`): link targets, posture, loops,
  mimic joints, and `LinkFrameFactorBatch` for SE(3) factors on links
  (`cunls/factor/kinematics/AGENTS.md`).

## Factors in this directory

- Reprojection: `reprojection_factor_batch.h`
- PnP (fixed 3D points, pose-only Jacobian): `pnp_factor_batch.h`
- Registration: `point_to_point_factor_batch.h`, `point_to_plane_factor_batch.h`,
  `symmetric_point_to_plane_factor_batch.h`
- Generic weighting wrapper: `weighted_factor_batch.h`
- IMU: `imu_factor_batch.h` (`ImuFactorBatch`, `ImuParameters`): keyframe pair
  (T_a, v_a, b_a, T_b, v_b, b_b; T = world_from_rig, as reprojection / PnP), raw samples in a ragged CSR device array
  (7 floats each, `sample_offsets`); the intermediate states of the Euler chain
  are marginalized inside the factor every evaluation (covariance-form Riccati
  sweep; chains split over 1-32 lanes as composable summaries, sized from
  `num_samples / capacity`), residual 15 = whitened keyframe-b defect (9) +
  bias random walk (6). Guide: `docs/sphinx/imu.rst`.
- Constraints (augmented Lagrangian, solved by `AugmentedLagrangianMinimizer`):
  `constraint_factor_batch.h` (`ConstraintKind`, `ConstraintFactorBatchBase`:
  multipliers per row and penalties per factor, AL residuals in `Evaluate`,
  raw values in `EvaluateConstraint`; `ConstraintFactorBatch`: non-owning
  wrapper turning any batch's rows into `s*r = 0` / `s*r <= 0`),
  `bound_factor_batch.h` (`BoundFactorBatch<Dim>`: box bounds on a vector
  state, itself an inequality batch), `halfspace_factor_batch.h`
  (`HalfspaceFactorBatch<Dim>`: `a^T x - b`, to be wrapped). A constraint must be the
  outermost wrapper: AL finds constraints by type, so one wrapped inside another batch
  (information, weighting, `LinkFrameFactorBatch`) would be solved as a plain residual.

## Data contracts

- Inputs are typically device pointers and must remain valid for factor lifetime.
- Constructed with a capacity (measurements the buffers hold), fixed for the
  batch's lifetime. The active count `NumActiveFactors()` starts at 0:
  `SetNumActiveFactors(n)` (host-only) must be called before the first solve and may
  change it between solves up to `Capacity()`. Custom factors pass the
  capacity to `SizedFactorBatch(capacity)` instead of overriding
  `NumActiveFactors()`; wrappers forward the size calls to the wrapped batch.
- `Evaluate()` writes residuals and optionally Jacobians in contiguous batched
  layouts documented in each header.
- `state_pointers` is a flattened pointer array, grouped per factor.
- `Evaluate(residuals, jacobians, state_pointers, stream, factor_ids = nullptr,
  num_factor_ids = 0)` evaluates n items (n = num_factor_ids, or NumActiveFactors()
  when 0): item t is factor `factor_ids[t]` (or t % NumActiveFactors()) with its own
  pointer row `state_pointers[t * B ..]`; outputs are item-major. Results must
  match a plain evaluation bitwise. All built-in factors implement it (device
  helper `FactorMeasurementIndex`, defined in each factor's `.cu`); RANSAC relies
  on it.

## Related modules

- States consumed from `state/`.
- Optional robustification through `robustifier/` via `ResidualBatch`.
