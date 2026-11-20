# Module: factor/information

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Weighting by square-root information matrices: whitening a factor's residuals and Jacobians.

## Files

- `information_factor_batch.h` (+ `information_factor_batch_kernels.cu`): the generic
  `InformationFactorBatch<T>` wrapper (residual and Jacobian of factor f multiplied by its
  row-major sqrt-information matrix), `EvaluateWithSqrtInformation` (the same for any inner
  batch; the Python wrapper uses it), and the item-contract helpers
  `ApplyInformationToResidualItems` / `ApplyInformationToJacobianItems`.
- `motion_prior_information.{h,cu}`: closed-form `Q(dt)^-1 = M(dt) (x) Qc^-1` for the
  `cunls/factor/motion/` factors, and `MotionPriorInformationFactorBatch<T, Dim>` (the motion
  factor with that information fused in; the matrices are computed and owned through a
  base-from-member helper, so they exist before the wrapper base reads them).

## Conventions

- Matrices are per factor and indexed by the item's factor id (`factor_ids[t]` or
  `t % N`), with a fixed summation order: results do not depend on the item count.
- Wrappers forward sizes to the wrapped batch. Do not wrap a constraint batch (it would be
  hidden from `AugmentedLagrangianMinimizer`; the Python binding rejects it).
