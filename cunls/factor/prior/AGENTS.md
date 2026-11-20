# Module: factor/prior

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Prior factors: one state should equal a measured value.

## Files

- `<group>_prior_factor_batch.{h,cu}` for so2, so3, se2, se3, similarity2, similarity3, sl4,
  and `prior_vector_factor_batch.{h,cu}` (`PriorVectorFactorBatch<Dim>`).
- `prior_factor_batch.h`: the facade `PriorFactorBatch<Manifold>`, deduced from the target
  pointer's type like `BetweenFactorBatch` (`cunls/factor/between/AGENTS.md`).

## Conventions

- One state, residual `Log(T_target^-1 * T_current)` (vector: `x - target`), right
  perturbation.
- Priors fix their states (no gauge freedom); use them to anchor a pose graph.
- On robot links: `SE3PriorFactorBatch` inside `LinkFrameFactorBatch` is a frame target in
  the SE(3) body tangent (`LinkPoseFactorBatch` is the built-in, world-frame equivalent).
