# Module: factor/between

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Relative-pose factors: a measured transform between two states of the same manifold.

## Files

- `<group>_between_factor_batch.{h,cu}` for so2, so3, se2, se3, similarity2, similarity3, sl4
  and vector: one hand-written kernel per manifold, analytic Jacobians.
- `between_factor_batch.h`: the facade `BetweenFactorBatch<Manifold>`, a zero-cost alias of the
  concrete class (no members, no extra virtual dispatch). The manifold is deduced (CTAD) from the
  deltas pointer's type (`SE3Transform`, `SO3Rotation`, ...: distinct types in
  `cunls/common/types.h`); generic code may spell `<manifold::SE3>`.

## Conventions

- Two states (left, right), residual `Log(Delta * T_left^-1 * T_right)` in the group's tangent
  (SE3: `[omega, v]`); `Delta` is the measurement, one per factor.
- `Delta = T_right^-1 * T_left` at the solution: the **inverse** of the usual relative pose
  `T_left^-1 * T_right`. Users get this wrong most often (`python/pycunls/AGENT_GUIDE.md`,
  section 4).
- Right perturbation of both states (`T Exp(delta)`), as every pose factor here.
- Between factors only constrain states relative to each other: a graph of them has gauge
  freedom (pycunls diagnostics flag it through `relative=True` in `python/pycunls/catalog.py`).
- On robot links: `SE3BetweenFactorBatch` inside `LinkFrameFactorBatch`
  (`cunls/factor/kinematics/AGENTS.md`).
