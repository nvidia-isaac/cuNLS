# Module: factor/clearance

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Obstacle clearance as constraint functions, for planning and MPC.

## Files

- `se2_disk_clearance_factor_batch.{h,cu}`: `SE2DiskClearanceFactorBatch`, SE(2) pose vs. disk
  obstacles (3 floats per factor: center x, y, radius).
- `se3_sphere_clearance_factor_batch.{h,cu}`: `SE3SphereClearanceFactorBatch`, SE(3) pose vs.
  sphere obstacles (4 floats: center x, y, z, radius).

## Conventions

- One row per factor, `c = (radius + margin) - |p - center|`, feasible when `c <= 0`: wrap
  in `ConstraintFactorBatch(..., ConstraintKind::kInequality)`. The robot's own radius goes
  into `margin`.
- Only the pose origin matters: the Jacobian is zero on the rotation part (and at the center).
- On robot links: `SE3SphereClearanceFactorBatch` inside `LinkFrameFactorBatch` (the frame
  offset picks the point on the link); wrap the `LinkFrameFactorBatch` as the inequality.
  Example: `python/examples/robot_custom_objective.py`. Guide: `docs/sphinx/dynamics.rst`.
