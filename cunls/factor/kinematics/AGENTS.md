# Module: factor/kinematics

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.
> The state side (topology, cache, `Plus`): `cunls/state/AGENTS.md`. User guide with every
> convention: `docs/sphinx/articulations.rst` (keep it in sync when a convention changes).

## Purpose

Factors on articulated robots (`ArticulationStateBatch`): link targets, posture, closed loops,
mimic joints, and a wrapper that puts any SE(3) factor on robot links.

## Files

- `articulation_factor_batch.h`: `ArticulationFactorBatch`, the base: topology + residual size;
  state slot 0 is an articulation instance (tangent = DOF count). `StateSizes()` is overridable
  only so `LinkFrameFactorBatch` can append its inner factor's other slots.
- `link_target_factor_batch.{h,cu}`: `LinkPositionFactorBatch` (3 rows),
  `LinkOrientationFactorBatch` (3), `LinkPoseFactorBatch` (6); one templated kernel.
- `posture_factor_batch.{h,cu}`: `PostureFactorBatch`, `W (q - q*)` per DOF (DOF-count rows).
- `loop_closure_factor_batch.{h,cu}`: `LoopClosureFactorBatch`, a loop-closing joint as
  constraint rows (BALL 3, REVOLUTE 6, PRISMATIC 5, FIXED 6), wrapped as an equality.
- `mimic_joint_factor_batch.{h,cu}`: `MimicJointFactorBatch`, `q_m - (a + b q_ref)`.
- `link_frame_factor_batch.{h,cu}`: `LinkFrameFactorBatch`, see below.

## Conventions every kernel here relies on

- **Index spaces**: joint, link, coordinate (`joint_q`), DOF (tangent). Link ids index the
  cache's poses and the ancestor bitsets; DOF ids index Jacobian columns, bounds and motion
  columns; `scalar_coord[d]` maps a scalar DOF to its coordinate (-1 for BALL / FREE).
- **Cache** (per instance, after `joint_q`; offsets from `ArticulationTopology`): link pose
  `l` = 12 floats, row-major 3x4 `[R | p]`, world_from_link; motion column of DOF `j` = 6
  floats `(omega, v)`, the world twist at the world origin per unit of `q_j` (a point `x`
  moves with `omega x x + v`).
- **Ancestors**: bit `j` of link `l`'s bitset (`ancestor_words` words per link) is set iff DOF
  `j` moves link `l`. Columns of other DOFs must be written as exact zeros.
- **Quaternions** (x, y, z, w) in every data array; poses in factor data are 7 floats
  (position, quaternion). Rotation residuals of the link factors are world-frame rotation
  vectors with `J_l^-1` Jacobians; posture rotation rows are in each joint's parent-anchor frame.
- **Factors never run kinematics**: the state's `Plus` refreshes the cache; a factor only reads
  it. Out-of-range link / DOF indices give NaN residuals and zero Jacobian rows (no read past the
  cache); `pycunls.diagnose` names the cause.

## Kernel layout

One group of 8/16/32 lanes per item, chosen by DOF count: every lane computes the residual
(cheap), lane 0 writes it, lane `l` writes Jacobian columns `l, l + G, ...` of every row, so each
row is written with consecutive stores. The dense rows bound an articulation by the Hessian
assembler's shared-memory staging (about 140 DOFs for 6-row factors, about 85 with
`PostureFactorBatch`).

## LinkFrameFactorBatch

Wraps an inner factor over SE(3) poses (built-in or `CustomFactorBatch` /
`pycunls.warp.WarpFactorBatch`). Terms: `pose_slots` = inner state slots that receive a link
frame pose instead of a problem state; `topology_links` = per factor and pose slot, the link
carrying the frame; `frame_offsets` = the frame in that link (7 floats, optional).

Evaluation, all on the stream:

1. `GatherLinkFramesKernel` (one thread per item): `F_s = X_L X_off` as a row-major 4x4 into
   scratch `poses_` (NaN for an out-of-range link), and the inner state-pointer table (pose
   slots -> scratch, other slots -> this item's own pointers).
2. `inner->Evaluate(...)` with the same `factor_ids` / item count: its residuals are this batch's;
   its Jacobian goes to scratch `inner_jacobians_`.
3. `CarryJacobianKernel` (one thread per item and output column): DOF column `j` =
   sum over pose slots whose link DOF `j` moves of `J_inner[:, slot] . b`, with the body twist
   `b = (R^T omega_j, R^T (omega_j x p + v_j))` of `F_s = (R, p)`; columns of the other slots
   are copied.

Invariants: pose slots follow `SE3StateBatch`'s convention (right perturbation, body tangent
`[omega, v]`); the inner factor must not be a constraint batch (AL finds constraints by type; the
constructor throws); scratch grows to the largest item count and is never shrunk; the problem's
state-layout check covers slot 0 because this class derives from `ArticulationFactorBatch`.

## Tests

`tests/articulation_test.cpp` (central-difference Jacobians through `PlusIgnoringBounds`,
item-contract checks with `evaluate_items_check.h`, the `LinkFrameFactor` tests against the
inner factor evaluated directly) and `python/tests/test_articulation.py`. Example:
`python/examples/robot_custom_objective.py`.
