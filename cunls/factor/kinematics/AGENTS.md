# Module: factor/kinematics

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.
> The state side (topology, cache, `Plus`): `cunls/state/AGENTS.md`. User guide with every
> convention: `docs/sphinx/articulations.rst` (keep it in sync when a convention changes).

## Purpose

Factors on articulated robots (`ArticulationStateBatch`): link targets, posture, closed loops,
mimic joints, and a wrapper that puts existing SE(3), vector and SO(3) factors on robot links.
Closed loops have no class of their own: they are equality constraints on `LinkFrameFactorBatch`
views (between factors, plus `SE3AxisAlignmentFactorBatch` / `SE3AxisOffsetFactorBatch` from
`cunls/factor/se3_joint_axis_factor_batch.h` for revolute / prismatic loops), built by
`pycunls.articulation.loop_closure` (`python/pycunls/articulation.py`).

## Files

- `articulation_factor_batch.h`: `ArticulationFactorBatch`, the base: topology + residual size;
  state slot 0 is an articulation instance (tangent = DOF count). `StateSizes()` is overridable
  only so `LinkFrameFactorBatch` can append its inner factor's other slots.
- `link_target_factor_batch.{h,cu}`: `LinkPositionFactorBatch` (3 rows),
  `LinkOrientationFactorBatch` (3), `LinkPoseFactorBatch` (6); one templated kernel.
- `posture_factor_batch.{h,cu}`: `PostureFactorBatch`, `W (q - q*)` per DOF (DOF-count rows).
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

Wraps an inner factor (built-in or `CustomFactorBatch` / `pycunls.warp.WarpFactorBatch`) whose
frame slots receive a **view** of a frame on a link instead of a problem state. Terms:
`frame_slots` = (inner state slot, `FrameView`) pairs; `topology_links` = per factor and frame
slot, the link carrying the frame; `frame_offsets` = the frame in that link (7 floats, optional).
Views use the layout and tangent convention of an existing state type, so that type's factors
work unchanged:

| `FrameView` | Scratch layout | As | Motion per unit of DOF `j` |
|---|---|---|---|
| `kPose` | 16, row-major 4x4 | `SE3StateBatch` | `(R^T omega_j, R^T (omega_j x p + v_j))` |
| `kPosition` | 3, world point | `VectorStateBatch<3>` | `omega_j x p + v_j` |
| `kOrientation` | 9, row-major 3x3 | `SO3StateBatch` | `R^T omega_j` |

Evaluation, all on the stream:

1. `GatherLinkFramesKernel` (one thread per item): `F_s = X_L X_off`, written as the slot's view
   into scratch `frames_` (16 floats per item and slot; NaN for an out-of-range link), and the
   inner state-pointer table (frame slots -> scratch, other slots -> this item's own pointers).
   Each view holds what its motion needs (pose: R and p; position: p; orientation: R), so no
   second copy of the frame is kept.
2. `inner->Evaluate(...)` with the same `factor_ids` / item count: its residuals are this batch's;
   its Jacobian goes to scratch `inner_jacobians_`.
3. `CarryJacobianKernel` (one thread per item and output column): DOF column `j` = sum over frame
   slots whose link DOF `j` moves of `J_inner[:, slot] . b` with the view's motion `b` above;
   columns of the other slots are copied.

Invariants: the view of a slot is explicit and must match the slot's tangent size (6 / 3 / 3);
the inner factor must not be a constraint batch (AL finds constraints by type; the constructor
throws); scratch grows to the largest item count and is never shrunk; the problem's
state-layout check covers slot 0 because this class derives from `ArticulationFactorBatch`.

Performance (7-DOF IK, 32k problems, Blackwell): a `kPosition` prior plus a `kOrientation` prior
cost 1.26x the fused `LinkPoseFactorBatch` (gather ~23% of the gap, the rest the extra batch,
the carry and the inner factors), so the fused link factors stay the fast path; measurements in
`ArticulationBenchmark.DISABLED_IkThroughput` (`IK_BENCH_OBJECTIVES=fused,views,pose_view`).

## Tests

`tests/articulation_test.cpp` (central-difference Jacobians through `PlusIgnoringBounds`,
item-contract checks with `evaluate_items_check.h`, the `LinkFrameFactor` tests against the
inner factor evaluated directly) and `python/tests/test_articulation.py`. Example:
`python/examples/robot_custom_objective.py`.
