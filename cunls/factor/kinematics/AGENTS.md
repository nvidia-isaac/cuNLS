# Module: factor/kinematics

Factors on articulated robots (`ArticulationStateBatch`). Shared factor contracts:
`cunls/factor/AGENTS.md`; state side (topology, cache, `Plus`): `cunls/state/AGENTS.md`; user
guide with every convention: `docs/sphinx/articulations.rst` (keep it in sync). Closed loops have
no class: they are equality constraints on `LinkFrameFactorBatch` views (between factors, plus the
SE(3) axis factors for revolute / prismatic loops), built by `pycunls.articulation.loop_closure`.

## Files

- `articulation_factor_batch.h`: base (topology + residual size; slot 0 is an articulation, tangent
  = DOF count). `StateSizes()` is overridable only for `LinkFrameFactorBatch`'s extra slots.
- `link_target_factor_batch.{h,cu}`: `LinkPosition` (3 rows), `LinkOrientation` (3), `LinkPose`
  (6) factor batches; one templated kernel.
- `posture_factor_batch.{h,cu}` (`W (q - q*)` per DOF), `mimic_joint_factor_batch.{h,cu}`
  (`q_m - (a + b q_ref)`), `link_frame_factor_batch.{h,cu}`.

## Conventions every kernel relies on

- Index spaces: joint, link, coordinate (`joint_q`), DOF (tangent). Link ids index cache poses and
  ancestor bitsets; DOF ids index Jacobian columns, bounds and motion columns; `scalar_coord[d]`
  maps a scalar DOF to its coordinate (-1 for BALL / FREE).
- Cache per instance after `joint_q`: link pose = 12 floats, row-major 3x4 `[R | p]`,
  world_from_link; motion column of DOF j = 6 floats `(omega, v)`, the world twist at the origin
  per unit of `q_j`. Bit j of link l's ancestor bitset is set iff DOF j moves link l; other DOFs'
  columns are written as exact zeros.
- Quaternions are (x, y, z, w); poses in factor data are 7 floats. Link rotation residuals are
  world-frame rotation vectors with `J_l^-1` Jacobians.
- Factors never run kinematics (the state's `Plus` refreshes the cache). Out-of-range link / DOF
  indices give NaN residuals and zero Jacobian rows; `pycunls.diagnose` names the cause.
- Kernels: one group of 8 / 16 / 32 lanes per item by DOF count; lane l writes Jacobian columns
  l, l + G, ... (consecutive stores). Dense rows bound an articulation by the assembler's shared
  memory (about 140 DOFs for 6-row factors, 85 with `PostureFactorBatch`).

## LinkFrameFactorBatch

Wraps an inner factor (built-in, `CustomFactorBatch` or `pycunls.warp.WarpFactorBatch`) whose
frame slots receive a view of a frame on a link: `frame_slots` = (inner slot, `FrameView`),
`topology_links` = the link per factor and frame slot, `frame_offsets` = the frame in that link (7
floats, optional). Views use an existing state type's layout and tangent, so its factors work
unchanged:

| `FrameView` | Layout | As | Motion per unit of DOF j |
|---|---|---|---|
| `kPose` | 4x4 row-major | `SE3StateBatch` | `(Rᵀ omega_j, Rᵀ (omega_j x p + v_j))` |
| `kPosition` | world point | `VectorStateBatch<3>` | `omega_j x p + v_j` |
| `kOrientation` | 3x3 row-major | `SO3StateBatch` | `Rᵀ omega_j` |

Evaluation: `GatherLinkFramesKernel` writes `X_L X_off` as each slot's view into scratch (NaN for
a bad link) and builds the inner pointer table; the inner factor evaluates (Jacobian to scratch);
`CarryJacobianKernel` sums, per DOF column, the inner Jacobian times the view's motion over the
frame slots that DOF moves. The view must match the slot's tangent (6 / 3 / 3); the inner factor
must not be a constraint batch (the constructor throws); scratch only grows. Views cost 1.26x the
fused `LinkPoseFactorBatch` (7-DOF IK, `ArticulationBenchmark.DISABLED_IkThroughput`), so the fused
link factors stay the fast path.

Tests: `tests/articulation_test.cpp` (central differences through `PlusIgnoringBounds`, item
contract, views against the inner factor), `python/tests/test_articulation.py`; example
`python/examples/robot_custom_objective.py`.
