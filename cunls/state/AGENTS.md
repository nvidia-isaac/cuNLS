# Module: state

Batched optimization variables and their `Plus` (tangent update).

## Files

- `state_batch.h` (`StateBatch`, `BoxBoundedStates`, `DerivedStateData`),
  `sized_state_batch.h` (`DynamicSizedStateBatch`: bookkeeping with run-time sizes;
  `SizedStateBatch<AmbientDim, TangentDim>` derives from it), `state_batch_ops.h`
  (`StateBatchOps`: reduced / full mapping, batched updates).
- `vector_state_batch.h` (`DynamicVectorStateBatch(dim, ...)`, bound in Python as
  `VectorStateBatch`; `VectorStateBatch<Dim>` derives from it), `so2_` / `so3_` / `se2_` /
  `se3_` / `similarity2_` / `similarity3_` / `sl4_state_batch.h`.
- Articulations: `articulation_topology.h` (`ArticulationDescription` flat joint arrays ->
  validated, parents-first `ArticulationTopology` with a kernel `DeviceView`) and
  `articulation_state_batch.h` (`ArticulationStateBatch`: owns its storage; ambient = joint_q +
  link poses (12 per link) + DOF motion columns (6 per DOF); `Plus` steps each joint, clamps moved
  scalar coordinates, runs forward kinematics into the cache; BALL / FREE quaternions are
  normalized whenever coordinates enter the cache, a zero one becomes the identity). Cache and
  index conventions: `cunls/factor/kinematics/AGENTS.md`.

## Memory model

- States live in caller-provided device memory (except `ArticulationStateBatch`, which owns its
  storage and its topology joint limits but borrows custom bounds (`SetBounds`, which must
  outlive the solves) and constant ids). Constant state ids mark fixed states.
- Capacity is fixed at construction; active counts start at 0 and `SetNumActiveStates(n,
  num_const)` (host) sets them before a solve. `StateDevicePtr(i)` is valid for i < `Capacity()`.
- `Plus(x, delta, out, stream, num_replicas = 1)` updates `num_replicas` contiguous copies
  (RANSAC: one per hypothesis); `out` does not overlap the inputs.
- Box bounds (`DynamicVectorStateBatch::SetBounds`, device arrays, ±inf allowed, not owned) are read
  through `BoxBoundedStates`; the bounded `Plus` clamps every moved component. Bounds are a
  constraint: only `AugmentedLagrangianMinimizer` solves bounded problems. Numeric differentiation
  perturbs through `PlusIgnoringBounds`; `BoundedComponentOffsets` (nullptr = identity) maps each
  bounded value (an articulation's scalar DOFs to coordinates).
- `DerivedStateData`: the minimizers call `RefreshDerivedStateData` on every batch at the start of
  each (inner) solve, and the Python diagnostics before evaluating.
