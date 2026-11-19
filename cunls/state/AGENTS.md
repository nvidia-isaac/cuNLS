# Module: state

> Notes for agents and contributors working in this directory (loaded automatically by coding agents that read nested `AGENTS.md`). Users of the library: see
> `python/pycunls/AGENT_GUIDE.md` and `docs/sphinx/`.

## Purpose

Represents optimization variables in batched manifold form and provides `Plus`
operations for applying tangent-space updates.

## Core interfaces

- `state_batch.h`: abstract base `StateBatch`
- `sized_state_batch.h`: `SizedStateBatch<AmbientDim, TangentDim>`
- `state_batch_ops.h`: `StateBatchOps` (reduced/full mapping + batched updates)

## Implementations

- Euclidean: `vector_state_batch.h`
- Rotation groups: `so2_state_batch.h`, `so3_state_batch.h`
- Rigid transforms: `se2_state_batch.h`, `se3_state_batch.h`
- Similarities: `similarity2_state_batch.h`, `similarity3_state_batch.h`
- Projective: `sl4_state_batch.h` (SL(4) homographies)
- Articulations: `articulation_topology.h` (`ArticulationDescription`: Newton's joint
  arrays; `ArticulationTopology`: validated, parents-first, uploaded; `DeviceView` for
  kernels) and `articulation_state_batch.h` (`ArticulationStateBatch`: owned storage,
  ambient = joint_q + link poses (12/link) + DOF motion columns (6/DOF); `Plus` applies the
  per-joint step, clamps moved scalar coordinates, then runs forward kinematics into the
  cache). Implements `BoxBoundedStates` (bounds per DOF; `BoundedComponentOffsets` =
  coordinate of each scalar DOF, -1 for BALL/FREE) and `DerivedStateData`.

## Memory model

- Backed by user-provided contiguous device memory, except `ArticulationStateBatch`, which
  allocates and owns its state storage; its custom bounds (`SetBounds`) and constant state IDs
  are still borrowed from the caller.
- Optional constant state IDs mark fixed states excluded from optimization.
- Constructed with a capacity (states the buffer holds; `const_capacity` for
  the constant-id buffer), fixed for the batch's lifetime. The active counts
  start at 0: `SetNumActiveStates(n, num_const)` (host-only) must be called
  before the first solve and may change them between solves up to the
  capacity. `NumActiveStates()` is the active count, `Capacity()` the capacity.
- `StateDevicePtr(i)` returns pointers into underlying batch storage,
  valid for any i < Capacity().
- `Plus(x, delta, out, stream, num_replicas = 1)` processes `num_replicas`
  contiguous copies of the batch (replica r = states [r*N, (r+1)*N)); out does
  not overlap the inputs. RANSAC passes one replica per hypothesis.
- Box bounds: `VectorStateBatch<Dim>::SetBounds(lower, upper)` (device arrays,
  Capacity()*Dim floats, ±inf allowed; not owned), read through the
  `BoxBoundedStates` interface (`HasBoxBounds(batch)`); `StateBatch` itself has
  no bounds API. The bounded `Plus` clamps every component with a nonzero
  delta into the box (constant states, delta 0, never move). Bounds are a
  constraint: only `AugmentedLagrangianMinimizer` solves bounded problems
  (it holds the active bounds in the inner systems). Numeric differentiation
  perturbs bounded states by plain addition (a clamped perturbation would
  give a one-sided derivative at a bound).

- Derived data: `DerivedStateData` (`state_batch.h`) for batches whose storage caches data
  computed from the coordinates; the minimizers call `RefreshDerivedStateData` on every
  problem batch at the start of each (inner) solve, before copying the states, and the
  Python diagnostics before evaluating.
- `BoxBoundedStates::PlusIgnoringBounds` is the perturbation of numeric differentiation;
  `BoundedComponentOffsets` (nullptr = identity) tells the AL where each bounded value is.

## Related modules

- Consumed by `factor/` through state pointer arrays.
- Updated by minimizers through `StateBatchOps`.
