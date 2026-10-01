# Reusable Buffers: Bind Once, Update Contents and Sizes Every Solve

Status: **implemented** on `dev/ak/api` (written against `a9ac4c6`; §11
lists where the implementation differs from the proposal below and the
measured performance).

## 0. Problem statement

Real-time users (tracking, sliding-window SLAM, per-frame registration) solve
a new problem every frame. Between two solves **everything may change**:

- the **measurements** of every factor (new observations, new matches);
- the **states**: their number and their initial values;
- the **connectivity**: which states each factor reads;
- the **sizes**: how many factors and states take part.

What does not change is the *shape*: the same factor and state types, the same
upper bounds. Such users allocate their device buffers once, at capacity, and
want to rewrite their contents in place every frame. They do not want to
allocate, construct objects, or hand cuNLS new pointers per frame.

Today that is not possible:

- a factor batch stores its measurement pointer and `num_factors` at
  construction, and neither can change;
- several factor batches copy derived data from the measurements once, in the
  constructor (inverted targets, measurement adjoints), so rewriting the
  measurements in place silently uses stale data;
- a state batch's state count and constant set are fixed at construction;
- the connectivity is a host `std::vector<float *>` per factor batch, copied
  into `Problem` at `AddFactorBatch` and never replaceable.

### Goals

- **G1 Bind once.** Every device buffer (measurements, states, constant ids,
  connectivity) is user-owned, allocated once at capacity, and bound to cuNLS
  once.
- **G2 Update in place.** Between solves the user rewrites buffer contents
  (on the GPU or by copy) and sets the active sizes with host calls. No
  allocation, no new objects, no new pointers.
- **G3 No performance cost.** Kernels run over the active sizes, not the
  capacity. Steady-state solves allocate nothing. No extra host
  synchronization compared with today.
- **G4 Simple implementation.** Mostly mechanical changes; the factor and
  state classes get simpler, not more complex.
- **G5 Compatible.** Existing code keeps compiling and behaving the same.

### Non-goals

Incremental solving (keeping the Hessian structure between solves, activity
masks, partial updates) is `docs/design/incremental_optimization.md`. This
design is its foundation and conflicts with nothing in it (§9).

## 1. Current conventions (read from the code at `a9ac4c6`)

- **Factor batches** keep `ptr` and `num_factors_` from the constructor (e.g.
  `PnPFactorBatch(observations, points, num, z_threshold)`). `NumActiveFactors()`
  drives every launch and every buffer size.
- **Derived data.** These batches compute data from the measurements in the
  constructor, on a private `CudaStream` with a synchronize:
  - priors on SE(2), SE(3), Sim(2), Sim(3), SL(4): `observations_inverse_`;
  - between factors on SO(3), SE(3), Sim(3), SL(4): `delta_adjoints_`;
  - `MotionPriorInformation`: `sqrt_information` from `dt` and `Qc`.
- **Scratch.** Many factor batches and all Lie-group state batches keep
  `mutable` device buffers (`poses_left_`, `transforms_error_`,
  `delta_transforms_`, ...) that `Evaluate` / `Plus` resize per call.
  `dvector::resize` never shrinks or reallocates below its capacity.
- **State batches** (`SizedStateBatch`) keep `ptr_`, `num_blocks_`, and a
  user-owned constant-id array with its count, all fixed at construction.
- **Connectivity** is `Problem::state_pointers_`, one host
  `std::vector<float *>` per residual batch. Its consumers:
  - `MinimizerState::CopyProblemStatePointersFromHost` uploads it on every
    `Minimize` (twice: current and candidate copies), then a kernel remaps each
    pointer into the minimizer's private state copy;
  - `HessianStructureBuilder::ResolveFactorColumns` uploads it again and finds
    each pointer's column with an address-range test on the GPU;
  - `Problem::CheckGraphConnectivity` (host hash map; called by the RANSAC
    minimizers, not by GN/LM);
  - `NumericDiffJacobianBuilder::PrepareResidualBatch` (host hash map);
  - `RansacLayout::ResolveBlocks` (host loop).
- **Every `Minimize` re-initializes.** `GaussNewtonMinimizer::Minimize` calls
  `Initialize`, which re-reads every size, rebuilds the Hessian structure,
  re-creates the state copies and re-uploads the connectivity. Buffers keep
  their capacity. The RANSAC minimizers do the same in `Prepare`. **So the
  solvers already accept a different problem size on every call**; what is
  missing is a way to change the sizes and the connectivity without
  rebuilding the objects.

## 2. Design decisions

- **D1 Capacity = the constructor's count; active sizes are zero until set.**
  The constructors keep their shapes; the count they take is the *capacity*
  (the parameter is named `capacity` / `const_capacity`), constant for the
  batch's lifetime. The active sizes start at 0: every user calls
  `SetNumActiveFactors(n)` / `SetNumActiveStates(n, num_const_states)` before the first
  solve, and `Minimize` rejects a problem with no active factors.
- **D2 Sizes are host setters.** `FactorBatch::SetNumActiveFactors(n)` and
  `StateBatch::SetNumActiveStates(n, num_const_states)`, with `n <= Capacity()`. They
  are plain host assignments: no stream, no kernel, no allocation.
- **D3 Factors cache nothing derived from measurements.** Derived data is
  computed inside the evaluation kernels. Rewriting measurements in place is
  then always correct, and `SetNumActiveFactors` needs no recomputation.
- **D4 Scratch is reserved at capacity** in the constructors. Solves never
  allocate (RANSAC's item counts above capacity grow it once).
- **D5 Connectivity is a device table.** A factor batch's connectivity is a
  user-owned device array bound once at `AddFactorBatch`, rewritten in place
  between solves. Two forms (§5): raw state pointers (general), or state
  *indices* into fixed state batches (recommended for dynamic graphs).
- **D6 The device table is the single source.** Every consumer reads the
  device table: no host copy, no per-solve upload. The host-vector
  `AddFactorBatch` stays as a convenience that copies into a library-owned
  device table.
- **D7 Validation moves to the GPU and is opt-in per solve.** Checking every
  connection on the host each frame would cost a readback and an O(N) host
  walk. A GPU validation kernel replaces it (§5.5).

## 3. Factors

### 3.1 API

```cpp
class FactorBatch {
 public:
  virtual size_t NumActiveFactors() const;          // the active count n (0 after construction)
  virtual size_t Capacity() const;            // fixed at construction
  virtual void SetNumActiveFactors(size_t n);       // n <= Capacity(), host only
 protected:
  FactorBatch() = default;
  explicit FactorBatch(size_t capacity);      // SizedFactorBatch(capacity) forwards it
};
```

`FactorBatch` holds `capacity_` and `num_factors_`, set by the protected
constructor (capacity) and `SetNumActiveFactors` (active count, 0 until called).
Every built-in factor passes its capacity to the base and reads `NumActiveFactors()`.
The three methods stay virtual only so wrappers (`InformationFactorBatch`,
`WeightedFactorBatch`) can forward them to the wrapped batch; custom factors
pass their capacity to the base instead of overriding `NumActiveFactors()` (one that
still overrides it reports `Capacity() == 0` and is rejected by
`Problem::CheckSizes`).

The measurement buffers stay user-owned. Their capacity is the constructor
count; the user may rewrite any of the first `n` entries between solves.

### 3.2 Removing derived data (D3)

| Factor batch | Cached today | Computed in the kernel instead |
|---|---|---|
| SE(2)/SE(3)/Sim(2)/Sim(3) prior | `observations_inverse_` | closed-form inverse: `[Rᵀ, −Rᵀt]` (Sim: `s⁻¹`) |
| SL(4) prior | `observations_inverse_` | 4×4 inverse via cofactors (benchmark, §8) |
| SO(3)/SE(3)/Sim(3)/SL(4) between | `delta_adjoints_` | adjoint of the measurement, per item |
| `MotionPriorInformation` | `sqrt_information` (from `dt`, `Qc`) | gains `Update(stream)` for the active count; see §3.4 |

The kernels already load each measurement; the extra arithmetic is a few
dozen flops in memory-bound kernels. Removing the caches also removes the
constructors' private stream and synchronize, and fixes the stale-data bug
when measurements are edited in place today.

### 3.3 Scratch (D4)

Constructors reserve every `mutable` scratch buffer for `capacity` items.
`Evaluate` keeps its `resize(num_items)` (a no-op within capacity). Several
factor families use three or four scratch passes (`poses_left_`,
`poses_right_`, `poses_left_inverse_`); fusing them is a separate
optimization, not required here.

### 3.4 Wrappers and helpers

- `WeightedFactorBatch<T>`, `InformationFactorBatch<T>`: forward
  `SetNumActiveFactors` to the wrapped batch. Their per-factor arrays (weights,
  sqrt-information matrices) are user buffers with the same capacity and are
  read through `factor_ids` / item index as today.
- `MotionPriorInformation` (helper that fills sqrt-information matrices from
  per-factor `dt`): `Update(stream, n)` recomputes the first `n` matrices from
  the current `dt` buffer. Called by the user after rewriting `dt`.

### 3.5 Python

```python
pnp = pycunls.PnPFactorBatch(obs_gpu, pts_gpu, capacity)  # unchanged signature
pnp.set_num_active_factors(n)
pnp.capacity                                               # read-only
```

## 4. States

### 4.1 API

```cpp
class StateBatch {
 public:
  // Existing: NumActiveStates(), StateDevicePtr(i), ConstStateIds(),
  //           NumConstStates(), Plus(...)
  // New:
  size_t Capacity() const;
  size_t ConstCapacity() const;
  void SetNumActiveStates(size_t n, size_t num_const_states = 0);  // host only
};
```

Implemented once in `SizedStateBatch`:

- the state buffer passed to the constructor has `Capacity()` states; the
  active states are `[0, n)`;
- the constant-id buffer passed to the constructor (if any) has
  `ConstCapacity()` entries; the first `num_const_states` are active and must be
  `< n`;
- Lie-group batches reserve their `Plus` scratch for `capacity` states.

`Plus` already processes `NumActiveStates() * num_replicas` states of the
arrays it is given, so it needs no change.

### 4.2 What the minimizer does with a new size

On every `Minimize` the minimizer already copies the `NumActiveStates()`
active states into its private copies, builds the column map from the active
constant ids, and writes the result back to the first `n` states of the user
buffer. States `[n, Capacity())` are never read or written.

### 4.3 Python

```python
landmarks = pycunls.VectorStateBatch3(pts_gpu, capacity)
landmarks.set_num_active_states(m)
poses = pycunls.SE3StateBatch(poses_gpu, capacity, const_ids_gpu, const_capacity)
poses.set_num_active_states(k, num_const_states=1)
```

## 5. Connectivity

### 5.1 The two forms

**Pointer table (general).** A device array of `float *`, `Capacity() × B`
entries (`B = StateSizes().size()`), entry `f * B + b` pointing at the
state that factor `f` reads in its slot `b`. This is today's format,
moved to the device.

```cpp
problem.AddFactorBatch(&factor, d_state_pointers /* float ** */);
```

**Index table (recommended for dynamic graphs).** The state batch of each
slot is fixed at registration; the device table holds state **indices**,
`Capacity() × B` ints:

```cpp
problem.AddFactorBatch(&reprojection, {&poses, &landmarks}, d_state_indices /* int * */);
// factor f reads poses[d_state_indices[2f]] and landmarks[d_state_indices[2f + 1]]
```

Indices are what user code naturally has (camera id, landmark id, track id),
they do not depend on buffer addresses, and validating them is a range check.
Every built-in factor has a fixed state batch per slot, so this form covers
all of them; the pointer form remains for custom factors whose slots mix
state batches.

The existing host-vector overload stays:

```cpp
problem.AddFactorBatch(&factor, host_pointer_vector);   // copied to a library-owned table
problem.SetStatePointers(residual_batch_index, host_pointer_vector);  // replace later
```

Only the first `NumActiveFactors() × B` entries of any table are read.

### 5.2 How connectivity is updated every solve

The user rewrites the table in place, on the GPU, ordered before the next
`Minimize`. Nothing is re-registered.

```
frame k:
  user stream S:  write measurements[0..n)      (kernel or cudaMemcpyAsync)
                  write state values[0..m)       (initial guess)
                  write connectivity[0..n*B)     (indices or pointers)
  host:           factor.SetNumActiveFactors(n); states.SetNumActiveStates(m, c)
  host:           minimizer.Minimize(S, problem)
                    ├─ (index form) one kernel: indices -> pointers into the
                    │    minimizer's state copies (replaces today's remap)
                    ├─ column resolution reads the device table directly
                    ├─ Hessian structure build, solve, iterations (unchanged)
                    └─ write back states[0..m)
  user:           read results from the state buffers
```

Ordering: all writes must be ordered before `Minimize` on the GPU. The
simplest way is to issue them on the stream passed to `Minimize`; otherwise
record an event and make that stream wait on it. The table, measurements and
states must not be rewritten while a `Minimize` that uses them is in flight.

### 5.3 What changes inside `Minimize`

| Step | Today | With device tables |
|---|---|---|
| Connectivity to device | host vector → H2D upload, twice (`MinimizerState` current/candidate) | none |
| Pointers into the minimizer's state copies | remap kernel by address-range search | index form: `copy_base[batch(b)] + idx * ambient`; pointer form: today's remap kernel, reading the device table |
| Column resolution (`HessianStructureBuilder`) | third upload of the host pointers, then address-range kernel | reads the device table; index form uses the index directly |
| Numeric-diff block map | host hash map over the host pointers | small kernel over the device table |
| RANSAC block map (`RansacLayout::ResolveBlocks`) | host loop over the host pointers | small kernel over the device table |
| Validation | host walk in `CheckConsistency` (RANSAC only) | GPU kernel, opt-in (§5.5) |

Net effect per solve: three host→device uploads of `Σ N × B` pointers
disappear, replaced by one elementwise kernel (index form) or nothing
(pointer form). Everything else is unchanged.

### 5.4 Library-owned table for the host-vector convenience

`AddFactorBatch(&factor, host_vector)` allocates a device table of
`Capacity() × B` entries and copies the vector into it.
`SetStatePointers(i, host_vector)` copies again (size must be
`NumActiveFactors() × B`). The rest of the pipeline only sees device tables.

### 5.5 Validation

Per-solve host validation of device data would need a readback and an O(N)
host walk, defeating the purpose. Instead:

- `Problem::Validate(cudaStream_t stream)` runs one GPU kernel over all active
  connections and one readback, and reports the first failure. It checks:
  every pointer lies in an active state of a registered state batch (pointer
  form) or every index is `< NumActiveStates()` (index form); every active
  constant id is `< n`; every active, non-constant state is referenced by at
  least one active factor (the current "unconstrained state" rule).
- `MinimizerOptions::validate_problem` (default `false`) calls it at the start
  of every `Minimize`, for debugging.
- `CheckConsistency()` keeps its current behavior for host tables and calls
  `Validate` on the default stream for device tables.

Without validation, an out-of-range index or a dangling pointer is undefined
behavior, exactly like an out-of-range pointer passed to a factor today.

## 6. Usage examples

### 6.1 Sliding-window bundle adjustment (C++), everything changes each frame

```cpp
#include "cunls/cunls.h"

constexpr size_t kMaxPoses = 32, kMaxLandmarks = 50'000, kMaxObs = 400'000;

// ---- Once ----------------------------------------------------------------
cunls::dvector<cunls::SE3Transform> poses(kMaxPoses);
cunls::dvector<cunls::Vector<3>> landmarks(kMaxLandmarks);
cunls::dvector<cunls::Vector<2>> observations(kMaxObs);
cunls::dvector<int> obs_indices(2 * kMaxObs);        // [pose index, landmark index] per factor
cunls::dvector<int> fixed_poses(1);                  // oldest pose is the gauge anchor

cunls::SE3StateBatch pose_states(reinterpret_cast<float *>(poses.data()), kMaxPoses,
                                 fixed_poses.data(), /*const capacity=*/1);
cunls::VectorStateBatch<3> landmark_states(reinterpret_cast<float *>(landmarks.data()),
                                           kMaxLandmarks);
cunls::ReprojectionFactorBatch reprojection(observations.data(), kMaxObs, /*z_threshold=*/1e-3f);
cunls::CauchyLossFunctionBatch loss(/*b=*/1.0f, /*c=*/1.0f);

cunls::Problem problem;
problem.AddStateBatch(&pose_states);
problem.AddStateBatch(&landmark_states);
problem.AddFactorBatch(&reprojection, &loss, {&pose_states, &landmark_states},
                       obs_indices.data());

cunls::LevenbergMarquardtMinimizer minimizer;
cunls::CudaStream stream;

// ---- Every frame ---------------------------------------------------------
for (const Frame &frame : frames) {
  // 1. Rewrite contents in place on `stream` (here from the user's tracker,
  //    which produces device data; cudaMemcpyAsync from host works too).
  tracker.WriteWindow(stream.GetStream(), frame,
                      poses.data(), landmarks.data(),          // initial guesses
                      observations.data(), obs_indices.data(),  // measurements, connectivity
                      fixed_poses.data());                       // gauge: index of oldest pose
  // 2. Set the active sizes (host values the tracker already knows).
  pose_states.SetNumActiveStates(frame.num_poses, /*num_const_states=*/1);
  landmark_states.SetNumActiveStates(frame.num_landmarks);
  reprojection.SetNumActiveFactors(frame.num_observations);
  // 3. Solve; results land in poses[0..num_poses) and landmarks[0..num_landmarks).
  const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  tracker.ReadWindow(stream.GetStream(), poses.data(), landmarks.data());
}
```

### 6.2 Per-frame PnP with RANSAC: connectivity written once

Every PnP factor reads the one pose, so the index table is all zeros and never
changes. Only the matches and their count change.

```cpp
// ---- Once ----
cunls::dvector<cunls::SE3Transform> pose(1);
cunls::dvector<cunls::Vector<2>> obs(kMaxMatches);
cunls::dvector<cunls::Vector<3>> pts(kMaxMatches);
cunls::dvector<int> pose_index(kMaxMatches);
cudaMemset(pose_index.data(), 0, kMaxMatches * sizeof(int));  // every factor reads pose 0

cunls::SE3StateBatch pose_state(reinterpret_cast<float *>(pose.data()), 1);
cunls::PnPFactorBatch pnp(obs.data(), pts.data(), kMaxMatches);
cunls::Problem problem;
problem.AddStateBatch(&pose_state);
problem.AddFactorBatch(&pnp, {&pose_state}, pose_index.data());

cunls::RansacLevenbergMarquardtMinimizerOptions options;
options.base_options.factor_batches = {{cunls::RansacRole::kSampled, 0.01f}};
cunls::RansacLevenbergMarquardtMinimizer ransac(options);

// ---- Every frame ----
matcher.Match(stream, frame, obs.data(), pts.data(), &num_matches);  // device writes, host count
cudaMemcpyAsync(pose.data(), &predicted_pose, sizeof(cunls::SE3Transform),
                cudaMemcpyHostToDevice, stream);                      // initial guess
pnp.SetNumActiveFactors(num_matches);
const cunls::RansacSummary summary = ransac.Minimize(stream, problem);
// pose[0] holds the estimate; ransac.InlierMask(0) / InlierMaskSize(0) the inliers.
```

### 6.3 Connectivity built on the host

When data association runs on the CPU, the user keeps a pinned host mirror and
copies it into the bound table:

```cpp
cunls::dvector<int> obs_indices(2 * kMaxObs);
int *host_indices;  cudaMallocHost(&host_indices, 2 * kMaxObs * sizeof(int));
// ... once: problem.AddFactorBatch(&reprojection, {&poses, &landmarks}, obs_indices.data());

// every frame:
association.Fill(host_indices, &n);                                   // CPU
cudaMemcpyAsync(obs_indices.data(), host_indices, 2 * n * sizeof(int),
                cudaMemcpyHostToDevice, stream);
reprojection.SetNumActiveFactors(n);
minimizer.Minimize(stream, problem);
```

### 6.4 Python (CuPy)

```python
import cupy as cp
import pycunls

MAX_POSES, MAX_LANDMARKS, MAX_OBS = 32, 50_000, 400_000

# Once
poses_gpu = cp.zeros(MAX_POSES * 16, dtype=cp.float32)
landmarks_gpu = cp.zeros(MAX_LANDMARKS * 3, dtype=cp.float32)
obs_gpu = cp.zeros(MAX_OBS * 2, dtype=cp.float32)
indices_gpu = cp.zeros(MAX_OBS * 2, dtype=cp.int32)        # [pose, landmark] per factor
fixed_gpu = cp.zeros(1, dtype=cp.int32)

poses = pycunls.SE3StateBatch(poses_gpu, MAX_POSES, fixed_gpu, 1)
landmarks = pycunls.VectorStateBatch3(landmarks_gpu, MAX_LANDMARKS)
reprojection = pycunls.ReprojectionFactorBatch(obs_gpu, MAX_OBS, 1e-3)

problem = pycunls.Problem()
problem.add_state_batch(poses)
problem.add_state_batch(landmarks)
problem.add_factor_batch(reprojection, [poses, landmarks], indices_gpu)

minimizer = pycunls.LevenbergMarquardtMinimizer()
stream = pycunls.CudaStream()

# Every frame
for frame in frames:
    k, m, n = frame.num_poses, frame.num_landmarks, frame.num_obs
    poses_gpu[: k * 16] = cp.asarray(frame.poses.reshape(-1))      # in-place writes
    landmarks_gpu[: m * 3] = cp.asarray(frame.landmarks.reshape(-1))
    obs_gpu[: n * 2] = cp.asarray(frame.observations.reshape(-1))
    indices_gpu[: n * 2] = cp.asarray(frame.indices.reshape(-1))
    poses.set_num_active_states(k, num_const_states=1)
    landmarks.set_num_active_states(m)
    reprojection.set_num_active_factors(n)
    summary = minimizer.minimize(stream, problem)
```

(CuPy writes run on CuPy's current stream; issue them inside
`cupy_stream(stream.get_stream())` or synchronize before `minimize`.)

## 7. Performance

- **Kernels** launch over active sizes; capacity costs memory, not time.
- **Per-solve work** is today's minus three host→device uploads of the
  connectivity (`Σ N·B` pointers each), plus one elementwise kernel for index
  tables. Everything else (`Initialize`, structure build, iterations) is
  unchanged; making that incremental is `incremental_optimization.md`.
- **No allocations** in steady state: minimizer buffers keep capacity from the
  largest problem seen; factor and state scratch is reserved at construction.
- **No new synchronization.** Setters are host-only. Validation is opt-in.
- **In-kernel derived data** (§3.2): a few dozen flops per item in
  memory-bound kernels; expected neutral, confirmed per factor by the
  benchmark in §8 (SL(4)'s 4×4 inverse is the one to watch).

## 8. Implementation plan

1. **Base classes.** `FactorBatch` capacity/size members and setters;
   `SizedStateBatch` capacity, constant capacity, `SetNumActiveStates`.
   Tests: setters, bounds checks, `NumActiveFactors()` follows the setter.
2. **Derived data into kernels** (§3.2), one factor family at a time, each
   with a test that capacity-built factors evaluated at `n < capacity` match
   exact-size factors, and that rewriting measurements in place between two
   evaluations gives the new results. Micro-benchmark per family (old vs. new
   kernel time).
3. **Scratch at capacity** for factors and Lie-group states.
4. **Device connectivity.** `Problem` stores per residual batch a table
   descriptor `{kind: pointers | indices, device ptr, state batch per slot,
   owned storage for the host-vector overload}`. Port the five consumers
   (§5.3) to read it: `MinimizerState` remap, `HessianStructureBuilder`
   column resolution, `NumericDiffJacobianBuilder`, `RansacLayout`,
   validation.
5. **Validation kernel** and `MinimizerOptions::validate_problem`.
6. **Python bindings** for setters, capacities and both table forms.
7. **Frame-loop tests** (C++ and Python): a sequence of frames with changing
   sizes, measurements, states and connectivity, each solved through the
   bound buffers and compared with a freshly constructed problem; RANSAC
   included. Plus an allocation test: after the first frame, no `cudaMalloc`
   (count via a CUDA memory-pool hook or `cudaMemGetInfo`).
8. **Docs and examples**: a dynamic-problem example (sliding-window BA) in
   C++ and Python.

## 9. Relation to incremental optimization

`incremental_optimization.md` keeps the problem alive across solves and edits
it with activity masks so that `Minimize` does not rebuild the Hessian
structure. This design gives it its data model for free: user-owned slots at
capacity (D1, its D1), contents rewritten in place, and a device-side
connectivity table that its incremental structure updates can read. The
active sizes here correspond to its high-water marks.

## 10. Open questions

1. **Naming.** `SetNumActiveFactors` / `SetNumActiveStates` mirror the getters.
   Alternatives: `SetActiveSize`, `Resize` (rejected: implies allocation).
2. **Device-side sizes.** Sizes are host values, so a GPU-computed match count
   costs one readback. A later option: a device size pointer, with kernels
   launched over capacity and early exit; the minimizer would size for
   capacity. Needs the incremental design's high-water marks.
3. **Constant states.** Keep the constant-id list (proposed), or switch to a
   per-state constant mask that the user rewrites (simpler to update, one
   byte per state, matches the incremental design's `fixed[s]`)?
4. **Index width.** `int32` indices cap a state batch at 2³¹ states; enough?
5. **Mixed forms.** Should one residual batch allow some slots by index and
   others by pointer? Proposed: no; one form per batch.
6. **Deprecating virtual `NumActiveFactors()`.** Custom factors override it today.
   Keep it virtual, or make it final in the base after one release?

## 11. Implementation notes and measured performance

**Deviations from the proposal.**

- Constructor counts are renamed `capacity` (`const_capacity` for constant
  ids), and **the active counts start at 0**: a batch must be activated with
  `SetNumActiveFactors` / `SetNumActiveStates` before it is solved (user decision).
- `StateDevicePtr(i)` is valid for `i < Capacity()`, not only for active
  states, so the connectivity of the next solve can be built before resizing.
- Every minimizer runs `Problem::CheckSizes()` at the start of `Minimize`: a
  host-only loop over batches that throws when nothing is active, a count
  exceeds its capacity, or connectivity does not cover the active factors.
- The numeric-differentiation plan was cached per residual batch forever (a
  latent bug when a minimizer is reused). It is now refreshed every solve when
  the batch's active state pointers differ from the ones it was built from
  (`NumericDiffJacobianBuilder::Refresh`).
- `Problem::HostStatePointers(i)` returns a view: no copy for host lists.

**D3 (in-kernel derived data), A/B against `HEAD` on an RTX A6000.** All 9
affected factors produce **bitwise identical** residuals and Jacobians. Per
`Evaluate` call (median of 100, two interleaved runs, 10k/100k/1M factors,
residuals only and with Jacobians), no case is slower in both runs; notable
changes at 1M factors:

| Factor | base ms | D3 ms | change |
|---|---|---|---|
| SE3 prior (res / res+jac) | 0.68 / 1.72 | 0.60 / 1.65 | −11% / −4% |
| Sim3 prior (res) | 0.46 | 0.43 | −7% |
| SL4 prior (res) | 0.61 | 0.56 | −8% |
| Sim3 between (res+jac) | 6.14 | 5.70 | −7% |
| SL4 between (res+jac) | 14.27 | 12.25 | −14% |
| SL4 between constructor | 73 | 0.08 | precompute and sync removed |

What it took (Nsight Compute): the first version regressed up to +31%. Fixes:
read R and t of the SE(3) adjoint from global memory instead of dynamically
indexed local arrays (removed local-memory spills that doubled DRAM writes);
issue the measurement and state loads together before the inverse (the
reciprocal's slow-path branch had split them into two memory round trips);
`__frcp_rn` instead of IEEE division (same bits); 16-byte vector loads/stores
for 4x4 transforms; shared-memory staging so the Sim(2) error is written with
16-byte stores.

**Whole solves** (same GPU, interleaved): sparse bundle adjustment with PCG
−2% to −7%; dense LDLT unchanged; a 50k-pose pose graph +0.6% (median) over 8
runs, every kernel equal or faster per call: the difference is the PCG
iteration count, which varies run to run because the Hessian is assembled with
atomics. RANSAC PnP (100 to 1M points) and the multi-camera rig: within about
±1–2%.
