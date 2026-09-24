# Design: Constant-Velocity / Constant-Acceleration Motion-Prior Factors

Status: implemented (§3 factors, §6 tests items 1-2 and a linear-system
convergence check for SE(3); §4.5's fused combined state remains a
follow-up, not implemented)
Scope: `cunls/factor`, `cunls/state`, `cunls/math`

## 0. Implementation summary

All eight factors from §3 are implemented, tested (finite-difference
Jacobian checks + zero-residual checks for every group; an additional
linear-system Gauss-Newton convergence test for
`ConstantVelocitySE3FactorBatch`), and demonstrated end-to-end:

- `cunls/factor/constant_{velocity,acceleration}_{se3,so3,se2,so2}_factor_batch.{h,cu}`
- `tests/constant_{velocity,acceleration}_{se3,so3,se2,so2}_factor_batch_test.cpp`
- `cunls/factor/motion_prior_information.{h,cu}` — closed-form `Q(Δt)⁻¹`
  covariance fusion via the existing `InformationFactorBatch` wrapper, plus
  a `MotionPriorInformationFactorBatch<T, Dim>` convenience wrapper and 8
  named aliases (e.g. `ConstantVelocityInformationSE3FactorBatch`) so
  callers get a single drop-in constructor instead of hand-composing
  `InformationFactorBatch` themselves (§3.1); `tests/motion_prior_information_test.cpp`.
  Fixed a latent transpose bug in `InformationFactorBatch`'s residual
  weighting along the way (§3.1).
- `examples/motion_prior/main.cpp` (constant-velocity SE(3) chain)
- API docs: `docs/sphinx/api/factor.rst`, "Motion prior factors" section

No new math-library primitives were needed (§5's SE(2) gap was closed via
the `J_l^{-1}(x) = J_r^{-1}(-x)` identity instead of a new function, as
foreshadowed there); no changes to `Problem`, `StateBatchOps`,
`HessianStructureBuilder`, or `BlockHessianAssembler` were needed, matching
the §4.2 prediction.

## 1. Goal

Add two new factor types that constrain how a pose evolves between two
consecutive timestamps, given an explicit velocity (and, for the
constant-acceleration factor, an explicit acceleration) state:

1. **Constant-velocity (CV) factor** — connects `(pose_k, pose_{k+1}, vel_k,
   vel_{k+1})`.
2. **Constant-acceleration (CA) factor** — connects `(pose_k, pose_{k+1},
   vel_k, vel_{k+1}, accel_k, accel_{k+1})`.

Both must work for **SE(3), SE(2), SO(3), SO(2)** poses (rotation-only groups
use "velocity"/"acceleration" to mean angular velocity/acceleration in the
group's own tangent space).

The second half of this document answers the "combined state" question: can
`[pose, velocity]` / `[pose, velocity, acceleration]` be one state block, or
should they stay separate blocks connected by a factor, and can this be
generated automatically.

The theoretical basis (§2) is Tang/Yoon/Barfoot's WNOA/WNOJ motion-prior
paper; §2.3 and §4.5 draw on patterns observed in other C++ factor-graph
implementations of pose+velocity motion models, described generically below
without attribution to a specific codebase.

## 2. Theoretical basis

Reference: Tang, Yoon, Barfoot, *"A White-Noise-On-Jerk Motion Prior for
Continuous-Time Trajectory Estimation on SE(3)"*, RA-L 2019
([arXiv:1809.06518](https://arxiv.org/abs/1809.06518)). This paper is the
direct source for the CV factor and the structural basis for the CA factor.

### 2.1 What the paper actually does

The paper is a continuous-time trajectory-estimation (GP-regression /
STEAM) paper, not a discrete factor-graph paper, but it derives two
**discrete prior factors between consecutive knots** that are exactly what
we want, plus the continuous-time machinery used to justify their
information matrices. It is SE(3)-specific in its worked equations, but the
underlying template (linear time-invariant SDE with state-transition matrix
Φ and process noise Q, propagated through a Lie group's left Jacobian) is
group-agnostic.

**State per knot:**
- White-noise-on-acceleration (WNOA) model: `x = {T, ϖ}` — pose `T ∈ SE(3)`
  and **body-frame generalized velocity** `ϖ ∈ ℝ⁶` (linear + angular,
  defined by `Ṫ = ϖ^ T`, i.e. `ϖ^ = Ṫ T⁻¹`).
- White-noise-on-jerk (WNOJ) model: `x = {T, ϖ, ϖ̇}` — adds body-frame
  generalized **acceleration** `ϖ̇ ∈ ℝ⁶`.

Crucially: **in both models, pose and velocity(/acceleration) are separate
state variables**, each retracted with its own update rule — `T` via the
group's exponential map, `ϖ`/`ϖ̇` via plain vector addition (Eq. 10 in the
paper). The paper never fuses them into one manifold. This directly answers
part of question 2 below: the authors' own answer is "separate states
connected by a factor," not "one combined state."

**Verified (I read this part of the paper verbatim) CV/WNOA discrete prior
residual between consecutive knots `i` and `i+1`, `Δt = t_{i+1} - t_i`**
(paper's Eq. 16; `Jᵢ,ᵢ₊₁ := J_l(Log(T_{i+1} T_i^{-1}))`, the SE(3) left
Jacobian evaluated at the relative-pose log):

```
e = [ Log(T_{i+1} T_i^{-1}) - Δt · ϖ_i          ]   (pose block, ∈ tangent space)
    [ J_{i+1,i}^{-1} ϖ_{i+1} - ϖ_i               ]   (velocity block)
```

Reading: the relative pose's log should equal `Δt` times the body velocity
at `i`; and the velocity at `i+1`, *transported back into the local frame at
i* through the inverse left Jacobian, should equal the velocity at `i`. The
Mahalanobis weight is a closed-form `Q(Δt)⁻¹` (Eq. 15) that only depends on
`Δt` and a tunable process-noise PSD `Qc` — no integration needed at
runtime.

**WNOJ / CA structure (verified through Eq. 32; the paper's exact final
residual expression beyond that point was not reliably extracted — see
§2.2):** the state-transition matrix for `[ξ; ξ̇; ξ̈]` (relative-pose log and
its 1st/2nd derivatives) is the textbook constant-jerk Φ:

```
Φ(Δt) = [ 1   Δt·1   ½Δt²·1 ]
        [ 0    1      Δt·1  ]
        [ 0    0       1    ]
```

i.e. structurally a second-order Taylor/constant-acceleration prediction.

### 2.2 What we adopt vs. what we simplify

- **CV factor**: adopt the paper's Eq. 16 residual essentially verbatim
  (§3.1). This part is fully verified against the paper text.
- **CA factor**: the paper's *exact* WNOJ global-state residual involves a
  "curlywedge" correction term (`(·)^⋏`) coupling the transport of
  acceleration through the *derivative* of the left Jacobian, which is not
  reliably known from this research pass (low-confidence secondary
  extraction only). Rather than guess at unverified coefficients, we define
  our own CA residual (§3.2) as the direct discrete analogue of the verified
  Φ matrix above — i.e. we transport velocity and acceleration through the
  same inverse-left-Jacobian operator the CV factor already uses, instead of
  the paper's more exact (but unconfirmed here) curlywedge term. This is a
  documented, deliberate simplification, not a literal reproduction of the
  paper's WNOJ equations. It is first-order-consistent with the paper's
  Φ/Q structure and should be revisited (re-derive from the primary PDF, pp.
  5-6) if higher fidelity to the exact WNOJ paper is required later.
- We do **not** implement the continuous-time GP interpolation machinery
  (Eqs. 21-23, the paper's main contribution) — only the discrete prior
  factor between consecutive knots, which is all a discrete factor-graph
  optimizer needs.

### 2.3 Patterns observed in other factor-graph implementations

Other C++ factor-graph libraries that model navigation state (pose +
velocity, sometimes + IMU bias) offer useful comparison points for both the
factor residual (§3) and the combined-state question (§4):

**A "constant-velocity between-state" factor** is a common pattern: a binary
factor between two combined pose+velocity states plus a scalar `dt`, of the
shape `error = predict(x1, dt) ⊟ x2`, where `predict` integrates the current
velocity forward by `dt` and leaves velocity unchanged. One widely-seen
implementation of this pattern is **structurally weaker than the WNOA
paper's residual we adopt in §3.1**: it hardcodes angular velocity to zero
in the prediction step, so rotation is never driven by a velocity state at
all — only *position* is constrained via *linear* velocity, leaving rotation
an independent, only weakly-coupled degree of freedom. Our §3.1 design
instead ties rotation directly to a genuine body-twist/angular-velocity
state (as the WNOA paper does), which is required since we explicitly want
CV/CA support for pure-rotation groups (SO(2)/SO(3)) where "velocity" *is*
the angular velocity — a zero-angular-velocity shortcut wouldn't make sense
there. Keep §3.1's twist-based residual; the simpler position-only variant
is worth keeping in mind as a cheaper fallback if angular-velocity coupling
turns out not to be needed.

**A dedicated "constant-acceleration" factor is not a common pattern.**
Acceleration is more often handled via IMU-preintegration factor families,
which integrate real measured accel/gyro readings between keyframes and
estimate a slowly-varying *bias*, rather than treating acceleration as an
explicit per-knot state variable the way our CA factor and the WNOJ paper
do. This is a different paradigm (measurement-driven preintegration vs. an
acceleration-state prior) with no directly reusable residual for us — it
reinforces, rather than resolves, the §2.2 caveat that our CA residual is
our own discretization, not a literal port of an existing implementation.

**A combined pose+velocity state type** is exactly the "combined state"
from question 2 — see §4.5 below, where the underlying generic
group-theoretic construction behind it turns out to be directly relevant to
the "can combined states be generated automatically" question.

## 3. Factor definitions

### 3.1 Constant-velocity factor

**States touched (4):** `pose_k`, `pose_{k+1}`, `vel_k`, `vel_{k+1}`.

**Residual** (tangent-space dimension = pose tangent dim + velocity dim,
e.g. 12 for SE(3)/velocity ∈ ℝ⁶, 6 for SE(2)/ℝ³, 4 for SO(3)/ℝ³ pose won't
match — see note below):

```
r_pose = Log( pose_k^{-1} · pose_{k+1} ) - Δt · vel_k
r_vel  = J_l^{-1}( Log(pose_k^{-1} · pose_{k+1}) ) · vel_{k+1} - vel_k
```

where `J_l^{-1}(·)` is the group's inverse-left-Jacobian (already
implemented for SO(3)/SE(3); needs adding for SE(2), trivial/identity for
SO(2) — see §5).

**Note on rotation-only groups (SO(2)/SO(3)):** there `vel` is angular
velocity in the group's own tangent space (`ℝ¹` for SO(2), `ℝ³` for SO(3)),
so `r_pose` and `r_vel` have the same dimension as the pose tangent — no
mismatch. For SE(2)/SE(3), velocity is the full body twist, same dimension
as the pose tangent, too. So in **all four groups**, `dim(vel) ==
dim(pose tangent)`, and the residual dimension is simply `2 * pose_tangent_dim`
(2, 6, 6, 12 for SO(2), SO(3)/SE(2), SE(3) respectively — SO(3) and SE(2)
both happen to have tangent dim 3).

**Jacobians:** with respect to `pose_k`, `pose_{k+1}`, `vel_k`,
`vel_{k+1}`. The pose-pose block is structurally identical to
`SE3BetweenFactorBatch`'s Jacobian (`Ad`, `J_l^{-1}`, `J_r^{-1}` of the
relative-pose log — see `cunls/factor/se3_between_factor_batch.cu:135-291,
300-445` for the closed-form pattern to reuse/adapt), the velocity blocks
are `±Identity` / `±J_l^{-1}` — all closed-form, no numerical
differentiation needed.

**Template declaration** (tangent sizes, following the
`ReprojectionFactorBatch`/`SE3BetweenFactorBatch` convention of declaring
*tangent*, not ambient, sizes):

```cpp
// SE(3): pose tangent 6, velocity ∈ R^6
class ConstantVelocitySE3FactorBatch
    : public SizedFactorBatch</*residual=*/12, /*pose_k=*/6, /*pose_k1=*/6,
                              /*vel_k=*/6, /*vel_k1=*/6> { ... };

// SE(2): pose tangent 3, velocity (body twist) ∈ R^3
class ConstantVelocitySE2FactorBatch
    : public SizedFactorBatch</*residual=*/6, 3, 3, 3, 3> { ... };

// SO(3): pose tangent 3, angular velocity ∈ R^3
class ConstantVelocitySO3FactorBatch
    : public SizedFactorBatch</*residual=*/6, 3, 3, 3, 3> { ... };

// SO(2): pose tangent 1, angular velocity ∈ R^1
class ConstantVelocitySO2FactorBatch
    : public SizedFactorBatch</*residual=*/2, 1, 1, 1, 1> { ... };
```

Each factor batch stores a per-factor `Δt` (device array, one float per
factor instance — timestamps are rarely uniform).

**Covariance fusion (implemented).** The factor itself is unweighted (unit
information); the paper's closed-form `Q(Δt)⁻¹` (Eq. 15 for CV, Eq. 32 for
CA) is fused into the residual and Jacobian — not inside the factor kernel,
but by composing it with the existing generic `InformationFactorBatch<T>`
wrapper (`cunls/factor/information_factor_batch.h`), which already
implements exactly "`r' = S·r`, `J' = S·J`" so that `J'ᵀJ' = JᵀInformation·J`
and `J'ᵀr' = JᵀInformation·r` come out correct in the normal equations.

Requiring every caller to hand-compute `S` and manually compose it with
`InformationFactorBatch<T>` turned out to be a bad API in practice (real
feedback: "nobody really wants to call those functions on their own") — so
`cunls/factor/motion_prior_information.h` provides a **generic convenience
wrapper**, `MotionPriorInformationFactorBatch<T, Dim>`, that computes `S`
internally and composes it with `InformationFactorBatch<T>` itself, plus 8
named aliases (`ConstantVelocityInformationSE3FactorBatch`, ...,
`ConstantAccelerationInformationSO2FactorBatch`) so the common case is a
single drop-in constructor call:

```cpp
ConstantVelocityInformationSE3FactorBatch factor(
    cublas_handle, stream, dt_ptr, qc_diag_ptr, num_factors);
```

Still just **one** template implementation (not 8 separate classes) —
`MotionPriorInformationFactorBatch<T, Dim>` uses the "base-from-member"
idiom (a private base-class member computes and owns the `S` buffer, listed
before the `InformationFactorBatch<T>` base in the inheritance list, so it's
guaranteed fully constructed first) to own the sqrt-information buffer for
the lifetime of the object, since `InformationFactorBatch<T>` itself only
stores a non-owning pointer. The two lower-level function *templates*,
`ComputeConstantVelocitySqrtInformation<Dim>` and
`ComputeConstantAccelerationSqrtInformation<Dim>` (covering all four groups
via `Dim = 1, 3, 6`), remain available directly for callers who want the raw
`S` matrix without going through `InformationFactorBatch` at all.

Deriving `S` (the square-root information `InformationFactorBatch` expects,
satisfying `SᵀS = Q(Δt)⁻¹`) in closed form, with no numerical Cholesky, uses
one key structural fact: `Q(Δt)⁻¹` is a **Kronecker product**
`M(Δt) ⊗ Qc⁻¹` of a tiny 2×2 (CV) or 3×3 (CA) *scalar* matrix `M(Δt)` — the
paper's Eq. 15 / Eq. 32, read as "the same coefficient repeated once per
tangent DOF" — with the (assumed diagonal) continuous-time process-noise
PSD `Qc` (one entry per DOF, constant across the whole batch — a property of
the noise process, not of any individual factor/timestep — while `Δt`
varies per factor). Since `chol(A⊗B) = chol(A) ⊗ chol(B)` for SPD `A, B`,
and `Qc` diagonal makes `chol(Qc⁻¹)` trivially `diag(1/√qc)`, only `M(Δt)`'s
tiny Cholesky factor needs deriving — and it has closed form:

```
CV: L_M(Δt) = [ 2√3·Δt^-1.5        0    ]
              [ -√3·Δt^-0.5    Δt^-0.5  ]

CA: L_M(Δt) = [ 12√5·Δt^-2.5      0            0      ]
              [ -6√5·Δt^-1.5   2√3·Δt^-1.5      0      ]
              [  √5·Δt^-0.5    -√3·Δt^-0.5   Δt^-0.5   ]
```

(verified: `L_M L_Mᵀ == M` by direct symbolic expansion). `S = L_Mᵀ ⊗ diag(1/√qc)`
is then computed directly per factor — a handful of FLOPs, no dense linear
algebra — and passed to `InformationFactorBatch<ConstantVelocitySE3FactorBatch>`
(etc.) as the `sqrt_information` argument. See
`cunls/factor/motion_prior_information.h` for the exact formulas and
`tests/motion_prior_information_test.cpp` for verification against the
paper's `Q(Δt)⁻¹` directly.

**Bug found and fixed along the way**: `ApplyInformationToResiduals` (used
by `InformationFactorBatch`) applied `Sᵀ·r` while `ApplyInformationToJacobians`
applied `S·J` — an inconsistent transpose between the two (traced to a
`CUBLAS_OP_N` that should have been `CUBLAS_OP_T`, given cuBLAS reads our
row-major `Matrix<N>` buffers column-major). This was invisible before
because every existing caller only ever passed an identity matrix
(symmetric, so the bug had no observable effect); it surfaces immediately
with a genuinely non-symmetric `S` like a Cholesky factor. Fixed in
`cunls/factor/information_factor_batch.cpp`.

### 3.2 Constant-acceleration factor

**States touched (6):** `pose_k`, `pose_{k+1}`, `vel_k`, `vel_{k+1}`,
`accel_k`, `accel_{k+1}`.

**Residual** (our discretization of the verified Φ structure in §2.1; see
§2.2 for the caveat versus the paper's exact WNOJ term):

```
r_pose  = Log( pose_k^{-1} · pose_{k+1} ) - Δt · vel_k - ½Δt² · accel_k
r_vel   = J_l^{-1}( Log(pose_k^{-1} · pose_{k+1}) ) · vel_{k+1} - vel_k - Δt · accel_k
r_accel = J_l^{-1}( Log(pose_k^{-1} · pose_{k+1}) ) · accel_{k+1} - accel_k
```

`r_accel` reuses the same transport operator as `r_vel` (applying the
inverse left Jacobian of the relative-pose log to carry the endpoint value
back into the start frame) — this is the specific simplification flagged in
§2.2.

**Template declaration:** same pattern as §3.1 but with 6 state blocks and
residual dimension `3 * pose_tangent_dim` (18 for SE(3), 9 for
SE(2)/SO(3), 3 for SO(2)):

```cpp
class ConstantAccelerationSE3FactorBatch
    : public SizedFactorBatch</*residual=*/18, 6, 6, 6, 6, 6, 6> { ... };
```

### 3.3 Implementation plan (files to add)

Mirror the existing per-group file layout exactly (one header + one `.cu`
per factor per group, following `se3_between_factor_batch.h/.cu` as the
primary template for the pose-Jacobian machinery, and
`reprojection_factor_batch.h/.cu` for the "factor touches N heterogeneous
state blocks" wiring pattern):

```
cunls/factor/
  constant_velocity_se3_factor_batch.h / .cu
  constant_velocity_se2_factor_batch.h / .cu
  constant_velocity_so3_factor_batch.h / .cu
  constant_velocity_so2_factor_batch.h / .cu
  constant_acceleration_se3_factor_batch.h / .cu
  constant_acceleration_se2_factor_batch.h / .cu
  constant_acceleration_so3_factor_batch.h / .cu
  constant_acceleration_so2_factor_batch.h / .cu
```

Each `Evaluate()` kernel:
1. Reads `state_pointers[...]` positionally (hardcoded ambient offsets per
   block, exactly like `reprojection_fused_kernel`,
   `cunls/factor/reprojection_factor_batch.cu:71-187`, which already mixes
   a 16-float SE3 ambient block with a 3-float vector block in one kernel).
2. Computes `Log(pose_k^{-1} pose_{k+1})` via the group's existing
   `ComputeLogXXX` (already implemented for all four groups).
3. Computes `J_l^{-1}` at that log via the group's existing
   `ComputeJacobianLeftInverseXXX` (already implemented for SO(3)/SE(3);
   needs adding for SE(2)/SO(2), see §5).
4. Assembles the residual and, if requested, the dense Jacobian block in
   the state-block order declared by the `SizedFactorBatch<...>` template
   parameters (same convention as every existing factor).

No changes are needed to `FactorBatch`, `SizedFactorBatch`, `Problem`,
`HessianStructureBuilder`, or `BlockHessianAssembler` — see §4.

Registering a chain of `N` knots in a `Problem`:

```cpp
Problem problem;
problem.AddStateBatch(&poses);      // SE3StateBatch, N blocks
problem.AddStateBatch(&velocities); // VectorStateBatch<6>, N blocks
// ... construct a ConstantVelocitySE3FactorBatch with N-1 factors ...
std::vector<float *> state_pointers;
for (size_t k = 0; k + 1 < N; ++k) {
  state_pointers.push_back(poses.StateBlockDevicePtr(k));
  state_pointers.push_back(poses.StateBlockDevicePtr(k + 1));
  state_pointers.push_back(velocities.StateBlockDevicePtr(k));
  state_pointers.push_back(velocities.StateBlockDevicePtr(k + 1));
}
problem.AddFactorBatch(&cv_factors, state_pointers);
```

This is exactly the existing `AddFactorBatch`/`state_pointers` convention
(`cunls/minimizer/problem.h:56-74`) — no new API surface.

## 4. Separate states vs. one combined state

### 4.1 Question restated

Should `[pose, velocity]` (and `[pose, velocity, acceleration]`) be:

(a) **separate `StateBatch` instances** (e.g. `SE3StateBatch` +
    `VectorStateBatch<6>`), connected by the CV/CA factors above, or

(b) **one combined `StateBatch`**, e.g. a single 22-float-ambient /
    12-tangent block for `[SE3 pose, R^6 velocity]`?

And: can (b) be generated automatically (a generic "product manifold" state
type), rather than hand-written per combination?

### 4.2 Recommendation: (a), separate state batches

This is also what the reference paper itself does (§2.1) — pose and
body velocity/acceleration are always distinct state variables in their
formulation, never fused.

**Architectural reasons, from the codebase survey:**

- `Problem`/`StateBatchOps`/`HessianStructureBuilder` are **already fully
  general over heterogeneous state batches and multi-block factors**: state
  batches are stored as `std::vector<StateBatch*>` (type-erased,
  `cunls/minimizer/problem.h`), factors declare an arbitrary list of tangent
  sizes via variadic `SizedFactorBatch<residual, size0, size1, ...>`
  (`cunls/factor/sized_factor_batch.h`), and Hessian columns are resolved
  purely by matching a factor's raw device pointer against each registered
  state batch's own address range and `AmbientSize()`/`TangentSize()`
  (`cunls/minimizer/hessian_structure.cu:334-407`). Bundle adjustment
  already exercises exactly this path today (SE3 poses + `Vector<3>`
  landmarks via `ReprojectionFactorBatch`). **Option (a) requires zero
  changes to `Problem`, `StateBatchOps`, `HessianStructureBuilder`, or
  `BlockHessianAssembler`** — only new `FactorBatch` and (already-existing,
  for velocity/acceleration — `VectorStateBatch<Dim>`) `StateBatch` classes.
- Option (b) is **not free**. Every existing `StateBatch` leaf
  (`SE3StateBatch`, `SO3StateBatch`, ...) is a hand-written class with one
  bespoke, fully-unrolled `Plus` CUDA kernel per manifold
  (`cunls/state/se3_state_batch.cu:33-97` is the canonical example); there
  is no generic "apply manifold M's Plus to sub-range `[a, b)` of a bigger
  block" primitive, and no `ProductStateBatch<M1, M2, ...>` template exists
  today. A combined `[pose, velocity]` block's `Plus` kernel would need to
  apply the group's `Exp`-retraction to the pose sub-range and plain vector
  addition to the velocity sub-range within *one* fused kernel — new,
  non-reusable code, one such class per group (SE3+vel, SE3+vel+accel,
  SE2+vel, ..., 8 combinations for this task alone).

### 4.3 Can it be automated ("compose states automatically")? — naive approach

A naive generic composition (bolt unrelated manifolds together, apply each
one's own `Plus` to its own sub-range of one bigger block) is possible but
is itself new, unprincipled infrastructure. §4.5 below describes a
better-founded alternative, based on recognizing that pose+velocity(+accel)
is genuinely one Lie group with real closed-form group structure, rather
than a grab-bag of independently-retracted sub-blocks — that construction is
worth building instead of the naive one.

### 4.4 `VectorStateBatch<Dim>` velocity/acceleration state — no new infra

Regardless of the (a)-vs-(b) decision, the *velocity* and *acceleration*
state itself needs no new code: `cunls/state/vector_state_batch.h`
(`VectorStateBatch<Dim>`) already exists and is exactly "a batch of plain
`ℝ^Dim` state blocks, `Plus` = elementwise add" — precisely what a body
twist/angular-velocity/acceleration state is. Use `VectorStateBatch<6>` for
SE(3) twists, `VectorStateBatch<3>` for SE(2)/SO(3), `VectorStateBatch<1>`
for SO(2).

### 4.5 A better-founded combined state: `SE_K(3)`-style product groups

Some implementations model a combined pose+velocity state not as an ad hoc
struct but as a genuinely generic **matrix Lie group `SE_K(3)`**: the
semidirect product of `SO(3)` with `K` copies of `ℝ³`, state
`(R, x_1, ..., x_K)`, tangent `[ω, ρ_1, ..., ρ_K] ∈ ℝ^{3+3K}`. Pose+velocity
is the `K=2` instance; **`K=3` is exactly `[pose, velocity, acceleration]`**
— a natural, well-defined extension of the same construction, not a
hypothetical. This is a materially better answer to "can combined states be
generated automatically" than the naive sketch in §4.3: instead of composing
*unrelated* manifold operations sub-block-by-sub-block, `SE_K(3)` recognizes
that pose+velocity(+accel) genuinely *is* one Lie group, with one clean
closed-form `Exp`/`Log`/`Ad` family, generic in `K`:

```
Exp([ω; ρ_1; ...; ρ_K]) = (Exp_SO3(ω), Jl(ω)·ρ_1, ..., Jl(ω)·ρ_K)
Log(R, x_1, ..., x_K)   = [Log_SO3(R); Jl(ω)⁻¹·x_1; ...; Jl(ω)⁻¹·x_K]
Ad(R, x_1..x_K) block-row i (i=1..K): [skew(x_i)·R  |  0 .. R .. 0]
composition: (R, x_i) * (R', x_i') = (R·R', {x_i + R·x_i'})
```

Every column `x_i` is transformed by the *same* `Jl(ω)`/`Jl(ω)⁻¹` already
used for the ordinary SE(3) translation block — i.e. `SE_K(3)`'s `Exp`/`Log`
is literally "do what `SE3StateBatch`'s existing fused kernel already does
for the translation column (`cunls/state/se3_state_batch.cu:33-97`, which
the architecture survey confirms already uses closed-form Rodrigues + the
SO(3) left Jacobian for that column), once per extra column." Concretely,
for cuNLS this would look like:

```cpp
// Ambient: 9 (R, row-major 3x3) + 3K (K columns of R^3) = 9 + 3K floats.
// Tangent: 3 (ω) + 3K (ρ_1..ρ_K)                          = 3 + 3K.
template <int K>
class SE_K3StateBatch : public SizedStateBatch<9 + 3 * K, 3 + 3 * K> {
  // Plus(): one fused kernel — Rodrigues + Jl(ω) exactly as
  // se3_state_batch.cu already computes for its translation block, run K
  // times (once per column) instead of once.
};
using PoseVelocityStateBatch = SE_K3StateBatch<2>;              // K=2: [pose, vel]
using PoseVelocityAccelStateBatch = SE_K3StateBatch<3>;         // K=3: [pose, vel, accel]
```

This is a **credible, precedented "phase 2" option**, not a hand-wave: it
reuses the exact math primitives cuNLS already has (`ComputeExpSO3`,
`ComputeJacobianLeftSO3`/`...InverseSO3`), generalizes the existing
`SE3StateBatch` kernel pattern by a loop over `K` columns instead of writing
unrelated per-manifold `Plus` calls, and is analogous for `SE_K(2)` (K
copies of `ℝ²` attached to `SO(2)`) for the SE(2)/SO(2) side. It is still
**new code** (a new templated `StateBatch` + new CV/CA factors that consume
its layout), so it does not change the §4.2 recommendation for *this* task,
but it is the right target to build toward if/when a fused
`[pose, velocity, acceleration]` state is actually wanted, rather than the
weaker "just glue arbitrary sub-block Plus calls together" idea originally
sketched in §4.3.

One caveat worth carrying over: some implementations of this construction
default their optimizer retraction to something *cheaper* than the full
`SE_K(3)` group `Expmap`/`Logmap` — a component-wise chart (rotation via
`Exp_SO3`; position/velocity via "rotate the tangent delta into world frame
by the current attitude and add"), with the true group Expmap available only
as an opt-in variant. If cuNLS ever implements `SE_K3StateBatch`, the same
tradeoff applies: the exact `Jl(ω)` group retraction is more faithful but
costs one SO(3) Jacobian evaluation per column per `Plus` call; a
component-wise chart (rotate-and-add, no `Jl(ω)`) is cheaper — worth
deciding empirically rather than assuming the "mathematically purest" option
is also the best performance/accuracy tradeoff.

### 4.6 Decision

Use **separate state batches** (`XxxStateBatch` for pose,
`VectorStateBatch<pose_tangent_dim>` for velocity, another
`VectorStateBatch<pose_tangent_dim>` for acceleration), connected by the
CV/CA factors of §3, for this task. Revisit a fused `SE_K3StateBatch`-style
combined state (§4.5) only if profiling shows the extra state-batch
bookkeeping (vs. one fused array) is an actual bottleneck.

## 5. New math-library primitives required

Everything needed for SO(3)/SE(3) already exists
(`cunls/math/so_se_lie_math.h`: `ComputeExpSO3/SE3`, `ComputeLogSO3/SE3`,
`ComputeAdjointSE3`, `ComputeInverseAdjointSE3`, and all four of
`ComputeJacobian{Left,Right}{,Inverse}{SO3,SE3}`).

Gaps to fill for SE(2)/SO(2):

| Group | Have | Missing |
|---|---|---|
| SO(2) | `ComputeExpSO2`, `ComputeLogSO2`, `ComputeTransposeSO2` | None — SO(2) is abelian, so `J_l = J_r = 1` (identity/no-op) and `Adjoint = 1`; the CV/CA residuals for SO(2) reduce to plain scalar arithmetic, no new primitive needed. |
| SE(2) | `ComputeExpSE2`, `ComputeLogSE2`, `ComputeInverseSE2`, `ComputeJacobianRightInverseSE2` | `ComputeJacobianLeftInverseSE2` (the CV/CA residual as written uses `J_l^{-1}`; SE(2)'s left Jacobian has a simple closed form since `so(2)` is 1-D/abelian — no infinite series needed, unlike SE(3)). No `ComputeAdjointSE2` exists either; needed for the pose-pose Jacobian block of the CV/CA factor (mirroring how `SE3BetweenFactorBatch` uses `ComputeAdjointSE3` for its `J_left` block). |

Both are small, closed-form additions (SE(2)'s Lie algebra structure is far
simpler than SE(3)'s — no Bernoulli/BCH series needed), following the
existing pattern in `cunls/math/so_se_lie_math.cu` for the other SE(2)
Jacobian.

## 6. Testing plan

Mirror the existing test patterns (`tests/se3_between_factor_batch_test.cpp`
-style, plus `tests/synthetic_pgo_test.cpp` for end-to-end usage):

1. **Unit tests per group** (`constant_velocity_<group>_factor_batch_test.cpp`,
   analogous for acceleration): finite-difference check of every analytic
   Jacobian block against numerical differentiation, for random poses,
   velocities, (accelerations), and `Δt`.
2. **Zero-residual sanity check**: given `pose_{k+1} = pose_k ⊞ (Δt · vel_k)`
   exactly (and, for CA, `vel_{k+1}`/`accel_{k+1}` consistent with constant
   acceleration), the residual must be ~0.
3. **Synthetic trajectory test**: generate a constant-velocity (resp.
   constant-acceleration) ground-truth trajectory, add noise to poses only,
   solve a small `Problem` with CV (resp. CA) factors plus pose priors, and
   check recovered velocities/accelerations converge to ground truth —
   analogous to `tests/synthetic_pgo_test.cpp`.
4. **Mixed-manifold connectivity test**: confirm `Problem::CheckConsistency()`
   accepts a graph mixing an `SE3StateBatch`, two `VectorStateBatch<6>`s, and
   a `ConstantVelocitySE3FactorBatch` with no changes needed
   (`cunls/minimizer/problem.cpp:74-181`), i.e. a regression test that the
   "zero minimizer changes" claim in §3.3/§4.2 actually holds.

## 7. Open questions for follow-up (not blocking a first implementation)

- ~~Whether to weight the CV/CA residual with the paper's closed-form
  `Q(Δt)⁻¹`~~ — **done**: `cunls/factor/motion_prior_information.h` +
  `InformationFactorBatch` (§3.1).
- Whether to re-derive the exact WNOJ curlywedge residual term from the
  primary source PDF for a higher-fidelity CA factor (§2.2), if the
  simplified transport-based residual proves numerically insufficient in
  practice.
- Whether a fused `SE_K3StateBatch` combined state (§4.5) is worth building
  later for memory-locality reasons, and if so, whether its `Plus` should
  use the exact group `Expmap` or a cheaper component-wise chart (§4.5).
- Whether the simpler position-only constant-velocity variant (angular
  velocity fixed at zero, §2.3) is worth offering as a cheaper alternative
  CV factor alongside the twist-based one in §3.1.
