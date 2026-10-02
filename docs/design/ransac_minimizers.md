# RANSAC Minimizers: `RansacGaussNewtonMinimizer` and `RansacLevenbergMarquardtMinimizer`

Status: phases 1–2 implemented (generic path). LO-RANSAC from phase 3 was
implemented, measured, and removed (see below).
Written against the code at `de4e9cb`. Claims about current behavior cite the
file they were read from. Measured performance is in §15.

**Implementation status and deviations from the design below:**

- Public API in `cunls/minimizer/ransac_minimizer.{h,cpp}`; internals in
  `cunls/minimizer/ransac/`: `RansacLayout` (validated problem metadata),
  `SlotSet` (slots with per-slot state replicas, evaluation buffers, minimal
  samples and solver state; one GN/LM iteration for all slots),
  `HypothesisScorer` (chunked / two-stage scoring), `RansacContext` (rounds,
  LO, refinement), and kernels grouped by topic
  (`normal_equations_kernels.cu`, `dense_solve_kernels.cu`,
  `scoring_kernels.cu`, `slot_kernels.cu`). Tests in
  `tests/ransac_minimizer_test.cpp`, benchmarks in
  `tests/ransac_benchmark_test.cpp` (`CUNLS_RANSAC_BENCHMARK=1`).
- Sampling uses the keyed Feistel permutation directly (splitmix64 keys), not
  Philox. The permutation *is* the sampler, so no separate RNG state is needed.
- Kernels use no atomics and no serial thread-0 loops. Normal equations:
  warp-per-slot for small systems, block-per-slot otherwise, Jacobian rows
  gathered into local columns, H accumulated in 4x4 register tiles, large
  slots split across blocks with a deterministic second-pass sum. Dense
  solve: warp per slot for `D <= 32`, 128-thread block above; pivot search and
  substitutions on one warp. Reductions use warp shuffles. The system is
  **Jacobi-equilibrated** before factorization, so the rank tolerance doesn't
  depend on variable scaling. Without it, a focal length next to a pose looked
  rank-deficient. The default solver is `kLDLT`, not `kCholesky`, because
  minimal samples are often near-singular.
- **New: informative inliers** (`require_informative_inliers`, default on). A
  factor counts as an inlier only if its Jacobian is non-zero on a free block.
  `PnPFactorBatch` reports zero residual and zero Jacobian for points behind the
  camera, so without this a hypothesis that flips the scene behind the camera
  scored as a perfect fit with every factor an inlier. Tests caught this at 30%
  outliers and above. It costs one Jacobian evaluation per scored factor.
- **New: the refinement never makes things worse.** If the final refinement
  scores worse than the best hypothesis, the best hypothesis is returned
  (`RansacSummary::refinement_reverted`).
- **No local optimization (LO-RANSAC removed).** Measured on the benchmarks
  (PnP 100–1M points, 0–90% uniform outliers, coherent outliers, small and
  large initial errors): disabling LO and scoring 1 finalist instead of 16
  left every success rate (100%) and inlier mask unchanged, while RANSAC-GN at
  1M points went 64 -> 14 ms. The final refinement on the winner's inliers
  already does what LO did. Option `lo_*` fields are gone;
  `scoring_finalists` defaults to 4. LO may still help with high noise or
  near-degenerate data; no current test shows it.
- No CUDA graphs (removed; see §4.2).
- **Large-problem paths:** two-stage scoring (`scoring_subset_size`,
  `scoring_finalists`: every hypothesis scored on a random 16k-factor subset,
  the best 4 on all factors) and convergence-based early exit of the
  refinement (one readback per iteration, enabled once an iteration touches
  >= 2^17 factor evaluations).
- **Item / replica parameters of `Evaluate` and `Plus` (replace the
  optional `EvaluateIndexed` / `PlusReplicated` of the first implementation,
  §15):** `FactorBatch::Evaluate(res, jac, ptrs, stream, factor_ids = nullptr,
  num_factor_ids = 0)` evaluates n items, item t being factor `factor_ids[t]`
  (or `t % NumActiveFactors()`) against its own pointer row, and
  `StateBatch::Plus(x, delta, out, stream, num_replicas = 1)` processes
  contiguous copies of the batch. The defaults give the old behavior. Every
  built-in factor and state implements them (bitwise equal to plain
  evaluation, tested per factor in `tests/evaluate_items_*_test.cpp`);
  `ResidualBatch::Evaluate` forwards them and applies the loss per item. RANSAC
  requires them and makes one call per batch for all slots, with no fallback.
- **Minimal samples are per slot, not waves (§4.1 superseded):** slot p's
  sample is the first s elements of the keyed permutation
  `PermutationKey(seed, round, p)`, and one `Evaluate` per sampled batch
  evaluates every slot's sample through factor ids. Samples of different
  hypotheses may overlap, i.e. hypotheses are independent.
- Not implemented yet: internal fast paths for built-ins (§4.4), linearized
  always-on factors (§4.3), numeric Jacobians (§4.6, rejected with an error),
  PROSAC ordering, the hypothesis generator hook (§11.3), and a status code
  for "no model" (§13 Q1).

## 0. Goal

Two new minimizers that robustly fit a problem containing outliers by RANSAC,
then refine on the inliers:

- `RansacGaussNewtonMinimizer`: hypotheses and refinement use Gauss-Newton.
- `RansacLevenbergMarquardtMinimizer`: the same, with Levenberg-Marquardt
  damping per hypothesis.

Requirements:

- **R1: no change to the user-facing contracts.** They consume an ordinary
  `Problem` built from existing `FactorBatch`, `StateBatch` and
  `LossFunctionBatch` objects (built-in, custom C++, or Python/Warp) through
  the **existing** `Evaluate` and `Plus` methods only. `FactorBatch`,
  `StateBatch`, `LossFunctionBatch`, `WarpFactorBatch` and `WarpStateBatch`
  gain no new virtual methods, and users implement nothing new (§4).
- **R2: the only limit is the total state dimension.** The optimized state
  can be any combination of supported state types (`SE3`, `SO3`, `SE2`,
  `SO2`, `Similarity2/3`, `SL4`, `Vector<N>`, custom C++ or Warp states), in any
  number of state batches. Any number of factor batches of any type, each with
  any number of factors, can constrain it. The single restriction is on the
  total **free** tangent dimension `D` (sum of `TangentSize()` over all
  non-constant states, across all state batches), which must not exceed
  `kMaxRansacTangentDim` (§3.3). Otherwise the minimizer fails with an
  actionable error. Constant states don't count toward `D`, whatever
  their number.
- **R3: their own options, a restricted linear solver set.** Per-hypothesis
  normal equations are tiny and dense, so only batched in-kernel dense solvers
  are offered (§7).
- **R4: efficient.** All hypotheses of a round are processed together, with no
  host synchronization inside a round (§9). No GPU sorting and no thrust
  anywhere in the new code.

Typical uses: PnP / camera resection with 2D–3D outliers, rig or multi-camera
pose from several PnP batches, pose + intrinsics, point-cloud registration from
putative correspondences, relative pose with a motion prior, triangulation of a
point from noisy rays.

Non-goals: RANSAC for large problems (SBA, PGO). Those need a different
robustness mechanism, graduated non-convexity over the existing robustifiers,
not hypothesis sampling (§14).

## 1. Current conventions (read from the code at `de4e9cb`)

- `FactorBatch::Evaluate(residuals, jacobians, state_pointers, stream)`
  (`cunls/factor/factor_batch.h`) evaluates **all** `NumActiveFactors()` factors in
  one host-launched call. Factor *i* reads its measurement at index *i* and its
  states through `state_pointers[i * num_blocks + b]`. **The state pointers are
  arbitrary per factor**, and that's what §4.1 exploits: one call can evaluate
  different factors against different hypotheses' states.
- `StateBatch::Plus(x, delta, x_plus_delta, stream)` (`cunls/state/state_batch.h`)
  updates exactly `NumActiveStates()` blocks from arbitrary `x`/`delta`/output
  pointers, so it can be applied to any replica of a batch's storage.
- **Many built-in objects keep `mutable` scratch buffers used inside
  `Evaluate`/`Plus`:** every Lie-group state (`SO2/SO3/SE2/SE3/Similarity2/3/SL4StateBatch`)
  and many factors (between, prior and motion families, e.g.
  `SE3BetweenFactorBatch::poses_left_inverse_`,
  `SE3PriorFactorBatch::transforms_error_`). Two calls on the **same object** must
  therefore never run concurrently. `NumericDiffJacobianBuilder`
  (`cunls/minimizer/numeric_diff_jacobian.cu`) already relies on this: it
  pins all calls on one object to one stream. RANSAC follows the same rule.
- `NumericDiffJacobianBuilder` also already evaluates one factor batch against
  many state copies through a replicated pointer table and one `Evaluate` per
  copy, noting this "cannot be collapsed into a single launch for arbitrary,
  unmodified shipped factor batches." §4 is built around that fact instead of
  changing the interface.
- `LossFunctionBatch::Evaluate(s, out, num_losses, stream)`
  (`cunls/robustifier/loss_function_batch.h`) is elementwise over `num_losses`,
  so it applies unchanged to any residual buffer.
- `LevenbergMarquardtMinimizer` derives from `GaussNewtonMinimizer` and
  `LevenbergMarquardtMinimizerOptions` wraps `MinimizerOptions base_options`
  (`cunls/minimizer/levenberg_marquardt_minimizer.h`). The RANSAC pair mirrors
  this.
- Minimizer failures are logged with `LogError` and thrown as
  `std::runtime_error` (`GaussNewtonMinimizer::Minimize`).

## 2. Algorithm

Per `Minimize(stream, problem)`:

```
validate problem, compute D, build local column map          (once per Initialize)
repeat rounds until the adaptive stopping rule or max_rounds:
  1. sample:      K minimal samples of sampled factors,
                  disjoint within a wave (§5)                 (device RNG, no sort)
  2. hypothesize: K hypotheses = inner GN/LM on each sample
                  (+ always-on factors), from the initial guess
                  or from a user hypothesis generator
  3. score:       evaluate every sampled factor for every hypothesis,
                  MSAC score + inlier count per hypothesis
  4. select:      best-so-far update; top-M of this round
  5. local optimization (if a new best appeared):
                  top-M hypotheses re-solved on their inliers,
                  re-scored, repeated lo_iterations times
  6. readback:    best score + inlier count (one sync per round)
                  → host updates the required hypothesis count
final: polish the best hypothesis on its inliers, write states back,
       publish inlier masks, fill RansacSummary
```

Steps 2 and 5 share one engine: *K independent small NLS problems, solved
together* (§6). The only difference between the GN and LM minimizers is that
engine's step policy.

## 3. Problem roles, dimensions, limits

### 3.1 Factor batch roles

Each residual batch in the `Problem` gets a role through the RANSAC options,
indexed like `Problem::GetResidualBatches()`:

- **`kSampled`** (default): data factors that may be outliers. Minimal samples
  are drawn from these, each one is classified inlier/outlier, and each has an
  inlier threshold `τ_b` in that batch's residual units (whitened residual
  norm, if the batch is wrapped in `WeightedFactorBatch`/`InformationFactorBatch`).
- **`kAlwaysOn`**: trusted factors such as priors, motion priors, and
  rig/extrinsic constraints. They're included in every hypothesis solve, every
  LO solve and the final polish, and are never classified. By default their
  cost is added to the hypothesis score (`score_always_on = true`), so a
  hypothesis that violates a prior is penalized.

Samples are drawn from the union of all sampled batches, indexed as the
concatenation of those batches' factors (`N` in total).

**Sample size** `s` is set by the user or chosen automatically:
`s = ceil(D / m_min)`, where `m_min` is the smallest residual dimension among
sampled batches. The rank contributed by always-on factors isn't inferred
(§13, Q3). PnP with one SE3 gives `s = 3`.

### 3.2 States

- **Free** states (not constant) are the hypothesis unknowns. Each
  hypothesis gets its own copy (*replica*) of every state batch that contains
  at least one free block. The whole batch is replicated, not just its free
  blocks, so each replica is a valid argument for that batch's own `Plus`.
  Constant blocks inside a replicated batch receive zero deltas, which is
  today's behavior for constant states.
- **Fully constant** state batches (e.g. landmark positions held fixed in a
  camera-resection problem) are **not** replicated. All hypotheses' pointers
  refer to one shared copy, so thousands of constant points cost nothing extra.
- Replica memory is `(K + 1) × Σ_{replicated batches} NumActiveStates × AmbientSize`.
  The extra replica is a *parking* copy of the initial guess (§4.1). Users
  should keep constant blocks in their own state batch; the minimizer logs a
  warning when a replicated batch is more than half constant blocks.

### 3.3 Dimension limit

`D = Σ_{free blocks} TangentSize`. `kMaxRansacTangentDim = 64`, a compile-time
constant: a 64×64 float system is 16 KB of shared memory per hypothesis.
Two code paths:

- `D ≤ 32`: one **warp** per hypothesis system. The Cholesky runs with one
  column per lane, and several hypotheses share a thread block.
- `32 < D ≤ 64`: one **thread block** per hypothesis system.

If `D > 64` or `D == 0`, `Initialize` logs and throws `std::invalid_argument`
with the computed `D`, the per-batch breakdown, and the fix ("mark states
constant or use GaussNewtonMinimizer with a robust loss"). This matches how the
existing minimizers report failures (§1). A status-code alternative is Q1 in §13.

Nothing else is restricted. `D` can be split across any number of state
batches of mixed types, e.g. SE3 pose (6) + `Vector<4>` intrinsics (4) +
`Vector<5>` distortion (5) = 15, or five SE3 rig poses = 30. The engine never
specializes on manifold type: states are updated only through each batch's own
`Plus`, and the local column map just concatenates the free blocks' tangent
ranges. Factor batches and factors are unrestricted in count and type; factor
count only affects scoring throughput (§8).

## 4. Working with the existing `Evaluate` and `Plus` only

RANSAC needs three evaluation patterns. Each maps onto the existing contracts
as follows.

### 4.1 Sampled factors in hypothesis solves: disjoint-sample waves

> Superseded: with the item parameters of `Evaluate`, each slot draws its own
> sample and one call evaluates all slots' samples (see the status notes at
> the top). Kept for the design history.

A hypothesis solve needs its `s` sampled factors evaluated at *its* states.
One `Evaluate` call evaluates all `N_b` factors of a batch, each against
whatever pointers the table gives it. So **if every factor belongs to at most
one hypothesis, one call serves all of them at once:**

- A **wave** is a random permutation `π` of the `N` sampled factors, cut into
  `⌊N/s⌋` disjoint samples: sample `j` is `π(j·s), …, π(j·s + s − 1)`.
- Factor `π(j·s + t)`'s pointer entries point into hypothesis `j`'s replica.
  Factors not assigned in the wave (the `N mod s` leftovers, or samples beyond
  the `K` needed) point into the parking replica, and their outputs are
  ignored.
- One `Evaluate` per sampled batch per wave evaluates every hypothesis's sample
  at once. Waves per round: `W = ⌈K / ⌊N/s⌋⌉`. Typically `W = 1`, e.g. PnP with
  `N = 2000`, `s = 3` gives 666 hypotheses per wave.
- The hypothesis's normal-equation kernel reads its rows at factor indices
  `π(j·s + t)` straight from the batch's output buffers, with no gather or
  compaction.

The work is `W · N` evaluations for the `K · s` that are needed, i.e. almost
no waste, in `W` launches per batch.

Statistically, each sample is still a uniformly random `s`-subset. Samples
within a wave are disjoint rather than independent, which is sampling without
replacement across hypotheses. For `K·s ≪ N` the difference is negligible;
otherwise it gives more even coverage of the data. The permutation is
generated without sorting (§5).

### 4.2 Per-hypothesis calls: always-on factors, `Plus`, scoring

Three things inherently need one call per hypothesis under the existing
contracts:

- **Always-on factors:** every hypothesis needs all of them, at its own states.
- **`Plus`:** one call updates one replica.
- **Scoring:** every sampled factor at every hypothesis's states, `K × N`
  evaluations, i.e. `K` calls of `N` each.

Rules for these calls:

- **Serialized per object, in a fixed order, on one stream.** Required by the
  `mutable` scratch in built-in factors and states (§1), and it makes the
  output deterministic.
- **Pointer tables are built once per round** by one kernel per residual
  batch: `ptr[h][i][b] = replica_base[batch(b)][h] + block(i, b) · ambient`
  for replicated batches, and the shared constant copy otherwise. `block(i, b)`
  is resolved once at `Initialize` from the problem's pointer lists. There are
  two table sets, for the *current* and *candidate* replicas.
- **No CUDA graphs.** An earlier revision replayed these sequences as CUDA
  graphs; they were removed as unnecessary complexity at this stage. All
  launches are direct.

### 4.3 Always-on factors, linearized, during hypothesis generation (opt-in)

When hypotheses start from the common initial guess `x0`, always-on factors can
be linearized **once per round** at `x0`: evaluate `r0`, `J0` in one call per
batch, then add the constant `J0ᵀJ0` and `J0ᵀ(r0 + J0 ξ_h)` to every hypothesis's
system. Here `ξ_h` is the hypothesis's accumulated tangent step from `x0`, a
first-order approximation of `x_h ⊟ x0`, which the `StateBatch` contract doesn't
provide. This removes the `K` always-on calls per inner iteration.

It's an approximation, but only in candidate *generation*. LO, scoring
(unless `score_always_on` uses the linear model too, which it does in this
mode) and the final polish evaluate always-on factors exactly. Not available
with a hypothesis generator (§11.3), since generated hypotheses don't share a
linearization point. Option: `linearize_always_on_in_hypotheses` (default
`false`; phase 0 decides whether it should default to on).

### 4.4 Internal fast paths for built-in types (not public API)

The generic paths above are correct for every factor and state. For built-in
types, the library can do better without touching the public contracts:
built-in classes additionally implement an **internal** interface declared in
an internal header that isn't part of `cunls.h`, for example:

```cpp
namespace cunls::internal {
class ReplicatedEvaluator {  // implemented only by built-in factor batches
 public:
  virtual bool EvaluateReplicated(float *residuals, float *jacobians,
                                  float const *const *state_pointers,
                                  size_t num_replicas, cudaStream_t stream) const = 0;
};
class ReplicatedPlus { /* same idea for built-in state batches */ };
}
```

The RANSAC minimizer detects these with `dynamic_cast` and uses one launch
where the generic path uses `K`. Users never see, implement or depend on them;
custom C++ and Warp objects simply take the generic path. Wrappers
(`WeightedFactorBatch`, `InformationFactorBatch`) implement the internal
interface only if the wrapped batch does.

This is a pure optimization, added only for the built-ins that phase 0 shows
dominate the time (likely `SE3StateBatch`, `SO3StateBatch`,
`VectorStateBatch`, `PnPFactorBatch`, `ReprojectionFactorBatch`, the prior
factors).

### 4.5 Launch counts

Per inner iteration of the hypothesis engine, with `S_b` sampled batches,
`A_b` always-on batches, `R` replicated state batches, and `W` waves:

| Work | Generic path | Built-in fast path |
|---|---|---|
| Sampled factors: residual + Jacobian, and candidate cost | `2 · W · S_b` | `2 · W · S_b` |
| Always-on factors, exact | `2 · K · A_b` | `2 · A_b` |
| Always-on factors, linearized (§4.3) | `0` per iteration, `A_b` per round | same |
| `Plus` | `K · R` | `R` |
| Own kernels (normal equations, solve, accept) | ~4 | ~4 |

Per round, scoring takes `K · S_b` calls (generic) or `⌈K/C⌉ · S_b` (fast path,
`C` hypotheses per chunk).

Rough example (unmeasured): PnP + SE3 prior, one SE3 state, `K = 256`,
5 iterations. The generic path is about 5 × (2 + 2·256 + 256 + 4) ≈ 3,900
serialized small kernels plus 256 scoring calls. At about 2–4 µs per launch,
that's roughly 4–8 ms per round, or about half that with the linearized
prior. The fast path is about 5 × (2 + 2 + 1 + 4) + 1 ≈ 50 launches, about
0.1–0.3 ms. Both are well within real-time for a single PnP. The gap is why
§4.4 exists, and phase 0 measures it.

### 4.6 Numeric Jacobians

`JacobianMode::kNumeric` (global or per batch) works through the same paths:
`NumericDiffJacobianBuilder`'s perturbation replicas compose with hypothesis
replicas, one `Evaluate` per (hypothesis, perturbation slot) in the generic
path. With `D` perturbation slots, that multiplies calls by `2D` (central
differences), so the minimizer logs a warning when numeric mode meets a batch
without a fast path. Sampled batches keep the wave trick: one call per
(wave, perturbation slot).

## 5. Sampling

- **RNG:** Philox4x32-10 via `curand_kernel.h` device functions, keyed by
  `(seed, round, wave)`. It's counter-based: no RNG state in memory, and
  results are reproducible for a given `seed` (§10).
- **Permutation without sorting:** `π` is a keyed bijection on `[0, N)`: a
  4-round Feistel network on the next power of two with cycle-walking to stay
  in range. Each thread computes `π(t)` independently, with no sort, no
  shuffle pass, and no thrust. Hypothesis `j` of a wave takes positions
  `[j·s, j·s + s)`.
- **Degenerate samples:** no explicit degeneracy test in v1. Degenerate samples
  show up as a failed or ill-conditioned solve (§7) and are discarded.
- **Guided sampling (PROSAC-style), optional:** if the user supplies a
  per-factor quality *order* (e.g. by descriptor score, computed however they
  like), wave `w` permutes only the PROSAC prefix of size `n_w` of that order,
  giving `⌊n_w/s⌋` hypotheses per wave (early waves are smaller). The
  minimizer never sorts. The order is an input.

## 6. The batched small-problem engine

The core of both minimizers: solve `K` independent NLS problems that share the
problem's structure but have their own state replicas and their own sampled
factors.

### 6.1 One inner iteration (all hypotheses at once)

1. **Evaluate** residuals + Jacobians: sampled batches through waves (§4.1),
   always-on batches per hypothesis or linearized (§4.2, §4.3). Apply the loss
   via `ResidualBatch`, as today.
2. **Normal equations:** one kernel. A warp or block per hypothesis
   accumulates `JᵀJ` (D×D, upper triangle) and `−Jᵀr` over its sampled rows
   (`π(j·s + t)` in the wave's output) and its always-on rows, into shared
   memory. Each row's block columns are mapped through the local column map
   (free block → local offset in `[0, D)`, shared by all hypotheses). The
   reduction order is fixed (tree reduce, no atomics), which makes the result
   deterministic (§10).
3. **Step policy** (the GN/LM difference, §6.3), then an in-kernel **dense
   solve** (§7). Hypotheses whose solve fails are marked invalid.
4. **Update:** `Plus` per hypothesis per replicated state batch (§4.2), from
   current to candidate replicas.
5. **Candidate cost:** residual-only evaluation against the candidate tables
   (waves for sampled batches), then per-hypothesis cost reduction (one kernel).
6. **Accept/reject** per hypothesis (one kernel): copy the accepted candidate
   states to current; update per-hypothesis convergence flags.

The iteration count is fixed (`hypothesis_iterations`, typically 3–10) and
there's no host sync. Converged or invalid hypotheses still receive their
per-hypothesis calls, since the calls are host-issued. In
the generic path, their outputs are ignored and their state is left unchanged.

### 6.2 Masked solves (LO, final polish)

`M` hypotheses (the top-M; `M = 1` for the final polish) are each solved on
**all** sampled factors plus always-on factors. Per iteration that's `M` full
`Evaluate` calls per batch and `M` `Plus` calls per replicated batch, which is
cheap since `M` is small (default 8). After evaluation, a select (not a
multiply: outliers can produce `NaN`/`Inf`) zeroes the residuals and Jacobians
of rows that are outliers for that hypothesis. This is the same dead-factor
rule as `docs/design/incremental_optimization.md` §5.2. Always-on factors are
evaluated exactly here.

### 6.3 Step policies

| | `RansacGaussNewtonMinimizer` | `RansacLevenbergMarquardtMinimizer` |
|---|---|---|
| System | `JᵀJ δ = −Jᵀr` + tiny Tikhonov floor `ε·diag` for rank-deficient minimal samples | `(JᵀJ + λ_h · diag(JᵀJ)) δ = −Jᵀr` |
| Accept | cost decreases | gain ratio ≥ `step_accept_threshold` |
| On reject | hypothesis converged | `λ_h *= lambda_upscale`, retry next iteration |
| On accept | continue | `λ_h *= lambda_downscale` if gain ratio ≥ `lambda_downscale_threshold` |
| Per-hypothesis state | cost, flags | cost, flags, `λ_h` |

The LM parameters mirror `LevenbergMarquardtMinimizerOptions` field for field
(§11), so users recognize them. `λ_h` lives in a device array of length `K` and
resets to `initial_lambda` at every new hypothesis.

## 7. Linear solvers (restricted set)

`RansacLinearSolverType`:

- **`kCholesky`** (default): in-kernel Cholesky of the per-hypothesis D×D
  system, a warp per system for `D ≤ 32`, a block per system for `D ≤ 64`.
  A non-positive pivot marks the hypothesis invalid.
- **`kLDLT`**: in-kernel pivoted LDLT, factored out of the existing
  `factorize_symmetric_pivoted_ldlt_kernel`
  (`cunls/linear_solver/dense_linear_solver.cu`) into a `__device__` routine
  shared by both. It's more tolerant of the near-singular systems minimal
  samples produce.

The existing sparse and dense solvers (PCG, cuDSS, cuSOLVER dense) are
rejected with an error if requested, because they solve one system per call
and would reintroduce K sequential solves.

## 8. Scoring and selection

- **Scoring** evaluates every sampled factor for every hypothesis, residual
  only. Generic path: one `Evaluate` per hypothesis per sampled batch, each
  writing into that hypothesis's slice of the residual buffer (§4.2). Fast
  path: one replicated call per chunk. Hypotheses are processed in chunks of
  `C`, which bounds the residual buffer to `C × N × m` floats (e.g. `C = 128`,
  `N = 10⁴`, `m = 2` gives 10 MB). `C` is derived at `Initialize` from
  `scoring_memory_budget_bytes`.
- **Per row:** `e = ‖r‖²` on the raw (pre-loss) residual. Inlier iff
  `e ≤ τ_b²`. MSAC contribution `min(e, τ_b²)`; inlier-count scoring is also
  available (`RansacScoring::kInlierCount`).
- **Per hypothesis:** a fixed-order segmented reduction of MSAC cost + inlier
  count (+ always-on cost if enabled). Invalid hypotheses score `+∞`.
- **Selection:** the round's top-M by M repeated block-wide argmin passes over
  `K` scores (ties to the lowest index), with no sort. The best-so-far
  hypothesis (score, states, inlier mask) is kept on device and replaced only
  on strict improvement.
- **Inlier masks:** one `uint8` per sampled factor, written only for the
  best-so-far hypothesis (from its scoring pass) and for LO candidates.

## 9. Adaptive stopping, memory, host interaction

- One round is a fixed batch of `K` hypotheses through §5 → §6 → §8 → LO.
- At the end of each round there's exactly **one** device-to-host readback:
  best score, best inlier count, and valid-hypothesis count. The host computes
  the inlier ratio `w`, the required total
  `k* = log(1 − confidence) / log(1 − wˢ)`, and stops when hypotheses drawn
  ≥ `k*`, when `max_rounds` is reached, or when `w ≥ early_stop_inlier_ratio`.
- **Memory is allocated once at `Initialize`,** sized from `K`, `M`, `C`,
  `s`, `N` and `D`: replicas, pointer tables, per-batch output buffers, scores
  and masks. Rounds allocate nothing, which is also what makes the call
  sequences free of allocations.
- The implementation uses no thrust and no sorting. Reductions, argmins and the
  permutation are hand-written kernels. `ReduceSumToDevice`
  (`cunls/minimizer/device_reduction.cu`) is reused where it fits.

## 10. Determinism

For a fixed `seed`, problem and options, results are bitwise reproducible on
the same GPU and driver, whether the generic path or the fast path is
used. Sampling and permutations are counter-based, per-hypothesis reductions
run in a fixed order, calls on each object are serialized in a fixed order, and
selection breaks ties by index. Built-in factor and state kernels are
elementwise. Custom objects are reproducible if their own kernels are. This is
stricter than the existing minimizers, whose assembler uses `atomicAdd`
(`block_hessian_assembler.cu`), and it's worth keeping: reproducible RANSAC
is much easier to debug.

## 11. API

### 11.1 C++

No changes to `FactorBatch`, `StateBatch`, `LossFunctionBatch` or `Problem`.

```cpp
struct RansacFactorBatchOptions {
  RansacRole role = RansacRole::kSampled;  // or kAlwaysOn
  float inlier_threshold = 1.0f;           // τ_b, residual units; ignored for kAlwaysOn
};

struct RansacMinimizerOptions {
  // Sampling / rounds
  size_t hypotheses_per_round = 256;
  size_t max_rounds = 8;
  size_t sample_size = 0;                  // 0 = auto (§3.1)
  float confidence = 0.999f;
  float early_stop_inlier_ratio = 1.0f;    // 1.0 disables
  uint64_t seed = 0;
  const int *sampling_order = nullptr;     // optional PROSAC order (device), §5

  // Per residual batch, indexed like Problem::GetResidualBatches().
  // Empty = every batch kSampled with default_inlier_threshold.
  std::vector<RansacFactorBatchOptions> factor_batches;
  float default_inlier_threshold = 1.0f;

  // Scoring
  RansacScoring scoring = RansacScoring::kMSAC;
  bool score_always_on = true;
  size_t scoring_memory_budget_bytes = 64ull << 20;

  // Inner solves
  size_t hypothesis_iterations = 5;
  size_t final_iterations = 20;
  float state_tolerance = 1e-6f;
  float cost_tolerance = 1e-6f;
  bool linearize_always_on_in_hypotheses = false;  // §4.3
  RansacLinearSolverType linear_solver = RansacLinearSolverType::kCholesky;
  JacobianMode jacobian_mode = JacobianMode::kAnalytic;
  NumericDiffOptions numeric_diff_options = {};

  // Optional: closed-form or learned hypothesis source (§11.3)
  RansacHypothesisGenerator *hypothesis_generator = nullptr;
};

struct RansacLevenbergMarquardtMinimizerOptions {
  RansacMinimizerOptions base_options;
  float initial_lambda = 1e-3f;
  float lambda_upscale = 2.0f;
  float lambda_downscale = 0.5f;
  float lambda_max = 1e6f;
  float lambda_min = 1e-6f;
  float step_accept_threshold = 0.25f;
  float lambda_downscale_threshold = 0.75f;
};

struct RansacSummary : MinimizerSummary {  // num_iterations = final polish iterations
  size_t num_rounds = 0;
  size_t num_hypotheses = 0;
  size_t num_valid_hypotheses = 0;
  size_t num_inliers = 0;
  float inlier_ratio = 0.f;
  float best_score = 0.f;
};

class RansacGaussNewtonMinimizer {
 public:
  explicit RansacGaussNewtonMinimizer(const RansacMinimizerOptions &options = {});
  virtual ~RansacGaussNewtonMinimizer() = default;
  RansacSummary Minimize(cudaStream_t stream, Problem &problem);

  /** Device mask (1 = inlier) over the factors of a kSampled batch, valid until
   *  the next Minimize. nullptr for kAlwaysOn batches. */
  const uint8_t *InlierMask(size_t residual_batch_index) const;
 protected:
  /* step policy hooks (§6.3), overridden by the LM variant */
};

class RansacLevenbergMarquardtMinimizer : public RansacGaussNewtonMinimizer { ... };
```

Usage, PnP with a pose prior. The problem setup is exactly what it would be for
`GaussNewtonMinimizer`:

```cpp
SE3StateBatch pose(d_pose, 1);                   // initial guess in d_pose
PnPFactorBatch pnp(d_obs, d_points, N);
PriorFactorBatch prior(d_pose_prior, 1);

Problem problem;
problem.AddStateBatch(&pose);
problem.AddFactorBatch(&pnp, pnp_state_ptrs);    // residual batch 0
problem.AddFactorBatch(&prior, prior_state_ptrs); // residual batch 1

RansacMinimizerOptions opt;
opt.factor_batches = {{RansacRole::kSampled, 2.0f / focal},
                      {RansacRole::kAlwaysOn}};
RansacGaussNewtonMinimizer ransac(opt);
RansacSummary summary = ransac.Minimize(stream, problem);  // refined pose written to d_pose
const uint8_t *inliers = ransac.InlierMask(0);             // N device flags
```

### 11.2 Python

Custom `WarpFactorBatch`/`WarpStateBatch` subclasses work as they are.

```python
opt = pycunls.RansacMinimizerOptions()
opt.factor_batches = [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.SAMPLED, 2.0 / f),
                      pycunls.RansacFactorBatchOptions(pycunls.RansacRole.ALWAYS_ON)]
ransac = pycunls.RansacLevenbergMarquardtMinimizer(
    pycunls.RansacLevenbergMarquardtMinimizerOptions(base_options=opt))
summary = ransac.minimize(stream, problem)
inliers = ransac.inlier_mask(0)          # cupy uint8 array view
```

Python-implemented objects go through a nanobind trampoline on each call, so
the generic path's per-hypothesis calls cost host time per call.

### 11.3 Hypothesis generator hook

Hypotheses from an initial guess plus a few GN/LM iterations need a guess
inside the basin of convergence. That covers tracking (IMU or motion-model
prediction), but not relocalization from scratch. For that, an optional,
separate interface. It's a new, *opt-in* extension point, not a change to
existing classes:

```cpp
class RansacHypothesisGenerator {
 public:
  virtual ~RansacHypothesisGenerator() = default;
  /**
   * Writes up to `max_solutions` hypotheses per sample into the replicated
   * state buffers (layout of §3.2) and a per-hypothesis validity flag.
   * sample_factor_ids: K × s indices into the concatenated sampled factors.
   */
  virtual bool Generate(cudaStream_t stream, const Problem &problem,
                        const int *sample_factor_ids, size_t num_samples,
                        size_t max_solutions, float *const *replica_states,
                        uint8_t *valid) = 0;
  virtual size_t MaxSolutionsPerSample() const = 0;
};
```

Generated hypotheses skip or shorten the inner solve
(`hypothesis_iterations` still applies, usually set to 0–2). A built-in
`P3PHypothesisGenerator` (Lambda Twist) for `PnPFactorBatch` is phase 5.

## 12. Implementation plan

**Phase 0: prototype and measure the generic path.** PnP + SE3 prior, one
`SE3StateBatch`, using only `Evaluate`/`Plus`: waves, per-hypothesis calls,
warp-per-hypothesis Cholesky, scoring, selection. Benchmarks on
GPU 1: `N ∈ {500, 2000, 10000}`, outlier ratio 10–70%, `K ∈ {128, 512, 2048}`:
- time per round, with and without linearized always-on factors;
- the same through a prototype internal fast path for `PnPFactorBatch` and
  `SE3StateBatch`, to size the §4.4 gap;
- a `WarpFactorBatch` version of PnP, to measure Python trampoline cost and

- comparison against OpenCV `solvePnPRansac` (CPU) for time and accuracy.

**Phase 1: `RansacGaussNewtonMinimizer`, generic path.** Roles, `D`
validation, sampling, waves, the batched engine, both dense solvers, scoring,
selection, adaptive rounds, final polish, inlier masks, summary. Tests:
- synthetic PnP at 0–70% outliers: recovered inliers ⊇ 99% of the true
  inliers, and the final pose matches `GaussNewtonMinimizer` run on the true
  inlier set to float tolerance;
- mixed-type multi-batch problems: SE3 pose + `Vector<4>` intrinsics
  (`D = 10`), two PnP batches from a rig plus an always-on prior, SO3 + vector
  (`D = 6`), and a partially constant state batch;
- `D > 64` fails with the documented message;
- a custom C++ factor and a Warp factor work unmodified;
- same seed gives bitwise-identical results across runs.

**Phase 2: `RansacLevenbergMarquardtMinimizer`.** Per-hypothesis `λ_h`, gain
ratio. Tests: agrees with the GN variant on well-conditioned cases; converges
from worse initial guesses where GN diverges.

**Phase 3: LO-RANSAC, PROSAC ordering, numeric Jacobians, linearized
always-on factors.**

**Phase 4: internal fast paths (§4.4)** for the built-ins phase 0 identifies
as dominant. Tests: fast and generic paths give bitwise-identical results.

**Phase 5: `P3PHypothesisGenerator`.**

## 13. Open questions

1. **Error reporting.** Throw (consistent with the existing minimizers) or
   return a status in `RansacSummary`? RANSAC failing to find a model
   (every hypothesis invalid, or zero inliers) is a legitimate outcome, not
   an exception. Proposal: throw for configuration errors (`D` limit, an
   unsupported solver, an invalid role vector), and report "no model found" via
   a `RansacSummary::status` without touching the states.
2. **Correspondence groups.** A physical correspondence can span several
   factors (a stereo observation as two PnP factors in two batches). Should
   sampling and classification work on groups via an optional per-factor group
   id? This is needed for multi-camera rigs. It fits the wave scheme: permute
   groups, not factors.
3. **Always-on factors in the minimal-sample dimension count.** Auto `s` ignores
   the rank that always-on factors provide (a full SE3 prior makes a smaller
   `s` meaningful). Should the minimizer try to infer it, or leave it to the user?
4. **Batching independent RANSAC problems.** One `Problem` is one RANSAC. Many
   cameras each needing their own PnP RANSAC means many `Minimize` calls. A
   later `Minimize(stream, std::span<Problem*>)` could batch problems of
   identical structure.
5. **Inlier threshold units.** `τ_b` in whitened residual units is precise but
   unintuitive for PnP (normalized image coordinates). Should built-in factors
   expose a helper for a pixel threshold?

## 14. Related: robust estimation for large problems

For problems with `D > 64`, the counterpart is **graduated non-convexity**:
run the existing minimizer while annealing a robust loss's scale (Geman-McClure
or truncated least squares) across outer iterations. It uses the existing
`LossFunctionBatch` machinery plus a scale schedule, and it yields a similar
inlier mask (the final loss weights). That's a separate design, noted here so
the two features share the inlier-mask output convention.

## 15. Measured results

Run with `CUNLS_RANSAC_BENCHMARK=1 ./bin/nls_tests --gtest_filter='RansacBenchmark.*'`
(CSV in `/tmp/cunls_ransac_bench/`). Hardware: RTX A6000 (the device
`CUDA_VISIBLE_DEVICES=1` selects on the development machine), Release build.
Another job shared the GPU during part of the timing runs, so single timings
are noisy by up to ~2x. Synthetic PnP: normalized coordinates, inlier noise
sigma = 1e-3 (~0.5 px at f = 500), tau = 5 sigma, outliers at least 4 tau off.
The regular minimizers use DenseLDLT unless noted.

**Speed with EvaluateIndexed / PlusReplicated** (idle GPU, PnP, 30% outliers,
median ms, K = 256 hypotheses per round):

| points | LM+Cauchy | RANSAC-GN before | RANSAC-GN after |
|---|---|---|---|
| 100 | 0.63 | 8.3 | 3.1 |
| 1k | 0.66 | 7.4 | 2.1 |
| 10k | 0.90 | 8.9 | 4.4 |
| 100k | 3.7 | 31.8 | 25.4 |
| 1M | 31.5 | 254 | 247 |

With two-stage scoring and early exit (large problems): 1M points RANSAC-GN
247 -> 67 ms (scoring 84 -> 7 ms, local optimization 120 -> 47 ms,
refinement 35 -> 2.6 ms), RANSAC-LM 117 ms; 100k points 25 -> 12.6 ms.
LM+Cauchy: 31–40 ms at 1M.

After folding the batched methods into `Evaluate` / `Plus` (item and replica
parameters, per-slot samples, no fallbacks; same device, idle,
`PnPSizeSweep`): RANSAC-GN 3.0 / 2.1 / 4.4 / 12.5 / 64.9 ms and RANSAC-LM
3.2 / 2.3 / 4.5 / 16.4 / 104.6 ms at 100 / 1k / 10k / 100k / 1M points.
Without local optimization and with 4 finalists: RANSAC-GN 1.8 / 1.4 / 2.4 /
7.3 / 14.9 ms and RANSAC-LM 1.9 / 1.4 / 2.5 / 7.5 / 47.3 ms (LM+Cauchy: 0.6 /
0.7 / 0.9 / 3.7 / 31.6 ms). Rig, 1000 points per camera: 1.6 / 3.1 / 11.1 /
18.4 ms for 1 / 2 / 5 / 10 cameras.

One round of RANSAC-GN at 1k points: 4,096 hypotheses in 3.3 ms (was 68 ms).
Rig with an always-on between factor, 2 cameras: 4.8 ms (was 116 ms);
10 cameras (D = 60): 34 ms (was 195 ms). Kernel launches for one 600-point
PnP test: 2,823 -> 720 (`Plus`: 1,424 -> 44, PnP `Evaluate`: 601 -> 98). At 1M
points the time is the scoring work itself (256 hypotheses x 1M factors with
Jacobians for the informative-inlier test), not launches.

**Speed before the batched methods** (PnP, 30% outliers, median ms, K = 256):

| points | GN | LM+Cauchy | RANSAC-GN | RANSAC-LM |
|---|---|---|---|---|
| 100 | 0.6 (wrong pose) | 0.6 | 8.4 | 8.5 |
| 1k | 0.8 (wrong pose) | 0.8 | 7.7 | 7.6 |
| 10k | 1.2 (wrong pose) | 0.9 | 9.9 | 9.4 |
| 100k | 6.7 (wrong pose) | 4.7 | 41.5 | 39.4 |
| 1M | 49 (wrong pose) | 39 | 283 | 286 |

RANSAC costs about 20 ms per 1,000 hypotheses at large K (hypothesis-count
sweep), and 4–6 ms of fixed cost per call. It is launch-bound: each hypothesis
issues its own `Plus` and scoring `Evaluate` calls (§4.2). Always-on factors
multiply that: a 2-camera rig with one always-on between factor takes
115–160 ms vs 20–24 ms for one camera. §4.3 (linearized always-on) and §4.4
(internal fast paths) target exactly this.

**Robustness** (PnP 1000 points, 20 trials, success = rotation < 0.5° and
translation < 0.05):

- Uniformly random outliers: plain GN/LM fail from 10% outliers, LM+Huber
  from 60%. LM+Cauchy succeeds up to 90% from a small initial error, but only
  65% of the time at 90% outliers from a large one (0.3 rad, 0.8). RANSAC-GN
  and RANSAC-LM succeed in 100% of trials in every cell.
- Coherent outliers (a competing, self-consistent wrong pose): from a large
  initial error, LM+Cauchy drops to 70% (40% outliers) and 60% (45%);
  LM+Huber keeps the pose within tolerance but is biased by the outliers
  (inlier recall 0.91 at 40%, 0.67 at 45%). RANSAC: 100% success and perfect
  classification in every cell.

**Detection** (inlier mask vs. ground truth):

- Precision (fraction of kept factors that are true inliers) and outlier
  recall are 1.000 in every RANSAC cell above, including outliers only 1.2 tau
  off.
- Inlier recall is set by the threshold, as expected for Gaussian noise:
  0.86 at tau = 2 sigma, 0.99 at 3 sigma, 1.00 at 5 sigma.
- LM+Cauchy classified by |r| <= tau at its final pose matches RANSAC when it
  converges, and drops to 0.66 inlier recall at 90% uniform outliers from a
  large initial error.

**Kernels** (Nsight Compute, isolated replay; before -> after the rewrite
that removed atomics and serial loops):

| kernel, workload | before | after |
|---|---|---|
| normal equations, 1 slot, D = 64, 3000 rows | 3.76 ms | 58 us |
| normal equations, 8 slots, D = 64, 3000 rows | 3.76 ms | 122 us |
| normal equations, 1 slot, D = 6, 600 PnP rows | 52 us | 19 us |
| normal equations, 256 hypotheses, D = 6 (warp) | 23 us | 11 us |
| select top-8 of 256 | 11 us | 5 us |

The dense solve is latency-bound at large D (about 150 us per D = 64 slot:
64 dependent elimination steps of shared-memory round trips). A
register-resident warp factorization is the next step if it matters; end to
end, the host-issued per-hypothesis calls cost more.
