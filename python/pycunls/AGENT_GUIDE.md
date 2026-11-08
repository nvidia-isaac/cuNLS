# pycunls: a guide for agents (and busy people)

pycunls solves nonlinear least-squares problems on the GPU: minimize
`0.5 * sum_i rho(|r_i(x)|^2)` over states `x` (vectors, poses, rotations). This guide is in
the installed package: `python -m pycunls guide` prints it, `python -m pycunls doctor` checks
the environment. Every class and function also documents itself: `help(pycunls.X)`,
`pycunls.describe("SE3BetweenFactorBatch")`.

## 1. Pick the entry point

| You have | Use | Example |
|---|---|---|
| data `x, y` and a formula `y = f(x, p)` (curve fit, calibration, per-pixel fits) | `pycunls.fit(model, x, y, p0)` | `python/examples/model_fitting.py` |
| thousands of independent fits of the same model | `pycunls.fit` with `y` of shape `(B, n)`: one GPU solve | same |
| outliers in a fit | `fit(..., loss="cauchy", loss_scale=2, sigma=...)` | same |
| parameters with physical limits | `fit(..., bounds=(lower, upper))` | same |
| a fit that lands in a wrong local minimum (frequencies, phases) | `fit(..., bounds=..., starts=64)` | same |
| poses / points / velocities linked by built-in factors (pose graph, bundle adjustment, ICP, PnP) | `pycunls.ProblemBuilder` or the one-call `pycunls.solve` | below |
| gross outliers in geometric matches (PnP, registration) | `RansacLevenbergMarquardtMinimizer` | `python/examples/ransac_pnp.py` |
| hard constraints, MPC | `AugmentedLagrangianMinimizer`, `pycunls.mpc` | `python/examples/supermarket_drones.py` |
| a residual no built-in factor computes, inside a larger graph | a Warp factor (`pycunls.warp.WarpFactorBatch`) | `python/examples/custom_warp_factor.py` |

`pycunls.catalog("factor")` lists the built-in factors; `pycunls.describe(name)` gives one's
states, residual, measurement layout and conventions.

## 2. Model fitting with `fit`

```python
import numpy as np, pycunls

def model(x, p):            # p: (rows, P), one row of parameters per sample; no loops
    return p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2]

res = pycunls.fit(model, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
print(res.report())
res.params, res.param_errors, res.converged, res.reduced_chi2
```

- `model(x, p)` must be row-wise: row `i` of the output depends only on row `i` of `x` and `p`
  (NumPy functions such as `np.exp` work on the CuPy arrays it receives). Return `(rows,)` or
  `(rows, m)` for `m` outputs per sample. A Warp `@wp.func model(x: float, p: wp.vec3) -> float`
  also works and gets autodiff Jacobians.
- Batches: `y` of shape `(B, n)` with `x` shared `(n,)` or per fit `(B, n)`; `p0` `(P,)` or
  `(B, P)`. Pad shorter fits with NaN in `y`. Results gain a leading `B` axis; `res.converged` is
  per fit.
- Pass `sigma` (the measurement noise) when you know it: then `reduced_chi2` is about 1 for a
  model that explains the data, and clearly larger (say > 2) when it does not. Without `sigma`,
  `param_errors` are scaled by the residual scatter (as `scipy.optimize.curve_fit`).
- `starts=K` needs finite `bounds` (starts are drawn in the box); `starts=[[...], ...]` gives
  them explicitly. The lowest objective wins per fit (`res.start_index`); starts where the
  model returns NaN never win.
- Up to 12 parameters per fit; beyond that build the problem with `ProblemBuilder`.
- A model given by an ODE: use its closed-form solution when there is one; otherwise integrate
  inside the model with a fixed-step scheme (e.g. RK4 from 0 to each row's `x`, all rows at once
  with array operations). Frequencies, phases and other periodic parameters make the cost
  multi-modal: give `bounds` and `starts`.

## 3. Graph problems with `ProblemBuilder`

```python
pb = pycunls.ProblemBuilder()
poses = pb.add_states(pycunls.SE3StateBatch, initial=T0, constant=[0])   # T0: (N, 4, 4)
pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges))  # edges: (M, 2)
result = pb.solve()                    # Levenberg-Marquardt; diagnose() runs first
result.converged, result.message
T = result[poses]                      # (N, 4, 4) NumPy
print(result.evaluate())               # residual statistics per factor batch
```

Several state types: `connect=[(cams, cam_ids), (points, point_ids)]` (one index array per state
slot of the factor). One state batch and one factor batch:
`pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=T0, connect=edges, constant=[0])`.
Options go by name: `pb.solve(max_num_iterations=100, cost_tolerance=1e-10)`.

## 4. Conventions that cost a turn when missed

- **GPU float32 arrays.** The low-level API takes CuPy float32, C-contiguous arrays (or int
  device pointers). `fit` and `ProblemBuilder` convert NumPy for you.
- **Poses are `world_from_body`, row-major 4x4 (SE3) or 3x3 (SE2).** Updates are on the right:
  `T' = T Exp(delta)`; se(3) tangent order is `[omega, v]` (rotation first), se(2) `[v_x, v_y,
  theta]`.
- **Between factors measure `Delta = T_right^-1 T_left`**, the inverse of the usual relative
  pose `T_left^-1 T_right`. Given `T_i^-1 T_j` per edge `(i, j)`, pass its inverse.
- **Gauge freedom.** Relative factors (between, reprojection, motion priors) leave the whole
  graph free to move: make one state constant (`constant=[0]`) or add a prior. `diagnose()`
  warns about it.
- **Capacity vs. active count (low-level API).** Batches are constructed with a capacity and
  start with **0 active** items: call `set_num_active_states(n[, num_const])` and
  `set_num_active_factors(n)` before solving.
- **Absolute tolerances.** `cost_tolerance` (1e-6) and `state_tolerance` (1e-6, on the squared
  step) are absolute: with exact data or tiny residual units lower them, or the solve stops
  "converged" early (the message says which criterion fired).
- **Reprojection observations are normalized** image coordinates (intrinsics removed).
- **Keep arrays alive.** Batches read the arrays you pass; do not let them be garbage collected.

## 5. After a solve: check, do not assume

1. `summary.converged()` (or `res.converged`); if False, read `summary.message`: it names the
   cause and the fix. `summary.termination_reason` is a `pycunls.TerminationReason`;
   `summary.to_dict()` is JSON-ready.
2. Cost: `summary.initial_cost -> final_cost`. A cost that barely moved from a bad start means a
   wrong layout, a wrong convention or a wrong Jacobian.
3. Residuals: `pycunls.evaluate(problem)` (or `result.evaluate()`): RMS per factor batch should
   be near the measurement noise; `worst_factor` points at outliers.
4. Custom factors: `pycunls.check_jacobians(problem)` must report `ok` for every batch.
5. Model adequacy (fits): `res.reduced_chi2` with the true `sigma`.

## 6. When it fails

| Symptom (message / reason) | Likely cause | Fix |
|---|---|---|
| `TypeError: ... is a host (NumPy) array` | NumPy array to the low-level API | `cupy.asarray(a, dtype=cupy.float32)` |
| `ValueError: data has N float32 values, but M are needed` | wrong array shape or capacity | check `pycunls.describe(cls)` for floats per item |
| `NonFiniteCost` | NaN/Inf in states or measurements, or a factor outside its domain | `pycunls.diagnose(problem)` names the batch and index |
| `NonFiniteStep` | singular system (gauge freedom, unconstrained state) or NaN at trial states | `diagnose()`; fix a state constant; use LM |
| `MaxRejectedSteps` with a high final cost | wrong Jacobian or bad initial guess | `check_jacobians`; better `p0`; `starts` |
| `MaxRejectedSteps` with a cost that stopped changing | float32 round-off at the minimum | usually fine: check residuals |
| `MaxIterations` | not finished | raise `max_num_iterations`; check the cost history |
| `No residual batch has active factors` | forgot `set_num_active_factors` | set the active counts |
| `... box-bounded states ... AugmentedLagrangianMinimizer` | bounds with GN/LM | wrap the minimizer in `AugmentedLagrangianMinimizer` (`fit`/`ProblemBuilder` do) |
| warning `Gauge freedom` | relative factors only | `constant=[0]` or a prior |
| fit: `model(x, p) mixes rows` | sums/means over samples in the model | make the model row-wise |
| fit: high `reduced_chi2` with the right `sigma` | the model does not explain the data | change the model |

More detail: `pycunls.log_to_python(verbosity=pycunls.Verbosity.Message)` (or the environment
variable `CUNLS_LOG_LEVEL=message`) logs every iteration and the final message through Python's
`logging`.

## 7. Low-level API in one screen

```python
import cupy as cp, pycunls
poses = cp.asarray(T0)                            # (N, 4, 4) float32
const = cp.zeros(1, cp.int32)                     # state 0 fixed
states = pycunls.SE3StateBatch(poses, N, const, 1)
states.set_num_active_states(N, 1)
between = pycunls.SE3BetweenFactorBatch(cp.asarray(deltas), M)
between.set_num_active_factors(M)
problem = pycunls.Problem()
problem.add_state_batch(states)
problem.add_factor_batch(between, [states, states], cp.asarray(edges.reshape(-1), cp.int32))
for issue in pycunls.diagnose(problem):           # optional: catches most setup mistakes
    print(issue)
summary = pycunls.LevenbergMarquardtMinimizer().minimize(pycunls.CudaStream(), problem)
print(summary.report())                           # poses is updated in place
```
