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
| a robot (URDF or joint arrays): inverse kinematics with hard joint limits | `ArticulationTopology` (`pycunls.articulation.topology_from_urdf`, or `topology_from_joint_arrays` for one articulation of a simulator scene's joint arrays), `ArticulationStateBatch`, `LinkPoseFactorBatch` + `AugmentedLagrangianMinimizer` | `python/tests/test_articulation.py`, `docs/sphinx/articulations.rst` |
| a robot objective of your own (look-at, stay above a table, clearance, a frame relative to an object pose) | an existing factor on a link frame view through `LinkFrameFactorBatch` (`FrameView.Pose`: SE3 factors; `Position`: vector factors such as `HalfspaceFactorBatch3`; `Orientation`: SO3 factors), or your own: `pycunls.warp.WarpFunctionFactorBatch` (a residual function of poses, rotations and vectors; autodiff Jacobian) or a `WarpFactorBatch` (analytic) | `python/examples/robot_custom_objective.py`, `docs/sphinx/articulations.rst` |
| a robot with closed loops or mimic joints | the same, plus `pycunls.articulation.loop_closure(...)` constraints / `MimicJointFactorBatch` wrapped in `ConstraintFactorBatch(..., ConstraintKind.Equality)` | `docs/sphinx/articulations.rst` |
| a residual no built-in factor computes, inside a larger graph | a Warp factor (`pycunls.warp.WarpFactorBatch`) | `python/examples/custom_warp_factor.py` |

`pycunls.catalog("factor")` lists the built-in factors; `pycunls.describe(name)` gives one's
states, residual, measurement layout and conventions.

## 2. Model fitting with `fit`

```python
import numpy as np, warp as wp, pycunls

@wp.func
def model(x: float, p: wp.vec3) -> float:   # one sample: y = a exp(-b x) + c
    return p[0] * wp.exp(-p[1] * x) + p[2]

res = pycunls.fit(model, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
print(res.report())
res.params, res.param_errors, res.converged, res.reduced_chi2
```

- The model is a **Warp function of one sample** (`@wp.func`), with annotated types: `x` a
  `float` or `wp.vecN` (one sample's inputs), `p` a `wp.vecP` (a `float` for one parameter;
  `wp.types.vector(length=5, dtype=float)` for sizes without a named type), the result a `float`
  or `wp.vecM` (`m` outputs per sample). Use `wp.exp`, `wp.sin`, ... inside. A Python or NumPy
  function is a `TypeError`.
- The Jacobian comes from Warp's automatic differentiation, checked against central differences
  at `p0` (`jacobian="auto"`; on a mismatch it warns and uses central differences).
  `jacobian="autodiff"` or `"numeric"` forces one.
- Batches: `y` of shape `(B, n)` with `x` shared `(n,)` or per fit `(B, n)`; `p0` `(P,)` or
  `(B, P)`. Pad shorter fits with NaN in `y`. Results gain a leading `B` axis; `res.converged` is
  per fit.
- Pass `sigma` (the measurement noise) when you know it: then `reduced_chi2` is about 1 for a
  model that explains the data, and clearly larger (say > 2) when it does not. Without `sigma`,
  `param_errors` are scaled by the residual scatter (as `scipy.optimize.curve_fit`).
- `starts=K` needs finite `bounds` (starts are drawn in the box); `starts=[[...], ...]` gives
  them explicitly. The lowest objective wins per fit (`res.start_index`); starts where the
  model returns NaN never win.
- Any number of parameters per fit: past 4 declare `p` as
  `wp.types.vector(length=P, dtype=float)`.
- A model given by an ODE: use its closed-form solution when there is one; otherwise integrate
  inside the model with a fixed-step scheme (e.g. RK4 from 0 to the sample's `x`, `h = x / 16.0`,
  in a `for i in range(16)` loop: Warp unrolls and differentiates loops of up to 16 steps;
  longer loops make `jacobian="auto"` fall back to central differences). Frequencies, phases and other periodic parameters make the cost
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

- **GPU float32 arrays.** The low-level API takes float32, C-contiguous GPU arrays: `wp.array`
  (any `__cuda_array_interface__` array such as PyTorch or CuPy works too) or int device
  pointers. `fit` and `ProblemBuilder` convert NumPy for you. Read results back with
  `a.numpy()`; write into a bound buffer with `a.assign(np_array)`.
- **Streams.** Warp work (uploads, `assign`, kernels) runs on Warp's current stream. Every
  `stream=` argument of pycunls accepts a `wp.Stream` (or any `__cuda_stream__` object, or a
  `CudaStream`), so the simplest order is to solve on Warp's stream itself:
  `stream = wp.get_stream()`; uploads, solves and reads then run in order without extra
  synchronization. With a separate `pycunls.CudaStream()`, call
  `wp.synchronize_stream(wp.get_stream())` before `minimize`. `minimize` synchronizes its stream
  before returning, except on the real-time paths (`MinimizerOptions.real_time`, the augmented
  Lagrangian's `real_time`): read their results on the same stream, or synchronize it first.
  To capture a real-time solve in a CUDA graph (`wp.ScopedCapture`), set `reuse_structure` and
  run one `minimize` call before the capture (the first call sets up and synchronizes); a captured
  call that would synchronize raises `ValueError` with this fix.
- **Devices.** A problem runs on the device of its state arrays: `minimize`, `evaluate_batch` and
  the state helpers make that device current themselves (`problem.device`), so no
  `wp.ScopedDevice` is needed around them. `ArticulationTopology` (and
  `topology_from_joint_arrays`, `topology_from_urdf`) takes `device=`, defaulting to the device of
  the GPU arrays it is given; its `ArticulationStateBatch`es live on its device. Objects made
  without device arrays (`WarpFactorBatch`, `CudaStream()`) use Warp's current device: create
  them inside `wp.ScopedDevice(device)` on a multi-GPU machine.
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
- **Articulations.** `joint_q` holds each joint's coordinates in joint order (quaternions x, y, z, w); write it with
  `ArticulationStateBatch.set_joint_q`, read it with `get_joint_q` (both take a `stride` for
  instances inside a larger array); `get_link_transforms` gives every link's world pose
  (position, quaternion x, y, z, w). Joint limits are hard bounds
  (solve with `AugmentedLagrangianMinimizer`). Loop-closing joints are not tree joints: add them
  as the equality constraints `pycunls.articulation.loop_closure` returns.
- **Warp factor kernels.** For poses, rotations and vectors prefer
  `pycunls.warp.WarpFunctionFactorBatch` (a residual `@wp.func`, no kernel). A kernel factor is
  `WarpFactorBatch(kernel, inputs=[...], residual_size=M, state_sizes=[...], capacity=n)`: the
  kernel's own parameters (`inputs`) first, then exactly `factor_ids: wp.array(dtype=wp.int32),
  states: wp.array2d(dtype=wp.uint64), residuals: wp.array2d(dtype=float), jacobians:
  wp.array3d(dtype=float)`. Thread `t` reads its data at `factor_ids[t]` and its states with
  `pycunls.warp.scalar_at` / `point_at` / `pose_at` / `rotation_at(states[t, k])`, writes
  `residuals[t, i]`, and writes `jacobians[t, i, c]` only under `if jacobians.shape[0] > 0`.
  A custom manifold is `WarpStateBatch(plus_kernel, data=..., ambient_size=A, tangent_size=T,
  capacity=n)`, the kernel ending with `x: wp.array2d(dtype=float)` (n, A), `delta` (n, T) and
  `x_plus_delta` (n, A): one thread per state, replicas included.

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
| `TypeError: ... is a host (NumPy) array` | NumPy array to the low-level API | `wp.array(a, dtype=wp.float32)` |
| `ValueError: data has N float32 values, but M are needed` | wrong array shape or capacity | check `pycunls.describe(cls)` for floats per item |
| `NonFiniteCost` | NaN/Inf in states or measurements, or a factor outside its domain | `pycunls.diagnose(problem)` names the batch and index |
| `NonFiniteStep` | singular system (gauge freedom, unconstrained state) or NaN at trial states | `diagnose()`; fix a state constant; use LM |
| `MaxRejectedSteps` with a high final cost | wrong Jacobian or bad initial guess | `check_jacobians`; better `p0`; `starts` |
| `MaxRejectedSteps` with a cost that stopped changing | float32 round-off at the minimum | usually fine: check residuals |
| `MaxIterations` | not finished | raise `max_num_iterations`; check the cost history |
| `No residual batch has active factors` | forgot `set_num_active_factors` | set the active counts |
| `... box-bounded states ... AugmentedLagrangianMinimizer` | bounds with GN/LM | solve with `AugmentedLagrangianMinimizer()` (settings in `AugmentedLagrangianMinimizerOptions`; `fit`/`ProblemBuilder` do; `method="gn"` with bounds raises) |
| `ImportError: the driver supports CUDA 12.x but pycunls is built with CUDA 13` | pre-R580 driver | install an R580+ driver; on CUDA 12 systems use the C++ library |
| warning `Gauge freedom` | relative factors only | `constant=[0]` or a prior |
| fit: `TypeError: model must be a Warp function` | a Python/NumPy model | rewrite it as `@wp.func` of one sample |
| fit: high `reduced_chi2` with the right `sigma` | the model does not explain the data | change the model |

More detail: `pycunls.log_to_python(verbosity=pycunls.Verbosity.Message)` (or the environment
variable `CUNLS_LOG_LEVEL=message`) logs every iteration and the final message through Python's
`logging`.

## 7. Low-level API in one screen

```python
import numpy as np, warp as wp, pycunls
poses = wp.array(T0, dtype=wp.float32)            # (N, 4, 4)
const = wp.zeros(1, dtype=wp.int32)               # state 0 fixed
states = pycunls.SE3StateBatch(poses, N, const, 1)
states.set_num_active_states(N, 1)
deltas_gpu = wp.array(deltas, dtype=wp.float32)   # keep every array alive
between = pycunls.SE3BetweenFactorBatch(deltas_gpu, M)
between.set_num_active_factors(M)
problem = pycunls.Problem()
problem.add_state_batch(states)
edge_ids = wp.array(edges.reshape(-1), dtype=wp.int32)
problem.add_factor_batch(between, [states, states], edge_ids)
for issue in pycunls.diagnose(problem):           # optional: catches most setup mistakes
    print(issue)
wp.synchronize_stream(wp.get_stream())            # the uploads finish before the solve
summary = pycunls.LevenbergMarquardtMinimizer().minimize(pycunls.CudaStream(), problem)
print(summary.report())
T = poses.numpy()                                 # poses is updated in place
```
