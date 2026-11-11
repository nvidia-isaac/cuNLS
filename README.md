<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/sphinx/_static/cuNLS_logo_dark.svg">
    <img src="docs/sphinx/_static/cuNLS_logo_light.svg" alt="cuNLS" width="360">
  </picture>
</p>

<h3 align="center">GPU-Accelerated Nonlinear Least Squares</h3>

<p align="center">
  <code>Python</code>&ensp;·&ensp;<code>CUDA/C++</code>&ensp;·&ensp;<code>Factor graphs</code>&ensp;·&ensp;<code>Manifolds</code>&ensp;·&ensp;<code>RANSAC</code>&ensp;·&ensp;<code>Constraints</code>&ensp;·&ensp;<code>Agent-ready</code>
</p>

<p align="center">
  <img src="assets/cunls_overview.gif" alt="cuNLS overview: problem, benchmarks, RANSAC, MPC, Python" width="100%">
</p>

---

**cuNLS** is a general-purpose nonlinear least-squares solver that runs entirely on the GPU. It
solves any factor graph over Euclidean and manifold states, with robust losses, constraints and
bounds:

$$x^* = \arg\min_x \sum_i \rho_i\!\left(\left\|f_i(x)\right\|^2_{\Sigma_i}\right) \quad \text{s.t.} \quad c_j(x) = 0,\ \ c_k(x) \le 0,\ \ l \le x \le u$$

where $x$ are the states (vectors, rotations, poses on their manifolds), $f_i$ the residuals,
$\rho_i$ optional robust losses, $\Sigma_i$ the measurement covariances
($\|v\|^2_{\Sigma} = v^T \Sigma^{-1} v$), and $c$ optional constraints. Factors are evaluated
in parallel, assembled straight into a block-sparse system and factored on the GPU; every step
of every iteration stays on the device.

The same solver covers curve fitting and calibration, thousands of small independent problems
in one batched call, and large robotics problems: bundle adjustment, pose graphs,
visual-inertial odometry with RANSAC, and model-predictive control. On the
[BAL](https://grail.cs.washington.edu/projects/bal/) Venice problem (1,776 cameras, 993k
points, 5M observations) it converges in 0.53 s, 53× faster than Ceres at the same cost;
on the torus3D pose graph it is 24–34× faster than Ceres, GTSAM and g2o.<sup>1</sup>

### Two APIs

- **Python (`pycunls`)**: the main way to use cuNLS. It works directly on
  [NVIDIA Warp](https://github.com/NVIDIA/warp) arrays (or any GPU array with
  `__cuda_array_interface__`, such as PyTorch or CuPy), and you write custom factors, states and
  `fit` models as Warp kernels and functions in Python. `pycunls.fit` and
  `pycunls.ProblemBuilder` build problems from NumPy arrays; the full API underneath gives
  fixed-buffer, real-time re-solves.
- **C++ (`cunls`)**: the native CUDA/C++ library (shared or static, CMake package) for
  applications and embedded targets, with custom factors and states as CUDA kernels. It runs on
  x86_64 and on Jetson (Orin, Thor).

### Built for agentic pipelines

cuNLS is designed to be driven by coding agents (and to save humans the same debugging):

- **Results explain themselves.** Every summary has `converged()`, a `termination_reason` (per
  subproblem too) and a `message` with the fix; `summary.to_dict()` serializes to strict JSON.
- **Mistakes are caught before solving.** `ProblemBuilder` checks shapes, index ranges and state
  types as factors are added, and `solve` runs `pycunls.diagnose` first (gauge freedom, NaN or
  Inf states and residuals, states no factor constrains). Errors name what was expected, what
  was given, and the fix.
- **The package documents itself.** `python -m pycunls guide` prints the
  [agent guide](python/pycunls/AGENT_GUIDE.md) shipped in the wheel,
  `python -m pycunls doctor` checks the installation, `pycunls.catalog()` and
  `pycunls.describe(name)` list the factors and states with their residuals, array layouts
  and conventions.
- **Measured.** `python/agent_tasks` holds tasks with hidden ground truth and
  `scripts/run_agent_eval.sh` scores an agent on them.

## Features

| Category | Details |
|---|---|
| **APIs** | Python (`pycunls`, nanobind bindings; constructors accept a `wp.array`, any other `__cuda_array_interface__` array, or a raw device pointer) and C++ (`cunls`, shared or static library) |
| **Quick setup** | `pycunls.fit(model, x, y, p0)` for model fitting (a Warp function of one sample with autodiff Jacobians, thousands of independent fits in one solve, measurement errors, robust losses, bounds, multiple starts, standard errors and reduced chi-square per fit); `pycunls.ProblemBuilder` for graph problems from arrays; see [Quick setup](docs/sphinx/simple_api.rst) |
| **Manifold support** | SO(2), SO(3), SE(2), SE(3), Sim(2), Sim(3), SL(4), Euclidean vectors |
| **Solvers** | Gauss-Newton, Levenberg-Marquardt with adaptive damping; batched subproblems, each with its own steps and stopping rule |
| **Constraints** | `AugmentedLagrangianMinimizer`: equality and inequality constraints (any factor can be a constraint) and box bounds on vector states (projected); see [Constraints](docs/sphinx/constraints.rst) |
| **Robust estimation (RANSAC)** | `RansacGaussNewtonMinimizer`, `RansacLevenbergMarquardtMinimizer`: hundreds of hypotheses solved in parallel on the GPU from minimal samples, MSAC scoring, adaptive stopping, refinement on the inliers, per-factor inlier mask. Works with any factor and state type (built-in or custom) whose total free dimension is ≤ 64; any number of factors. Deterministic for a fixed seed; see [RANSAC](docs/sphinx/ransac.rst) |
| **Robust losses** | Huber, Cauchy, Arctan, SoftL1, Tolerant, Tukey, Scaled |
| **Built-in factors** | Reprojection, PnP, between (SO(2)/SO(3)/SE(2)/SE(3)/Sim(2)/Sim(3)/SL(4)/vector), point-to-point, point-to-plane, symmetric point-to-plane, prior, IMU preintegration ([IMU](docs/sphinx/imu.rst)), constant-velocity/constant-acceleration motion priors (SO(2)/SO(3)/SE(2)/SE(3)), dynamics (kinematics, differential drive, bicycle, quadrotor, quadruped; [Dynamics](docs/sphinx/dynamics.rst)), clearance and half-space constraints |
| **Model-predictive control** | `pycunls.mpc`: batched receding-horizon problems from the dynamics factors, costs and constraints, solved in closed loop; see [MPC](docs/sphinx/mpc.rst) |
| **Custom factors and states** | NVIDIA Warp kernels in Python, or user-defined CUDA kernels via `SizedFactorBatch` / `SizedStateBatch` in C++; the same types work with every minimizer, including RANSAC — see [Custom factors and states](docs/sphinx/custom_factors_and_states.rst) |
| **Numeric Jacobians** | Finite-difference Jacobians for any factor batch (manifold-aware, reuses each state's `Plus` retraction), selectable globally (`MinimizerOptions.jacobian_mode`) or per factor group (the `jacobian_mode_override` of `Problem.add_factor_batch`) — write a factor with only a residual and let cuNLS differentiate it; see [Numeric Jacobians](docs/sphinx/numeric_jacobians.rst) |
| **Linear solver** | Supernodal block-sparse Cholesky (built in, default: exact steps; also solves the Schur complement of bundle adjustment), block-sparse PCG (variable block-Jacobi preconditioner), NVIDIA cuDSS (optional, loaded via `dlopen()` at runtime — see [INSTALL.md](INSTALL.md#cudss)), block-tridiagonal Cholesky, dense LDLT, dense Cholesky (cuSOLVER), dense QR (cuSOLVER). Guide: [Linear systems and solvers](docs/sphinx/linear_solvers.rst) |
| **Self-explaining results** | Every summary has `converged()` and a `message` with the fix; Gauss-Newton, Levenberg-Marquardt and RANSAC summaries also give a `termination_reason` (per subproblem too), the augmented Lagrangian one a `status`; `pycunls.diagnose` / `evaluate` / `check_jacobians` check a problem before and after solving; argument errors name the expected shape and dtype; see [Troubleshooting](docs/sphinx/troubleshooting.rst) |
| **Safety checks** | Optional runtime validation (linear-solver diagnostics and more), off by default for low-latency solves — enable with `MinimizerOptions::disable_safety_checks = false` when debugging |
| **Execution model** | Fully asynchronous via CUDA streams; setup once, then re-solve with new data and active counts without reallocation |

## Installation

cuNLS is built from source (Python wheels on PyPI are coming). With Docker and the NVIDIA
Container Toolkit:

```bash
./scripts/build_pycunls_in_docker.sh        # Python wheel -> ./dist
pip install ./dist/pycunls*.whl
python -m pycunls doctor                    # checks the GPU, the native module, a solve

./scripts/build_cunls_in_docker.sh Release  # C++ library, shared + static -> ./build_docker
```

Local builds, the C++ CMake package, Jetson, cuDSS, testing and building the docs:
[INSTALL.md](INSTALL.md).

## Quick start (Python)

### Fit a model to data

```python
import numpy as np
import warp as wp
import pycunls

@wp.func
def decay(x: float, p: wp.vec3) -> float:   # y = a exp(-b x) + c, for one sample
    return p[0] * wp.exp(-p[1] * x) + p[2]

rng = np.random.default_rng(0)
x = np.linspace(0.0, 4.0, 200)
y = 2.0 * np.exp(-1.3 * x) + 0.5 + rng.normal(0.0, 0.02, x.size)

res = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
print(res.report())                     # parameters ± errors, reduced chi2, why it stopped
```

```text
FitResult: converged (autodiff Jacobian)
  Converged: the predicted relative cost reduction fell below relative_reduction_tolerance; cost 1.166e+05 -> 91.28 in 5 iterations.
  p = [1.9964 ± 0.0063, 1.28707 ± 0.0087, 0.497294 ± 0.0027], reduced chi2 0.927, RelativeReductionTolerance
```

Thousands of fits are one call: give `y` one row per fit. Each fit takes its own steps and stops
on its own:

```python
res = pycunls.fit(decay, x, Y, p0=[1.0, 1.0, 0.0], sigma=0.02)   # Y: (10000, 200)
res.params.shape, res.converged.all()                            # (10000, 3), True
```

The Jacobian comes from Warp's automatic differentiation, checked against central differences.
Robust losses (`loss="cauchy"`), bounds and multiple starts are one argument each: [Quick setup](docs/sphinx/simple_api.rst) and
[`python/examples/model_fitting.py`](python/examples/model_fitting.py).

### Solve a factor graph from arrays

A loop of 32 SE(2) poses with odometry and one loop closure. `ProblemBuilder` takes the initial
poses and an edge list; it checks the problem before solving.

```python
import numpy as np
import pycunls

def se2(x, y, theta):                   # 3x3 pose [R t; 0 1]
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

n = 32
angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
truth = np.stack([se2(np.cos(a), np.sin(a), a + np.pi / 2) for a in angles])
edges = np.array([(i, (i + 1) % n) for i in range(n)])          # (M, 2): [left, right]
deltas = np.linalg.inv(truth[edges[:, 1]]) @ truth[edges[:, 0]]  # T_right^-1 T_left
rng = np.random.default_rng(0)
start = truth @ np.stack([se2(*rng.normal(0.0, 0.05, 3)) for _ in range(n)])
start[0] = truth[0]

pb = pycunls.ProblemBuilder()
poses = pb.add_states(pycunls.SE2StateBatch, start, constant=[0])   # pose 0 fixes the gauge
pb.add_factors(pycunls.SE2BetweenFactorBatch, deltas, connect=(poses, edges))
result = pb.solve()                     # checks the problem first, then Levenberg-Marquardt
print(result.message)                   # why it stopped, and the fix if it failed
T = result[poses]                       # solved poses, NumPy (32, 3, 3)
print(result.evaluate())                # residuals per factor batch
```

`pycunls.describe("SE2BetweenFactorBatch")` gives a factor's states, residual, array layout
and conventions; `pycunls.catalog("factor")` lists them all. Bundle adjustment (two state
types, Schur complement) works the same way:
[`python/examples/sparse_bundle_adjustment.py`](python/examples/sparse_bundle_adjustment.py).

### When something is wrong

Problems are checked before solving and every outcome explains itself. Leave out the anchor
(`constant=[0]`) above, or connect the wrong state type, and you get:

```text
UserWarning: Gauge freedom: 32 connected free states (SE2StateBatch; e.g. state 0 of state batch 0)
are only constrained relative to each other (between / reprojection / motion factors), so the whole
group can move without changing the cost. [...] Fix: make one state of the group constant
(constant ids) or add a prior factor on it.

ValueError: slot 0 of SE3BetweenFactorBatch needs a SE3StateBatch, got a SE2StateBatch: the factor
reads the state layout of SE3StateBatch (4x4 matrix [R t; 0 1], row-major (16 floats)).
Fix: add the states as pycunls.SE3StateBatch.
```

For a pipeline, `result.summary.to_dict()` is a plain dict with no NaN or Inf (non-finite
values become `null`), so it serializes to strict JSON:

```python
import json
print(json.dumps(result.summary.to_dict(), allow_nan=False, indent=1))
```

```json
{
 "converged": true,
 "termination_reason": "CostTolerance",
 "message": "Converged: the cost fell below cost_tolerance (1e-06); cost 0.2329 -> 1.659e-08 in 2 iterations.",
 "num_iterations": 2,
 "initial_cost": 0.23288413882255554,
 "final_cost": 1.658583670405278e-08,
 "num_free_parameters": 93,
 "solve_time_ms": 1.40308,
 "num_problems_by_reason": {"CostTolerance": 1},
 "iteration_costs": [0.23288413882255554, 9.848973422776908e-05]
}
```

More in [Diagnostics and troubleshooting](docs/sphinx/troubleshooting.rst).

### Robust estimation with RANSAC

When some measurements are gross outliers (wrong matches), swap the minimizer; the problem
stays the same. Each factor batch is either *sampled* (may contain outliers, classified with an
inlier threshold in residual units) or *always on* (trusted priors, IMU):

```python
options = pycunls.RansacLevenbergMarquardtMinimizerOptions()
options.base_options.factor_batches = [
    pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, 0.01)]
minimizer = pycunls.RansacLevenbergMarquardtMinimizer(options)
summary = minimizer.minimize(stream, problem)  # estimate written back
mask = minimizer.inlier_mask(0)                # numpy uint8, 1 = inlier
```

On PnP it recovers the pose at up to 90% outliers, where least squares (even with a Huber loss)
fails, in 1.4 ms for 1,000 correspondences and 15 ms for 1,000,000. See
[RANSAC](docs/sphinx/ransac.rst) and [`python/examples/ransac_pnp.py`](python/examples/ransac_pnp.py).

### The full API

`fit` and `ProblemBuilder` build an ordinary `pycunls.Problem`. Built directly, it works on your
own GPU buffers (Warp arrays), which is what real-time loops use: allocate once, then change the data and
the active counts every frame.

```python
import warp as wp
import pycunls

stream = pycunls.CudaStream()
state_gpu = wp.array([0.0], dtype=wp.float32)   # initial guess x = 0
obs_gpu = wp.array([2.0], dtype=wp.float32)     # observation o = 2

state_batch = pycunls.VectorStateBatch1(state_gpu, 1)   # capacity: fixed buffer size
state_batch.set_num_active_states(1)                     # active count: starts at 0
prior = pycunls.PriorVectorFactorBatch1(obs_gpu, 1)
prior.set_num_active_factors(1)

problem = pycunls.Problem()
problem.add_state_batch(state_batch)
problem.add_factor_batch(prior, [state_batch.state_device_ptr(0)])

wp.synchronize_stream(wp.get_stream())         # uploads (Warp's stream) before the solve
summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
print(summary.message, state_gpu.numpy())       # x = 2, updated in place
```

> [!IMPORTANT]
> **Capacity vs. active count.** Every factor and state batch has two sizes: the **capacity**
> its device buffers hold (constructor, fixed) and the **active count** (starts at **zero**,
> set with `set_num_active_factors(n)` / `set_num_active_states(n)`, any `n` up to the
> capacity). `fit` and `ProblemBuilder` set both for you. See
> [Capacity and active count](docs/sphinx/introduction.rst).

Custom factors and states in Warp: [Custom factors and states](docs/sphinx/custom_factors_and_states.rst)
and [`python/examples/custom_warp_factor.py`](python/examples/custom_warp_factor.py). The C++
API mirrors the Python one: [C++ quick start](docs/sphinx/quick_start.rst).

## Examples

- [**Python examples**](python/examples/README.md): model fitting, pose graphs, bundle
  adjustment, RANSAC PnP, visual-inertial bundle adjustment, custom Warp factors and states,
  and two showcases on real data: visual-inertial odometry with RANSAC and a drone fleet flying
  batched MPC through a supermarket.
- [**C++ examples**](examples/README.md): standalone CMake projects for bundle adjustment, pose
  graphs, PnP, RANSAC PnP, motion priors and a custom CUDA factor.

## Documentation

The [documentation](docs/sphinx/index.rst) covers the concepts, the Python and C++ tutorials,
every factor, state, minimizer and linear solver, and troubleshooting. Build it with
`python -m sphinx -b html docs/sphinx docs/sphinx/_build`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, tests and sign-off, and
[AGENTS.md](AGENTS.md) for the architecture and conventions.

## License

cuNLS is licensed under the [Apache License 2.0](LICENSE). Third-party license notices are in
`NOTICE` and `third_party/LICENSES/`.

---

<sup>1</sup> cuNLS (fp32) on an NVIDIA RTX PRO 5000 Blackwell; Ceres, GTSAM and g2o (fp64) on an
Intel Core i9-14900K with OpenBLAS; same problem, start and stopping rule, median of 5 runs.
