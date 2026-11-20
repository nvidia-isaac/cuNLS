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

```bash
pip install pycunls-cu13 "cuda-toolkit[cudart,cublas,cusparse,cusolver,nvjitlink]==13.2.*"   # CUDA 12: -cu12, ==12.8.*
python -m pycunls doctor
```

Wheels for x86_64 Linux, Python 3.10-3.14; CUDA libraries, cuDSS and source builds (C++ library,
Jetson): [INSTALL.md](INSTALL.md).

**cuNLS** solves nonlinear least-squares problems entirely on the GPU: any factor graph over
Euclidean and manifold states, with robust losses, constraints and bounds,

$$x^* = \arg\min_x \sum_i \rho_i\!\left(\left\|f_i(x)\right\|^2_{\Sigma_i}\right) \quad \text{s.t.} \quad c_j(x) = 0,\ \ c_k(x) \le 0,\ \ l \le x \le u$$

from curve fitting and thousands of small independent problems in one call to bundle
adjustment, pose graphs, visual-inertial odometry and model-predictive control. On the
[BAL](https://grail.cs.washington.edu/projects/bal/) Venice problem (1,776 cameras, 993k points)
it converges in 0.53 s, 53× faster than Ceres; on the torus3D pose graph it is 24-34× faster
than Ceres, GTSAM and g2o.<sup>1</sup>

- **Python (`pycunls`)** works on [NVIDIA Warp](https://github.com/NVIDIA/warp) arrays (or any
  `__cuda_array_interface__` array); custom factors, states and `fit` models are Warp kernels.
  **C++ (`cunls`)**: shared or static library, x86_64 and Jetson.
- **Agent-ready**: problems are checked before solving, every result carries a
  `termination_reason` and a `message` with the fix, and `python -m pycunls guide` prints the
  [agent guide](python/pycunls/AGENT_GUIDE.md) shipped in the wheel.

## Features

| | |
|---|---|
| **States** | Vectors, SO(2), SO(3), SE(2), SE(3), Sim(2), Sim(3), SL(4), articulated robots |
| **Factors** | Reprojection, PnP, between, prior, point-to-point / plane, IMU, motion priors, vehicle and robot dynamics, robot kinematics, clearance; custom factors in Warp or CUDA |
| **Solvers** | Gauss-Newton, Levenberg-Marquardt; batched subproblems with their own steps and stopping |
| **Constraints** | Equality, inequality and box bounds (augmented Lagrangian) |
| **RANSAC** | Hundreds of hypotheses in parallel, any factor and state type (free dimension ≤ 64) |
| **Robust losses** | Huber, Cauchy, Arctan, SoftL1, Tolerant, Tukey, Scaled |
| **Linear solvers** | Supernodal block-sparse Cholesky (default, also for the Schur complement), block-sparse PCG, cuDSS, block-tridiagonal, dense LDLT / Cholesky / QR |
| **Jacobians** | Analytic, Warp autodiff, or finite differences per factor |
| **Real time** | Fully asynchronous on CUDA streams; fixed buffers re-solved with new data, no reallocation |

## Quick start

### Fit a model

```python
import numpy as np, warp as wp, pycunls

@wp.func
def decay(x: float, p: wp.vec3) -> float:   # y = a exp(-b x) + c, for one sample
    return p[0] * wp.exp(-p[1] * x) + p[2]

x = np.linspace(0.0, 4.0, 200)
y = 2.0 * np.exp(-1.3 * x) + 0.5 + np.random.default_rng(0).normal(0.0, 0.02, x.size)
res = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
print(res.report())   # p = [1.9964 ± 0.0063, 1.28707 ± 0.0087, 0.497294 ± 0.0027], reduced chi2 0.927
```

Give `y` one row per fit (e.g. `(10000, 200)`) and all fits are solved in one call, each with its
own steps and stopping rule. Robust losses, bounds and multiple starts are one argument each:
[Quick setup](docs/sphinx/simple_api.rst).

### Solve a factor graph

```python
pb = pycunls.ProblemBuilder()
poses = pb.add_states(pycunls.SE2StateBatch, start, constant=[0])   # (N, 3, 3); pose 0 fixes the gauge
pb.add_factors(pycunls.SE2BetweenFactorBatch, deltas, connect=(poses, edges))  # edges (M, 2)
result = pb.solve()     # checks the problem, then Levenberg-Marquardt
print(result.message)   # why it stopped, and the fix if it failed
T = result[poses]       # solved poses
```

`pycunls.describe("SE2BetweenFactorBatch")` gives a factor's states, residual, layout and
conventions; `pycunls.catalog()` lists them all. Mistakes such as a missing anchor or a wrong state
type are reported with the fix before solving: [Troubleshooting](docs/sphinx/troubleshooting.rst).

### RANSAC

Swap the minimizer; the problem stays the same. Sampled factor batches may contain outliers:

```python
options = pycunls.RansacLevenbergMarquardtMinimizerOptions()
options.base_options.factor_batches = [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, 0.01)]
minimizer = pycunls.RansacLevenbergMarquardtMinimizer(options)
summary = minimizer.minimize(stream, problem)
inliers = minimizer.inlier_mask(0)
```

PnP with up to 90% outliers: 1.4 ms for 1,000 correspondences ([RANSAC](docs/sphinx/ransac.rst)).

### Real-time loops

`fit` and `ProblemBuilder` build an ordinary `pycunls.Problem`; built directly on your own Warp
buffers it is re-solved every frame without reallocation. Every batch has a fixed **capacity** and
an **active count** that starts at zero (`set_num_active_states` / `set_num_active_factors`):
[Capacity and active count](docs/sphinx/introduction.rst), [C++ quick start](docs/sphinx/quick_start.rst),
[Custom factors and states](docs/sphinx/custom_factors_and_states.rst).

## More

- [Python examples](python/examples/README.md) (fitting, pose graphs, bundle adjustment, RANSAC
  PnP, visual-inertial odometry, batched MPC of a drone fleet) and
  [C++ examples](examples/README.md).
- [Documentation](docs/sphinx/index.rst) (build: `python -m sphinx -b html docs/sphinx
  docs/sphinx/_build`), [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) for the
  architecture.
- License: [Apache 2.0](LICENSE); third-party notices in `NOTICE` and `third_party/LICENSES/`.

---

<sup>1</sup> cuNLS (fp32) on an NVIDIA RTX PRO 5000 Blackwell; Ceres, GTSAM and g2o (fp64) on an
Intel Core i9-14900K with OpenBLAS; same problem, start and stopping rule, median of 5 runs.
