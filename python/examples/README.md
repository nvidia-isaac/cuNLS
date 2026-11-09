# pycunls examples

End-to-end programs using `pycunls`, from a one-line curve fit to visual-inertial odometry and a
drone fleet on real data. Each script keeps its `pycunls` calls inline and reads top to bottom
(data, problem, solve, results); `example_utils/` holds everything else (synthetic data, metrics,
SE(3) helpers, Rerun visualization). The GIFs below come from the two showcases and two
renders of large problems ([gallery](#gallery-large-problems-one-iteration-per-frame)).

| Example | What it shows | Needs |
|---|---|---|
| [`model_fitting.py`](#model-fitting-model_fittingpy) | `pycunls.fit` in six scenes: errors, 20,000 fits in one call, outliers, bounds, multiple starts, model check | CuPy (Warp optional) |
| [`pose_graph_optimization.py`](#pose-graph-optimization-pose_graph_optimizationpy) | SE(3) pose graph with `ProblemBuilder`: NumPy poses and an edge list in, solved poses out | CuPy |
| [`sparse_bundle_adjustment.py`](#sparse-bundle-adjustment-sparse_bundle_adjustmentpy) | Cameras and points with `ProblemBuilder`, points eliminated by a Schur complement | CuPy |
| [`ransac_pnp.py`](#robust-pnp-with-ransac-ransac_pnppy) | PnP with 50% wrong matches: `RansacLevenbergMarquardtMinimizer` vs least squares | CuPy |
| [`imu_bundle_adjustment.py`](#visual-inertial-bundle-adjustment-imu_bundle_adjustmentpy) | `ImuFactorBatch` + reprojection factors: poses, velocities, IMU biases | CuPy |
| [`custom_warp_factor.py`](#custom-factor-in-warp-custom_warp_factorpy) | A factor written as a Warp kernel, with an analytic or a numeric Jacobian | CuPy, Warp |
| [`custom_warp_state.py`](#custom-state-in-warp-custom_warp_statepy) | A state type written in Warp: positive scalars with a multiplicative retraction | CuPy, Warp |
| [`tartan_vio.py`](#visual-inertial-odometry-with-ransac-on-tartanground) | RGB-D inertial odometry on TartanGround with injected outliers: IMU factor + RANSAC | data, OpenCV, Rerun |
| [`supermarket_drones.py`](#a-drone-fleet-in-a-supermarket-batched-mpc) | A fleet of quadrotors in a real store with walking shoppers: batched MPC | data, SciPy, Rerun |

The C++ counterparts are in [`examples/`](../../examples) at the repository root.

## Setup

Build the `pycunls` wheel ([INSTALL.md](../../INSTALL.md)), then, from the repository root,
install it and run the examples from this folder so that `example_utils` is importable:

```bash
pip install ./dist/pycunls*.whl "warp-lang<1.18"   # the wheel brings the matching CuPy
python -m pycunls doctor                            # GPU, native module, a test solve
cd python/examples
python model_fitting.py
```

The small examples (the first seven) make their own synthetic data, run in a few seconds and print
what happened; all but `model_fitting.py` and `imu_bundle_adjustment.py` exit with an error if
the result is wrong. The test suite runs them (`python/tests/test_examples.py`). The two
showcases need downloaded data and log to [Rerun](https://rerun.io)
(`pip install tartanair rerun-sdk opencv-python scipy`).

## Model fitting (`model_fitting.py`)

`pycunls.fit(model, x, y, p0)` fits the parameters of `y ~ model(x, p)`: no states, factors or
device pointers. The model is a NumPy-style function of `(x, p)` (one parameter row per sample);
a Warp `@wp.func` works too and gets autodiff Jacobians. Six scenes, one call each:

1. one decay curve with measurement errors: parameters ± standard errors, reduced chi-square;
2. 20,000 curves (one per pixel of an image) in one batched GPU solve, each fit with its own
   steps and stopping rule;
3. a line with 25% gross outliers: the squared loss is pulled away, a Cauchy loss is not;
4. a parameter that must stay in a range: `bounds=(lower, upper)`;
5. a sine whose frequency the initial guess gets wrong: 64 starts solved together;
6. the model check: reduced chi-square rejects one exponential and accepts two.

```bash
python model_fitting.py            # or: python model_fitting.py --quick (a smaller batch)
```

```text
FitResult: converged (numeric Jacobian)
  p = [2.00779 ± 0.013, 1.31179 ± 0.017, 0.495582 ± 0.0044], reduced chi2 0.82, RelativeReductionTolerance
  truth: [2.0, 1.3, 0.5]
  All 20000 subproblems converged (2708 StateTolerance, 17292 RelativeReductionTolerance); ...
  squared loss: m, q = [1.928, 2.155]  (pulled by the outliers)
  Cauchy loss:  m, q = [2.001, -1.002]  (truth [2.0, -1.0])
  one start:  A, w = [0.075, 1.073], reduced chi2 460.1  (a local minimum)
  64 starts:  A, w = [1.502, 3.699], reduced chi2 0.88  (start #15 won)
  one exponential:  reduced chi2   28.95  -> rejected
  two exponentials: reduced chi2    1.20  -> p = [0.996, 0.496, 1.001, 2.996]
```

Guide: [Quick setup](../../docs/sphinx/simple_api.rst).

## Pose graph optimization (`pose_graph_optimization.py`)

A chain of 201 SE(3) poses joined by relative-pose (between) factors, starting from a perturbed
guess. `ProblemBuilder` takes the poses as an `(N, 4, 4)` NumPy array and the edges as an
`(M, 2)` index array; it creates the state and factor batches, capacities, active counts and
state pointers, and runs `pycunls.diagnose` before solving (a missing anchor, NaN data or a
wrong state type are reported with the fix). Pose 0 is held constant to fix the gauge. The
script prints the summary, the residuals per factor batch (`result.evaluate()`) and the position
error against the ground truth. Python version of
[`examples/pose_graph_optimization`](../../examples/pose_graph_optimization).

```bash
python pose_graph_optimization.py
```

```text
Converged: the cost fell below cost_tolerance (1e-08); cost 48.89 -> 7.45e-09 in 12 iterations.
batch                                     factors        cost        rms     max|r|   worst non-finite
#0 SE3BetweenFactorBatch                      200    7.45e-09  3.524e-06  1.285e-05      58          0
  Max position error: 1.166 -> 0.0145
```

## Sparse bundle adjustment (`sparse_bundle_adjustment.py`)

Jointly refines camera poses (`world_from_camera`) and 3D points from reprojection factors in
normalized image coordinates. Two state types in one `ProblemBuilder`; each factor connects one
camera and one point (`connect=[(cameras, cam_ids), (points, pt_ids)]`). Cameras 0 and 1 are
held constant to fix the gauge (rigid transform and scale). The points are eliminated with a
Schur complement (`problem.set_eliminated_state_batches` +
`linear_system=LinearSystemType.SchurComplement`), and the reduced camera system is solved with
the built-in supernodal Cholesky. Python version of
[`examples/sparse_bundle_adjustment`](../../examples/sparse_bundle_adjustment).

```bash
python sparse_bundle_adjustment.py
```

```text
Sparse Bundle Adjustment (pycunls)
  Initial cost : 5.038471
  Final cost   : 0.000000
  Iterations   : 3
  Point MSE    : 0.040003 -> 0.000000
```

Guide: [Linear systems and solvers](../../docs/sphinx/linear_solvers.rst).

## Robust PnP with RANSAC (`ransac_pnp.py`)

Recovers a camera pose from 3D-2D correspondences of which half are wrong matches. The problem
is built as for any minimizer (one `PnPFactorBatch`); only the minimizer changes:
`RansacLevenbergMarquardtMinimizer` solves 256 hypotheses (minimal samples of three matches) in one batched GPU solve,
scores them (MSAC), refines the best on its inliers and returns a per-factor inlier mask. A
plain `LevenbergMarquardtMinimizer` on the same data is shown for comparison. Python version of
[`examples/ransac_pnp`](../../examples/ransac_pnp).

```bash
python ransac_pnp.py                                   # defaults: 2000 points, 50% outliers
python ransac_pnp.py --num-points 10000 --outlier-ratio 0.8
```

```text
RANSAC PnP example (pycunls): 2000 correspondences, 50% outliers
  Initial guess       : rotation error 6.5019 deg, translation error 0.3855
  LevenbergMarquardt  : rotation error 3.6522 deg, translation error 5.2653
  RansacLM            : rotation error 0.0192 deg, translation error 0.0029
  RANSAC: 1 round(s), 256 hypotheses, 1023 inliers (51.2%)
  Inlier mask: 1023 of 1028 true inliers kept, 0 outliers accepted
```

Guide: [RANSAC](../../docs/sphinx/ransac.rst).

## Visual-inertial bundle adjustment (`imu_bundle_adjustment.py`)

Keyframes of a synthetic trajectory are joined by `ImuFactorBatch` factors (the raw gyro and
accelerometer samples between consecutive keyframes) and observe
landmarks through `ReprojectionFactorBatch`. Both factor types read the same `world_from_rig`
pose states. Velocities and IMU biases start at zero and are recovered by the solve; poses and
landmarks start perturbed. Landmarks are eliminated with a Schur complement.

```bash
python imu_bundle_adjustment.py                        # defaults: 10 keyframes, 40 samples each
python imu_bundle_adjustment.py --keyframes 20 --samples-per-keyframe 80
```

```text
10 keyframes, 40 IMU samples each, 300 landmarks, 1320 observations
LM: 6 iterations, cost 2.5e+05 -> 1.99e-06
max position error   1.26e-03 m
gyro bias  [ 0.01000014 -0.02000053  0.01500006]  (true [ 0.01  -0.02   0.015])
accel bias [ 0.09974974 -0.05017724  0.07994397]  (true [ 0.1  -0.05  0.08])
```

Guide: [IMU preintegration](../../docs/sphinx/imu.rst).

## Custom factor in Warp (`custom_warp_factor.py`)

A residual no built-in factor computes, written as an NVIDIA Warp kernel and solved inside a
regular `Problem` with a built-in prior. A chain of scalars is solved from difference
measurements twice: with a kernel that writes the residual and its analytic Jacobian, and with a
residual-only kernel whose Jacobian pycunls computes by finite differences
(`jacobian_mode_override=pycunls.JacobianMode.numeric`). Python version of
[`examples/custom_factor`](../../examples/custom_factor).

```bash
python custom_warp_factor.py
```

```text
Part 1: analytic Jacobian (Warp)
  Initial cost : 9.964937   Final cost : 0.000000   Iterations : 8
Part 2: residual-only, numeric Jacobian (Warp)
  Initial cost : 9.964937   Final cost : 0.000000   Iterations : 8
```

Guides: [Custom factors and states](../../docs/sphinx/custom_factors_and_states.rst),
[Numeric Jacobians](../../docs/sphinx/numeric_jacobians.rst).

## Custom state in Warp (`custom_warp_state.py`)

A new state type: positive scalars whose retraction is multiplicative,
`x (+) delta = x * exp(delta)`, so the states stay positive while the tangent space is all of R
(scales, variances, rates). A chain of 128 positive values spanning eight orders of magnitude is
solved from log-ratio measurements and a prior, both custom Warp factors.

```bash
python custom_warp_state.py
```

```text
Custom Warp State Batch Example (positive-scalar manifold)
  Initial cost : 9.184696
  Final cost   : 0.000000
  Log MSE      : 0.077129 -> 0.000000
  Range gt     : [2.00, 79210576.00]
  Range opt    : [2.00, 79284784.00]
```

## Visual-inertial odometry with RANSAC on TartanGround

![tartan_vio](assets/tartan_vio.gif)

A legged robot walks 82 m through `OldTownFall` (sequence `Data_anymal/P2000`) of
[TartanGround](https://tartanair.org/tartanground/). Its front camera (640 x 640, 10 Hz, with
depth) and IMU (100 Hz) drive a small odometry:

- **front end** (OpenCV): KLT feature tracks; each track's landmark comes from the depth
  image at its first frame and stays fixed (a frame-to-map odometry);
- **back end** (cuNLS): every frame, one 30-dimensional problem on a two-frame window: pose,
  velocity and IMU bias of the previous and the current frame. `ImuFactorBatch` and the priors
  from the last solve are *always on*; one `PnPFactorBatch` factor per tracked landmark is
  *sampled* and classified by `RansacLevenbergMarquardtMinimizer`. The IMU predicts the motion,
  so two matches per hypothesis are enough, and mismatches that agree with a motion the IMU
  rules out are rejected.

Outliers are injected into the 2D matches the back end sees: swapped matches, a coherent
shift of a third of the tracks (repetitive texture), a camera blackout and 75% clutter.
Noise and biases are added to the dataset's ideal IMU. Visual-only RANSAC, the same window
with a Huber loss instead of RANSAC, and IMU dead reckoning run on the same data:

| method | final position error (82 m walked) |
|---|---|
| **inertial RANSAC (cuNLS)** | **0.72 m (0.9%)**, 5 of 58,106 injected outliers accepted |
| visual-only RANSAC | 2.36 m (2.9%) |
| inertial LM + Huber loss | diverges in the clutter segment |
| IMU dead reckoning | lost within seconds |

```bash
# Data (pip install tartanair; about 1.1 GB)
python -c "import tartanair as ta; ta.init('dataset/tartan_ground'); \
  ta.download_ground(env=['OldTownFall'], version=['anymal'], traj=['P2000'], \
  modality=['image', 'depth', 'imu'], camera_name=['lcam_front'], unzip=True)"

python tartan_vio.py --data dataset/tartan_ground/OldTownFall/Data_anymal/P2000 --spawn
# or --rrd tartan_vio.rrd, then: rerun tartan_vio.rrd
```

## A drone fleet in a supermarket: batched MPC

![supermarket_drones](assets/supermarket_drones.gif)

The TartanGround `Supermarket` is fused from the depth images of its ten trajectories into a
3D map (`example_utils/tartan_map.py`). Shoppers walk the paths the dataset's robot drove,
and a fleet of quadrotors flies deliveries between stations along A* routes through the
aisles. The whole fleet is **one batched MPC problem** (`pycunls.mpc.Horizon(batch=B)`, 40
steps of 25 ms, `QuadrotorFactorBatch` dynamics, thrust bounds), solved together every
control step with each drone its own subproblem. On every step of its horizon each drone
keeps clear of

- the map voxels nearest to its previous plan (shelves, pillars, the floor),
- the shoppers, grown by a 0.6 m personal space and predicted at constant velocity,
- the other drones' previous plans, with a 0.5 m separation zone,

all as sphere clearance constraints rewritten in `Horizon.obstacles` every step. The left
panel shows the store from above, the right one follows drone 0 from behind.

With 6 drones and 10 shoppers over 45 s (RTX A6000): 15 deliveries, drones at least 1.06 m
apart (1.15 kept), at least 0.90 m from a shopper's body (1.00 kept; the gap is prediction
error), about 30 ms per fleet solve. The GIF shows 10 drones and 20 shoppers.

```bash
# Data (about 8 GB: the image and depth of every Supermarket trajectory)
python -c "import tartanair as ta; ta.init('dataset/tartan_ground'); \
  ta.download_ground(env=['Supermarket'], version=['omni', 'diff'], \
  modality=['image', 'depth'], camera_name=['lcam_front'], unzip=True)"

python supermarket_drones.py --env dataset/tartan_ground/Supermarket --spawn \
    --drones 10 --shoppers 20 --duration 30
```

The first run fuses the map (about a minute) and caches it as `fused_map.npz` in the
environment folder (`--map` to choose the file). Guides: [MPC](../../docs/sphinx/mpc.rst),
[Constraints](../../docs/sphinx/constraints.rst).

## Gallery: large problems, one iteration per frame

Two renders of cuNLS refining large problems, one Gauss-Newton / Levenberg-Marquardt iteration
per frame. The rendering scripts are not part of the repository; the closest runnable examples
are named below.

<p align="center">
  <img src="assets/sphere_refine.gif" alt="Sphere pose-graph refinement" width="49%">
  <img src="assets/kepler_orbits.gif" alt="Kepler orbit fitting" width="49%">
</p>

- **Sphere pose-graph refinement** (left). 200k points that should lie on a sphere are
  connected by relative (between) factors and start as a disturbed blob; cuNLS drives them back
  onto the sphere. Color encodes the per-point error, cooling from hot to calm as the solve
  converges. Same building blocks as `pose_graph_optimization.py`.
- **Kepler orbit fitting** (right). A family of orbits is observed as noisy 3D points; cuNLS
  estimates the five Keplerian elements per orbit with a custom Warp factor on an R^5 state, and
  a jittered tangle of loops organizes into a nested rosette of tilted ellipses. Same building
  blocks as `custom_warp_factor.py`.
