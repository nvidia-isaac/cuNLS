# Constraints, dynamics factors and MPC in cuNLS

Status: design proposal, on branch `dev/ak/constraints_mpc` (from `main`).

### Dependencies on the experimental PyTorch branch

This work starts from `main`. Three things it relies on exist so far only on
the experimental branch `dev/ak/updates_v2` (differentiable solves):

| needed | on `dev/ak/updates_v2` | plan here |
|---|---|---|
| per-subproblem step control (`Problem::SetProblemPartition`, `ProblemPartition`): batched MPC instances that converge independently | commit "Per-subproblem convergence for batched problems" (core C++ only) | **port first** (phase C0): it has no PyTorch dependency |
| core bug fixes found there: LM accepting steps to a NaN cost; SO3/SE3 log and Jacobian precision; SE2 / Sim2 / Sim3 / SL4 math | several commits | **port with C0** (dynamics factors on SE(2)/SE(3) hit exactly these code paths) |
| implicit backward, `NLSLayer`, `pycunls.lie` | the PyTorch work | **wait**: differentiable MPC (phase DM1) starts once that work lands on `main` |

Everything else (constraints, dynamics factors, the MPC builder, the
real-time mode, the tridiagonal solver) depends only on `main`.

## 1. Summary

cuNLS minimizes ½ Σ‖r_i(x)‖² without constraints. Optimal control, motion
planning, articulated-body fitting and much of engineering design need
constraints: control limits, joint limits, dynamics that must hold exactly,
friction cones, obstacle clearance, goal conditions. This document adds three
things:

1. **Constraints**: equality `c(x) = 0`, inequality `c(x) ≤ 0` and bounds
   `lb ≤ x ≤ ub`, solved by an **augmented Lagrangian** (AL) outer loop around
   the existing Gauss-Newton / Levenberg-Marquardt solve. A constraint is just
   a factor batch whose "residual" is the constraint value, so every existing
   and future factor (C++, Warp, numeric-diff) can be used as a constraint.
2. **Dynamics factors**: a library of discrete-time models (wheeled robots,
   quadrotors, legged robots, generic integrators) that tie consecutive states
   and controls together, plus a template for user models.
3. **MPC support**: a Python horizon builder, warm-start shifting, a
   real-time mode (fixed iterations, no host round trips, CUDA-graph capture),
   a batched block-tridiagonal linear solver for time chains, and
   differentiable MPC through the existing backward pass.

The goal is to show cuNLS as a general GPU nonlinear least-squares engine:
the same core solves estimation, control and planning, batched over thousands
of instances, and differentiably.

### 1.1 Principles

- **Reuse the core.** The AL method turns a constrained problem into a sequence
  of unconstrained least-squares problems, exactly what cuNLS already solves
  well. No interior-point or QP machinery inside the minimizer.
- **Constraints are factors.** One wrapper (`ConstraintFactorBatch`) turns any
  factor batch into AL residuals. No separate constraint API, no second
  Jacobian pipeline.
- **Device-resident decisions.** Multiplier and penalty updates, active sets
  and convergence run in kernels, per subproblem, like the partition step
  control. The host reads a few scalars per outer iteration (none in
  real-time mode).
- **Batched first.** One call solves thousands of independent MPC problems
  (robots, sampled initial states, scenario rollouts), each with its own
  penalties and stopping.
- **Analytic Jacobians** for the shipped dynamics models; user models may use
  numeric differentiation (existing `JacobianMode::kNumeric`) or Warp.

### 1.2 Non-goals (first version)

- Exact interior-point or active-set QP solvers (acados/HPIPM territory).
  Constraint satisfaction is to a tolerance (default 1e-4), not to machine
  precision.
- Mixed-integer problems (contact sequence or gear selection as decisions).
  Contact schedules are inputs.
- Hard real-time guarantees. The real-time mode bounds the work per call; it
  does not certify it.
- Whole-body legged dynamics (joint-space, rigid-body algorithms). The
  single-rigid-body model covers the common MPC use; whole-body is a later
  phase.

---

## 2. User view

### 2.1 Constraints on any problem (Python)

```python
# Box limits on a vector state batch (controls): -1 <= u <= 1.
limits = pycunls.BoundFactorBatch3(lower=lo_buf, upper=hi_buf, capacity=n)
problem.add_factor_batch(limits, [controls], control_ids)

# Any factor batch as a constraint: here "the robot ends at the goal".
goal = pycunls.SE2PriorFactorBatch(goal_buf, 1)
problem.add_factor_batch(pycunls.ConstraintFactorBatch(goal, pycunls.ConstraintKind.Equality),
                         [poses], last_pose_id)

# Inequality c(x) <= 0: keep 0.3 m from every obstacle (signed distance).
clear = pycunls.SphereClearanceFactorBatch(centers_buf, radii_buf, margin=0.3, capacity=k)
problem.add_factor_batch(pycunls.ConstraintFactorBatch(clear, pycunls.ConstraintKind.Inequality),
                         [poses], pose_ids)

solver = pycunls.ConstrainedMinimizer(pycunls.LevenbergMarquardtMinimizer(lm_options),
                                      pycunls.ConstrainedOptions(constraint_tolerance=1e-4))
summary = solver.minimize(stream, problem)
summary.max_violation, summary.outer_iterations
```

`ConstrainedMinimizer` works on any problem: with no constraint factors it
reduces to the wrapped minimizer.

### 2.2 MPC (Python)

```python
from pycunls import mpc

model = mpc.Quadrotor(mass=1.0, inertia=(0.01, 0.01, 0.02), arm=0.17)
ocp = mpc.Horizon(model, steps=40, dt=0.025, batch=1024)        # 1024 drones
ocp.cost.track_state(reference=ref_traj, weight=state_weights)   # [1024, 41, ...]
ocp.cost.control_effort(weight=1e-2)
ocp.cost.control_rate(weight=1e-1)
ocp.constraints.control_bounds(lower=0.0, upper=8.0)             # rotor thrusts [N]
ocp.constraints.state_bounds("velocity", lower=-5, upper=5)
solver = ocp.build(real_time=mpc.RealTime(iterations=3))         # graph-captured

for t in control_loop:
    u0 = solver.step(x_measured)        # shift warm start, set x_0, solve, return u_0
```

`mpc.Horizon` is a thin builder over the general API: it creates state and
control batches, dynamics factors, cost factors (existing priors, betweens and
weights), constraint wrappers and the subproblem partition. Everything it
builds is an ordinary `Problem`, so users can add their own factors.

### 2.3 Differentiable MPC (after the PyTorch work lands)

`ocp.layer()` returns an `NLSLayer` / `BatchedNLSLayer` whose inputs are the
reference, the cost weights, model parameters (mass, inertia, drag, wheelbase,
...) and the initial state; outputs are the optimal trajectories and controls.
Training through it learns cost weights from demonstrations (inverse optimal
control), dynamics parameters from logs (system identification through the
controller) or a policy network that outputs MPC references.

---

## 3. Constraints: the math

### 3.1 Problem

```
minimize    ½ Σ_i ‖r_i(x)‖²
subject to  c_E(x) = 0              (m_E equality rows)
            c_I(x) ≤ 0              (m_I inequality rows)
            lb ≤ x_j ≤ ub           (bounds on components of vector states)
```

Bounds are a special case of inequality constraints (`x_j − ub ≤ 0`,
`lb − x_j ≤ 0`) with constant Jacobians; they get their own factor batch for
speed and convenience, not their own algorithm.

### 3.2 Augmented Lagrangian in least-squares form

With multipliers λ (equality), μ ≥ 0 (inequality) and penalties ρ > 0, the
AL function of one constraint row is

```
equality:    λ c + ½ ρ c²              = ½ ρ (c + λ/ρ)²           − λ²/(2ρ)
inequality:  (1/(2ρ)) (max(0, μ + ρ c)² − μ²)
           = ½ ρ max(0, c + μ/ρ)²                                 − μ²/(2ρ)
```

The constant terms do not depend on x, so for fixed (λ, μ, ρ) minimizing the
AL is a **least-squares problem** with one extra residual per constraint row:

```
r_E = √ρ (c_E(x) + λ/ρ)                      J_E = √ρ ∂c_E/∂x
r_I = √ρ max(0, c_I(x) + μ/ρ)                J_I = √ρ ∂c_I/∂x   on active rows, 0 otherwise
```

`r_I` is continuously differentiable in the cost (its square is C¹), and the
Gauss-Newton model with the masked Jacobian is the standard generalized
Gauss-Newton treatment of the max. So the inner solve is exactly the
existing Gauss-Newton / LM iteration, with constraint factors contributing
residuals and Jacobian blocks like any other factor.

### 3.3 Outer loop (per subproblem)

```
initialize λ = 0, μ = 0, ρ = ρ0 (per constraint batch, default 10)
repeat (outer iteration k):
    x ← inner_minimize(x)                       warm-started, at most `inner_iterations`
    v_k ← max violation:  max(|c_E|, max(0, c_I))     (per subproblem)
    λ ← λ + ρ c_E
    μ ← max(0, μ + ρ c_I)
    if v_k > η · v_{k−1}:  ρ ← min(β ρ, ρ_max)        (η = 0.25, β = 10)
until v_k ≤ tol_c and the inner solve converged (stationarity)
```

These are the standard first-order multiplier updates (Powell, Hestenes,
Rockafellar; as in ALTRO and ALGENCAN). All of them are elementwise kernels
over the constraint rows plus one segmented max-reduction per subproblem.
With a partition, each subproblem has its own ρ and stops on its own
(`ProblemPartition` gets per-subproblem violation, ρ and an outer "active"
flag, analogous to the inner step control).

Practical details that matter on the GPU:

- **Inner iterations are few.** The inner solve is warm-started, and an
  accurate inner solve is wasted while multipliers are still far off. Default
  `inner_iterations = 5`, then tightened on the last outer iterations
  (inexact AL, as in ALGENCAN's inner tolerance schedule).
- **Scaling.** Constraint rows are scaled per batch (`ConstraintFactorBatch`
  takes a scale) so that tolerances mean the same in meters, radians and
  newtons.
- **ρ is per constraint batch and per subproblem**, stored in a device array
  `[num_constraint_batches × num_problems]`, so a stiff dynamics constraint
  does not force a huge penalty on a loose obstacle constraint.
- **Penalty caps.** ρ_max (default 1e8 relative to the batch scale) keeps the
  float32 system usable; the outer loop stops with `kMaxPenalty` status if
  the violation cannot be reduced further.

### 3.4 Where this sits in the code

| piece | location | notes |
|---|---|---|
| `ConstraintFactorBatch(inner, kind, scale)` | `cunls/factor/constraint_factor_batch.{h,cu}` | wrapper like `WeightedFactorBatch`: evaluates the inner batch, applies the AL transform with the device multipliers and penalties, masks inactive inequality rows |
| `BoundFactorBatch<Dim>` | `cunls/factor/bound_factor_batch.{h,cu}` | unary on a vector state; lower/upper per component (±inf allowed); produces both inequality rows of each bounded component, fused |
| multiplier storage | owned by the wrapper (`dvector<float> multipliers_`, one per constraint row) | sized from capacity; reset on demand |
| outer loop | `cunls/minimizer/constrained_minimizer.{h,cu}` | wraps a `GaussNewtonMinimizer` (or LM); calls `Minimize` with a reduced iteration cap; runs the update kernels |
| per-subproblem penalties | `ProblemPartition` extension | violation reduction and ρ update per subproblem; one read-back per outer iteration |

The inner minimizer is reused unchanged except for one hook: a
`ConstrainedMinimizer` must be able to call it with "keep the linear solver
and structures; only rerun the iterations" (today `Minimize` re-runs
`Initialize`; the constrained loop needs `Minimize(stream, problem,
{reuse_structure = true})`, which skips the structure rebuild when the
problem has not changed).

### 3.5 Bounds without a penalty (later)

For pure box constraints on vector states, a projected method is cheaper
and exact: clamp in `Plus`, and freeze the active components in the linear
solve. It is a second implementation of the same semantics, so it is
deferred until profiles show the AL bounds to be a bottleneck.

### 3.6 Implicit differentiation with constraints

At an AL solution, the inner problem's stationarity holds at fixed (λ, μ, ρ).
The existing backward pass differentiates exactly that inner problem, so it
works unchanged, but it treats the multipliers as constants. The exact KKT
sensitivity (multipliers respond to θ as well) differs in the constrained
directions by O(1/ρ):

```
exact:        [ H    A_actᵀ ] [w]   [∂L/∂x]          inner (v1):   (H + ρ A_actᵀ A_act) w = ∂L/∂x
              [ A_act  0    ] [ν] = [  0  ]
```

The inner system is the KKT system with the zero block replaced by
`−(1/ρ) I` (eliminate ν = ρ A w). Plan:

- **v1:** inner-problem backward (already implemented); documented error
  O(1/ρ) in the constrained directions, tested against finite differences
  of the whole constrained solve.
- **v2:** exact KKT. Solve with the regularized system and remove the
  regularization by iterative refinement on the KKT residual (each pass is
  one solve with the same matrix; it converges at rate ~1/(ρ λ_min(A H⁻¹Aᵀ))),
  i.e. reuse the backward refinement loop that exists for PCG.

Active-set changes make x\*(θ) nondifferentiable at the switching points
(a known property of constrained optimization layers, also in OptNet and
cvxpylayers); gradients there are one-sided.

---

## 4. Dynamics factors

### 4.1 The common shape

A discrete-time model advances the state from step k to k+1 under control
u_k and parameters p:

```
x_{k+1} = Φ(x_k, u_k, p, dt_k)
```

The dynamics factor connects `(x_k, u_k, x_{k+1})` (plus optional parameter
states) with the multiple-shooting defect

```
r = x_{k+1} ⊖ Φ(x_k, u_k, p, dt)          (⊖: Log(Φ⁻¹ x_{k+1}) on Lie parts, difference on vector parts)
```

and is used either as a **hard constraint** (wrapped in
`ConstraintFactorBatch(..., Equality)`: the trajectory is dynamically
feasible at convergence) or as a **soft factor** with a weight (useful for
estimation-style problems and for robust warm starts).

Jacobians: ∂r/∂x_{k+1} = J_r⁻¹(r) on Lie parts (identity on vector parts);
∂r/∂x_k and ∂r/∂u_k = −J_l⁻¹(r) Ad(…) ∂Φ/∂(·) composed from the model's
continuous Jacobians and the integrator's chain rule (§4.6). Parameters are
inputs (differentiable through `InputVjp`), or vector state batches when they
are to be estimated (system identification: one parameter state shared by
every factor of a trajectory).

State and control layout use existing state batches: poses in
`SE2StateBatch` / `SE3StateBatch` / `SO3StateBatch`, velocities, rates,
steering, controls in `VectorStateBatchN`.

### 4.2 Wheeled robots

**W1. Unicycle / differential drive** (most mobile bases, warehouse AMRs).

- State: pose `T ∈ SE(2)`. Control: `u = (v, ω)` body speed and yaw rate,
  or wheel speeds `(ω_L, ω_R)` with `v = r(ω_R + ω_L)/2`,
  `ω = r(ω_R − ω_L)/b` (wheel radius r, track b).
- Exact discretization (constant twist over dt): `T_{k+1} = T_k Exp(dt [v, 0, ω])`.
- Residual: `Log((T_k Exp(dt ξ(u_k)))⁻¹ T_{k+1}) ∈ R³`.
- Parameters: r, b (wheel-odometry calibration through the same factor).

**W2. Kinematic bicycle (car-like, Ackermann).**

- State: pose `T ∈ SE(2)`, speed v; optional steering angle δ as a state with
  rate control. Control: acceleration a and steering δ (or steering rate).
- Body twist with slip angle `β = atan(l_r tan δ / L)`:
  `ξ = [v cos β, v sin β, v cos β tan δ / L]`; `T_{k+1} = T_k Exp(dt ξ)`,
  `v_{k+1} = v_k + dt a`.
- Parameters: wheelbase L, rear axle distance l_r.
- Typical constraints: |δ| ≤ δ_max, |a| ≤ a_max, steering-rate bounds, lateral
  acceleration `v² tan δ / L ≤ a_lat`.

**W3. Dynamic bicycle with linear tires** (racing, high speed).

- State: pose `T ∈ SE(2)`, body velocities `(v_x, v_y, r)`. Control: drive
  force (or throttle) and steering δ.
- Tire slip angles `α_f = δ − atan((v_y + l_f r)/v_x)`,
  `α_r = −atan((v_y − l_r r)/v_x)`; lateral forces `F_y = C α` (linear) or a
  simplified Pacejka `F_y = D sin(C atan(B α))`.
- `m(v̇_x − v_y r) = F_x − F_yf sin δ`, `m(v̇_y + v_x r) = F_yr + F_yf cos δ`,
  `I_z ṙ = l_f F_yf cos δ − l_r F_yr`; pose by `Exp` of the body twist.
- Integrator: RK4 on the velocity part, Lie-Euler on the pose (§4.6).
- Parameters: m, I_z, l_f, l_r, cornering stiffnesses (learnable).

**W4. Skid-steer** (tracked robots, 4-wheel skid steer): unicycle with
instantaneous-center-of-rotation slip parameters, `v = r(ω_R + ω_L)/2 · s_v`,
`ω = r(ω_R − ω_L)/(χ b)`; χ (effective track) and s_v are the usual
calibrated quantities.

### 4.3 Quadrotors

**Q1. Rigid body with rotor thrusts** (full model).

- State: pose `T = (R, p) ∈ SE(3)`, world velocity `v ∈ R³`, body rates
  `ω ∈ R³`. Control: rotor thrusts `f ∈ R⁴` (or collective thrust and body
  torques).
- Allocation: `[T_c; τ] = M f`, with M from arm length l, rotor positions and
  drag-torque coefficient k_m (X and + layouts, also hexa/octo by M).
- Continuous dynamics: `ṗ = v`, `v̇ = g + R e₃ T_c / m − D v` (optional linear
  drag D), `Ṙ = R ω^`, `ω̇ = J⁻¹(τ − ω × J ω)`.
- Discretization: semi-implicit Lie-Euler (default) or RK4 on (p, v, ω) with
  `R_{k+1} = R_k Exp(dt ω̄)` (RKMK for the rotation; §4.6).
- Parameters: m, J (diagonal or full), l, k_m, D (all learnable: system
  identification of a drone through its controller).
- Typical constraints: `0 ≤ f_i ≤ f_max`, tilt `e₃ᵀ R e₃ ≥ cos θ_max`,
  speed limits, obstacle clearance.

**Q2. Thrust and body-rate control** (the interface most autopilots
expose, e.g. PX4 offboard): state `(T, v)`; control `(T_c, ω_cmd)`; the rate
loop is assumed fast. `R_{k+1} = R_k Exp(dt ω_cmd)`,
`v_{k+1} = v_k + dt (g + R e₃ T_c/m)`. Cheaper and often better conditioned
than Q1 for trajectory tracking.

**Q3. Point mass with acceleration control** (planning layer, differential
flatness): state `(p, v) ∈ R⁶`, control `a ∈ R³`, exact double integrator;
thrust/tilt limits become the constraint `‖a − g‖ ≤ T_max/m` and a cone on
the direction of `a − g`.

### 4.4 Legged robots

**L1. Single rigid body with contact forces** (the standard quadruped MPC
model, e.g. MIT Cheetah convex MPC, here with full nonlinear rotation).

- State: base pose `T ∈ SE(3)`, world velocity v, body rates ω. Control:
  ground reaction forces `f_i ∈ R³` for each foot i.
- Inputs per step: contact schedule `s_{k,i} ∈ {0, 1}` and foot positions
  `p_{k,i}` (from the gait planner; differentiable inputs).
- Dynamics: `m v̇ = Σ_i s_i f_i + m g`;
  `J ω̇ + ω × J ω = Rᵀ Σ_i s_i (p_i − p) × f_i`; `Ṙ = R ω^`.
- Constraints (where the AL machinery earns its keep): unilateral normal
  force `f_n ≥ 0` (and `≤ f_max`), friction cone `‖f_t‖ ≤ μ f_n` (exact
  second-order cone as one smooth inequality `‖f_t‖² − μ² f_n² ≤ 0` with
  `f_n ≥ 0`, or the 4-sided pyramid), zero force in swing (`s = 0`: force
  bounds collapse to 0).
- Parameters: m, J, μ.

**L2. Linear inverted pendulum** (biped walking, ZMP preview control).

- State: CoM position and velocity in the plane `(c, ċ) ∈ R⁴`. Control: ZMP
  `z ∈ R²` (or CoM jerk with ZMP as output).
- `c̈ = (g / h)(c − z)`; exact discretization with `cosh`/`sinh` of
  `dt sqrt(g/h)` (a linear factor with constant Jacobians).
- Constraint: ZMP inside the support polygon of the stance foot or feet
  (linear inequalities per edge, from the footstep plan).

**L3. Centroidal dynamics with kinematics** (later phase): adds joint
positions and the centroidal momentum matrix; depends on a rigid-body
dynamics library and is out of scope for the first version.

### 4.5 Generic models

**G1. Integrator chains**: single, double, triple integrators on `Rⁿ`
(point robots, smooth reference generation, joint-space planning for arms
with acceleration or jerk control). Exact linear discretization.

**G2. Lie-group kinematics**: `T_{k+1} = T_k Exp(dt ξ_k)` with the twist as
control, on SE(2), SE(3), SO(3) (the kinematic core of W1, Q2, camera
motion, satellite attitude).

**G3. User models**: `ContinuousDynamics` + integrator template (§4.6) in
C++ or Warp, analytic or numeric Jacobians.

### 4.6 Integrators and the user-model template

A model provides continuous dynamics and their Jacobians on its tangent
space:

```cpp
struct MyModel {
  static constexpr int kState = 9, kControl = 4, kParams = 3;   // tangent sizes
  __device__ static void Derivative(const State &x, const float *u, const float *p,
                                    float *xdot,                // tangent, body frame
                                    float *A, float *B);        // ∂xdot/∂x, ∂xdot/∂u (nullable)
};
using MyDynamicsFactorBatch = DynamicsFactorBatch<MyModel, RK4>;
```

Integrators compose the stage evaluations and their Jacobians (forward-mode
chain rule through the stages), so the factor has exact Jacobians of the
discrete map:

| integrator | order | Lie parts | cost (model evaluations) |
|---|---|---|---|
| `Euler` | 1 | `X Exp(dt ẋ)` | 1 |
| `SemiImplicitEuler` | 1 (symplectic for mechanical systems) | velocities first, then pose | 1 |
| `Midpoint` | 2 | RKMK-2 | 2 |
| `RK4` | 4 | RKMK-4 (Munthe-Kaas: stages in the Lie algebra with `dexp⁻¹` corrections) | 4 |
| `Exact` | — | model-specific closed form (W1, G1, G2, L2) | 1 |

Python users get the same through Warp (`pycunls.warp` kernels for
`Derivative`, the integrator applied by the library) or through numeric
Jacobians for prototyping.

### 4.7 Cost and constraint factors for MPC

Mostly existing factors, plus a few new ones:

| purpose | factor | status |
|---|---|---|
| state tracking | SE2/SE3 priors, vector priors (per-step weights via `WeightedFactorBatch` / `InformationFactorBatch`) | exists |
| control effort | vector prior toward 0 or a nominal control | exists |
| control smoothness | vector between on consecutive controls | exists |
| terminal cost / goal | priors on the last state, or equality constraint | exists / wrapper |
| control and state bounds | `BoundFactorBatch<Dim>` | new |
| obstacle clearance | `SphereClearanceFactorBatch` (point/sphere robot vs spheres), `EllipsoidClearance`, signed distance field lookup (`SdfClearanceFactorBatch`, trilinear in a 3D texture) | new |
| friction cone, force limits | `FrictionConeFactorBatch` (smooth cone), bounds | new |
| support polygon (ZMP) | `HalfspaceFactorBatch` (aᵀ x ≤ b, per-factor a, b) | new |
| tilt / attitude cone | `AttitudeConeFactorBatch` | new |
| actuator rate limits | between on controls + bounds via `ConstraintFactorBatch` | wrapper |

---

## 5. MPC support

### 5.1 Problem layout

For a horizon of N steps and B instances: state batches hold B·(N+1) states
(`x_0` constant: the measured state), control batches B·N controls; factors
connect step k to k+1 within each instance; the subproblem partition makes
every instance independent (own convergence, own penalties). `mpc.Horizon`
builds exactly this with the existing APIs.

### 5.2 Warm start and the receding horizon

`solver.step(x_measured)`:

1. **Shift**: `x_k ← x_{k+1}`, `u_k ← u_{k+1}` (one kernel per batch), the last
   step extrapolated by repeating the final control and integrating; the
   multipliers shift the same way (warm multipliers are what make one or two
   AL iterations enough in steady state, as in SQP-RTI).
2. Write `x_measured` into the constant `x_0`.
3. Solve (fixed iterations in real-time mode).
4. Return `u_0` (and the full plan).

### 5.3 Real-time mode

Today one `Minimize` performs one host synchronization per iteration
(convergence read-back) plus PCG's periodic convergence polls, and
re-initializes structures on every call. The real-time mode removes all of
them:

- **Structure reuse**: `Initialize` once; later calls skip the structure
  build when the problem topology has not changed (a topology version
  counter on `Problem`).
- **Fixed work**: `inner_iterations` and `outer_iterations` fixed; converged
  subproblems are predicated off on the device (the partition step control
  already decides acceptance and convergence on the device; the single
  problem case is a partition of one).
- **No read-backs**: step control, multiplier and penalty updates write only
  device memory; the summary is read lazily (or never) by the caller.
- **Linear solver without polls**: PCG with a fixed iteration count, or the
  block-tridiagonal solver (§5.4), which is direct.
- **CUDA graph capture**: with the above, a whole solve is a fixed sequence
  of kernels; `solver.step` replays a captured graph, removing launch
  overhead (dominant for small problems: a 40-step horizon has kernels of a
  few microseconds each).

### 5.4 Batched block-tridiagonal solver

Ordered by time, the normal equations of a trajectory are block tridiagonal:
blocks for `(x_k, u_k)` couple only to step k−1 and k+1. A direct solver
exploits this:

- **`SparseLinearSolverType::BlockTridiagonal`**: one instance per thread
  block, block Thomas (Riccati-equivalent) forward elimination and back
  substitution with small dense blocks in shared memory/registers; for long
  horizons with few instances, parallel cyclic reduction across steps.
- The ordering comes from the problem: `mpc.Horizon` declares a time index per
  state (a new `Problem::SetStateOrdering`), and the solver checks that the
  pattern is block tridiagonal under it.
- Expected effect: O(N) work per instance, no iteration-count variability
  (PCG on long chains converges slowly: the condition number grows with N²
  for chain-like systems), fully capturable.

cuDSS also handles these systems (and is the fallback for general
structures); the dedicated solver targets the many-small-instances regime
where cuDSS's per-call overhead dominates.

### 5.5 Differentiable MPC (depends on the PyTorch work)

The MPC problem is an ordinary cuNLS problem, so `NLSLayer` /
`BatchedNLSLayer` apply directly; the new parts are input gradients for the
dynamics-model parameters (`InputVjp` on the dynamics factors: "params",
"dt", "contact_schedule" is non-differentiable) and the constrained backward
(§3.6). Learning tests (as in the differentiable-solve work):

- **M1. Cost weights from demonstrations**: a diff-drive robot tracks paths
  with unknown expert weights; learn the weights so the MPC reproduces the
  expert trajectories.
- **M2. System identification through the controller**: quadrotor with
  wrong mass and drag; learn them from logged closed-loop trajectories.
- **M3. Learned reference**: a small network outputs references for a batch
  of quadrupeds (L1) and is trained on a task loss through the MPC.

---

## 6. Phases

| phase | content | exit criteria |
|---|---|---|
| **C0. Port prerequisites** | subproblem partition and the core bug fixes from `dev/ak/updates_v2` (§ Dependencies) | their tests pass on this branch; no PyTorch code involved |
| **C1. Constraints** | `ConstraintFactorBatch` (equality, inequality), `BoundFactorBatch<Dim>`, `ConstrainedMinimizer` (AL outer loop, per-subproblem penalties), `HalfspaceFactorBatch`; Python bindings; docs | Hock-Schittkowski subset and random convex QPs match a reference solver (cvxpy / scipy) to 1e-4; constrained batched problems with per-subproblem stopping; no regressions |
| **D1. Dynamics library, tier 1** | `DynamicsFactorBatch<Model, Integrator>` template; models W1, W2, Q2, Q3, L2, G1, G2; integrators Euler, semi-implicit, RK4, exact | every model's Jacobians against central differences (float64 reference); trajectories against an independent float64 integration |
| **M1. MPC (eager)** | `pycunls.mpc.Horizon`, warm-start shift, cost and constraint helpers; examples: diff-drive path tracking, car (bicycle) racing line, quadrotor (Q2) waypoint flight, biped LIPM walking | closed-loop simulations track their references; obstacle and bound constraints hold to tolerance |
| **D2. Dynamics library, tier 2** | Q1 (rotor thrusts), W3 (dynamic bicycle), W4 (skid steer), L1 (single rigid body with contacts), `FrictionConeFactorBatch`, `AttitudeConeFactorBatch`, clearance factors | as D1; quadruped trotting in simulation with friction-cone satisfaction |
| **R1. Real-time** | structure reuse, device-only control, fixed-iteration PCG, CUDA-graph capture, `BlockTridiagonal` solver | latency and throughput targets (§7) on a reference GPU |
| **DM1. Differentiable MPC** (after the PyTorch work is on `main`) | parameter `InputVjp` on dynamics factors, constrained backward v1, learning tests M1-M3; then exact KKT backward (v2) | gradients against finite differences of the closed-loop solve; M1-M3 learn |

C0 comes first. C1 is independent of MPC and useful on its own (articulated-body fitting
with joint limits, calibration with physical bounds, any engineering
least-squares problem with limits). D1 does not depend on C1 (soft dynamics
work without constraints); M1 needs both.

## 7. Targets (to be measured)

| scenario | instances | horizon | target |
|---|---|---|---|
| diff-drive tracking, warm, real-time mode | 1 | 50 | < 0.5 ms per step |
| quadrotor (Q2) waypoints, warm | 1 | 40 | < 1 ms per step |
| quadrotor (Q1) batched | 4096 | 40 | < 10 ms per step |
| quadruped (L1) with friction cones | 1 / 1024 | 20 | < 2 ms / < 15 ms per step |

References for comparison: acados (SQP-RTI with HPIPM, CPU; the latency
reference for single instances), Crocoddyl / Aligator (DDP, CPU), ALTRO
(AL-iLQR), cuRobo (GPU, manipulators), GPU MPPI implementations
(sampling-based; the throughput reference for batches). The claim to test:
cuNLS matches CPU solvers on single-instance latency within a small factor
and exceeds them by orders of magnitude in batched throughput, while also
providing gradients.

## 8. Testing

- **Constraint core**: closed-form problems (projection onto a ball, a box,
  a halfspace), equality-constrained least squares against the KKT solution,
  random convex QPs and selected Hock-Schittkowski problems against scipy /
  cvxpy in float64; batched variants where every subproblem has a different
  active set.
- **Dynamics**: Jacobians against finite differences (float64 reference
  implementations in Python, as for the Lie groups); integrator order checks
  (error vs dt slope); energy and momentum behavior of the semi-implicit
  integrator on a free rigid body.
- **MPC**: closed-loop simulations with the model as the plant (nominal) and
  with mismatch (robustness), constraint satisfaction over the whole run,
  determinism of the real-time mode (same inputs, same outputs, same
  iteration count).
- **Differentiable**: finite differences of the whole constrained solve; the
  learning tests M1-M3.

## 9. Open questions

1. **Constraint wrapper vs. separate constraint list.** A wrapper reuses all
   factor infrastructure but mixes constraint rows into the residual vector;
   a separate list on `Problem` would make constraint-specific reporting
   (violation per batch) simpler. Proposal: wrapper, with the
   `ConstrainedMinimizer` discovering wrappers by type.
2. **Inequality handling in the inner solve.** The masked Jacobian is
   standard but can chatter at the activation boundary; a smoothed max
   (softplus with a small temperature) is the alternative if it does.
3. **Multipliers in the backward pass.** Whether v1 (inner problem) is good
   enough for learning tasks in practice, or v2 is needed before DM1 ships.
4. **Ordering for the tridiagonal solver**: explicit time indices from the
   builder (proposed) vs. automatic detection (bandwidth-reducing ordering
   of the Hessian pattern).
5. **Python model authoring**: Warp kernels (consistent with custom factors)
   vs. PyTorch-defined dynamics with `torch.func` Jacobians compiled into a
   factor (lower barrier, slower).
