# Differentiable MPC in cuNLS

Status: design proposal, **deferred**. Work starts once (a) the experimental
PyTorch integration (`dev/ak/updates_v2`: implicit backward, `NLSLayer`,
`BatchedNLSLayer`, `pycunls.lie`; design in
`docs/design/differentiable_pytorch.md` on that branch) is on `main`, and (b)
constraints, dynamics factors and the MPC builder exist
(`docs/design/constraints_and_mpc.md`).

## 1. Summary

A model-predictive controller solves, at every control tick, an optimal
control problem: given the current state x₀, a reference, cost weights and a
dynamics model with parameters, find the controls (and predicted states) over
a horizon that minimize the cost subject to the dynamics and constraints. It
is a function

```
MPC:  (x₀, reference, weights, model parameters, ...)  →  (x*_{1..N}, u*_{0..N-1})
```

**Differentiable MPC** computes the gradients of this function, so that its
inputs can be learned from data instead of tuned by hand:

- cost weights from expert demonstrations (inverse optimal control),
- dynamics parameters (mass, inertia, drag, tire stiffness, wheel radius)
  from logged behavior (system identification through the controller),
- references, terrain or contact parameters produced by a neural network
  trained end to end on a task loss, with the MPC guaranteeing that the
  executed plan respects the dynamics and the constraints.

In cuNLS, an MPC problem is an ordinary (constrained) least-squares problem,
so differentiable MPC is the differentiable-solve machinery applied to it.
This document covers what is specific: gradients with respect to dynamics
parameters, differentiation through constraints, and gradients through many
MPC steps of a closed loop.

### 1.1 What it builds on

| piece | from |
|---|---|
| implicit backward at x\* (one linear solve), `input_gradient`, constant-state gradients | PyTorch branch (core C++) |
| `NLSLayer`, `BatchedNLSLayer`, tangent-space Lie tensors | PyTorch branch (Python) |
| constraints (augmented Lagrangian), dynamics factors, `mpc.Horizon` | constraints/MPC work |

### 1.2 Non-goals (first version)

- Differentiating through the active-set switches (gradients are one-sided
  where a constraint becomes active or inactive).
- Second-order derivatives (Hessians of the loss through the solver).
- Differentiating the iteration count or the real-time mode's truncated
  solve as such: gradients assume the forward converged (see §3.5).

---

## 2. User view

```python
from pycunls import mpc

model = mpc.DiffDrive(wheel_radius=0.05, track=0.4)
ocp = mpc.Horizon(model, steps=30, dt=0.05, batch=256)
ocp.cost.track_state(reference=None, weight=None)       # both provided per call
ocp.cost.control_effort(weight=None)
ocp.constraints.control_bounds(lower=-1.0, upper=1.0)
layer = ocp.layer(learn=["state_weight", "effort_weight", "model.track"])

# Inverse optimal control: fit weights so the MPC reproduces demonstrations.
log_w = torch.zeros(3, requires_grad=True)               # positive weights via exp
opt = torch.optim.Adam([log_w], lr=0.05)
for x0, ref, demo in loader:                              # [B, ...] each
    x, u = layer(x0=x0, reference=ref, state_weight=log_w[:2].exp(),
                 effort_weight=log_w[2].exp())
    loss = ((x - demo) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

`ocp.layer(...)` returns a `BatchedNLSLayer` (one subproblem per instance)
whose inputs are the declared learnable quantities plus x₀ and the reference,
and whose outputs are the planned states and controls. Inputs not declared
learnable stay ordinary buffers.

---

## 3. The math

### 3.1 Recap: implicit gradient at the solution

For an unconstrained solve, at x\* with Jacobian J and residuals r, the
backward pass solves `(JᵀJ) w = ∂L/∂x*` and returns
`∂L/∂θ = −∂/∂θ[(J w)ᵀ r]` for every input θ (exact for inputs whose Jacobian
depends on them, such as weights). This is what the PyTorch branch
implements.

### 3.2 Gradients with respect to dynamics parameters

A dynamics factor has the residual `r = x_{k+1} ⊖ Φ(x_k, u_k, p, dt)`. For a
parameter p (per factor, or shared by a trajectory and broadcast):

```
∂r/∂p = −J_l⁻¹(r) · ∂Φ/∂p          (Lie parts; plain −∂Φ/∂p on vector parts)
```

so each dynamics factor batch gets an `InputVjp("params", u, w, ...)` built
from ∂Φ/∂p. The integrators (Euler, semi-implicit, RK4, exact) already
propagate ∂/∂x and ∂/∂u through their stages (forward mode); ∂/∂p is one more
column block through the same chain rule. Model-specific ∂f/∂p are part of
each model (mass, inertia, drag, wheelbase, cornering stiffness, wheel
radius, track width, friction coefficient).

When a dynamics factor is a hard constraint, its Jacobian enters through the
augmented-Lagrangian residual `√ρ (c + λ/ρ)`; the mixed term
`(∂J/∂p)ᵀ r` is included exactly as for weights (the factor knows w).

Other inputs:

| input | gradient | notes |
|---|---|---|
| x₀ (the measured state) | constant-state gradient | exists on the PyTorch branch |
| reference | prior measurements' `InputVjp` | exists (vector and Lie priors) |
| cost weights | `WeightedFactorBatch` / `InformationFactorBatch` inputs | exists |
| dt (per step) | dynamics `InputVjp("dt")` | for learned time allocation |
| contact schedule, footstep plan | not differentiable (discrete) / foot positions differentiable | L1 model inputs |

### 3.3 Differentiating through constraints

At an augmented-Lagrangian solution the inner problem's stationarity holds at
fixed multipliers and penalties (λ, μ, ρ). Differentiating that inner problem
(the existing backward pass, unchanged) treats the multipliers as constants.
The exact sensitivity of the constrained problem comes from the KKT system
over the active constraints A_act:

```
exact:   [ H      A_actᵀ ] [w]   [∂L/∂x*]        inner (v1):   (H + ρ A_actᵀ A_act) w = ∂L/∂x*
         [ A_act    0    ] [ν] = [   0   ]
```

The inner system is the KKT system with its zero block replaced by
`−(1/ρ) I` (eliminate ν = ρ A_act w), so the two differ by O(1/ρ) in the
constrained directions.

- **v1: inner-problem backward.** Free (already implemented); error O(1/ρ)
  in constrained directions. Tested against finite differences of the whole
  constrained solve, reporting the error as a function of ρ.
- **v2: exact KKT.** Solve with the regularized matrix and remove the
  regularization by iterative refinement on the KKT residual: each pass is
  one more solve with the same factorization / preconditioner and converges
  at rate ~1/(ρ λ_min(A H⁻¹ Aᵀ)). This reuses the refinement loop the
  backward pass already has for iterative solvers.

Inequality constraints contribute only while active (μ > 0 or c > 0 at the
solution); at the switching points x\*(θ) is not differentiable and the
gradient is the one-sided one of the current active set, as in OptNet,
cvxpylayers and the box-DDP differentiable MPC of Amos et al. (2018).

### 3.4 Open loop and closed loop

Two different gradients are useful:

- **Open loop (one solve)**: the loss depends on the plan of a single MPC
  call (imitation of a demonstrated plan, M1 below). One backward pass per
  sample.
- **Closed loop (many solves)**: the loss depends on what happens when the
  MPC drives a plant (simulator or learned model) for T ticks:
  `x_{t+1} = plant(x_t, MPC(x_t, θ)_u0)`. The gradient chains the MPC's
  input gradient (∂u₀/∂x_t and ∂u₀/∂θ) with the plant's Jacobians over time
  (back-propagation through time). Each tick's backward is one implicit
  solve; the chain is ordinary autograd over the T layer calls, so it works
  as long as each call keeps its own linearization. The current one-
  linearization-per-layer rule (calling a layer twice before backward
  raises) needs either one layer per tick or saved linearizations (§5).

### 3.5 Convergence and the real-time mode

Implicit gradients assume a stationary point. A real-time MPC that runs a
fixed small number of iterations is not converged in general; its gradient
is then that of the stationary point it is heading to, not of the truncated
iterate. For learning, use converged solves (training is offline); for
deployment, the learned parameters are used by the real-time controller.
Measuring how much this mismatch matters is test M2b.

---

## 4. Learning tests

As in the differentiable-solve work, each feature ships with a learning test
that shows a model learns something meaningful, small and at scale.

| test | setup | learns | success |
|---|---|---|---|
| **M1. Cost weights from demonstrations** | diff-drive robots track paths; demonstrations produced by an MPC with hidden weights | state / effort / smoothness weights | recovered to a few percent; demonstrations reproduced |
| **M2. System identification through the controller** | quadrotor (Q2) closed loop; plant has true mass and drag, MPC starts with wrong ones | mass, drag | parameters recovered; tracking error drops to the oracle's |
| **M2b. Truncated solves** | M2 with the real-time mode in the loop | as M2 | quantifies the converged-vs-truncated gradient mismatch |
| **M3. Learned references** | batch of quadrupeds (L1 model) on varying terrain; a small network outputs CoM references | network weights | task loss (velocity tracking, foot slip) improves vs. a fixed reference |
| **M4. Constrained gradients** | problems with known analytic sensitivities (projection onto boxes and halfspaces, LQR with input bounds) | — | v1 error matches O(1/ρ); v2 matches finite differences to float32 precision |
| **M-large** | M1 with 4096 instances | as M1 | backward / forward per training step within the 1.5x target |

---

## 5. Implementation plan

| phase | content | exit criteria |
|---|---|---|
| **DM0. Prerequisites** | PyTorch integration on `main`; constraints, dynamics tier 1 and `mpc.Horizon` from the constraints/MPC work | — |
| **DM1. Parameter gradients** | `InputVjp("params")` and `("dt")` on every dynamics factor, through all integrators; `ocp.layer(learn=...)` | every parameter gradient against finite differences (float64 reference); M1 learns |
| **DM2. Constrained backward v1** | AL problems through the existing backward; documentation of the O(1/ρ) error | M4 (v1 part); M2 learns |
| **DM3. Exact KKT backward (v2)** | KKT refinement in the backward pass | M4 (v2 part) |
| **DM4. Closed loop** | saved linearizations so one layer can be called T times before backward (or a per-tick layer pool), BPTT examples | M2b, M3 |
| **DM5. Scale** | batched benchmarks, performance work | M-large |

## 6. Related work

- Amos, Rodriguez, Sacks, Boots, Kolter, "Differentiable MPC for End-to-end
  Planning and Control", NeurIPS 2018: box-constrained iLQR with implicit
  gradients at the fixed point; PyTorch, CPU/GPU, small problems.
- OptNet and cvxpylayers: differentiable convex optimization layers (QPs,
  cone programs) by implicit differentiation of the KKT conditions.
- Theseus (Meta): differentiable nonlinear least squares in PyTorch, with
  implicit and truncated backward modes.
- DiffTaichi, Brax, MJX: differentiable simulators; complementary (the plant
  in closed-loop training, §3.4).

cuNLS's angle: second-order (Gauss-Newton) MPC with constraints, batched over
thousands of instances on the GPU, Lie-group states with tangent-space
gradients, and analytic dynamics Jacobians.

## 7. Open questions

1. **Exact KKT (v2) before or after the first users?** v1 may suffice for
   learning tasks where constraints are rarely active at the solution.
2. **Closed-loop gradients**: saved linearizations per call (memory grows
   with T × problem size) vs. recomputing the linearization in backward from
   saved x\* (time instead of memory).
3. **Gradient through the real-time mode**: whether to offer a truncated
   (unrolled) backward for the fixed-iteration solve, as Theseus does, in
   addition to the implicit one.
4. **Learnable discrete structure** (contact timing): smoothing or
   score-function estimators are out of scope; keep the schedule an input.
