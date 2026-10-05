###############################################################################
Model Predictive Control
###############################################################################

``pycunls.mpc`` builds batched receding-horizon optimal control problems from
the dynamics factors (:doc:`dynamics`), the existing cost factors and the
constraints (:doc:`constraints`), and runs them in closed loop. It is a thin
builder: everything it creates is an ordinary :class:`pycunls.Problem` solved
by :class:`pycunls.AugmentedLagrangianMinimizer`, so any factor can be added
to it.

===============================================================================
Example
===============================================================================

.. code-block:: python

   import numpy as np
   import pycunls
   from pycunls import mpc

   stream = pycunls.CudaStream()
   h = mpc.Horizon(mpc.Car(wheelbase=2.75), steps=20, dt=0.1, batch=1)
   h.track_pose(reference, weight=[0.5, 2.0, 1.0])     # reference [1, 21, 3, 3]
   h.track_vector("speed_steer", np.tile([5.0, 0.0], (1, 21, 1)), weight=[1.0, 0.1])
   h.control_effort(weight=0.05)
   h.control_bounds([-3.0, -0.5], [3.0, 0.5])         # acceleration, steering rate
   h.vector_bounds("speed_steer", [0.0, -0.4], [10.0, 0.4])
   ctrl = h.build()

   for t in range(T):
       h.pose_reference[...] = next_reference(t)       # [1, 20, 3, 3], steps 1..N
       u = ctrl.step(stream, pose=measured_pose, speed_steer=measured_speed_steer)
       apply(u[0])

===============================================================================
Horizon
===============================================================================

``Horizon(model, steps, dt, batch=1, hard_dynamics=True, dynamics_weight=100)``
creates ``batch`` independent trajectories (one subproblem each, solved and
stopped independently) of ``steps`` steps:

- ``poses``: ``[B, N + 1, 3, 3]`` (SE(2)) or ``[B, N + 1, 4, 4]`` (SE(3));
- ``vectors[name]``: ``[B, N + 1, d]`` for each vector state of the model;
- ``controls``: ``[B, N, nu]``;
- ``time_steps``: ``[B * N]`` step durations (rewrite for non-uniform steps).

These are CuPy arrays owned by the horizon; write to them to set initial
guesses or to change the problem between solves. The first state of every
trajectory is constant (the measured state). The dynamics are equality
constraints by default, or soft factors with ``hard_dynamics=False``.

Models
   ``Carter(wheel_radius, track_width, terrain=False)`` (wheel speeds),
   ``Car(wheelbase, terrain=False)`` (vector state ``speed_steer``; controls
   acceleration and steering rate), ``Quadrotor(parameters)`` (vector states
   ``velocity``, ``rates``; rotor thrusts), ``Quadruped(parameters)`` (foot
   forces; gait inputs ``contacts`` and ``foot_positions``). ``terrain=True``
   uses SE(3) poses.

Costs
   ``track_pose(reference, weight)``, ``track_vector(name, reference, weight)``
   (references in ``pose_reference`` / ``vector_reference[name]``, rewritable),
   ``control_effort(weight, nominal=None)``, ``control_rate(weight)``. A weight
   is a scalar or one value per residual component.

Constraints
   ``control_bounds(lower, upper)`` and ``vector_bounds(name, lower, upper)``
   are bounds on the state batches, enforced by projection: the controls and
   bounded states stay inside their limits at every iteration (bounds in
   ``control_lower`` / ``control_upper`` and ``vector_lower[name]`` /
   ``vector_upper[name]``). ``goal_pose(goal)`` (equality) and
   ``disk_obstacles(obstacles, margin)`` (inequality; disks for SE(2) models,
   spheres for SE(3)) go through the augmented Lagrangian loop.

``build(minimizer=None, options=None, real_time=None)``
assembles the problem. It declares the stages of the states (x_k and u_k in
stage k, :meth:`pycunls.Problem.set_state_stages`), so the default inner
minimizer, Levenberg-Marquardt, uses the block-tridiagonal linear solver
(``SparseLinearSolverType.BlockTridiagonal``: the Riccati recursion, one warp
per trajectory). The default AL options enable the warm start and the reuse of
the problem structure across steps and cap the penalty at ``mpc.MAX_PENALTY``.

===============================================================================
Real-time mode
===============================================================================

.. code-block:: python

   ctrl = h.build(real_time=(1, 1))   # 1 AL iteration of 1 LM iteration per step

The first :meth:`Controller.step` solves to convergence; every later step runs
a fixed budget of ``outer`` augmented-Lagrangian iterations of ``inner``
Levenberg-Marquardt iterations with all control on the GPU and a single
read-back. The warm start carries the plan and the multipliers
forward, so the iterations of consecutive steps add up (as in real-time
iteration schemes); the penalty is capped at ``mpc.REAL_TIME_MAX_PENALTY``.
On an RTX PRO 5000 a step takes about 0.3 ms for a Carter with a 50-step
horizon, 0.6 ms for a quadrotor (40 steps) and 2.3 ms for 1024 quadrupeds
(20 steps); the closed-loop tests track as well as with converged steps.

The underlying options are :attr:`pycunls.AugmentedLagrangianMinimizerOptions.real_time`
and ``reuse_structure``, and the solver's ``options``
property (assigning keeps the warm-start state).

===============================================================================
Controller
===============================================================================

``Controller.step(stream, pose=..., **vectors)`` shifts the previous solution
one step (states, controls and the constraint multipliers; the last step
repeats), writes the measured state into the first state of every trajectory,
solves, and returns the first controls ``[B, nu]``. The first call fills the
whole horizon with the measured state ("stay where you are"). ``solve``,
``shift`` and ``set_initial_state`` are the individual parts; ``summary`` is
the last :class:`AugmentedLagrangianMinimizerSummary`.

The solver is local: it finds the solution near the initial guess (the
shifted previous plan). An obstacle almost centered on the reference path is
close to a saddle (passing left or right is equally good); start such
problems from a guess that already passes on one side.
