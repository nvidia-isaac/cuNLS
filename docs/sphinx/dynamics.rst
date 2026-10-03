###############################################################################
Dynamics factors
###############################################################################

A dynamics factor ties consecutive states of a trajectory together through a
discrete-time model. Its residual is the multiple-shooting defect of an Euler
step over the step duration :math:`\Delta t_k`,

.. math::

   r = \mathrm{Log}\big((T_k\,\mathrm{Exp}(\Delta t_k\, \xi_k))^{-1} T_{k+1}\big)

on the pose (and the difference on vector states), with analytic Jacobians.
Use it as a **soft** factor (weighted, e.g. for estimation) or as a **hard**
constraint wrapped in ``ConstraintFactorBatch(..., ConstraintKind.Equality)``
and solved with ``AugmentedLagrangianMinimizer`` (see :doc:`constraints`).

===============================================================================
Factors
===============================================================================

``SE2DifferentialDriveFactorBatch(time_steps, wheel_radius, track_width, capacity)``
   Two driven wheels plus passive casters (NVIDIA Carter), kinematic. States:
   pose :math:`T_k` (``SE2StateBatch``), wheel speeds
   :math:`(\omega_L, \omega_R)` (``VectorStateBatch2``), pose :math:`T_{k+1}`.
   Body twist :math:`\xi = [r(\omega_R + \omega_L)/2,\ 0,\ r(\omega_R - \omega_L)/b]`;
   the step is exact for wheel speeds held over the step.

``SE2KinematicBicycleFactorBatch(time_steps, wheelbase, capacity)``
   Car with Ackermann steering (kinematic bicycle at the rear axle). States:
   pose :math:`T_k`, :math:`z_k = (v, \delta)`, control
   :math:`u_k = (a, \dot\delta)`, pose :math:`T_{k+1}`, :math:`z_{k+1}`.
   Body twist :math:`\xi = [v,\ 0,\ v \tan\delta / L]`; the speed and
   steering rows are :math:`z_{k+1} - z_k - \Delta t_k u_k`. Valid at moderate
   lateral acceleration (no tire slip).

``SE3DifferentialDriveFactorBatch``, ``SE3KinematicBicycleFactorBatch``
   The same vehicles on non-planar terrain (slopes, ramps): the pose is an
   ``SE3StateBatch`` (vehicle frame x forward, z up) and the wheels set the
   body twist :math:`[0, 0, \omega, v, 0, 0]` (tangent order
   :math:`[\phi, \rho]`). The residual keeps four of the six components of
   :math:`\mathrm{Log}((T_k \mathrm{Exp}(\Delta t_k \xi))^{-1} T_{k+1})`: yaw,
   forward, lateral (no sideslip) and vertical (no hop). Roll and pitch
   changes follow the terrain and come from other factors (an attitude or
   terrain-normal prior, a height map, ground contact); without them they are
   unobserved.

``SE2KinematicsFactorBatch``, ``SO3KinematicsFactorBatch``, ``SE3KinematicsFactorBatch``
   Lie-group kinematics :math:`X_{k+1} = X_k \mathrm{Exp}(\Delta t_k \xi_k)` with
   the body twist as control (a ``VectorStateBatch3`` / ``VectorStateBatch6``):
   planar and 3-D camera or robot motion with commanded velocities, attitude
   kinematics. Exact for a twist held over the step.

``QuadrotorFactorBatch(time_steps, parameters, capacity)``
   Quadrotor rigid body with the four rotor thrusts as controls. States: pose
   (``SE3StateBatch``), world velocity, body rates, thrusts
   (``VectorStateBatch4``), then the next pose, velocity and rates. Euler step
   of :math:`\dot p = v`, :math:`\dot R = R \omega^\wedge`,
   :math:`\dot v = -g e_3 + R e_3 \sum f_i / m - D v`,
   :math:`J \dot\omega = \tau(f) - \omega \times J\omega`, with the X-layout
   allocation of ``QuadrotorParameters``. Needs short steps (10-25 ms).

``QuadrupedFactorBatch(time_steps, contacts, foot_positions, parameters, capacity)``
   Quadruped base as a single rigid body driven by the four foot forces
   (world frame, ``VectorStateBatch12``):
   :math:`m \dot v = \sum s_i f_i - m g e_3`,
   :math:`J \dot\omega + \omega \times J\omega = R^\top \sum s_i (p_i - p) \times f_i`.
   The contact flags :math:`s_i` and foot positions :math:`p_i` are per-step
   inputs from the gait planner. Contact constraints (normal force, friction
   cone, zero force in swing) are separate constraint factors.

``time_steps`` is a device buffer of one step duration per factor.

===============================================================================
Example: a Carter drives to a goal
===============================================================================

.. code-block:: python

   import cupy as cp, numpy as np, pycunls

   steps, dt = 30, 0.1
   poses = cp.asarray(np.tile(np.eye(3, dtype=np.float32), (steps + 1, 1, 1)))
   u = cp.zeros((steps, 2), dtype=cp.float32)
   x = pycunls.SE2StateBatch(poses, steps + 1, cp.asarray([0], dtype=cp.int32), 1)
   x.set_num_active_states(steps + 1, 1)                       # x_0 is fixed
   uu = pycunls.VectorStateBatch2(u, steps)
   uu.set_num_active_states(steps)

   dts = cp.full(steps, dt, dtype=cp.float32)
   dyn = pycunls.SE2DifferentialDriveFactorBatch(dts, 0.125, 0.5, steps)   # r, b
   dyn.set_num_active_factors(steps)
   hard = pycunls.ConstraintFactorBatch(dyn, pycunls.ConstraintKind.Equality)
   # ... goal: SE2 prior on the last pose wrapped as an equality,
   #     effort: weighted vector prior on u, limits: BoundFactorBatch2 on u.

   problem = pycunls.Problem()
   problem.add_state_batch(x)
   problem.add_state_batch(uu)
   ptrs = []
   for k in range(steps):
       ptrs += [x.state_device_ptr(k), uu.state_device_ptr(k), x.state_device_ptr(k + 1)]
   problem.add_factor_batch(hard, ptrs)
   # problem.add_factor_batch(goal / effort / limits ...)
   pycunls.AugmentedLagrangianMinimizer(pycunls.GaussNewtonMinimizer()).minimize(
       pycunls.CudaStream(), problem)
