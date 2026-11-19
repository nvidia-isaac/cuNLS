###############################################################################
Articulations (kinematic trees)
###############################################################################

cuNLS optimizes over the generalized coordinates of articulated mechanisms (robot arms, hands,
humanoids, closed-chain linkages): inverse kinematics with hard joint limits, closed loops and
mimic joints, many problems per call.

* ``ArticulationTopology``: the kinematic tree, built once from joint arrays (Newton's layout)
  or a URDF file (``pycunls.articulation.topology_from_urdf``).
* ``ArticulationStateBatch``: many instances of one topology; each holds its ``joint_q`` and a
  cache of link poses that the factors read.
* Kinematic factors: ``LinkPositionFactorBatch``, ``LinkOrientationFactorBatch``,
  ``LinkPoseFactorBatch``, ``PostureFactorBatch``, ``LoopClosureFactorBatch`` and
  ``MimicJointFactorBatch``. Every Jacobian is analytic.

===============================================================================
The convention
===============================================================================

A topology has J joints and J links: link *i* is the child of exactly one tree joint. Per joint:
``joint_type``, ``joint_parent`` (a link, or -1 for the root frame ``root_pose``),
``joint_child``, ``joint_X_p`` and ``joint_X_c`` (the joint frame in the parent and in the child
link frame: position, quaternion x, y, z, w), ``joint_q_start`` / ``joint_qd_start`` (coordinate
and DOF offsets, J + 1 entries), ``joint_dof_dim`` (linear and angular axis counts of D6 joints);
per DOF ``joint_axis`` and the limits. These are Newton's ``Model`` arrays for one articulation.

Forward kinematics, with A the parent-anchor frame:

.. math::

   A = X_{world}(\text{parent})\; X_p, \qquad
   X_{world}(\text{child}) = A\; X_{joint}(q)\; X_c^{-1}

.. list-table::
   :header-rows: 1

   * - Joint
     - Coordinates
     - Tangent (DOFs)
     - :math:`X_{joint}(q)`
   * - PRISMATIC
     - 1
     - 1
     - translation :math:`a\,q`
   * - REVOLUTE
     - 1
     - 1
     - rotation about :math:`a` by :math:`q`
   * - BALL
     - 4 (quaternion x, y, z, w)
     - 3: rotation vector in A, :math:`quat \leftarrow Exp(\delta) \otimes quat`
     - rotation :math:`quat`
   * - FREE
     - 7 (position, quaternion)
     - 6: :math:`(\delta p, \delta\theta)` in A
     - :math:`(p, quat)`
   * - FIXED
     - 0
     - 0
     - identity
   * - D6
     - linear + angular axes
     - one per axis
     - translation :math:`\sum a_k q_k`, then the angular axes as intrinsic rotations (each axis
       rotated by the rotations before it); angular axes must be orthogonal

The tangent is cuNLS's own chart: FREE's :math:`\delta p` is the change of the joint position in
A, not Newton's center-of-mass referenced ``joint_qd``. Loop-closing joints are **not** part of the
tree; add each one as a ``LoopClosureFactorBatch``.

===============================================================================
Python example: bounded inverse kinematics from a URDF
===============================================================================

.. code-block:: python

   import numpy as np
   import warp as wp
   import pycunls
   from pycunls.articulation import topology_from_urdf

   robot = topology_from_urdf("fr3.urdf")          # limits become hard bounds
   topo, n = robot.topology, 1024                    # 1024 IK problems
   states = pycunls.ArticulationStateBatch(topo, n)
   states.set_num_active_states(n)
   stream = pycunls.CudaStream()
   states.set_joint_q(wp.array(q_initial), n, stream)  # (n, topo.coord_count)

   hand = robot.link_names.index("fr3_hand")
   links = wp.full(n, hand, dtype=wp.int32)
   pose = pycunls.LinkPoseFactorBatch(topo, links, None, wp.array(targets), None, n)  # (n, 7)
   pose.set_num_active_factors(n)

   problem = pycunls.Problem()
   problem.add_state_batch(states)
   instance = wp.array(np.arange(n, dtype=np.int32))       # factor i acts on instance i
   problem.add_factor_batch(pose, [states], instance)
   problem.set_problem_partition(n, [instance])             # each problem converges and stops alone

   solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer())
   summary = solver.minimize(stream, problem)
   q = wp.zeros((n, topo.coord_count), dtype=float)
   states.get_joint_q(q, n, stream)

Joint limits are box bounds of the state: the iterates never leave them (projected Gauss-Newton,
see :doc:`constraints`), so solve with ``AugmentedLagrangianMinimizer``; the plain minimizers
reject a bounded problem. ``ArticulationStateBatch(topo, n, apply_joint_limits=False)`` leaves
the coordinates unbounded.

===============================================================================
Building an inverse-kinematics solver
===============================================================================

cuNLS has no IK solver class: an application (a simulator, a robot stack) assembles one from the
pieces above, which keeps its objectives, seeding and result selection in its own hands. The
pieces that make that cheap:

* **Scene arrays in place.** ``set_joint_q`` / ``get_joint_q`` take a ``stride`` (floats between
  instances), so instances embedded in a larger array are read and written without a gather:
  with one articulation per replicated world of a scene-wide ``joint_q``,
  ``states.set_joint_q(scene_q[first:], worlds, stream, stride=coords_per_world)``.
  ``ArticulationTopology`` accepts GPU arrays (copied once to the host), so the joint arrays of
  a scene can be passed as they are.
* **Forward kinematics and link poses.** ``set_joint_q`` runs forward kinematics;
  ``get_link_transforms(out, n, stream, stride=0)`` writes every link's world pose (7 floats:
  position, quaternion x, y, z, w; the convention of the factor targets) of every instance, with
  a stride for writing into a scene-wide pose array. Use it to check a solution (position and
  angle errors) or to pick the best of several seeds.
* **Targets change, the problem does not.** Factors read their device arrays (targets, weights,
  links) at every evaluation: write the next targets into the same arrays (``wp.copy``, a
  kernel) and call ``minimize`` again. With ``reuse_structure = True`` in the minimizer options,
  repeated solves skip the structure setup.
* **Several seeds per problem.** Make ``problems x seeds`` instances (seeds next to each other),
  give each instance its own subproblem, solve once, then compare the link-pose errors of each
  problem's seeds and keep the best one.
* **Frozen DOFs.** ``set_bounds(lower, upper)`` with ``lower == upper == q`` on a scalar DOF holds
  it at ``q`` (a box of zero width); other DOFs keep their limits.
* **Objectives are factors.** Position, orientation and pose targets (with per-row weights),
  posture regularization, loop-closing joints and mimic joints are factors; constraints are
  factors wrapped in ``ConstraintFactorBatch``. Several targets per instance are several factors
  (or several rows of one batch) connected to the same instance.

===============================================================================
Closed loops and mimic joints
===============================================================================

A loop-closing joint keeps two joint frames together in the directions its type does not allow
(BALL 3 rows, REVOLUTE 5, PRISMATIC 5, FIXED 6). Wrap it as an equality constraint:

.. code-block:: python

   loop = pycunls.LoopClosureFactorBatch(topo, pycunls.ArticulationJointType.Revolute,
                                         parent_links, child_links, frames, axes, n)
   problem.add_factor_batch(pycunls.ConstraintFactorBatch(loop, pycunls.ConstraintKind.Equality),
                            ptrs)

``frames`` holds ``joint_X_p`` and ``joint_X_c`` of the loop-closing joint (14 floats per factor).
Rows may be redundant (a planar four-bar has 5 rows of rank 2); the augmented Lagrangian
tolerates that. To assemble a mechanism at given actuated angles, add a ``PostureFactorBatch``
with a high weight on the actuated DOFs and a low one on the passive DOFs: the result is the
nearest valid configuration.

A mimic joint ``q_m = a + b q_ref`` (URDF ``<mimic>``, Newton ``joint_mimic_*``) is a
``MimicJointFactorBatch`` wrapped the same way; ``topology_from_urdf`` returns the couplings in
``mimic_joints``.

===============================================================================
Performance notes
===============================================================================

* The state's ``Plus`` runs forward kinematics once per update and stores link poses and motion
  columns; factors read them, so their cost does not grow with the chain length.
* One subproblem per IK problem (``set_problem_partition``): each one converges and stops on its
  own.
* The Jacobian of a factor is dense over the articulation's DOFs; the Hessian assembler stages
  each factor's block in shared memory, which bounds an articulation at roughly 140 DOFs.
* After writing coordinates through ``state_device_ptr`` directly, call ``refresh(stream)``
  before ``pycunls.evaluate``; ``set_joint_q`` and the minimizers refresh on their own.
