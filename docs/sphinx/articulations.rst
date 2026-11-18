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
   problem.add_factor_batch(pose, [states.state_device_ptr(i) for i in range(n)])
   problem.set_problem_partition(n, [wp.array(np.arange(n, dtype=np.int32))])  # stop each alone

   solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer())
   summary = solver.minimize(stream, problem)
   q = wp.zeros(n * topo.coord_count, dtype=float)
   states.get_joint_q(q, n, stream)

Joint limits are box bounds of the state: the iterates never leave them (projected Gauss-Newton,
see :doc:`constraints`), so solve with ``AugmentedLagrangianMinimizer``; the plain minimizers
reject a bounded problem. ``ArticulationStateBatch(topo, n, apply_joint_limits=False)`` leaves
the coordinates unbounded.

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
