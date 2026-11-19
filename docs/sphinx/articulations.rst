###############################################################################
Articulations (kinematic trees)
###############################################################################

cuNLS optimizes over the generalized coordinates of articulated mechanisms (robot arms, hands,
humanoids, closed-chain linkages): inverse kinematics with hard joint limits, closed loops and
mimic joints, many problems per call.

* ``ArticulationTopology``: the kinematic tree, built once from flat joint arrays
  or a URDF file (``pycunls.articulation.topology_from_urdf``).
* ``ArticulationStateBatch``: many instances of one topology; each holds its ``joint_q`` and a
  cache of link poses that the factors read.
* Kinematic factors: ``LinkPositionFactorBatch``, ``LinkOrientationFactorBatch``,
  ``LinkPoseFactorBatch``, ``PostureFactorBatch``, ``LoopClosureFactorBatch`` and
  ``MimicJointFactorBatch``. Every Jacobian is analytic.
* ``LinkFrameFactorBatch``: any factor over SE(3) poses (built-in or a Warp kernel) on frames of
  links, for objectives of your own (see `Custom kinematic objectives`_).

Read `Conventions`_ first: an articulation has several frames, four index spaces and three pose
layouts, and most mistakes come from mixing them up.

===============================================================================
Conventions
===============================================================================

-------------------------------------------------------------------------------
Frames
-------------------------------------------------------------------------------

All poses are ``world_from_frame`` transforms: a pose :math:`X = (R, p)` maps a point from the
frame into the world, :math:`x_{world} = R\,x + p`. Lengths are in meters, angles in radians.

* **World**: the frame every target, link pose and motion column is expressed in.
* **Root frame** ``root_pose``: where joints with parent -1 attach (the identity by default; set
  it to place a fixed-base robot on a table). A loop closure with parent -1 means the world.
* **Link frame**: each link's own frame; :math:`X_L(q)` is its world pose.
* **Joint frame**: each joint has a frame fixed on its parent link, ``joint_X_p``, and one fixed
  on its child link, ``joint_X_c``; the joint motion :math:`X_{joint}(q)` happens between them.
  :math:`A = X_{parent}(q)\,X_p` is the **parent-anchor frame**: the joint frame as the parent
  carries it, in which the joint axes, BALL rotations and FREE positions are expressed.
* **A frame on a link** (a tool tip, a camera mount): an offset :math:`X_{off}` in the link
  frame; its world pose is :math:`X_L(q)\,X_{off}` (the ``offsets`` / ``points`` /
  ``frame_offsets`` arguments of the link factors).

Forward kinematics, joint by joint from the root (parents first):

.. math::

   A = X_{world}(\text{parent})\; X_p, \qquad
   X_{world}(\text{child}) = A\; X_{joint}(q)\; X_c^{-1}

-------------------------------------------------------------------------------
Index spaces
-------------------------------------------------------------------------------

Four kinds of integer indices describe one articulation; each argument says which one it takes.

.. list-table::
   :header-rows: 1

   * - Index
     - Range
     - Where it comes from
     - Taken by
   * - joint *j*
     - 0 .. ``joint_count`` - 1
     - order of the joint arrays (``joint_type``, ...); URDF: ``joint_names``
     - the topology arrays
   * - link *l*
     - 0 .. ``link_count`` - 1 (= ``joint_count``)
     - link *l* is the child of the joint with ``joint_child[j] == l``; URDF: ``link_names``
     - ``links``, ``parent_links``, ``child_links``, ``topology_links``, ``link_pose_offset``,
       ``get_link_transforms`` output order
   * - coordinate *c*
     - 0 .. ``coord_count`` - 1
     - joint *j* owns ``joint_q[joint_q_start[j] : joint_q_start[j+1]]``
     - ``set_joint_q`` / ``get_joint_q``, ``PostureFactorBatch`` targets
   * - DOF *d*
     - 0 .. ``dof_count`` - 1
     - joint *j* owns ``joint_qd_start[j] : joint_qd_start[j+1]``; URDF: ``dof_names``
     - bounds and limits, posture weights, ``MimicJointFactorBatch`` DOFs, Jacobian columns

Joint *j* and link *j* are the same index only when ``joint_child[j] == j`` (always true for
``topology_from_urdf``, not required for joint arrays). Coordinates and DOFs differ for BALL (4
coordinates, 3 DOFs) and FREE (7, 6) joints; ``topology.scalar_coord[d]`` maps a scalar DOF to its
coordinate (-1 for BALL and FREE DOFs). Two more indices belong to the problem, not the robot:
the **instance** (which articulation of the state batch, ``0 .. capacity - 1``) and the **state
slot** (a factor's *k*-th state, in the order of its ``state_sizes``).

-------------------------------------------------------------------------------
Pose layouts
-------------------------------------------------------------------------------

The same pose appears in three layouts. Quaternions are always (x, y, z, w), as in
Warp's ``wp.transform`` and SciPy's ``Rotation.as_quat()``; libraries that put w first (Eigen's
constructor, MuJoCo) need reordering.

.. list-table::
   :header-rows: 1

   * - Layout
     - Floats
     - Used by
   * - position, quaternion
     - 7: ``px, py, pz, qx, qy, qz, qw``
     - ``joint_X_p`` / ``joint_X_c``, ``root_pose``, link factor targets and offsets,
       ``frame_offsets``, ``get_link_transforms`` (one ``wp.transform`` per link)
   * - row-major 3x4 ``[R | p]``
     - 12
     - the state's cache of link poses (``link_pose_offset``); read-only for factors
   * - row-major 4x4
     - 16
     - ``SE3StateBatch`` states, and the pose a ``LinkFrameFactorBatch`` hands its inner factor

-------------------------------------------------------------------------------
Joint types, coordinates and tangents
-------------------------------------------------------------------------------

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
A, not a velocity referenced to a center of mass. BALL and FREE quaternions are stored
normalized (a zero one becomes the identity). Loop-closing joints are **not** part of the tree;
add each one as a ``LoopClosureFactorBatch``.

-------------------------------------------------------------------------------
Perturbations: which side, which frame
-------------------------------------------------------------------------------

A Jacobian is a derivative with respect to a small step :math:`\delta`; what :math:`\delta`
means differs between the pieces, and a custom factor must use the right one.

.. list-table::
   :header-rows: 1

   * - Quantity
     - Step :math:`\delta`
     - Frame
   * - articulation scalar DOF
     - :math:`q \leftarrow q + \delta`
     - (scalar)
   * - articulation BALL / FREE rotation
     - left: :math:`quat \leftarrow Exp(\delta) \otimes quat`
     - parent-anchor frame A
   * - ``SE3StateBatch`` pose, ``LinkFrameFactorBatch`` pose slot
     - right: :math:`X \leftarrow X\,Exp(\delta)`, :math:`\delta = [\omega, v]`
     - body frame of the pose
   * - motion column of DOF *j* (the cache)
     - world twist :math:`(\omega_j, v_j)` per unit of :math:`q_j`
     - world, at the world origin: a point *x* moves with :math:`\omega_j \times x + v_j`

The built-in orientation residuals (``LinkOrientationFactorBatch``, ``LinkPoseFactorBatch``) are
world-frame rotation vectors :math:`Log(R_{cur} R^{*T})`; ``PostureFactorBatch``'s rotation rows
live in each joint's frame A.

-------------------------------------------------------------------------------
Common mistakes
-------------------------------------------------------------------------------

* Passing a joint index where a link index is expected (or a DOF where a coordinate is
  expected): they coincide only for simple chains of 1-DOF joints.
* Quaternions w first: the rotation is silently wrong. cuNLS reads (x, y, z, w) everywhere.
* Reading the cache as 4x4: link poses are 12 floats (3x4); only pose slots of
  ``LinkFrameFactorBatch`` are 4x4.
* A custom SE(3) factor with a world-frame or left Jacobian: pose slots use the right (body)
  perturbation, like ``SE3StateBatch``. ``pycunls.check_jacobians`` catches it.
* Writing ``state_device_ptr`` coordinates without ``refresh(stream)``: the cache keeps the old
  link poses until the next solve.

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
   stream = pycunls.CudaStream()
   # Warp work on the cuNLS stream: uploads, target updates and reads stay in order with solves.
   wp_stream = wp.Stream(wp.get_cuda_device(), cuda_stream=stream.get_stream())
   with wp.ScopedStream(wp_stream):
       q0 = wp.array(q_initial, dtype=wp.float32)      # (n, topo.coord_count)
       targets = wp.array(target_poses, dtype=wp.float32)  # (n, 7)
       links = wp.full(n, robot.link_names.index("fr3_hand"), dtype=wp.int32)
       instance = wp.array(np.arange(n, dtype=np.int32))    # factor i acts on instance i

   states = pycunls.ArticulationStateBatch(topo, n)
   states.set_num_active_states(n)
   states.set_joint_q(q0, n, stream)

   pose = pycunls.LinkPoseFactorBatch(topo, links, None, targets, None, n)
   pose.set_num_active_factors(n)

   problem = pycunls.Problem()
   problem.add_state_batch(states)
   problem.add_factor_batch(pose, [states], instance)
   problem.set_problem_partition(n, [instance])             # each problem converges and stops alone

   solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer())
   summary = solver.minimize(stream, problem)
   with wp.ScopedStream(wp_stream):
       q = wp.zeros((n, topo.coord_count), dtype=wp.float32)
   states.get_joint_q(q, n, stream)   # asynchronous: read q on wp_stream (q.numpy() inside the scope)

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
  kernel) on the cuNLS stream (``wp.ScopedStream`` as above; work on another stream is not
  ordered with the solve) and call ``minimize`` again. Repeated solves skip the structure
  setup with ``reuse_structure`` set on the augmented Lagrangian's own options (it passes its
  flag to the inner solves):

  .. code-block:: python

     al = pycunls.AugmentedLagrangianMinimizerOptions()
     al.reuse_structure = True
     solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(), al)
* **Several seeds per problem.** Make ``problems x seeds`` instances (seeds next to each other),
  give each instance its own subproblem, solve once, then compare the link-pose errors of each
  problem's seeds and keep the best one.
* **Frozen DOFs.** A box of zero width, ``lower == upper == q``, holds a scalar DOF at ``q``.
  ``set_bounds(lower, upper)`` replaces all bounds, so start from the joint limits and change only
  the frozen entries:

  .. code-block:: python

     lower = np.tile(topo.lower_limits, (n, 1)).astype(np.float32)   # (n, topo.dof_count)
     upper = np.tile(topo.upper_limits, (n, 1)).astype(np.float32)
     lower[:, dof] = upper[:, dof] = q_frozen
     with wp.ScopedStream(wp_stream):
         states.set_bounds(wp.array(lower), wp.array(upper))
* **Objectives are factors.** Position, orientation and pose targets (with per-row weights),
  posture regularization, loop-closing joints and mimic joints are factors; constraints are
  factors wrapped in ``ConstraintFactorBatch``. Several targets per instance are several factors
  (or several rows of one batch) connected to the same instance.

===============================================================================
Custom kinematic objectives
===============================================================================

An objective the built-in factors do not cover (look at a point, stay on a plane, keep clear of
an obstacle, match an object's pose) is written as an **ordinary factor over SE(3) poses**, which
knows nothing about robots. ``LinkFrameFactorBatch`` puts it on the robot.

-------------------------------------------------------------------------------
How it works
-------------------------------------------------------------------------------

The wrapped factor (the *inner* factor) has state slots, e.g. ``state_sizes=[6]`` for one SE(3)
pose. Normally each slot connects to a state of the problem. ``pose_slots`` lists the slots that
do **not**: each receives, at every evaluation, the world pose of a frame on a link instead.

.. code-block:: text

   inner factor slots          [0: SE3 pose]                [0: SE3 pose, 1: SE3 pose]
   pose_slots                  [0]                          [0]
   LinkFrameFactorBatch slots  [0: articulation instance]   [0: articulation instance,
                                                             1: SE3 pose (a problem state)]

For factor *f* and pose slot *s* (the position of the slot in ``pose_slots``; nP =
``len(pose_slots)``):

#. ``L = topology_links[f * nP + s]``: the topology link that carries the frame (a **link**
   index, e.g. ``robot.link_names.index("wrist")``).
#. ``X_off = frame_offsets[f * nP + s]``: where the frame sits on that link, in the link frame
   (7 floats: position, quaternion x, y, z, w; ``None``: the link frame itself).
#. The inner factor receives :math:`F_s = X_L(q)\,X_{off}` as a row-major 4x4 matrix and
   evaluates as if it were an ``SE3StateBatch`` state, returning its residual and its 6
   Jacobian columns for that slot (body tangent :math:`[\omega, v]`).
#. The wrapper turns those columns into joint columns: DOF *j* moves the frame by the body
   twist :math:`b_j = (R^T\omega_j,\; R^T(\omega_j \times p + v_j))` (with :math:`F_s = (R,
   p)` and :math:`(\omega_j, v_j)` the DOF's world motion column), so
   :math:`\partial r / \partial q_j = \sum_s (\partial r / \partial \delta_s)\, b_j` over the
   pose slots whose link DOF *j* moves.

The residual size and active count are the inner factor's (``set_num_active_factors`` on the
wrapper sets the inner one).

-------------------------------------------------------------------------------
Writing the inner factor
-------------------------------------------------------------------------------

A ``pycunls.warp.WarpFactorBatch`` with ``state_sizes=[6]`` per pose (see
:doc:`custom_factors_and_states`). In the kernel, item *t*'s pose slot is a pointer to 16 floats:

.. code-block:: python

   X = wp.array(ptr=state_pointers[t], shape=(16,), dtype=float)   # row-major 4x4
   R = wp.mat33(X[0], X[1], X[2], X[4], X[5], X[6], X[8], X[9], X[10])
   p = wp.vec3(X[3], X[7], X[11])

Jacobian columns are with respect to the right perturbation :math:`X\,Exp([\omega, v])`, under
which, to first order, :math:`p \to p + R\,v` and :math:`R \to R\,(I + [\omega]_\times)`. Two
useful consequences: a world point fixed in the frame, :math:`x = R c + p`, has
:math:`\partial x / \partial \omega = -R [c]_\times` and :math:`\partial x / \partial v = R`; a
world point *t* seen in the frame, :math:`d = R^T (t - p)`, has
:math:`\partial d / \partial \omega = [d]_\times` and :math:`\partial d / \partial v = -I`.
Check the result with ``pycunls.check_jacobians(problem)`` before solving.

-------------------------------------------------------------------------------
Recipes
-------------------------------------------------------------------------------

* **A frame target or relation, built in**: ``SE3PriorFactorBatch`` (a frame should reach a
  pose), ``SE3BetweenFactorBatch`` with ``pose_slots=[0, 1]`` (the relative pose of two links of
  one instance; ``topology_links`` then holds 2 links per factor),
  ``SE3SphereClearanceFactorBatch`` (keep a frame's origin off a sphere; wrap the
  ``LinkFrameFactorBatch`` in ``ConstraintFactorBatch(..., ConstraintKind.Inequality)``).
* **A link and an object pose solved together** (grasping, hand-eye): an
  ``SE3BetweenFactorBatch`` with ``pose_slots=[0]`` keeps slot 1 as a state, so connect
  ``[articulation_states, object_poses]``; the object is an ``SE3StateBatch``.
* **Constraints**: always wrap the ``LinkFrameFactorBatch`` (``ConstraintFactorBatch`` around
  it), never its inner factor: a constraint inside would be hidden from the augmented Lagrangian,
  and the constructor rejects it.
* **Joint-space objectives** (soft limits, joint velocity between instances) need no wrapper:
  a custom factor on the articulation state reads DOF *d* of a scalar joint at float
  ``topology.scalar_coord[d]`` of its instance's storage.

``python/examples/robot_custom_objective.py`` puts it together: a 7-DOF arm aims a wrist camera
at 512 points with a Warp look-at factor while ``SE3SphereClearanceFactorBatch`` keeps the elbow
off an obstacle, under hard joint limits. An evaluation runs three steps (gather the frames, the
inner factor, the chain rule), so a built-in kinematic factor stays the faster choice for what it
covers.

===============================================================================
Closed loops and mimic joints
===============================================================================

A loop-closing joint keeps two joint frames together in the directions its type does not allow
(BALL 3 rows, REVOLUTE 6, PRISMATIC 5, FIXED 6). Wrap it as an equality constraint:

.. code-block:: python

   loop = pycunls.LoopClosureFactorBatch(topo, pycunls.ArticulationJointType.Revolute,
                                         parent_links, child_links, frames, axes, n)
   loop.set_num_active_factors(n)                   # batches start with 0 active factors
   problem.add_factor_batch(pycunls.ConstraintFactorBatch(loop, pycunls.ConstraintKind.Equality),
                            [states], instance)   # one loop factor per instance

``frames`` holds ``joint_X_p`` and ``joint_X_c`` of the loop-closing joint (14 floats per factor).
Rows may be redundant (a planar four-bar has 6 rows of rank 2); the augmented Lagrangian
tolerates that. To assemble a mechanism at given actuated angles, add a ``PostureFactorBatch``
with a high weight on the actuated DOFs and a low one on the passive DOFs: the result is the
nearest valid configuration.

A mimic joint ``q_m = a + b q_ref`` (URDF ``<mimic>``) is a
``MimicJointFactorBatch`` wrapped the same way; ``topology_from_urdf`` returns the couplings in
``mimic_joints``.

===============================================================================
Performance notes
===============================================================================

* The state's ``Plus`` runs forward kinematics once per update and stores link poses and motion
  columns; factors read them, so evaluating a residual does not traverse the tree. Jacobians
  still visit every DOF: their cost grows with the number of movable joints (dense rows).
* One subproblem per IK problem (``set_problem_partition``): each one converges and stops on its
  own.
* The Jacobian of a factor is dense over the articulation's DOFs; the Hessian assembler stages
  each factor's block in shared memory, which bounds an articulation at roughly 140 DOFs for
  6-row factors and roughly 85 DOFs with ``PostureFactorBatch`` (one row per DOF).
* After writing coordinates through ``state_device_ptr`` directly, call ``refresh(stream)``
  before ``pycunls.evaluate``; ``set_joint_q`` and the minimizers refresh on their own.
