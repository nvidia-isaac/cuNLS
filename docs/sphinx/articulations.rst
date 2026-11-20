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
  ``LinkPoseFactorBatch``, ``PostureFactorBatch`` and ``MimicJointFactorBatch``. Every Jacobian
  is analytic.
* Closed loops: ``pycunls.articulation.loop_closure`` builds a loop-closing joint as equality
  constraints from existing factors (see `Closed loops and mimic joints`_).
* ``LinkFrameFactorBatch``: existing factors over SE(3) poses, 3D points or SO(3) rotations
  (built-in or a Warp kernel) on frames of links, for objectives of your own (see
  `Custom kinematic objectives`_).

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
     - ``links`` (link factors), ``topology_links`` (``LinkFrameFactorBatch``; -1: the world),
       the ``links`` of ``pycunls.articulation.loop_closure``, ``link_pose_offset``,
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
     - ``joint_X_p`` / ``joint_X_c``, ``root_pose``, ``LinkPoseFactorBatch`` targets and
       offsets, ``frame_offsets``, ``get_link_transforms`` (one ``wp.transform`` per link)
   * - position
     - 3: ``px, py, pz``
     - ``LinkPositionFactorBatch`` points (in the link frame) and targets (in the world)
   * - quaternion
     - 4: ``qx, qy, qz, qw``
     - ``LinkOrientationFactorBatch`` offsets and targets
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
add each one as constraints (``pycunls.articulation.loop_closure``).

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
   * - ``SE3StateBatch`` pose, ``FrameView.Pose`` slot
     - right: :math:`X \leftarrow X\,Exp(\delta)`, :math:`\delta = [\omega, v]`
     - body frame of the pose
   * - ``VectorStateBatch3`` point, ``FrameView.Position`` slot
     - :math:`p \leftarrow p + \delta`
     - world
   * - ``SO3StateBatch`` rotation, ``FrameView.Orientation`` slot
     - right: :math:`R \leftarrow R\,Exp(\delta)`
     - body frame of the rotation
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
* Reading the cache as 4x4: link poses are 12 floats (3x4); only ``FrameView.Pose`` slots of
  ``LinkFrameFactorBatch`` are 4x4.
* A custom SE(3) or SO(3) factor with a world-frame or left Jacobian: ``Pose`` and
  ``Orientation`` slots use the right (body) perturbation, like ``SE3StateBatch`` and
  ``SO3StateBatch``. ``pycunls.check_jacobians`` catches it.
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
   # Solve on Warp's stream: uploads, target updates, solves and reads stay in order.
   stream = wp.get_stream()
   q0 = wp.array(q_initial, dtype=wp.float32)        # (n, topo.coord_count)
   targets = wp.array(target_poses, dtype=wp.transform)  # (n,): position, quaternion x, y, z, w
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

   solver = pycunls.AugmentedLagrangianMinimizer()
   summary = solver.minimize(stream, problem)
   q = wp.zeros((n, topo.coord_count), dtype=wp.float32)
   states.get_joint_q(q, n, stream)   # asynchronous, on the same stream as q.numpy()

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

* **A topology from a scene's joint arrays.** A simulator stores the joints of all its robots
  and worlds in flat arrays with scene indices (bodies, ``joint_q`` / ``joint_qd`` offsets,
  per-DOF axes and limits). ``pycunls.articulation.topology_from_joint_arrays(...,
  joints=range(first, last))`` selects one articulation's joints, remaps them to local indices
  and returns the topology with the maps between the two: ``link_of_body`` (scene body ->
  link, for the ``links`` of factors), ``bodies`` (link -> scene body), ``coord_index`` /
  ``dof_index`` (local -> scene), ``coord_start`` (where the articulation's coordinates start
  in the scene ``joint_q`` when they are contiguous) and ``loop_joints`` (joints whose child
  already has a tree joint: add them with ``pycunls.articulation.loop_closure``). The arrays
  may be GPU arrays (copied once to the host).
* **Scene arrays in place.** ``set_joint_q`` / ``get_joint_q`` take a ``stride`` (floats between
  instances), so instances embedded in a larger array are read and written without a gather:
  with one articulation per replicated world of a scene-wide ``joint_q``,
  ``states.set_joint_q(scene_q[art.coord_start:], worlds, stream, stride=coords_per_world)``.
* **Forward kinematics and link poses.** ``set_joint_q`` runs forward kinematics;
  ``get_link_transforms(out, n, stream, stride=0)`` writes every link's world pose (7 floats:
  position, quaternion x, y, z, w; the convention of the factor targets) of every instance, with
  a stride for writing into a scene-wide pose array. Use it to check a solution (position and
  angle errors) or to pick the best of several seeds.
* **Targets change, the problem does not.** Factors read their device arrays (targets, weights,
  links) at every evaluation: write the next targets into the same arrays (``wp.copy``, a
  kernel) on the stream the solve runs on (with ``stream = wp.get_stream()`` as above, any Warp
  work is; work on another stream is not ordered with the solve) and call ``minimize`` again. Repeated solves skip the structure
  setup with ``base_options.reuse_structure`` of the augmented Lagrangian's options:

  .. code-block:: python

     options = pycunls.AugmentedLagrangianMinimizerOptions()
     options.base_options.reuse_structure = True
     solver = pycunls.AugmentedLagrangianMinimizer(options)
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
     states.set_bounds(wp.array(lower), wp.array(upper))
* **Objectives are factors.** Position, orientation and pose targets (with per-row weights),
  posture regularization, loop-closing joints and mimic joints are factors; constraints are
  factors wrapped in ``ConstraintFactorBatch``. Several targets per instance are several factors
  (or several rows of one batch) connected to the same instance.

===============================================================================
Custom kinematic objectives
===============================================================================

An objective the built-in factors do not cover (look at a point, stay above a table, keep clear
of an obstacle, match an object's pose) is written as an **ordinary factor** over SE(3) poses,
3D points or SO(3) rotations, which knows nothing about robots. ``LinkFrameFactorBatch`` puts it
on the robot, so the factors cuNLS already has work on robot links.

-------------------------------------------------------------------------------
How it works
-------------------------------------------------------------------------------

The wrapped factor (the *inner* factor) has state slots, e.g. ``state_sizes=[6]`` for one SE(3)
pose. Normally each slot connects to a state of the problem. ``frame_slots`` lists the slots that
do **not**: each receives, at every evaluation, a **view** of a frame on a link instead, in the
layout and convention of an existing state type.

.. list-table::
   :header-rows: 1

   * - View
     - The slot receives
     - As the state type
     - Slot tangent
   * - ``FrameView.Pose``
     - the frame's pose, row-major 4x4 ``world_from_frame``
     - ``SE3StateBatch`` (body :math:`[\omega, v]`, :math:`X' = X\,Exp(\delta)`)
     - 6
   * - ``FrameView.Position``
     - the frame's origin in the world (3 floats)
     - ``VectorStateBatch3`` (additive world step)
     - 3
   * - ``FrameView.Orientation``
     - the frame's rotation, row-major 3x3 ``world_from_frame``
     - ``SO3StateBatch`` (body :math:`\omega`, :math:`R' = R\,Exp(\omega)`)
     - 3

The view is named explicitly because the tangent size alone is ambiguous (6: a pose or a
6-vector; 3: a rotation or a point).

.. code-block:: text

   inner factor slots          [0: pose (6)]                [0: pose (6), 1: pose (6)]
   frame_slots                 [(0, Pose)]                  [(0, Pose)]
   LinkFrameFactorBatch slots  [0: articulation instance]   [0: articulation instance,
                                                             1: SE3 pose (a problem state)]

For factor *f* and frame slot *s* (the position of the slot in ``frame_slots``; nF =
``len(frame_slots)``):

#. ``L = topology_links[f * nF + s]``: the topology link that carries the frame (a **link**
   index, e.g. ``robot.link_names.index("wrist")``).
#. ``X_off = frame_offsets[f * nF + s]``: where the frame sits on that link, in the link frame
   (7 floats: position, quaternion x, y, z, w; ``None``: the link frame itself).
#. The frame is :math:`F_s = X_L(q)\,X_{off} = (R, p)`; the inner factor receives its view and
   evaluates as if the slot were a state of the view's type, returning its residual and its
   Jacobian columns for that slot (6 for a pose, 3 otherwise).
#. The wrapper turns those columns into joint columns. Per unit of DOF *j* (world motion column
   :math:`(\omega_j, v_j)`), a pose moves by the body twist :math:`(R^T\omega_j,\;
   R^T(\omega_j \times p + v_j))`, a position by :math:`\omega_j \times p + v_j`, an orientation
   by :math:`R^T\omega_j`; :math:`\partial r / \partial q_j` sums these over the frame slots
   whose link DOF *j* moves.

The residual size and active count are the inner factor's (``set_num_active_factors`` on the
wrapper sets the inner one).

.. code-block:: python

   # The tool tip stays above a table at z = h: a halfspace a^T p <= b on the tip's position.
   plane = pycunls.HalfspaceFactorBatch3(normals, offsets, n)        # a = (0, 0, -1), b = -h
   tip = pycunls.LinkFrameFactorBatch(topo, plane, frame_slots=[(0, pycunls.FrameView.Position)],
                                      topology_links=tool_links, frame_offsets=tip_offsets)
   tip.set_num_active_factors(n)
   problem.add_factor_batch(pycunls.ConstraintFactorBatch(tip, pycunls.ConstraintKind.Inequality),
                            [states], instance)

-------------------------------------------------------------------------------
Writing the inner factor
-------------------------------------------------------------------------------

Two ways, both in ``pycunls.warp``:

* **A residual function only** (``WarpFunctionFactorBatch``, see
  :ref:`py-warp-function-factor-batch`): a Warp function of the frames (a ``wp.transform`` for a
  ``Pose`` view, ``wp.vec3`` for ``Position``, ``wp.quat`` for ``Orientation``) and of one
  per-factor data item; Warp's automatic differentiation provides the Jacobian (checked once
  against central differences):

  .. code-block:: python

     @wp.func
     def look_at(X: wp.transform, target: wp.vec3) -> wp.vec4:
         d = wp.quat_rotate_inv(wp.transform_get_rotation(X), target - wp.transform_get_translation(X))
         u = wp.normalize(d)
         return wp.vec4(u[0], u[1], u[2] - 1.0, wp.length(d) - 0.35)   # look along +z from 0.35 m

     inner = pycunls.warp.WarpFunctionFactorBatch(look_at, data=wp.array(points, dtype=wp.vec3),
                                                  capacity=n)
     camera = pycunls.LinkFrameFactorBatch(topo, inner, frame_slots=[(0, pycunls.FrameView.Pose)],
                                           topology_links=wrist_links, frame_offsets=mounts)

* **An analytic Jacobian** (a ``WarpFactorBatch`` whose ``state_sizes`` match the views: 6 for a
  pose, 3 for a position or orientation; see :doc:`custom_factors_and_states`). The device
  functions ``pose_at``, ``point_at`` and ``rotation_at`` read one entry of the item-major
  pointer table in its view's layout (a ``wp.transform``, a ``wp.vec3``, a ``wp.mat33``): state
  *k* of item *t* is entry ``t * K + k`` for a factor with *K* states. ``jacobian_index`` gives
  the position of a Jacobian entry:

  .. code-block:: python

     X = pycunls.warp.pose_at(state_pointers, t * K + 0)  # state 0 of item t, a wp.transform
     jacobians[pycunls.warp.jacobian_index(t, row, column, num_rows, num_columns)] = value

  For a pose, Jacobian columns are with respect to the right perturbation
  :math:`X\,Exp([\omega, v])`, under which, to first order, :math:`p \to p + R\,v` and
  :math:`R \to R\,(I + [\omega]_\times)`. Two useful consequences: a world point fixed in the
  frame, :math:`x = R c + p`, has :math:`\partial x / \partial \omega = -R [c]_\times` and
  :math:`\partial x / \partial v = R`; a world point *t* seen in the frame,
  :math:`d = R^T (t - p)`, has :math:`\partial d / \partial \omega = [d]_\times` and
  :math:`\partial d / \partial v = -I`.

The analytic form is faster (one kernel, while the reverse pass of the residual-function form
runs the function once per residual row); the residual-function form is quicker to write and a
reference for the analytic one. When only the
origin or only the rotation matters, a position or orientation view keeps the factor smaller.
Check either with ``pycunls.check_jacobians(problem)`` before solving.

-------------------------------------------------------------------------------
Recipes
-------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Objective
     - Inner factor and views
   * - a point on a link reaches a target
     - ``PriorVectorFactorBatch3``, ``[(0, Position)]`` (the same as ``LinkPositionFactorBatch``,
       which is faster)
   * - stay above a table / on one side of a wall
     - ``HalfspaceFactorBatch3``, ``[(0, Position)]``, as an inequality
   * - stay inside a workspace box
     - ``HalfspaceFactorBatch3`` with six factors per instance (normals :math:`\pm x, \pm y,
       \pm z`), ``[(0, Position)]``, as an inequality (``BoundFactorBatch3`` is itself a
       constraint and cannot be an inner factor)
   * - a frame reaches a target orientation
     - ``SO3PriorFactorBatch``, ``[(0, Orientation)]`` (body-frame residual)
   * - relative orientation of two links
     - ``SO3BetweenFactorBatch``, ``[(0, Orientation), (1, Orientation)]``
   * - a frame reaches a target pose
     - ``SE3PriorFactorBatch``, ``[(0, Pose)]`` (coupled SE(3) log residual;
       ``LinkPoseFactorBatch`` is the faster, decoupled one)
   * - relative pose of two links
     - ``SE3BetweenFactorBatch``, ``[(0, Pose), (1, Pose)]`` (``topology_links``: 2 per factor)
   * - keep a point off a sphere
     - ``SE3SphereClearanceFactorBatch``, ``[(0, Pose)]``, as an inequality
   * - a link and an object pose solved together (grasping, hand-eye)
     - ``SE3BetweenFactorBatch``, ``[(0, Pose)]``: slot 1 stays a state, so connect
       ``[articulation_states, object_poses]`` (an ``SE3StateBatch``)

* **Constraints**: always wrap the ``LinkFrameFactorBatch`` (``ConstraintFactorBatch`` around
  it), never its inner factor: a constraint inside would be hidden from the augmented Lagrangian,
  and the constructor rejects it.
* **Joint-space objectives** (soft limits, preferred ranges) need no wrapper: a custom factor on
  the articulation state reads DOF *d* of a scalar joint at float ``topology.scalar_coord[d]`` of
  its instance's storage. Hard joint boxes are the state's bounds (``set_bounds``).
* **Speed**: an evaluation runs three steps (copy the views, the inner factor, the chain rule).
  For batched IK (7-DOF arm, 32k problems) a position plus an orientation view cost about 1.26x
  the fused ``LinkPoseFactorBatch``; use the built-in kinematic factors for what they cover.

``python/examples/robot_custom_objective.py`` puts it together: a 7-DOF arm aims a wrist camera
at 512 points with a Warp look-at factor (pose view), keeps the elbow off an obstacle
(``SE3SphereClearanceFactorBatch``, pose view) and the camera above a table
(``HalfspaceFactorBatch3``, position view), under hard joint limits.

===============================================================================
Closed loops and mimic joints
===============================================================================

A loop-closing joint keeps two joint frames together in the directions its type does not allow:
the parent frame :math:`X_P X_p` (:math:`X_P` the parent link's world pose, the identity for the
world) and the child frame :math:`X_C X_c`. It is not a factor class of its own but equality
constraints built from existing factors on ``LinkFrameFactorBatch`` views of the two frames
(slot 0 the parent frame, slot 1 the child frame; see `Custom kinematic objectives`_):

.. list-table::
   :header-rows: 1

   * - Joint
     - Rows
   * - BALL
     - origins coincide: ``VectorBetweenFactorBatch3`` on two ``Position`` views (3)
   * - REVOLUTE
     - origins coincide (3), and ``SE3AxisAlignmentFactorBatch`` on two ``Pose`` views: the
       joint axes point the same way (3)
   * - PRISMATIC
     - rotations equal: ``SO3BetweenFactorBatch`` on two ``Orientation`` views (3), and
       ``SE3AxisOffsetFactorBatch`` on two ``Pose`` views: the child frame's origin stays on the
       axis (2)
   * - FIXED
     - poses equal: ``SE3BetweenFactorBatch`` on two ``Pose`` views (6)

``pycunls.articulation.loop_closure`` builds them:

.. code-block:: python

   from pycunls.articulation import loop_closure

   # links: (parent link or -1 for the world, child link) per loop; frames: joint_X_p then
   # joint_X_c per loop (14 floats); axes: the joint axis in the joint frame (3 floats).
   loop = loop_closure(topo, pycunls.ArticulationJointType.Revolute, links, frames, axes)
   loop.set_num_active_factors(n)                   # batches start with 0 active factors
   for constraint in loop.constraints:
       problem.add_factor_batch(constraint, [states], instance)   # one loop per instance

The arrays are used in place (``frames`` is exactly the views' ``frame_offsets``), and
``loop.factors`` are the unconstrained rows, for ``pycunls.check_jacobians``. Rows may be
redundant (a planar four-bar's revolute loop has 6 rows of rank 2); the augmented Lagrangian
tolerates that, and constraint rows only define where the loop is closed, so their
parameterization does not change the solution. To assemble a mechanism at given actuated
angles, add a ``PostureFactorBatch`` with a high weight on the actuated DOFs and a low one on
the passive DOFs: when the solve converges, the result is a nearby closed configuration. The
posture is a soft preference, so the actuated angles may move if the loop cannot close at them;
to hold them exactly, give the actuated DOFs zero-width bounds (``lower == upper``) and check
that the solve converged (a loop that cannot close at those angles then fails the
constraints).

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
