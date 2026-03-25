###############################################################################
Python Tutorial
###############################################################################

===============================================================================
Overview
===============================================================================

This tutorial walks through four complete pycunls examples, each demonstrating
a different optimization pattern. Every example follows the same high-level
flow described in the :doc:`introduction`:

1. Allocate state data on the GPU (CuPy arrays).
2. Wrap the GPU memory in one or more
   :ref:`state batch <py-state-batch-interface>` objects.
3. Build one or more
   :ref:`factor batch <py-factor-batch-interface>` objects from observations.
4. Add state batches and factor batches to a
   :ref:`Problem <py-problem-label>`.
5. Run a :ref:`minimizer <py-lm-label>` and inspect
   :ref:`MinimizerSummary <py-minimizer-summary-label>`.

The examples increase in complexity:

- :ref:`pycunls_tutorial:Sparse Bundle Adjustment` — uses
  :ref:`ReprojectionFactorBatch <py-reprojection-factor>` to jointly
  optimize camera poses and 3D landmarks from multi-view observations.
- :ref:`pycunls_tutorial:Pose Graph Optimization` — uses
  :ref:`SE3BetweenFactorBatch <py-se3-between-factor>` to recover a chain
  of SE(3) poses from consecutive relative-transform measurements.
- :ref:`pycunls_tutorial:Custom Warp Factor` — shows how to implement a
  user-defined factor kernel using `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_ and
  :ref:`WarpFactorBatch <py-warp-factor-batch>`.
- :ref:`pycunls_tutorial:Custom Warp State` — shows how to implement a
  custom manifold retraction using `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_ and
  :ref:`WarpStateBatch <py-warp-state-batch>`.

Full source code for all examples lives in the ``python/examples/``
directory.

===============================================================================
Sparse Bundle Adjustment
===============================================================================

- **Source**: python/examples/sparse_bundle_adjustment.py

SBA problem statement
---------------------

This is a Python port of the C++ bundle adjustment example (see
:ref:`tutorial:Sparse Bundle Adjustment` for the full mathematical
formulation). Given :math:`M` cameras with poses
:math:`T_1, \ldots, T_M \in \mathrm{SE}(3)` and :math:`N` 3D landmarks
:math:`\mathbf{p}_1, \ldots, \mathbf{p}_N \in \mathbb{R}^3`, we minimize
the sum of squared reprojection errors across all :math:`K` observations:

.. math::

   \min_{T_1,\ldots,T_M,\; \mathbf{p}_1,\ldots,\mathbf{p}_N}
     \frac{1}{2} \sum_{k=1}^{K}
       \left\| \pi(T_{i_k}, \mathbf{p}_{j_k}) - \mathbf{z}_k \right\|^2

Pose :math:`T_0` is held constant as a gauge anchor.

SBA API used
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - :ref:`SE3StateBatch <py-lie-state-batches>`
     - Stores camera poses on the SE(3) manifold. The first pose is marked
       constant via ``const_ids``; the rest are optimized.
   * - :ref:`VectorStateBatch3 <py-vector-state-batches>`
     - Stores 3D landmark coordinates in :math:`\mathbb{R}^3`. All points
       are optimization variables.
   * - :ref:`ReprojectionFactorBatch <py-reprojection-factor>`
     - Computes normalized reprojection residuals and Jacobians for each
       (pose, point) pair.
   * - :ref:`Problem <py-problem-label>`
     - Assembles the factor graph.
   * - :ref:`LevenbergMarquardtMinimizer <py-lm-label>`
     - Iteratively solves the nonlinear least-squares problem with adaptive
       damping.

SBA code walkthrough
~~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate synthetic data.**
Random camera poses and 3D points are created on the host using NumPy.
Each ground-truth point is filtered so that it has positive depth in all
cameras. Poses :math:`T_1, \ldots, T_{M-1}` and all points are perturbed
to create noisy initial estimates; :math:`T_0` stays at ground truth.

.. code-block:: python

   import numpy as np
   import cupy as cp
   import pycunls
   from se3_utils import twist_to_se3, compose_se3, project_normalized

   num_poses  = 6
   num_points = 800
   num_observations = num_poses * num_points
   rng = np.random.default_rng(1234)

   # Ground-truth SE(3) camera poses from random twists.
   gt_poses = [twist_to_se3(twist) for twist in ...]

   # 3D points visible from all cameras (positive depth).
   gt_points = ...  # (num_points, 3) float32

   # Perturbed initial estimates.
   initial_points = gt_points + rng.uniform(-0.35, 0.35, gt_points.shape)
   initial_poses  = [...]  # T_0 at ground truth, T_1..T_{M-1} perturbed

**Step 2 — Create 2D observations.**
Each camera observes every 3D point. Observations are in normalized image
coordinates — the format
:ref:`ReprojectionFactorBatch <py-reprojection-factor>` expects.

.. code-block:: python

   observations = np.empty((num_observations, 2), dtype=np.float32)
   for pi in range(num_poses):
       for qi in range(num_points):
           observations[pi * num_points + qi] = project_normalized(
               gt_poses[pi], gt_points[qi])

**Step 3 — Upload data to the GPU and build state batches.**
CuPy arrays serve as GPU storage.
:ref:`SE3StateBatch <py-lie-state-batches>` wraps the pose memory with only
:math:`T_0` marked constant;
:ref:`VectorStateBatch3 <py-vector-state-batches>` wraps the landmark
memory.

.. code-block:: python

   poses_gpu  = cp.asarray(initial_poses_flat, dtype=cp.float32)
   points_gpu = cp.asarray(initial_points.reshape(-1), dtype=cp.float32)
   obs_gpu    = cp.asarray(observations.reshape(-1), dtype=cp.float32)
   const_ids  = cp.array([0], dtype=cp.int32)

   cublas = pycunls.CublasHandle()
   pose_states  = pycunls.SE3StateBatch(cublas, poses_gpu, num_poses, const_ids, 1)
   point_states = pycunls.VectorStateBatch3(points_gpu, num_points)

**Step 4 — Build the reprojection factor batch.**

.. code-block:: python

   reproj_factor = pycunls.ReprojectionFactorBatch(
       cublas, obs_gpu, num_observations, z_threshold=1e-3)

**Step 5 — Create the state-pointer list and assemble the problem.**
The state-pointer list is a flat sequence of device pointers, two per
factor: ``[pose_ptr, point_ptr]``. This tells the solver which state
blocks each factor reads.

.. code-block:: python

   state_pointers = []
   for pi in range(num_poses):
       for qi in range(num_points):
           state_pointers.append(pose_states.state_block_device_ptr(pi))
           state_pointers.append(point_states.state_block_device_ptr(qi))

   problem = pycunls.Problem()
   problem.add_state_batch(pose_states)
   problem.add_state_batch(point_states)
   problem.add_factor_batch(reproj_factor, state_pointers)
   assert problem.check_consistency()

**Step 6 — Configure and run** :ref:`Levenberg-Marquardt <py-lm-label>` **.**

.. code-block:: python

   opts = pycunls.MinimizerOptions()
   opts.max_num_iterations = 80
   opts.state_tolerance = 1e-8
   opts.cost_tolerance  = 1e-8

   lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
   lm_opts.base_options  = opts
   lm_opts.initial_lambda = 1e-3

   stream    = pycunls.CudaStream()
   minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
   summary   = minimizer.minimize(stream, problem)

   cp.cuda.runtime.streamSynchronize(stream.get_stream())

**Step 7 — Read back results.**
After optimization, the CuPy arrays contain the updated state. Copy them
to the host with ``cp.asnumpy`` and compare against ground truth.

.. code-block:: python

   optimized_points = cp.asnumpy(points_gpu).reshape(-1, 3)
   point_mse = float(np.mean((optimized_points - gt_points) ** 2))

   print(f"Initial cost: {summary.initial_cost:.6f}")
   print(f"Final cost:   {summary.final_cost:.6f}")
   print(f"Iterations:   {summary.num_iterations}")
   print(f"Point MSE:    {point_mse:.6f}")

===============================================================================
Pose Graph Optimization
===============================================================================

- **Source**: python/examples/pose_graph_optimization.py

PGO problem statement
---------------------

This is a Python port of the C++ pose graph example (see
:ref:`tutorial:Pose Graph Optimization` for the full mathematical
formulation). A chain of :math:`N` SE(3) poses is connected by
:math:`N{-}1` between constraints. The residual for constraint :math:`i`
is:

.. math::

   r_i = \mathrm{Log}\!\left(
     \Delta_i \, T_i^{-1} \, T_{i+1}
   \right) \in \mathbb{R}^6

Pose :math:`T_0` is held constant as a gauge anchor.

PGO API used
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - :ref:`SE3StateBatch <py-lie-state-batches>`
     - A single instance stores the full pose chain. The first pose is
       marked constant; the rest are optimized.
   * - :ref:`SE3BetweenFactorBatch <py-se3-between-factor>`
     - Computes the relative-transform residual and its Jacobians w.r.t.
       both pose blocks.
   * - :ref:`Problem <py-problem-label>`
     - Assembles the factor graph.
   * - :ref:`LevenbergMarquardtMinimizer <py-lm-label>`
     - Solves the nonlinear system.

PGO code walkthrough
~~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate synthetic pose chain and measurements.**
A random anchor and random relative transforms are generated on the host.
Ground-truth poses are built by chaining the transforms, then all poses
except the anchor are perturbed.

.. code-block:: python

   num_poses = 201
   num_constraints = num_poses - 1
   rng = np.random.default_rng(9012)

   anchor = random_se3()
   deltas = [random_se3() for _ in range(num_constraints)]

   gt_poses = [anchor]
   for i in range(num_constraints):
       gt_poses.append(compose_se3(gt_poses[-1], se3_inverse(deltas[i])))

   initial_poses = [gt_poses[0].copy()]
   for i in range(1, num_poses):
       perturbation = random_se3(rot_scale=0.05, trans_scale=0.3)
       initial_poses.append(compose_se3(perturbation, gt_poses[i]))

**Step 2 — Upload data and build the state batch.**

.. code-block:: python

   poses_gpu  = cp.asarray(np.stack(initial_poses).reshape(-1), dtype=cp.float32)
   deltas_gpu = cp.asarray(np.stack(deltas).reshape(-1), dtype=cp.float32)
   const_ids  = cp.array([0], dtype=cp.int32)

   cublas = pycunls.CublasHandle()
   pose_states = pycunls.SE3StateBatch(
       cublas, poses_gpu, num_poses, const_ids, 1)

**Step 3 — Build the between-factor batch.**

.. code-block:: python

   between_factor = pycunls.SE3BetweenFactorBatch(
       cublas, deltas_gpu, num_constraints)

**Step 4 — Wire state pointers and assemble the problem.**
Each between factor reads two state blocks: ``[T_i, T_{i+1}]``.

.. code-block:: python

   state_pointers = []
   for i in range(num_constraints):
       state_pointers.append(pose_states.state_block_device_ptr(i))
       state_pointers.append(pose_states.state_block_device_ptr(i + 1))

   problem = pycunls.Problem()
   problem.add_state_batch(pose_states)
   problem.add_factor_batch(between_factor, state_pointers)
   assert problem.check_consistency()

**Step 5 — Solve with Levenberg-Marquardt.**

.. code-block:: python

   opts = pycunls.MinimizerOptions()
   opts.max_num_iterations = 60
   opts.state_tolerance = 1e-8
   opts.cost_tolerance  = 1e-8

   lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
   lm_opts.base_options  = opts
   lm_opts.initial_lambda = 1e-3

   stream    = pycunls.CudaStream()
   minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
   summary   = minimizer.minimize(stream, problem)

   cp.cuda.runtime.streamSynchronize(stream.get_stream())

   print(f"Initial cost: {summary.initial_cost:.6f}")
   print(f"Final cost:   {summary.final_cost:.6f}")
   print(f"Iterations:   {summary.num_iterations}")

===============================================================================
Custom Warp Factor
===============================================================================

- **Source**: python/examples/custom_warp_factor.py

Custom Warp factor problem statement
-------------------------------------

This is a Python port of the C++ custom factor example (see
:ref:`tutorial:Custom Factor`). It builds a chain of scalar states
connected by difference constraints implemented as an `NVIDIA Warp
<https://developer.nvidia.com/warp-python>`_ kernel
through :ref:`WarpFactorBatch <py-warp-factor-batch>` (from
``pycunls.warp``):

.. math::

   r_i = (x_{i+1} - x_i) - m_i, \qquad i = 0, \ldots, N{-}2

A prior factor anchors :math:`x_0` to its observed value:

.. math::

   r_{\mathrm{prior}} = x_0 - x_0^{\mathrm{obs}}

Warp factor API used
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - :ref:`VectorStateBatch1 <py-vector-state-batches>`
     - Stores all :math:`N` scalar states in :math:`\mathbb{R}^1`.
   * - :ref:`WarpFactorBatch <py-warp-factor-batch>`
     - Base class for user-defined factors evaluated via Warp kernels.
       Provides ``wrap_array`` and ``make_warp_stream`` helpers.
   * - :ref:`PriorVectorFactorBatch1 <py-prior-vector-factor>`
     - Built-in prior factor that anchors :math:`x_0`.
   * - :ref:`Problem <py-problem-label>`
     - Assembles the factor graph.
   * - :ref:`LevenbergMarquardtMinimizer <py-lm-label>`
     - Solves the nonlinear system.

Warp factor code walkthrough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1 — Define the Warp kernel.**
The kernel computes the scalar difference residual and constant Jacobian
``[-1, +1]`` for each factor. State values are first gathered into
contiguous buffers (since Warp kernels operate on contiguous
``wp.array`` objects and cannot perform the double-pointer indirection
that raw CUDA kernels do).

.. code-block:: python

   import warp as wp
   from pycunls.warp import WarpFactorBatch

   @wp.kernel
   def scalar_diff_kernel(
       measurements: wp.array(dtype=wp.float32),
       left_vals:    wp.array(dtype=wp.float32),
       right_vals:   wp.array(dtype=wp.float32),
       residuals:    wp.array(dtype=wp.float32),
       jacobians:    wp.array(dtype=wp.float32),
       num_factors:  int,
       write_jacobians: int,
   ):
       i = wp.tid()
       if i >= num_factors:
           return
       residuals[i] = (right_vals[i] - left_vals[i]) - measurements[i]
       if write_jacobians != 0:
           jacobians[i * 2]     = -1.0
           jacobians[i * 2 + 1] =  1.0

**Step 2 — Subclass** :ref:`WarpFactorBatch <py-warp-factor-batch>`.
The ``evaluate`` method gathers scattered state pointers into contiguous
CuPy arrays, wraps them as ``wp.array`` objects, and launches the Warp
kernel.

.. code-block:: python

   class ScalarDiffFactor(WarpFactorBatch):
       def __init__(self, measurements_wp, num_factors):
           super().__init__(
               residual_size=1,
               state_block_sizes=[1, 1],
               num_factors=num_factors,
           )
           self.measurements = measurements_wp
           self._num_factors = num_factors

       def evaluate(self, residuals_ptr, jacobians_ptr,
                    state_pointers_ptr, stream_handle):
           n = self._num_factors

           all_vals   = _gather_state_values(state_pointers_ptr, n * 2)
           left_vals  = all_vals[0::2].copy()
           right_vals = all_vals[1::2].copy()

           left_wp  = wp.array(ptr=int(left_vals.data.ptr), ...)
           right_wp = wp.array(ptr=int(right_vals.data.ptr), ...)

           res = self.wrap_array(residuals_ptr, wp.float32, n)
           jac = self.wrap_array(jacobians_ptr, wp.float32, n * 2)

           stream = self.make_warp_stream(stream_handle)
           wp.launch(scalar_diff_kernel, dim=n,
                     inputs=[self.measurements, left_wp, right_wp,
                             res, jac, n, 1],
                     stream=stream)
           return True

**Step 3 — Build the problem and solve.**
A :ref:`VectorStateBatch1 <py-vector-state-batches>` holds all scalar
states.  Two factor batches are added: the custom Warp difference factor
and a built-in :ref:`prior anchor <py-prior-vector-factor>` on
:math:`x_0`.

.. code-block:: python

   states_gpu      = cp.asarray(initial)
   measurements_wp = wp.array(measurements_np, dtype=wp.float32, device="cuda:0")
   prior_obs_gpu   = cp.asarray(gt[:1])

   state_batch  = pycunls.VectorStateBatch1(states_gpu, num_states)
   diff_factor  = ScalarDiffFactor(measurements_wp, num_diff)
   prior_factor = pycunls.PriorVectorFactorBatch1(prior_obs_gpu, 1)

   diff_ptrs = []
   for i in range(num_diff):
       diff_ptrs.append(state_batch.state_block_device_ptr(i))
       diff_ptrs.append(state_batch.state_block_device_ptr(i + 1))

   prior_ptrs = [state_batch.state_block_device_ptr(0)]

   problem = pycunls.Problem()
   problem.add_state_batch(state_batch)
   problem.add_factor_batch(diff_factor, diff_ptrs)
   problem.add_factor_batch(prior_factor, prior_ptrs)
   assert problem.check_consistency()

   minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
   summary   = minimizer.minimize(stream, problem)

===============================================================================
Custom Warp State
===============================================================================

- **Source**: python/examples/custom_warp_state.py

Custom Warp state problem statement
-------------------------------------

This example demonstrates :ref:`WarpStateBatch <py-warp-state-batch>` (from ``pycunls.warp``) by
defining a **positive-scalar** manifold where the Plus (retraction)
operation is multiplicative:

.. math::

   x \oplus \delta = x \cdot \exp(\delta)

This makes the tangent space the reals (:math:`\delta \in \mathbb{R}`),
while states stay strictly positive — the natural parameterization for
quantities like scales, variances, or rates.

A chain of positive scalars is connected by log-ratio between-factors:

.. math::

   r_{\mathrm{prior}} = \log(x_0) - \log(t_0), \qquad
   r_i = \log(x_{i+1} / x_i) - m_i

All Jacobians equal :math:`\pm 1` because the problem is linear in the
tangent (log) space.

Warp state API used
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - :ref:`WarpStateBatch <py-warp-state-batch>`
     - Base class for user-defined state batches with a Warp-based Plus.
       Provides ``wrap_array`` and ``make_warp_stream`` helpers.
   * - :ref:`WarpFactorBatch <py-warp-factor-batch>`
     - Base class for the custom log-prior and log-ratio factors.
   * - :ref:`Problem <py-problem-label>`
     - Assembles the factor graph.
   * - :ref:`LevenbergMarquardtMinimizer <py-lm-label>`
     - Solves the nonlinear system.

Warp state code walkthrough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1 — Define the Warp Plus kernel.**
The kernel computes the multiplicative retraction for each state block.

.. code-block:: python

   @wp.kernel
   def positive_plus_kernel(
       x:             wp.array(dtype=wp.float32),
       delta:         wp.array(dtype=wp.float32),
       x_plus_delta:  wp.array(dtype=wp.float32),
       n: int,
   ):
       i = wp.tid()
       if i >= n:
           return
       x_plus_delta[i] = x[i] * wp.exp(delta[i])

**Step 2 — Subclass** :ref:`WarpStateBatch <py-warp-state-batch>`.
The ``plus`` method wraps the raw device pointers as ``wp.array`` objects
and launches the kernel.

.. code-block:: python

   from pycunls.warp import WarpStateBatch

   class PositiveScalarStateBatch(WarpStateBatch):
       def __init__(self, data, num_blocks, **kwargs):
           super().__init__(data, ambient_size=1, tangent_size=1,
                            num_blocks=num_blocks, **kwargs)
           self._num = num_blocks

       def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle):
           n = self._num
           x     = self.wrap_array(x_ptr, wp.float32, n)
           delta  = self.wrap_array(delta_ptr, wp.float32, n)
           x_out = self.wrap_array(x_plus_delta_ptr, wp.float32, n)
           stream = self.make_warp_stream(stream_handle)
           wp.launch(positive_plus_kernel, dim=n,
                     inputs=[x, delta, x_out, n], stream=stream)

**Step 3 — Define custom factors and build the problem.**
Two custom :ref:`WarpFactorBatch <py-warp-factor-batch>` subclasses compute
the log-prior and log-ratio residuals.  The problem is assembled the same
way as previous examples.

.. code-block:: python

   states_gpu = cp.asarray(initial)
   state_batch = PositiveScalarStateBatch(states_gpu, num_states)

   between_factor = LogRatioBetweenFactor(log_ratios_wp, num_between)
   prior_factor   = LogPriorFactor(prior_obs_wp, 1)

   problem = pycunls.Problem()
   problem.add_state_batch(state_batch)
   problem.add_factor_batch(between_factor, between_ptrs)
   problem.add_factor_batch(prior_factor, prior_ptrs)
   assert problem.check_consistency()

   minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
   summary   = minimizer.minimize(stream, problem)

Because the problem is linear in log-space, Gauss-Newton converges in a
single iteration.
