###############################################################################
Tutorial
###############################################################################

.. important::

   **Capacity vs. active count** (the custom Warp examples and the full API).
   Factor and state batches are constructed with their *capacity* (how many
   factors / states their buffers hold) and start with **zero** active
   entries: call ``set_num_active_factors(n)`` / ``set_num_active_states(n)``
   before solving, and again whenever the problem size changes. See
   :ref:`capacity-and-active-count`. ``pycunls.ProblemBuilder`` (the first two
   examples) does this for you.

===============================================================================
Overview
===============================================================================

This tutorial walks through four complete pycunls examples, each demonstrating
a different optimization pattern. The first two build their problems with
:doc:`pycunls.ProblemBuilder <simple_api>` from NumPy arrays:

1. Add the states from their initial values (``add_states``).
2. Add the factors from their measurements and the states each one connects
   (``add_factors``).
3. Solve (``solve``: checks the problem, then runs a minimizer) and read the
   states back as NumPy.

The two custom Warp examples use the full API, which the builder sets up
underneath (the flow of the :doc:`introduction` and :doc:`pycunls_quick_start`):

1. Allocate state data on the GPU (Warp arrays).
2. Wrap the GPU memory in one or more
   :ref:`state batch <py-state-batch-interface>` objects.
3. Build one or more
   :ref:`factor batch <py-factor-batch-interface>` objects from observations.
4. Add state batches and factor batches to a
   :ref:`Problem <py-problem-label>`.
5. Run a :ref:`minimizer <py-lm-label>` and inspect
   :ref:`MinimizerSummary <py-minimizer-summary-label>`.

The examples increase in complexity. For problems with gross outliers, see
:doc:`ransac` and ``python/examples/ransac_pnp.py``; for visual-inertial
bundle adjustment, see :doc:`imu` and
``python/examples/imu_bundle_adjustment.py``; for frame-by-frame inertial PnP
with RANSAC, see ``python/examples/tartan_vio.py``; for the full custom-type
contract (needed by RANSAC), see :doc:`custom_factors_and_states`.

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

.. note::

   **Using cuNLS from C++.** The C++ versions of these examples (and a custom
   factor written as a CUDA kernel) are in :doc:`tutorial`.

===============================================================================
Sparse Bundle Adjustment
===============================================================================

- **Source**: python/examples/sparse_bundle_adjustment.py

SBA problem statement
---------------------

This is the Python version of the C++ bundle adjustment example (see
:ref:`tutorial:Sparse Bundle Adjustment` for the full mathematical
formulation). Given :math:`M` cameras with poses
:math:`T_1, \ldots, T_M \in \mathrm{SE}(3)` (world_from_camera) and :math:`N` 3D landmarks
:math:`\mathbf{p}_1, \ldots, \mathbf{p}_N \in \mathbb{R}^3`, we minimize
the sum of squared reprojection errors across all :math:`K` observations:

.. math::

   \min_{T_1,\ldots,T_M,\; \mathbf{p}_1,\ldots,\mathbf{p}_N}
     \frac{1}{2} \sum_{k=1}^{K}
       \left\| \pi(T_{i_k}, \mathbf{p}_{j_k}) - \mathbf{z}_k \right\|^2

Poses :math:`T_0` and :math:`T_1` are held constant as gauge anchors (frame
and scale).

SBA API used
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - :doc:`ProblemBuilder <simple_api>`
     - Builds the problem from NumPy arrays, checks it, solves it and returns
       the states as NumPy.
   * - :ref:`SE3StateBatch <py-lie-state-batches>`
     - Camera poses (world_from_camera) on the SE(3) manifold; the first two
       are constant (frame and scale gauge).
   * - :ref:`VectorStateBatch3 <py-vector-state-batches>`
     - 3D landmark coordinates in :math:`\mathbb{R}^3`.
   * - :ref:`ReprojectionFactorBatch <py-reprojection-factor>`
     - Normalized reprojection residual and Jacobians of each (pose, point)
       pair.

SBA code walkthrough
~~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate synthetic data.**
``bundle_adjustment_scene`` returns ground-truth SE(3) poses and 3D points,
a perturbed initial guess (pose 0 stays exact) and the normalized observation of
every point in every camera, as NumPy arrays.
The data comes from ``python/examples/example_utils/datasets.py``; the
generators are ordinary NumPy code and are not part of the lesson.

.. literalinclude:: ../../python/examples/sparse_bundle_adjustment.py
   :language: python
   :start-at: # 1.
   :end-before: # 2.
   :dedent: 4

**Step 2 — Add the state batches.**
``add_states`` takes the class and the initial values (``(N, 4, 4)`` row-major
poses, ``(M, 3)`` points); ``constant`` lists the states held fixed. It returns
a handle for connecting factors and reading the solution.

.. literalinclude:: ../../python/examples/sparse_bundle_adjustment.py
   :language: python
   :start-at: # 2.
   :end-before: # 3.
   :dedent: 4

**Step 3 — Add the reprojection factors.**
``add_factors`` takes the class, its measurement arrays (``pycunls.describe``
lists them) and ``connect``: which states each factor reads. A reprojection
factor reads ``[pose, point]`` from two batches, so ``connect`` has one
``(handle, indices)`` pair per slot. Shapes, state types and index ranges are
checked here, with the fix in the message.

.. literalinclude:: ../../python/examples/sparse_bundle_adjustment.py
   :language: python
   :start-at: # 3.
   :end-before: # 4.
   :dedent: 4

**Step 4 — Solve on the Schur complement.**
Options go by name (fields of ``MinimizerOptions`` and
``LevenbergMarquardtMinimizerOptions``). Eliminating the points is a setting
of the underlying ``Problem`` (``pb.problem``), the full API the builder
created. ``solve`` runs ``pycunls.diagnose`` first and writes the solution into
the state arrays.

.. literalinclude:: ../../python/examples/sparse_bundle_adjustment.py
   :language: python
   :start-at: # 4.
   :end-before: # 5.
   :dedent: 4

**Step 5 — Read back and validate.**
``result[points]`` is the solved ``(M, 3)`` NumPy array; ``result.converged``
and ``result.message`` say how the solve ended.

.. literalinclude:: ../../python/examples/sparse_bundle_adjustment.py
   :language: python
   :start-at: # 5.
   :end-before: if __name__
   :dedent: 4

===============================================================================
Pose Graph Optimization
===============================================================================

- **Source**: python/examples/pose_graph_optimization.py

PGO problem statement
---------------------

This is the Python version of the C++ pose graph example (see
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
   * - :doc:`ProblemBuilder <simple_api>`
     - Builds, checks and solves the problem from NumPy arrays.
   * - :ref:`SE3StateBatch <py-lie-state-batches>`
     - The pose chain; the first pose is constant.
   * - :ref:`SE3BetweenFactorBatch <py-se3-between-factor>`
     - Relative-transform residual and its Jacobians w.r.t. both poses.

PGO code walkthrough
~~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate the pose chain and its measurements.**
``pose_chain`` returns a ground-truth chain, the relative transform between
each consecutive pair, and a perturbed initial guess.
The data comes from ``python/examples/example_utils/datasets.py``; the
generators are ordinary NumPy code and are not part of the lesson.

.. literalinclude:: ../../python/examples/pose_graph_optimization.py
   :language: python
   :start-at: # 1.
   :end-before: # 2.
   :dedent: 4

**Step 2 — Build the graph.**
One state batch with pose 0 constant; factor :math:`i` connects
``[T_i, T_{i+1}]`` through an ``(N-1, 2)`` edge array. Note the between
convention: ``Delta_i = T_{i+1}^-1 T_i``.

.. literalinclude:: ../../python/examples/pose_graph_optimization.py
   :language: python
   :start-at: # 2.
   :end-before: # 3.
   :dedent: 4

**Step 3 — Solve with Levenberg-Marquardt.**
The checks run first: without ``constant=[0]`` the builder would warn about
the gauge freedom of the chain.

.. literalinclude:: ../../python/examples/pose_graph_optimization.py
   :language: python
   :start-at: # 3.
   :end-before: # 4.
   :dedent: 4

**Step 4 — Read back and check.**
``result.evaluate()`` prints the residual statistics per factor batch (RMS,
worst factor), the first thing to look at when a solve disappoints.

.. literalinclude:: ../../python/examples/pose_graph_optimization.py
   :language: python
   :start-at: # 4.
   :end-before: # 5.
   :dedent: 4

**Step 5 — The full API underneath.**
The builder's objects stay reachable for real-time use, where batches are
allocated once and re-solved with changing active counts (see
:doc:`pycunls_quick_start` and ``python/examples/tartan_vio.py``).

.. literalinclude:: ../../python/examples/pose_graph_optimization.py
   :language: python
   :start-at: # 5.
   :end-before: if __name__
   :dedent: 4

===============================================================================
Custom Warp Factor
===============================================================================

- **Source**: python/examples/custom_warp_factor.py

.. note::

   For the full ``evaluate`` / ``plus`` contract (items, ``factor_ids_ptr``,
   ``num_factor_ids``, ``num_replicas``), a self-contained custom factor and
   state, and how to use them with the RANSAC minimizers, see
   :doc:`custom_factors_and_states`.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solve lives in ``run_chain_example``, which ``main`` calls twice: Part 1
with ``ScalarDiffFactor``, Part 2 with the residual-only factor and numeric
Jacobians.

**Step 1 — Define the Warp kernel.**
One thread per *item*: one factor evaluated at one set of states. Item ``t``
reads the measurement of its factor ``ids[t]`` and its own two state values
(through the state pointer table, with ``state_value``), and writes row ``t``. The regular minimizers evaluate each factor once
(``ids[t] == t``); the RANSAC minimizers evaluate many items per factor (see
:doc:`custom_factors_and_states`).

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :pyobject: scalar_diff_kernel

**Step 2 — Subclass WarpFactorBatch.**
``evaluate`` receives raw device pointers. ``factor_ids`` gives the factor
of every item (``t % num_active_factors`` when cuNLS passes none), the
state pointer table (``2 * n`` addresses, item ``t``'s states at ``2t`` and
``2t + 1``) is wrapped as a ``wp.uint64`` array for the kernel, and the
kernel runs on cuNLS's stream. When ``jacobians_ptr`` is 0 only residuals are wanted. The
constructor passes the capacity (measurements the buffer holds) to the base
class; ``num_active_factors`` is the active count, 0 until
``set_num_active_factors``.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :pyobject: ScalarDiffFactor

**Step 3 — The state reader.**
``state_value`` lives in ``example_utils/gpu.py`` and is reusable for any
Warp factor: it reads one component of the state a table entry points to.

.. literalinclude:: ../../python/examples/example_utils/gpu.py
   :language: python
   :pyobject: state_value

**Step 4 — Generate synthetic data.**
``scalar_chain`` returns a monotonic chain, exact consecutive differences and a
noisy initial guess.
The data comes from ``python/examples/example_utils/datasets.py``; the
generators are ordinary NumPy code and are not part of the lesson.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 1.
   :end-before: # 2.
   :dedent: 4

**Step 5 — Upload to the GPU.**
States, the prior target and the measurements (read inside the Warp kernel)
go to Warp arrays.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 2.
   :end-before: # 3.
   :dedent: 4

**Step 6 — Build states, factors and state pointers.**
Difference factors read ``[x_i, x_{i+1}]``; the built-in prior anchors
``x_0``. Every batch starts with 0 active entries and is activated with
``set_num_active_states`` / ``set_num_active_factors``.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 3.
   :end-before: # 4.
   :dedent: 4

**Step 7 — Assemble the problem.**
Part 2 registers the residual-only variant (same file,
``ScalarDiffResidualOnlyFactor``) with ``JacobianMode.numeric``; the prior
keeps its analytic Jacobian (see :doc:`numeric_jacobians`).

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 4.
   :end-before: # 5.
   :dedent: 4

**Step 8 — Solve with Levenberg-Marquardt.**
Same solver as in the previous examples. The uploads ran on Warp's stream and
the solve runs on its own ``CudaStream``, so Warp's stream is synchronized
first; ``minimize`` synchronizes its stream before returning.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 5.
   :end-before: # 6.
   :dedent: 4

**Step 9 — Report and check.**
Compare with the ground truth and check.

.. literalinclude:: ../../python/examples/custom_warp_factor.py
   :language: python
   :start-at: # 6.
   :end-before: def main
   :dedent: 4



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1 — Define the Plus kernel.**
The retraction :math:`x \oplus \delta = x\,e^{\delta}` keeps every state
positive.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :pyobject: positive_plus_kernel

**Step 2 — Subclass WarpStateBatch.**
``plus`` receives ``num_replicas`` contiguous copies of the batch (1 for the
regular minimizers, one per hypothesis for RANSAC). Every state is independent,
so all copies are one flat launch over ``num_replicas * num_active_states``
states (``num_active_states`` is the active count, at most the capacity).

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :pyobject: PositiveScalarStateBatch

**Step 3 — Define the factors.**
Two custom factors in log space: a prior on ``x_0`` and a log-ratio between
consecutive states. Both follow the same item pattern as in the custom factor
example: measurement by ``ids[t]``, states and outputs by ``t``.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :pyobject: log_ratio_kernel

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :pyobject: LogRatioBetweenFactor

**Step 4 — Generate synthetic data.**
``positive_chain`` returns a growing positive chain, its log-ratio
measurements and a noisy initial guess.
The data comes from ``python/examples/example_utils/datasets.py``; the
generators are ordinary NumPy code and are not part of the lesson.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 1.
   :end-before: # 2.
   :dedent: 4

**Step 5 — Upload to the GPU.**
States and factor data go to Warp arrays.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 2.
   :end-before: # 3.
   :dedent: 4

**Step 6 — Build the custom state and factor batches.**
The custom state batch wraps the Warp array; the factors read
``[x_i, x_{i+1}]`` and ``x_0``. Custom batches follow the same capacity rule
as the built-in ones: 0 active entries until ``set_num_*`` is called.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 3.
   :end-before: # 4.
   :dedent: 4

**Step 7 — Assemble the problem.**
Register the state batch and both factor batches.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 4.
   :end-before: # 5.
   :dedent: 4

**Step 8 — Solve with Levenberg-Marquardt.**
Same solver as in the previous examples; every step goes through the custom
``plus``.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 5.
   :end-before: # 6.
   :dedent: 4

**Step 9 — Report and check.**
Errors are compared in log space; all states must stay positive.

.. literalinclude:: ../../python/examples/custom_warp_state.py
   :language: python
   :start-at: # 6.
   :end-before: if __name__
   :dedent: 4
