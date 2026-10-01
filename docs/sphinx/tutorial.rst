###############################################################################
Tutorial
###############################################################################

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``SetNumActiveFactors(n)`` /
   ``SetNumActiveStates(n)`` before solving, and again whenever the problem size
   changes. See :ref:`capacity-and-active-count`.

===============================================================================
Overview
===============================================================================

This tutorial walks through three complete cuNLS examples, each demonstrating a
different optimization pattern. Every example follows the same high-level flow
described in the :doc:`introduction`:

1. Allocate state data on the GPU.
2. Wrap state memory in one or more `StateBatch` objects
   (see :doc:`api/state`).
3. Build one or more `FactorBatch` objects from observations
   (see :doc:`api/factor`).
4. Add state batches and factor batches to a `Problem`
   (see :doc:`api/minimizer`).
5. Run a minimizer and inspect `MinimizerSummary`.

The examples increase in complexity:

- :ref:`tutorial:Sparse Bundle Adjustment` — uses built-in
  `ReprojectionFactorBatch` to jointly optimize camera poses and 3D
  landmarks from multi-view observations.
- :ref:`tutorial:Pose Graph Optimization` — uses `BetweenFactorBatch` (the
  manifold-generic facade, deduced to SE(3) here) to recover a chain of
  SE(3) poses from consecutive relative-transform measurements.
- :ref:`tutorial:Custom Factor` — shows how to implement a user-defined CUDA
  factor kernel by subclassing `SizedFactorBatch`.

Full source code for all examples lives in the ``examples/`` directory and is
built by the shared ``examples/CMakeLists.txt``.

===============================================================================
Common build commands
===============================================================================

Build all examples locally:

.. code-block:: bash

   cmake -S examples -B build/examples/all \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUNLS_INSTALL_DIR=/path/to/cunls_install
   cmake --build build/examples/all -j

Build all examples in Docker and export binaries:

.. code-block:: bash

   ./examples/build_in_docker.sh Release ./artifacts/examples

===============================================================================
Sparse Bundle Adjustment
===============================================================================

- **Source**: examples/sparse_bundle_adjustment/main.cpp

Problem statement
-----------------

`Bundle adjustment <https://en.wikipedia.org/wiki/Bundle_adjustment>`_ is the
problem of jointly refining 3D structure and camera parameters to minimize
**reprojection error** — the difference between where a 3D point actually
projects into an image and where it was observed.

Given :math:`M` cameras with poses :math:`T_1, \ldots, T_M \in \mathrm{SE}(3)`
and :math:`N` 3D landmarks :math:`\mathbf{p}_1, \ldots, \mathbf{p}_N \in
\mathbb{R}^3`, we form one residual per observation. The projection model
transforms a world point :math:`\mathbf{p}` into camera :math:`i`'s frame and
divides by depth to obtain **normalized image coordinates**:

.. math::

   \mathbf{p}_{\mathrm{cam}} = T_i \, \mathbf{p}, \qquad
   \hat{\mathbf{z}} = \begin{bmatrix}
     p_{\mathrm{cam},x} / p_{\mathrm{cam},z} \\
     p_{\mathrm{cam},y} / p_{\mathrm{cam},z}
   \end{bmatrix}

The reprojection residual for observation :math:`k`, which pairs camera
:math:`i` with point :math:`j`, is:

.. math::

   r_k = \hat{\mathbf{z}}_k - \mathbf{z}_k
       = \pi(T_i,\, \mathbf{p}_j) - \mathbf{z}_k

where :math:`\mathbf{z}_k` is the measured 2D observation and :math:`\pi` is
the normalized projection function above.

The full bundle adjustment objective minimizes the sum of squared reprojection
errors across all :math:`K` observations:

.. math::

   \min_{T_1,\ldots,T_M,\; \mathbf{p}_1,\ldots,\mathbf{p}_N}
     \frac{1}{2} \sum_{k=1}^{K}
       \left\| \pi(T_{i_k}, \mathbf{p}_{j_k}) - \mathbf{z}_k \right\|^2

Because applying a rigid transform to every pose and point leaves all
reprojection residuals unchanged, the system has a 6-DOF gauge freedom.
Fixing one camera pose as a **gauge anchor** removes this freedom. In this
example, the first pose :math:`T_0` is held constant while the remaining
poses :math:`T_1, \ldots, T_{M-1}` and all 3D points are jointly optimized
— the classic full bundle adjustment problem.

BA factor graph
~~~~~~~~~~~~~~~

The problem has a **bipartite** factor graph: camera-pose variable nodes on one
side, 3D-point variable nodes on the other, and reprojection factor nodes
connecting them.

.. raw:: html

   <div style="display:flex; justify-content:center; margin:1rem 0;">
     <div style="max-width:80%; width:100%;">
       <img class="only-light" src="_static/sba_fg.png" alt="Bundle adjustment factor graph" style="width:100%; height:auto;">
       <img class="only-dark" src="_static/sba_fg_dark.png" alt="Bundle adjustment factor graph" style="width:100%; height:auto;">
     </div>
   </div>

.. rst-class:: centered

   *Factor graph for sparse bundle adjustment. The blue circle is the fixed
   anchor pose*
   :math:`T_0`\ *, green circles are optimized camera poses
   (*\ `SE3StateBatch`\ *) and 3D point variables
   (*\ `VectorStateBatch<3>`\ *), and orange squares are reprojection factors
   (*\ `ReprojectionFactorBatch`\ *). Each factor connects one camera and one
   point.*

BA API used
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - `SE3StateBatch` (:doc:`api/state`)
     - Stores camera poses on the SE(3) manifold. The first pose is marked
       constant via ``device_constant_state_ids``; the rest are optimized.
   * - `VectorStateBatch<3>` (:doc:`api/state`)
     - Stores 3D landmark coordinates in :math:`\mathbb{R}^3`. All points
       are optimization variables.
   * - `ReprojectionFactorBatch` (:doc:`api/factor`)
     - Computes normalized reprojection residuals and Jacobians for each
       (pose, point) pair.
   * - `Problem` (:doc:`api/minimizer`)
     - Assembles the factor graph: connects factors to states via device
       pointers.
   * - `LevenbergMarquardtMinimizer` (:doc:`api/minimizer`)
     - Iteratively solves the nonlinear least-squares problem with adaptive
       damping.

BA code walkthrough
~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate synthetic data.**
``MakeBundleAdjustmentScene`` returns ground-truth SE(3) poses and 3D points
(every point in front of every camera), a perturbed initial guess (pose
:math:`T_0` stays exact), and the normalized observation of every point in
every camera.
The scene comes from ``examples/utils/datasets.h``; the generators are ordinary
host code and are not part of the lesson.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 1.
   :end-before: // 2.
   :dedent: 4

**Step 2 — Upload to the GPU.**
``dvector`` owns device memory. The initial guess is uploaded into the
buffers the solver will update in place; ``constant_pose_ids`` lists the
gauge anchor.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 2.
   :end-before: // 3.
   :dedent: 4

**Step 3 — Wrap the device memory in state batches.**
A state batch wraps device memory without copying it. Poses use
`SE3StateBatch` with state 0 marked constant; points use `VectorStateBatch<3>`
(see :doc:`api/state`).

**Capacity and active count.** The count passed to a batch constructor is
its *capacity*: how many states (state batches) or factors (factor batches)
the bound device buffers hold, fixed for the batch's lifetime. Right after
construction nothing is active. ``SetNumActiveStates`` /
``SetNumActiveFactors`` set the *active count*: how many of the first states
or factors the next solve uses. They are host-only (no allocation) and may be
called again between solves with any count up to the capacity, so one set of
batches serves problems of changing size. In this example every slot is
used, so active = capacity.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 3.
   :end-before: // 4.
   :dedent: 4

**Step 4 — Build the reprojection factor batch and its state pointers.**
Each reprojection factor reads two states, ``[pose, point]``. The
state-pointer list is flattened in factor order, two pointers per factor, and
tells cuNLS which states every factor reads. ``z_threshold`` guards points
almost behind a camera (see :doc:`api/factor`). Like state batches, a factor
batch is constructed with its capacity and starts with 0 active factors:
``SetNumActiveFactors`` activates them.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 4.
   :end-before: // 5.
   :dedent: 4

**Step 5 — Assemble the problem.**
`Problem` connects state batches and factor batches. ``CheckConsistency``
verifies that every state pointer lies inside a registered state batch.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 5.
   :end-before: // 6.
   :dedent: 4

**Step 6 — Solve with Levenberg-Marquardt.**
`LevenbergMarquardtMinimizer` (see :doc:`api/minimizer`) solves the damped
normal equations each iteration and adapts :math:`\lambda` from the step
quality. ``Minimize`` writes the solution back into the state batches'
memory.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 6.
   :end-before: // 7.
   :dedent: 4

**Step 7 — Read back and validate.**
Copy the optimized poses and points back to the host, compare them with
the ground truth, print the summary, and turn the quality check into the
exit code.

.. literalinclude:: ../../examples/sparse_bundle_adjustment/main.cpp
   :language: cpp
   :start-at: // 7.
   :end-before: } catch
   :dedent: 4



===============================================================================
Pose Graph Optimization
===============================================================================

- **Source**: examples/pose_graph_optimization/main.cpp

PGO problem statement
---------------------

`Pose graph optimization
<https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping>`_ (PGO) is
a key building block in Simultaneous Localization and Mapping (SLAM). Given a
set of poses and **pairwise relative-transform measurements** between them, the
goal is to find the configuration of poses that best satisfies all measurements.

This example models a **pose chain**: :math:`N` poses
:math:`T_0, T_1, \ldots, T_{N-1} \in \mathrm{SE}(3)` connected by
:math:`N{-}1` consecutive between constraints. Each constraint :math:`i`
connects pose :math:`T_i` to pose :math:`T_{i+1}` and carries a measured
relative transform :math:`\Delta_i`. The relative-transform residual is
defined on the SE(3) Lie algebra:

.. math::

   r_i = \mathrm{Log}\!\left(
     \Delta_i \, T_i^{-1} \, T_{i+1}
   \right) \in \mathbb{R}^6

The residual :math:`r_i` is the 6-DOF twist that measures how far the
observed relative transform deviates from the measurement. When the constraint
is exactly satisfied, the argument of :math:`\mathrm{Log}` is the identity
and :math:`r_i = 0`.

Because adding a rigid transform to every pose leaves all relative residuals
unchanged, the system is rank-deficient without further constraints. Fixing
the first pose :math:`T_0` as a **gauge anchor** removes this freedom.

The optimization objective minimizes the sum of squared residuals over all
non-fixed poses:

.. math::

   \min_{T_1,\ldots,T_{N-1}}
     \frac{1}{2} \sum_{i=0}^{N-2}
       \left\| \mathrm{Log}\!\left(
         \Delta_i \, T_i^{-1} \, T_{i+1}
       \right) \right\|^2

For a thorough introduction to graph-based SLAM see Grisetti et al.,
`A Tutorial on Graph-Based SLAM
<http://ais.informatik.uni-freiburg.de/teaching/ws11/robotics2/pdfs/ls-slam-tutorial.pdf>`_,
IEEE Intelligent Transportation Systems Magazine, 2010.

PGO factor graph
~~~~~~~~~~~~~~~~

The factor graph is a **chain**: each pose node connects to its neighbors
through between factors, and the first pose :math:`T_0` is held fixed.

.. raw:: html

   <div style="display:flex; justify-content:center; margin:1rem 0;">
     <div style="max-width:80%; width:100%;">
       <img class="only-light" src="_static/pgo_fg.png" alt="Pose graph optimization factor graph" style="width:100%; height:auto;">
       <img class="only-dark" src="_static/pgo_fg_dark.png" alt="Pose graph optimization factor graph" style="width:100%; height:auto;">
     </div>
   </div>

.. rst-class:: centered

   *Factor graph for pose graph optimization. The blue circle is the fixed
   anchor pose*
   :math:`T_0`\ *, green circles are optimized poses
   (*\ `SE3StateBatch`\ *), and orange squares are between factors
   (*\ `BetweenFactorBatch`\ *, deduced to SE(3)). Each factor encodes a
   measured relative transform*
   :math:`\Delta_i`\ *.*

PGO API used
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - `SE3StateBatch` (:doc:`api/state`)
     - A single instance stores the full pose chain. The first pose is
       marked constant via ``device_constant_state_ids``; the rest are
       optimized.
   * - `BetweenFactorBatch<Manifold>` (:doc:`api/factor`)
     - Manifold-generic facade; deduced here to SE(3) via CTAD from the
       deltas pointer's type. Computes the relative-transform residual
       :math:`\mathrm{Log}(\Delta \, T_i^{-1} \, T_{i+1})` and its
       Jacobians w.r.t. both poses.
   * - `Problem` (:doc:`api/minimizer`)
     - Assembles the factor graph.
   * - `LevenbergMarquardtMinimizer` (:doc:`api/minimizer`)
     - Solves the nonlinear system.

PGO code walkthrough
~~~~~~~~~~~~~~~~~~~~

**Step 1 — Generate the pose chain and its measurements.**
``MakePoseChainScene`` returns a ground-truth chain of SE(3) poses, the
relative transform (``delta``) between each consecutive pair, and a disturbed
initial guess.
The scene comes from ``examples/utils/datasets.h``; the generators are ordinary
host code and are not part of the lesson.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 1.
   :end-before: // 2.
   :dedent: 4

**Step 2 — Upload to the GPU.**
Upload the initial guess (updated in place by the solver) and the
measurements. Pose :math:`T_0` is the gauge anchor.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 2.
   :end-before: // 3.
   :dedent: 4

**Step 3 — Wrap the chain in one state batch.**
All poses live in a single `SE3StateBatch`; only state 0 is constant.
``SetNumActiveStates`` activates all poses and the constant state (batches
start with 0 active entries).

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 3.
   :end-before: // 4.
   :dedent: 4

**Step 4 — Build the between factors and their state pointers.**
`BetweenFactorBatch` deduces its manifold (SE(3)) from the type of the
measurements. Factor :math:`i` reads ``[T_i, T_{i+1}]``. ``SetNumActiveFactors``
activates all of them.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 4.
   :end-before: // 5.
   :dedent: 4

**Step 5 — Assemble the problem.**
Register the state batch and the factor batch with their state pointers.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 5.
   :end-before: // 6.
   :dedent: 4

**Step 6 — Solve with Levenberg-Marquardt.**
Same solver as in the bundle adjustment example.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 6.
   :end-before: // 7.
   :dedent: 4

**Step 7 — Read back and validate.**
Copy the chain back and measure how well it satisfies its relative
constraints before and after the solve.

.. literalinclude:: ../../examples/pose_graph_optimization/main.cpp
   :language: cpp
   :start-at: // 7.
   :end-before: } catch
   :dedent: 4



===============================================================================
Custom Factor
===============================================================================

- **Source**: examples/custom_factor/main.cu

.. note::

   This walkthrough shows the mechanics on a small example. The complete
   contract of ``Evaluate`` (items, ``factor_ids``, ``num_factor_ids``) and
   ``Plus`` (``num_replicas``), which every custom type must honor to work
   with the RANSAC minimizers, is explained in :doc:`custom_factors_and_states`.

Custom factor problem statement
-------------------------------

This example shows how to implement a **user-defined factor** by subclassing
`SizedFactorBatch` (see :doc:`api/factor`) and writing a CUDA kernel that
computes residuals and Jacobians.

We model a 1-D chain of :math:`N` scalar states
:math:`x_0, x_1, \ldots, x_{N-1}` connected by :math:`N{-}1` **difference
constraints**. Each constraint carries a measurement :math:`m_i` of the
expected difference between consecutive states:

.. math::

   r_i = (x_{i+1} - x_i) - m_i, \qquad i = 0, \ldots, N{-}2

The Jacobians are trivially constant:

.. math::

   \frac{\partial r_i}{\partial x_i} = -1, \qquad
   \frac{\partial r_i}{\partial x_{i+1}} = +1

Because adding a constant to every state leaves all difference residuals
unchanged, the system is rank-deficient without further constraints. A
**prior factor** (anchor) on the first state removes this gauge freedom:

.. math::

   r_{\mathrm{prior}} = x_0 - x_0^{\mathrm{obs}}

The full objective is:

.. math::

   \min_{x_0, \ldots, x_{N-1}}
     \frac{1}{2} \left\| x_0 - x_0^{\mathrm{obs}} \right\|^2
     + \frac{1}{2} \sum_{i=0}^{N-2}
       \left\| (x_{i+1} - x_i) - m_i \right\|^2

This is a simple linear-in-state problem, but it demonstrates the full
workflow for authoring custom factors.

Custom factor graph
~~~~~~~~~~~~~~~~~~~

The factor graph is a **chain**: each variable node connects to its neighbors
through difference factors, and a prior factor anchors :math:`x_0`.

.. raw:: html

   <div style="display:flex; justify-content:center; margin:1rem 0;">
     <div style="max-width:80%; width:100%;">
       <img class="only-light" src="_static/custom_fg.png" alt="Custom factor (1D chain) factor graph" style="width:100%; height:auto;">
       <img class="only-dark" src="_static/custom_fg_dark.png" alt="Custom factor (1D chain) factor graph" style="width:100%; height:auto;">
     </div>
   </div>

.. rst-class:: centered

   *Factor graph for the custom factor example. Green circles are scalar state
   variables (*\ `VectorStateBatch<1>`\ *), orange squares are custom
   difference factors (*\ `ScalarDifferenceFactorBatch`\ *), and the purple
   square is the anchor prior (*\ `PriorFactorBatch<manifold::Vector<1>>`\ *)
   on*
   :math:`x_0`\ *.*

Custom factor API used
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Role
   * - `VectorStateBatch<1>` (:doc:`api/state`)
     - Stores all :math:`N` scalar states in :math:`\mathbb{R}^1`.
   * - `SizedFactorBatch<1, 1, 1>` (:doc:`api/factor`)
     - Compile-time base for the custom factor (residual dim = 1, two states
       of tangent dim 1 each).
   * - `PriorFactorBatch<manifold::Vector<1>>` (:doc:`api/factor`)
     - Manifold-generic facade over the built-in prior factor that pulls
       :math:`x_0` toward the observed anchor value.
   * - `Problem` (:doc:`api/minimizer`)
     - Assembles the factor graph.
   * - `LevenbergMarquardtMinimizer` (:doc:`api/minimizer`)
     - Solves the nonlinear system.

Custom factor code walkthrough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All steps of the solve live in ``RunChainExample``, which ``main`` calls twice: Part 1 with the analytic factor, Part 2 with the residual-only factor and numeric Jacobians.

**Step 1 — Implement the CUDA kernel.**
The kernel is launched with one thread per *item*: one factor evaluated
against its own set of states. Item ``idx`` reads the measurement of its
factor (``factor_ids[idx]``, or ``idx % num_factors`` when ``factor_ids`` is
null), its two state pointers ``state_pointers[2 * idx ..]``, and writes
row ``idx`` of the outputs. The regular minimizers pass ``factor_ids ==
nullptr`` and one item per factor; the RANSAC minimizers evaluate many items
per factor (see :doc:`custom_factors_and_states`).

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cuda
   :start-at: __global__ void ScalarDifferenceKernel
   :end-before: // -------

**Step 2 — Subclass SizedFactorBatch<1, 1, 1>.**
The template arguments encode the residual dimension (1) and the tangent
dimensions of the two states (1, 1). The constructor passes the
capacity (the number of measurements the buffer holds) to the base class, which
keeps the active count ``NumActiveFactors()``: 0 until ``SetNumActiveFactors`` is called.
The class stores a device pointer to the measurements and launches the kernel
in ``Evaluate`` over the active factors.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: class ScalarDifferenceFactorBatch
   :end-at: };

**Step 3 — Generate synthetic data.**
``MakeScalarChainScene`` returns a monotonic ground-truth chain, a noisy
initial guess, and the exact differences of consecutive states.
The scene comes from ``examples/utils/datasets.h``; the generators are ordinary
host code and are not part of the lesson.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 1.
   :end-before: // 2.
   :dedent: 2

**Step 4 — Upload to the GPU.**
Upload the states and the differences. The prior's target anchors
:math:`x_0`: without it, adding a constant to every state would leave all
differences unchanged.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 2.
   :end-before: // 3.
   :dedent: 2

**Step 5 — Build the state batch.**
All scalar states share one `VectorStateBatch<1>`, activated with
``SetNumActiveStates``.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 3.
   :end-before: // 4.
   :dedent: 2

**Step 6 — Build the factor batches and their state pointers.**
Difference factors read ``[x_i, x_{i+1}]``; the shipped prior reads
``x_0``. The residual-only class is used in Part 2 (see
:doc:`numeric_jacobians`). Every factor batch is activated with
``SetNumActiveFactors``.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 4.
   :end-before: // 5.
   :dedent: 2

**Step 7 — Assemble the problem.**
Part 2 registers the residual-only factor with
``JacobianMode::kNumeric``; the prior stays analytic in the same problem.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 5.
   :end-before: // 6.
   :dedent: 2

**Step 8 — Solve with Levenberg-Marquardt.**
Same solver as in the previous examples.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 6.
   :end-before: // 7.
   :dedent: 2

**Step 9 — Read back and validate.**
Compare the solved chain with the ground truth and report.

.. literalinclude:: ../../examples/custom_factor/main.cu
   :language: cpp
   :start-at: // 7.
   :end-at: mse_after <= mse_before * 0.02f);
   :dedent: 2
