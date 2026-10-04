.. raw:: html

   <div style="display:flex; justify-content:center; margin: 0.5rem 0 1.0rem 0;">
     <div style="max-width: 420px; width: 100%;">
      <img class="only-light" src="_static/cuNLS_logo_light.png" alt="cuNLS logo" style="width:100%; height:auto;">
      <img class="only-dark" src="_static/cuNLS_logo_dark.png" alt="cuNLS logo" style="width:100%; height:auto;">
     </div>
   </div>

###############################################################################
Introduction
###############################################################################

===============================================================================
Purpose
===============================================================================

cuNLS is a library for solving nonlinear least-squares problems on the GPU.
It is designed around batched factor evaluation, sparse Jacobian assembly,
and sparse linear solvers tailored for large scale minimization problems.

cuNLS is used primarily from Python through **pycunls**, a package that
exposes the full library through CuPy-based GPU arrays (see
:doc:`pycunls_installation` and :doc:`pycunls_quick_start`). For advanced
extensibility, pycunls integrates with `NVIDIA Warp
<https://developer.nvidia.com/warp-python>`_ to let users author custom
factor and state kernels in Python (see :doc:`pycunls_tutorial`).

The same functionality is available as a native **C++/CUDA API**
(``libcunls``), for applications written in C++ or for custom factors
implemented directly as CUDA kernels (see :doc:`installation`,
:doc:`quick_start` and :doc:`tutorial`).

===============================================================================
Nonlinear least-squares problems
===============================================================================

At a high level, cuNLS solves optimization problems of the form:

.. math::
   x^* = \arg\min_x \sum_i \rho_i\left(\left\|f_i(x)\right\|^2_{\Sigma_i}\right)

where:

- :math:`x` is the optimization variable (often living on a manifold),
- :math:`f_i(x)` are residual/error functions,
- :math:`\rho_i(\cdot)` are optional robust loss functions,
- :math:`\left\|v\right\|^2_{\Sigma} = v^T \Sigma^{-1} v` is the Mahalanobis norm.

Using a square-root information matrix :math:`R_i` such that
:math:`\Sigma_i^{-1} = R_i^T R_i`, each term can be rewritten as:

.. math::
   \left\|f_i(x)\right\|^2_{\Sigma_i} = \left\|R_i f_i(x)\right\|^2

This is exactly why cuNLS has a dedicated ``InformationFactorBatch``: it applies
this whitening step directly to residuals and Jacobians.  ``WeightedFactorBatch``
does the same for scalar weighting.  In C++, the template
``InformationFactorBatch<T>`` inherits ``T::sized_layout`` (the same
``SizedFactorBatch`` as the inner batch), and so does
``WeightedFactorBatch<T>``.

To solve the nonlinear problem, cuNLS linearizes around the current estimate
:math:`x_0`:

.. math::
   f_i(x_0 + \Delta x) \approx f_i(x_0) + J_i \Delta x

Stacks all blocks into a global Jacobian :math:`J` and vector :math:`b`, then
solves a sparse linearized system (Gauss-Newton / Levenberg-Marquardt):

.. math::
   \Delta x^* = \arg\min_{\Delta x} \left\|J\Delta x + b\right\|^2

with normal equations:

.. math::
   J^T J \Delta x^* = -J^T b

and updates:

.. math::
   x^* = x_0 \oplus \Delta x^*

where :math:`\oplus` is the manifold plus operation implemented by state
batches (for Euclidean states, this reduces to simple addition).

===============================================================================
Factor Graphs
===============================================================================

A common way to set up nonlinear least-squares problems is to create a factor graph:
a graph where nodes represent variables and edges represent constraints between them.

.. raw:: html

   <div style="display:flex; justify-content:center; margin:1rem 0;">
     <div style="max-width:80%; width:100%;">
       <img class="only-light" src="_static/fg.png" alt="Factor graph illustration" style="width:100%; height:auto;">
       <img class="only-dark" src="_static/fg_dark.png" alt="Factor graph illustration" style="width:100%; height:auto;">
     </div>
   </div>

.. rst-class:: centered

   Example factor-graph structure used to represent sparse nonlinear least-squares problems.

The constraints between variables are called factors, which are nonlinear functions representing mean error.
Each factor is also associated with a covariance matrix.
Together the mean and the covariance represent multivariate normal distribution for a given factor.

This way factor graph is a probabilistic graphical model,
which represents a joint probability distribution of all factors

.. math::
   p(x) \propto \prod_i p_i(x_i)

and the MAP estimate is:

.. math::
   x^* = \arg\max_x p(x)

For Gaussian-like factors:

.. math::
   p_i(x_i) \propto \exp\left(-\frac{1}{2}\left\|f_i(x_i)\right\|^2_{\Sigma_i}\right)

maximizing the posterior is equivalent to minimizing the sum of squared (and
optionally robustified) residuals.

cuNLS allows setting up variables and factors in batches for higher GPU utilization.
A `FactorBatch` is a collection of same type factors that are connected to a list of `StateBatch` objects —
collections of same type variables.
The `Problem` is a collection of `FactorBatch` objects and connected `StateBatch` objects, that together form the Factor Graph.

===============================================================================
Core concepts
===============================================================================

- **State batches** store optimization variables on manifolds (for example,
  ``SE3StateBatch`` for rigid transforms, ``VectorStateBatch3`` for Euclidean
  3D vectors; in C++, ``VectorStateBatch<Dim>``).
- **Factor batches** compute residuals and Jacobians in parallel for many
  observations.
- **Problems** connect factors to states via device pointers
  (``state_device_ptr(i)``; C++ ``StateDevicePtr(i)``).
- **Minimizers** (``GaussNewtonMinimizer``, ``LevenbergMarquardtMinimizer``)
  solve for state updates (``minimize``; C++ ``Minimize``).
- **Loss functions** robustify residuals to reduce outlier influence.
- **RANSAC minimizers** (``RansacGaussNewtonMinimizer``,
  ``RansacLevenbergMarquardtMinimizer``) solve the same problems when many
  measurements are gross outliers, and return the inlier set
  (:doc:`ransac`).

.. _capacity-and-active-count:

===============================================================================
Capacity and active count
===============================================================================

.. important::

   Every factor batch and state batch has **two sizes**. The constructor takes
   the **capacity**; the **active count** starts at **0** and must be set with
   ``set_num_active_factors`` / ``set_num_active_states`` (C++:
   ``SetNumActiveFactors`` / ``SetNumActiveStates``) before solving. A
   solve with nothing active raises ``ValueError`` (C++:
   ``std::invalid_argument``).

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - **Capacity**
     - **Active count** (``num_active_factors`` / ``num_active_states``;
       C++ ``NumActiveFactors()`` / ``NumActiveStates()``)
   * - What it is
     - How many factors (states) the batch's device buffers hold
       (``capacity``; C++ ``Capacity()``).
     - How many of the *first* factors (states) the next solve uses.
   * - Set by
     - The constructor. Fixed for the batch's lifetime.
     - ``set_num_active_factors(n)`` /
       ``set_num_active_states(n, num_const_states=0)``, any
       ``n <= capacity`` (C++: ``SetNumActiveFactors(n)`` /
       ``SetNumActiveStates(n, num_const_states)``, ``n <= Capacity()``).
       Starts at 0.
   * - Cost of changing
     - Not changeable.
     - Host-only assignment: no allocation, no device work, no sync.
   * - Typical value
     - The largest problem you expect.
     - The size of the problem you are solving now.

**Why two sizes.** Real-time applications (tracking, sliding-window SLAM,
per-frame registration) solve a new problem every frame, with a different
number of measurements and states each time. They allocate their device
buffers *once*, for the largest problem, construct the batches *once* with that
capacity, and then, every frame, rewrite the buffer contents in place and set
the active counts. Nothing is reallocated or reconstructed, and the solve only
pays for the active part. A one-shot solve simply sets the active count equal
to the capacity.

.. code-block:: python

   import cupy as cp
   import pycunls

   # Once: buffers and batches sized for the largest problem (the capacity).
   max_points = 100000
   obs_gpu = cp.zeros((max_points, 2), dtype=cp.float32)
   pts_gpu = cp.zeros((max_points, 3), dtype=cp.float32)
   pnp = pycunls.PnPFactorBatch(obs_gpu, pts_gpu, max_points)   # capacity
   # ... state batch, problem, minimizer ...

   # Every frame: write the first num_points entries, then set the active count.
   pnp.set_num_active_factors(num_points)       # num_points <= max_points
   pose_state.set_num_active_states(1)
   minimizer.minimize(stream, problem)

The same in C++:

.. code-block:: cpp

   // Once: buffers and batches sized for the largest problem (the capacity).
   const size_t max_points = 100000;
   cunls::dvector<Vector<2>> obs(max_points);
   cunls::dvector<Vector<3>> pts(max_points);
   cunls::PnPFactorBatch pnp(obs.data(), pts.data(), /*capacity=*/max_points);
   // ... state batch, problem, minimizer ...

   // Every frame: write the first num_points entries, then set the active count.
   pnp.SetNumActiveFactors(num_points);       // num_points <= max_points
   pose_state.SetNumActiveStates(1);
   minimizer.Minimize(stream, problem);

**Rules.**

- Every buffer bound to a batch must hold its **capacity**: measurements,
  state values, constant ids, connectivity tables.
- Only the **active** part is read and written by a solve: factors
  ``[0, num_active_factors)``, states ``[0, num_active_states)``, the first
  ``num_const_states`` constant ids (C++: ``NumActiveFactors()``,
  ``NumActiveStates()``, ``NumConstStates()``). Results are written back to the active
  states only.
- Factors may only reference **active** states. State addresses
  (``state_device_ptr(i)``; C++ ``StateDevicePtr(i)``) are valid for every
  ``i < capacity``, so the
  connectivity of the next solve can be built before the counts are set.
- Connectivity must cover the active factors: a host pointer list needs at
  least ``num_active_factors * B`` entries (``Problem.set_state_pointers``;
  C++ ``Problem::SetStatePointers`` replaces it); device tables are read for
  their first ``num_active_factors * B`` entries.
- Never change sizes or buffer contents while a solve that uses them runs.
- Every minimizer checks the sizes at the start of ``minimize`` (C++
  ``Minimize``, via ``Problem::CheckSizes``, host-only): nothing active, a
  count above its capacity, or connectivity shorter than the active factors
  raises an error with an explanatory message.

How connectivity is rewritten between solves (host lists, device pointer
tables, device index tables) is described in the ``Problem`` API reference
(:doc:`api/minimizer`).

===============================================================================
High-level solve flow
===============================================================================

1. Allocate state and measurement data on the GPU (CuPy arrays in Python,
   device buffers in C++), sized for the largest problem (the capacity).
2. Wrap state memory in one or more ``StateBatch`` objects (constructed with
   their capacity).
3. Build one or more ``FactorBatch`` objects from observations (constructed
   with their capacity).
4. Set the active counts: ``set_num_active_states`` /
   ``set_num_active_factors`` (C++: ``SetNumActiveStates`` /
   ``SetNumActiveFactors``; see :ref:`capacity-and-active-count`).
5. Add state batches and factor batches to a ``Problem``
   (``add_state_batch`` / ``add_factor_batch``; C++ ``AddStateBatch`` /
   ``AddFactorBatch``).
6. Run a minimizer (``minimize``; C++ ``Minimize``) and inspect the returned
   ``MinimizerSummary``. To solve the next problem, rewrite the buffers, set
   the new active counts, and solve again.

===============================================================================
Supported optimization patterns
===============================================================================

- Pose graph optimization with between factors.
- Bundle-adjustment style reprojection optimization.
- ICP-like alignment (point-to-point / point-to-plane factors).
- Custom user-defined factors: in Python through ``CustomFactorBatch`` /
  ``CustomStateBatch`` (CuPy kernels) or ``WarpFactorBatch`` /
  ``WarpStateBatch`` (NVIDIA Warp); in C++ through ``FactorBatch`` /
  ``SizedFactorBatch`` (see :doc:`custom_factors_and_states`).

See :doc:`pycunls_tutorial` for complete Python examples, :doc:`tutorial` for
complete C++ working pipelines, and :doc:`api/index` for class-level API
details.
