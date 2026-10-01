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

cuNLS is a CUDA/C++ library for solving nonlinear least-squares problems on the
GPU. It is designed around batched factor evaluation, sparse Jacobian assembly,
and sparse linear solvers tailored for large scale minimization problems.

cuNLS also provides **pycunls**, a Python package that exposes the full C++
API through CuPy-based GPU arrays (see :doc:`pycunls_installation`). For
advanced extensibility, pycunls integrates with `NVIDIA Warp
<https://developer.nvidia.com/warp-python>`_ to let users
author custom factor and state kernels in Python (see
:doc:`pycunls_tutorial`).

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
this whitening step directly to residuals and Jacobians.  In C++, the template
``InformationFactorBatch<T>`` inherits ``T::sized_layout`` (the same
``SizedFactorBatch`` as the inner batch).  ``WeightedFactorBatch<T>`` does the
same for scalar weighting.

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
  `SE3StateBatch` for rigid transforms, `VectorStateBatch<Dim>` for Euclidean
  vectors).
- **Factor batches** compute residuals and Jacobians in parallel for many
  observations.
- **Problems** connect factors to states via device pointers.
- **Minimizers** (`GaussNewtonMinimizer`, `LevenbergMarquardtMinimizer`) solve
  for state updates.
- **Loss functions** robustify residuals to reduce outlier influence.
- **RANSAC minimizers** (`RansacGaussNewtonMinimizer`,
  `RansacLevenbergMarquardtMinimizer`) solve the same problems when many
  measurements are gross outliers, and return the inlier set
  (:doc:`ransac`).

.. _capacity-and-active-count:

===============================================================================
Capacity and active count
===============================================================================

.. important::

   Every factor batch and state batch has **two sizes**. The constructor takes
   the **capacity**; the **active count** starts at **0** and must be set with
   ``SetNumActiveFactors`` / ``SetNumActiveStates`` (Python:
   ``set_num_active_factors`` / ``set_num_active_states``) before solving. A
   solve with nothing active throws ``std::invalid_argument`` (Python
   ``ValueError``).

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - **Capacity**
     - **Active count** (``NumActiveFactors()`` / ``NumActiveStates()``)
   * - What it is
     - How many factors (states) the batch's device buffers hold.
     - How many of the *first* factors (states) the next solve uses.
   * - Set by
     - The constructor. Fixed for the batch's lifetime.
     - ``SetNumActiveFactors(n)`` /
       ``SetNumActiveStates(n, num_const_states)``, any ``n <= Capacity()``.
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

.. code-block:: python

   pnp = pycunls.PnPFactorBatch(obs_gpu, pts_gpu, max_points)   # capacity
   pnp.set_num_active_factors(num_points)                        # active count

**Rules.**

- Every buffer bound to a batch must hold its **capacity**: measurements,
  state values, constant ids, connectivity tables.
- Only the **active** part is read and written by a solve: factors
  ``[0, NumActiveFactors())``, states ``[0, NumActiveStates())``, the first
  ``num_const_states`` constant ids. Results are written back to the active
  states only.
- Factors may only reference **active** states. State addresses
  (``StateDevicePtr(i)``) are valid for every ``i < Capacity()``, so the
  connectivity of the next solve can be built before the counts are set.
- Connectivity must cover the active factors: a host pointer list needs at
  least ``NumActiveFactors() * B`` entries (``Problem::SetStatePointers``
  replaces it); device tables are read for their first
  ``NumActiveFactors() * B`` entries.
- Never change sizes or buffer contents while a solve that uses them runs.
- Every minimizer checks the sizes at the start of ``Minimize``
  (``Problem::CheckSizes``, host-only): nothing active, a count above its
  capacity, or connectivity shorter than the active factors throws with an
  explanatory message.

How connectivity is rewritten between solves (host lists, device pointer
tables, device index tables) is described in
``docs/design/reusable_buffers.md`` and in the ``Problem`` API reference
(:doc:`api/minimizer`).

===============================================================================
High-level solve flow
===============================================================================

1. Allocate state and measurement data on the GPU, sized for the largest
   problem (the capacity).
2. Wrap state memory in one or more `StateBatch` objects (constructed with
   their capacity).
3. Build one or more `FactorBatch` objects from observations (constructed
   with their capacity).
4. Set the active counts: ``SetNumActiveStates`` / ``SetNumActiveFactors``
   (see :ref:`capacity-and-active-count`).
5. Add state batches and factor batches to a `Problem`.
6. Run a minimizer and inspect `MinimizerSummary`. To solve the next problem,
   rewrite the buffers, set the new active counts, and solve again.

===============================================================================
Supported optimization patterns
===============================================================================

- Pose graph optimization with between factors.
- Bundle-adjustment style reprojection optimization.
- ICP-like alignment (point-to-point / point-to-plane factors).
- Custom user-defined factors through `FactorBatch` / `SizedFactorBatch`.

See :doc:`tutorial` for complete C++ working pipelines, :doc:`pycunls_tutorial`
for Python examples, and :doc:`api/index` for class-level API details.
