################################################################################
Factor API
################################################################################

The factor module provides batched residual and Jacobian models for non-linear
least squares. Each factor computes a residual vector from one or more
**states**. States lie on **manifolds**; the factor API uses
**tangent-space dimensions** for Jacobian layout and solver variables. This
page documents the Python factor classes first, then the C++ API, which holds
the full residual and Jacobian formulas. In the C++ API, links to the
corresponding state batch types are in the :ref:`factor-inputs` section and in
each factor’s **Inputs** subsection.

- **Python** — ``pycunls``
- **C++** — ``cunls/factor``

================================================================================
Python API (``pycunls``)
================================================================================

All Python factor batches inherit from the abstract ``FactorBatch`` base
class.  Every constructor argument documented as ``DevicePointer`` accepts
either a ``cupy.ndarray`` (the device pointer is extracted automatically via
``.data.ptr``) or a raw ``int`` GPU device address.

The residual formulas, Jacobian structure, and state layouts are identical to
the C++ versions documented in :ref:`factor-cpp-api` below (each Python entry
links to its C++ counterpart) — this section focuses on the Python
constructor signatures, methods, and properties.

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``set_num_active_factors(n)`` /
   ``set_num_active_states(n)`` (C++: ``SetNumActiveFactors`` /
   ``SetNumActiveStates``) before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

.. _py-factor-batch-interface:

Common ``FactorBatch`` interface
--------------------------------------------------------------------------------

Every factor batch — built-in or user-defined — exposes the following
read-only properties and methods.

**Read-only properties**

- **num_active_factors** (``int``) — number of active factors (0 until
  ``set_num_active_factors``; at most ``capacity``).
- **capacity** (``int``) — number of factors the measurement buffers hold
  (the constructor's ``capacity``); constant.
- **residuals_size** (``int``) — residual dimension per factor (e.g. 2 for
  ``ReprojectionFactorBatch``, 6 for ``SE3BetweenFactorBatch``).

**Methods**

- ``set_num_active_factors(num_active_factors)`` — sets the active factor
  count (at most ``capacity``). Every batch starts with 0 active factors: call it before
  the first solve, and again whenever the count changes. Host-only; takes
  effect at the next ``minimize``. Raises ``ValueError`` above the capacity.
- ``state_sizes() -> list[int]`` — returns a list of tangent-space
  dimensions for each state consumed by one factor.  For example,
  ``ReprojectionFactorBatch`` returns ``[6, 3]`` (SE(3) pose then
  :math:`\mathbb{R}^3` point), and ``PnPFactorBatch`` returns ``[6]`` (pose
  only; 3-D points are fixed in the constructor).

**C++ reference:** :ref:`cpp-factor-batch`.

.. _py-prior-vector-factor:

``pycunls.PriorVectorFactorBatch1`` / ``PriorVectorFactorBatch2`` / ``PriorVectorFactorBatch3`` / ``PriorVectorFactorBatch6``
------------------------------------------------------------------------------------------------------------------------------

Prior on a Euclidean vector.  Residual = :math:`x - o` with identity
Jacobian.  The suffix indicates the dimension.

**Constructor**

.. code-block:: python

   fb = pycunls.PriorVectorFactorBatch3(observations, capacity)

- **observations** (``DevicePointer``) — contiguous GPU buffer of
  ``capacity × Dim`` floats holding the observed (target) vectors.  The
  factor batch does **not** copy the data; the caller must keep the
  allocation alive.
- **capacity** (``int``) — number of prior factors the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from the corresponding
``VectorStateBatch`` (see :ref:`py-vector-state-batches`).

**C++ reference:** :ref:`cpp-prior-vector-factor-batch`.

.. _py-so2-prior-factor:

``pycunls.SO2PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 2-D rotation.  Residual = :math:`\mathrm{Log}(R_\mathrm{target}^\top R)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SO2PriorFactorBatch(observations, capacity)

- **observations** (``DevicePointer``) — ``capacity × 4`` floats holding
  row-major 2×2 target rotation matrices.
- **capacity** (``int``) — number of prior factors the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from ``SO2StateBatch``
(see :ref:`py-lie-state-batches`).

**C++ reference:** :ref:`cpp-so2-prior-factor-batch`.

.. _py-so3-prior-factor:

``pycunls.SO3PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 3-D rotation.  Residual =
:math:`\mathrm{Log}(R_\mathrm{target}^\top R)`, Jacobian =
:math:`J_r^{-1}(r)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SO3PriorFactorBatch(observations, capacity)

- **observations** (``DevicePointer``) — ``capacity × 9`` floats holding
  row-major 3×3 target rotation matrices.
- **capacity** (``int``) — number of prior factors the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from ``SO3StateBatch``.

**C++ reference:** :ref:`cpp-so3-prior-factor-batch`.

.. _py-se3-prior-factor:

``pycunls.SE3PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 3-D rigid transform.  Residual =
:math:`\mathrm{Log}(T_\mathrm{target}^{-1} T)`, Jacobian = :math:`J_r^{-1}(r)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SE3PriorFactorBatch(observations, capacity)

- **observations** (``DevicePointer``) — ``capacity × 16`` floats
  holding row-major 4×4 target homogeneous matrices.
- **capacity** (``int``) — number of prior factors the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

**C++ reference:** :ref:`cpp-se3-prior-factor-batch`.

.. _py-sl4-prior-factor:

``pycunls.SL4PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on an SL(4) transform.  Residual =
:math:`\mathrm{Log}(T_\mathrm{target}^{-1} T)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SL4PriorFactorBatch(observations, capacity)

- **observations** (``DevicePointer``) — ``capacity × 16`` floats
  holding row-major 4×4 SL(4) target transforms.
- **capacity** (``int``) — number of prior factors the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from ``SL4StateBatch``.

**C++ reference:** :ref:`cpp-sl4-prior-factor-batch`.

.. _py-se3-between-factor:

``pycunls.SE3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative pose between two SE(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SE3BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[T_left, T_right]`` — both from
:ref:`SE3StateBatch <py-lie-state-batches>`.  The state-pointer list must
therefore contain ``2 × num_active_factors`` entries.

**C++ reference:** :ref:`cpp-se3-between-factor-batch`.

.. _py-se2-between-factor:

``pycunls.SE2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two SE(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SE2BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 9`` floats holding
  row-major 3×3 measured relative transforms :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[T_left, T_right]`` — both from
``SE2StateBatch``.

**C++ reference:** :ref:`cpp-se2-between-factor-batch`.

.. _py-so2-between-factor:

``pycunls.SO2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative rotation between two SO(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^\top R_l^\top R_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SO2BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 4`` floats holding
  row-major 2×2 measured relative rotations :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[R_left, R_right]`` — both from
``SO2StateBatch``.

**C++ reference:** :ref:`cpp-so2-between-factor-batch`.

.. _py-so3-between-factor:

``pycunls.SO3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative rotation between two SO(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^\top R_l^\top R_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SO3BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 9`` floats holding
  row-major 3×3 measured relative rotations :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[R_left, R_right]`` — both from
``SO3StateBatch``.

**C++ reference:** :ref:`cpp-so3-between-factor-batch`.

.. _py-similarity2-between-factor:

``pycunls.Similarity2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two Sim(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.Similarity2BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 9`` floats holding
  row-major 3×3 measured relative transforms :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[T_left, T_right]`` — both from
``Similarity2StateBatch``.

**C++ reference:** :ref:`cpp-similarity2-between-factor-batch`.

.. _py-similarity3-between-factor:

``pycunls.Similarity3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two Sim(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.Similarity3BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[T_left, T_right]`` — both from
``Similarity3StateBatch``.

**C++ reference:** :ref:`cpp-similarity3-between-factor-batch`.

.. _py-sl4-between-factor:

``pycunls.SL4BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two SL(4) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two states per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SL4BetweenFactorBatch(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta` (unit determinant).
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor — ``[T_left, T_right]`` — both from
``SL4StateBatch``.

**C++ reference:** :ref:`cpp-sl4-between-factor-batch`.

.. _py-vector-between-factor:

``pycunls.VectorBetweenFactorBatch1`` / ``VectorBetweenFactorBatch2`` / ``VectorBetweenFactorBatch3`` / ``VectorBetweenFactorBatch6``
--------------------------------------------------------------------------------------------------------------------------------------

Between factor on Euclidean vectors.  Residual =
:math:`x_l - x_r - \delta`.  Two states per factor.

**Constructor**

.. code-block:: python

   fb = pycunls.VectorBetweenFactorBatch3(deltas, capacity)

- **deltas** (``DevicePointer``) — ``capacity × Dim`` floats holding
  the measured difference vectors :math:`\delta`.
- **capacity** (``int``) — number of between constraints the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** two states per factor from the corresponding
``VectorStateBatch`` (see :ref:`py-vector-state-batches`).

**C++ reference:** :ref:`cpp-vector-between-factor-batch`.

.. _py-reprojection-factor:

``pycunls.ReprojectionFactorBatch``
--------------------------------------------------------------------------------

Reprojection error for bundle adjustment.  Observations must be in
**normalized image coordinates** (intrinsic calibration already applied):
:math:`r = \pi(T, P) - z`.

**Constructor**

.. code-block:: python

   fb = pycunls.ReprojectionFactorBatch(
       observations, capacity, z_threshold=1e-3)

- **observations** (``DevicePointer``) — ``capacity × 2`` floats
  holding normalized 2-D observations :math:`(x_n, y_n)`.
- **capacity** (``int``) — number of reprojection factors the buffers hold; 0 are active until ``set_num_active_factors``.
- **z_threshold** (``float``, default ``1e-3``) — minimum valid depth
  :math:`z` in camera frame.  Points with :math:`z < z_\text{threshold}`
  produce zero residuals and Jacobians to avoid singularities.

**State layout:** two states per factor — ``[SE3 pose, R^3 point]`` — from
:ref:`SE3StateBatch <py-lie-state-batches>` and
:ref:`VectorStateBatch3 <py-vector-state-batches>` respectively.

**C++ reference:** :ref:`cpp-reprojection-factor-batch`.

.. _py-pnp-factor:

``pycunls.PnPFactorBatch``
--------------------------------------------------------------------------------

PnP-style reprojection: **fixed** 3-D points in the constructor, **one** SE(3)
state per correspondence (typically the same camera pose pointer repeated).

**Constructor (identity camera-from-rig)**

.. code-block:: python

   fb = pycunls.PnPFactorBatch(
       observations, points_world, capacity, z_threshold=1e-3)

**Constructor (with camera-from-rig extrinsics per factor)**

.. code-block:: python

   fb = pycunls.PnPFactorBatch(
       observations, poses_camera_from_rig, points_world,
       capacity, z_threshold=1e-3)

- **observations** — ``capacity × 2`` normalized image coordinates.
- **points_world** — ``capacity × 3`` fixed world points (not
  optimized).
- **poses_camera_from_rig** — ``capacity × 16`` row-major SE(3)
  matrices (optional overload).
- **z_threshold** — minimum valid depth in the camera frame (same role as
  :ref:`ReprojectionFactorBatch <py-reprojection-factor>`).

**State layout:** one ``SE3StateBatch`` state per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

**C++ reference:** :ref:`cpp-pnp-factor-batch`.

.. _py-icp-factors:

``pycunls.PointToPointFactorBatch``
--------------------------------------------------------------------------------

Point-to-point ICP factor.  Residual = :math:`p - T q`.

**Constructor**

.. code-block:: python

   fb = pycunls.PointToPointFactorBatch(p_observations, q_observations, capacity)

- **p_observations** (``DevicePointer``) — ``capacity × 3`` floats
  holding target points :math:`p`.
- **q_observations** (``DevicePointer``) — ``capacity × 3`` floats
  holding source points :math:`q`.
- **capacity** (``int``) — number of point correspondences the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

**C++ reference:** :ref:`cpp-point-to-point-factor-batch`.

``pycunls.PointToPlaneFactorBatch``
--------------------------------------------------------------------------------

Point-to-plane ICP factor.  Residual = :math:`n_q^\top (p - T q)`.

**Constructor**

.. code-block:: python

   fb = pycunls.PointToPlaneFactorBatch(
       p_observations, q_observations, nq_observations, capacity)

- **p_observations** (``DevicePointer``) — target points (``× 3`` floats).
- **q_observations** (``DevicePointer``) — source points (``× 3`` floats).
- **nq_observations** (``DevicePointer``) — source normals (``× 3`` floats).
- **capacity** (``int``) — number of correspondences the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

**C++ reference:** :ref:`cpp-point-to-plane-factor-batch`.

``pycunls.SymmetricPointToPlaneFactorBatch``
--------------------------------------------------------------------------------

Symmetric point-to-plane ICP factor.  Both frames contribute normals;
:math:`N = n_p + n_q`.

**Constructor**

.. code-block:: python

   fb = pycunls.SymmetricPointToPlaneFactorBatch(
       p_observations, q_observations,
       np_observations, nq_observations, capacity)

- **p_observations** (``DevicePointer``) — target points (``× 3`` floats).
- **q_observations** (``DevicePointer``) — source points (``× 3`` floats).
- **np_observations** (``DevicePointer``) — target normals (``× 3`` floats).
- **nq_observations** (``DevicePointer``) — source normals (``× 3`` floats).
- **capacity** (``int``) — number of correspondences the buffers hold; 0 are active until ``set_num_active_factors``.

**State layout:** one state per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

**C++ reference:** :ref:`cpp-symmetric-point-to-plane-factor-batch`.

.. _py-information-factor-batch:

``pycunls.InformationFactorBatch``
--------------------------------------------------------------------------------

Wraps **any** factor batch and left-multiplies residuals and Jacobians by
per-factor square-root information matrices
:math:`\Omega^{1/2}`.  Unlike the C++ template, the Python class accepts any
``FactorBatch`` — no template specialization is needed.

**C++ contrast:** The C++ template ``InformationFactorBatch<T>`` also inherits
``T::sized_layout`` (a ``SizedFactorBatch`` with the same compile-time layout as
``T``). The Python wrapper is a dynamic ``FactorBatch`` only.

**Constructor**

.. code-block:: python

   info_fb = pycunls.InformationFactorBatch(
       inner_factor, sqrt_information_matrices)

- **inner_factor** (``FactorBatch``) — the factor batch to wrap.  The wrapper
  delegates ``Evaluate`` to this factor first, then applies the information
  matrices.  The inner factor must be kept alive for the lifetime of the
  wrapper.
- **sqrt_information_matrices** (``DevicePointer``) —
  ``capacity × residual_size × residual_size`` contiguous floats holding
  one row-major square-root information matrix per factor.

**Example**

.. code-block:: python

   inner = pycunls.SE3BetweenFactorBatch(deltas, N)
   info  = pycunls.InformationFactorBatch(inner, sqrt_info_gpu)
   info.set_num_active_factors(N)  # forwarded to inner

   problem.add_factor_batch(info, state_pointers)

**C++ reference:** :ref:`cpp-information-factor-batch`.

.. _py-weighted-factor-batch:

``pycunls.WeightedFactorBatch``
--------------------------------------------------------------------------------

Wraps **any** factor batch and scales residuals and Jacobians by a scalar
weight.  Two construction modes are supported:

1. **Uniform weight** (``float``) — the same scalar is applied to every factor.
2. **Per-factor weights** (``DevicePointer``) — one weight per factor from a
   GPU array.

**C++ contrast:** ``WeightedFactorBatch<T>`` inherits ``T::sized_layout``; the
Python wrapper subclasses ``FactorBatch`` only.

**Constructors**

.. code-block:: python

   # Uniform weight
   wfb = pycunls.WeightedFactorBatch(inner_factor, weight=2.0)

   # Per-factor weights
   wfb = pycunls.WeightedFactorBatch(inner_factor, weights=weights_gpu)

- **inner_factor** (``FactorBatch``) — the factor batch to wrap.
- **weight** (``float``) — uniform scalar weight applied to all factors.
- **weights** (``DevicePointer``, keyword-only) — ``inner_factor.capacity`` contiguous
  floats, one weight per factor.

Exactly one of ``weight`` or ``weights`` must be provided.

**Example**

.. code-block:: python

   inner = pycunls.PriorVectorFactorBatch3(obs_gpu, N)
   wfb   = pycunls.WeightedFactorBatch(inner, weight=5.0)
   wfb.set_num_active_factors(N)  # forwarded to inner

   problem.add_factor_batch(wfb, state_pointers)

**C++ reference:** :ref:`cpp-weighted-factor-batch`.

.. _py-custom-factor-batch:

``pycunls.CustomFactorBatch``
--------------------------------------------------------------------------------

Base class for user-defined factors.  Subclass this to implement a residual
and Jacobian computation that is not available as a built-in factor.

**Constructor**

.. code-block:: python

   class MyFactor(pycunls.CustomFactorBatch):
       def __init__(self, capacity):
           super().__init__(
               residual_size=...,
               state_sizes=[...],
               capacity=capacity,
           )

- **residual_size** (``int``) — dimension of the residual vector per
  factor.
- **state_sizes** (``Sequence[int]``) — list of tangent-space
  dimensions for each state consumed by one factor (e.g. ``[1, 1]``
  for a factor reading two scalar states).
- **capacity** (``int``) — number of factor instances the buffers hold; 0 are active until ``set_num_active_factors``.

**Methods to override**

- ``evaluate(residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
  factor_ids_ptr, num_factor_ids) -> bool``
  — computes residuals and Jacobians on the GPU for ``n = num_factor_ids``
  *items* (the same contract as C++ :cpp:func:`FactorBatch::Evaluate`): item
  *t* is factor ``f(t)`` evaluated at its own state pointers. All six
  arguments are raw ``int`` values:

  - *residuals_ptr* — device pointer to the output residual buffer.
    Layout: ``n × residual_size`` contiguous floats; item *t* writes row *t*.
  - *jacobians_ptr* — device pointer to the output Jacobian buffer.
    Layout: ``n × residual_size × sum(state_sizes)``
    contiguous floats (row-major per item, blocks concatenated in state
    order).  May be ``0`` (null) when the minimizer only needs residuals
    (e.g. for cost evaluation); in that case skip Jacobian writes.
  - *state_pointers_ptr* — device pointer to an array of ``float*``
    pointers.  The array has ``n × len(state_sizes)``
    entries; item *t*'s state *b* is entry ``t * len(state_sizes) + b``
    (the device address of that state's ambient-space storage).  Because Warp
    kernels cannot perform ``float**`` double-pointer indirection, custom
    factors typically gather state values into contiguous CuPy arrays
    before launching a kernel (see the
    :ref:`Custom Warp Factor tutorial <pycunls_tutorial:Custom Warp Factor>`).
  - *stream_handle* — ``cudaStream_t`` cast to ``int``.  All GPU work
    **must** be launched on this stream.
  - *factor_ids_ptr* — device pointer to ``n`` ``int32`` factor indices in
    ``[0, num_active_factors)`` with ``f(t) = factor_ids[t]``, or ``0`` (null)
    for ``f(t) = t % num_active_factors``. Read *measurements* through ``f(t)``;
    everything else (state pointers, outputs) is indexed by *t*.
  - *num_factor_ids* — the item count ``n``. Unlike C++, where ``0`` means
    ``num_active_factors``, Python always receives the actual count (``> 0``).

  The regular minimizers call ``evaluate`` with ``factor_ids_ptr == 0`` and
  ``num_factor_ids == num_active_factors``; the RANSAC minimizers evaluate many items
  per factor, so a factor used with RANSAC must honor both arguments (see
  :doc:`../custom_factors_and_states`).

  Return ``True`` on success.  The default implementation raises
  ``NotImplementedError``.

**Skipping the Jacobian entirely.** ``evaluate`` only has to write to
``jacobians_ptr`` when it is non-zero and you intend to supply an analytic
Jacobian. A custom factor that never writes to it — even when
``jacobians_ptr`` is non-zero — still satisfies the contract, and can be
registered with ``jacobian_mode_override=pycunls.JacobianMode.numeric`` in
:py:meth:`Problem.add_factor_batch` to have cuNLS differentiate it via
finite differences instead. See :doc:`../numeric_jacobians` for details and
:ref:`pycunls_tutorial:Warp factor code walkthrough` for a worked
Python example.

**C++ reference:** :ref:`cpp-factor-batch`.

.. _py-warp-factor-batch:

``pycunls.warp.WarpFactorBatch``
--------------------------------------------------------------------------------

Convenience base for custom factors implemented with `NVIDIA Warp
<https://developer.nvidia.com/warp-python>`_ kernels.
Inherits from ``CustomFactorBatch`` and provides helper methods for
zero-copy pointer wrapping so you never need to manually construct
``wp.array`` objects from raw device addresses.  Requires ``warp-lang``.

**Constructor**

.. code-block:: python

   from pycunls.warp import WarpFactorBatch

   class MyWarpFactor(WarpFactorBatch):
       def __init__(self, capacity):
           super().__init__(
               residual_size=...,
               state_sizes=[...],
               capacity=capacity,
               device="cuda:0",
           )

- **device** (``str``, default ``"cuda:0"``) — Warp device string used when
  creating ``wp.array`` wrappers via ``wrap_array``.

**Helper methods** (inherited — do not override)

- ``wrap_array(ptr: int, dtype, shape) -> wp.array`` — zero-copy wrap of an
  existing GPU allocation as a Warp array.  *ptr* is the device address,
  *dtype* a Warp data type (e.g. ``wp.float32``), and *shape* an ``int`` or
  tuple giving the array dimensions.  The returned ``wp.array`` shares the
  memory; no allocation or copy occurs.

- ``factor_ids(factor_ids_ptr: int, num_items: int) -> wp.array`` — the
  factor index of every item as an ``int32`` Warp array: wraps
  ``factor_ids_ptr`` when it is non-null, otherwise returns (and caches)
  ``arange(num_items) % num_active_factors``. Kernels can then always read
  ``ids[t]``.

- ``make_warp_stream(stream_handle: int) -> wp.Stream`` — wraps a raw
  ``cudaStream_t`` (passed as ``int``) as a ``wp.Stream``.  Use the
  returned stream in ``wp.launch(..., stream=stream)`` to ensure the Warp
  kernel executes on the minimizer's CUDA stream.

**Methods to override**

- ``evaluate(residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
  factor_ids_ptr, num_factor_ids) -> bool``
  — same contract as ``CustomFactorBatch.evaluate``.  Typical
  implementations:

  1. Gather scattered state pointers into contiguous CuPy arrays (using a
     CuPy ``RawKernel`` or ``cp.ndarray`` indexing), one entry per item.
  2. Get per-item factor indices with ``self.factor_ids(factor_ids_ptr,
     num_factor_ids)`` and read measurements through them.
  3. Wrap the contiguous arrays and output buffers with
     ``self.wrap_array``.
  4. Build a ``wp.Stream`` with ``self.make_warp_stream``.
  5. Launch a ``@wp.kernel`` with ``dim=num_factor_ids`` on that stream.

See :ref:`pycunls_tutorial:Custom Warp Factor` for a complete example.

.. _factor-cpp-api:

================================================================================
C++ API
================================================================================

.. _factor-inputs:

Factor inputs
-------------

Each factor's **Inputs** subsection below and the :doc:`state` API list the
required state batch types (e.g. :code:`VectorStateBatch<Dim>`, :code:`SO3StateBatch`).

.. _cpp-factor-batch:

FactorBatch
-----------

Abstract base (:code:`cunls/factor/factor_batch.h`).

.. math::
   r = f(x),\qquad J = \frac{\partial f}{\partial x}

.. cpp:function:: bool Evaluate(float* residuals, float* jacobians, float const* const* state_pointers, cudaStream_t stream, const int* factor_ids = nullptr, size_t num_factor_ids = 0) const

  Evaluates residuals and, optionally, Jacobians for a list of **items**.

  **Terms.** :math:`N` = ``NumActiveFactors()`` (factors/measurements in the batch),
  :math:`B` = ``StateSizes().size()`` (states per factor),
  :math:`m` = ``ResidualsSize()``, :math:`J` = sum of ``StateSizes()``
  (Jacobian columns per factor). An *item* is one factor evaluated at one set
  of :math:`B` states. The call evaluates :math:`n` items
  :math:`t = 0 \ldots n-1`; item :math:`t` reads the measurement of factor
  :math:`f(t)` and its own state pointers, and writes its own output rows.

  With the default arguments item :math:`t` is simply factor :math:`t`
  (:math:`n = N`, :math:`f(t) = t`) — exactly the behavior of the classic
  4-argument call, used by the regular minimizers. The last two arguments let
  one call evaluate the same factors at many state sets; the
  :doc:`RANSAC minimizers <../ransac>` evaluate every hypothesis this way.

  :param ``residuals``: [out] Device array of :math:`n \cdot m` floats. Item
    :math:`t` writes ``residuals[t * m + r]`` for :math:`r \in [0, m)`.
  :param ``jacobians``: [out] Device array of :math:`n \cdot m \cdot J` floats,
    or ``nullptr`` when only residuals are needed. Item :math:`t` writes a
    row-major :math:`m \times J` block; element :math:`(r, c)` is
    ``jacobians[(t * m + r) * J + c]``. Columns follow the states in
    order, each contributing its tangent size.
  :param ``state_pointers``: [in] Device array of :math:`n \cdot B` device
    pointers. Item :math:`t` reads state :math:`b` from
    ``state_pointers[t * B + b]``. Different items may point to the same state
    (e.g. every PnP factor points to the one camera pose).
  :param ``stream``: [in] CUDA stream on which all work is enqueued; the call
    may return before the work completes.
  :param ``factor_ids``: [in] Which factor each item evaluates. ``nullptr``
    (default): :math:`f(t) = t \bmod N` — with :math:`n = kN` this evaluates
    the whole batch :math:`k` times, copy :math:`c` being items
    :math:`[cN, (c+1)N)`. Otherwise a device array of :math:`n` indices in
    :math:`[0, N)` with :math:`f(t) =` ``factor_ids[t]`` (any order, repeats
    allowed).
  :param ``num_factor_ids``: [in] Number of items :math:`n`; ``0`` (default)
    means :math:`n = N`. When ``factor_ids`` is given it is that array's length.
  :returns: [out] ``true`` on success.

  **Examples** (:math:`N = 3`, :math:`B = 1`, :math:`m = 2`; ``ptrs[t]`` is the
  state pointer of item :math:`t`, ``Pk`` the state of set P for factor k,
  ``res rows`` the range of ``residuals`` item :math:`t` writes):

  .. code-block:: text

     1. Plain evaluation: Evaluate(res, jac, ptrs, stream)        n = 3
          item t          0     1     2
          factor f(t)     0     1     2
          ptrs[t]         x0    x1    x2
          res rows        [0,2) [2,4) [4,6)

     2. Whole batch at two state sets P and Q:
        Evaluate(res, jac, ptrs, stream, nullptr, 6)              n = 6
          item t          0     1     2     3     4     5
          factor f(t)     0     1     2     0     1     2      (t % 3)
          ptrs[t]         P0    P1    P2    Q0    Q1    Q2
          res rows        [0,2) [2,4) [4,6) [6,8) [8,10) [10,12)

     3. Chosen factors: ids = {2, 0, 2, 1} (device array)
        Evaluate(res, jac, ptrs, stream, ids, 4)                  n = 4
          item t          0     1     2     3
          factor f(t)     2     0     2     1
          ptrs[t]         P     P     Q     Q
          res rows        [0,2) [2,4) [4,6) [6,8)

  **Requirements for implementations.** Launch one thread per item; index
  *measurements* by :math:`f(t)` and *everything else* (state pointers,
  outputs) by :math:`t`. Item :math:`t` must produce exactly what a plain
  evaluation produces for factor :math:`f(t)` at item :math:`t`'s states (all
  built-in batches are bitwise equal). Size any internal per-factor scratch
  for :math:`n` items, not :math:`N`. Do not assume :math:`n = N` or
  :math:`f(t) = t`. See :doc:`../custom_factors_and_states` for a complete
  walkthrough (C++ and Python).

.. cpp:function:: size_t ResidualsSize() const

  :returns: [out] Residual dimension per factor.

.. cpp:function:: std::vector<size_t> StateSizes() const

  :returns: [out] State **tangent** dimensions consumed by each factor.

.. _factor-active-size:

.. cpp:function:: size_t NumActiveFactors() const

  :returns: [out] Number of active factors: the first ``NumActiveFactors()``
    measurements are used. 0 after construction, until ``SetNumActiveFactors``.

.. cpp:function:: size_t Capacity() const

  :returns: [out] Number of factors the measurement buffers hold: the
    ``capacity`` passed to the constructor. Constant for the batch's lifetime.
    A custom batch constructed without a capacity (one that overrides
    ``NumActiveFactors()`` instead) has a fixed size, and its capacity is
    ``NumActiveFactors()``.

.. cpp:function:: void SetNumActiveFactors(size_t num_active_factors)

  Sets the active factor count. Every batch starts with 0 active factors:
  call this before the first solve, and again whenever the count changes
  (e.g. per frame, after rewriting the measurement buffers in place).
  Host-only (no allocation, no device work); takes effect at the next
  ``Minimize``. Wrappers (``InformationFactorBatch``,
  ``WeightedFactorBatch``) forward it to the wrapped batch.

  :param ``num_active_factors``: [in] Active count, at most ``Capacity()``.
  :throws std::invalid_argument: if ``num_active_factors > Capacity()``.
  :throws std::logic_error: if a subclass overrides ``NumActiveFactors()``, so the
    set would have no effect. Custom batches pass their capacity to
    ``SizedFactorBatch(capacity)`` instead.

**Residual-only factors.** ``Evaluate`` must support ``jacobians == nullptr``
(residual-only evaluation) — this is required for cost-only evaluation, and
it is also all that's needed to opt a factor into cuNLS's numeric
(finite-difference) Jacobians: a factor whose ``Evaluate`` never writes to
``jacobians`` at all still satisfies this interface, and can be solved by
registering it with ``JacobianMode::kNumeric`` (see
:doc:`../numeric_jacobians` and :ref:`minimizer-jacobian-mode-label`) instead
of implementing a Jacobian by hand.

.. _cpp-sized-factor-batch:

SizedFactorBatch<kResidualSize, ...kStateSizes>
----------------------------------------------------

Compile-time convenience base (:code:`cunls/factor/sized_factor_batch.h`) that fixes
residual and state (tangent) dimensions at compile time.

Each specialization exposes **sized_layout** — an alias for the same
``SizedFactorBatch<kResidualSize, kStateSizes...>`` type. Wrapper templates
such as ``InformationFactorBatch<T>`` and ``WeightedFactorBatch<T>`` inherit
``public T::sized_layout`` so they remain full ``SizedFactorBatch`` instances with
the same layout as the inner batch ``T``.

.. _cpp-prior-vector-factor-batch:

PriorVectorFactorBatch<Dim>
----------------------------

Header: :code:`cunls/factor/prior/prior_vector_factor_batch.h`

Prior on a Euclidean vector (e.g. bias, landmark). Pulls the state toward observed values.

.. list-table::
   :header-rows: 1
   :widths: 25 14 25 18 15

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = x - o`
     - :math:`\mathrm{Dim}`
     - :math:`I`
     - :math:`\mathrm{Dim} \times \mathrm{Dim}`
     - :math:`\mathbb{R}^{\mathrm{Dim}}`

**Inputs:** :math:`x` = state vector, :math:`o` = observation (constructor). State: one state from :code:`VectorStateBatch<Dim>` (see :doc:`state`).

Constructor:

.. code-block:: cpp

   PriorVectorFactorBatch(const Vector<Dim>* observations_ptr, size_t capacity)

- ``observations_ptr`` — [in] Device pointer to observed vectors.
- ``capacity`` — [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.

.. _cpp-so2-prior-factor-batch:

SO2PriorFactorBatch
-------------------

Header: :code:`cunls/factor/prior/so2_prior_factor_batch.h`

Prior on a 2D rotation (e.g. heading). Penalizes deviation from a target rotation.

.. list-table::
   :header-rows: 1
   :widths: 25 14 25 18 15

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(R_{\mathrm{target}}^\top R)`
     - 1
     - :math:`1`
     - :math:`1 \times 1`
     - SO(2)

**Inputs:** :math:`R` = current rotation (state). State: one state from :code:`SO2StateBatch` (see :doc:`state`).

.. cpp:function:: SO2PriorFactorBatch(const Matrix<2>* observations_ptr, size_t capacity)

  :param ``observations_ptr``: [in] Device pointer to SO(2) observations (2×2 row-major).
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-so3-prior-factor-batch:

SO3PriorFactorBatch
-------------------

Header: :code:`cunls/factor/prior/so3_prior_factor_batch.h`

Prior on a 3D rotation. Penalizes deviation from a target orientation.

.. list-table::
   :header-rows: 1
   :widths: 25 14 25 18 15

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(R_{\mathrm{target}}^\top R)`
     - 3
     - :math:`J_r^{-1}(r)`
     - :math:`3 \times 3`
     - SO(3)

**Inputs:** :math:`R` = current rotation (state). State: one state from :code:`SO3StateBatch` (see :doc:`state`).

.. cpp:function:: SO3PriorFactorBatch(const Matrix<3>* observations_ptr, size_t capacity)

  :param ``observations_ptr``: [in] Device pointer to SO(3) observations (3×3 row-major).
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-se2-prior-factor-batch:

SE2PriorFactorBatch
-------------------

Header: :code:`cunls/factor/prior/se2_prior_factor_batch.h`

Prior on 2D rigid transform. State: one state from :code:`SE2StateBatch` (see :doc:`state`).

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(T_{\mathrm{target}}^{-1} T)`
     - 3
     - :math:`J_r^{-1}(r)`
     - :math:`3 \times 3`
     - SE(2)

.. _cpp-se3-prior-factor-batch:

SE3PriorFactorBatch
-------------------

Header: :code:`cunls/factor/prior/se3_prior_factor_batch.h`

Prior on 3D rigid transform. State: one state from :code:`SE3StateBatch` (see :doc:`state`).

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(T_{\mathrm{target}}^{-1} T)`
     - 6
     - :math:`J_r^{-1}(r)`
     - :math:`6 \times 6`
     - SE(3)

.. _cpp-similarity2-prior-factor-batch:

Similarity2PriorFactorBatch
---------------------------

Header: :code:`cunls/factor/prior/similarity2_prior_factor_batch.h`

Prior on 2D similarity transform. State: one state from :code:`Similarity2StateBatch` (see :doc:`state`).

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 14 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(T_{\mathrm{target}}^{-1} T)`
     - 4
     - :math:`J_r^{-1}(r)`
     - :math:`4 \times 4`
     - Sim(2)

.. _cpp-similarity3-prior-factor-batch:

Similarity3PriorFactorBatch
----------------------------

Header: :code:`cunls/factor/prior/similarity3_prior_factor_batch.h`

Prior on 3D similarity transform. State: one state from :code:`Similarity3StateBatch` (see :doc:`state`).

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 14 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(T_{\mathrm{target}}^{-1} T)`
     - 7
     - :math:`J_r^{-1}(r)`
     - :math:`7 \times 7`
     - Sim(3)

**Constructors (all four prior classes above):**

.. cpp:function:: ClassName(const ObsType* observations_ptr, size_t capacity)

  :param ``observations_ptr``: [in] Device pointer to observation transforms.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-sl4-prior-factor-batch:

SL4PriorFactorBatch
-------------------

Header: :code:`cunls/factor/prior/sl4_prior_factor_batch.h`

Prior on an SL(4) transform. State: one state from :code:`SL4StateBatch` (see :doc:`state`).

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`r = \mathrm{Log}(T_{\mathrm{target}}^{-1} T)`
     - 15
     - :math:`I`
     - :math:`15 \times 15`
     - SL(4)

.. cpp:function:: SL4PriorFactorBatch(const SL4Transform* observations_ptr, size_t capacity)

  :param ``observations_ptr``: [in] Device pointer to SL(4) target transforms (row-major 4×4).
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-se3-between-factor-batch:

SE3BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/between/se3_between_factor_batch.h`

Constrains the relative pose between two SE(3) frames (e.g. odometry, loop closure).

.. math::
   r = \mathrm{Log}\bigl( \Delta^{-1} \, T_{\mathrm{left}}^{-1} \, T_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 6
     - left/right SE(3) Jacobians
     - :math:`6 \times 12`
     - SE(3) × SE(3)

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two poses (states). State: two states from :code:`SE3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SE3BetweenFactorBatch(const SE3Transform* pose_deltas_ptr, size_t capacity)

  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms.
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-se2-between-factor-batch:

SE2BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/between/se2_between_factor_batch.h`

Constrains the relative transform between two SE(2) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{-1} \, T_{\mathrm{left}}^{-1} \, T_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 3
     - left/right SE(2) Jacobians
     - :math:`3 \times 6`
     - SE(2) × SE(2)

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two poses (states). State: two states from :code:`SE2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SE2BetweenFactorBatch(const Matrix<3>* pose_deltas_ptr, size_t capacity)

  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 3×3).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-so2-between-factor-batch:

SO2BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/between/so2_between_factor_batch.h`

Constrains the relative rotation between two SO(2) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{\top} \, R_{\mathrm{left}}^{\top} \, R_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 1
     - left/right SO(2) Jacobians
     - :math:`1 \times 2`
     - SO(2) × SO(2)

**Inputs:** :math:`R_{\mathrm{left}}`, :math:`R_{\mathrm{right}}` = two rotations (states). State: two states from :code:`SO2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative rotation (constructor).

.. cpp:function:: SO2BetweenFactorBatch(const Matrix<2>* rotation_deltas_ptr, size_t capacity)

  :param ``rotation_deltas_ptr``: [in] Device pointer to measured relative rotations (row-major 2×2).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-so3-between-factor-batch:

SO3BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/between/so3_between_factor_batch.h`

Constrains the relative rotation between two SO(3) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{\top} \, R_{\mathrm{left}}^{\top} \, R_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 3
     - left/right SO(3) Jacobians
     - :math:`3 \times 6`
     - SO(3) × SO(3)

**Inputs:** :math:`R_{\mathrm{left}}`, :math:`R_{\mathrm{right}}` = two rotations (states). State: two states from :code:`SO3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative rotation (constructor).

.. cpp:function:: SO3BetweenFactorBatch(const Matrix<3>* rotation_deltas_ptr, size_t capacity)

  :param ``rotation_deltas_ptr``: [in] Device pointer to measured relative rotations (row-major 3×3).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-similarity2-between-factor-batch:

Similarity2BetweenFactorBatch
-----------------------------

Header: :code:`cunls/factor/between/similarity2_between_factor_batch.h`

Constrains the relative transform between two Sim(2) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{-1} \, T_{\mathrm{left}}^{-1} \, T_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 4
     - left/right Sim(2) Jacobians
     - :math:`4 \times 8`
     - Sim(2) × Sim(2)

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (states). State: two states from :code:`Similarity2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: Similarity2BetweenFactorBatch(const Matrix<3>* pose_deltas_ptr, size_t capacity)

  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 3×3).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-similarity3-between-factor-batch:

Similarity3BetweenFactorBatch
-----------------------------

Header: :code:`cunls/factor/between/similarity3_between_factor_batch.h`

Constrains the relative transform between two Sim(3) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{-1} \, T_{\mathrm{left}}^{-1} \, T_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 7
     - left/right Sim(3) Jacobians
     - :math:`7 \times 14`
     - Sim(3) × Sim(3)

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (states). State: two states from :code:`Similarity3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: Similarity3BetweenFactorBatch(const Matrix<4>* pose_deltas_ptr, size_t capacity)

  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 4×4).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-sl4-between-factor-batch:

SL4BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/between/sl4_between_factor_batch.h`

Constrains the relative transform between two SL(4) frames.

.. math::
   r = \mathrm{Log}\bigl( \Delta^{-1} \, T_{\mathrm{left}}^{-1} \, T_{\mathrm{right}} \bigr)

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 12

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 15
     - left/right SL(4) Jacobians
     - :math:`15 \times 30`
     - SL(4) × SL(4)

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (states). State: two states from :code:`SL4StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SL4BetweenFactorBatch(const SL4Transform* pose_deltas_ptr, size_t capacity)

  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 4×4, unit determinant).
  :param ``capacity``: [in] Number of between constraints the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-vector-between-factor-batch:

VectorBetweenFactorBatch<Dim>
-----------------------------

Header: :code:`cunls/factor/between/vector_between_factor_batch.h`

Constrains the difference between two Euclidean vector states.

.. math::
   r = x_{\mathrm{left}} - x_{\mathrm{right}} - \delta

.. list-table::
   :header-rows: 1
   :widths: 25 14 25 18 15

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`x_l - x_r - \delta`
     - :math:`\mathrm{Dim}`
     - :math:`[I \;|\; {-}I]`
     - :math:`\mathrm{Dim} \times 2\mathrm{Dim}`
     - :math:`\mathbb{R}^{\mathrm{Dim}} \times \mathbb{R}^{\mathrm{Dim}}`

**Inputs:** :math:`x_l`, :math:`x_r` = two vector states. State: two states from :code:`VectorStateBatch<Dim>` (see :doc:`state`). :math:`\delta` = measured difference (constructor).

Constructor:

.. code-block:: cpp

   VectorBetweenFactorBatch(const Vector<Dim>* deltas_ptr, size_t capacity)

- ``deltas_ptr`` — [in] Device pointer to measured difference vectors.
- ``capacity`` — [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.

Manifold facade
----------------

The per-manifold classes above (``SE3BetweenFactorBatch``,
``SO2PriorFactorBatch``, ...) are the hand-optimized implementations. Two
zero-cost, manifold-generic facades let callers write fewer distinct class
names:

.. cpp:class:: template <class Manifold> BetweenFactorBatch

  Header: :code:`cunls/factor/between/between_factor_batch.h`. A compile-time
  alias for the matching ``XxxBetweenFactorBatch``: each specialization adds
  no data members or virtual dispatch, so
  ``sizeof(BetweenFactorBatch<manifold::SE3>) == sizeof(SE3BetweenFactorBatch)``
  and the two are interchangeable everywhere a ``FactorBatch*`` is expected.
  ``Manifold`` is one of the tags in ``cunls::manifold`` (``SE3``, ``SO3``,
  ``SE2``, ``SO2``, ``Similarity2``, ``Similarity3``, ``SL4``, or
  ``Vector<Dim>``) and is usually **deduced via CTAD** from the deltas
  pointer's own (now manifold-distinct) type, so ``<Manifold>`` need not be
  written explicitly:

  .. code-block:: cpp

     cunls::BetweenFactorBatch between(deltas_ptr, capacity);  // manifold deduced

  CTAD deduction is unavailable for ``Vector<Dim>`` (a C++ template-argument-deduction
  limitation: ``Vector``'s ``int Dim`` cannot be deduced from the ``size_t``
  extent of the underlying array type), so that one specialization requires
  ``<manifold::Vector<Dim>>`` explicitly.

.. cpp:class:: template <class Manifold> PriorFactorBatch

  Header: :code:`cunls/factor/prior/prior_factor_batch.h`. Same mechanism as
  ``BetweenFactorBatch<Manifold>``, for ``XxxPriorFactorBatch`` instead of
  ``XxxBetweenFactorBatch``; ``Manifold`` is deduced from the observations
  pointer's type.

.. cpp:class:: template <class Manifold> ConstantVelocityFactorBatch

  Header: :code:`cunls/factor/motion/constant_velocity_factor_batch.h`. Same
  zero-cost specialization mechanism, but **always requires the manifold as
  an explicit template argument**: every ``ConstantVelocityXxxFactorBatch``
  constructor is ``(const float* dt_ptr, size_t capacity)``, so there is
  no manifold-specific argument type to deduce from.

  .. code-block:: cpp

     cunls::ConstantVelocityFactorBatch<cunls::manifold::SE3> factor(dt_ptr, capacity);

.. cpp:class:: template <class Manifold> ConstantAccelerationFactorBatch

  Header: :code:`cunls/factor/motion/constant_acceleration_factor_batch.h`.
  Identical mechanism and explicit-template-argument-only convention as
  ``ConstantVelocityFactorBatch<Manifold>``.

Motion prior factors
---------------------

Constant-velocity (CV) and constant-acceleration (CA) motion-prior factors
constrain how a pose evolves between two consecutive timestamps, given an
explicit body-velocity (and, for CA, body-acceleration) state at each
timestamp. Pose, velocity, and acceleration are kept as **separate**
states (an ``SE3StateBatch``/``SO3StateBatch``/``SE2StateBatch``/
``SO2StateBatch`` for the pose, ``VectorStateBatch<Dim>`` for
velocity/acceleration, ``Dim`` matching the pose's tangent size) connected by
one of the factors below.

For a pose group with ``Log``/``Exp`` and inverse-left-Jacobian
:math:`J_l^{-1}`, and ``twist`` :math:`:= \mathrm{Log}(T_k^{-1} T_{k+1})`:

.. math::
   r_{\mathrm{pose}} &= \mathrm{twist} - \Delta t \, v_k \;\;(- \tfrac{1}{2}\Delta t^2 a_k \text{ for CA}) \\
   r_{\mathrm{vel}} &= J_l^{-1}(\mathrm{twist}) \, v_{k+1} - v_k \;\;(- \Delta t \, a_k \text{ for CA}) \\
   r_{\mathrm{accel}} &= J_l^{-1}(\mathrm{twist}) \, a_{k+1} - a_k \quad \text{(CA only)}

i.e. the relative pose should match a first- (CV) or second-order (CA)
Taylor prediction from the velocity/acceleration at :math:`k`, and the
velocity/acceleration at :math:`k+1`, transported back into the local frame
at :math:`k` through the inverse left Jacobian, should match the value at
:math:`k`. The Jacobians of :math:`r_{\mathrm{vel}}` and
:math:`r_{\mathrm{accel}}` with respect to the *pose* states are treated as
zero — a documented simplification (the residuals themselves are exact; only
that specific curvature term is dropped). SO(2) is abelian
(:math:`J_l^{-1} = 1`), so its factors reduce to scalar arithmetic; SE(2)
obtains :math:`J_l^{-1}` from the identity :math:`J_l^{-1}(x) = J_r^{-1}(-x)`
rather than a dedicated left-Jacobian primitive.

ConstantVelocitySE3FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_velocity_se3_factor_batch.h`

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 25

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}]`
     - 12
     - closed-form (see above)
     - :math:`12 \times 24`
     - SE(3) × SE(3) × :math:`\mathbb{R}^6` × :math:`\mathbb{R}^6`

**Inputs:** :math:`T_k, T_{k+1}` = two states from :code:`SE3StateBatch`; :math:`v_k, v_{k+1}` = two states from :code:`VectorStateBatch<6>` (body twist). :math:`\Delta t` (constructor).

.. cpp:function:: ConstantVelocitySE3FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantVelocitySO3FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_velocity_so3_factor_batch.h`. Same
construction as ``ConstantVelocitySE3FactorBatch``, specialized to SO(3)
(``vel`` = angular velocity in :math:`\mathbb{R}^3`).

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 25

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}]`
     - 6
     - closed-form (see above)
     - :math:`6 \times 12`
     - SO(3) × SO(3) × :math:`\mathbb{R}^3` × :math:`\mathbb{R}^3`

**Inputs:** :math:`R_k, R_{k+1}` = two states from :code:`SO3StateBatch`; :math:`v_k, v_{k+1}` = two states from :code:`VectorStateBatch<3>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantVelocitySO3FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantVelocitySE2FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_velocity_se2_factor_batch.h`. Same
construction, specialized to SE(2).

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 25

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}]`
     - 6
     - closed-form (see above)
     - :math:`6 \times 12`
     - SE(2) × SE(2) × :math:`\mathbb{R}^3` × :math:`\mathbb{R}^3`

**Inputs:** :math:`T_k, T_{k+1}` = two states from :code:`SE2StateBatch`; :math:`v_k, v_{k+1}` = two states from :code:`VectorStateBatch<3>` (body twist). :math:`\Delta t` (constructor).

.. cpp:function:: ConstantVelocitySE2FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantVelocitySO2FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_velocity_so2_factor_batch.h`. SO(2) is
abelian, so the residual/Jacobian reduce to scalar arithmetic.

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 25

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}]`
     - 2
     - closed-form scalars
     - :math:`2 \times 4`
     - SO(2) × SO(2) × :math:`\mathbb{R}` × :math:`\mathbb{R}`

**Inputs:** :math:`\theta_k, \theta_{k+1}` = two states from :code:`SO2StateBatch`; :math:`v_k, v_{k+1}` = two states from :code:`VectorStateBatch<1>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantVelocitySO2FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantAccelerationSE3FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_acceleration_se3_factor_batch.h`. Adds
an acceleration state to ``ConstantVelocitySE3FactorBatch``'s construction;
see the general formula above.

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 30

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}; r_{\mathrm{accel}}]`
     - 18
     - closed-form (see above)
     - :math:`18 \times 36`
     - SE(3) × SE(3) × :math:`(\mathbb{R}^6)^4`

**Inputs:** :math:`T_k, T_{k+1}` = two states from :code:`SE3StateBatch`; :math:`v_k, v_{k+1}, a_k, a_{k+1}` = four states from :code:`VectorStateBatch<6>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantAccelerationSE3FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantAccelerationSO3FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_acceleration_so3_factor_batch.h`. Same
construction, specialized to SO(3).

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 30

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}; r_{\mathrm{accel}}]`
     - 9
     - closed-form (see above)
     - :math:`9 \times 18`
     - SO(3) × SO(3) × :math:`(\mathbb{R}^3)^4`

**Inputs:** :math:`R_k, R_{k+1}` = two states from :code:`SO3StateBatch`; :math:`v_k, v_{k+1}, a_k, a_{k+1}` = four states from :code:`VectorStateBatch<3>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantAccelerationSO3FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantAccelerationSE2FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_acceleration_se2_factor_batch.h`. Same
construction, specialized to SE(2).

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 30

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}; r_{\mathrm{accel}}]`
     - 9
     - closed-form (see above)
     - :math:`9 \times 18`
     - SE(2) × SE(2) × :math:`(\mathbb{R}^3)^4`

**Inputs:** :math:`T_k, T_{k+1}` = two states from :code:`SE2StateBatch`; :math:`v_k, v_{k+1}, a_k, a_{k+1}` = four states from :code:`VectorStateBatch<3>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantAccelerationSE2FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

ConstantAccelerationSO2FactorBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/motion/constant_acceleration_so2_factor_batch.h`. SO(2)
is abelian, so the residual/Jacobian reduce to scalar arithmetic.

.. list-table::
   :header-rows: 1
   :widths: 18 14 28 15 30

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - :math:`[r_{\mathrm{pose}}; r_{\mathrm{vel}}; r_{\mathrm{accel}}]`
     - 3
     - closed-form scalars
     - :math:`3 \times 6`
     - SO(2) × SO(2) × :math:`\mathbb{R}^4`

**Inputs:** :math:`\theta_k, \theta_{k+1}` = two states from :code:`SO2StateBatch`; :math:`v_k, v_{k+1}, a_k, a_{k+1}` = four states from :code:`VectorStateBatch<1>`. :math:`\Delta t` (constructor).

.. cpp:function:: ConstantAccelerationSO2FactorBatch(const float* dt_ptr, size_t capacity)

  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

Motion prior covariance weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/factor/information/motion_prior_information.h`. The eight factors
above are unweighted (unit information) on their own. To fuse the paper's
closed-form process-noise covariance :math:`Q(\Delta t)^{-1}` into the
residual/Jacobian, construct the corresponding **named alias** below instead
of the plain factor — it is a drop-in replacement that takes two extra
arguments (a stream for the one-off precomputation and the process-noise PSD) and internally
computes the square-root information and composes it with the existing
:code:`InformationFactorBatch<T>` (see above), so callers never see the
Kronecker-product math or manage a separate information buffer:

.. code-block:: cpp

   ConstantVelocityInformationSE3FactorBatch factor(
       stream, dt_ptr, qc_diag_ptr, capacity);
   factor.SetNumActiveFactors(num_factors);  // active count, at most capacity

Available aliases (one per factor above, same residual/Jacobian shape as
the wrapped factor): :code:`ConstantVelocityInformationSE3FactorBatch`,
:code:`ConstantVelocityInformationSO3FactorBatch`,
:code:`ConstantVelocityInformationSE2FactorBatch`,
:code:`ConstantVelocityInformationSO2FactorBatch`,
:code:`ConstantAccelerationInformationSE3FactorBatch`,
:code:`ConstantAccelerationInformationSO3FactorBatch`,
:code:`ConstantAccelerationInformationSE2FactorBatch`,
:code:`ConstantAccelerationInformationSO2FactorBatch`.

.. cpp:class:: template <class T, int Dim> MotionPriorInformationFactorBatch

  The generic template all eight aliases instantiate; :code:`T` is the
  wrapped :code:`ConstantVelocityXxxFactorBatch`/
  :code:`ConstantAccelerationXxxFactorBatch` and :code:`Dim` its pose
  tangent size (6/3/3/1 for SE(3)/SO(3)/SE(2)/SO(2)). Prefer the named
  aliases; only spell this out directly for a factor/Dim combination that
  doesn't have one yet.

.. cpp:function:: MotionPriorInformationFactorBatch(cudaStream_t stream, const float* dt_ptr, const float* qc_diag_ptr, size_t capacity)

  :param ``stream``: [in] CUDA stream used to precompute the sqrt-information matrices at construction time.
  :param ``dt_ptr``: [in] Device pointer to per-factor time deltas; also forwarded to the wrapped factor's own constructor.
  :param ``qc_diag_ptr``: [in] Device pointer to the continuous-time process-noise PSD diagonal (``Dim`` floats), constant across the batch.
  :param ``capacity``: [in] Number of factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

The two function templates behind this wrapper,
:code:`ComputeConstantVelocitySqrtInformation<Dim>` and
:code:`ComputeConstantAccelerationSqrtInformation<Dim>`, are still available
directly (same header) for callers who want the raw square-root information
matrix without going through :code:`InformationFactorBatch` — it's a
Kronecker product with an analytic Cholesky factor, no numerical linear
algebra.

.. _cpp-reprojection-factor-batch:

ReprojectionFactorBatch
-----------------------

Header: :code:`cunls/factor/reprojection_factor_batch.h`

Reprojection error for bundle adjustment. Observations in **normalized** image coordinates.

.. math::
   P_{\mathrm{cam}} = T_{\mathrm{cam}} P,\qquad
   r = \begin{bmatrix} P_{\mathrm{cam},x}/P_{\mathrm{cam},z} - x_n \\
                       P_{\mathrm{cam},y}/P_{\mathrm{cam},z} - y_n \end{bmatrix}

.. list-table::
   :header-rows: 1
   :widths: 18 14 22 15 18

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 2
     - chain rule on projection
     - :math:`2 \times 9`
     - SE(3) × :math:`\mathbb{R}^3`

**Inputs:** Pose :math:`T_{\mathrm{cam}}` (state 1), 3D point :math:`P` (state 2). State: :code:`SE3StateBatch` then :code:`VectorStateBatch<3>` (see :doc:`state`). Observations :math:`(x_n, y_n)` and optional camera-from-rig from constructor.

.. cpp:function:: ReprojectionFactorBatch(const Vector<2>* observations, size_t capacity, float z_threshold = 1e-3f)
.. cpp:function:: ReprojectionFactorBatch(const Vector<2>* observations, const SE3Transform* poses_camera_from_rig, size_t capacity, float z_threshold = 1e-3f)

  :param ``observations``: [in] Device pointer to normalized observations.
  :param ``poses_camera_from_rig``: [in] Optional device pointer to camera extrinsics (second overload).
  :param ``capacity``: [in] Number of reprojection factors the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :param ``z_threshold``: [in] Minimum valid depth.
  :returns: Constructor has no return value.

.. _cpp-pnp-factor-batch:

PnPFactorBatch
--------------

Header: :code:`cunls/factor/pnp_factor_batch.h`

Fixed-structure **Perspective-n-Point** reprojection: the same normalized
pinhole residual as `ReprojectionFactorBatch`, but each 3D landmark is held in
device memory passed to the constructor (not a state variable). Only the SE(3)
pose is optimized; the analytic Jacobian is therefore :math:`2 \times 6`.

.. math::
   P_{\mathrm{cam}} = T_{\mathrm{cam}\leftarrow\mathrm{world}}\, P_{\mathrm{world}},\qquad
   r = \begin{bmatrix} P_{\mathrm{cam},x}/P_{\mathrm{cam},z} - x_n \\
                       P_{\mathrm{cam},y}/P_{\mathrm{cam},z} - y_n \end{bmatrix}

.. list-table::
   :header-rows: 1
   :widths: 18 14 22 15 18

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (same pinhole model as reprojection)
     - 2
     - pose chain rule only
     - :math:`2 \times 6`
     - SE(3)

**Inputs:** Normalized observations :math:`(x_n,y_n)` and matching world points
:math:`P_{\mathrm{world}}` from the constructor (one pair per factor). State: a
single :code:`SE3StateBatch` state per factor (rig-from-world pose, or
world-to-camera according to your convention—match how you built the
observations). Optional ``poses_camera_from_rig`` uses the same composition as
`ReprojectionFactorBatch`.

.. cpp:function:: PnPFactorBatch(const Vector<2>* observations, const Vector<3>* points_world, size_t capacity, float z_threshold = 1e-3f)
.. cpp:function:: PnPFactorBatch(const Vector<2>* observations, const SE3Transform* poses_camera_from_rig, const Vector<3>* points_world, size_t capacity, float z_threshold = 1e-3f)

  :param ``observations``: [in] Device pointer to normalized 2-D observations.
  :param ``points_world``: [in] Device pointer to fixed world points :math:`P`.
  :param ``poses_camera_from_rig``: [in] Optional per-factor rig extrinsics (second overload).
  :param ``capacity``: [in] Number of PnP correspondences the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :param ``z_threshold``: [in] Minimum valid camera-frame depth.
  :returns: Constructor has no return value.

.. _cpp-point-to-point-factor-batch:

PointToPointFactorBatch
-----------------------

Header: :code:`cunls/factor/point_to_point_factor_batch.h`

Point cloud registration (e.g. ICP). Residual = target point minus transformed source point.

.. math::
   r = p - T q = p - (R q + t)

.. list-table::
   :header-rows: 1
   :widths: 22 14 28 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 3
     - :math:`[\partial r/\partial\omega\;\partial r/\partial\rho] = [R[q]_\times\;{-}R]`
     - :math:`3 \times 6`
     - SE(3)

**Inputs:** :math:`T` = pose (state). State: one state from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`q` = target/source points (constructor).

.. cpp:function:: PointToPointFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, size_t capacity)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``capacity``: [in] Number of correspondences the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-point-to-plane-factor-batch:

PointToPlaneFactorBatch
-----------------------

Header: :code:`cunls/factor/point_to_plane_factor_batch.h`

Plane-based ICP: signed distance from transformed source point to target plane.

.. math::
   r = n_q^\top (p - T q) = n_q \cdot (p - (R q + t))

With :math:`n' = R^\top n_q`, the Jacobian row is :math:`[n'^\top [q]_\times,\; -n'^\top]`.

.. list-table::
   :header-rows: 1
   :widths: 20 14 28 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 1
     - :math:`[n'^\top [q]_\times\;{-}n'^\top]`
     - :math:`1 \times 6`
     - SE(3)

**Inputs:** :math:`T` = pose (state). State: one state from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`q`, :math:`n_q` = target point, source point, source normal (constructor).

.. cpp:function:: PointToPlaneFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, const Vector<3>* nq_observations_ptr, size_t capacity)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``nq_observations_ptr``: [in] Device pointer to source normals.
  :param ``capacity``: [in] Number of correspondences the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-symmetric-point-to-plane-factor-batch:

SymmetricPointToPlaneFactorBatch
--------------------------------

Header: :code:`cunls/factor/symmetric_point_to_plane_factor_batch.h`

Symmetric point-to-plane: both frames contribute normals; :math:`N = n_p + n_q`.

.. math::
   r = N^\top \bigl( T p - T^{-1} q \bigr)
     = \bigl( (R p + t) - R^\top(q - t) \bigr)^\top N

.. list-table::
   :header-rows: 1
   :widths: 18 14 22 14 10

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - 1
     - :math:`\mathrm{d}(T p,\, T^{-1} q)` w.r.t. :math:`T`
     - :math:`1 \times 6`
     - SE(3)

**Inputs:** :math:`T` = pose (state). State: one state from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`n_p`, :math:`q`, :math:`n_q` = target/source points and normals (constructor).

.. cpp:function:: SymmetricPointToPlaneFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, const Vector<3>* np_observations_ptr, const Vector<3>* nq_observations_ptr, size_t capacity)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``np_observations_ptr``: [in] Device pointer to target normals.
  :param ``nq_observations_ptr``: [in] Device pointer to source normals.
  :param ``capacity``: [in] Number of correspondences the buffers hold. 0 are active until ``SetNumActiveFactors``.
  :returns: Constructor has no return value.

.. _cpp-information-factor-batch:

InformationFactorBatch<T>
-------------------------

Header: :code:`cunls/factor/information/information_factor_batch.h`

**Item parameters.** ``Evaluate`` forwards ``factor_ids`` / ``num_factor_ids``
to the wrapped factor and weights item :math:`t` with the matrix of its factor
:math:`f(t)`, so the wrapper works under the :doc:`RANSAC minimizers
<../ransac>`. The sqrt-information product uses deterministic CUDA kernels
(fixed summation order per item). Residual sizes up to 96 stage each vector in
shared memory; larger sizes read their inputs directly with a stream-ordered
scratch buffer.

**Inheritance:** ``class InformationFactorBatch : public T::sized_layout`` — i.e.
the same ``SizedFactorBatch<kResidualSize, ...>`` as the wrapped type ``T``.
Residual and state sizes come from that base; this class adds
``NumActiveFactors``, storage for ``T``, and an ``Evaluate`` that applies
:math:`\Omega^{1/2}` after the inner factor.

``T`` must derive from some ``SizedFactorBatch`` (see type trait
``IsDerivedFromAnySizedFactorBatch``).

Wraps a factor to apply a square-root information matrix :math:`\Omega^{1/2}` (e.g. from measurement covariance).

.. math::
   r_{\mathrm{weighted}} = \Omega^{1/2} r,\qquad
   J_{\mathrm{weighted}} = \Omega^{1/2} J

.. list-table::
   :header-rows: 1
   :widths: 22 14 22 18 18

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - (per :math:`T`)
     - (see above)
     - (per :math:`T`)
     - same as wrapped :math:`T`

**Inputs:** Same state layout as the wrapped factor :code:`T`. Constructor also takes :code:`sqrt_information_matrices_ptr` (per-factor :math:`\Omega^{1/2}`).

.. cpp:function:: template <class... Args> InformationFactorBatch(const Matrix<T::residual_size_>* sqrt_information_matrices_ptr, size_t capacity, Args&&... sized_factor_batch_args)

  :param ``sqrt_information_matrices_ptr``: [in] Device pointer to per-factor square-root information matrices.
  :param ``capacity``: [in] Number of square-root information matrices; must equal ``T::Capacity()`` after ``T`` is constructed. The active count is the wrapped batch's (``SetNumActiveFactors`` is forwarded).
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T`` (same order as ``T``'s constructor, or ``(weight, …)`` when ``T`` is ``WeightedFactorBatch<U>``).
  :returns: Constructor has no return value.

.. _cpp-weighted-factor-batch:

WeightedFactorBatch<T>
-------------------------

Header: :code:`cunls/factor/weighted_factor_batch.h`

**Inheritance:** ``class WeightedFactorBatch : public T::sized_layout`` (same
``SizedFactorBatch`` specialization as ``T``), with ``NumActiveFactors`` and ``Evaluate``
extended for scalar weighting.

``T`` must derive from ``SizedFactorBatch``.

**Item parameters.** ``Evaluate`` forwards ``factor_ids`` / ``num_factor_ids``
to the wrapped factor; with per-factor weights, item :math:`t` is scaled by
the weight of its factor :math:`f(t)`.

Wraps a factor to apply scalar weight(s) to residuals and Jacobians. Supports
two modes: a single uniform weight applied to every factor, or per-factor
weights from a device array.

.. math::
   r_{\mathrm{weighted}} = w \, r,\qquad
   J_{\mathrm{weighted}} = w \, J

.. list-table::
   :header-rows: 1
   :widths: 22 14 22 18 18

   * - Residual
     - Residual dim
     - Jacobian
     - Jacobian dims
     - Manifold
   * - (see above)
     - (per :math:`T`)
     - (see above)
     - (per :math:`T`)
     - same as wrapped :math:`T`

**Inputs:** Same state layout as the wrapped factor :code:`T`. Constructor also
takes either a single :code:`float` weight or a :code:`const float*` device
pointer to per-factor weights.

.. cpp:function:: template <class... Args> WeightedFactorBatch(float weight, Args&&... sized_factor_batch_args)

  Uniform weight constructor. Multiplies every factor's residual and Jacobian
  by the same scalar :code:`weight`. The batch size is the inner factor's
  :code:`NumActiveFactors()`.

  :param ``weight``: [in] Scalar weight applied to all factors.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T``.
  :returns: Constructor has no return value.

.. cpp:function:: template <class... Args> WeightedFactorBatch(const float* per_factor_weights, size_t capacity, Args&&... sized_factor_batch_args)

  Per-factor weight constructor. Factor *i* has its residual and Jacobian
  multiplied by :code:`per_factor_weights[i]`.

  :param ``per_factor_weights``: [in] Device pointer to per-factor weights (at least ``capacity`` floats).
  :param ``capacity``: [in] Number of weights; must equal ``T::Capacity()`` for the constructed inner batch.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T``.
  :returns: Constructor has no return value.
