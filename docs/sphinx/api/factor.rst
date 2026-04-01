################################################################################
Factor API
################################################################################

The factor module provides batched residual and Jacobian models for non-linear
least squares. Each factor computes a residual vector from one or more **state
blocks**. State blocks lie on **manifolds**; the factor API uses
**tangent-space dimensions** for Jacobian layout and solver variables. Links to
the corresponding state batch types are in the :ref:`factor-inputs` section
and in each factor’s **Inputs** subsection.

**C++** — ``cunls/factor``
  |  **Python** — ``pycunls``

================================================================================
C++ API
================================================================================

.. _factor-inputs:

Factor inputs
-------------

Each factor's **Inputs** subsection below and the :doc:`state` API list the
required state batch types (e.g. :code:`VectorStateBatch<Dim>`, :code:`SO3StateBatch`).

FactorBatch
-----------

Abstract base (:code:`cunls/factor/factor_batch.h`).

.. math::
   r = f(x),\qquad J = \frac{\partial f}{\partial x}

.. cpp:function:: bool Evaluate(float* residuals, float* jacobians, float const* const* state_pointers, cudaStream_t stream) const

  :param ``residuals``: [out] Residual output buffer.
  :param ``jacobians``: [out] Optional Jacobian output buffer (``nullptr`` to skip).
  :param ``state_pointers``: [in] Device pointer array mapping factor inputs to state blocks.
  :param ``stream``: [in] CUDA stream for asynchronous execution.
  :returns: [out] ``true`` on success.

.. cpp:function:: size_t ResidualsSize() const

  :returns: [out] Residual dimension per factor.

.. cpp:function:: std::vector<size_t> StateBlockSizes() const

  :returns: [out] State block **tangent** dimensions consumed by each factor.

.. cpp:function:: size_t NumFactors() const

  :returns: [out] Number of factors in the batch.

SizedFactorBatch<kResidualSize, ...kStateBlockSizes>
----------------------------------------------------

Compile-time convenience base (:code:`cunls/factor/sized_factor_batch.h`) that fixes
residual and state (tangent) dimensions at compile time.

Each specialization exposes **sized_layout** — an alias for the same
``SizedFactorBatch<kResidualSize, kStateBlockSizes...>`` type. Wrapper templates
such as ``InformationFactorBatch<T>`` and ``WeightedFactorBatch<T>`` inherit
``public T::sized_layout`` so they remain full ``SizedFactorBatch`` instances with
the same layout as the inner batch ``T``.

PriorVectorFactorBatch<Dim>
----------------------------

Header: :code:`cunls/factor/prior_vector_factor_batch.h`

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

**Inputs:** :math:`x` = state vector, :math:`o` = observation (constructor). State: one block from :code:`VectorStateBatch<Dim>` (see :doc:`state`).

Constructor:

.. code-block:: cpp

   PriorVectorFactorBatch(const Vector<Dim>* observations_ptr, size_t num_factors)

- ``observations_ptr`` — [in] Device pointer to observed vectors.
- ``num_factors`` — [in] Number of factors in this batch.

SO2PriorFactorBatch
-------------------

Header: :code:`cunls/factor/so2_prior_factor_batch.h`

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

**Inputs:** :math:`R` = current rotation (state). State: one block from :code:`SO2StateBatch` (see :doc:`state`).

.. cpp:function:: SO2PriorFactorBatch(const Matrix<2>* observations_ptr, size_t num_factors)

  :param ``observations_ptr``: [in] Device pointer to SO(2) observations (2×2 row-major).
  :param ``num_factors``: [in] Number of factors.
  :returns: Constructor has no return value.

SO3PriorFactorBatch
-------------------

Header: :code:`cunls/factor/so3_prior_factor_batch.h`

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

**Inputs:** :math:`R` = current rotation (state). State: one block from :code:`SO3StateBatch` (see :doc:`state`).

.. cpp:function:: SO3PriorFactorBatch(cuBLASHandle& cublas_handle, const Matrix<3>* observations_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``observations_ptr``: [in] Device pointer to SO(3) observations (3×3 row-major).
  :param ``num_factors``: [in] Number of factors.
  :returns: Constructor has no return value.

SE2PriorFactorBatch
-------------------

Header: :code:`cunls/factor/se2_prior_factor_batch.h`

Prior on 2D rigid transform. State: one block from :code:`SE2StateBatch` (see :doc:`state`).

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

SE3PriorFactorBatch
-------------------

Header: :code:`cunls/factor/se3_prior_factor_batch.h`

Prior on 3D rigid transform. State: one block from :code:`SE3StateBatch` (see :doc:`state`).

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

Similarity2PriorFactorBatch
---------------------------

Header: :code:`cunls/factor/similarity2_prior_factor_batch.h`

Prior on 2D similarity transform. State: one block from :code:`Similarity2StateBatch` (see :doc:`state`).

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

Similarity3PriorFactorBatch
----------------------------

Header: :code:`cunls/factor/similarity3_prior_factor_batch.h`

Prior on 3D similarity transform. State: one block from :code:`Similarity3StateBatch` (see :doc:`state`).

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

.. cpp:function:: ClassName(cuBLASHandle& cublas_handle, const ObsType* observations_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``observations_ptr``: [in] Device pointer to observation transforms.
  :param ``num_factors``: [in] Number of factors.
  :returns: Constructor has no return value.

SL4PriorFactorBatch
-------------------

Header: :code:`cunls/factor/sl4_prior_factor_batch.h`

Prior on an SL(4) transform. State: one block from :code:`SL4StateBatch` (see :doc:`state`).

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

.. cpp:function:: SL4PriorFactorBatch(cuBLASHandle& cublas_handle, const SL4Transform* observations_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``observations_ptr``: [in] Device pointer to SL(4) target transforms (row-major 4×4).
  :param ``num_factors``: [in] Number of factors.
  :returns: Constructor has no return value.

SE3BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/se3_between_factor_batch.h`

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

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two poses (state blocks). State: two blocks from :code:`SE3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SE3BetweenFactorBatch(cuBLASHandle& cublas_handle, const SE3Transform* pose_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms.
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

SE2BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/se2_between_factor_batch.h`

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

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two poses (state blocks). State: two blocks from :code:`SE2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SE2BetweenFactorBatch(cuBLASHandle& cublas_handle, const Matrix<3>* pose_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 3×3).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

SO2BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/so2_between_factor_batch.h`

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

**Inputs:** :math:`R_{\mathrm{left}}`, :math:`R_{\mathrm{right}}` = two rotations (state blocks). State: two blocks from :code:`SO2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative rotation (constructor).

.. cpp:function:: SO2BetweenFactorBatch(cuBLASHandle& cublas_handle, const Matrix<2>* rotation_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``rotation_deltas_ptr``: [in] Device pointer to measured relative rotations (row-major 2×2).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

SO3BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/so3_between_factor_batch.h`

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

**Inputs:** :math:`R_{\mathrm{left}}`, :math:`R_{\mathrm{right}}` = two rotations (state blocks). State: two blocks from :code:`SO3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative rotation (constructor).

.. cpp:function:: SO3BetweenFactorBatch(cuBLASHandle& cublas_handle, const Matrix<3>* rotation_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``rotation_deltas_ptr``: [in] Device pointer to measured relative rotations (row-major 3×3).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

Similarity2BetweenFactorBatch
-----------------------------

Header: :code:`cunls/factor/similarity2_between_factor_batch.h`

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

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (state blocks). State: two blocks from :code:`Similarity2StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: Similarity2BetweenFactorBatch(cuBLASHandle& cublas_handle, const Matrix<3>* pose_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 3×3).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

Similarity3BetweenFactorBatch
-----------------------------

Header: :code:`cunls/factor/similarity3_between_factor_batch.h`

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

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (state blocks). State: two blocks from :code:`Similarity3StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: Similarity3BetweenFactorBatch(cuBLASHandle& cublas_handle, const Matrix<4>* pose_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 4×4).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

SL4BetweenFactorBatch
---------------------

Header: :code:`cunls/factor/sl4_between_factor_batch.h`

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

**Inputs:** :math:`T_{\mathrm{left}}`, :math:`T_{\mathrm{right}}` = two transforms (state blocks). State: two blocks from :code:`SL4StateBatch` (see :doc:`state`). :math:`\Delta` = measured relative transform (constructor).

.. cpp:function:: SL4BetweenFactorBatch(cuBLASHandle& cublas_handle, const SL4Transform* pose_deltas_ptr, size_t num_factors)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``pose_deltas_ptr``: [in] Device pointer to measured relative transforms (row-major 4×4, unit determinant).
  :param ``num_factors``: [in] Number of between constraints.
  :returns: Constructor has no return value.

VectorBetweenFactorBatch<Dim>
-----------------------------

Header: :code:`cunls/factor/vector_between_factor_batch.h`

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

**Inputs:** :math:`x_l`, :math:`x_r` = two vector states. State: two blocks from :code:`VectorStateBatch<Dim>` (see :doc:`state`). :math:`\delta` = measured difference (constructor).

Constructor:

.. code-block:: cpp

   VectorBetweenFactorBatch(const Vector<Dim>* deltas_ptr, size_t num_factors)

- ``deltas_ptr`` — [in] Device pointer to measured difference vectors.
- ``num_factors`` — [in] Number of factors in this batch.

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

**Inputs:** Pose :math:`T_{\mathrm{cam}}` (state block 1), 3D point :math:`P` (state block 2). State: :code:`SE3StateBatch` then :code:`VectorStateBatch<3>` (see :doc:`state`). Observations :math:`(x_n, y_n)` and optional camera-from-rig from constructor.

.. cpp:function:: ReprojectionFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations, size_t num_observations, float z_threshold = 1e-3f)
.. cpp:function:: ReprojectionFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations, const SE3Transform* poses_camera_from_rig, size_t num_observations, float z_threshold = 1e-3f)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``observations``: [in] Device pointer to normalized observations.
  :param ``poses_camera_from_rig``: [in] Optional device pointer to camera extrinsics (second overload).
  :param ``num_observations``: [in] Number of reprojection factors.
  :param ``z_threshold``: [in] Minimum valid depth.
  :returns: Constructor has no return value.

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
single :code:`SE3StateBatch` block per factor (rig-from-world pose, or
world-to-camera according to your convention—match how you built the
observations). Optional ``poses_camera_from_rig`` uses the same composition as
`ReprojectionFactorBatch`.

.. cpp:function:: PnPFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations, const Vector<3>* points_world, size_t num_observations, float z_threshold = 1e-3f)
.. cpp:function:: PnPFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations, const SE3Transform* poses_camera_from_rig, const Vector<3>* points_world, size_t num_observations, float z_threshold = 1e-3f)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``observations``: [in] Device pointer to normalized 2-D observations.
  :param ``points_world``: [in] Device pointer to fixed world points :math:`P`.
  :param ``poses_camera_from_rig``: [in] Optional per-factor rig extrinsics (second overload).
  :param ``num_observations``: [in] Number of PnP correspondences.
  :param ``z_threshold``: [in] Minimum valid camera-frame depth.
  :returns: Constructor has no return value.

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

**Inputs:** :math:`T` = pose (state). State: one block from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`q` = target/source points (constructor).

.. cpp:function:: PointToPointFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, size_t num_factors)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``num_factors``: [in] Number of correspondences.
  :returns: Constructor has no return value.

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

**Inputs:** :math:`T` = pose (state). State: one block from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`q`, :math:`n_q` = target point, source point, source normal (constructor).

.. cpp:function:: PointToPlaneFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, const Vector<3>* nq_observations_ptr, size_t num_factors)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``nq_observations_ptr``: [in] Device pointer to source normals.
  :param ``num_factors``: [in] Number of correspondences.
  :returns: Constructor has no return value.

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

**Inputs:** :math:`T` = pose (state). State: one block from :code:`SE3StateBatch` (see :doc:`state`). :math:`p`, :math:`n_p`, :math:`q`, :math:`n_q` = target/source points and normals (constructor).

.. cpp:function:: SymmetricPointToPlaneFactorBatch(const Vector<3>* p_observations_ptr, const Vector<3>* q_observations_ptr, const Vector<3>* np_observations_ptr, const Vector<3>* nq_observations_ptr, size_t num_factors)

  :param ``p_observations_ptr``: [in] Device pointer to target points.
  :param ``q_observations_ptr``: [in] Device pointer to source points.
  :param ``np_observations_ptr``: [in] Device pointer to target normals.
  :param ``nq_observations_ptr``: [in] Device pointer to source normals.
  :param ``num_factors``: [in] Number of correspondences.
  :returns: Constructor has no return value.

InformationFactorBatch<T>
-------------------------

Header: :code:`cunls/factor/information_factor_batch.h`

**Inheritance:** ``class InformationFactorBatch : public T::sized_layout`` — i.e.
the same ``SizedFactorBatch<kResidualSize, ...>`` as the wrapped type ``T``.
Residual and state-block sizes come from that base; this class adds
``NumFactors``, storage for ``T``, and an ``Evaluate`` that applies
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

.. cpp:function:: template <class... Args> InformationFactorBatch(cuBLASHandle& cublas_handle, const Matrix<T::residual_size_>* sqrt_information_matrices_ptr, size_t num_matrices, Args&&... sized_factor_batch_args)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``sqrt_information_matrices_ptr``: [in] Device pointer to per-factor square-root information matrices.
  :param ``num_matrices``: [in] Number of square-root information matrices; must equal ``T::NumFactors()`` after ``T`` is constructed.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T`` (same order as ``T``'s constructor — include a leading ``cuBLASHandle`` when ``T`` requires one, or ``(weight, …)`` when ``T`` is ``WeightedFactorBatch<U>``).
  :returns: Constructor has no return value.

WeightedFactorBatch<T>
-------------------------

Header: :code:`cunls/factor/weighted_factor_batch.h`

**Inheritance:** ``class WeightedFactorBatch : public T::sized_layout`` (same
``SizedFactorBatch`` specialization as ``T``), with ``NumFactors`` and ``Evaluate``
extended for scalar weighting.

``T`` must derive from ``SizedFactorBatch``.

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
  :code:`NumFactors()`.

  :param ``weight``: [in] Scalar weight applied to all factors.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T``.
  :returns: Constructor has no return value.

.. cpp:function:: template <class... Args> WeightedFactorBatch(const float* per_factor_weights, size_t num_weights, Args&&... sized_factor_batch_args)

  Per-factor weight constructor. Factor *i* has its residual and Jacobian
  multiplied by :code:`per_factor_weights[i]`.

  :param ``per_factor_weights``: [in] Device pointer to per-factor weights (at least ``num_weights`` floats).
  :param ``num_weights``: [in] Number of weights; must equal ``T::NumFactors()`` for the constructed inner batch.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T``.
  :returns: Constructor has no return value.

================================================================================
Python API (``pycunls``)
================================================================================

All Python factor batches inherit from the abstract ``FactorBatch`` base
class.  Every constructor argument documented as ``DevicePointer`` accepts
either a ``cupy.ndarray`` (the device pointer is extracted automatically via
``.data.ptr``) or a raw ``int`` GPU device address.

The residual formulas, Jacobian structure, and state layouts are identical to
the C++ versions documented in the sections above — this section focuses on
the Python constructor signatures, methods, and properties.

.. _py-factor-batch-interface:

Common ``FactorBatch`` interface
--------------------------------------------------------------------------------

Every factor batch — built-in or user-defined — exposes the following
read-only properties and methods.

**Read-only properties**

- **num_factors** (``int``) — number of factor instances in the batch.
- **residuals_size** (``int``) — residual dimension per factor (e.g. 2 for
  ``ReprojectionFactorBatch``, 6 for ``SE3BetweenFactorBatch``).

**Methods**

- ``state_block_sizes() -> list[int]`` — returns a list of tangent-space
  dimensions for each state block consumed by one factor.  For example,
  ``ReprojectionFactorBatch`` returns ``[6, 3]`` (SE(3) pose then
  :math:`\mathbb{R}^3` point), and ``PnPFactorBatch`` returns ``[6]`` (pose
  only; 3-D points are fixed in the constructor).

.. _py-prior-vector-factor:

``pycunls.PriorVectorFactorBatch1`` / ``PriorVectorFactorBatch2`` / ``PriorVectorFactorBatch3`` / ``PriorVectorFactorBatch6``
------------------------------------------------------------------------------------------------------------------------------

Prior on a Euclidean vector.  Residual = :math:`x - o` with identity
Jacobian.  The suffix indicates the dimension.

**Constructor**

.. code-block:: python

   fb = pycunls.PriorVectorFactorBatch3(observations, num_factors)

- **observations** (``DevicePointer``) — contiguous GPU buffer of
  ``num_factors × Dim`` floats holding the observed (target) vectors.  The
  factor batch does **not** copy the data; the caller must keep the
  allocation alive.
- **num_factors** (``int``) — number of prior factors.

**State layout:** one block per factor from the corresponding
``VectorStateBatch`` (see :ref:`py-vector-state-batches`).

.. _py-so2-prior-factor:

``pycunls.SO2PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 2-D rotation.  Residual = :math:`\mathrm{Log}(R_\mathrm{target}^\top R)`.
Does **not** require a ``CublasHandle``.

**Constructor**

.. code-block:: python

   fb = pycunls.SO2PriorFactorBatch(observations, num_factors)

- **observations** (``DevicePointer``) — ``num_factors × 4`` floats holding
  row-major 2×2 target rotation matrices.
- **num_factors** (``int``) — number of prior factors.

**State layout:** one block per factor from ``SO2StateBatch``
(see :ref:`py-lie-state-batches`).

.. _py-so3-prior-factor:

``pycunls.SO3PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 3-D rotation.  Residual =
:math:`\mathrm{Log}(R_\mathrm{target}^\top R)`, Jacobian =
:math:`J_r^{-1}(r)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SO3PriorFactorBatch(cublas, observations, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **observations** (``DevicePointer``) — ``num_factors × 9`` floats holding
  row-major 3×3 target rotation matrices.
- **num_factors** (``int``) — number of prior factors.

**State layout:** one block per factor from ``SO3StateBatch``.

.. _py-se3-prior-factor:

``pycunls.SE3PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on a 3-D rigid transform.  Residual =
:math:`\mathrm{Log}(T_\mathrm{target}^{-1} T)`, Jacobian = :math:`J_r^{-1}(r)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SE3PriorFactorBatch(cublas, observations, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **observations** (``DevicePointer``) — ``num_factors × 16`` floats
  holding row-major 4×4 target homogeneous matrices.
- **num_factors** (``int``) — number of prior factors.

**State layout:** one block per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

.. _py-sl4-prior-factor:

``pycunls.SL4PriorFactorBatch``
--------------------------------------------------------------------------------

Prior on an SL(4) transform.  Residual =
:math:`\mathrm{Log}(T_\mathrm{target}^{-1} T)`.

**Constructor**

.. code-block:: python

   fb = pycunls.SL4PriorFactorBatch(cublas, observations, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **observations** (``DevicePointer``) — ``num_factors × 16`` floats
  holding row-major 4×4 SL(4) target transforms.
- **num_factors** (``int``) — number of prior factors.

**State layout:** one block per factor from ``SL4StateBatch``.

.. _py-se3-between-factor:

``pycunls.SE3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative pose between two SE(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SE3BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[T_left, T_right]`` — both from
:ref:`SE3StateBatch <py-lie-state-batches>`.  The state-pointer list must
therefore contain ``2 × num_factors`` entries.

.. _py-se2-between-factor:

``pycunls.SE2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two SE(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SE2BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 9`` floats holding
  row-major 3×3 measured relative transforms :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[T_left, T_right]`` — both from
``SE2StateBatch``.

.. _py-so2-between-factor:

``pycunls.SO2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative rotation between two SO(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^\top R_l^\top R_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SO2BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 4`` floats holding
  row-major 2×2 measured relative rotations :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[R_left, R_right]`` — both from
``SO2StateBatch``.

.. _py-so3-between-factor:

``pycunls.SO3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative rotation between two SO(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^\top R_l^\top R_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SO3BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 9`` floats holding
  row-major 3×3 measured relative rotations :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[R_left, R_right]`` — both from
``SO3StateBatch``.

.. _py-similarity2-between-factor:

``pycunls.Similarity2BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two Sim(2) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.Similarity2BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 9`` floats holding
  row-major 3×3 measured relative transforms :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[T_left, T_right]`` — both from
``Similarity2StateBatch``.

.. _py-similarity3-between-factor:

``pycunls.Similarity3BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two Sim(3) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.Similarity3BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[T_left, T_right]`` — both from
``Similarity3StateBatch``.

.. _py-sl4-between-factor:

``pycunls.SL4BetweenFactorBatch``
--------------------------------------------------------------------------------

Constrains the relative transform between two SL(4) frames.  Residual =
:math:`\mathrm{Log}(\Delta^{-1} T_l^{-1} T_r)`.  Two state blocks per
factor.

**Constructor**

.. code-block:: python

   fb = pycunls.SL4BetweenFactorBatch(cublas, deltas, num_factors)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **deltas** (``DevicePointer``) — ``num_factors × 16`` floats holding
  row-major 4×4 measured relative transforms :math:`\Delta` (unit determinant).
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor — ``[T_left, T_right]`` — both from
``SL4StateBatch``.

.. _py-vector-between-factor:

``pycunls.VectorBetweenFactorBatch1`` / ``VectorBetweenFactorBatch2`` / ``VectorBetweenFactorBatch3`` / ``VectorBetweenFactorBatch6``
--------------------------------------------------------------------------------------------------------------------------------------

Between factor on Euclidean vectors.  Residual =
:math:`x_l - x_r - \delta`.  Two state blocks per factor.

**Constructor**

.. code-block:: python

   fb = pycunls.VectorBetweenFactorBatch3(deltas, num_factors)

- **deltas** (``DevicePointer``) — ``num_factors × Dim`` floats holding
  the measured difference vectors :math:`\delta`.
- **num_factors** (``int``) — number of between constraints.

**State layout:** two blocks per factor from the corresponding
``VectorStateBatch`` (see :ref:`py-vector-state-batches`).

.. _py-reprojection-factor:

``pycunls.ReprojectionFactorBatch``
--------------------------------------------------------------------------------

Reprojection error for bundle adjustment.  Observations must be in
**normalized image coordinates** (intrinsic calibration already applied):
:math:`r = \pi(T, P) - z`.

**Constructor**

.. code-block:: python

   fb = pycunls.ReprojectionFactorBatch(
       cublas, observations, num_observations, z_threshold=1e-3)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **observations** (``DevicePointer``) — ``num_observations × 2`` floats
  holding normalized 2-D observations :math:`(x_n, y_n)`.
- **num_observations** (``int``) — number of reprojection factors.
- **z_threshold** (``float``, default ``1e-3``) — minimum valid depth
  :math:`z` in camera frame.  Points with :math:`z < z_\text{threshold}`
  produce zero residuals and Jacobians to avoid singularities.

**State layout:** two blocks per factor — ``[SE3 pose, R^3 point]`` — from
:ref:`SE3StateBatch <py-lie-state-batches>` and
:ref:`VectorStateBatch3 <py-vector-state-batches>` respectively.

.. _py-pnp-factor:

``pycunls.PnPFactorBatch``
--------------------------------------------------------------------------------

PnP-style reprojection: **fixed** 3-D points in the constructor, **one** SE(3)
state per correspondence (typically the same camera pose pointer repeated).

**Constructor (identity camera-from-rig)**

.. code-block:: python

   fb = pycunls.PnPFactorBatch(
       cublas, observations, points_world, num_observations, z_threshold=1e-3)

**Constructor (with camera-from-rig extrinsics per factor)**

.. code-block:: python

   fb = pycunls.PnPFactorBatch(
       cublas, observations, poses_camera_from_rig, points_world,
       num_observations, z_threshold=1e-3)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **observations** — ``num_observations × 2`` normalized image coordinates.
- **points_world** — ``num_observations × 3`` fixed world points (not
  optimized).
- **poses_camera_from_rig** — ``num_observations × 16`` row-major SE(3)
  matrices (optional overload).
- **z_threshold** — minimum valid depth in the camera frame (same role as
  :ref:`ReprojectionFactorBatch <py-reprojection-factor>`).

**State layout:** one ``SE3StateBatch`` block per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

.. _py-icp-factors:

``pycunls.PointToPointFactorBatch``
--------------------------------------------------------------------------------

Point-to-point ICP factor.  Residual = :math:`p - T q`.

**Constructor**

.. code-block:: python

   fb = pycunls.PointToPointFactorBatch(p_observations, q_observations, num_factors)

- **p_observations** (``DevicePointer``) — ``num_factors × 3`` floats
  holding target points :math:`p`.
- **q_observations** (``DevicePointer``) — ``num_factors × 3`` floats
  holding source points :math:`q`.
- **num_factors** (``int``) — number of point correspondences.

**State layout:** one block per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

``pycunls.PointToPlaneFactorBatch``
--------------------------------------------------------------------------------

Point-to-plane ICP factor.  Residual = :math:`n_q^\top (p - T q)`.

**Constructor**

.. code-block:: python

   fb = pycunls.PointToPlaneFactorBatch(
       p_observations, q_observations, nq_observations, num_factors)

- **p_observations** (``DevicePointer``) — target points (``× 3`` floats).
- **q_observations** (``DevicePointer``) — source points (``× 3`` floats).
- **nq_observations** (``DevicePointer``) — source normals (``× 3`` floats).
- **num_factors** (``int``) — number of correspondences.

**State layout:** one block per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

``pycunls.SymmetricPointToPlaneFactorBatch``
--------------------------------------------------------------------------------

Symmetric point-to-plane ICP factor.  Both frames contribute normals;
:math:`N = n_p + n_q`.

**Constructor**

.. code-block:: python

   fb = pycunls.SymmetricPointToPlaneFactorBatch(
       p_observations, q_observations,
       np_observations, nq_observations, num_factors)

- **p_observations** (``DevicePointer``) — target points (``× 3`` floats).
- **q_observations** (``DevicePointer``) — source points (``× 3`` floats).
- **np_observations** (``DevicePointer``) — target normals (``× 3`` floats).
- **nq_observations** (``DevicePointer``) — source normals (``× 3`` floats).
- **num_factors** (``int``) — number of correspondences.

**State layout:** one block per factor from
:ref:`SE3StateBatch <py-lie-state-batches>`.

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
       cublas, inner_factor, sqrt_information_matrices)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle.
- **inner_factor** (``FactorBatch``) — the factor batch to wrap.  The wrapper
  delegates ``Evaluate`` to this factor first, then applies the information
  matrices.  The inner factor must be kept alive for the lifetime of the
  wrapper.
- **sqrt_information_matrices** (``DevicePointer``) —
  ``num_factors × residual_size × residual_size`` contiguous floats holding
  one row-major square-root information matrix per factor.

**Example**

.. code-block:: python

   inner = pycunls.SE3BetweenFactorBatch(cublas, deltas, N)
   info  = pycunls.InformationFactorBatch(cublas, inner, sqrt_info_gpu)

   problem.add_factor_batch(info, state_pointers)

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
- **weights** (``DevicePointer``) — ``num_factors`` contiguous floats, one
  weight per factor.

Exactly one of ``weight`` or ``weights`` must be provided.

**Example**

.. code-block:: python

   inner = pycunls.PriorVectorFactorBatch3(obs_gpu, N)
   wfb   = pycunls.WeightedFactorBatch(inner, weight=5.0)

   problem.add_factor_batch(wfb, state_pointers)

.. _py-custom-factor-batch:

``pycunls.CustomFactorBatch``
--------------------------------------------------------------------------------

Base class for user-defined factors.  Subclass this to implement a residual
and Jacobian computation that is not available as a built-in factor.

**Constructor**

.. code-block:: python

   class MyFactor(pycunls.CustomFactorBatch):
       def __init__(self, num_factors):
           super().__init__(
               residual_size=...,
               state_block_sizes=[...],
               num_factors=num_factors,
           )

- **residual_size** (``int``) — dimension of the residual vector per
  factor.
- **state_block_sizes** (``Sequence[int]``) — list of tangent-space
  dimensions for each state block consumed by one factor (e.g. ``[1, 1]``
  for a factor reading two scalar states).
- **num_factors** (``int``) — number of factor instances.

**Methods to override**

- ``evaluate(residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle) -> bool``
  — computes residuals and Jacobians on the GPU for all factors in the
  batch.  All four arguments are raw ``int`` handles:

  - *residuals_ptr* — device pointer to the output residual buffer.
    Layout: ``num_factors × residual_size`` contiguous floats.
  - *jacobians_ptr* — device pointer to the output Jacobian buffer.
    Layout: ``num_factors × residual_size × sum(state_block_sizes)``
    contiguous floats (row-major per factor, blocks concatenated in state
    order).  May be ``0`` (null) when the minimizer only needs residuals
    (e.g. for cost evaluation); in that case skip Jacobian writes.
  - *state_pointers_ptr* — device pointer to an array of ``float*``
    pointers.  The array has ``num_factors × len(state_block_sizes)``
    entries.  Each entry is the device address of the ambient-space storage
    for one (factor, state-block) pair, in row-major order.  Because Warp
    kernels cannot perform ``float**`` double-pointer indirection, custom
    factors typically gather state values into contiguous CuPy arrays
    before launching a kernel (see the
    :ref:`Custom Warp Factor tutorial <pycunls_tutorial:Custom Warp Factor>`).
  - *stream_handle* — ``cudaStream_t`` cast to ``int``.  All GPU work
    **must** be launched on this stream.

  Return ``True`` on success.  The default implementation raises
  ``NotImplementedError``.

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
       def __init__(self, num_factors):
           super().__init__(
               residual_size=...,
               state_block_sizes=[...],
               num_factors=num_factors,
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

- ``make_warp_stream(stream_handle: int) -> wp.Stream`` — wraps a raw
  ``cudaStream_t`` (passed as ``int``) as a ``wp.Stream``.  Use the
  returned stream in ``wp.launch(..., stream=stream)`` to ensure the Warp
  kernel executes on the minimizer's CUDA stream.

**Methods to override**

- ``evaluate(residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle) -> bool``
  — same contract as ``CustomFactorBatch.evaluate``.  Typical
  implementations:

  1. Gather scattered state pointers into contiguous CuPy arrays (using a
     CuPy ``RawKernel`` or ``cp.ndarray`` indexing).
  2. Wrap the contiguous arrays and output buffers with
     ``self.wrap_array``.
  3. Build a ``wp.Stream`` with ``self.make_warp_stream``.
  4. Launch a ``@wp.kernel`` on that stream.

See :ref:`pycunls_tutorial:Custom Warp Factor` for a complete example.
