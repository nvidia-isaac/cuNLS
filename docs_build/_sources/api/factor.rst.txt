################################################################################
Factor API
################################################################################

The ``cunls/factor`` module provides batched residual and Jacobian models for
non-linear least squares. Each factor computes a residual vector from one or
more **state blocks**. State blocks lie on **manifolds**; the factor API uses
**tangent-space dimensions** for Jacobian layout and solver variables. Links to
the corresponding state batch types are in the :ref:`factor-inputs` section
and in each factor’s **Inputs** subsection.

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

.. cpp:function:: template <class... Args> InformationFactorBatch(cuBLASHandle& cublas_handle, const Matrix<T::residual_size_>* sqrt_information_matrices_ptr, size_t num_factors, Args&&... sized_factor_batch_args)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``sqrt_information_matrices_ptr``: [in] Device pointer to per-factor square-root information matrices.
  :param ``num_factors``: [in] Number of factors and weighting matrices.
  :param ``sized_factor_batch_args``: [in] Constructor arguments forwarded to wrapped factor ``T``.
  :returns: Constructor has no return value.
