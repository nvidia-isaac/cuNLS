################################################################################
State API
################################################################################

The ``cunls/state`` module provides batched storage and **manifold** updates for
optimization variables. State batches implement the **Plus** (retraction) operation
so the solver can update states in tangent space while keeping them on the manifold.

================================================================================
Manifolds
================================================================================

**What is a manifold?**

Many variables in nonlinear least squares do not live in :math:`\mathbb{R}^n` but
on curved spaces: 2D/3D rotations (SO(2), SO(3)), rigid or similarity transforms
(SE(2), SE(3), Sim(2), Sim(3)), or other constrained sets. Such a space is a
**manifold**: at each point :math:`x` there is a **tangent space** (a linear space
of “directions”) whose dimension is the **intrinsic** dimension of the manifold.
The **ambient space** is the larger Euclidean space in which the manifold is
embedded (e.g. 3×3 matrices for SO(3), so ambient dimension 9).

**Why use manifolds?**

1. **Constraint satisfaction:** Updates are applied in the tangent space and then
   mapped back onto the manifold, so the state never leaves the constraint set
   (e.g. rotation matrices stay orthogonal).
2. **Correct dimension:** The solver only works with as many unknowns as the
   tangent dimension (e.g. 3 for SO(3) instead of 9), which improves numerics
   and efficiency.

**Plus (retraction)**

The **Plus** operation (in the literature often written :math:`\boxplus`) takes
a point :math:`x` on the manifold and a tangent vector :math:`\Delta` and returns
a new point on the manifold:

.. math::
   x \oplus \Delta = \mathrm{Plus}(x,\, \Delta)

So the solver computes an update :math:`\Delta` in tangent space (e.g. from
Gauss-Newton or Levenberg-Marquardt) and then sets
:math:`x_{\mathrm{new}} = x \oplus \Delta`. For Euclidean space,
:math:`x \oplus \Delta = x + \Delta`. For Lie groups (SO, SE, Sim), Plus is
implemented as right-multiplication by the exponential of the Lie algebra
element: :math:`x \oplus \Delta = x \cdot \mathrm{Exp}(\Delta)`.

**How the minimizer uses state batches**

The minimizer holds a current state :math:`x` in ambient storage. It solves for
a tangent update :math:`\Delta` (using Jacobians that are w.r.t. tangent space).
Then it calls :cpp:func:`StateBatch::Plus` (or :cpp:class:`StateBatchOps::Plus`
over multiple batches) to write :math:`x \oplus \Delta` back into the state
buffer. So the state batch is the object that knows how to apply :math:`\oplus`
for its manifold.

================================================================================
StateBatch Interface
================================================================================

.. cpp:function:: size_t TangentSize() const

  :returns: [out] Tangent-space dimension per state block.

.. cpp:function:: size_t AmbientSize() const

  :returns: [out] Ambient/storage dimension per state block.

.. cpp:function:: size_t NumStateBlocks() const

  :returns: [out] Number of state blocks in this batch.

.. cpp:function:: void Plus(const float* x, const float* delta, float* x_plus_delta, cudaStream_t stream)

  Computes :math:`x_{\mathrm{out}} = x \oplus \delta` for each block in the batch.

  :param ``x``: [in] Device pointer to the current state values (ambient).
  :param ``delta``: [in] Device pointer to tangent-space updates.
  :param ``x_plus_delta``: [out] Device pointer to updated state values (ambient).
  :param ``stream``: [in] CUDA stream for asynchronous execution.
  :returns: [out] No return value.

.. cpp:function:: float* StateBlockDevicePtr(size_t state_block_idx)

  :param ``state_block_idx``: [in] Zero-based index of state block.
  :returns: [out] Mutable device pointer for the selected block, or ``nullptr`` when out-of-range.

.. cpp:function:: const float* StateBlockDevicePtr(size_t state_block_idx) const

  :param ``state_block_idx``: [in] Zero-based index of state block.
  :returns: [out] Const device pointer for the selected block, or ``nullptr`` when out-of-range.

.. cpp:function:: const int* ConstStateIds() const

  :returns: [out] Device pointer to constant-state indices, or ``nullptr`` when none are set.

.. cpp:function:: size_t NumConstStateBlocks() const

  :returns: [out] Number of constant (non-optimized) state blocks.

================================================================================
State batch types (tables)
================================================================================

Each state batch type corresponds to a manifold. The table columns are: **Plus**
formula, **Ambient** dimension, **Tangent** dimension, **Ambient space**
description, **Tangent space** description, and **Memory layout** of one state
block in device memory.

--------------------------------------------------------------------------------
SizedStateBatch<AmbientDim, TangentDim>
--------------------------------------------------------------------------------

Generic base with compile-time ambient and tangent dimensions. Storage layout:
contiguous blocks, each of **AmbientDim** floats. Derived classes implement
:cpp:func:`Plus` for their manifold.

--------------------------------------------------------------------------------
VectorStateBatch<Dim>
--------------------------------------------------------------------------------

Header: :code:`cunls/state/vector_state_batch.h`

Euclidean vector state (e.g. landmarks, biases). Tangent and ambient spaces coincide.

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x + \delta`
     - :math:`\mathrm{Dim}`
     - :math:`\mathrm{Dim}`
     - :math:`\mathbb{R}^{\mathrm{Dim}}`
     - :math:`\mathbb{R}^{\mathrm{Dim}}`
     - :math:`\mathrm{Dim}` floats per block, contiguous

**Constructors:** Same as :code:`SizedStateBatch` with both dimensions equal to
:code:`Dim`. See :ref:`state-constructors` below.

--------------------------------------------------------------------------------
SO2StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/so2_state_batch.h`

2D rotations (heading angle). Tangent = 1 (angle in radians).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\delta)`
     - 4
     - 1
     - 2×2 rotation matrix
     - angle (radians)
     - row-major 2×2: :math:`[\cos\theta,\, -\sin\theta,\, \sin\theta,\, \cos\theta]`

--------------------------------------------------------------------------------
SO3StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/so3_state_batch.h`

3D rotations. Tangent = 3 (axis-angle / rotation vector).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\mathrm{skew}(\delta))`
     - 9
     - 3
     - 3×3 rotation matrix
     - 3D rotation vector
     - row-major 3×3 (9 floats)

--------------------------------------------------------------------------------
SE2StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/se2_state_batch.h`

2D rigid transform (rotation + translation). Tangent = 3 (:math:`v_x,\, v_y`, angle).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\delta)`
     - 9
     - 3
     - 3×3 homogeneous matrix
     - :math:`[v_x,\, v_y,\, \theta]`
     - row-major 3×3: :math:`[\cos\theta,\, -\sin\theta,\, t_x,\, \sin\theta,\, \cos\theta,\, t_y,\, 0,\, 0,\, 1]`

--------------------------------------------------------------------------------
SE3StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/se3_state_batch.h`

3D rigid transform (rotation + translation). Tangent = 6 (twist: rotation vector + translation).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\mathrm{skew}(\delta))`
     - 16
     - 6
     - 4×4 homogeneous matrix
     - 6D twist :math:`[\omega; \rho]`
     - row-major 4×4: :math:`[R\,|\,t;\; 0\; 0\; 0\; 1]` (16 floats)

--------------------------------------------------------------------------------
Similarity2StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/similarity2_state_batch.h`

2D similarity (rotation + translation + scale). Tangent = 4 (:math:`u_x,\, u_y,\, \theta,\, \lambda=\log s`).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\delta)`
     - 9
     - 4
     - 3×3 sim. matrix
     - :math:`[u_x,\, u_y,\, \theta,\, \lambda]`
     - row-major 3×3: :math:`[\cos\theta,\, -\sin\theta,\, t_x,\, \sin\theta,\, \cos\theta,\, t_y,\, 0,\, 0,\, 1/s]`

--------------------------------------------------------------------------------
Similarity3StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/similarity3_state_batch.h`

3D similarity (rotation + translation + scale). Tangent = 7 (:math:`\omega,\, u,\, \lambda=\log s`).

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 20 20 22

   * - Plus
     - Ambient
     - Tangent
     - Ambient space
     - Tangent space
     - Memory layout
   * - :math:`x \cdot \mathrm{Exp}(\delta)`
     - 16
     - 7
     - 4×4 sim. matrix
     - :math:`[\omega; u; \lambda]`
     - row-major 4×4: :math:`[R\,|\,t;\; 0\; 0\; 0\; 1/s]` (16 floats)

.. _state-constructors:

================================================================================
Constructors
================================================================================

--------------------------------------------------------------------------------
SizedStateBatch<AmbientDim, TangentDim> (constructors)
--------------------------------------------------------------------------------

.. cpp:function:: SizedStateBatch(const float* device_ptr, size_t num_blocks)

  :param ``device_ptr``: [in] Device pointer to contiguous state storage (num_blocks × AmbientDim floats).
  :param ``num_blocks``: [in] Number of state blocks.
  :returns: [out] Constructor has no return value.

.. cpp:function:: SizedStateBatch(const float* device_ptr, size_t num_blocks, const int* device_constant_state_ids, size_t num_const_state_blocks)

  :param ``device_ptr``: [in] Device pointer to contiguous state storage.
  :param ``num_blocks``: [in] Number of state blocks.
  :param ``device_constant_state_ids``: [in] Device pointer to indices of constant blocks.
  :param ``num_const_state_blocks``: [in] Number of constant block indices.
  :returns: [out] Constructor has no return value.

--------------------------------------------------------------------------------
VectorStateBatch<Dim> (constructors)
--------------------------------------------------------------------------------

Uses the same constructor signatures as :code:`SizedStateBatch` with ambient and
tangent dimension :code:`Dim`.

--------------------------------------------------------------------------------
StateBatch constructors
--------------------------------------------------------------------------------

Each StateBatch-derived class has constructors equivalent to:

.. cpp:function:: ClassName(cuBLASHandle& cublas_handle, const float* device_ptr, size_t num_blocks)
.. cpp:function:: ClassName(cuBLASHandle& cublas_handle, const float* device_ptr, size_t num_blocks, const int* device_constant_state_ids, size_t num_const_state_blocks)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``device_ptr``: [in] Device pointer to contiguous state storage.
  :param ``num_blocks``: [in] Number of state blocks.
  :param ``device_constant_state_ids``: [in] Device pointer to constant block indices.
  :param ``num_const_state_blocks``: [in] Number of constant block indices.
  :returns: [out] Constructor has no return value.

================================================================================
StateBatchOps
================================================================================

Orchestrates :cpp:func:`Plus` across multiple state batches: gathers tangent
updates from a single reduced vector, scatters to per-batch deltas, and calls
each batch’s :cpp:func:`Plus`.

.. cpp:function:: StateBatchOps()

  :returns: [out] Constructor has no return value.

.. cpp:function:: StateBatchOps(cudaStream_t stream, const std::vector<StateBatch*>& state_batches)

  :param ``stream``: [in] CUDA stream used to initialize mappings.
  :param ``state_batches``: [in] Ordered list of state batches.
  :returns: [out] Constructor has no return value.

.. cpp:function:: void Preprocess(cudaStream_t stream, const std::vector<StateBatch*>& state_batches)

  :param ``stream``: [in] CUDA stream for mapping/buffer initialization.
  :param ``state_batches``: [in] State batches used to build reduced/full mappings.
  :returns: [out] No return value.

.. cpp:function:: void Plus(cudaStream_t stream, const std::vector<const float*>& x_ptrs, const DeviceVector<float>& delta, std::vector<float*>& x_plus_delta_ptrs)

  :param ``stream``: [in] CUDA stream for scatter/update operations.
  :param ``x_ptrs``: [in] Current per-batch state pointers.
  :param ``delta``: [in] Reduced tangent update vector.
  :param ``x_plus_delta_ptrs``: [out] Per-batch pointers for updated states.
  :returns: [out] No return value.

.. cpp:function:: size_t NumReducedStates() const

  :returns: [out] Number of scalar optimization variables after removing constant states.
