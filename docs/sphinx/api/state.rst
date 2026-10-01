################################################################################
State API
################################################################################

The state module provides batched storage and **manifold** updates for
optimization variables. State batches implement the **Plus** (retraction)
operation so the solver can update states in tangent space while keeping them on
the manifold.

**C++** — ``cunls/state``
  |  **Python** — ``pycunls``

================================================================================
Manifolds
================================================================================

**What is a manifold?**

Many variables in nonlinear least squares do not live in :math:`\mathbb{R}^n` but
on curved spaces: 2D/3D rotations (SO(2), SO(3)), rigid or similarity transforms
(SE(2), SE(3), Sim(2), Sim(3)), projective linear groups (SL(4)), or other constrained sets. Such a space is a
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

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / state blocks their buffers hold) and
   start with **zero** active entries: call ``SetNumFactors(n)`` /
   ``SetNumStateBlocks(n)`` (Python: ``set_num_factors`` /
   ``set_num_state_blocks``) before solving, and again whenever the problem size
   changes. See :ref:`capacity-and-active-count`.

.. cpp:function:: size_t TangentSize() const

  :returns: [out] Tangent-space dimension per state block.

.. cpp:function:: size_t AmbientSize() const

  :returns: [out] Ambient/storage dimension per state block.

.. cpp:function:: size_t NumStateBlocks() const

  :returns: [out] Number of active state blocks (the first
    ``NumStateBlocks()`` blocks of the buffer). 0 after construction, until
    ``SetNumStateBlocks``.

.. cpp:function:: size_t StateBatch::Capacity() const

  :returns: [out] Number of state blocks the buffer holds: the ``capacity``
    passed to the constructor. Constant for the batch's lifetime.
    ``StateBlockDevicePtr(i)`` is valid for any ``i < Capacity()``.

.. cpp:function:: size_t ConstCapacity() const

  :returns: [out] Number of entries the constant-id buffer holds (0 without
    one).

.. cpp:function:: void SetNumStateBlocks(size_t num_blocks, size_t num_const_state_blocks = 0)

  Sets the active block count and the active constant-id count (the first
  ``num_const_state_blocks`` entries of the constant-id buffer, each below
  ``num_blocks``). Every batch starts with 0 active blocks: call this before
  the first solve, and again whenever the sizes change. Host-only (no
  allocation, no device work); takes effect at the next ``Plus`` /
  ``Minimize``.

  :param ``num_blocks``: [in] Active block count, at most ``Capacity()``.
  :param ``num_const_state_blocks``: [in] Active constant count, at most ``ConstCapacity()``.
  :throws std::invalid_argument: if a count exceeds its capacity.

.. cpp:function:: void Plus(const float* x, const float* delta, float* x_plus_delta, cudaStream_t stream, size_t num_replicas = 1)

  Computes :math:`x_{\mathrm{out}} = x \oplus \delta` for every state block
  in the arrays.

  **Terms.** :math:`N` = ``NumStateBlocks()``, :math:`A` = ``AmbientSize()``
  (floats stored per block, e.g. 16 for an SE(3) matrix), :math:`T` =
  ``TangentSize()`` (floats per update, e.g. 6 for SE(3)), :math:`R` =
  ``num_replicas``. The arrays hold :math:`R` contiguous copies
  ("replicas") of the batch, :math:`R \cdot N` blocks in total; replica
  :math:`r` is blocks :math:`[rN, (r+1)N)`. The regular minimizers pass
  :math:`R = 1` (the classic 4-argument call); the
  :doc:`RANSAC minimizers <../ransac>` keep one replica per hypothesis and
  update all of them in one call.

  Every block is updated independently: output block :math:`i` depends only
  on block :math:`i` of ``x`` and block :math:`i` of ``delta``.

  :param ``x``: [in] Device array of :math:`R \cdot N \cdot A` floats; block
    :math:`i` is ``x[i * A .. (i + 1) * A)``.
  :param ``delta``: [in] Device array of :math:`R \cdot N \cdot T` floats;
    block :math:`i`'s update is ``delta[i * T .. (i + 1) * T)``.
  :param ``x_plus_delta``: [out] Device array of :math:`R \cdot N \cdot A`
    floats, same layout as ``x``. Must not overlap ``x`` or ``delta``.
  :param ``stream``: [in] CUDA stream on which all work is enqueued; the call
    may return before the work completes.
  :param ``num_replicas``: [in] :math:`R \geq 1` (default 1).
  :returns: [out] No return value.

  **Example** (:math:`N = 2` blocks, :math:`R = 3` replicas: 6 blocks in every
  array, block :math:`i` of ``x`` at ``x + i * A``, of ``delta`` at
  ``delta + i * T``):

  .. code-block:: text

     global block i     0     1  |  2     3  |  4     5
     replica r          0     0  |  1     1  |  2     2
     block within r     0     1  |  0     1  |  0     1

  **Implementing it.** Treat the arrays as one batch of :math:`R \cdot N`
  blocks, e.g. one thread per block with :math:`i < R N`, and size any
  internal scratch for :math:`R \cdot N` blocks, not :math:`N`. See
  :doc:`../custom_factors_and_states`.

.. cpp:function:: float* StateBlockDevicePtr(size_t state_block_idx)

  :param ``state_block_idx``: [in] Zero-based index of state block.
  :returns: [out] Mutable device pointer for the selected block, or ``nullptr`` when out-of-range.

.. cpp:function:: const float* StateBlockDevicePtr(size_t state_block_idx) const

  :param ``state_block_idx``: [in] Zero-based index of state block.
  :returns: [out] Const device pointer for the selected block, or ``nullptr`` when out-of-range.

.. cpp:function:: const int* ConstStateIds() const

  :returns: [out] Device pointer to constant-state indices, or ``nullptr`` when none are set.

.. cpp:function:: size_t NumConstStateBlocks() const

  :returns: [out] Number of active constant (non-optimized) state blocks.

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

--------------------------------------------------------------------------------
SL4StateBatch
--------------------------------------------------------------------------------

Header: :code:`cunls/state/sl4_state_batch.h`

Projective special linear group SL(4). The tangent space is the 15-dimensional
Lie algebra :math:`\mathfrak{sl}(4)`
(:math:`\mathfrak{so}(4) \oplus \mathrm{sym\_off}(4) \oplus \mathrm{diag}_0(4)`).

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
     - 15
     - 4×4 matrix with unit determinant
     - 15D :math:`\mathfrak{sl}(4)` Lie algebra
     - row-major 4×4 (16 floats)

.. _state-constructors:

================================================================================
Constructors
================================================================================

--------------------------------------------------------------------------------
SizedStateBatch<AmbientDim, TangentDim> (constructors)
--------------------------------------------------------------------------------

.. cpp:function:: SizedStateBatch(const float* device_ptr, size_t capacity)

  :param ``device_ptr``: [in] Device pointer to contiguous state storage (capacity × AmbientDim floats).
  :param ``capacity``: [in] Number of state blocks the buffer holds. 0 are active until ``SetNumStateBlocks``.
  :returns: [out] Constructor has no return value.

.. cpp:function:: SizedStateBatch(const float* device_ptr, size_t capacity, const int* device_constant_state_ids, size_t const_capacity)

  :param ``device_ptr``: [in] Device pointer to contiguous state storage.
  :param ``capacity``: [in] Number of state blocks the buffer holds. 0 are active until ``SetNumStateBlocks``.
  :param ``device_constant_state_ids``: [in] Device pointer to indices of constant blocks.
  :param ``const_capacity``: [in] Number of entries the constant-id buffer holds. 0 are active until ``SetNumStateBlocks``.
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

.. cpp:function:: ClassName(cuBLASHandle& cublas_handle, const float* device_ptr, size_t capacity)
.. cpp:function:: ClassName(cuBLASHandle& cublas_handle, const float* device_ptr, size_t capacity, const int* device_constant_state_ids, size_t const_capacity)

  :param ``cublas_handle``: [in] External cuBLAS handle wrapper.
  :param ``device_ptr``: [in] Device pointer to contiguous state storage.
  :param ``capacity``: [in] Number of state blocks the buffer holds. 0 are active until ``SetNumStateBlocks``.
  :param ``device_constant_state_ids``: [in] Device pointer to constant block indices.
  :param ``const_capacity``: [in] Number of entries the constant-id buffer holds. 0 are active until ``SetNumStateBlocks``.
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

================================================================================
Python API (``pycunls``)
================================================================================

All Python state batches inherit from the abstract ``StateBatch`` base class.
Every constructor argument documented as ``DevicePointer`` accepts either a
``cupy.ndarray`` (the device pointer is extracted automatically via
``.data.ptr``) or a raw ``int`` GPU device address.

.. _py-state-batch-interface:

--------------------------------------------------------------------------------
Common ``StateBatch`` interface
--------------------------------------------------------------------------------

Every state batch — built-in or user-defined — exposes the following methods
and properties.

**Methods**

- ``state_block_device_ptr(index: int) -> int`` — returns the GPU device
  pointer (as an ``int``) for state block *index*.  The returned value is
  the address of the first float in the block's ambient storage.  Use these
  pointers to build the ``state_pointers`` list passed to
  :ref:`Problem.add_factor_batch <py-problem-label>`.  *index* is
  zero-based and may be any slot below ``capacity`` (so connectivity can be
  built before the active count is set); passing a value ``>= capacity``
  returns ``0`` (null pointer).
- ``set_num_state_blocks(num_blocks, num_const_state_blocks=0)`` — sets the
  active block count (the first ``num_blocks`` blocks of the buffer) and the
  active constant-id count. Every batch starts with 0 active blocks: call it
  before the first solve, and again whenever the sizes change. Host-only;
  takes effect at the next ``minimize``. Raises ``ValueError`` above the
  capacity.

**Read-only properties**

- **num_state_blocks** (``int``) — number of active state blocks, including
  any active constant blocks (0 until ``set_num_state_blocks``).
- **capacity** (``int``) — number of state blocks the buffer holds (the
  constructor's ``capacity``); constant.
- **const_capacity** (``int``) — number of entries the constant-id buffer
  holds (the constructor's ``const_capacity``, 0 without one).
- **tangent_size** (``int``) — tangent-space dimension per state block.
  This is the number of unknowns the solver allocates per block (e.g. 6 for
  SE(3), 3 for SO(3)).
- **ambient_size** (``int``) — ambient/storage dimension per state block.
  The GPU buffer stores ``capacity * ambient_size`` contiguous
  floats (e.g. 16 for SE(3) = row-major 4×4 matrix).

.. _py-vector-state-batches:

------------------------------------------------------------------------------------------------------
``pycunls.VectorStateBatch1`` / ``VectorStateBatch2`` / ``VectorStateBatch3`` / ``VectorStateBatch6``
------------------------------------------------------------------------------------------------------

Euclidean vector states where tangent and ambient dimensions coincide.  The
suffix indicates the dimension (1, 2, 3, or 6).  Plus is simple addition:
:math:`x \oplus \delta = x + \delta`.

**Constructors**

.. code-block:: python

   # All optimizable:
   sb = pycunls.VectorStateBatch3(data, capacity)

   # With constant (frozen) blocks:
   sb = pycunls.VectorStateBatch3(data, capacity, const_state_ids, const_capacity)

- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``capacity × Dim`` floats.  The state batch does **not** copy the data;
  it stores the pointer and reads/writes the buffer directly.  The caller
  must keep the underlying allocation alive for the lifetime of the state
  batch.
- **capacity** (``int``) — number of state blocks the buffer holds. The
  batch starts with 0 active blocks: call ``set_num_state_blocks`` before
  solving.
- **const_state_ids** (``DevicePointer``, optional) — GPU ``int32`` array
  containing the zero-based indices of blocks that should be held constant
  during optimization.  Constant blocks are excluded from the solver's
  tangent vector; their ambient values are never modified.
- **const_capacity** (``int``, optional) — number of entries the
  *const_state_ids* buffer holds (the active count is set with
  ``set_num_state_blocks``).

.. _py-lie-state-batches:

--------------------------------------------------------------------------------
``pycunls.SE3StateBatch``
--------------------------------------------------------------------------------

3-D rigid-body transform state batch.  Ambient = 16 (row-major 4×4
homogeneous matrix), Tangent = 6 (twist :math:`[\omega; \rho]`).  Plus is
right-multiplication by the exponential map:
:math:`T \oplus \delta = T \cdot \mathrm{Exp}(\delta)`.

**Constructors**

.. code-block:: python

   cublas = pycunls.CublasHandle()

   # All optimizable:
   sb = pycunls.SE3StateBatch(cublas, data, capacity)

   # With constant blocks:
   sb = pycunls.SE3StateBatch(cublas, data, capacity, const_ids, const_capacity)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle used internally for matrix operations in the exponential map.
- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``capacity × 16`` floats (row-major 4×4 matrices).
- **capacity** (``int``) — number of state blocks (poses) the buffer
  holds; 0 are active until ``set_num_state_blocks``.
- **const_ids** (``DevicePointer``, optional) — GPU ``int32`` array of
  constant-block indices (e.g. a gauge anchor).
- **const_capacity** (``int``, optional) — number of entries the
  *const_ids* buffer holds.

--------------------------------------------------------------------------------
``pycunls.SO3StateBatch``
--------------------------------------------------------------------------------

3-D rotation state batch.  Ambient = 9 (row-major 3×3 rotation matrix),
Tangent = 3 (rotation vector / axis-angle).  Plus:
:math:`R \oplus \delta = R \cdot \mathrm{Exp}(\mathrm{skew}(\delta))`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SO3StateBatch(cublas, data, capacity)
   sb = pycunls.SO3StateBatch(cublas, data, capacity, const_ids, const_capacity)

- **data** — ``capacity × 9`` floats (row-major 3×3).

--------------------------------------------------------------------------------
``pycunls.SO2StateBatch``
--------------------------------------------------------------------------------

2-D rotation state batch.  Ambient = 4 (row-major 2×2 rotation matrix),
Tangent = 1 (angle in radians).  Plus:
:math:`R \oplus \delta = R \cdot \mathrm{Exp}(\delta)`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SO2StateBatch(cublas, data, capacity)
   sb = pycunls.SO2StateBatch(cublas, data, capacity, const_ids, const_capacity)

- **data** — ``capacity × 4`` floats
  (:math:`[\cos\theta,\,-\sin\theta,\,\sin\theta,\,\cos\theta]`).

--------------------------------------------------------------------------------
``pycunls.SE2StateBatch``
--------------------------------------------------------------------------------

2-D rigid-body transform state batch.  Ambient = 9 (row-major 3×3
homogeneous matrix), Tangent = 3 (:math:`[v_x, v_y, \theta]`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SE2StateBatch(cublas, data, capacity)
   sb = pycunls.SE2StateBatch(cublas, data, capacity, const_ids, const_capacity)

- **data** — ``capacity × 9`` floats (row-major 3×3).

.. _py-similarity-state-batches:

--------------------------------------------------------------------------------
``pycunls.Similarity2StateBatch``
--------------------------------------------------------------------------------

2-D similarity transform state batch.  Ambient = 9, Tangent = 4
(:math:`[u_x, u_y, \theta, \lambda]` where :math:`\lambda = \log s`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.Similarity2StateBatch(cublas, data, capacity)
   sb = pycunls.Similarity2StateBatch(cublas, data, capacity, const_ids, const_capacity)

--------------------------------------------------------------------------------
``pycunls.Similarity3StateBatch``
--------------------------------------------------------------------------------

3-D similarity transform state batch.  Ambient = 16, Tangent = 7
(:math:`[\omega; u; \lambda]` where :math:`\lambda = \log s`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.Similarity3StateBatch(cublas, data, capacity)
   sb = pycunls.Similarity3StateBatch(cublas, data, capacity, const_ids, const_capacity)

--------------------------------------------------------------------------------
``pycunls.SL4StateBatch``
--------------------------------------------------------------------------------

SL(4) state batch.  Ambient = 16 (row-major 4×4 matrix with unit determinant),
Tangent = 15 (:math:`\mathfrak{sl}(4)` Lie algebra).
Plus: :math:`T \oplus \delta = T \cdot \mathrm{Exp}(\delta)`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SL4StateBatch(cublas, data, capacity)
   sb = pycunls.SL4StateBatch(cublas, data, capacity, const_ids, const_capacity)

- **data** — ``capacity × 16`` floats (row-major 4×4).

.. _py-custom-state-batch:

--------------------------------------------------------------------------------
``pycunls.CustomStateBatch``
--------------------------------------------------------------------------------

Base class for user-defined state batches.  Subclass this to implement a
manifold retraction that is not available as a built-in (e.g. positive
scalars, quaternions, constrained subspaces).

**Constructor**

.. code-block:: python

   class MyState(pycunls.CustomStateBatch):
       def __init__(self, data, capacity):
           super().__init__(
               data,
               ambient_size=...,
               tangent_size=...,
               capacity=capacity,
           )

- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``capacity × ambient_size`` floats.
- **ambient_size** (``int``) — number of floats per state block in GPU
  memory.
- **tangent_size** (``int``) — number of tangent-space unknowns per block.
- **capacity** (``int``) — number of state blocks the buffer holds. The
  batch starts with 0 active blocks: call ``set_num_state_blocks`` before
  solving.
- **const_state_ids** (``DevicePointer``, optional) — GPU ``int32`` array
  of constant-block indices.
- **const_capacity** (``int``, default ``0``) — number of entries the
  *const_state_ids* buffer holds.

**Methods to override**

- ``plus(x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas) -> None`` —
  implements the manifold retraction
  :math:`x_{\mathrm{out}} = x \oplus \delta` for **all blocks** in the
  arrays (the same contract as C++ :cpp:func:`StateBatch::Plus`).  The
  arrays hold ``num_replicas`` contiguous copies of the batch, i.e.
  ``R × num_state_blocks`` blocks with ``R = num_replicas``; replica *r* is blocks
  ``[r * num_state_blocks, (r + 1) * num_state_blocks)``.  All five arguments are raw
  ``int`` values:

  - *x_ptr* — device pointer to the current ambient state
    (``R × num_state_blocks × ambient_size`` floats, read-only).
  - *delta_ptr* — device pointer to the tangent-space updates
    (``R × num_state_blocks × tangent_size`` floats, read-only).
  - *x_plus_delta_ptr* — device pointer to the output buffer
    (``R × num_state_blocks × ambient_size`` floats, write; does not overlap the
    inputs).
  - *stream_handle* — ``cudaStream_t`` cast to ``int``.  All GPU work
    **must** be launched on this stream so the minimizer can serialize
    operations correctly.
  - *num_replicas* — ``R >= 1``. The regular minimizers pass ``1``; the
    RANSAC minimizers pass one replica per hypothesis, so a state used with
    RANSAC must process every replica (see
    :doc:`../custom_factors_and_states`).

  The default implementation raises ``NotImplementedError``.

.. _py-warp-state-batch:

--------------------------------------------------------------------------------
``pycunls.warp.WarpStateBatch``
--------------------------------------------------------------------------------

Convenience base for custom state batches implemented with `NVIDIA Warp
<https://developer.nvidia.com/warp-python>`_ kernels.  Inherits from ``CustomStateBatch`` and provides helper methods for
zero-copy pointer wrapping so you never need to manually construct
``wp.array`` objects from raw device addresses.  Requires ``warp-lang``.

**Constructor**

.. code-block:: python

   from pycunls.warp import WarpStateBatch

   class MyWarpState(WarpStateBatch):
       def __init__(self, data, capacity):
           super().__init__(
               data,
               ambient_size=...,
               tangent_size=...,
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

- ``make_warp_stream(stream_handle: int) -> wp.Stream`` — wraps a raw
  ``cudaStream_t`` (passed as ``int``) as a ``wp.Stream``.  Use the
  returned stream in ``wp.launch(..., stream=stream)`` to ensure the Warp
  kernel executes on the minimizer's CUDA stream.

**Methods to override**

- ``plus(x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas) -> None`` —
  same contract as ``CustomStateBatch.plus``.  Typical implementations wrap
  the pointers with ``self.wrap_array`` (sized for
  ``num_replicas × num_state_blocks`` blocks), build a ``wp.Stream`` with
  ``self.make_warp_stream``, and launch a ``@wp.kernel`` with one thread per
  block.

See :ref:`pycunls_tutorial:Custom Warp State` for a complete example.
