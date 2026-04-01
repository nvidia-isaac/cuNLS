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
  zero-based; passing a value ``>= num_state_blocks`` returns ``0`` (null
  pointer).

**Read-only properties**

- **num_state_blocks** (``int``) — total number of state blocks in the
  batch, including any constant blocks.
- **tangent_size** (``int``) — tangent-space dimension per state block.
  This is the number of unknowns the solver allocates per block (e.g. 6 for
  SE(3), 3 for SO(3)).
- **ambient_size** (``int``) — ambient/storage dimension per state block.
  The GPU buffer stores ``num_state_blocks * ambient_size`` contiguous
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
   sb = pycunls.VectorStateBatch3(data, num_blocks)

   # With constant (frozen) blocks:
   sb = pycunls.VectorStateBatch3(data, num_blocks, const_state_ids, num_const)

- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``num_blocks × Dim`` floats.  The state batch does **not** copy the data;
  it stores the pointer and reads/writes the buffer directly.  The caller
  must keep the underlying allocation alive for the lifetime of the state
  batch.
- **num_blocks** (``int``) — number of state blocks in the batch.
- **const_state_ids** (``DevicePointer``, optional) — GPU ``int32`` array
  containing the zero-based indices of blocks that should be held constant
  during optimization.  Constant blocks are excluded from the solver's
  tangent vector; their ambient values are never modified.
- **num_const** (``int``, optional) — number of entries in
  *const_state_ids*.

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
   sb = pycunls.SE3StateBatch(cublas, data, num_blocks)

   # With constant blocks:
   sb = pycunls.SE3StateBatch(cublas, data, num_blocks, const_ids, num_const)

- **cublas** (:ref:`CublasHandle <py-cublas-handle-label>`) — shared cuBLAS
  handle used internally for matrix operations in the exponential map.
- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``num_blocks × 16`` floats (row-major 4×4 matrices).
- **num_blocks** (``int``) — number of state blocks (poses).
- **const_ids** (``DevicePointer``, optional) — GPU ``int32`` array of
  constant-block indices (e.g. a gauge anchor).
- **num_const** (``int``, optional) — number of constant blocks.

--------------------------------------------------------------------------------
``pycunls.SO3StateBatch``
--------------------------------------------------------------------------------

3-D rotation state batch.  Ambient = 9 (row-major 3×3 rotation matrix),
Tangent = 3 (rotation vector / axis-angle).  Plus:
:math:`R \oplus \delta = R \cdot \mathrm{Exp}(\mathrm{skew}(\delta))`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SO3StateBatch(cublas, data, num_blocks)
   sb = pycunls.SO3StateBatch(cublas, data, num_blocks, const_ids, num_const)

- **data** — ``num_blocks × 9`` floats (row-major 3×3).

--------------------------------------------------------------------------------
``pycunls.SO2StateBatch``
--------------------------------------------------------------------------------

2-D rotation state batch.  Ambient = 4 (row-major 2×2 rotation matrix),
Tangent = 1 (angle in radians).  Plus:
:math:`R \oplus \delta = R \cdot \mathrm{Exp}(\delta)`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SO2StateBatch(cublas, data, num_blocks)
   sb = pycunls.SO2StateBatch(cublas, data, num_blocks, const_ids, num_const)

- **data** — ``num_blocks × 4`` floats
  (:math:`[\cos\theta,\,-\sin\theta,\,\sin\theta,\,\cos\theta]`).

--------------------------------------------------------------------------------
``pycunls.SE2StateBatch``
--------------------------------------------------------------------------------

2-D rigid-body transform state batch.  Ambient = 9 (row-major 3×3
homogeneous matrix), Tangent = 3 (:math:`[v_x, v_y, \theta]`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SE2StateBatch(cublas, data, num_blocks)
   sb = pycunls.SE2StateBatch(cublas, data, num_blocks, const_ids, num_const)

- **data** — ``num_blocks × 9`` floats (row-major 3×3).

.. _py-similarity-state-batches:

--------------------------------------------------------------------------------
``pycunls.Similarity2StateBatch``
--------------------------------------------------------------------------------

2-D similarity transform state batch.  Ambient = 9, Tangent = 4
(:math:`[u_x, u_y, \theta, \lambda]` where :math:`\lambda = \log s`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.Similarity2StateBatch(cublas, data, num_blocks)
   sb = pycunls.Similarity2StateBatch(cublas, data, num_blocks, const_ids, num_const)

--------------------------------------------------------------------------------
``pycunls.Similarity3StateBatch``
--------------------------------------------------------------------------------

3-D similarity transform state batch.  Ambient = 16, Tangent = 7
(:math:`[\omega; u; \lambda]` where :math:`\lambda = \log s`).

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.Similarity3StateBatch(cublas, data, num_blocks)
   sb = pycunls.Similarity3StateBatch(cublas, data, num_blocks, const_ids, num_const)

--------------------------------------------------------------------------------
``pycunls.SL4StateBatch``
--------------------------------------------------------------------------------

SL(4) state batch.  Ambient = 16 (row-major 4×4 matrix with unit determinant),
Tangent = 15 (:math:`\mathfrak{sl}(4)` Lie algebra).
Plus: :math:`T \oplus \delta = T \cdot \mathrm{Exp}(\delta)`.

**Constructors** — same pattern as ``SE3StateBatch``:

.. code-block:: python

   sb = pycunls.SL4StateBatch(cublas, data, num_blocks)
   sb = pycunls.SL4StateBatch(cublas, data, num_blocks, const_ids, num_const)

- **data** — ``num_blocks × 16`` floats (row-major 4×4).

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
       def __init__(self, data, num_blocks):
           super().__init__(
               data,
               ambient_size=...,
               tangent_size=...,
               num_blocks=num_blocks,
           )

- **data** (``DevicePointer``) — contiguous GPU buffer of
  ``num_blocks × ambient_size`` floats.
- **ambient_size** (``int``) — number of floats per state block in GPU
  memory.
- **tangent_size** (``int``) — number of tangent-space unknowns per block.
- **num_blocks** (``int``) — number of state blocks.
- **const_state_ids** (``DevicePointer``, optional) — GPU ``int32`` array
  of constant-block indices.
- **num_const_state_blocks** (``int``, default ``0``) — number of constant
  blocks.

**Methods to override**

- ``plus(x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle) -> None`` —
  implements the manifold retraction
  :math:`x_{\mathrm{out}} = x \oplus \delta` for **all blocks** in the
  batch.  All four arguments are raw ``int`` handles:

  - *x_ptr* — device pointer to the current ambient state
    (``num_blocks × ambient_size`` floats, read-only).
  - *delta_ptr* — device pointer to the tangent-space updates
    (``num_blocks × tangent_size`` floats, read-only).
  - *x_plus_delta_ptr* — device pointer to the output buffer
    (``num_blocks × ambient_size`` floats, write).
  - *stream_handle* — ``cudaStream_t`` cast to ``int``.  All GPU work
    **must** be launched on this stream so the minimizer can serialize
    operations correctly.

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
       def __init__(self, data, num_blocks):
           super().__init__(
               data,
               ambient_size=...,
               tangent_size=...,
               num_blocks=num_blocks,
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

- ``plus(x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle) -> None`` —
  same contract as ``CustomStateBatch.plus``.  Typical implementations wrap
  the pointers with ``self.wrap_array``, build a ``wp.Stream`` with
  ``self.make_warp_stream``, and launch a ``@wp.kernel``.

See :ref:`pycunls_tutorial:Custom Warp State` for a complete example.
