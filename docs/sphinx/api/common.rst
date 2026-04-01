################################################################################
Common API
################################################################################

``cunls/common`` provides utility containers, type aliases, profiling wrappers,
and CUDA library handle abstractions.

**C++** — ``cunls/common``
  |  **Python** — ``pycunls``

Core types
----------

Defined in `cunls/common/types.h`:

- `Vector<Dim>`: fixed-size float vector (`cuda::std::array`).
- `Matrix<Dim>`: row-major square matrix.
- `SE3Transform`: alias of `Matrix<4>`.
- `dvector<T>`: alias for `DeviceVector<T>`.
- `hvector<T>`: alias for `std::vector<T>`.

Sparse matrix structs
---------------------

- `CSRSparseMatrix`
  - `row_offsets` - [in/out] CSR row offsets.
  - `col_ids` - [in/out] CSR column indices.
  - `values` - [in/out] CSR non-zero values.
  - methods: `NumRows()`, `NumNonZeros()`.
- `TripletSparseStructure`
  - `row_ids` - [in/out] triplet row indices.
  - `col_ids` - [in/out] triplet column indices.
- `SparseJacobian`
  - `structure` - [in/out] sparse structure.
  - `values` - [in/out] sparse values.

DeviceVector<T>
---------------

Header: `cunls/common/device_vector.h`

RAII GPU vector with sync/async host-device copies.

Constructors:

- `DeviceVector()`
- `DeviceVector(size_t num_elements)`
- `DeviceVector(const T& fill_value, size_t num_elements)`
- `DeviceVector(const std::vector<T>& host_vector)`
- move constructor / move assignment

Primary copy APIs
^^^^^^^^^^^^^^^^^

CopyFromHost
^^^^^^^^^^^^

``void CopyFromHost(const T* src, size_t num_elements)`` — ``src`` [in] Host pointer source buffer; ``num_elements`` [in] number of elements to copy to device.

CopyToHost
^^^^^^^^^^

``void CopyToHost(T* dst, size_t num_elements) const`` — ``dst`` [out] Host pointer destination buffer; ``num_elements`` [in] number of elements to copy from device.

CopyFromHostAsync
^^^^^^^^^^^^^^^^^

``void CopyFromHostAsync(const T* src, size_t num_elements, CudaStream& stream)`` — ``src`` [in] host source (prefer pinned for async); ``num_elements`` [in]; ``stream`` [in].

CopyToHostAsync
^^^^^^^^^^^^^^^^

``void CopyToHostAsync(T* dst, size_t num_elements, CudaStream& stream) const`` — ``dst`` [out] host destination; ``num_elements`` [in]; ``stream`` [in].

PinnedVector<T>
---------------

Header: `cunls/common/pinned_vector.h`

RAII container for CUDA page-locked (pinned) host memory. Pinned memory
enables fully asynchronous device-to-host and host-to-device transfers via
``cudaMemcpyAsync``. Alias: ``pvector<T>`` (defined in ``types.h``).

Constructors:

- ``PinnedVector()``
- ``PinnedVector(size_t num_elements)``
- move constructor / move assignment

Accessors:

- ``T* data()`` — [out] pointer to pinned host memory.
- ``size_t size()`` — [out] number of elements.
- ``size_t capacity()`` — [out] number of elements allocated.
- ``T& operator[](size_t i)`` — element access.

Mutators:

- ``void resize(size_t new_size, bool preserve_data = false)`` —
  ``new_size`` [in] desired element count; ``preserve_data`` [in] if true,
  existing data is copied when reallocation occurs. Reuses the existing
  allocation when ``new_size <= capacity``.

CudaStream
----------

Header: `cunls/common/cuda_stream.h`

Constructor:

- `CudaStream(bool sync_on_destroy = false)`

Method:

- ``cudaStream_t& GetStream()`` — [out] underlying CUDA stream handle.

cuBLASHandle
------------

Header: `cunls/common/cublas_helper.h`

Method:

- ``cublasHandle_t GetHandle(cudaStream_t stream)`` — ``stream`` [in] stream to bind; returns [out] cuBLAS handle.

cuSPARSE wrappers
-----------------

Header: `cunls/common/cusparse_helper.h`

- ``cuSPARSEHandle`` — ``cusparseHandle_t GetHandle(cudaStream_t stream)``; ``stream`` [in]; returns [out] cuSPARSE handle.
- ``cuSPARSEMatrixDescription`` — CSR/empty constructors; ``void UpdatePointers(const CSRSparseMatrix& matrix)`` with ``matrix`` [in]; ``cusparseSpMatDescr_t GetDescription()`` returns [out] matrix descriptor.
- ``cuSPARSEVectorDescription`` — ``cuSPARSEVectorDescription(const dvector<float>& vec)`` with ``vec`` [in]; ``cusparseDnVecDescr_t GetDescription()`` returns [out] vector descriptor.

cuSolver wrappers
-----------------

Header: `cunls/common/cusolver_helper.h`

- ``cuSolverHandle`` — ``cusolverDnHandle_t GetHandle(cudaStream_t stream)``; ``stream`` [in]; returns [out] cuSolver handle.
- ``cuSolverInfo`` — ``syevjInfo_t GetInfo() const`` returns [out] eigen-solver info handle.

cuDSS wrappers and pool
-----------------------

Header: `cunls/common/cudss_helper.h`

- ``cuDSSDeviceMemPool`` — ``Alloc(ptr, size, stream)``: ``ptr`` [out], ``size`` [in], ``stream`` [in]; ``Dealloc(ptr, size, stream)``: ``ptr`` [in], ``size`` [in], ``stream`` [in].
- ``cuDSSHandle`` — ``void* GetHandle(cudaStream_t stream)``; ``stream`` [in]; returns [out] opaque cuDSS handle.
- ``cuDSSDescription`` — constructors from ``CSRSparseMatrix`` or ``dvector<float>``; ``void* GetDescription()`` returns [out] matrix/vector descriptor.
- ``cuDSSConfig`` — ``cuDSSConfig(reordering_algorithm, nthreads)``: ``reordering_algorithm`` [in], ``nthreads`` [in]; ``void* GetData() const`` returns [out] config pointer.
- ``cuDSSData`` — ``void* GetData(void* handle)``: ``handle`` [in]; returns [out] opaque cuDSS data pointer.

Profiling utilities
-------------------

Header: `cunls/common/profiler.h`

- ``profiler::ScopedRange(const std::string& name)`` — ``name`` [in] NVTX range label.
- ``profiler::Domain(const std::string& name)`` — ``name`` [in] NVTX domain label.
- ``profiler::Domain::CreateDomainRange(const std::string& name)`` — ``name`` [in] range label; returns [out] scoped range RAII object.

Python API (``pycunls``)
------------------------

The Python bindings expose CUDA stream and cuBLAS handle wrappers through the
``pycunls`` package.  These are utility types shared by state batches, factor
batches, and minimizers.

.. _py-cuda-stream-label:

``pycunls.CudaStream``
^^^^^^^^^^^^^^^^^^^^^^

RAII wrapper around a ``cudaStream_t``.  Every pycunls ``minimize`` call
requires a stream to serialize asynchronous GPU work (factor evaluation,
linear algebra, state updates).

**Constructor**

.. code-block:: python

   stream = pycunls.CudaStream(sync_on_destroy=False)

- **sync_on_destroy** (``bool``, default ``False``) — when ``True``, the
  destructor calls ``cudaStreamSynchronize`` before destroying the stream.
  Useful for debugging; in production leave ``False`` and synchronize
  explicitly after ``minimize``.

**Methods**

- ``get_stream() -> int`` — returns the underlying ``cudaStream_t`` cast to
  an integer.  Pass this value to ``cp.cuda.runtime.streamSynchronize`` to
  wait for all GPU work issued on this stream, or to `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_
  (``wp.Stream(cuda_stream=handle)``) when authoring custom kernels.

.. _py-cublas-handle-label:

``pycunls.CublasHandle``
^^^^^^^^^^^^^^^^^^^^^^^^

RAII wrapper around a ``cublasHandle_t``.  Required as the first argument by
every Lie-group and similarity state batch constructor
(``SE3StateBatch``, ``SO3StateBatch``, ``SO2StateBatch``, ``SE2StateBatch``,
``Similarity2StateBatch``, ``Similarity3StateBatch``) and by factor batches
that operate on those manifolds (``SE3BetweenFactorBatch``,
``ReprojectionFactorBatch``, ``PnPFactorBatch``, ``SE3PriorFactorBatch``,
``SO3PriorFactorBatch``).
Euclidean state and factor batches (``VectorStateBatch*``,
``PriorVectorFactorBatch*``, ICP factors) do **not** need it.

**Constructor**

.. code-block:: python

   cublas = pycunls.CublasHandle()

Creates the handle lazily; the first cuBLAS call binds it to the active CUDA
device.  A single ``CublasHandle`` instance may be shared across all state
and factor batches in a problem.
