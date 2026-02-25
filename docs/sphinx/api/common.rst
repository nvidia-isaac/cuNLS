################################################################################
Common API
################################################################################

`cunls/common` provides utility containers, type aliases, profiling wrappers,
and CUDA library handle abstractions.

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
