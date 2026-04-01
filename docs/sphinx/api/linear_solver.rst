################################################################################
Linear Solver API
################################################################################

`cunls/linear_solver` hosts linear-system abstractions (cuDSS integration and
a dense pivoted LDLT solver) behind a common CSR-based interface.

SparseLinearSolverType
----------------------

Enum in `cunls/linear_solver/sparse_linear_solver.h`:

- `cuDSS`
- `DenseLDLT`

SparseLinearSolverConfig
------------------------

Struct in `sparse_linear_solver.h`. Only the member corresponding to the
chosen ``SparseLinearSolverType`` is used:

- `cudss_solver_options` - [in] cuDSS-specific backend options (ignored when
  using ``DenseLDLT``).

CSRSparseLinearSolver
---------------------

Abstract base (`cunls/linear_solver/csr_sparse_linear_solver.h`).

.. cpp:function:: bool Initialize(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Performs setup work (at minimum symbolic analysis; some modes also perform
  an initial factorization) for the given sparsity pattern. Must be called
  before :cpp:func:`Solve` and re-called whenever the matrix structure changes.

  Both ``rhs`` and ``result`` must be pre-allocated to the same number of
  elements as the number of rows in ``spd_matrix``; the solver does **not**
  resize them.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations.
  :param ``spd_matrix``: [in] Symmetric matrix in CSR format (backend-specific
    definiteness requirements may apply).
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool Solve(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Performs factorization and solve phases to compute the solution ``x`` of
  ``A x = b``. Both ``rhs`` and ``result`` must be pre-allocated to the same
  number of elements as the number of rows in ``spd_matrix``; the solver does
  **not** resize them. Call :cpp:func:`Initialize` first for the current
  sparsity pattern.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations.
  :param ``spd_matrix``: [in] Symmetric matrix in CSR format (backend-specific
    definiteness requirements may apply).
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

cuDSSLinearSolverMode
---------------------

Enum in `cunls/linear_solver/cudss_sparse_linear_solver.h`:

- `SlowInitFastSolve` — ``Initialize`` performs symbolic analysis only;
  ``Solve`` performs a full factorization followed by the solve phase.
  Best when the matrix values change frequently relative to its structure.
- `FastInitSlowSolve` — ``Initialize`` performs symbolic analysis **and**
  an initial factorization; ``Solve`` performs refactorization followed by
  the solve phase. Best when the sparsity pattern is stable and fast
  initialization is more important than per-solve speed.

cuDSSLinearSolverOptions
------------------------

Struct fields:

- `mode` - [in] Trade-off between setup and repeated solve speed (see
  ``cuDSSLinearSolverMode``).
- `nthreads` - [in] Host thread count for cuDSS host-side stages.
- `threading_lib_path` - [in] Optional threading runtime path for multi-threaded
  host processing. Empty string disables multi-threading.

cuDSSLinearSolver
-----------------

Concrete implementation in `cudss_sparse_linear_solver.h`.

.. cpp:function:: cuDSSLinearSolver(cuDSSLinearSolverOptions options = cuDSSLinearSolverOptions())

  :param ``options``: [in] cuDSS mode/threading configuration.
  :returns: Constructor has no return value.

.. cpp:function:: bool cuDSSLinearSolver::Initialize(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  For ``SlowInitFastSolve``: runs the cuDSS symbolic analysis phase only.
  For ``FastInitSlowSolve``: runs symbolic analysis followed by an initial
  factorization so that subsequent :cpp:func:`cuDSSLinearSolver::Solve` calls
  can use cheaper refactorization.

  Both ``rhs`` and ``result`` must be pre-allocated to the number of rows in
  ``spd_matrix``; the solver does **not** resize them. The internal cuDSS
  handle is lazily initialized on the first call using the provided stream.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations. Also
    used to lazily initialize the internal cuDSS handle on first call.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

DenseLDLTSolver
---------------

Concrete implementation in `dense_linear_solver.h`.

.. cpp:function:: bool DenseLDLTSolver::Initialize(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Validates dimensions and allocates reusable dense buffers used by the dense
  pivoted LDLT pipeline.

  :param ``stream``: [in] CUDA stream (unused; buffers are allocated synchronously).
  :param ``spd_matrix``: [in] Symmetric matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool DenseLDLTSolver::Solve(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Converts CSR to dense row-major storage, computes a CUDA pivoted LDLT
  factorization ``P^T A P = L D L^T``, then solves in the permuted system.
  Handles symmetric matrices including indefinite ones (not limited to SPD).
  Kernel success/failure (singular pivot, zero diagonal) is propagated back
  to the host via a single async device-to-pinned-host copy after both
  kernels complete.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations.
  :param ``spd_matrix``: [in] Symmetric matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` on dimension mismatch, singular
    pivot during factorization, or zero diagonal during the solve phase.

.. cpp:function:: bool cuDSSLinearSolver::Solve(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  For ``SlowInitFastSolve``: performs full factorization then solve. For
  ``FastInitSlowSolve``: performs refactorization (reusing the symbolic
  analysis from ``Initialize``) then solve.

  Both ``rhs`` and ``result`` must be pre-allocated to the number of rows in
  ``spd_matrix``; the solver does **not** resize them and returns ``false``
  on any dimension mismatch.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations. Also
    used to lazily initialize the internal cuDSS handle if needed.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

Factory
-------

Function in `sparse_linear_solver.h`:

.. cpp:function:: SparseLinearSolverPtr CreateCSRSparseLinearSolver(SparseLinearSolverType type, const SparseLinearSolverConfig& config)

  :param ``type``: [in] Backend type to instantiate.
  :param ``config``: [in] Backend-specific configuration blob.
  :returns: [out] Heap-allocated solver instance.
