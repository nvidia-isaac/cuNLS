################################################################################
Linear Solver API
################################################################################

`cunls/linear_solver` hosts linear-system abstractions (cuDSS integration,
dense pivoted LDLT, dense Cholesky, and dense QR solvers) behind a common
CSR-based interface.

SparseLinearSolverType
----------------------

Enum in `cunls/linear_solver/sparse_linear_solver.h`:

- `cuDSS`
- `DenseLDLT`
- `DenseCholesky`
- `DenseQR`

SparseLinearSolverConfig
------------------------

Struct in `sparse_linear_solver.h`. Only the member corresponding to the
chosen ``SparseLinearSolverType`` is used:

- `cudss_solver_options` - [in] cuDSS-specific backend options (ignored when
  using ``DenseLDLT``, ``DenseCholesky``, or ``DenseQR``).

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

.. cpp:function:: void DisableSafetyChecks()

  Disables runtime safety checks in the solver.  By default (safety checks
  enabled), dense solvers validate every factorization and solve step:
  Cholesky checks cuSOLVER ``devInfo`` after ``potrf`` and ``potrs``; QR
  inspects the diagonal of ``R`` for rank deficiency; LDLT performs in-kernel
  pivot and diagonal checks.  Failures cause ``Solve()`` to return ``false``
  with a diagnostic via ``LogError()``.  Calling this method skips all of
  the above (no device-to-host memcpy, no stream synchronization, no
  in-kernel validation), which can noticeably reduce per-iteration latency
  for small systems but may produce silently incorrect results for singular
  or ill-conditioned matrices.  Normally called by the minimizer when
  ``MinimizerOptions::disable_safety_checks`` is ``true``.

.. cpp:function:: bool SafetyChecksEnabled() const

  :returns: ``true`` when post-factorization safety checks are enabled
    (the default).

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

DenseCholeskySolver
-------------------

Concrete implementation in `dense_cholesky_solver.h`. Uses the cuSOLVER
``cusolverDnSpotrf`` / ``cusolverDnSpotrs`` routines (Cholesky factorization
``A = L L^T``). Requires the input matrix to be symmetric positive-definite.

.. cpp:function:: bool DenseCholeskySolver::Initialize(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Validates dimensions and pre-allocates internal dense buffers and the
  cuSOLVER workspace for the given matrix size.

  :param ``stream``: [in] CUDA stream used to query the cuSOLVER workspace size.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool DenseCholeskySolver::Solve(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Converts CSR to dense, performs Cholesky factorization via
  ``cusolverDnSpotrf``, then solves via ``cusolverDnSpotrs``. When safety
  checks are enabled, inspects cuSOLVER ``devInfo`` after both ``potrf``
  and ``potrs``: returns ``false`` if the matrix is not positive-definite
  (``devInfo > 0`` from ``potrf``) or if ``potrs`` reports an invalid
  parameter (``devInfo < 0``).

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` on dimension mismatch,
    non-positive-definite matrix, or invalid parameter from ``potrs``.

DenseQRSolver
-------------

Concrete implementation in `dense_qr_solver.h`. Uses the cuSOLVER
``cusolverDnSgeqrf`` / ``cusolverDnSormqr`` routines for QR factorization
(``A = Q R``) followed by a cuBLAS ``cublasStrsm`` upper-triangular solve.
Works for any non-singular square matrix (not limited to SPD).

.. cpp:function:: bool DenseQRSolver::Initialize(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Validates dimensions and pre-allocates internal dense buffers, Householder
  tau vector, and the cuSOLVER workspace for the given matrix size.

  :param ``stream``: [in] CUDA stream used to query the cuSOLVER workspace size.
  :param ``spd_matrix``: [in] Matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool DenseQRSolver::Solve(cudaStream_t stream, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Converts CSR to dense, performs QR factorization via ``cusolverDnSgeqrf``,
  applies ``Q^T`` to the RHS via ``cusolverDnSormqr``, then solves the
  upper-triangular system ``R x = Q^T b`` via ``cublasStrsm``.

  :param ``stream``: [in] CUDA stream for asynchronous GPU operations.
  :param ``spd_matrix``: [in] Matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` on dimension mismatch or
    singular matrix (cuSOLVER ``devInfo != 0``).

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
