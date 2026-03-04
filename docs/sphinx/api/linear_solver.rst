################################################################################
Linear Solver API
################################################################################

`cunls/linear_solver` hosts sparse linear-system abstractions and cuDSS
integration.

SparseLinearSolverType
----------------------

Enum in `cunls/linear_solver/sparse_linear_solver.h`:

- `cuDSS`

SparseLinearSolverConfig
------------------------

Struct in `sparse_linear_solver.h`:

- `cudss_solver_options` - [in] cuDSS-specific backend options.

CSRSparseLinearSolver
---------------------

Abstract base (`cunls/linear_solver/csr_sparse_linear_solver.h`).

.. cpp:function:: bool Initialize(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Performs setup work (at minimum symbolic analysis; some modes also perform
  an initial factorization) for the given sparsity pattern. Must be called
  before :cpp:func:`Solve` and re-called whenever the matrix structure changes.

  Both ``rhs`` and ``result`` must be pre-allocated to the same number of
  elements as the number of rows in ``spd_matrix``; the solver does **not**
  resize them.

  :param ``handle``: [in] Opaque backend context (e.g. ``cuDSSHandle_t``).
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool Solve(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  Performs factorization and solve phases to compute the solution ``x`` of
  ``A x = b``. Both ``rhs`` and ``result`` must be pre-allocated to the same
  number of elements as the number of rows in ``spd_matrix``; the solver does
  **not** resize them. Call :cpp:func:`Initialize` first for the current
  sparsity pattern.

  :param ``handle``: [in] Opaque backend context from :cpp:func:`Initialize`.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
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

.. cpp:function:: bool cuDSSLinearSolver::Initialize(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  For ``SlowInitFastSolve``: runs the cuDSS symbolic analysis phase only.
  For ``FastInitSlowSolve``: runs symbolic analysis followed by an initial
  factorization so that subsequent :cpp:func:`cuDSSLinearSolver::Solve` calls
  can use cheaper refactorization.

  Both ``rhs`` and ``result`` must be pre-allocated to the number of rows in
  ``spd_matrix``; the solver does **not** resize them.

  :param ``handle``: [in] ``cuDSSHandle_t`` provided by the caller.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector ``b`` (size must equal matrix rows).
  :param ``result``: [out] Solution vector ``x`` (size must equal matrix rows).
  :returns: ``true`` on success; ``false`` if a dimension mismatch is detected.

.. cpp:function:: bool cuDSSLinearSolver::Solve(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  For ``SlowInitFastSolve``: performs full factorization then solve. For
  ``FastInitSlowSolve``: performs refactorization (reusing the symbolic
  analysis from ``Initialize``) then solve.

  Both ``rhs`` and ``result`` must be pre-allocated to the number of rows in
  ``spd_matrix``; the solver does **not** resize them and returns ``false``
  on any dimension mismatch.

  :param ``handle``: [in] ``cuDSSHandle_t`` from :cpp:func:`cuDSSLinearSolver::Initialize`.
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
