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

.. cpp:function:: void Initialize(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  :param ``handle``: [in] Opaque backend context.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector.
  :param ``result``: [out] Output/working solution vector.
  :returns: [out] No return value.

.. cpp:function:: void Solve(void* handle, const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs, dvector<float>& result)

  :param ``handle``: [in] Opaque backend context.
  :param ``spd_matrix``: [in] SPD matrix in CSR format.
  :param ``rhs``: [in] Right-hand-side vector.
  :param ``result``: [out] Solution vector.
  :returns: [out] No return value.

cuDSSLinearSolverMode
---------------------

Enum in `cunls/linear_solver/cudss_sparse_linear_solver.h`:

- `SlowInitFastSolve`
- `FastInitSlowSolve`

cuDSSLinearSolverOptions
------------------------

Struct fields:

- `mode` - [in] Trade-off between setup and repeated solve speed.
- `nthreads` - [in] Host thread count for cuDSS host-side stages.
- `threading_lib_path` - [in] Optional threading runtime path for multi-threaded
  host processing.

cuDSSLinearSolver
-----------------

Concrete implementation in `cudss_sparse_linear_solver.h`.

.. cpp:function:: cuDSSLinearSolver(cuDSSLinearSolverOptions options = cuDSSLinearSolverOptions())

  :param ``options``: [in] cuDSS mode/threading configuration.
  :returns: [out] Constructor has no return value.

Factory
-------

Function in `sparse_linear_solver.h`:

.. cpp:function:: SparseLinearSolverPtr CreateCSRSparseLinearSolver(SparseLinearSolverType type, const SparseLinearSolverConfig& config)

  :param ``type``: [in] Backend type to instantiate.
  :param ``config``: [in] Backend-specific configuration blob.
  :returns: [out] Heap-allocated solver instance.
