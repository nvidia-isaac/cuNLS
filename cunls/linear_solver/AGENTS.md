# Module: linear_solver

Sparse linear solvers for the normal equations (SPD systems as CSR, or BSR where
`SupportsBlockStorage()`). Design guide: `docs/sphinx/linear_solvers.rst`.

## Contract (`sparse_linear_solver_base.h`)

`SparseLinearSolver`: `Initialize` analyzes the pattern; `MatrixStorage()` is the solver's own
matrix buffer plus the map from each pattern value to its slot (null = identity);
`NormalEquations` writes the working matrix straight into it (no copy on the solve path);
`ClearMatrixStorage` zeroes it; `SolveStoredMatrix` factors and solves. `Solve(matrix)` is the
convenience form (clear, copy through the map, solve), used by `SchurSystem`; PCG and cuDSS read
the matrix in place. `SetRowActivity` is advisory. Factory: `sparse_linear_solver.h`
(`SparseLinearSolverType`, `CreateSparseLinearSolver`).

## Backends

- Dense (`dense_linear_solver_base.h` owns the dense matrix and the entry map; subclasses supply
  `FactorizeAndSolve`): `DenseLDLTSolver` (pivoted, indefinite OK), `DenseCholeskySolver`,
  `DenseQRSolver` (cuSOLVER).
- `cuDSSLinearSolver` (CSR only; `SlowInitFastSolve` / `FastInitSlowSolve`).
- `BlockSparsePCGSolver`: block-Jacobi LDLT preconditioner, `H p` by
  `MultiplyBSRValuesByDenseVector` (`minimizer/bsr_matrix.h`), cuSPARSE
  (`CUSPARSE_SPMV_CSR_ALG2` in `deterministic_mode`) or `BaHessianProduct`
  (`ba_hessian_vector_product.h`: bundle adjustment, fixed-point sums, bit-identical); block
  layout as `(count, size)` segments; convergence polled every K iterations.
- `BlockTridiagonalSolver`: Riccati recursion for systems block tridiagonal in time (needs
  `Problem::SetStateStages`; stage blocks <= 32); one warp per subproblem, cp.async prefetch;
  graph-capturable unless safety checks read back the singular-pivot count.
- `BlockSparseCholeskySolver` (`block_sparse_cholesky_{solver,analysis}.h`): the default and the
  Schur complement's solver. `Initialize`: supervariables and their graph on the GPU,
  minimum-degree or nested-dissection ordering by a cost model (`AnalyzeCholesky`, host),
  supernodes amalgamated and cut into panels of <= 64 columns, level schedule, assembly and
  update maps. `SolveStoredMatrix`: per level a factor kernel (one CTA per panel and 64 rows,
  forward substitution folded in) and an update kernel (64 x 64 tiles), then backward
  substitution by level.

## BlockSparseCholesky warp path

A level with >= 256 panels of <= 32 rows (`kCholeskyMinSmallPanelsPerLevel`,
`kCholeskySmallPanelRows`) gives each one warp, a row per lane in registers
(`FactorKernelByWarp`, `UpdateKernelByWarp`, `BackwardKernelByWarp`); fewer stay on the CTA
kernels (less latency per panel). A small root (no rows below, e.g. a whole IK problem) is also
solved backward by its factor warp (no L_D store, no backward launch). The warp kernels check
their own row activity; first-level small panels read A's diagonal from registers. Measured
lessons, keep them:

- Keep the column loop rolled (rows shift down a register per column): a fully unrolled 32 x 32
  nest is instruction-fetch bound with few warps (3x slower).
- Predicate the inner loops, never early-exit: a branch per column serializes the shuffles (2x).
- One launch per level, sized by the level's widest class (8 / 16 / 32 columns,
  `small_width_class`): one size for all took 96 registers for 7-wide panels; one launch per class
  cost large graphs 2-4%.
- Small panels store L_D column-major (row-major stores cost 2x once the panels exceed L2).
- The diagonal-block lanes own the forward-substitution entries (a per-lane copy cost 5-25%).
- 33-64-row panels on one warp lost to the CTA kernels (a 64-row panel is what a CTA is sized for).

`SetRowActivity` skips panels whose rows are all idle (no factorization, no updates, zero
solution); exact because a panel never mixes connected components (supernodes merge only along the
elimination tree): keep it that way when changing the analysis.
