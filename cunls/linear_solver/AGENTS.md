# Module: linear_solver

> Notes for agents and contributors working in this directory (loaded automatically by coding agents that read nested `AGENTS.md`). Users of the library: see
> `python/pycunls/AGENT_GUIDE.md` and `docs/sphinx/`.

## Purpose

Sparse linear system abstraction and implementation used by minimizers.

## Key types

- `sparse_linear_solver_base.h`: abstract `SparseLinearSolver`, one
  contract for every backend (so they are interchangeable): `Initialize`
  analyzes the pattern (CSR pure virtual; BSR defaulted and guarded by
  `SupportsBlockStorage()`); `MatrixStorage()` returns the solver's own
  matrix buffer plus the map from each pattern value to its slot
  (`SolverMatrixStorage`, null map = identity); `ClearMatrixStorage` zeroes
  it; `SolveStoredMatrix` factors what was written and solves.
  `NormalEquations` writes the working matrix straight into the storage (one
  kernel), so no matrix is copied on the solve path. `Solve(matrix)` is the
  default convenience form (clear, copy through the map, solve;
  `sparse_linear_solver_base.cu`), used by `SchurSystem`; PCG and cuDSS
  override it to read the matrix in place.
- `sparse_linear_solver.h`:
  - `SparseLinearSolverType`
  - `SparseLinearSolverConfig`
  - `CreateSparseLinearSolver(...)`
- `dense_linear_solver_base.h`:
  - `DenseLinearSolverBase` — shared machinery for the backends that densify
    before factorizing.  Owns `dense_matrix_` (its `MatrixStorage()`),
    dimension validation and the entry -> dense slot map built in
    `Initialize` (64-bit when n² > 2^31); subclasses supply only
    `EnsureBuffersSize` and `FactorizeAndSolve`.  `SupportsBlockStorage()` is
    `true`: the sparse layout does not survive the scatter, so neither layout
    can be better for them.
- `dense_linear_solver.h`: `DenseLDLTSolver` — custom pivoted-LDLT kernels;
  handles indefinite symmetric matrices.
- `dense_cholesky_solver.h`: `DenseCholeskySolver` — cuSOLVER potrf/potrs.
- `dense_qr_solver.h`: `DenseQRSolver` — cuSOLVER geqrf/ormqr + cuBLAS trsm.
- `cudss_sparse_linear_solver.h`:
  - `cuDSSLinearSolverOptions`
  - `cuDSSLinearSolver` — declines block storage (CSR only).
- `block_sparse_pcg_solver.h`:
  - `BlockSparsePCGOptions` — `block_size` / `block_layout`,
    `max_iterations`, `relative_tolerance`, `absolute_tolerance`,
    `pivot_floor`, `check_period`, `deterministic_mode` (fixed-order dot
    products + `CUSPARSE_SPMV_CSR_ALG2`, so repeated solves return the same
    bits; off by default), `hessian_product` (`kBundleAdjustment`:
    deterministic bundle-adjustment `H p`, see `ba_hessian_vector_product.h`).
  - `BlockSparsePCGSolver` — iterative SPD solver. `H p` by one of three
    paths: in-house BSR kernels for block storage, cuSPARSE SpMV for CSR
    (`CUSPARSE_SPMV_CSR_ALG2` in `deterministic_mode`), or `BaHessianProduct`
    for `kBundleAdjustment`; block-Jacobi LDLT preconditioner (dense B x B diagonal tiles factored
    in shared memory).  Supports a uniform block size or a layout vector
    of `(count, block_size)` segments matching the host state-batch
    structure (e.g. `{{N_poses, 6}, {N_points, 3}}` for SBA).  Inner loop
    computes alpha/beta on device; convergence is polled every K
    iterations to keep CPU launches ahead of the GPU.
- `ba_hessian_vector_product.h`: `BaHessianProduct` — the
  `kBundleAdjustment` `H p`: one cooperative launch that reads each
  camera-point block once from the point rows and sums the pose rows in
  fixed point (bit-identical run to run).  `Build` validates the pattern,
  sizes and GPU and records `FailureReason()`; per-call counters live in
  the instance's accumulator buffer, so instances are independent.

- `block_tridiagonal_solver.h`: `BlockTridiagonalSolver` — direct solver for
  systems that are block tridiagonal in time (trajectories, MPC). Needs
  `Problem::SetStateStages` (stage = time index per state) and optionally the
  subproblem partition. `Initialize` maps every CSR entry to a stage block on
  the host (once per structure; rejects couplings of non-adjacent stages);
  the map is its `MatrixStorage()` into the D_k / C_k blocks;
  `SolveStoredMatrix` runs the block Cholesky (Riccati)
  recursion, one warp per subproblem, registers + shuffles, cp.async
  prefetch of the next stage, kernel templated on the stage block size
  M <= 32 (dispatch table). No host synchronization (graph-capturable) unless
  safety checks are on: then the solve reads back the singular-pivot count.
- `block_sparse_cholesky_solver.h` / `block_sparse_cholesky_analysis.h`:
  `BlockSparseCholeskySolver` — supernodal sparse Cholesky, no external
  dependency; the default backend (`MinimizerOptions::sparse_linear_solver_type`)
  and the reduced-system solver of `LinearSystemType::SchurComplement`. CSR
  only (declines block storage). `Initialize` finds supervariables and their
  graph on the GPU, picks approximate minimum degree (quotient graph) or
  nested dissection (balanced BFS level-set separators) by a cost model
  (`AnalyzeCholesky`, host),
  amalgamates supernodes and cuts them into panels of <= 64 columns, and
  builds the assembly map (its `MatrixStorage()` map into the permuted panels)
  and the relative map of the updates on the GPU.
  `SolveStoredMatrix`: per level a factor kernel (one
  CTA per panel and 64 rows; forward substitution folded in) and an update
  kernel (64 x 64 tiles), then a level-scheduled backward substitution.
  Small panels (`kCholeskySmallPanelRows` = 32 rows: diagonal block plus
  rows below) in a level with at least `kCholeskyMinSmallPanelsPerLevel`
  (256) of them take warp kernels instead (`FactorKernelByWarp`,
  `UpdateKernelByWarp`, `BackwardKernelByWarp`: a row per lane in registers,
  shuffles; `CholeskyPlan::small_panels` / `small_runs` by level instead of
  factor / update items): batches of small subproblems. Fewer stay on the
  CTA kernels (less latency per panel, no extra launches). Measured lessons:
  keep the warp factorization's column loop rolled (rows shift down a
  register per column; widths 8 / 16 / 32 compiled apart, also for the update
  and backward kernels): a fully unrolled 32 x 32 loop nest is bound by
  instruction fetch when few warps run it (3x slower than the CTA kernel on
  16 panels). Keep the inner loops predicated, not early-exited: a branch per
  column serializes the shuffles (2x slower). The lanes of the diagonal-block
  rows own the forward-substitution entries (a copy of y_p per lane cost
  5-25%). Small panels store L_D column-major (each column one coalesced
  store; row-major scattered every store over 32 lines and, with the panels
  larger than L2, cost 2x on an A6000); only the warp kernels read it. Panels
  of 33-64 rows on one warp (two rows per lane) were tried and lost to the
  CTA kernels except for narrow ones (<= 16 columns): a 64-row panel is what
  the CTA kernel is sized for.
  `SetRowActivity` (advisory, from the minimizers of partitioned problems)
  skips panels whose rows are all idle: no factorization, no updates from
  them, zero solution. Exact because a panel never mixes connected components
  (supervariables need identical patterns; supernodes merge only along the
  elimination tree): keep it that way when changing the analysis.

## Expected system form

- Solves SPD systems given as `CSRSparseMatrix` or, for backends that report
  `SupportsBlockStorage()`, `BSRSparseMatrix` — typically normal equations
  assembled by `GaussNewtonMinimizer` / `LevenbergMarquardtMinimizer`.  The
  layout is chosen in `NormalEquations`; no conversion runs on the solve path,
  and the matrix is written into the solver's `MatrixStorage()`, not passed.

## cuDSS performance modes (`cuDSSLinearSolverOptions::mode`)

- `SlowInitFastSolve`: expensive setup, fast repeated solves.
- `FastInitSlowSolve`: faster setup, slower repeated solves.
