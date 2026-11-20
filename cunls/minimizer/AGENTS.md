# Module: minimizer

Problem assembly, normal equations, and the GN / LM / augmented-Lagrangian / RANSAC loops. User
guides: `docs/sphinx/constraints.rst`, `docs/sphinx/ransac.rst`.

## Files

- `problem.h`: `Problem` (non-owning graph of states and factors; `FindStructureError` returns the
  first structural problem with its fix). `residual_batch.h`: `ResidualBatch` (factor + loss),
  `EvaluateFactorCosts`.
- `minimizer.h`: `TerminationReason` (ordered by severity), `MinimizerOptions`, `MinimizerSummary`,
  `Minimizer` (the one iteration loop: linear system, line search, structure reuse, per-subproblem
  bookkeeping; the private hooks `UpdateSystem` / `ClassifySteps` are the whole GN / LM
  difference; behaviour lives in free functions in `minimizer.cu`). `internal::InnerSolve` /
  `InnerMinimize` are the AL's channel into the loop; `internal::SetUpStructure` /
  `BuildSystem` also serve tests.
- `gauss_newton_minimizer.h`, `levenberg_marquardt_minimizer.h` (per-subproblem λ, fused
  λ-update + damping kernel; all LM state lives there).
- `problem_partition.h`: `ProblemPartition` (subproblem maps and sums, the device step decision
  `StepControl`, deterministic reductions, one-subproblem fast paths).
- `augmented_lagrangian_minimizer.h`: AL outer loop around a GN / LM `Minimizer &`; finds
  constraint batches by type; per-subproblem penalties and stopping on the device. The only solver
  of box-bounded states: clamps initial free states into the box, and `HoldActiveBounds` (as
  `InnerSolve::restrict_system`) holds active bounds plus up to `max_bound_refinements`
  active-set refinements. GN / LM / RANSAC reject bounded problems.
- `normal_equations.h`: `NormalEquations` owns H and its layout (BSR when tangent sizes share a
  factor and the solver supports it, else CSR; chosen here only) and writes the working LHS
  (scale, damping, mask) into the solver's `MatrixStorage()` in one kernel.
- `hessian_structure.h` (GPU sparsity), `block_hessian_assembler.h` (one kernel for H and rhs),
  `bsr_matrix.h`, `sparse_matrix.h` (CSR diagonal positions / extraction, SpMV, element-wise
  ops), `minimizer_state.h` (snapshots).
- `ransac_minimizer.h`: RANSAC over an ordinary `Problem` (free tangent <= 64);
  internals in `ransac/` (`RansacLayout`, `SlotSet`, `HypothesisScorer`, `RunRansac`, kernels).

## Iteration

1. Evaluate residuals and per-factor dense Jacobian blocks; contract them locally into
   `H = JᵀJ`, `-Jᵀr` (the global Jacobian is never formed); solve; apply the manifold update.
2. Classify and take / reject each subproblem's step on the device (`StepControl`); one 2-float
   read-back per iteration. `ClassifySteps` writes a `TerminationReason` per subproblem;
   `FinishTermination` counts them in the final synchronization (none in real time).
3. With `max_line_search_steps > 0` a non-decreasing step is halved along its direction.
4. Stopped subproblems take no step and no factorization work: the loop declares the active rows
   (`ActiveRows` -> `SparseLinearSolver::SetRowActivity`) and BlockSparseCholesky skips the rest.

## Invariants

- A subproblem never waits for another; without a partition the problem is one subproblem with
  single-problem results and iteration counts. The AL gives each feasible subproblem its own full
  inner solve while infeasible ones continue capped solves.
- Problem pointers are non-owning; all buffers are on the device.
- RANSAC uses only `FactorBatch::Evaluate` and `StateBatch::Plus` through their item
  (`factor_ids`) and replica (`num_replicas`) parameters, which every batch must honor. Each
  hypothesis owns a replica of every free state batch; samples are keyed Feistel permutations;
  per-hypothesis normal equations, factorization and scoring use fixed reduction orders (same
  seed, same bits). By default an inlier needs a non-zero Jacobian on a free state.
- Real time: `reuse_structure` skips structure setup when the caller vouches for it (a host size
  signature is checked); `MinimizerOptions::real_time` (`InnerSolve::fixed_iterations`, also the
  AL's `base_options.real_time`) fixes the line-search and refinement counts and reads nothing back
  until the end (summary costs NaN; AL without constraint batches reads nothing back at all).
- `AugmentedLagrangianMinimizer` owns a private `LevenbergMarquardtMinimizer` built from its
  options (`AugmentedLagrangianMinimizerOptions` extends `LevenbergMarquardtMinimizerOptions`);
  `base_options.max_num_iterations` / `max_line_search_steps` / `real_time` / `reuse_structure` are
  the budget of the whole call (one rule with or without constraint batches; a problem without them
  starts every subproblem as feasible, so its first inner solve is the full solve). `SetOptions`
  changes the outer loop and the budget; the inner solver keeps its construction settings.
