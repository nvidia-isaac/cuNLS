# cuNLS Core Source Tree

> Notes for agents and contributors working in this directory (loaded automatically by coding agents that read nested `AGENTS.md`). Users of the library: see
> `python/pycunls/AGENT_GUIDE.md` and `docs/sphinx/`.

## Purpose

This directory contains all core library modules and the public umbrella header
`cunls.h`.

## Public include

- `cunls.h`: includes the primary public classes used by applications
  (factors, states, minimizers, robustifiers).

## Submodules

- `common/AGENTS.md`: CUDA handles, device vectors, sparse/dense type aliases.
- `factor/AGENTS.md`: factor interfaces and built-in factor implementations.
- `linear_solver/AGENTS.md`: sparse solver interface, BlockSparseCholesky
  direct solver (default), BlockSparsePCG iterative solver, BlockTridiagonal
  (trajectories / MPC), cuDSS direct solver, and dense fallbacks.
- `math/AGENTS.md`: Lie and matrix math kernels/utilities.
- `minimizer/AGENTS.md`: `Problem` (with subproblem partitions), residual wrappers, GN/LM
  solvers, the assembled and Schur-complement linear systems, RANSAC minimizers and
  `AugmentedLagrangianMinimizer` (constraints, bounds).
- `robustifier/AGENTS.md`: robust loss batches.
- `state/AGENTS.md`: manifold state batch implementations and update orchestration.

## Typical dependency direction

`factor/state` -> `minimizer` -> `linear_solver`/`math` + `common`. One exception:
`SparseLinearSolver::Initialize` takes the `Problem` (to derive the block layout or
the stages), so `linear_solver` includes `minimizer/problem.h`.
