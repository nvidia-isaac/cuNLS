# Module: math

> Notes for agents and contributors working in this directory (loaded automatically by coding agents that read nested `AGENTS.md`). Users of the library: see
> `python/pycunls/AGENT_GUIDE.md` and `docs/sphinx/`.

## Purpose

Low-level math kernels and Lie-group utilities used by factors/states.

## Key files

- `so_se_lie_math.h` / `.cu`: SO(2), SO(3), SE(2), SE(3) operations (exp, log,
  adjoints, Jacobians); `sim_lie_math.h` / `.cu`: Sim(2), Sim(3);
  `sl_lie_math.h` / `.cu`: SL(4); `lie_device.cuh`: device-side helpers shared by
  the kernels.
- `dense_matrix_ops.h` / `dense_matrix_ops.cu`: dense matrix helper kernels.

## Usage notes

- Mostly internal APIs; external applications should prefer high-level factor
  and state classes.
- Functions are designed for batched GPU execution and often assume contiguous
  device memory layouts.

## Dependents

- `state/*_state_batch.*`
- `factor/*_factor_batch.*`
- `minimizer/*` (for some algebra utilities)
