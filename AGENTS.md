# AGENTS.md

cuNLS: GPU nonlinear least-squares solver, CUDA/C++17 with nanobind Python bindings (`pycunls`),
Linux x86_64 / aarch64. CMake 3.22+; GoogleTest and pytest (both need a GPU). Dependencies (cuDSS,
spdlog, CUDA toolkit libraries) come in through CMake `FetchContent`. Python wheels
(`pycunls-cu13`) are CUDA 13 only and link `libcunls.a` statically; the C++ library supports CUDA
12.6+ and 13 (the CUDA 12 CI cells build and test C++ only, which keeps CUDA 13-only APIs out of
library code). Each directory under `cunls/` has an `AGENTS.md`
(classes, data flow, invariants): read it before changing that module.

## Layout

`cunls/` library · `tests/` C++ tests · `python/` bindings (`src/`), package (`pycunls/`), tests,
examples · `examples/` CMake examples · `docs/sphinx/` docs · `scripts/` build/test scripts and the
one `Dockerfile` · `cmake/` modules · `.github/workflows/` CI.

## Build and test (all in Docker; `INSTALL.md` has local builds, Jetson, docs)

```bash
./scripts/build_cunls_in_docker.sh Release          # C++, shared + static
./scripts/test_cunls_in_docker.sh ./build_docker    # C++ tests
./scripts/build_pycunls_in_docker.sh                # wheel
./scripts/test_pycunls_in_docker.sh ./dist          # Python tests
```

CI runs the same four commands, parameterized by `CUDA_VERSION` / `UBUNTU_VERSION`.

## What the library is

- `StateBatch`: manifold variables (vectors, SO/SE(2/3), Sim(2/3), SL(4), articulated robots:
  `ArticulationTopology` + `ArticulationStateBatch`). `FactorBatch`: batched residuals with
  analytic Jacobians (kinematic factors and `LinkFrameFactorBatch` in `cunls/factor/kinematics/`).
  `Problem` composes them; `GaussNewtonMinimizer` / `LevenbergMarquardtMinimizer`, their RANSAC
  versions, and `AugmentedLagrangianMinimizer` (constraints, box bounds; its own LM inner solves) solve; a
  problem runs on the device of its states (`Problem::Device`). Umbrella header:
  `cunls/cunls.h`.
- `python/pycunls`: bindings plus `fitting.py` (`fit`), `builder.py` (`ProblemBuilder`, `solve`),
  `diagnostics.py`, `catalog.py`, `articulation.py`, `mpc.py`, `warp.py` (`WarpFactorBatch`,
  `WarpStateBatch`, `WarpFunctionFactorBatch`). Every `stream=` also takes a `wp.Stream` or any
  `__cuda_stream__` object. `AGENT_GUIDE.md` ships in the wheel (`python -m pycunls guide`); its
  section 4 lists the conventions users must get right.

## How to write library code

1. Every public item (class, method, member, hook, parameter in an installed header) has a doc
   comment saying what it is for, when it runs, what it reads and writes. Every Python binding has
   a docstring (`python/tests/test_api_docs.py`); regenerate the stub with
   `python scripts/generate_pycunls_stubs.py`.
2. Errors: `<Class>::<Method>: <what is wrong> (got X, expected Y). Fix: <action>.` Python
   boundary checks live in `python/src/bind_types.cpp`.
3. A solver outcome is a `TerminationReason` plus a `message` with the fix, never a bare bool.
4. No cost in the hot path: diagnostics run on failure, at the end, or on request; no host
   synchronization per iteration; real-time paths read nothing back.
5. Analytic Jacobians for built-in factors; concrete classes; no templates or CUDA in public
   headers; coalesced kernels. Numeric Jacobians are judged by accuracy, not speed.
6. Lean code: no per-class virtual overrides just to enable a check (keep such metadata in one
   table, as `python/pycunls/catalog.py`); no tiny shared `.cuh` headers; a base class holds only
   what every subclass shares.
7. No automatic backend switching: the linear solver is an explicit choice.
8. Docs stay consistent: `python/tests/test_docs_consistency.py` checks paths and `pycunls.X`
   names in docs and `AGENTS.md`; the Sphinx build has no warnings; `llms.txt` is generated.
9. Agent-facing doc or API changes must not lower the pass rate of `scripts/run_agent_eval.sh`
   (tasks in `python/agent_tasks`).

## CI and build principles

- Local-CI parity: CI calls the scripts developers run; workflow YAML has no cmake, make, ctest,
  pip or pytest. Scripts own the "how", workflows the "when / where / what".
- One job per script: `*_in_docker.sh` manages Docker, the inner script builds or tests; build
  scripts never test and test scripts never build.
- One parameterized `scripts/Dockerfile` (`CUDA_VERSION`, `UBUNTU_VERSION`); add dependencies
  there. Distro-default GCC and CMake; pin a version only for a documented incompatibility.
- Scripts work in a terminal and in CI unchanged (`[ -t 0 ] && TTY_FLAG="-it"`, env-var
  parameters with defaults) and fail fast with the command that fixes the problem; GPU jobs
  start with `nvidia-smi`.
- No reusable workflows or composite actions with fewer than two callers. x86 GPU jobs run on
  `[self-hosted, gpu]`, Jetson on `jetson-orin` / `jetson-thor`; CUDA / OS versions are Docker
  build args, not runner labels.
- Artifacts move through the filesystem within a job; upload / download only across jobs.
  Nightlies publish a dated release (`nightly-YYYY-MM-DD`) with, per matrix entry, one C++
  tarball (shared + static), a wheel and JUnit XMLs.
- Containers mount the output directory at `/cunls_install` (CTest files embed absolute paths)
  and the source read-only at `/cunls`. Shared and static builds install to `$OUTPUT/shared` and
  `$OUTPUT/static` (one install prefix would overwrite the CMake targets file).

## Pitfalls seen before

- `docker login -p $SECRET` exposes the secret: pipe it to `--password-stdin` from `env:`.
- `grep ... | head -1 || echo "?"` never falls back (`||` binds to `head`): use `${VAR:-"?"}`.
- `wp.Stream(device, cuda_stream=h)` registers `h` in Warp's stream table (one entry per handle,
  no reference count) and its `__del__` erases the entry: never wrap a handle another
  `wp.Stream` owns (cuNLS passes Warp's own stream to callbacks). Use
  `pycunls._arrays.borrow_stream`, which leaves the table alone.
- `.dockerignore` must sit at the build-context root. Use `@v5` actions with
  `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true`. Build CI images from the Dockerfile, not from a
  private registry.
