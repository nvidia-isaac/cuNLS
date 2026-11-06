# AGENTS.md

## Repository Overview

cuNLS is a GPU-accelerated nonlinear least-squares solver library written in CUDA/C++. It targets NVIDIA GPUs and runs on Linux (x86_64 and aarch64).

### Key facts

- **Language**: CUDA/C++ (C++17), with Python bindings via nanobind
- **Build system**: CMake 3.22+, `nvcc`, Unix Makefiles
- **Testing**: GoogleTest (C++), pytest (Python). Tests require a GPU.
- **Dependencies**: cuDSS, spdlog, CUDA Toolkit (cusparse, cublas, cusolver). All fetched via CMake `FetchContent`.
- **Docker**: All builds run inside Docker containers via `scripts/Dockerfile`. Parameterized by `CUDA_VERSION` and `UBUNTU_VERSION` build args.
- **CI**: GitHub Actions on nvidia-isaac org shared self-hosted GPU runners (AWS, auto-scaling, `[self-hosted, gpu]`), plus the org's Jetson runners (`jetson-orin`, `jetson-thor`). Nightly matrix: x86_64 CUDA 12.6/12.8/13.2 x Ubuntu 22.04/24.04, Orin (JetPack 6.1, CUDA 12.6), Thor (JetPack 7.1, CUDA 13.0). PR verification runs the same x86_64 entries plus Thor.
- **Python package**: `pycunls` -- statically links `libcunls.a`, dynamically links CUDA runtime libs. Wheel is CUDA-version-specific.

### Repository structure

```
cunls/              # Core library source (C++/CUDA)
tests/              # GoogleTest C++ tests
python/             # Python bindings (nanobind) + pytest tests
examples/           # Standalone CMake example projects
docs/sphinx/        # Sphinx documentation source
scripts/            # Build/test shell scripts + Dockerfile
  Dockerfile        # Single parameterized Docker image for all workflows
  build_cunls.sh    # CMake configure + build (runs inside container)
  build_cunls_in_docker.sh   # Docker lifecycle for C++ build
  build_pycunls.sh           # pip wheel build (runs inside container)
  build_pycunls_in_docker.sh # Docker lifecycle for Python build
  test_cunls_in_docker.sh    # Docker lifecycle for C++ tests
  test_pycunls_in_docker.sh  # Docker lifecycle for Python tests
  generate_pycunls_stubs.py  # regenerate python/pycunls/_pycunls_core.pyi
  run_agent_eval.sh          # run an agent on python/agent_tasks and score it
cmake/              # CMake modules (AddCUDSS, AddSpdlog, BundleStaticDeps)
.github/workflows/  # CI workflows (pr-verify, build-and-test, lint, nightly, release, sync-rulesets)
```

Each directory under `cunls/` has an `AGENTS.md` with its architecture notes (key classes, data
flow, invariants); read it before changing that module.

### Build and test locally

Full instructions (local builds, Jetson, cuDSS, docs): `INSTALL.md`.

```bash
# C++ build (shared + static, installed to separate dirs)
./scripts/build_cunls_in_docker.sh Release

# C++ tests (requires build output from above)
./scripts/test_cunls_in_docker.sh ./build_docker

# Python wheel
./scripts/build_pycunls_in_docker.sh

# Python tests (requires wheel from above)
./scripts/test_pycunls_in_docker.sh ./dist
```

CI runs the exact same 4 commands with `CUDA_VERSION` and `UBUNTU_VERSION` env vars for matrix parameterization.

---

## Library Architecture and Conventions

### What the library is

- `StateBatch` implementations hold manifold-aware variables (vectors, SO/SE(2/3), Sim(2/3),
  SL(4)); `FactorBatch` implementations evaluate batched residuals and analytic Jacobians;
  `Problem` composes them; `GaussNewtonMinimizer` / `LevenbergMarquardtMinimizer` solve,
  `RansacGaussNewtonMinimizer` / `RansacLevenbergMarquardtMinimizer` add robust estimation,
  `AugmentedLagrangianMinimizer` adds constraints and box bounds. Public umbrella include:
  `cunls/cunls.h`.
- Python (`python/pycunls`): the nanobind bindings plus pure-Python layers -- `fitting.py`
  (`fit`), `builder.py` (`ProblemBuilder`, `solve`), `diagnostics.py` (`diagnose`,
  `evaluate`, `check_jacobians`), `catalog.py` (`catalog`, `describe`, class docstrings),
  `mpc.py`, `warp.py`. `AGENT_GUIDE.md` ships in the wheel (`python -m pycunls guide`).
- Conventions users must get right are listed in `python/pycunls/AGENT_GUIDE.md` (section 4):
  capacity vs. active count, `world_from_body` row-major poses, between factors measure
  `T_right^-1 T_left`, gauge freedom, absolute tolerances.

### How to write library code

1. **Every public item explains itself.** Each class, method, member, hook and parameter in an
   installed header gets a clear doc comment in the same commit: what it is for, when it runs,
   what it reads and writes. Every Python binding gets a docstring (enforced by
   `python/tests/test_api_docs.py`); regenerate the stub with
   `python scripts/generate_pycunls_stubs.py` after changing bindings. Names state exactly what
   a function does.
2. **Errors say what, got, expected, and the fix**:
   `<Class>::<Method>: <what is wrong> (got X, expected Y). Fix: <action>.` Python boundary
   checks (dtype, host vs. device, size) live in the bindings (`python/src/bind_types.cpp`
   `inspect_device_array`, `device_items`).
3. **Results explain themselves.** A solver outcome is a `TerminationReason` plus a `message`
   with the fix, not a bool; new failure modes get a reason or a message, never silence.
4. **No cost in the hot path.** Diagnostics run on failure paths, at the end of a solve, or on
   request; never add a host synchronization to the per-iteration loop. Real-time
   (`fixed_iterations`) paths read nothing back.
5. **Analytic Jacobians** for built-in factors (no autodiff); concrete classes like the between
   factors; no templates or CUDA in public headers; coalesced kernels. Numeric (finite
   difference) Jacobians are judged by accuracy, not speed.
6. **Lean code.** No per-class virtual overrides just to enable a check (keep such metadata in
   one table, as `pycunls/catalog.py`); no tiny shared `.cuh` headers for a few device helpers
   (inline them into the `.cu`); a base class holds only what every subclass shares (no pimpl,
   no optional feature slots).
7. **No automatic backend switching.** The linear solver is an explicit choice; a message may
   suggest another one.
8. **Docs stay consistent.** `python/tests/test_docs_consistency.py` checks that file paths in
   the docs and `AGENTS.md` files exist and that every `pycunls.X` the docs name exists; the
   Sphinx build must stay free of warnings. `llms.txt` is generated into the docs site by
   `docs/sphinx/conf.py`, not written by hand.
9. **Measure agent-facing changes.** `python/agent_tasks` holds tasks with hidden truths;
   `scripts/run_agent_eval.sh` runs an agent on them. A doc or API change aimed at agents should
   not lower the pass rate.

---

## Engineering Principles

Follow these principles when modifying this repository. They reflect decisions made during CI/CD implementation and are informed by the nvidia-isaac infrastructure constraints.

### 1. Local-CI parity

CI must call the same scripts developers run locally. Never inline build or test commands in workflow YAML. If a developer can't reproduce a CI failure by running the same script on their machine, the CI design is wrong.

### 2. Scripts own the "how", pipelines own the "when/where/what"

Build logic, test logic, and Docker lifecycle live in shell scripts under `scripts/`. Workflow YAML handles triggers, matrix definitions, runner selection, artifact upload, and releases. The YAML should contain zero cmake, make, ctest, pip, or pytest invocations.

### 3. Single Responsibility for scripts

Each script does one thing:
- `build_cunls_in_docker.sh` -- Docker lifecycle for C++ build
- `build_cunls.sh` -- cmake configure + make + install (runs inside container)
- `test_cunls_in_docker.sh` -- Docker lifecycle for C++ tests
- `build_pycunls_in_docker.sh` -- Docker lifecycle for Python wheel build
- `build_pycunls.sh` -- pip wheel (runs inside container)
- `test_pycunls_in_docker.sh` -- Docker lifecycle for Python tests

`_in_docker.sh` wrappers manage Docker (build image, mount volumes, run container). Inner scripts run the actual build tools. Build scripts don't test. Test scripts don't build. Dependencies are enforced by ordering, not by scripts calling each other.

### 4. One Dockerfile, parameterized

All Docker-based workflows use `scripts/Dockerfile` with `ARG CUDA_VERSION` and `ARG UBUNTU_VERSION` that default to the team's standard values (currently CUDA 12.8.1, Ubuntu 24.04). Do not create additional Dockerfiles. When adding a dependency, add it to the existing Dockerfile.

### 5. Rely on platform defaults

Use the distro-default GCC and CMake (not pinned versions unless there's a proven compatibility issue). Ubuntu 22.04 ships GCC 11, Ubuntu 24.04 ships GCC 13 -- both support C++17. Only pin versions when there is a documented, tested incompatibility.

### 6. CI adapts to infrastructure, not the other way around

- Use `[ -t 0 ] && TTY_FLAG="-it"` for conditional TTY flags -- CI has no terminal, local does.
- Use env vars (`CUDA_VERSION`, `UBUNTU_VERSION`, `EXTRA_CMAKE_ARGS`) for CI parameterization -- defaults make local usage identical to before.
- Scripts must work in both interactive terminals and non-interactive CI without modification by the developer.

### 7. Fail fast with clear messages

- Test scripts check that build output exists before running and print actionable instructions if it doesn't.
- GPU verification (`nvidia-smi`) runs as the first step of every GPU job.
- Precondition failures must tell the user exactly what to run to fix the problem.

### 8. Question every hardcoded value

Before adding a version pin, PPA, or platform-specific workaround, ask: does the project actually require this specific version, or does the platform default work? Most hardcoded values exist by accident, not by necessity. The burden of proof is on the person adding the pin.

### 9. Minimize file count and indirection

Do not create reusable workflows (`workflow_call`) or composite actions until there are at least two consumers. A single-caller reusable workflow adds input threading, secret forwarding, and debugging indirection for zero reuse benefit. Inline is better than indirection when there's only one caller.

### 10. Runner tags reflect infrastructure, not build parameters

x86_64 GPU jobs use `runs-on: [self-hosted, gpu]`. CUDA version, Ubuntu version, and CUDA architectures are controlled by Docker build args, not runner labels. The nvidia-isaac org has a shared pool of 10 auto-scaling GPU runners -- do not encode CUDA or OS versions in runner tags. Jetson jobs select hardware with `jetson-orin` / `jetson-thor`, because the device is infrastructure; their CUDA version is fixed by the host JetPack driver, and Orin overrides the Docker `BASE_IMAGE` with an L4T-compatible NGC CUDA image.

### 11. Artifacts flow through the filesystem, not between jobs

Within a single job, steps share a host directory (e.g., `./output/`). Use GitHub artifact upload/download only for cross-job transfer (e.g., from build-matrix to nightly-status). Do not upload and re-download artifacts between steps in the same job.

### 12. Every nightly produces a complete, dated release

A nightly release tagged `nightly-YYYY-MM-DD` (with the workflow run number appended when that day's tag already points to a different commit) includes, per matrix entry: a single C++ tarball containing both shared (`.so`) and static (`.a`) libraries with their respective CMake configs, headers, a Python wheel, and JUnit test result XMLs. All attached as GitHub Release assets with filenames that include the CUDA version and Ubuntu version.

---

## Container Mount Convention

All `_in_docker.sh` scripts mount the host output directory at `/cunls_install` inside the container. This path must be consistent across all scripts because:
- CTest discovery generates include files with absolute container paths during the build
- Test scripts run in a new `docker run` and must mount at the same path for CTest to find its generated files

The source tree is always mounted read-only at `/cunls`.

Inner scripts that need a configurable output path (e.g., `build_pycunls.sh`) use an env var with a default: `OUTPUT=${OUTPUT_DIR:-/output}`.

---

## Build Artifacts Layout

`build_cunls_in_docker.sh` installs shared and static variants to separate directories to avoid CMake package config overwrites. CI packages both into a single tarball per matrix entry so users get one download per CUDA/Ubuntu combination:

```
$OUTPUT_DIR/                    # host build output
  shared/                       # make install from shared build
    lib/libcunls.so
    lib/cmake/cunls/            # CMake config for shared
    include/cunls/
  static/                       # make install from static build
    lib/libcunls.a
    lib/cmake/cunls/            # CMake config for static
    include/cunls/
  build_shared/                 # cmake build directory (persists for ctest)
  build_static/                 # cmake build directory
  cpp-test-results.xml

cunls-x86_64-cuda13.2.0-ubuntu24.04.tar.gz   # release asset (also orin-*, thor-*)
  shared/                       # set CMAKE_PREFIX_PATH here for .so
  static/                       # set CMAKE_PREFIX_PATH here for .a
```

Headers are identical between shared and static. CMake configs differ (each references its own library file and link interface).

---

## Common Pitfalls

These are mistakes encountered during implementation. Avoid repeating them.

| Pitfall | What happens | How to avoid |
|---|---|---|
| Hardcoding GCC version (e.g., `g++-13`) | Fails on Ubuntu 22.04 which doesn't ship GCC 13 | Use distro default `g++` -- C++17 works on GCC 11+ |
| Unnecessarily pinning CMake | Hides compatibility regressions and adds an external APT repository | Use the distro-provided CMake 3.22+ |
| `docker login -p $SECRET` | Exposes secret in process list | Use `env:` block + `echo "$VAR" \| docker login --password-stdin` |
| `grep ... \| head -1 \|\| echo "?"` | `\|\|` applies to `head` (exit 0), not `grep` | Use `${VAR:-"?"}` after the pipeline |
| Different container mount paths between build and test | CTest include files reference absolute paths from build container | Mount at `/cunls_install` in all scripts |
| Same `CMAKE_INSTALL_PREFIX` for shared and static | CMake targets file gets overwritten by the second build | Install to `$DIR/shared` and `$DIR/static` |
| `.dockerignore` in wrong directory | Docker reads it from build context root, not from Dockerfile directory | Place `.dockerignore` at repo root |
| `actions/checkout@v4` and other v4 actions | Node.js 20 deprecation warnings | Use `@v5` and set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` |
| Creating reusable workflows for single consumers | Adds `workflow_call` input threading, secret forwarding, dual `workflow_dispatch` boilerplate | Inline until there are 2+ consumers |
| NGC/private registry dependency for CI images | Adds auth complexity, fails without secrets | Build images locally from Dockerfile; Docker layer cache makes it fast |
