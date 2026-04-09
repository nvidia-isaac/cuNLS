# AGENTS.md

## Repository Overview

cuNLS is a GPU-accelerated nonlinear least-squares solver library written in CUDA/C++. It targets NVIDIA GPUs and runs on Linux (x86_64 and aarch64).

### Key facts

- **Language**: CUDA/C++ (C++17), with Python bindings via nanobind
- **Build system**: CMake 3.24+, `nvcc`, Unix Makefiles
- **Testing**: GoogleTest (C++), pytest (Python). Tests require a GPU.
- **Dependencies**: cuDSS, spdlog, CUDA Toolkit (cusparse, cublas, cusolver). All fetched via CMake `FetchContent`.
- **Docker**: All builds run inside Docker containers via `scripts/Dockerfile`. Parameterized by `CUDA_VERSION` and `UBUNTU_VERSION` build args.
- **CI**: GitHub Actions on nvidia-isaac org shared self-hosted GPU runners (AWS, auto-scaling, `[self-hosted, gpu]`). Nightly schedule with 4-entry matrix (CUDA 12/13 x Ubuntu 22.04/24.04).
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
cmake/              # CMake modules (AddCUDSS, AddSpdlog, BundleStaticDeps)
.github/workflows/  # CI workflows (nightly, docker-ci, static_docs)
```

### Build and test locally

```bash
# C++ build (shared + static)
./scripts/build_cunls_in_docker.sh Release

# C++ tests
./scripts/test_cunls_in_docker.sh ./build_docker

# Python wheel
./scripts/build_pycunls_in_docker.sh

# Python tests
./scripts/test_pycunls_in_docker.sh ./dist
```

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

Use the distro-default GCC (not a pinned version). Use the Kitware-latest CMake (not a pinned version unless there's a proven compatibility issue). Ubuntu 22.04 ships GCC 11, Ubuntu 24.04 ships GCC 13 -- both support C++17. Only pin versions when there is a documented, tested incompatibility.

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

All GPU jobs use `runs-on: [self-hosted, gpu]`. CUDA version, Ubuntu version, and architecture are controlled by Docker build args, not runner labels. The nvidia-isaac org has a shared pool of 10 auto-scaling GPU runners -- do not encode CUDA or OS versions in runner tags.

### 11. Artifacts flow through the filesystem, not between jobs

Within a single job, steps share a host directory (e.g., `./output/`). Use GitHub artifact upload/download only for cross-job transfer (e.g., from build-matrix to nightly-status). Do not upload and re-download artifacts between steps in the same job.

### 12. Every nightly produces a complete, dated release

A nightly release tagged `nightly-YYYY-MM-DD` includes, per matrix entry: C++ libraries (shared `.so` + static `.a`), headers, a Python wheel, and JUnit test result XMLs. All attached as GitHub Release assets with filenames that include the CUDA version and Ubuntu version.
