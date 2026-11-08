# Installing cuNLS

cuNLS ships as a Python package (`pycunls`) and as a CUDA/C++ library (`cunls`). Both need an
NVIDIA GPU and are built from source; Python wheels on PyPI are coming.

- [Python package](#python-package-pycunls): build the wheel in Docker or locally, install it,
  check it with `python -m pycunls doctor`.
- [C++ library](#c-library): shared and static builds, Jetson, cuDSS.
- [Testing](#testing) and [building the documentation](#building-the-documentation).

More detail: [Python installation](docs/sphinx/pycunls_installation.rst) and
[C++ installation](docs/sphinx/installation.rst) in the docs.

## Prerequisites

- NVIDIA GPU with a compatible driver (x86_64 or Jetson / aarch64, Linux)
- CUDA Toolkit (`nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`)
- CMake >= 3.22
- C++17 compiler
- GNU Make
- Python >= 3.10 (for `pycunls`)

The Docker builds need only Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
the toolchain comes with the image (`scripts/Dockerfile`).

## Python package (pycunls)

`pycunls` exposes cuNLS to Python via [nanobind](https://github.com/wjakob/nanobind), with
first-class [CuPy](https://cupy.dev/) interop and optional
[NVIDIA Warp](https://github.com/NVIDIA/warp) support for custom factor kernels and `fit()`
models. The wheel statically links the cuNLS core library and dynamically links the CUDA runtime
libraries, so it is specific to the CUDA version it was built with.

### Build the wheel in Docker

```bash
./scripts/build_pycunls_in_docker.sh [local_output_dir]
```

The output directory defaults to `./dist`. The script builds the wheel inside a container with
the source mounted read-only. Intermediate build directories live inside the container and are
discarded; only the final `.whl` file is written to the host output directory.

Docker wheels use the distribution name `pycunls-cu12` or `pycunls-cu13` and the matching CuPy
dependency; both use `import pycunls`. With `RELEASE_TAG=v<version>`, the Docker build uses the
committed checkout, excluding local changes and generated files. The tag version must match
`python/pyproject.toml` and `python/pycunls/__init__.py`. Wheels include the source commit and
measured CUDA toolkit version and are repaired with auditwheel while keeping CUDA libraries
external.

### Build the wheel locally

```bash
cd python
pip install scikit-build-core nanobind
pip wheel . --no-build-isolation --no-deps --wheel-dir ../dist
```

### Install the wheel

```bash
pip install ./dist/pycunls*.whl
python -m pycunls doctor     # checks Python, the GPU, the native module, a solve, a fit, Warp, cuDSS
```

### Editable install for development

For an editable (in-place) install that reflects source changes without rebuilding:

```bash
cd python
pip install scikit-build-core nanobind
pip install -e ".[test]" --no-build-isolation
```

This installs `pycunls` along with all test dependencies (`pytest`, `cupy-cuda12x`,
`warp-lang`). Other optional dependency groups:

```bash
pip install -e ".[warp]"   # warp-lang only
pip install -e ".[all]"    # all optional extras
```

After changing the bindings, regenerate the type stub with
`python scripts/generate_pycunls_stubs.py`.

## C++ library

### Build locally

```bash
./scripts/build_cunls.sh <build_dir> <Release|Coverage> [install_dir]
```

Example — release build with install:

```bash
./scripts/build_cunls.sh build Release /tmp/cunls_install
```

### Build with Docker

```bash
./scripts/build_cunls_in_docker.sh <Release|Coverage> [local_install_dir]
```

The Docker build produces **both** shared and static variants, installed to separate
directories so their CMake package configs don't overwrite each other. The CMake build
directories are kept in the output directory (needed by `test_cunls_in_docker.sh`).

Install artifacts (default `build_docker/`, or the specified directory):

```
<install_dir>/
  shared/               # set CMAKE_PREFIX_PATH here for libcunls.so
    include/cunls/      # headers
    lib/
      libcunls.so       # shared library
      cmake/cunls/      # CMake package config (shared)
  static/               # set CMAKE_PREFIX_PATH here for libcunls.a
    include/cunls/      # headers
    lib/
      libcunls.a        # static library (with bundled deps)
      cmake/cunls/      # CMake package config (static)
  cudss/                # cuDSS headers + libs (optional, dlopen()'d at runtime)
  build_shared/         # CMake build directories (used by the C++ tests)
  build_static/
```

### Direct CMake build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cunls_install
cmake --build build -j
cmake --install build
```

By default this builds a shared library. Pass `-DBUILD_SHARED_LIBS=OFF` to build a static library
instead.

CUDA compiler and architecture selections supplied by the caller are preserved. Without one,
`/usr/local/cuda/bin/nvcc` is used when it exists. For example, a native Jetson Orin build can
avoid PTX JIT compatibility requirements by compiling an SM 87 cubin:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=87-real
```

### cuDSS

The cuDSS archive (`CUDSS_PLATFORM=auto`) follows the CUDA Toolkit target: `linux-x86_64`,
`linux-aarch64` for Jetson toolkits, or `linux-sbsa` for Arm server toolkits and CUDA 13 on
Jetson. Toolkits installed without a `targets/` directory (e.g. distro packages under `/usr`) fall
back to `linux-x86_64` on x86_64 and require an explicit platform on aarch64. Set
`-DCUDSS_PLATFORM` explicitly to override it.

Pass `-DCUNLS_ENABLE_CUDSS=OFF` to build without the cuDSS backend: the cuDSS archive is not
downloaded and `SparseLinearSolverType::cuDSS` throws `std::runtime_error` when a solver is
created. All other solvers work as usual, and the C++ tests skip the cuDSS cases.

### Use it from your project

The install ships a CMake package with the imported target `cunls::cunls`:

```cmake
find_package(cunls REQUIRED)            # CMAKE_PREFIX_PATH=<install_dir>/shared or /static
target_link_libraries(my_app PRIVATE cunls::cunls)
```

The [C++ examples](examples) link `libcunls` and `CUDA::cudart` from an install prefix
(`CUNLS_INSTALL_DIR`); `examples/build_in_docker.sh` builds cuNLS and all examples in one
container.

## Testing

Tests require a GPU.

### Python tests

```bash
pytest -v python/tests
```

Or in Docker, against a wheel built by `build_pycunls_in_docker.sh`:

```bash
./scripts/test_pycunls_in_docker.sh ./dist
```

### C++ tests

```bash
cmake -S . -B build/tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/tests -j
ctest --test-dir build/tests --output-on-failure
```

Or run the test binary directly:

```bash
./build/tests/bin/nls_tests
```

Or in Docker, against the output of `build_cunls_in_docker.sh`:

```bash
./scripts/test_cunls_in_docker.sh ./build_docker
```

Coverage build:

```bash
./scripts/build_cunls.sh build/coverage Coverage
```

## Building the documentation

```bash
python -m pip install -r docs/sphinx/requirements.txt
python -m sphinx -b html docs/sphinx docs/sphinx/_build
```

Or build in Docker:

```bash
bash docs/build_in_docker.sh [output_dir]
```

Code style and contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
