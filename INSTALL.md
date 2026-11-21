# Installing cuNLS

cuNLS ships as a Python package (`pycunls`) and as a CUDA/C++ library (`cunls`). Both need an
NVIDIA GPU. Prebuilt Python wheels are on PyPI as `pycunls-cu13` ([from PyPI](#from-pypi)); the
rest of this page covers building from source.

| Part | CUDA | Driver | Platforms |
|---|---|---|---|
| C++ library (`libcunls`, CMake package) | 12.6+ and 13 | R525+ (CUDA 12), R580+ (CUDA 13) | x86_64; Jetson Orin (CUDA 12.6), Jetson Thor (CUDA 13) |
| Python (`pycunls-cu13`) | 13 | R580+ | x86_64 wheels on PyPI; aarch64 (Jetson Thor) from source |

What decides is the driver, not the installed toolkit: a machine with a CUDA 12 toolkit and an
R580 or newer driver runs the wheel with the CUDA 13 pip libraries. With an older driver (for
example Jetson Orin with JetPack 6), use the C++ library: importing `pycunls` on a driver older
than CUDA 13 raises an `ImportError` that says so.

- [Python package](#python-package-pycunls): install from PyPI, or build the wheel in Docker or
  locally, and check it with `python -m pycunls doctor`.
- [C++ library](#c-library): shared and static builds, Jetson, cuDSS.
- [Testing](#testing) and [building the documentation](#building-the-documentation).

More detail: [Python installation](docs/sphinx/pycunls_installation.rst) and
[C++ installation](docs/sphinx/installation.rst) in the docs.

## Prerequisites

- NVIDIA GPU with a compatible driver (x86_64 or Jetson / aarch64, Linux)
- CUDA Toolkit (`nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`): 12.6 or newer for the C++
  library, 13 for `pycunls`
- CMake >= 3.22
- C++17 compiler
- GNU Make
- Python >= 3.10 (for `pycunls`)

The Docker builds need only Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
the toolchain comes with the image (`scripts/Dockerfile`).

## Python package (pycunls)

`pycunls` exposes cuNLS to Python via [nanobind](https://github.com/wjakob/nanobind). Its
runtime dependencies are NumPy and [NVIDIA Warp](https://github.com/NVIDIA/warp)
(`warp-lang>=1.18`, whose PyPI wheels are CUDA 13 builds): Warp arrays hold the GPU data, and
custom factors, states and `fit()` models are Warp kernels and functions. The bindings accept any
GPU array with `__cuda_array_interface__` (Warp, PyTorch, CuPy, ...) or a raw device pointer. The
wheel statically links the cuNLS core library and dynamically links the CUDA 13 runtime
libraries; it needs an R580 or newer driver.

### From PyPI

Wheels for x86_64 Linux with glibc 2.35 or newer (Ubuntu 22.04 and later; build from source on
older distributions), Python 3.10 to 3.14, CUDA 13 (driver R580 or newer). They do not
bundle the CUDA libraries they link against (`cudart`, `cublas`, `cusparse`, `cusolver`,
`nvJitLink`): install them from NVIDIA's pip packages (pycunls finds them in the same
environment, no `LD_LIBRARY_PATH`) or use a system CUDA 13 toolkit.

```bash
pip install pycunls-cu13 "cuda-toolkit[cudart,cublas,cusparse,cusolver,nvjitlink]==13.2.*"
pip install "nvidia-cudss-cu13==0.8.*"   # optional, only for SparseLinearSolverType.cuDSS
python -m pycunls doctor                 # checks the GPU, the native module, a solve, cuDSS
```

Install the CUDA libraries through the `cuda-toolkit` metapackage, which pins one CUDA release:
unpinned `nvidia-*` packages can mix releases (the newest `nvidia-cusolver-cu13` needs the newest
`nvidia-cublas-cu13`), and the import then fails with `undefined symbol`. A directory on
`LD_LIBRARY_PATH` takes precedence over the pip packages. pycunls 0.4.0 does not search the pip
packages: add `<site-packages>/nvidia/cu13/lib` to `LD_LIBRARY_PATH`, or upgrade.

### Build the wheel in Docker

```bash
./scripts/build_pycunls_in_docker.sh [local_output_dir]
```

The output directory defaults to `./dist`. The script builds the wheel inside a container with
the source mounted read-only. Intermediate build directories live inside the container and are
discarded; only the final `.whl` file is written to the host output directory. The image uses
CUDA 13.2.0 unless `CUDA_VERSION` names another CUDA 13 release; a CUDA 12 `CUDA_VERSION` stops
with the fix, as the wheels are CUDA 13 only.

The wheel's distribution name is `pycunls-cu13`; it is imported as `pycunls`. With
`RELEASE_TAG=v<version>`, the Docker build uses the committed checkout, excluding local changes
and generated files. The tag version must match `python/pyproject.toml` and
`python/pycunls/__init__.py`. Wheels include the source commit and measured CUDA toolkit version
and are repaired with auditwheel while keeping CUDA libraries external.

### Build the wheel locally

```bash
pip install scikit-build-core nanobind
(cd python && pip wheel . --no-build-isolation --no-deps --wheel-dir ../dist)
```

The bindings need a CUDA 13 toolkit: configuring them with CUDA 12 stops with the fix. When
`/usr/local/cuda` is a CUDA 12 toolkit, name the CUDA 13 compiler, for example
`CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc"` before `pip wheel`. The C++
library alone (`BUILD_PYTHON_BINDINGS=OFF`, the default) also builds with CUDA 12.

### Install the wheel

From the repository root:

```bash
pip install ./dist/pycunls*.whl
python -m pycunls doctor     # checks Python, the GPU (via Warp), the native module, a solve, a fit, cuDSS
```

### Editable install for development

For an editable (in-place) install that reflects source changes without rebuilding:

```bash
cd python
pip install scikit-build-core nanobind
pip install -e ".[test]" --no-build-isolation
```

This installs `pycunls` with its runtime dependencies (`numpy`, `warp-lang`) and the test
dependencies (`pytest`, and `cupy-cuda13x` for the one CuPy interoperability test). `test` is the
only optional dependency group.

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
`/usr/local/cuda/bin/nvcc` is used when it exists, and the build compiles native code for every
GPU generation that toolkit supports (`cmake/CudaArchitectures.cmake`): SM 75, 80, 86, 89 and 90;
100 and 120 from CUDA 12.8; on Arm hosts 87 (Jetson Orin, CUDA 12) or 110 (Jetson Thor, CUDA 13).
PTX is kept only for the newest, so no supported GPU compiles kernels at load time. A build for one
GPU is faster, for example a native Jetson Orin build:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=87-real
```

### cuDSS

The cuDSS archive (`CUDSS_PLATFORM=auto`) follows the CUDA Toolkit target: `linux-x86_64`,
`linux-aarch64` for Jetson toolkits, or `linux-sbsa` for Arm server toolkits and CUDA 13 on
Jetson. Toolkits installed without a `targets/` directory (e.g. distro packages under `/usr`) fall
back to `linux-x86_64` on x86_64 and require an explicit platform on aarch64. Set
`-DCUDSS_PLATFORM` explicitly to override it.

Pass `-DCUNLS_ENABLE_CUDSS=OFF` to a local or direct CMake build to build without the cuDSS
backend (the Docker build always packages cuDSS): the cuDSS archive is not
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

Or in Docker, against a wheel built by `build_pycunls_in_docker.sh` (the test runs in the CUDA 13
image that script built):

```bash
./scripts/test_pycunls_in_docker.sh ./dist
WARP_REQUIREMENT="warp-lang==1.18.*" ./scripts/test_pycunls_in_docker.sh ./dist   # the lowest Warp
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

Coverage build (`build_cunls.sh` configures the source one level above the build directory, so
use a top-level build directory):

```bash
./scripts/build_cunls.sh build_coverage Coverage
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
