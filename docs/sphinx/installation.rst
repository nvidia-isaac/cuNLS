###############################################################################
Installation
###############################################################################

===============================================================================
Prerequisites
===============================================================================

- CUDA Toolkit (with `nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`)
- CMake >= 3.22
- C++17 compiler
- GNU Make (for the provided scripts)
- NVIDIA GPU driver compatible with your CUDA Toolkit

===============================================================================
Build locally
===============================================================================

From the repository root:

.. code-block:: bash

   ./scripts/build_cunls.sh <build_dir> <CMAKE_BUILD_TYPE = Release | Coverage> [install_dir]

Example (release build + install):

.. code-block:: bash

   ./scripts/build_cunls.sh build Release /tmp/cunls_install

By default, this builds a **shared library** (``libcunls.so``). Set the
``EXTRA_CMAKE_ARGS`` environment variable to override CMake options, for example
to build a static library:

.. code-block:: bash

   EXTRA_CMAKE_ARGS='-DBUILD_SHARED_LIBS=OFF' ./scripts/build_cunls.sh build Release /tmp/cunls_install

When an ``install_dir`` is provided, headers and the library binary are
installed there.

===============================================================================
Build with Docker
===============================================================================

1. Install NVIDIA Container Toolkit:
   `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
2. Run:

.. code-block:: bash

   ./scripts/build_cunls_in_docker.sh <CMAKE_BUILD_TYPE = Release | Coverage> [local_install_dir]

The Docker build produces **both** shared and static library variants.
Intermediate build directories live inside the container and are discarded
automatically; only the final install directory is mounted to the host.

Install artifacts (default ``build_docker/``, or the specified directory):

.. code-block:: text

   <install_dir>/
     include/cunls/          # headers
     lib/
       libcunls.so           # shared library
       libcunls.a            # static library (with bundled deps)
       cmake/cunls/          # CMake package config

===============================================================================
Direct CMake build (manual path)
===============================================================================

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cunls_install
   cmake --build build -j
   cmake --install build

Pass ``-DBUILD_SHARED_LIBS=OFF`` to build a static library instead of a shared
one.

===============================================================================
Notes
===============================================================================

- ``BUILD_SHARED_LIBS`` defaults to ``ON`` (shared library). Set to ``OFF``
  for a static archive with bundled dependencies.
- ``BUILD_TESTING`` is off by default.
- ``ENABLE_PROFILING=ON`` adds NVTX support.
- cuDSS is an **optional runtime dependency**. cunls never links against
  ``libcudss.so``; instead it loads it with ``dlopen()`` the first time
  ``SparseLinearSolverType::cuDSS`` is used (see ``cmake/AddCUDSS.cmake`` and
  ``cunls/common/cudss_dynamic.h``). This means:

  - Neither ``libcunls.so``/``libcunls.a`` nor the ``pycunls`` wheel bundle or
    depend on cuDSS at load time — every other solver (``DenseLDLT``,
    ``DenseCholesky``, ``DenseQR``, ``BlockSparsePCG``) works with no cuDSS
    installed at all.
  - To use the cuDSS solver, download a cuDSS release matching your CUDA
    major version from
    `NVIDIA's cuDSS redistributables <https://developer.download.nvidia.com/compute/cudss/redist/libcudss/>`_
    (or install it via your platform's package manager), then add the
    directory containing ``libcudss.so`` to ``LD_LIBRARY_PATH`` **in the
    environment the process is launched with** — e.g. ``export
    LD_LIBRARY_PATH=...`` in the shell before running your executable or
    ``python``. The dynamic loader reads ``LD_LIBRARY_PATH`` once at process
    startup, so setting it from inside an already-running process (for
    example via ``os.environ[...]`` in a live Python interpreter, before
    constructing a ``cuDSSLinearSolver``) has no effect on ``dlopen()``.
  - Requesting the cuDSS solver without ``libcudss.so`` reachable raises a
    ``std::runtime_error`` (Python: ``RuntimeError``) describing how to fix
    it, rather than a linker error or crash.
  - CI still downloads cuDSS at build time (for headers) and publishes its
    shared/static libraries as a separate ``cudss-*`` artifact, distinct from
    the ``cunls-*`` and ``pycunls-*`` artifacts.
- ``build_cunls.sh`` supports two environment variables for advanced use:

  - ``CUNLS_SOURCE_DIR`` — override the CMake source directory (defaults to
    the parent of the build directory).
  - ``EXTRA_CMAKE_ARGS`` — pass additional flags to the CMake configure step
    (e.g. ``-DBUILD_SHARED_LIBS=OFF``).
