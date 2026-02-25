###############################################################################
Installation
###############################################################################

===============================================================================
Prerequisites
===============================================================================

- CUDA Toolkit (with `nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`)
- CMake >= 3.24
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

This produces:

- `libcunls.so` in your build output
- headers and shared library in `install_dir` when provided

===============================================================================
Build with Docker
===============================================================================

1. Install NVIDIA Container Toolkit:
   `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
2. Run:

.. code-block:: bash

   ./scripts/build_cunls_in_docker.sh <CMAKE_BUILD_TYPE = Release | Coverage> [local_install_dir]

When `local_install_dir` is set, install artifacts are copied to:

.. code-block:: text

   <local_install_dir>/
     include/cunls/
     lib/

===============================================================================
Direct CMake build (manual path)
===============================================================================

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cunls_install
   cmake --build build -j
   cmake --install build

===============================================================================
Notes
===============================================================================

- `BUILD_TESTING` is off by default.
- `ENABLE_PROFILING=ON` adds NVTX support.
- cuDSS integration is configured through `cmake/AddCUDSS.cmake`.
