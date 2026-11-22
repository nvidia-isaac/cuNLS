###############################################################################
Installation
###############################################################################

cuNLS is installed for Python as the ``pycunls`` package. The wheel
statically links the cuNLS core library, so no separate C++ installation is
needed.

.. note::

   **Using cuNLS from C++.** To build and install the C++ library
   (``libcunls.so`` / ``libcunls.a``, headers and CMake config) instead, see
   :doc:`installation`.

.. list-table:: Supported platforms
   :header-rows: 1

   * - Part
     - CUDA
     - Driver
     - Platforms
   * - C++ library (``libcunls``, CMake package)
     - 12.6+ and 13
     - R525+ (CUDA 12), R580+ (CUDA 13)
     - x86_64; Jetson Orin (CUDA 12.6), Jetson Thor (CUDA 13)
   * - Python (``pycunls-cu13``)
     - 13
     - R580+
     - x86_64 wheels on PyPI; aarch64 (Jetson Thor) from source

On CUDA 12 systems (for example Jetson Orin with JetPack 6), use the C++
library: importing ``pycunls`` on a driver older than CUDA 13 raises an
``ImportError`` that says so.

===============================================================================
Install from PyPI
===============================================================================

Wheels for x86_64 Linux with glibc 2.35 or newer (Ubuntu 22.04 and later; build from
source on older distributions), Python 3.10 to 3.14, CUDA 13 (driver R580 or newer):

.. code-block:: bash

   pip install pycunls-cu13
   python -m pycunls doctor

The wheel depends on NVIDIA's pip packages for the CUDA libraries it links (see
`CUDA libraries`_).

===============================================================================
Prerequisites (source builds)
===============================================================================

- CUDA Toolkit 13 (with `nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`);
  configuring the bindings with CUDA 12 stops with the fix (the C++ library
  alone also builds with CUDA 12)
- CMake >= 3.22
- C++17 compiler
- Python >= 3.10
- NVIDIA GPU driver R580 or newer

===============================================================================
Install from source (pip)
===============================================================================

From the ``python/`` directory inside the repository:

.. code-block:: bash

   cd python
   pip install .

This uses `scikit-build-core` to configure CMake with
``-DBUILD_PYTHON_BINDINGS=ON -DBUILD_SHARED_LIBS=OFF``, compiles the
``_pycunls_core`` native extension via `nanobind`, statically links the
cuNLS core library, and installs the ``pycunls`` package into your Python
environment.

To also install the test dependencies:

.. code-block:: bash

   pip install ".[test]"   # pytest, and cupy-cuda13x for one interoperability test

===============================================================================
Build a wheel
===============================================================================

.. code-block:: bash

   cd python
   pip wheel . -w dist/

The resulting ``.whl`` file in ``dist/`` can be installed on any machine with
CUDA 13 libraries and an R580 or newer driver. When ``/usr/local/cuda`` is a
CUDA 12 toolkit, name the CUDA 13 compiler, for example
``CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc"``.

===============================================================================
Dependencies
===============================================================================

**Required (installed automatically by pip):**

- `warp-lang <https://nvidia.github.io/warp/>`_ (``>=1.18``, whose PyPI
  wheels are CUDA 13 builds) — `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_: GPU arrays (``wp.array``),
  custom factor and state kernels (see :doc:`pycunls_tutorial`) and
  ``pycunls.fit`` models. All pycunls constructors accept a ``wp.array``, any
  other array with ``__cuda_array_interface__`` (PyTorch, CuPy, ...) or a raw
  ``int`` device pointer.
- `numpy <https://numpy.org/>`_ — host arrays (results, ``fit`` and
  ``ProblemBuilder`` inputs).

**Build-time only:**

- `scikit-build-core <https://scikit-build-core.readthedocs.io/>`_ — CMake
  build backend for Python packaging.
- `nanobind <https://nanobind.readthedocs.io/>`_ — lightweight C++/Python
  binding library (fetched by CMake during build).

===============================================================================
Verify the installation
===============================================================================

.. code-block:: bash

   python -c "import pycunls; print(pycunls.__version__)"

You should see the installed version string (e.g. ``0.1.0``).

===============================================================================
Optional: cuDSS sparse solver
===============================================================================

cuDSS is an **optional runtime dependency**: the ``pycunls`` wheel neither
bundles nor requires it, and every other sparse linear solver works without
it. To use ``SparseLinearSolverType.cuDSS``, either ``pip install
"nvidia-cudss-cu13==0.8.*"`` into the same
environment, which ``pycunls`` finds without any setup, or make a cuDSS 0.8
``libcudss.so`` reachable through ``LD_LIBRARY_PATH`` *before* starting
``python``; requesting it otherwise raises ``RuntimeError``. See
:ref:`installation:Notes` for details.

===============================================================================
CUDA libraries
===============================================================================

The wheel links ``cudart``, ``cublas``, ``cusparse`` and ``cusolver`` without
bundling them, and depends on NVIDIA's pip packages for them
(``cuda-toolkit[cudart,cublas,cusparse,cusolver]>=13.0,<14``). PyTorch's CUDA 13
wheels depend on the same packages, pinned to one release; pycunls accepts any
CUDA 13 release, so one copy of each library serves both. The native module
searches, in order: ``LD_LIBRARY_PATH``, the NVIDIA pip packages next to it in
``site-packages`` (``nvidia/cu13/lib``), then the system library cache. A
directory on ``LD_LIBRARY_PATH`` (such as a system CUDA 13 toolkit's
``lib64``) therefore takes precedence over the pip packages.

Next: :doc:`pycunls_quick_start`.
