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

===============================================================================
Prerequisites
===============================================================================

- CUDA Toolkit (with `nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`)
- CMake >= 3.22
- C++17 compiler
- Python >= 3.10
- NVIDIA GPU driver compatible with your CUDA Toolkit

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

   pip install ".[test]"   # pytest, and cupy-cuda12x for one interoperability test

===============================================================================
Build a wheel
===============================================================================

.. code-block:: bash

   cd python
   pip wheel . -w dist/

The resulting ``.whl`` file in ``dist/`` can be installed on any machine with
a compatible CUDA Toolkit and GPU driver.

===============================================================================
Dependencies
===============================================================================

**Required (installed automatically by pip):**

- `warp-lang <https://nvidia.github.io/warp/>`_ (``<1.18``) — `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_: GPU arrays (``wp.array``),
  custom factor and state kernels (see :doc:`pycunls_tutorial`) and
  ``pycunls.fit`` models. All pycunls constructors accept a ``wp.array``, any
  other array with ``__cuda_array_interface__`` (PyTorch, CuPy, ...) or a raw
  ``int`` device pointer. Warp 1.18 is excluded because its PyPI wheels need
  an R580+ driver (older drivers, such as JetPack 6 on Jetson Orin, leave
  Warp CPU-only).
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
it. To use ``SparseLinearSolverType.cuDSS``, make ``libcudss.so`` reachable
through ``LD_LIBRARY_PATH`` *before* starting ``python``; requesting it
otherwise raises ``RuntimeError``. See :ref:`installation:Notes` for details.

Next: :doc:`pycunls_quick_start`.
