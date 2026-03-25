###############################################################################
pycunls Installation
###############################################################################

===============================================================================
Prerequisites
===============================================================================

- CUDA Toolkit (with `nvcc`, `cudart`, `cuBLAS`, `cuSPARSE`, `cuSOLVER`)
- CMake >= 3.24
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

To include the optional `NVIDIA Warp <https://developer.nvidia.com/warp-python>`_
integration and test dependencies:

.. code-block:: bash

   pip install ".[all]"

Or selectively:

.. code-block:: bash

   pip install ".[warp]"   # Warp support only
   pip install ".[test]"   # test dependencies (pytest, cupy, warp-lang)

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

- `cupy-cuda12x <https://cupy.dev/>`_ — GPU arrays and CUDA interop. All
  pycunls constructors accept either a ``cupy.ndarray`` or a raw ``int``
  device pointer.

**Optional:**

- `warp-lang <https://nvidia.github.io/warp/>`_ — `NVIDIA Warp
  <https://developer.nvidia.com/warp-python>`_ for authoring
  custom factor and state kernels in Python (see
  :doc:`pycunls_tutorial`). Install with ``pip install warp-lang`` or
  ``pip install "pycunls[warp]"``.

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
