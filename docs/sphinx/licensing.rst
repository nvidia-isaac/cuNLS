###############################################################################
Licensing
###############################################################################

The following third-party components are used by cuNLS.

===============================================================================
Runtime / core build dependencies
===============================================================================

- **spdlog**
  - License: MIT
  - URL: `https://github.com/gabime/spdlog`

- **NVIDIA CUDA Toolkit libraries** (`cudart`, `cuBLAS`, `cuSPARSE`,
  `cuSOLVER`)
  - License: NVIDIA CUDA End User License Agreement
  - URL: `https://docs.nvidia.com/cuda/eula/index.html`

===============================================================================
Optional dependency
===============================================================================

- **cuDSS** (only required to use `SparseLinearSolverType::cuDSS`; loaded via
  `dlopen()` at runtime, not linked or bundled — see :doc:`installation`)
  - License: Proprietary (NVIDIA)
  - URL: `https://developer.nvidia.com/cudss`

- **NVTX** (used when `ENABLE_PROFILING=ON`)
  - License: Apache-2.0
  - URL: `https://github.com/NVIDIA/NVTX`

===============================================================================
Python bindings (pycunls) dependencies
===============================================================================

- **nanobind** (build-time; C++/Python binding library)
  - License: BSD-3-Clause
  - URL: `https://github.com/wjakob/nanobind`

- **`NVIDIA Warp <https://developer.nvidia.com/warp-python>`_** (runtime; GPU arrays, custom
  Warp factor/state kernels and ``fit`` models)
  - License: Apache-2.0
  - URL: `https://github.com/NVIDIA/warp`

- **NumPy** (runtime; host arrays)
  - License: BSD-3-Clause
  - URL: `https://github.com/numpy/numpy`

===============================================================================
Test-only dependency
===============================================================================

- **GoogleTest** (used when `BUILD_TESTING=ON`)
  - License: BSD-3-Clause
  - URL: `https://github.com/google/googletest`

- **CuPy** (``pycunls[test]`` only; one interoperability test)
  - License: MIT
  - URL: `https://github.com/cupy/cupy`

===============================================================================
License texts in repository
===============================================================================

See:

- `NOTICE`
- `third_party/LICENSES/`
