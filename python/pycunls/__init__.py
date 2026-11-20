# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pycunls -- Python bindings for cuNLS CUDA-accelerated nonlinear least squares solver.

This package exposes the full cuNLS C++ API to Python through a nanobind-based
native extension module (``_pycunls_core``).  Device arrays are passed as any
object with ``__cuda_array_interface__`` (Warp, PyTorch, CuPy, ...) or as a plain
``int`` device pointer; the pure-Python layers (``fit``, ``ProblemBuilder``,
``mpc``, diagnostics) use Warp.

Typical workflow::

    import warp as wp
    import pycunls

    stream  = pycunls.CudaStream()

    # 1. Allocate state(s) on the GPU
    states_gpu = wp.zeros(num * dim, dtype=float)
    sb = pycunls.VectorStateBatch(states_gpu, num, dim=dim)

    # 2. Create factor(s) and observation data
    obs_gpu = wp.array(observations, dtype=float)
    fb = pycunls.PriorVectorFactorBatch(obs_gpu, num, dim=dim)

    # 3. Assemble the problem
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_device_ptr(i) for i in range(num)])

    # 4. Solve (optional: set ``MinimizerOptions.column_scaling`` for scaled
    #    normal equations, or ``LevenbergMarquardtMinimizerOptions.base_options``)
    minimizer = pycunls.LevenbergMarquardtMinimizer()
    summary   = minimizer.minimize(stream, problem)

For user-defined GPU factors implemented in NVIDIA Warp, see
:mod:`pycunls.warp`.
"""

import sys as _sys

try:
    from pycunls import _pycunls_core
    from pycunls._pycunls_core import *  # noqa: F401,F403  (every native class, enum, function)
    from pycunls import _summary  # noqa: F401  (adds to_dict() / report() to the summaries)
    from pycunls._vectors import NUMBERED as _NUMBERED
    from pycunls._logging import log_to_python
    from pycunls.catalog import annotate_classes as _annotate_classes, catalog, describe
    from pycunls.diagnostics import check_jacobians, diagnose, evaluate
    from pycunls.fitting import FitResult, fit
    from pycunls.builder import ProblemBuilder, SolveResult, solve

    globals().update(_NUMBERED)
    _annotate_classes(_pycunls_core)
    __all__ = ([n for n in vars(_pycunls_core) if not n.startswith("_")] + list(_NUMBERED) +
               ["log_to_python", "catalog", "describe", "diagnose", "evaluate", "check_jacobians",
                "fit", "FitResult", "ProblemBuilder", "SolveResult", "solve"])
except ImportError:
    # `python -m pycunls doctor` must start on a broken install (missing native module, Warp
    # or driver) to report what is broken; sys.argv[0] is "-m" only while `python -m` imports
    # this package. Every other import keeps the original error.
    if _sys.argv[:1] != ["-m"]:
        raise

__version__ = "0.4.3"
