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
native extension module (``_pycunls_core``).  All GPU memory is managed via
CuPy arrays; constructors accept either a ``cupy.ndarray`` or a plain ``int``
device pointer.

Typical workflow::

    import cupy as cp
    import pycunls

    stream  = pycunls.CudaStream()
    cublas  = pycunls.CublasHandle()

    # 1. Allocate state(s) on the GPU
    states_gpu = cp.zeros(num * dim, dtype=cp.float32)
    sb = pycunls.VectorStateBatch3(states_gpu, num)

    # 2. Create factor(s) and observation data
    obs_gpu = cp.asarray(observations)
    fb = pycunls.PriorVectorFactorBatch3(obs_gpu, num)

    # 3. Assemble the problem
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_block_device_ptr(i) for i in range(num)])

    # 4. Solve (optional: set ``MinimizerOptions.column_scaling`` for scaled
    #    normal equations, or ``LevenbergMarquardtMinimizerOptions.base_options``)
    minimizer = pycunls.LevenbergMarquardtMinimizer()
    summary   = minimizer.minimize(stream, problem)

For user-defined GPU factors implemented in NVIDIA Warp, see
:mod:`pycunls.warp`.
"""

from pycunls._pycunls_core import (
    # --- CUDA helpers ---
    CudaStream,
    CublasHandle,
    # --- Enumerations ---
    SparseLinearSolverType,
    SparseMatrixMultiplierType,
    ColumnScaling,
    # --- Minimizer options & summary ---
    MinimizerOptions,
    MinimizerSummary,
    LevenbergMarquardtMinimizerOptions,
    # --- Minimizers ---
    GaussNewtonMinimizer,
    LevenbergMarquardtMinimizer,
    # --- Problem ---
    Problem,
    # --- State batches (Euclidean) ---
    VectorStateBatch1,
    VectorStateBatch2,
    VectorStateBatch3,
    VectorStateBatch6,
    # --- State batches (Lie groups) ---
    SE3StateBatch,
    SO3StateBatch,
    SO2StateBatch,
    SE2StateBatch,
    Similarity2StateBatch,
    Similarity3StateBatch,
    SL4StateBatch,
    # --- Built-in factor batches ---
    ReprojectionFactorBatch,
    PnPFactorBatch,
    SE3BetweenFactorBatch,
    SE2BetweenFactorBatch,
    SO2BetweenFactorBatch,
    SO3BetweenFactorBatch,
    Similarity2BetweenFactorBatch,
    Similarity3BetweenFactorBatch,
    SL4BetweenFactorBatch,
    VectorBetweenFactorBatch1,
    VectorBetweenFactorBatch2,
    VectorBetweenFactorBatch3,
    VectorBetweenFactorBatch6,
    SE3PriorFactorBatch,
    SL4PriorFactorBatch,
    SO3PriorFactorBatch,
    SO2PriorFactorBatch,
    PriorVectorFactorBatch1,
    PriorVectorFactorBatch2,
    PriorVectorFactorBatch3,
    PriorVectorFactorBatch6,
    PointToPointFactorBatch,
    PointToPlaneFactorBatch,
    SymmetricPointToPlaneFactorBatch,
    # --- Wrapper factor batches ---
    InformationFactorBatch,
    WeightedFactorBatch,
    # --- Custom state trampoline ---
    CustomStateBatch,
    # --- Custom factor trampoline ---
    CustomFactorBatch,
    # --- Robust loss functions ---
    TrivialLossFunctionBatch,
    HuberLossFunctionBatch,
    CauchyLossFunctionBatch,
    ArctanLossFunctionBatch,
    SoftLOneLossFunctionBatch,
    TolerantLossFunctionBatch,
    TukeyLossFunctionBatch,
)

__version__ = "0.1.0"

__all__ = [
    "CudaStream",
    "CublasHandle",
    "SparseLinearSolverType",
    "SparseMatrixMultiplierType",
    "ColumnScaling",
    "MinimizerOptions",
    "MinimizerSummary",
    "LevenbergMarquardtMinimizerOptions",
    "GaussNewtonMinimizer",
    "LevenbergMarquardtMinimizer",
    "Problem",
    "VectorStateBatch1",
    "VectorStateBatch2",
    "VectorStateBatch3",
    "VectorStateBatch6",
    "SE3StateBatch",
    "SO3StateBatch",
    "SO2StateBatch",
    "SE2StateBatch",
    "Similarity2StateBatch",
    "Similarity3StateBatch",
    "SL4StateBatch",
    "ReprojectionFactorBatch",
    "PnPFactorBatch",
    "SE3BetweenFactorBatch",
    "SE2BetweenFactorBatch",
    "SO2BetweenFactorBatch",
    "SO3BetweenFactorBatch",
    "Similarity2BetweenFactorBatch",
    "Similarity3BetweenFactorBatch",
    "SL4BetweenFactorBatch",
    "VectorBetweenFactorBatch1",
    "VectorBetweenFactorBatch2",
    "VectorBetweenFactorBatch3",
    "VectorBetweenFactorBatch6",
    "SE3PriorFactorBatch",
    "SL4PriorFactorBatch",
    "SO3PriorFactorBatch",
    "SO2PriorFactorBatch",
    "PriorVectorFactorBatch1",
    "PriorVectorFactorBatch2",
    "PriorVectorFactorBatch3",
    "PriorVectorFactorBatch6",
    "PointToPointFactorBatch",
    "PointToPlaneFactorBatch",
    "SymmetricPointToPlaneFactorBatch",
    "InformationFactorBatch",
    "WeightedFactorBatch",
    "CustomStateBatch",
    "CustomFactorBatch",
    "TrivialLossFunctionBatch",
    "HuberLossFunctionBatch",
    "CauchyLossFunctionBatch",
    "ArctanLossFunctionBatch",
    "SoftLOneLossFunctionBatch",
    "TolerantLossFunctionBatch",
    "TukeyLossFunctionBatch",
]
