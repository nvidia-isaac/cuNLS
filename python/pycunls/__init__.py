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
    sb = pycunls.VectorStateBatch3(states_gpu, num)

    # 2. Create factor(s) and observation data
    obs_gpu = wp.array(observations, dtype=float)
    fb = pycunls.PriorVectorFactorBatch3(obs_gpu, num)

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
    from pycunls._pycunls_core import (
        # --- CUDA helpers ---
        CudaStream,
        # --- Logging ---
        Verbosity,
        set_log_verbosity,
        # --- Enumerations ---
        LinearSystemType,
        SparseLinearSolverType,
        ColumnScaling,
        JacobianMode,
        NumericDiffMethod,
        # --- Minimizer options & summary ---
        MinimizerOptions,
        NumericDiffOptions,
        MinimizerSummary,
        TerminationReason,
        LevenbergMarquardtMinimizerOptions,
        # --- Minimizers ---
        Minimizer,
        GaussNewtonMinimizer,
        LevenbergMarquardtMinimizer,
        # --- Constrained minimization (augmented Lagrangian) ---
        ConstraintKind,
        AugmentedLagrangianMinimizerOptions,
        AugmentedLagrangianMinimizerStatus,
        AugmentedLagrangianMinimizerSummary,
        AugmentedLagrangianMinimizer,
        # --- RANSAC minimizers ---
        RansacRole,
        RansacScoring,
        RansacLinearSolverType,
        RansacFactorBatchOptions,
        RansacMinimizerOptions,
        RansacLevenbergMarquardtMinimizerOptions,
        RansacSummary,
        RansacMinimizer,
        RansacGaussNewtonMinimizer,
        RansacLevenbergMarquardtMinimizer,
        # --- Problem and the abstract batch bases ---
        Problem,
        StateBatch,
        FactorBatch,
        LossFunctionBatch,
        # --- State batches (Euclidean) ---
        VectorStateBatch1,
        VectorStateBatch2,
        VectorStateBatch3,
        VectorStateBatch4,
        VectorStateBatch6,
        VectorStateBatch12,
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
        VectorBetweenFactorBatch4,
        VectorBetweenFactorBatch6,
        VectorBetweenFactorBatch12,
        SE2PriorFactorBatch,
        SE3PriorFactorBatch,
        Similarity2PriorFactorBatch,
        Similarity3PriorFactorBatch,
        SL4PriorFactorBatch,
        SO3PriorFactorBatch,
        SO2PriorFactorBatch,
        PriorVectorFactorBatch1,
        PriorVectorFactorBatch2,
        PriorVectorFactorBatch3,
        PriorVectorFactorBatch4,
        PriorVectorFactorBatch6,
        PriorVectorFactorBatch12,
        PointToPointFactorBatch,
        PointToPlaneFactorBatch,
        SymmetricPointToPlaneFactorBatch,
        # --- Wrapper factor batches ---
        InformationFactorBatch,
        WeightedFactorBatch,
        # --- Constraint factor batches ---
        ConstraintFactorBatchBase,
        ConstraintFactorBatch,
        BoundFactorBatch1,
        BoundFactorBatch2,
        BoundFactorBatch3,
        BoundFactorBatch4,
        BoundFactorBatch6,
        BoundFactorBatch12,
        HalfspaceFactorBatch1,
        HalfspaceFactorBatch2,
        HalfspaceFactorBatch3,
        HalfspaceFactorBatch4,
        HalfspaceFactorBatch6,
        HalfspaceFactorBatch12,
        # --- Dynamics factor batches ---
        SE2DifferentialDriveFactorBatch,
        SE2KinematicBicycleFactorBatch,
        SE3DifferentialDriveFactorBatch,
        SE3KinematicBicycleFactorBatch,
        SE2KinematicsFactorBatch,
        SO3KinematicsFactorBatch,
        SE3KinematicsFactorBatch,
        ConstantVelocitySE3FactorBatch,
        ConstantVelocityInformationSE3FactorBatch,
        ConstantAccelerationSE3FactorBatch,
        ConstantAccelerationInformationSE3FactorBatch,
        ConstantVelocitySO3FactorBatch,
        ConstantVelocityInformationSO3FactorBatch,
        ConstantAccelerationSO3FactorBatch,
        ConstantAccelerationInformationSO3FactorBatch,
        ConstantVelocitySE2FactorBatch,
        ConstantVelocityInformationSE2FactorBatch,
        ConstantAccelerationSE2FactorBatch,
        ConstantAccelerationInformationSE2FactorBatch,
        ConstantVelocitySO2FactorBatch,
        ConstantVelocityInformationSO2FactorBatch,
        ConstantAccelerationSO2FactorBatch,
        ConstantAccelerationInformationSO2FactorBatch,
        QuadrotorParameters,
        QuadrotorFactorBatch,
        QuadrupedParameters,
        QuadrupedFactorBatch,
        ImuParameters,
        ImuFactorBatch,
        SE2DiskClearanceFactorBatch,
        SE3SphereClearanceFactorBatch,
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
        ScaledLossFunctionBatch,
    )

    from pycunls import _pycunls_core
    from pycunls import _summary  # noqa: F401  (adds to_dict() / report() to the summaries)
    from pycunls._logging import log_to_python
    from pycunls.catalog import annotate_classes as _annotate_classes, catalog, describe
    from pycunls.diagnostics import check_jacobians, diagnose, evaluate
    from pycunls.fitting import FitResult, fit
    from pycunls.builder import ProblemBuilder, SolveResult, solve

    _annotate_classes(_pycunls_core)
except ImportError:
    # `python -m pycunls doctor` must start on a broken install (missing native module, Warp
    # or driver) to report what is broken; sys.argv[0] is "-m" only while `python -m` imports
    # this package. Every other import keeps the original error.
    if _sys.argv[:1] != ["-m"]:
        raise

__version__ = "0.3.0"

__all__ = [
    "CudaStream",
    "Verbosity",
    "set_log_verbosity",
    "log_to_python",
    "catalog",
    "describe",
    "diagnose",
    "evaluate",
    "check_jacobians",
    "fit",
    "FitResult",
    "ProblemBuilder",
    "SolveResult",
    "solve",
    "LinearSystemType",
    "SparseLinearSolverType",
    "ColumnScaling",
    "JacobianMode",
    "NumericDiffMethod",
    "MinimizerOptions",
    "NumericDiffOptions",
    "MinimizerSummary",
    "TerminationReason",
    "LevenbergMarquardtMinimizerOptions",
    "Minimizer",
    "GaussNewtonMinimizer",
    "LevenbergMarquardtMinimizer",
    "ConstraintKind",
    "AugmentedLagrangianMinimizerOptions",
    "AugmentedLagrangianMinimizerStatus",
    "AugmentedLagrangianMinimizerSummary",
    "AugmentedLagrangianMinimizer",
    "RansacRole",
    "RansacScoring",
    "RansacLinearSolverType",
    "RansacFactorBatchOptions",
    "RansacMinimizerOptions",
    "RansacLevenbergMarquardtMinimizerOptions",
    "RansacSummary",
    "RansacMinimizer",
    "RansacGaussNewtonMinimizer",
    "RansacLevenbergMarquardtMinimizer",
    "Problem",
    "StateBatch",
    "FactorBatch",
    "LossFunctionBatch",
    "VectorStateBatch1",
    "VectorStateBatch2",
    "VectorStateBatch3",
    "VectorStateBatch4",
    "VectorStateBatch6",
    "VectorStateBatch12",
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
    "VectorBetweenFactorBatch4",
    "VectorBetweenFactorBatch6",
    "VectorBetweenFactorBatch12",
    "SE2PriorFactorBatch",
    "SE3PriorFactorBatch",
    "Similarity2PriorFactorBatch",
    "Similarity3PriorFactorBatch",
    "SL4PriorFactorBatch",
    "SO3PriorFactorBatch",
    "SO2PriorFactorBatch",
    "PriorVectorFactorBatch1",
    "PriorVectorFactorBatch2",
    "PriorVectorFactorBatch3",
    "PriorVectorFactorBatch4",
    "PriorVectorFactorBatch6",
    "PriorVectorFactorBatch12",
    "PointToPointFactorBatch",
    "PointToPlaneFactorBatch",
    "SymmetricPointToPlaneFactorBatch",
    "InformationFactorBatch",
    "WeightedFactorBatch",
    "ConstraintFactorBatchBase",
    "ConstraintFactorBatch",
    "BoundFactorBatch1",
    "BoundFactorBatch2",
    "BoundFactorBatch3",
    "BoundFactorBatch4",
    "BoundFactorBatch6",
    "BoundFactorBatch12",
    "HalfspaceFactorBatch1",
    "HalfspaceFactorBatch2",
    "HalfspaceFactorBatch3",
    "HalfspaceFactorBatch4",
    "HalfspaceFactorBatch6",
    "HalfspaceFactorBatch12",
    "SE2DifferentialDriveFactorBatch",
    "SE2KinematicBicycleFactorBatch",
    "SE3DifferentialDriveFactorBatch",
    "SE3KinematicBicycleFactorBatch",
    "SE2KinematicsFactorBatch",
    "SO3KinematicsFactorBatch",
    "SE3KinematicsFactorBatch",
    "ConstantVelocitySE3FactorBatch",
    "ConstantVelocityInformationSE3FactorBatch",
    "ConstantAccelerationSE3FactorBatch",
    "ConstantAccelerationInformationSE3FactorBatch",
    "ConstantVelocitySO3FactorBatch",
    "ConstantVelocityInformationSO3FactorBatch",
    "ConstantAccelerationSO3FactorBatch",
    "ConstantAccelerationInformationSO3FactorBatch",
    "ConstantVelocitySE2FactorBatch",
    "ConstantVelocityInformationSE2FactorBatch",
    "ConstantAccelerationSE2FactorBatch",
    "ConstantAccelerationInformationSE2FactorBatch",
    "ConstantVelocitySO2FactorBatch",
    "ConstantVelocityInformationSO2FactorBatch",
    "ConstantAccelerationSO2FactorBatch",
    "ConstantAccelerationInformationSO2FactorBatch",
    "QuadrotorParameters",
    "QuadrotorFactorBatch",
    "QuadrupedParameters",
    "QuadrupedFactorBatch",
    "ImuParameters",
    "ImuFactorBatch",
    "SE2DiskClearanceFactorBatch",
    "SE3SphereClearanceFactorBatch",
    "CustomStateBatch",
    "CustomFactorBatch",
    "TrivialLossFunctionBatch",
    "HuberLossFunctionBatch",
    "CauchyLossFunctionBatch",
    "ArctanLossFunctionBatch",
    "SoftLOneLossFunctionBatch",
    "TolerantLossFunctionBatch",
    "TukeyLossFunctionBatch",
    "ScaledLossFunctionBatch",
]
