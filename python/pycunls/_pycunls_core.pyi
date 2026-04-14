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

"""Type stubs for the pycunls C++ extension module."""

from __future__ import annotations

import enum
from typing import Any, Sequence, overload

# ---------------------------------------------------------------------------
# Type alias for arguments that accept a GPU device pointer.
# Pass either a raw ``int`` address or any object with a ``.data.ptr``
# attribute (e.g. a ``cupy.ndarray``).
# ---------------------------------------------------------------------------
type DevicePointer = int | Any

# ===================================================================
# Utility types
# ===================================================================

class CudaStream:
    """RAII wrapper for a CUDA stream."""

    def __init__(self, sync_on_destroy: bool = False) -> None: ...
    def get_stream(self) -> int:
        """Return the underlying ``cudaStream_t`` as an integer handle."""
        ...

class CublasHandle:
    """RAII wrapper for a cuBLAS handle."""

    def __init__(self) -> None: ...

# ===================================================================
# Enumerations
# ===================================================================

class SparseLinearSolverType(enum.IntEnum):
    cuDSS = ...
    DenseLDLT = ...
    DenseCholesky = ...
    DenseQR = ...

class SparseMatrixMultiplierType(enum.IntEnum):
    cuSPARSE = ...
    Fast = ...

class ColumnScaling(enum.IntEnum):
    """Diagonal scaling mode for the GN/LM normal equations."""

    none = ...
    hessian_diagonal = ...
    jacobian_column_norm = ...

# ===================================================================
# Options and summary
# ===================================================================

class MinimizerOptions:
    """Options for Gauss-Newton and Levenberg-Marquardt minimizers."""

    max_num_iterations: int
    state_tolerance: float
    cost_tolerance: float
    max_consecutive_rejected_steps: int
    sparse_linear_solver_type: SparseLinearSolverType
    sparse_square_multiplier_type: SparseMatrixMultiplierType
    column_scaling: ColumnScaling
    disable_safety_checks: bool

    def __init__(self) -> None: ...

class MinimizerSummary:
    """Summary of a minimization run."""

    @property
    def num_iterations(self) -> int: ...
    @property
    def initial_cost(self) -> float: ...
    @property
    def final_cost(self) -> float: ...
    @property
    def iteration_costs(self) -> list[float]: ...
    def __repr__(self) -> str: ...

class LevenbergMarquardtMinimizerOptions:
    """Options for the Levenberg-Marquardt minimizer."""

    base_options: MinimizerOptions
    initial_lambda: float
    lambda_upscale: float
    lambda_downscale: float
    lambda_max: float
    lambda_min: float
    step_accept_threshold: float
    lambda_downscale_threshold: float

    def __init__(self) -> None: ...

# ===================================================================
# State batches
# ===================================================================

class StateBatch:
    """Abstract base class for batched state blocks on a manifold."""

    ...

class VectorStateBatch1(StateBatch):
    """Euclidean 1-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, num_blocks: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch2(StateBatch):
    """Euclidean 2-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, num_blocks: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch3(StateBatch):
    """Euclidean 3-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, num_blocks: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch6(StateBatch):
    """Euclidean 6-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, num_blocks: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SE3StateBatch(StateBatch):
    """SE(3) state batch. Ambient=16 (4x4 matrix), Tangent=6."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SO3StateBatch(StateBatch):
    """SO(3) state batch. Ambient=9 (3x3 matrix), Tangent=3."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SO2StateBatch(StateBatch):
    """SO(2) state batch. Ambient=4 (2x2 matrix), Tangent=1."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SE2StateBatch(StateBatch):
    """SE(2) state batch. Ambient=9 (3x3 matrix), Tangent=3."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class Similarity2StateBatch(StateBatch):
    """2D similarity state batch. Ambient=9, Tangent=4."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class Similarity3StateBatch(StateBatch):
    """3D similarity state batch. Ambient=16, Tangent=7."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SL4StateBatch(StateBatch):
    """SL(4) state batch. Ambient=16 (4x4 matrix), Tangent=15."""

    @overload
    def __init__(
        self, cublas_handle: CublasHandle, data: DevicePointer, num_blocks: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        cublas_handle: CublasHandle,
        data: DevicePointer,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class CustomStateBatch(StateBatch):
    """Base class for user-defined state batches. Override ``plus()`` in Python.

    The ``plus()`` method implements the manifold retraction:
    ``x_plus_delta = x (+) delta``.  All pointers are passed as integer handles.
    """

    @overload
    def __init__(
        self,
        data: DevicePointer,
        ambient_size: int,
        tangent_size: int,
        num_blocks: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        ambient_size: int,
        tangent_size: int,
        num_blocks: int,
        const_state_ids: DevicePointer,
        num_const_state_blocks: int,
    ) -> None: ...
    def plus(
        self,
        x_ptr: int,
        delta_ptr: int,
        x_plus_delta_ptr: int,
        stream_handle: int,
    ) -> None:
        """Apply manifold retraction. Override in subclasses."""
        ...
    def state_block_device_ptr(self, index: int) -> int: ...
    @property
    def num_state_blocks(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

# ===================================================================
# Factor batches
# ===================================================================

class FactorBatch:
    """Abstract base class for batched factors."""

    ...

class CustomFactorBatch(FactorBatch):
    """Base class for user-defined factors. Override ``evaluate()`` in Python."""

    def __init__(
        self,
        residual_size: int,
        state_block_sizes: Sequence[int],
        num_factors: int,
    ) -> None: ...
    def evaluate(
        self,
        residuals_ptr: int,
        jacobians_ptr: int,
        state_pointers_ptr: int,
        stream_handle: int,
    ) -> bool:
        """Compute residuals and Jacobians. Override in subclasses."""
        ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class ReprojectionFactorBatch(FactorBatch):
    """Batched 2D reprojection factor. Residual=2, States=[SE3(6), Point(3)].

    Observations must be in normalized image coordinates (K^-1 applied).
    """

    def __init__(
        self,
        observations: DevicePointer,
        num_observations: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PnPFactorBatch(FactorBatch):
    """PnP reprojection: fixed 3D points (constructor), pose-only Jacobian.

    Residual=2, single SE3 state per factor. Observations normalized (K^-1).
    """

    @overload
    def __init__(
        self,
        observations: DevicePointer,
        points_world: DevicePointer,
        num_observations: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    @overload
    def __init__(
        self,
        observations: DevicePointer,
        poses_camera_from_rig: DevicePointer,
        points_world: DevicePointer,
        num_observations: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SE3BetweenFactorBatch(FactorBatch):
    """Batched SE(3) between factor. Residual=6, States=[SE3(6), SE3(6)]."""

    def __init__(
        self,
        deltas: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SE3PriorFactorBatch(FactorBatch):
    """Batched SE(3) prior factor. Residual=6, States=[SE3(6)]."""

    def __init__(
        self,
        observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SO3PriorFactorBatch(FactorBatch):
    """Batched SO(3) prior factor. Residual=3, States=[SO3(3)]."""

    def __init__(
        self,
        observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SO2PriorFactorBatch(FactorBatch):
    """Batched SO(2) prior factor. Residual=1, States=[SO2(1)]."""

    def __init__(
        self,
        observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch1(FactorBatch):
    """Prior factor for 1-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch2(FactorBatch):
    """Prior factor for 2-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch3(FactorBatch):
    """Prior factor for 3-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch6(FactorBatch):
    """Prior factor for 6-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SE2BetweenFactorBatch(FactorBatch):
    """Batched SE(2) between factor. Residual=3, States=[SE2(3), SE2(3)]."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SO2BetweenFactorBatch(FactorBatch):
    """Batched SO(2) between factor. Residual=1, States=[SO2(1), SO2(1)]."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SO3BetweenFactorBatch(FactorBatch):
    """Batched SO(3) between factor. Residual=3, States=[SO3(3), SO3(3)]."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class Similarity2BetweenFactorBatch(FactorBatch):
    """Batched Sim(2) between factor. Residual=4, States=[Sim2(4), Sim2(4)]."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class Similarity3BetweenFactorBatch(FactorBatch):
    """Batched Sim(3) between factor. Residual=7, States=[Sim3(7), Sim3(7)]."""

    def __init__(
        self,
        cublas_handle: CublasHandle,
        deltas: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SL4PriorFactorBatch(FactorBatch):
    """Batched SL(4) prior factor. Residual=15, States=[SL4(15)]."""

    def __init__(
        self,
        observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SL4BetweenFactorBatch(FactorBatch):
    """Batched SL(4) between factor. Residual=15, States=[SL4(15), SL4(15)]."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch1(FactorBatch):
    """Between factor for 1-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch2(FactorBatch):
    """Between factor for 2-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch3(FactorBatch):
    """Between factor for 3-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch6(FactorBatch):
    """Between factor for 6-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, num_factors: int) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PointToPointFactorBatch(FactorBatch):
    """Batched point-to-point factor: residual = p - T*q. Residual=3, States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class PointToPlaneFactorBatch(FactorBatch):
    """Batched point-to-plane factor: residual = Nq^T*(p - T*q). Residual=1, States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        nq_observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class SymmetricPointToPlaneFactorBatch(FactorBatch):
    """Batched symmetric point-to-plane factor. Residual=1, States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        np_observations: DevicePointer,
        nq_observations: DevicePointer,
        num_factors: int,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class InformationFactorBatch(FactorBatch):
    """Wraps a factor with per-factor square-root information matrices (Python dynamic wrapper)."""

    def __init__(
        self,
        cublas_handle: CublasHandle,
        inner_factor: FactorBatch,
        sqrt_information_matrices: DevicePointer,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

class WeightedFactorBatch(FactorBatch):
    """Wraps a factor with uniform or per-factor scalar weights (Python dynamic wrapper)."""

    def __init__(
        self,
        inner_factor: FactorBatch,
        *,
        weight: float | None = None,
        weights: DevicePointer | None = None,
    ) -> None: ...
    @property
    def num_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_block_sizes(self) -> list[int]: ...

# ===================================================================
# Loss functions
# ===================================================================

class LossFunctionBatch:
    """Abstract base class for batched robust loss functions."""

    ...

class TrivialLossFunctionBatch(LossFunctionBatch):
    """Identity loss: rho(s)=s. Equivalent to standard least-squares."""

    def __init__(self) -> None: ...

class HuberLossFunctionBatch(LossFunctionBatch):
    """Huber loss: quadratic for small residuals, linear for large."""

    def __init__(self, delta: float) -> None: ...

class CauchyLossFunctionBatch(LossFunctionBatch):
    """Cauchy (Lorentzian) robust loss function."""

    def __init__(self, b: float, c: float) -> None: ...

class ArctanLossFunctionBatch(LossFunctionBatch):
    """Arctan robust loss function."""

    def __init__(self, a: float, b: float) -> None: ...

class SoftLOneLossFunctionBatch(LossFunctionBatch):
    """Soft L1 robust loss function."""

    def __init__(self, b: float, c: float) -> None: ...

class TolerantLossFunctionBatch(LossFunctionBatch):
    """Tolerant robust loss function."""

    def __init__(self, a: float, b: float) -> None: ...

class TukeyLossFunctionBatch(LossFunctionBatch):
    """Tukey's biweight robust loss function."""

    def __init__(self, a: float) -> None: ...

class ScaledLossFunctionBatch(LossFunctionBatch):
    """Scales another loss function by a positive scalar: rho(s) = a * f(s)."""

    def __init__(self, loss_function: LossFunctionBatch, a: float) -> None: ...

# ===================================================================
# Problem
# ===================================================================

class Problem:
    """Defines a nonlinear least-squares problem from state and factor batches."""

    def __init__(self) -> None: ...
    def add_state_batch(self, state_batch: StateBatch) -> None:
        """Register a state batch with the problem."""
        ...
    @overload
    def add_factor_batch(
        self,
        factor_batch: FactorBatch,
        state_pointers: Sequence[int],
    ) -> None:
        """Add a factor batch with its state pointer connectivity."""
        ...
    @overload
    def add_factor_batch(
        self,
        factor_batch: FactorBatch,
        loss_function: LossFunctionBatch,
        state_pointers: Sequence[int],
    ) -> None:
        """Add a factor batch with a loss function and state pointer connectivity."""
        ...
    def check_consistency(self) -> bool:
        """Validate that all state batches and factor batches are consistent."""
        ...

# ===================================================================
# Minimizers
# ===================================================================

class GaussNewtonMinimizer:
    """Gauss-Newton minimizer for nonlinear least-squares problems."""

    def __init__(self, options: MinimizerOptions = ...) -> None: ...
    def minimize(self, stream: CudaStream, problem: Problem) -> MinimizerSummary:
        """Run the Gauss-Newton optimizer. Returns a MinimizerSummary."""
        ...

class LevenbergMarquardtMinimizer(GaussNewtonMinimizer):
    """Levenberg-Marquardt minimizer (damped Gauss-Newton)."""

    def __init__(self, options: LevenbergMarquardtMinimizerOptions = ...) -> None: ...
    def minimize(self, stream: CudaStream, problem: Problem) -> MinimizerSummary:
        """Run the Levenberg-Marquardt optimizer. Returns a MinimizerSummary."""
        ...
