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

import numpy
from typing import Any, Optional, Sequence, overload

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

# ===================================================================
# Logging
# ===================================================================

class Verbosity(enum.Enum):
    Silent = ...
    Error = ...
    Warning = ...
    Message = ...
    Debug = ...

def set_log_verbosity(verbosity: Verbosity) -> None:
    """Set the library log level (default Silent); Message logs every minimizer iteration."""
    ...

# ===================================================================
# Enumerations
# ===================================================================

class SparseLinearSolverType(enum.Enum):
    cuDSS = ...
    DenseLDLT = ...
    DenseCholesky = ...
    DenseQR = ...
    BlockSparsePCG = ...
    BlockTridiagonal = ...

class ColumnScaling(enum.Enum):
    """Diagonal scaling mode for the GN/LM normal equations."""

    none = ...
    hessian_diagonal = ...

class JacobianMode(enum.Enum):
    """Selects how a factor batch's Jacobian is obtained: the factor's own
    analytic Evaluate() output, or finite differences on the manifold
    tangent space of each referenced state."""

    analytic = ...
    numeric = ...

class NumericDiffMethod(enum.Enum):
    """Finite-difference scheme used when a factor batch resolves to
    JacobianMode.numeric."""

    forward = ...
    central = ...

# ===================================================================
# Options and summary
# ===================================================================

class NumericDiffOptions:
    """Tuning knobs for numeric (finite-difference) Jacobian computation."""

    method: NumericDiffMethod
    relative_step_size: float

    def __init__(self) -> None: ...

class MinimizerOptions:
    """Options for Gauss-Newton and Levenberg-Marquardt minimizers."""

    max_num_iterations: int
    state_tolerance: float
    cost_tolerance: float
    max_consecutive_rejected_steps: int
    max_bound_refinements: int
    max_line_search_steps: int
    sparse_linear_solver_type: SparseLinearSolverType
    column_scaling: ColumnScaling
    jacobian_mode: JacobianMode
    numeric_diff_options: NumericDiffOptions
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
    relative_reduction_tolerance: float
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
    """Abstract base class for batched states on a manifold.

    Constructed with a ``capacity`` (states the buffer holds) and 0 active
    states: call :meth:`set_num_active_states` before the first solve.
    """

    @property
    def capacity(self) -> int:
        """States the state buffer holds (the constructor's capacity)."""
        ...
    @property
    def const_capacity(self) -> int:
        """Entries the constant-id buffer holds (the constructor's const_capacity)."""
        ...
    @property
    def num_const_states(self) -> int:
        """Active constant-id count: the first ``num_const_states`` entries of the
        constant-id buffer are held constant (0 until set_num_active_states)."""
        ...
    def set_num_active_states(self, num_active_states: int, num_const_states: int = 0) -> None:
        """Set the active state count (the first ``num_active_states`` states of
        the buffer) and the active constant-id count (``num_const_states``).
        Host-only; takes effect at the next minimize(). Raises ValueError above
        the capacity."""
        ...

class VectorStateBatch1(StateBatch):
    """Euclidean 1-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch2(StateBatch):
    """Euclidean 2-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch3(StateBatch):
    """Euclidean 3-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch4(StateBatch):
    """Euclidean 4-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch12(StateBatch):
    """Euclidean 12-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class VectorStateBatch6(StateBatch):
    """Euclidean 6-D vector state batch."""

    @overload
    def __init__(self, data: DevicePointer, capacity: int) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    def set_bounds(self, lower: Optional[DevicePointer], upper: Optional[DevicePointer]) -> None:
        """Box bounds lower <= x <= upper per component (capacity * dim floats;
        ±inf: unbounded), enforced by projection in GN/LM. None, None removes them."""
        ...
    @property
    def has_bounds(self) -> bool: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SE3StateBatch(StateBatch):
    """SE(3) state batch. Ambient=16 (4x4 matrix), Tangent=6."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SO3StateBatch(StateBatch):
    """SO(3) state batch. Ambient=9 (3x3 matrix), Tangent=3."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SO2StateBatch(StateBatch):
    """SO(2) state batch. Ambient=4 (2x2 matrix), Tangent=1."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SE2StateBatch(StateBatch):
    """SE(2) state batch. Ambient=9 (3x3 matrix), Tangent=3."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class Similarity2StateBatch(StateBatch):
    """2D similarity state batch. Ambient=9, Tangent=4."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class Similarity3StateBatch(StateBatch):
    """3D similarity state batch. Ambient=16, Tangent=7."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

class SL4StateBatch(StateBatch):
    """SL(4) state batch. Ambient=16 (4x4 matrix), Tangent=15."""

    @overload
    def __init__(
        self, data: DevicePointer, capacity: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
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
        capacity: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: DevicePointer,
        ambient_size: int,
        tangent_size: int,
        capacity: int,
        const_state_ids: DevicePointer,
        const_capacity: int,
    ) -> None: ...
    def plus(
        self,
        x_ptr: int,
        delta_ptr: int,
        x_plus_delta_ptr: int,
        stream_handle: int,
        num_replicas: int,
    ) -> None:
        """Apply manifold retraction to num_replicas contiguous copies. Override in subclasses."""
        ...
    def state_device_ptr(self, index: int) -> int: ...
    @property
    def num_active_states(self) -> int: ...
    @property
    def tangent_size(self) -> int: ...
    @property
    def ambient_size(self) -> int: ...

# ===================================================================
# Factor batches
# ===================================================================

class FactorBatch:
    """Abstract base class for batched factors.

    Constructed with a ``capacity`` (factors the measurement buffers hold) and
    0 active factors: call :meth:`set_num_active_factors` before the first solve.
    """

    @property
    def capacity(self) -> int:
        """Factors the measurement buffers hold (the constructor's capacity)."""
        ...
    def set_num_active_factors(self, num_active_factors: int) -> None:
        """Set the active factor count (the first ``num_active_factors`` measurements are
        used). Host-only; takes effect at the next minimize(). Raises ValueError
        above the capacity."""
        ...

class CustomFactorBatch(FactorBatch):
    """Base class for user-defined factors. Override ``evaluate()`` in Python."""

    def __init__(
        self,
        residual_size: int,
        state_sizes: Sequence[int],
        capacity: int,
    ) -> None: ...
    def evaluate(
        self,
        residuals_ptr: int,
        jacobians_ptr: int,
        state_pointers_ptr: int,
        stream_handle: int,
        factor_ids_ptr: int,
        num_factor_ids: int,
    ) -> bool:
        """Compute residuals and Jacobians of num_factor_ids items. Override in subclasses."""
        ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ReprojectionFactorBatch(FactorBatch):
    """Batched 2D reprojection factor. Residual=2, States=[SE3(6), Point(3)].

    The pose state is world_from_rig (the rig's pose in the world); the camera is the
    rig (identity camera_from_rig): P_cam = pose^-1 * P_world.
    Observations must be in normalized image coordinates (K^-1 applied).
    """

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PnPFactorBatch(FactorBatch):
    """PnP reprojection: fixed 3D points (constructor), pose-only Jacobian.

    Residual=2, single SE3 state per factor (world_from_rig). Observations normalized (K^-1).
    """

    @overload
    def __init__(
        self,
        observations: DevicePointer,
        points_world: DevicePointer,
        capacity: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    @overload
    def __init__(
        self,
        observations: DevicePointer,
        poses_camera_from_rig: DevicePointer,
        points_world: DevicePointer,
        capacity: int,
        z_threshold: float = 1e-3,
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3BetweenFactorBatch(FactorBatch):
    """Batched SE(3) between factor. Residual=6, States=[SE3(6), SE3(6)]."""

    def __init__(
        self,
        deltas: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3PriorFactorBatch(FactorBatch):
    """Batched SE(3) prior factor. Residual=6, States=[SE3(6)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SO3PriorFactorBatch(FactorBatch):
    """Batched SO(3) prior factor. Residual=3, States=[SO3(3)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE2PriorFactorBatch(FactorBatch):
    """Batched SE(2) prior factor. Residual=3, States=[SE2(3)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class Similarity2PriorFactorBatch(FactorBatch):
    """Batched Sim(2) prior factor. Residual=4, States=[Similarity2(4)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class Similarity3PriorFactorBatch(FactorBatch):
    """Batched Sim(3) prior factor. Residual=7, States=[Similarity3(7)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SO2PriorFactorBatch(FactorBatch):
    """Batched SO(2) prior factor. Residual=1, States=[SO2(1)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch1(FactorBatch):
    """Prior factor for 1-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch2(FactorBatch):
    """Prior factor for 2-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch3(FactorBatch):
    """Prior factor for 3-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch4(FactorBatch):
    """Prior factor for 4-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch12(FactorBatch):
    """Prior factor for 12-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PriorVectorFactorBatch6(FactorBatch):
    """Prior factor for 6-D vectors: residual = state - observation."""

    def __init__(self, observations: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE2BetweenFactorBatch(FactorBatch):
    """Batched SE(2) between factor. Residual=3, States=[SE2(3), SE2(3)]."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SO2BetweenFactorBatch(FactorBatch):
    """Batched SO(2) between factor. Residual=1, States=[SO2(1), SO2(1)]."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SO3BetweenFactorBatch(FactorBatch):
    """Batched SO(3) between factor. Residual=3, States=[SO3(3), SO3(3)]."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class Similarity2BetweenFactorBatch(FactorBatch):
    """Batched Sim(2) between factor. Residual=4, States=[Sim2(4), Sim2(4)]."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class Similarity3BetweenFactorBatch(FactorBatch):
    """Batched Sim(3) between factor. Residual=7, States=[Sim3(7), Sim3(7)]."""

    def __init__(
        self,
        deltas: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SL4PriorFactorBatch(FactorBatch):
    """Batched SL(4) prior factor. Residual=15, States=[SL4(15)]."""

    def __init__(
        self,
        observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SL4BetweenFactorBatch(FactorBatch):
    """Batched SL(4) between factor. Residual=15, States=[SL4(15), SL4(15)]."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch1(FactorBatch):
    """Between factor for 1-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch2(FactorBatch):
    """Between factor for 2-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch3(FactorBatch):
    """Between factor for 3-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch4(FactorBatch):
    """Between factor for 4-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch12(FactorBatch):
    """Between factor for 12-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class VectorBetweenFactorBatch6(FactorBatch):
    """Between factor for 6-D vectors: residual = left - right - delta."""

    def __init__(self, deltas: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PointToPointFactorBatch(FactorBatch):
    """Batched point-to-point factor: residual = p - T*q. Residual=3, States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class PointToPlaneFactorBatch(FactorBatch):
    """Point-to-plane factor: residual = Nq^T*(p - T*q), Nq in the target frame. States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        nq_observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SymmetricPointToPlaneFactorBatch(FactorBatch):
    """Batched symmetric point-to-plane factor. Residual=1, States=[SE3(6)]."""

    def __init__(
        self,
        p_observations: DevicePointer,
        q_observations: DevicePointer,
        np_observations: DevicePointer,
        nq_observations: DevicePointer,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class InformationFactorBatch(FactorBatch):
    """Wraps a factor with per-factor square-root information matrices (Python dynamic wrapper)."""

    def __init__(
        self,
        inner_factor: FactorBatch,
        sqrt_information_matrices: DevicePointer,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class WeightedFactorBatch(FactorBatch):
    """Wraps a factor with uniform or per-factor scalar weights (Python dynamic wrapper)."""

    @overload
    def __init__(self, inner_factor: FactorBatch, weight: float) -> None: ...
    @overload
    def __init__(self, inner_factor: FactorBatch, *, weights: DevicePointer) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

# ===================================================================
# Constraint factor batches (augmented Lagrangian)
# ===================================================================

class ConstraintKind(enum.Enum):
    """Kind of a constraint row."""

    Equality = 0
    """c(x) = 0"""
    Inequality = 1
    """c(x) <= 0"""

class ConstraintFactorBatchBase(FactorBatch):
    """A factor batch whose rows are constraints (augmented Lagrangian residuals)."""

    @property
    def kind(self) -> ConstraintKind: ...
    @property
    def scale(self) -> float: ...
    @property
    def multipliers_ptr(self) -> int:
        """Device pointer to capacity * residuals_size float multipliers."""
        ...
    @property
    def penalties_ptr(self) -> int:
        """Device pointer to capacity float penalties, one per factor."""
        ...
    def reset_multipliers(self, stream: CudaStream) -> None: ...
    def set_penalty(self, penalty: float, stream: CudaStream) -> None: ...

class ConstraintFactorBatch(ConstraintFactorBatchBase):
    """Turns any factor batch into constraint rows: scale * r(x) = 0 or <= 0."""

    def __init__(
        self, inner_factor: FactorBatch, kind: ConstraintKind, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch1(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 1-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class HalfspaceFactorBatch1(FactorBatch):
    """Signed halfspace value a^T x - b of a 1-D vector state; wrap in ConstraintFactorBatch."""

    def __init__(self, normals: DevicePointer, offsets: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch2(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 2-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class HalfspaceFactorBatch2(FactorBatch):
    """Signed halfspace value a^T x - b of a 2-D vector state; wrap in ConstraintFactorBatch."""

    def __init__(self, normals: DevicePointer, offsets: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch3(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 3-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch4(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 4-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch12(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 12-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class HalfspaceFactorBatch3(FactorBatch):
    """Signed halfspace value a^T x - b of a 3-D vector state; wrap in ConstraintFactorBatch."""

    def __init__(self, normals: DevicePointer, offsets: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class BoundFactorBatch6(ConstraintFactorBatchBase):
    """Box constraints lower <= x <= upper on a 6-D vector state (an inequality constraint batch)."""

    def __init__(
        self, lower: DevicePointer, upper: DevicePointer, capacity: int, scale: float = 1.0
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class HalfspaceFactorBatch6(FactorBatch):
    """Signed halfspace value a^T x - b of a 6-D vector state; wrap in ConstraintFactorBatch."""

    def __init__(self, normals: DevicePointer, offsets: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

# ===================================================================
# Dynamics factor batches
# ===================================================================

class SE2DifferentialDriveFactorBatch(FactorBatch):
    """Carter dynamics: states (SE2 pose, wheel speeds, next SE2 pose); analytic Jacobians."""

    def __init__(
        self, time_steps: DevicePointer, wheel_radius: float, track_width: float, capacity: int
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE2KinematicBicycleFactorBatch(FactorBatch):
    """Car (kinematic bicycle, Euler): states (pose, (v, delta), (a, delta rate), pose, (v, delta))."""

    def __init__(self, time_steps: DevicePointer, wheelbase: float, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3DifferentialDriveFactorBatch(FactorBatch):
    """Carter on terrain: states (SE3 pose, wheel speeds, next SE3 pose); roll/pitch free."""

    def __init__(
        self, time_steps: DevicePointer, wheel_radius: float, track_width: float, capacity: int
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3KinematicBicycleFactorBatch(FactorBatch):
    """Car on terrain: states (SE3 pose, (v, delta), (a, delta rate), SE3 pose, (v, delta))."""

    def __init__(self, time_steps: DevicePointer, wheelbase: float, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE2KinematicsFactorBatch(FactorBatch):
    """SE(2) kinematics with the body twist as control."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SO3KinematicsFactorBatch(FactorBatch):
    """SO(3) kinematics with the body rate as control."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3KinematicsFactorBatch(FactorBatch):
    """SE(3) kinematics with the body twist [omega, v] as control."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocitySE3FactorBatch(FactorBatch):
    """Constant-velocity motion prior: (pose_k, pose_k1, v_k, v_k1). SE3 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocityInformationSE3FactorBatch(FactorBatch):
    """ConstantVelocitySE3FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationSE3FactorBatch(FactorBatch):
    """Constant-acceleration motion prior: (pose_k, pose_k1, v_k, v_k1, a_k, a_k1). SE3 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationInformationSE3FactorBatch(FactorBatch):
    """ConstantAccelerationSE3FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocitySO3FactorBatch(FactorBatch):
    """Constant-velocity motion prior: (pose_k, pose_k1, v_k, v_k1). SO3 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocityInformationSO3FactorBatch(FactorBatch):
    """ConstantVelocitySO3FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationSO3FactorBatch(FactorBatch):
    """Constant-acceleration motion prior: (pose_k, pose_k1, v_k, v_k1, a_k, a_k1). SO3 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationInformationSO3FactorBatch(FactorBatch):
    """ConstantAccelerationSO3FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocitySE2FactorBatch(FactorBatch):
    """Constant-velocity motion prior: (pose_k, pose_k1, v_k, v_k1). SE2 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocityInformationSE2FactorBatch(FactorBatch):
    """ConstantVelocitySE2FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationSE2FactorBatch(FactorBatch):
    """Constant-acceleration motion prior: (pose_k, pose_k1, v_k, v_k1, a_k, a_k1). SE2 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationInformationSE2FactorBatch(FactorBatch):
    """ConstantAccelerationSE2FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocitySO2FactorBatch(FactorBatch):
    """Constant-velocity motion prior: (pose_k, pose_k1, v_k, v_k1). SO2 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantVelocityInformationSO2FactorBatch(FactorBatch):
    """ConstantVelocitySO2FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationSO2FactorBatch(FactorBatch):
    """Constant-acceleration motion prior: (pose_k, pose_k1, v_k, v_k1, a_k, a_k1). SO2 poses."""

    def __init__(self, time_steps: DevicePointer, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ConstantAccelerationInformationSO2FactorBatch(FactorBatch):
    """ConstantAccelerationSO2FactorBatch weighted by the closed-form process-noise information Q(dt)^-1."""

    def __init__(
        self, stream: CudaStream, time_steps: DevicePointer, qc_diag: DevicePointer, capacity: int
    ) -> None: ...
    def update(self, stream: CudaStream, num: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class QuadrotorParameters:
    """Quadrotor in X configuration (see the C++ documentation for the rotor layout)."""

    mass: float
    inertia: list[float]
    arm_length: float
    torque_coefficient: float
    linear_drag: float
    gravity: float

    def __init__(self) -> None: ...

class QuadrotorFactorBatch(FactorBatch):
    """Quadrotor dynamics: (pose, velocity, rates, thrusts, next pose, velocity, rates)."""

    def __init__(
        self, time_steps: DevicePointer, parameters: QuadrotorParameters, capacity: int
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class QuadrupedParameters:
    """Single-rigid-body quadruped: mass, inertia, gravity."""

    mass: float
    inertia: list[float]
    gravity: float

    def __init__(self) -> None: ...

class QuadrupedFactorBatch(FactorBatch):
    """Quadruped base dynamics: (pose, velocity, rates, foot forces, next pose, velocity, rates)."""

    def __init__(
        self,
        time_steps: DevicePointer,
        contacts: DevicePointer,
        foot_positions: DevicePointer,
        parameters: QuadrupedParameters,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class ImuParameters:
    """Sensor model of ImuFactorBatch (gravity, noise densities, IMU-body extrinsic)."""

    gravity: list[float]
    gyro_noise_density: float
    accel_noise_density: float
    integration_noise_density: float
    gyro_bias_random_walk: float
    accel_bias_random_walk: float
    body_from_imu: list[float]

    def __init__(self) -> None: ...

class ImuFactorBatch(FactorBatch):
    """IMU factor between keyframes (T_a, v_a, b_a, T_b, v_b, b_b), T = world_from_rig."""

    def __init__(
        self,
        imu_samples: DevicePointer,
        sample_offsets: DevicePointer,
        num_samples: int,
        parameters: ImuParameters,
        capacity: int,
    ) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE2DiskClearanceFactorBatch(FactorBatch):
    """Clearance (radius + margin) - |p - center| of an SE(2) pose to a disk (wrap as inequality)."""

    def __init__(self, obstacles: DevicePointer, margin: float, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

class SE3SphereClearanceFactorBatch(FactorBatch):
    """Clearance (radius + margin) - |p - center| of an SE(3) pose to a sphere (wrap as inequality)."""

    def __init__(self, obstacles: DevicePointer, margin: float, capacity: int) -> None: ...
    @property
    def num_active_factors(self) -> int: ...
    @property
    def residuals_size(self) -> int: ...
    def state_sizes(self) -> list[int]: ...

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
        jacobian_mode_override: JacobianMode | None = None,
    ) -> None:
        """Add a factor batch with its state pointer connectivity.

        jacobian_mode_override, when set, forces this factor batch to always
        use the given JacobianMode regardless of the minimizer's
        MinimizerOptions.jacobian_mode default.
        """
        ...
    @overload
    def add_factor_batch(
        self,
        factor_batch: FactorBatch,
        loss_function: LossFunctionBatch,
        state_pointers: Sequence[int],
        jacobian_mode_override: JacobianMode | None = None,
    ) -> None:
        """Add a factor batch with a loss function and state pointer connectivity.

        jacobian_mode_override, when set, forces this factor batch to always
        use the given JacobianMode regardless of the minimizer's
        MinimizerOptions.jacobian_mode default.
        """
        ...
    @overload
    def add_factor_batch(
        self,
        factor_batch: FactorBatch,
        *,
        state_pointer_table: DevicePointer,
        loss_function: LossFunctionBatch | None = None,
        jacobian_mode_override: JacobianMode | None = None,
    ) -> None:
        """Add a factor batch whose connectivity is a device table of state pointers.

        ``state_pointer_table`` is a uint64 device array of ``capacity * B``
        entries; entry ``f * B + b`` points at the state factor ``f`` reads in
        slot ``b``. Bound once; rewrite its contents between solves.
        """
        ...
    @overload
    def add_factor_batch(
        self,
        factor_batch: FactorBatch,
        slot_state_batches: Sequence[StateBatch],
        state_indices: DevicePointer,
        loss_function: LossFunctionBatch | None = None,
        jacobian_mode_override: JacobianMode | None = None,
    ) -> None:
        """Add a factor batch whose connectivity is a device table of state indices.

        ``state_indices`` is an int32 device array of ``capacity * B`` entries:
        factor ``f`` reads state ``state_indices[f * B + b]`` of
        ``slot_state_batches[b]``. Bound once; rewrite its contents between solves.
        """
        ...
    def set_state_pointers(self, residual_batch_index: int, state_pointers: Sequence[int]) -> None:
        """Replace the host-list connectivity of a residual batch
        (``num_active_factors * B`` pointers)."""
        ...
    def validate(self, stream: CudaStream) -> bool:
        """GPU check of every active connection (for connectivity rewritten on the device).

        Synchronizes the stream; logs the first failure.
        """
        ...
    def check_consistency(self) -> bool:
        """Validate that all state batches and factor batches are consistent."""
        ...
    def set_problem_partition(
        self, num_problems: int, state_problem_ids: list[DevicePointer]
    ) -> None:
        """Declare the problem a batch of independent subproblems: state s of state
        batch b belongs to subproblem state_problem_ids[b][s] (device int32 arrays,
        one per state batch). Every factor must stay within one subproblem. The
        minimizers then accept, damp and stop each subproblem on its own.
        num_problems <= 1 clears the partition."""
        ...
    def set_state_stages(self, state_stages: list[DevicePointer]) -> None:
        """Declare a time ordering of the states for the BlockTridiagonal linear
        solver: state s of state batch b belongs to stage state_stages[b][s]
        (device int32 arrays, one per state batch). Factors may only connect
        states of the same or adjacent stages within a subproblem. An empty list
        clears the stages."""
        ...
    @property
    def num_problems(self) -> int: ...

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

# ---------------------------------------------------------------------------
# Constrained minimization (augmented Lagrangian)
# ---------------------------------------------------------------------------

class AugmentedLagrangianMinimizerOptions:
    """Options of the augmented Lagrangian outer loop."""

    constraint_tolerance: float
    max_outer_iterations: int
    inner_iterations: int
    final_inner_iterations: int
    inner_line_search_steps: int
    initial_penalty: float
    penalty_increase: float
    max_penalty: float
    violation_decrease: float
    warm_start: bool
    reuse_structure: bool
    real_time: bool
    use_cuda_graph: bool

    def __init__(self) -> None: ...

class AugmentedLagrangianMinimizerStatus(enum.Enum):
    Converged = 0
    MaxOuterIterations = 1
    MaxPenalty = 2

class AugmentedLagrangianMinimizerSummary:
    """Summary of a constrained minimization run."""

    @property
    def status(self) -> AugmentedLagrangianMinimizerStatus: ...
    @property
    def outer_iterations(self) -> int: ...
    @property
    def inner_iterations(self) -> int: ...
    @property
    def max_violation(self) -> float: ...
    @property
    def initial_cost(self) -> float: ...
    @property
    def final_cost(self) -> float: ...
    @property
    def num_problems(self) -> int: ...
    @property
    def num_converged(self) -> int: ...
    @property
    def num_max_penalty(self) -> int: ...
    def __repr__(self) -> str: ...

class AugmentedLagrangianMinimizer:
    """Augmented Lagrangian solver for problems with constraint factor batches."""

    def __init__(
        self, minimizer: GaussNewtonMinimizer, options: AugmentedLagrangianMinimizerOptions = ...
    ) -> None: ...
    def minimize(self, stream: CudaStream, problem: Problem) -> AugmentedLagrangianMinimizerSummary:
        """Minimize the objective subject to the constraint batches."""
        ...
    @property
    def options(self) -> AugmentedLagrangianMinimizerOptions:
        """Options of the following calls (a copy; assign a modified one to change them)."""
        ...
    @options.setter
    def options(self, value: AugmentedLagrangianMinimizerOptions) -> None: ...
    @property
    def uses_cuda_graph(self) -> bool:
        """Whether the last call captured or replayed a CUDA graph."""
        ...

# ---------------------------------------------------------------------------
# RANSAC minimizers
# ---------------------------------------------------------------------------

class RansacRole(enum.Enum):
    """Role of a residual batch in RANSAC."""

    sampled = ...
    """Data factors that may be outliers: sampled and classified."""
    always_on = ...
    """Trusted factors (priors): in every solve, never classified."""

class RansacScoring(enum.Enum):
    """Hypothesis scoring rule."""

    msac = ...
    """Sum over sampled factors of min(|r|^2, tau^2)."""
    inlier_count = ...
    """Number of inliers (ties broken by MSAC)."""

class RansacLinearSolverType(enum.Enum):
    """Dense per-hypothesis solver."""

    cholesky = ...
    ldlt = ...

class RansacFactorBatchOptions:
    """RANSAC configuration of one residual batch."""

    def __init__(self, role: RansacRole = ..., inlier_threshold: float = 1.0) -> None: ...
    role: RansacRole
    inlier_threshold: float
    """Inlier iff |r| <= inlier_threshold (raw residual norm)."""

class RansacMinimizerOptions:
    """Options shared by both RANSAC minimizers."""

    def __init__(self) -> None: ...
    hypotheses_per_round: int
    max_rounds: int
    sample_size: int
    """Factors per minimal sample; 0 = ceil(D / m_min)."""
    confidence: float
    early_stop_inlier_ratio: float
    seed: int
    factor_batches: list[RansacFactorBatchOptions]
    """One entry per residual batch, in the order the batches were added.
    Assign a whole list (appending to the returned copy has no effect).
    Empty = every batch sampled with default_inlier_threshold."""
    default_inlier_threshold: float
    scoring: RansacScoring
    score_always_on: bool
    require_informative_inliers: bool
    scoring_memory_budget_bytes: int
    scoring_subset_size: int
    scoring_finalists: int
    hypothesis_iterations: int
    final_iterations: int
    state_tolerance: float
    cost_tolerance: float
    linear_solver: RansacLinearSolverType

class RansacLevenbergMarquardtMinimizerOptions:
    """Options of RansacLevenbergMarquardtMinimizer."""

    def __init__(self) -> None: ...
    base_options: RansacMinimizerOptions
    initial_lambda: float
    lambda_upscale: float
    lambda_downscale: float
    lambda_max: float
    lambda_min: float
    step_accept_threshold: float
    lambda_downscale_threshold: float

class RansacSummary(MinimizerSummary):
    """Result of a RANSAC minimization."""

    @property
    def num_rounds(self) -> int: ...
    @property
    def num_hypotheses(self) -> int: ...
    @property
    def num_valid_hypotheses(self) -> int: ...
    @property
    def num_inliers(self) -> int: ...
    @property
    def inlier_ratio(self) -> float: ...
    @property
    def best_score(self) -> float: ...
    @property
    def refinement_reverted(self) -> bool: ...

class RansacGaussNewtonMinimizer:
    """RANSAC with Gauss-Newton hypotheses and refinement (free tangent dim <= 64)."""

    def __init__(self, options: RansacMinimizerOptions = ...) -> None: ...
    def minimize(self, stream: CudaStream, problem: Problem) -> RansacSummary:
        """Run RANSAC; the estimate is written into the problem's state batches."""
        ...
    def inlier_mask(self, residual_batch_index: int) -> numpy.ndarray:
        """Inlier mask (uint8, 1 = inlier) of a sampled residual batch of the
        problem of the last minimize(). Raises RuntimeError for an out-of-range
        index, an always_on batch, or before any run."""
        ...

class RansacLevenbergMarquardtMinimizer(RansacGaussNewtonMinimizer):
    """RANSAC with Levenberg-Marquardt hypotheses and refinement."""

    def __init__(self, options: RansacLevenbergMarquardtMinimizerOptions = ...) -> None: ...
