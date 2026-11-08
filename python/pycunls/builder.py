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

"""Problems from arrays: :class:`ProblemBuilder` and the one-call :func:`solve`.

The builder sets up a :class:`pycunls.Problem` from NumPy or CuPy arrays: states from their
initial values, factors from their measurements and an index array saying which states each
factor connects. It owns the device arrays, the capacities, active counts, constant ids and the
connectivity table, checks shapes against the catalog, runs :func:`pycunls.diagnose` before the
first solve, and returns the solved states as NumPy arrays. The underlying objects stay reachable
(``builder.problem``, ``handle.batch``) to move on to the full API.

Example (a pose graph)::

    pb = pycunls.ProblemBuilder()
    poses = pb.add_states(pycunls.SE3StateBatch, initial=T0, constant=[0])     # T0: (N, 4, 4)
    pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges))  # edges (M, 2)
    result = pb.solve()
    result.converged, result.message
    T = result[poses]                                                           # (N, 4, 4)

Or in one call, when there is one state batch and one factor batch::

    result = pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=T0, connect=edges,
                           constant=[0])
"""

import dataclasses
import warnings
from typing import Any, List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from pycunls import _pycunls_core as _core
from pycunls.catalog import factor_info, state_info
from pycunls.diagnostics import diagnose, evaluate

_MINIMIZER_OPTIONS = ("max_num_iterations", "state_tolerance", "cost_tolerance",
                      "max_consecutive_rejected_steps", "max_line_search_steps",
                      "sparse_linear_solver_type", "linear_system", "column_scaling",
                      "jacobian_mode", "reuse_structure", "disable_safety_checks")
_LM_OPTIONS = ("initial_lambda", "relative_reduction_tolerance", "lambda_upscale",
               "lambda_downscale", "lambda_max", "lambda_min", "min_diagonal",
               "step_accept_threshold", "lambda_downscale_threshold")
_AL_OPTIONS = ("constraint_tolerance", "max_outer_iterations", "inner_iterations",
               "final_inner_iterations", "inner_line_search_steps", "initial_penalty",
               "penalty_increase", "max_penalty", "violation_decrease", "max_bound_refinements")


def _resolve(cls_or_name, base, kind: str):
    if isinstance(cls_or_name, str):
        cls = getattr(_core, cls_or_name, None)
        if cls is None:
            raise ValueError(f"unknown {kind} class {cls_or_name!r}; see pycunls.catalog('{kind}')")
        return cls
    if not (isinstance(cls_or_name, type) and issubclass(cls_or_name, base)):
        raise TypeError(f"expected a {kind} batch class (e.g. pycunls.SE3StateBatch) or its name; "
                        f"got {cls_or_name!r}")
    return cls_or_name


def _float32(a, name: str) -> cp.ndarray:
    try:
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{name} must be array-like (NumPy, CuPy, list); got {type(a)}") from exc


@dataclasses.dataclass(eq=False)
class StateHandle:
    """A state batch added with :meth:`ProblemBuilder.add_states`.

    Attributes:
        batch: The :class:`pycunls.StateBatch`.
        data: Its device array (CuPy, float32), shaped like the initial values; the solve updates
            it in place.
        index: Position among the builder's state batches.
    """
    batch: Any
    data: cp.ndarray
    index: int
    constant_ids: Optional[cp.ndarray] = None
    bounds: Optional[Tuple[cp.ndarray, cp.ndarray]] = None

    @property
    def count(self) -> int:
        """Number of states."""
        return self.data.shape[0]

    @property
    def values(self) -> np.ndarray:
        """The current states as a NumPy array (same shape as the initial values)."""
        return cp.asnumpy(self.data)

    def set(self, values) -> None:
        """Overwrites the states (same shape), e.g. a new initial guess before solving again."""
        new = _float32(values, "values")
        if new.shape != self.data.shape:
            raise ValueError(f"values must have shape {self.data.shape}, got {new.shape}")
        self.data[...] = new


@dataclasses.dataclass(eq=False)
class FactorHandle:
    """A factor batch added with :meth:`ProblemBuilder.add_factors`.

    Attributes:
        batch: The :class:`pycunls.FactorBatch`.
        measurements: Its device measurement arrays (CuPy, float32).
        connectivity: The (num_factors, num_slots) int32 state indices on the device.
        index: Residual batch index in the problem (as in :func:`pycunls.evaluate`).
    """
    batch: Any
    measurements: List[cp.ndarray]
    connectivity: cp.ndarray
    index: int
    loss: Any = None

    def set_measurements(self, *arrays) -> None:
        """Overwrites the measurement arrays (same shapes) before solving again."""
        if len(arrays) != len(self.measurements):
            raise ValueError(f"expected {len(self.measurements)} arrays, got {len(arrays)}")
        for old, new in zip(self.measurements, arrays):
            new = _float32(new, "measurements")
            if new.shape != old.shape:
                raise ValueError(f"measurement shape {new.shape} differs from {old.shape}")
            old[...] = new


@dataclasses.dataclass
class SolveResult:
    """Result of :meth:`ProblemBuilder.solve` / :func:`solve`.

    ``result[handle]`` (or ``result.states`` with one state batch) gives the solved states as
    NumPy arrays. ``converged``, ``message`` and ``summary`` come from the solver;
    ``evaluate()`` gives the residuals per factor batch at the solution.
    """
    summary: Any
    converged: bool
    message: str
    _builder: Any

    def __getitem__(self, handle: StateHandle) -> np.ndarray:
        return handle.values

    @property
    def states(self) -> np.ndarray:
        """The solved states of the only state batch (use ``result[handle]`` with several)."""
        handles = self._builder.state_handles
        if len(handles) != 1:
            raise ValueError("several state batches: use result[handle]")
        return handles[0].values

    def evaluate(self):
        """:func:`pycunls.evaluate` at the solution (cost, RMS, worst factor per batch)."""
        return evaluate(self._builder.problem, self._builder.stream)

    def report(self) -> str:
        """The solver summary's report."""
        return self.summary.report()


class ProblemBuilder:
    """Builds and solves a :class:`pycunls.Problem` from arrays (see the module docs).

    Args:
        stream: CUDA stream for every solve (a new one by default).
    """

    def __init__(self, stream: Optional[_core.CudaStream] = None):
        self.stream = stream if stream is not None else _core.CudaStream()
        self.problem = _core.Problem()
        self.state_handles: List[StateHandle] = []
        self.factor_handles: List[FactorHandle] = []
        self._solved_once = False
        self._minimizer = None

    # -- states -----------------------------------------------------------------------------------

    def add_states(self, state_class, initial, constant: Optional[Sequence[int]] = None,
                   bounds: Optional[Tuple[Any, Any]] = None) -> StateHandle:
        """Adds a state batch with its initial values.

        Args:
            state_class: A state batch class or its name, e.g. ``pycunls.SE3StateBatch``.
            initial: Initial values, ``(N, ...)`` with ``ambient_size`` floats per state (SE3:
                ``(N, 4, 4)`` row-major ``world_from_body``; VectorStateBatch3: ``(N, 3)``).
            constant: Indices of states held fixed (e.g. ``[0]`` to fix the gauge).
            bounds: Vector states only: ``(lower, upper)`` per component, scalars or arrays
                broadcastable to ``(N, dim)``; then the solve uses the augmented Lagrangian.

        Returns:
            A :class:`StateHandle` to connect factors to and read results from.
        """
        cls = _resolve(state_class, _core.StateBatch, "state")
        data = _float32(initial, "initial")
        info = state_info(cls.__name__)
        n = data.shape[0] if data.ndim else 0
        if info is not None and data.size != n * info.ambient:
            raise ValueError(f"{cls.__name__} stores {info.ambient} floats per state "
                             f"({info.layout}); initial has shape {data.shape}, i.e. "
                             f"{data.size / max(n, 1):g} per state")
        const_ids = None
        if constant is not None and len(constant):
            const_ids = cp.asarray(np.asarray(constant, dtype=np.int32).reshape(-1))
            if int(const_ids.max()) >= n or int(const_ids.min()) < 0:
                raise ValueError(f"constant ids must be in [0, {n}); got {constant}")
            batch = cls(data, n, const_ids, const_ids.size)
            batch.set_num_active_states(n, const_ids.size)
        else:
            batch = cls(data, n)
            batch.set_num_active_states(n)
        handle = StateHandle(batch, data, len(self.state_handles), const_ids)
        if bounds is not None:
            if not hasattr(batch, "set_bounds"):
                raise TypeError(f"{cls.__name__} has no bounds; only VectorStateBatchN does")
            dim = batch.tangent_size
            lo = cp.ascontiguousarray(cp.broadcast_to(_float32(bounds[0], "lower"), (n, dim)))
            hi = cp.ascontiguousarray(cp.broadcast_to(_float32(bounds[1], "upper"), (n, dim)))
            batch.set_bounds(lo, hi)
            handle.bounds = (lo, hi)
        self.problem.add_state_batch(batch)
        self.state_handles.append(handle)
        return handle

    # -- factors ----------------------------------------------------------------------------------

    def add_factors(self, factor_class, *measurements, connect, loss=None, **kwargs
                    ) -> FactorHandle:
        """Adds a factor batch: its measurements and the states each factor connects.

        Args:
            factor_class: A factor batch class or its name, e.g. ``pycunls.SE3BetweenFactorBatch``
                (``pycunls.describe(...)`` shows its states and measurement layout).
            *measurements: The constructor's measurement arrays in order (one item per factor),
                e.g. ``deltas`` ``(M, 4, 4)``; NumPy or CuPy, converted to float32.
            connect: Which states each factor reads, one entry per state slot of the factor:
                ``(handle, indices)`` with ``indices`` of shape ``(M, num_slots)`` (all slots in
                one state batch, e.g. ``(poses, edges)`` for between factors), or a list of
                ``(handle, indices_m)`` pairs, one per slot (e.g. ``[(cams, cam_ids),
                (points, point_ids)]``).
            loss: Optional robust loss (a :class:`pycunls.LossFunctionBatch`).
            **kwargs: Extra constructor arguments (e.g. ``z_threshold``).

        Returns:
            A :class:`FactorHandle`.
        """
        cls = _resolve(factor_class, _core.FactorBatch, "factor")
        handles, columns = self._connectivity(connect)
        num_factors = columns.shape[0]
        arrays = [_float32(m, f"measurement {i}") for i, m in enumerate(measurements)]
        try:
            batch = cls(*arrays, num_factors, **kwargs)
        except TypeError as exc:
            info = factor_info(cls.__name__)
            expected = (", ".join(info.arrays) if info is not None else "see help()")
            raise TypeError(f"{cls.__name__}({len(arrays)} measurement arrays, capacity="
                            f"{num_factors}) failed: {exc}. Expected measurement arrays: "
                            f"{expected}") from exc
        batch.set_num_active_factors(num_factors)
        sizes = list(batch.state_sizes())
        if len(sizes) != len(handles):
            raise ValueError(f"{cls.__name__} connects {len(sizes)} states per factor but connect "
                             f"gives {len(handles)} slots")
        for slot, (size, handle) in enumerate(zip(sizes, handles)):
            if handle.batch.tangent_size != size:
                expected = factor_info(cls.__name__)
                hint = f" (expects {expected.states[slot]})" if expected and expected.states else ""
                raise ValueError(f"slot {slot} of {cls.__name__} needs a state with tangent size "
                                 f"{size}{hint}; {type(handle.batch).__name__} has "
                                 f"{handle.batch.tangent_size}")
            col = columns[:, slot]
            if col.size and (int(col.min()) < 0 or int(col.max()) >= handle.count):
                raise ValueError(f"connect: slot {slot} indices must be in [0, {handle.count}) "
                                 f"for {type(handle.batch).__name__}; got [{int(col.min())}, "
                                 f"{int(col.max())}]")
        table = cp.ascontiguousarray(columns.reshape(-1))
        self.problem.add_factor_batch(batch, [h.batch for h in handles], table, loss)
        handle = FactorHandle(batch, arrays, columns, len(self.factor_handles), loss)
        self.factor_handles.append(handle)
        return handle

    def _connectivity(self, connect) -> Tuple[List[StateHandle], cp.ndarray]:
        if isinstance(connect, tuple) and len(connect) == 2 and isinstance(connect[0],
                                                                           StateHandle):
            handle, idx = connect
            idx = cp.asarray(idx, dtype=cp.int32)
            if idx.ndim == 1:
                idx = idx[:, None]
            return [handle] * idx.shape[1], cp.ascontiguousarray(idx)
        if isinstance(connect, (list, tuple)) and all(
                isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], StateHandle)
                for c in connect):
            handles = [c[0] for c in connect]
            cols = [cp.asarray(c[1], dtype=cp.int32).reshape(-1) for c in connect]
            if len({c.size for c in cols}) != 1:
                raise ValueError("connect: every slot needs one index per factor (equal lengths)")
            return handles, cp.ascontiguousarray(cp.stack(cols, axis=1))
        raise TypeError("connect must be (handle, indices[M, slots]) or a list of "
                        "(handle, indices[M]) pairs, one per state slot")

    # -- solve -------------------------------------------------------------------------------------

    def diagnose(self):
        """:func:`pycunls.diagnose` on the built problem."""
        return diagnose(self.problem, self.stream)

    def evaluate(self, keep_residuals: bool = False):
        """:func:`pycunls.evaluate` on the built problem at the current states."""
        return evaluate(self.problem, self.stream, keep_residuals)

    def solve(self, method: str = "lm", validate: bool = True, **options) -> SolveResult:
        """Solves the problem; the states are updated in place.

        Args:
            method: ``"lm"`` (Levenberg-Marquardt, default) or ``"gn"`` (Gauss-Newton). With box
                bounds (``add_states(..., bounds=...)``) or constraint factor batches the method
                runs inside the augmented Lagrangian minimizer.
            validate: Run :func:`pycunls.diagnose` before the first solve: errors raise
                ``ValueError`` (with the fix), warnings (e.g. a gauge freedom) warn.
            **options: Solver options by name: ``max_num_iterations``, ``state_tolerance``,
                ``cost_tolerance``, ``sparse_linear_solver_type``, ``linear_system``, ... (the
                fields of :class:`pycunls.MinimizerOptions`), Levenberg-Marquardt fields
                (``initial_lambda``, ...) and augmented Lagrangian fields
                (``constraint_tolerance``, ...). An unknown name raises with the valid ones.

        Returns:
            :class:`SolveResult`.
        """
        if not self.factor_handles:
            raise ValueError("add factors before solving (add_factors)")
        if validate and not self._solved_once:
            issues = self.diagnose()
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                raise ValueError("the problem is not ready to solve:\n" +
                                 "\n".join(str(i) for i in errors))
            for issue in issues:
                warnings.warn(issue.message, stacklevel=2)
        valid = _MINIMIZER_OPTIONS + _LM_OPTIONS + _AL_OPTIONS
        unknown = sorted(set(options) - set(valid))
        if unknown:
            raise ValueError(f"unknown option(s) {unknown}; valid: {', '.join(valid)}")
        base = _core.MinimizerOptions()
        for key in _MINIMIZER_OPTIONS:
            if key in options:
                setattr(base, key, options[key])
        if method == "lm":
            lm = _core.LevenbergMarquardtMinimizerOptions()
            lm.base_options = base
            for key in _LM_OPTIONS:
                if key in options:
                    setattr(lm, key, options[key])
            minimizer = _core.LevenbergMarquardtMinimizer(lm)
        elif method == "gn":
            if any(k in options for k in _LM_OPTIONS):
                raise ValueError("Levenberg-Marquardt options given with method='gn'")
            minimizer = _core.GaussNewtonMinimizer(base)
        else:
            raise ValueError(f"method must be 'lm' or 'gn', got {method!r}")
        constrained = any(h.bounds is not None for h in self.state_handles) or any(
            isinstance(f.batch, _core.ConstraintFactorBatchBase) for f in self.factor_handles)
        self._minimizer = minimizer
        if constrained:
            al = _core.AugmentedLagrangianMinimizerOptions()
            for key in _AL_OPTIONS:
                if key in options:
                    setattr(al, key, options[key])
            summary = _core.AugmentedLagrangianMinimizer(minimizer, al).minimize(self.stream,
                                                                                 self.problem)
        else:
            if any(k in options for k in _AL_OPTIONS):
                raise ValueError("augmented Lagrangian options given for a problem without "
                                 "bounds or constraint factors")
            summary = minimizer.minimize(self.stream, self.problem)
        cp.cuda.runtime.streamSynchronize(self.stream.get_stream())
        self._solved_once = True
        return SolveResult(summary, bool(summary.converged()), summary.message, self)


def solve(factor_class, *measurements, initial, connect, state_class=None,
          constant: Optional[Sequence[int]] = None, loss=None, method: str = "lm",
          **options) -> SolveResult:
    """Builds and solves a problem with one state batch and one factor batch in one call.

    Args:
        factor_class: The factor batch class (or name), e.g. ``pycunls.SE3BetweenFactorBatch``.
        *measurements: Its measurement arrays (``pycunls.describe(factor_class)``).
        initial: Initial states, e.g. ``(N, 4, 4)`` poses.
        connect: ``(M, num_slots)`` state indices per factor (e.g. ``(M, 2)`` edges).
        state_class: The state batch class; by default the factor's state type from the catalog.
        constant: Indices of states held fixed (e.g. ``[0]``).
        loss: Optional robust loss.
        method, **options: As :meth:`ProblemBuilder.solve`.

    Returns:
        :class:`SolveResult`; ``result.states`` are the solved states (NumPy).
    """
    if state_class is None:
        cls = _resolve(factor_class, _core.FactorBatch, "factor")
        info = factor_info(cls.__name__)
        if info is None or not info.states or len(set(info.states)) != 1:
            raise ValueError(f"pass state_class: {cls.__name__} has no single state type in the "
                             "catalog")
        state_class = info.states[0]
    builder = ProblemBuilder()
    states = builder.add_states(state_class, initial, constant=constant)
    builder.add_factors(factor_class, *measurements, connect=(states, connect), loss=loss)
    return builder.solve(method=method, **options)
