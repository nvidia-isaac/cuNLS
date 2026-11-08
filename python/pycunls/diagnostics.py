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

"""Checks of a :class:`~pycunls.Problem` before and after solving.

* :func:`diagnose` -- everything that can be checked without solving: the structure (connectivity,
  active counts), non-finite states and residuals, and gauge freedoms. Returns a list of
  :class:`Issue`, each with the fix. Run it when a solve fails or before the first solve.
* :func:`evaluate` -- the residuals of every factor batch at the current states: cost, RMS, the
  worst factor and non-finite counts. Before a solve it checks data layout and the initial guess;
  after a solve it shows where the remaining error is (outliers, a wrong model).
* :func:`check_jacobians` -- the Jacobian of every factor batch against central differences. Run
  it for every custom (Warp) factor.

All three read the problem's own device buffers and leave the states unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional

import cupy as cp
import numpy as np

from pycunls._summary import _plain
from pycunls.catalog import effective_factor
from pycunls._pycunls_core import CudaStream, Problem


# ---------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------

def _stream(stream: Optional[CudaStream]) -> CudaStream:
    return stream if stream is not None else CudaStream()


def _external(stream: CudaStream):
    """The cuNLS stream as a CuPy stream, so CuPy work is ordered with the solver's."""
    if hasattr(cp.cuda.Stream, "from_external"):  # CuPy >= 14
        return cp.cuda.Stream.from_external(stream)
    return cp.cuda.ExternalStream(stream.get_stream())


def _batch_name(problem: Problem, index: int) -> str:
    return f"#{index} {type(problem.factor_batch(index)).__name__}"


def _state_view(batch) -> cp.ndarray:
    """The active states of a state batch as a (num_active_states, ambient_size) CuPy view."""
    n, a = batch.num_active_states, batch.ambient_size
    if n == 0:
        return cp.empty((0, a), cp.float32)
    ptr = batch.state_device_ptr(0)
    mem = cp.cuda.UnownedMemory(ptr, n * a * 4, batch)
    return cp.ndarray((n, a), cp.float32, cp.cuda.MemoryPointer(mem, 0))


def _pointer_table(problem: Problem, index: int) -> np.ndarray:
    return np.asarray(problem.host_state_pointers(index), dtype=np.uint64)


# ---------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class BatchResiduals:
    """Residual statistics of one factor batch (see :func:`evaluate`).

    Attributes:
        index: Position of the batch in the problem (the order of ``add_factor_batch``).
        name: ``"#index ClassName"``.
        num_factors: Active factors.
        residual_size: Residual dimension of one factor.
        cost: ``0.5 * sum(|r|^2)`` over the finite residuals, before any robust loss.
        rms: Root mean square of the finite residual components.
        max_abs: Largest finite residual component.
        worst_factor: Factor with the largest ``|r|`` (-1 without factors).
        num_non_finite: Factors with a NaN or Inf residual component.
        first_non_finite: First such factor (-1 if none).
        has_loss: The batch has a robust loss (``cost`` is the plain squared cost).
        evaluate_ok: The factor batch did not report a failure.
        residuals: The (num_factors, residual_size) residuals when requested, else None.
    """
    index: int
    name: str
    num_factors: int
    residual_size: int
    cost: float
    rms: float
    max_abs: float
    worst_factor: int
    num_non_finite: int
    first_non_finite: int
    has_loss: bool
    evaluate_ok: bool
    residuals: Optional[cp.ndarray] = None

    def factor_norms(self) -> cp.ndarray:
        """Per-factor residual norm ``|r_f|`` (needs ``keep_residuals=True``)."""
        if self.residuals is None:
            raise ValueError("call evaluate(..., keep_residuals=True) to keep the residuals")
        return cp.linalg.norm(self.residuals, axis=1)


@dataclasses.dataclass
class Evaluation:
    """Result of :func:`evaluate`: one :class:`BatchResiduals` per factor batch."""
    batches: List[BatchResiduals]

    @property
    def total_cost(self) -> float:
        """Sum of the batch costs (plain squared cost, finite residuals only)."""
        return float(sum(b.cost for b in self.batches))

    @property
    def all_finite(self) -> bool:
        """No factor has a NaN or Inf residual."""
        return all(b.num_non_finite == 0 for b in self.batches)

    def to_dict(self) -> Dict:
        """JSON-ready dict (without the residual arrays)."""
        rows = []
        for b in self.batches:
            row = dataclasses.asdict(dataclasses.replace(b, residuals=None))
            row.pop("residuals")
            rows.append(row)
        return _plain({"total_cost": self.total_cost, "all_finite": self.all_finite,
                       "batches": rows})

    def __str__(self) -> str:
        head = (f"{'batch':<40} {'factors':>8} {'cost':>11} {'rms':>10} {'max|r|':>10} "
                f"{'worst':>7} {'non-finite':>10}")
        lines = [head]
        for b in self.batches:
            name = b.name + (" (+loss)" if b.has_loss else "")
            lines.append(f"{name[:40]:<40} {b.num_factors:>8} {b.cost:>11.4g} {b.rms:>10.4g} "
                         f"{b.max_abs:>10.4g} {b.worst_factor:>7} {b.num_non_finite:>10}")
        lines.append(f"total cost {self.total_cost:.6g}" +
                     ("" if self.all_finite else "  (NON-FINITE residuals present)"))
        return "\n".join(lines)


def evaluate(problem: Problem, stream: Optional[CudaStream] = None,
             keep_residuals: bool = False) -> Evaluation:
    """Evaluates every factor batch at the current states (no solve, states unchanged).

    Before solving, it checks the data and the initial guess: a NaN residual, a huge cost in one
    batch, or a residual far larger than the measurement noise points at a wrong layout (e.g. a
    transposed 4x4 pose) or a bad initial guess. After solving, it shows where the remaining error
    is: the RMS per batch should be near the measurement noise; ``worst_factor`` locates outliers.

    Args:
        problem: The problem (its state and factor batches must be set up as for a solve).
        stream: CUDA stream (a new one by default); synchronized before returning.
        keep_residuals: Keep each batch's (num_factors, residual_size) residuals.

    Returns:
        :class:`Evaluation`; ``print()`` it for a table, ``to_dict()`` for JSON.
    """
    stream = _stream(stream)
    batches = []
    with _external(stream) as ext:
        for i in range(problem.num_residual_batches):
            fb = problem.factor_batch(i)
            n, m = fb.num_active_factors, fb.residuals_size
            if n == 0:
                batches.append(BatchResiduals(i, _batch_name(problem, i), 0, m, 0.0, 0.0, 0.0,
                                              -1, 0, -1, problem.loss_function(i) is not None,
                                              True, None))
                continue
            table = cp.asarray(_pointer_table(problem, i))
            residuals = cp.empty((n, m), cp.float32)
            ok = fb.evaluate_batch(residuals, None, table, stream)
            finite_rows = cp.all(cp.isfinite(residuals), axis=1)
            clean = cp.where(finite_rows[:, None], residuals, 0.0)
            sq = cp.sum(clean.astype(cp.float64) ** 2, axis=1)
            num_bad = int(n - int(cp.count_nonzero(finite_rows)))
            first_bad = int(cp.argmin(finite_rows)) if num_bad else -1
            num_good = n - num_bad
            batches.append(BatchResiduals(
                index=i, name=_batch_name(problem, i), num_factors=n, residual_size=m,
                cost=0.5 * float(cp.sum(sq)),
                rms=math.sqrt(float(cp.sum(sq)) / (num_good * m)) if num_good else float("nan"),
                max_abs=float(cp.max(cp.abs(clean))) if num_good else float("nan"),
                worst_factor=int(cp.argmax(sq)) if num_good else -1,
                num_non_finite=num_bad, first_non_finite=first_bad,
                has_loss=problem.loss_function(i) is not None, evaluate_ok=bool(ok),
                residuals=residuals if keep_residuals else None))
        ext.synchronize()
    return Evaluation(batches)


# ---------------------------------------------------------------------------------------------
# check_jacobians
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class JacobianCheck:
    """Analytic vs. central-difference Jacobian of one factor batch (see :func:`check_jacobians`).

    Attributes:
        index, name: The batch.
        max_abs_error: Largest ``|J - J_fd|`` entry.
        relative_error: ``max_abs_error / max(1, max |J_fd|)``.
        worst_factor, worst_row, worst_column: Where the largest error is (column counts the
            state slots side by side, in tangent coordinates).
        ok: ``relative_error <= tolerance``.
    """
    index: int
    name: str
    max_abs_error: float
    relative_error: float
    worst_factor: int
    worst_row: int
    worst_column: int
    ok: bool

    def __str__(self) -> str:
        verdict = "ok" if self.ok else "MISMATCH"
        return (f"{self.name}: {verdict} (relative error {self.relative_error:.3g}, worst at "
                f"factor {self.worst_factor}, row {self.worst_row}, column {self.worst_column})")


def _locate(pointers: np.ndarray, batches) -> List:
    """For each pointer: (state batch index, state index), or (-1, -1)."""
    out = np.full((pointers.size, 2), -1, np.int64)
    for j, sb in enumerate(batches):
        n = sb.num_active_states
        if n == 0:
            continue
        base = np.uint64(sb.state_device_ptr(0))
        stride = np.uint64(sb.ambient_size * 4)
        rel = pointers - base
        inside = (pointers >= base) & (rel < stride * np.uint64(n))
        out[inside, 0] = j
        out[inside, 1] = (rel[inside] // stride).astype(np.int64)
    return out


def check_jacobians(problem: Problem, stream: Optional[CudaStream] = None, step: float = 1e-3,
                    tolerance: float = 1e-2, batches: Optional[List[int]] = None
                    ) -> List[JacobianCheck]:
    """Compares each factor batch's Jacobian with central differences at the current states.

    The difference is taken along the manifold update (``StateBatch.apply_plus``), so it checks
    exactly the Jacobian the minimizers use (with respect to the tangent-space step). Run it for
    every custom factor: a wrong sign, a swapped column or a Jacobian for the wrong state order
    shows up as a relative error of order 1, while float32 round-off stays well below the default
    tolerance. Factors with kinks (robust thresholds, clamps) can differ near the kink.

    Args:
        problem: The problem, set up as for a solve.
        stream: CUDA stream (a new one by default); synchronized before returning.
        step: Central-difference step in tangent coordinates.
        tolerance: Largest accepted relative error.
        batches: Residual batch indices to check (all by default).

    Returns:
        One :class:`JacobianCheck` per checked batch.
    """
    stream = _stream(stream)
    state_batches = problem.state_batches()
    indices = range(problem.num_residual_batches) if batches is None else batches
    results = []
    with _external(stream) as ext:
        for i in indices:
            fb = problem.factor_batch(i)
            n, m = fb.num_active_factors, fb.residuals_size
            sizes = list(fb.state_sizes())
            cols = sum(sizes)
            if n == 0 or cols == 0:
                continue
            pointers = _pointer_table(problem, i)
            table = cp.asarray(pointers)
            analytic = cp.empty((n, m, cols), cp.float32)
            fb.evaluate_batch(cp.empty((n, m), cp.float32), analytic, table, stream)
            numeric = cp.zeros((n, m, cols), cp.float32)
            where = _locate(pointers, state_batches).reshape(n, len(sizes), 2)
            col = 0
            for slot, tangent in enumerate(sizes):
                for j in np.unique(where[:, slot, 0]):
                    if j < 0:
                        continue
                    sb = state_batches[int(j)]
                    x = _state_view(sb)
                    base = np.uint64(sb.state_device_ptr(0))
                    rows = where[:, slot, 0] == j
                    for k in range(tangent):
                        r = []
                        for sign in (1.0, -1.0):
                            delta = cp.zeros((sb.num_active_states, sb.tangent_size), cp.float32)
                            delta[:, k] = sign * step
                            moved = cp.empty_like(x)
                            sb.apply_plus(x, delta, moved, stream)
                            # Slot `slot` of the factors on batch j reads the moved copy.
                            shifted = pointers.reshape(n, len(sizes)).copy()
                            offset = np.uint64(moved.data.ptr) - base
                            shifted[rows, slot] = shifted[rows, slot] + offset
                            out = cp.empty((n, m), cp.float32)
                            fb.evaluate_batch(out, None, cp.asarray(shifted.reshape(-1)), stream)
                            r.append(out)
                        diff = (r[0] - r[1]) / (2.0 * step)
                        rows_d = cp.asarray(rows)
                        numeric[:, :, col + k] = cp.where(rows_d[:, None], diff,
                                                          numeric[:, :, col + k])
                col += tangent
            error = cp.abs(analytic - numeric)
            error = cp.where(cp.isfinite(error), error, cp.inf)
            flat = int(cp.argmax(error))
            f, row, c = np.unravel_index(flat, error.shape)
            max_err = float(error.reshape(-1)[flat])
            scale = max(1.0, float(cp.max(cp.abs(cp.nan_to_num(numeric)))))
            rel = max_err / scale
            results.append(JacobianCheck(i, _batch_name(problem, i), max_err, rel, int(f),
                                         int(row), int(c), bool(rel <= tolerance)))
        ext.synchronize()
    return results


# ---------------------------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Issue:
    """One finding of :func:`diagnose`.

    Attributes:
        severity: ``"error"`` (the solve will fail or be wrong) or ``"warning"`` (likely a
            problem, e.g. a gauge freedom that Gauss-Newton cannot handle).
        message: What is wrong, where, and how to fix it.
    """
    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def _gauge_issues(problem: Problem, state_batches) -> List[Issue]:
    """Connected groups of free states that only relative factors touch (no prior, no constant)."""
    # Union-find over (state batch, state) keys.
    parent: Dict[tuple, tuple] = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    anchored = set()
    for i in range(problem.num_residual_batches):
        fb = problem.factor_batch(i)
        if fb.num_active_factors == 0:
            continue
        _, _, role = effective_factor(fb)
        if role == "inactive":
            continue
        slots = len(fb.state_sizes())
        where = _locate(_pointer_table(problem, i), state_batches).reshape(-1, slots, 2)
        keys = [[(int(a), int(b)) for a, b in row] for row in where]
        for row in keys:
            for key in row[1:]:
                parent[find(key)] = find(row[0])
            if role != "relative":  # priors, measurements; custom factors are assumed to anchor
                anchored.update(row)
    for j, sb in enumerate(state_batches):
        if sb.num_const_states:
            # Constant states anchor their group.
            anchored.update((j, int(s)) for s in sb.constant_state_ids())
    roots_anchored = {find(k) for k in anchored if k in parent}
    groups: Dict[tuple, List[tuple]] = {}
    for key in list(parent):
        if key[0] >= 0:
            groups.setdefault(find(key), []).append(key)
    issues = []
    for root, members in groups.items():
        if root in roots_anchored:
            continue
        batches = sorted({state_batches[k[0]].__class__.__name__ for k in members})
        issues.append(Issue(
            "warning",
            f"Gauge freedom: {len(members)} connected free states ({', '.join(batches)}; e.g. "
            f"state {members[0][1]} of state batch {members[0][0]}) are only constrained "
            "relative to each other (between / reprojection / motion factors), so the whole group "
            "can move without changing the cost. Gauss-Newton with a direct solver may fail on the "
            "singular system; Levenberg-Marquardt damps it but the result floats. Fix: make one "
            "state of the group constant (constant ids) or add a prior factor on it."))
    return issues


def diagnose(problem: Problem, stream: Optional[CudaStream] = None) -> List[Issue]:
    """Checks a problem without solving it; returns the issues found (empty: nothing found).

    Checks, in order: the structure (``Problem.find_structure_error``: active counts,
    connectivity, unconstrained states), non-finite state values, non-finite residuals at the
    current states (per factor batch, with the first bad factor), and gauge freedoms (groups of
    free states that only between-type factors connect, with no constant state or prior). The
    gauge check is a heuristic, not a rank test: wrapped factors are judged by the factor they
    wrap (``effective_factor`` in ``pycunls/catalog.py``), inequality constraints are ignored
    (they may be inactive), and custom factors are assumed to fix their states.

    Args:
        problem: The problem, set up as for a solve.
        stream: CUDA stream (a new one by default); synchronized before returning.

    Example::

        for issue in pycunls.diagnose(problem):
            print(issue)
    """
    stream = _stream(stream)
    issues: List[Issue] = []
    error = problem.find_structure_error(stream)
    if error:
        return [Issue("error", error)]
    state_batches = problem.state_batches()
    with _external(stream) as ext:
        for j, sb in enumerate(state_batches):
            if sb.num_active_states == 0:
                continue
            bad = cp.flatnonzero(~cp.all(cp.isfinite(_state_view(sb)), axis=1))
            if bad.size:
                issues.append(Issue(
                    "error",
                    f"State batch {j} ({type(sb).__name__}): {bad.size} states hold NaN or Inf, "
                    f"first state {int(bad[0])}. Fix: initialize every active state with finite "
                    "float32 values in the batch's layout."))
        ext.synchronize()
    if issues:
        return issues
    evaluation = evaluate(problem, stream)
    for b in evaluation.batches:
        if b.num_non_finite:
            issues.append(Issue(
                "error",
                f"Factor batch {b.name}: {b.num_non_finite} of {b.num_factors} factors have NaN "
                f"or Inf residuals at the current states, first factor {b.first_non_finite}. Fix: "
                "check the measurement data and layout for that factor and that its states are "
                "in the factor's domain (e.g. points in front of the camera)."))
        if not b.evaluate_ok:
            issues.append(Issue("error", f"Factor batch {b.name} reported an evaluation failure."))
    issues.extend(_gauge_issues(problem, state_batches))
    return issues
