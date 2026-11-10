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

"""``to_dict()`` and ``report()`` for the summaries of the native module.

The summaries are plain native structs; these two methods make them easy to log, compare and
read in a pipeline: ``to_dict()`` gives JSON-ready values (enums as names), ``report()`` a short
multi-line text with the outcome first.
"""

import enum
import math

import numpy as np

from pycunls._pycunls_core import (
    AugmentedLagrangianMinimizerSummary,
    MinimizerSummary,
    RansacSummary,
    TerminationReason,
)

# Fields of each summary, in report order (inherited fields first).
_MINIMIZER_FIELDS = ("termination_reason", "message", "num_iterations", "initial_cost",
                     "final_cost", "num_free_parameters", "solve_time_ms",
                     "num_problems_by_reason", "iteration_costs")
_RANSAC_FIELDS = _MINIMIZER_FIELDS + ("num_rounds", "num_hypotheses", "num_valid_hypotheses",
                                      "num_inliers", "inlier_ratio", "best_score",
                                      "refinement_reverted")
_AL_FIELDS = ("status", "message", "outer_iterations", "inner_iterations", "max_violation",
              "initial_cost", "final_cost", "num_problems", "num_converged", "num_max_penalty",
              "num_inner_failed", "solve_time_ms")


def _plain(value):
    """A value that ``json.dumps(..., allow_nan=False)`` accepts, for every ``to_dict()``.

    Enums become their names, non-finite numbers None, NumPy scalars Python numbers, arrays,
    lists and tuples lists, dicts dicts (all nested).
    """
    if isinstance(value, enum.Enum) or type(value).__name__ in ("TerminationReason",
                                                                 "AugmentedLagrangianMinimizerStatus"):
        return value.name
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _fields(summary):
    if isinstance(summary, RansacSummary):
        return _RANSAC_FIELDS
    if isinstance(summary, AugmentedLagrangianMinimizerSummary):
        return _AL_FIELDS
    return _MINIMIZER_FIELDS


def _to_dict(self):
    """All fields as a JSON-ready dict (enums as names, NaN as None) plus ``converged``.

    ``num_problems_by_reason`` becomes a ``{reason name: count}`` dict without zero counts.
    """
    out = {"converged": bool(self.converged())}
    for name in _fields(self):
        value = getattr(self, name)
        if name == "num_problems_by_reason":
            value = {TerminationReason(i).name: int(c) for i, c in enumerate(value) if c}
        out[name] = _plain(value)
    return out


def _report(self):
    """A short multi-line text: the outcome and message first, then the numbers."""
    d = self.to_dict()
    head = "converged" if d["converged"] else "NOT CONVERGED"
    lines = [f"{type(self).__name__}: {head}", f"  {d['message']}"]
    for name, value in d.items():
        if name in ("converged", "message", "iteration_costs"):
            continue
        if isinstance(value, float):
            value = f"{value:.6g}"
        lines.append(f"  {name}: {value}")
    costs = d.get("iteration_costs")
    if costs:
        shown = ", ".join("nan" if c is None else f"{c:.4g}" for c in costs[:8])
        more = f", ... ({len(costs)} total)" if len(costs) > 8 else ""
        lines.append(f"  iteration_costs: [{shown}{more}]")
    return "\n".join(lines)


for _cls in (MinimizerSummary, AugmentedLagrangianMinimizerSummary):
    _cls.to_dict = _to_dict
    _cls.report = _report
