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

"""Tests for factor batch bindings.

Each built-in factor is constructed with dummy (zeroed) Warp observation data and must report
its ``capacity``, ``num_active_factors`` (zero until set), ``residuals_size`` and
``state_sizes()``; the wrappers (weighted, information) forward them to the wrapped batch.
These are pure construction/metadata tests: evaluation correctness is tested end-to-end in
``test_minimizer.py`` and ``test_warp_factor.py``.
"""

import numpy as np
import pytest
import warp as wp

import pycunls

N = 10  # capacity of every batch below


def _zeros(n):
    return wp.zeros(n, dtype=wp.float32)


def _eyes(dim, n):
    """n identity matrices of size dim x dim (sqrt-information or rig poses)."""
    return wp.array(np.tile(np.eye(dim, dtype=np.float32).reshape(-1), n))


# (constructor of a batch of capacity N, residual size, state tangent sizes)
FACTORS = {
    "Reprojection": (lambda: pycunls.ReprojectionFactorBatch(_zeros(2 * N), N), 2, [6, 3]),
    "Reprojection_z_threshold": (
        lambda: pycunls.ReprojectionFactorBatch(_zeros(2 * N), N, z_threshold=0.1), 2, [6, 3]),
    "PnP": (lambda: pycunls.PnPFactorBatch(_zeros(2 * N), _zeros(3 * N), N), 2, [6]),
    "PnP_camera_from_rig": (
        lambda: pycunls.PnPFactorBatch(_zeros(2 * N), _eyes(4, N), _zeros(3 * N), N,
                                       z_threshold=0.02), 2, [6]),
    "SE3Between": (lambda: pycunls.SE3BetweenFactorBatch(_zeros(16 * N), N), 6, [6, 6]),
    "SE3Prior": (lambda: pycunls.SE3PriorFactorBatch(_zeros(16 * N), N), 6, [6]),
    "SO3Prior": (lambda: pycunls.SO3PriorFactorBatch(_zeros(9 * N), N), 3, [3]),
    "SO2Prior": (lambda: pycunls.SO2PriorFactorBatch(_zeros(4 * N), N), 1, [1]),
    **{f"PriorVector{d}": (lambda d=d: getattr(pycunls, f"PriorVectorFactorBatch{d}")(
        _zeros(d * N), N), d, [d]) for d in (1, 2, 3, 6)},
    # ICP: residual = p - T q (point), Nq'(p - T q) (plane), symmetric plane.
    "PointToPoint": (lambda: pycunls.PointToPointFactorBatch(_zeros(3 * N), _zeros(3 * N), N),
                     3, [6]),
    "PointToPlane": (lambda: pycunls.PointToPlaneFactorBatch(*[_zeros(3 * N)] * 3, N), 1, [6]),
    "SymmetricPointToPlane": (
        lambda: pycunls.SymmetricPointToPlaneFactorBatch(*[_zeros(3 * N)] * 4, N), 1, [6]),
}


@pytest.mark.parametrize("make, residual_size, state_sizes", FACTORS.values(), ids=FACTORS)
def test_factor_reports_capacity_and_sizes(make, residual_size, state_sizes):
    fb = make()
    assert fb.capacity == N
    assert fb.num_active_factors == 0  # zero until set
    fb.set_num_active_factors(N)
    assert fb.num_active_factors == N
    assert fb.residuals_size == residual_size
    assert fb.state_sizes() == state_sizes


# Inner batches (constructor, residual size, state sizes) and the wrappers around them.
INNER = {
    "PriorVector3": (lambda: pycunls.PriorVectorFactorBatch3(_zeros(3 * N), N), 3, [3]),
    "SE3Between": (lambda: pycunls.SE3BetweenFactorBatch(_zeros(16 * N), N), 6, [6, 6]),
}
WRAPPERS = {
    "weighted": lambda inner, dim: pycunls.WeightedFactorBatch(inner, weight=2.0),
    "per_factor_weights": lambda inner, dim: pycunls.WeightedFactorBatch(
        inner, weights=wp.ones(N, dtype=wp.float32)),
    "information": lambda inner, dim: pycunls.InformationFactorBatch(inner, _eyes(dim, N)),
    "information_of_weighted": lambda inner, dim: pycunls.InformationFactorBatch(
        pycunls.WeightedFactorBatch(inner, weight=1.5), _eyes(dim, N)),
    "weighted_information": lambda inner, dim: pycunls.WeightedFactorBatch(
        pycunls.InformationFactorBatch(inner, _eyes(dim, N)), weight=1.5),
}


@pytest.mark.parametrize("wrap", WRAPPERS.values(), ids=WRAPPERS)
@pytest.mark.parametrize("make_inner, residual_size, state_sizes", INNER.values(), ids=INNER)
def test_wrapper_forwards_sizes_and_active_count(make_inner, residual_size, state_sizes, wrap):
    inner = make_inner()
    fb = wrap(inner, residual_size)
    assert fb.capacity == N
    fb.set_num_active_factors(N - 1)  # forwarded to the wrapped batch
    assert inner.num_active_factors == N - 1
    inner.set_num_active_factors(N)  # and read back from it
    assert fb.num_active_factors == N
    assert fb.residuals_size == residual_size
    assert fb.state_sizes() == state_sizes


def test_positional_number_is_a_scalar_weight():
    # ``weights`` is keyword-only: a positional number never becomes a pointer.
    inner = pycunls.PriorVectorFactorBatch3(_zeros(3), 1)
    fb = pycunls.WeightedFactorBatch(inner, 2)
    assert fb.capacity == 1
    with pytest.raises(TypeError):
        pycunls.WeightedFactorBatch(inner, wp.ones(4, dtype=wp.float32))


class TestCustomFactorBatch:
    """Python-side trampoline for user-defined factors."""
    def test_creation(self):
        fb = pycunls.CustomFactorBatch(2, [6, 3], 100)
        assert fb.capacity == 100
        assert fb.num_active_factors == 0  # zero until set
        fb.set_num_active_factors(40)
        assert fb.num_active_factors == 40
        with pytest.raises(ValueError):
            fb.set_num_active_factors(101)
        assert fb.residuals_size == 2
        assert fb.state_sizes() == [6, 3]

    def test_evaluate_raises_without_override(self):
        fb = pycunls.CustomFactorBatch(1, [1], 10)
        with pytest.raises(RuntimeError, match="must be overridden"):
            fb.evaluate(0, 0, 0, 0, 0, 1)
