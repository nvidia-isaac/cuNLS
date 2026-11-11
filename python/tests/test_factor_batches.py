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

Each test class constructs one factor type with dummy (zeroed) Warp observation
data and verifies that ``num_active_factors``, ``residuals_size``, and
``state_sizes()`` report the expected values.  These are pure
construction/metadata tests — evaluation correctness is tested end-to-end in
``test_minimizer.py`` and ``test_warp_factor.py``.
"""

import numpy as np
import pytest
import warp as wp

import pycunls


class TestReprojectionFactorBatch:
    """2D reprojection factor: residual=2, states=[SE3(6), Point3(3)]."""
    def test_creation(self):
        num_obs = 100
        obs = wp.zeros(num_obs * 2, dtype=wp.float32)
        fb = pycunls.ReprojectionFactorBatch(obs, num_obs)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num_obs
        assert fb.residuals_size == 2
        assert fb.state_sizes() == [6, 3]

    def test_custom_z_threshold(self):
        obs = wp.zeros(20, dtype=wp.float32)
        fb = pycunls.ReprojectionFactorBatch(obs, 10, z_threshold=0.1)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == 10


class TestPnPFactorBatch:
    """PnP factor: residual=2, single SE3(6); 3D points fixed at construction."""
    def test_creation(self):
        num_obs = 20
        obs = wp.zeros(num_obs * 2, dtype=wp.float32)
        pts = wp.zeros(num_obs * 3, dtype=wp.float32)
        fb = pycunls.PnPFactorBatch(obs, pts, num_obs)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num_obs
        assert fb.residuals_size == 2
        assert fb.state_sizes() == [6]

    def test_creation_with_camera_from_rig(self):
        n = 8
        obs = wp.zeros(n * 2, dtype=wp.float32)
        rig = wp.array(np.tile(np.eye(4, dtype=np.float32).reshape(-1), n))
        pts = wp.zeros(n * 3, dtype=wp.float32)
        fb = pycunls.PnPFactorBatch(obs, rig, pts, n, z_threshold=0.02)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == n
        assert fb.state_sizes() == [6]


class TestSE3BetweenFactorBatch:
    """SE(3) between factor: residual=6, states=[SE3(6), SE3(6)]."""
    def test_creation(self):
        num = 50
        deltas = wp.zeros(num * 16, dtype=wp.float32)
        fb = pycunls.SE3BetweenFactorBatch(deltas, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 6
        assert fb.state_sizes() == [6, 6]


class TestSE3PriorFactorBatch:
    """SE(3) prior factor: residual=6, states=[SE3(6)]."""
    def test_creation(self):
        num = 10
        obs = wp.zeros(num * 16, dtype=wp.float32)
        fb = pycunls.SE3PriorFactorBatch(obs, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 6
        assert fb.state_sizes() == [6]


class TestSO3PriorFactorBatch:
    """SO(3) prior factor: residual=3, states=[SO3(3)]."""
    def test_creation(self):
        num = 10
        obs = wp.zeros(num * 9, dtype=wp.float32)
        fb = pycunls.SO3PriorFactorBatch(obs, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 3
        assert fb.state_sizes() == [3]


class TestSO2PriorFactorBatch:
    """SO(2) prior factor: residual=1, states=[SO2(1)]."""
    def test_creation(self):
        num = 10
        obs = wp.zeros(num * 4, dtype=wp.float32)
        fb = pycunls.SO2PriorFactorBatch(obs, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 1
        assert fb.state_sizes() == [1]


class TestPriorVectorFactorBatches:
    """Euclidean vector prior: residual = state - observation."""
    @pytest.mark.parametrize("cls,dim", [
        (pycunls.PriorVectorFactorBatch1, 1),
        (pycunls.PriorVectorFactorBatch2, 2),
        (pycunls.PriorVectorFactorBatch3, 3),
        (pycunls.PriorVectorFactorBatch6, 6),
    ])
    def test_creation(self, cls, dim):
        num = 20
        obs = wp.zeros(num * dim, dtype=wp.float32)
        fb = cls(obs, num)
        assert fb.capacity == num
        assert fb.num_active_factors == 0  # zero until set
        fb.set_num_active_factors(num)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]


class TestPointToPointFactorBatch:
    """Point-to-point ICP factor: residual = p - T*q, states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = wp.zeros(num * 3, dtype=wp.float32)
        q = wp.zeros(num * 3, dtype=wp.float32)
        fb = pycunls.PointToPointFactorBatch(p, q, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 3
        assert fb.state_sizes() == [6]


class TestPointToPlaneFactorBatch:
    """Point-to-plane ICP factor: residual = Nq'(p - T*q), states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = wp.zeros(num * 3, dtype=wp.float32)
        q = wp.zeros(num * 3, dtype=wp.float32)
        nq = wp.zeros(num * 3, dtype=wp.float32)
        fb = pycunls.PointToPlaneFactorBatch(p, q, nq, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 1
        assert fb.state_sizes() == [6]


class TestSymmetricPointToPlaneFactorBatch:
    """Symmetric point-to-plane ICP factor, states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = wp.zeros(num * 3, dtype=wp.float32)
        q = wp.zeros(num * 3, dtype=wp.float32)
        np_ = wp.zeros(num * 3, dtype=wp.float32)
        nq = wp.zeros(num * 3, dtype=wp.float32)
        fb = pycunls.SymmetricPointToPlaneFactorBatch(p, q, np_, nq, num)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 1
        assert fb.state_sizes() == [6]


class TestInformationFactorBatch:
    """Wraps any factor and applies per-factor sqrt-information matrices."""
    def test_creation(self):
        num = 20
        dim = 3
        obs = wp.zeros(num * dim, dtype=wp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        inner.set_num_active_factors(inner.capacity)
        sqrt_info = wp.array(np.tile(np.eye(dim, dtype=np.float32).reshape(-1), num))
        fb = pycunls.InformationFactorBatch(inner, sqrt_info)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]

    def test_wraps_se3_between(self):
        num = 10
        deltas = wp.zeros(num * 16, dtype=wp.float32)
        inner = pycunls.SE3BetweenFactorBatch(deltas, num)
        inner.set_num_active_factors(inner.capacity)
        sqrt_info = wp.array(np.tile(np.eye(6, dtype=np.float32).reshape(-1), num))
        fb = pycunls.InformationFactorBatch(inner, sqrt_info)
        assert fb.num_active_factors == num
        assert fb.residuals_size == 6
        assert fb.state_sizes() == [6, 6]

    def test_wraps_weighted_prior_vector(self):
        num = 20
        dim = 3
        obs = wp.zeros(num * dim, dtype=wp.float32)
        base = pycunls.PriorVectorFactorBatch3(obs, num)
        base.set_num_active_factors(base.capacity)
        weighted = pycunls.WeightedFactorBatch(base, weight=1.5)
        sqrt_info = wp.array(np.tile(np.eye(dim, dtype=np.float32).reshape(-1), num))
        fb = pycunls.InformationFactorBatch(weighted, sqrt_info)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]


class TestWeightedFactorBatch:
    """Wraps any factor and scales by uniform or per-factor weights."""
    def test_uniform_weight(self):
        num = 20
        dim = 3
        obs = wp.zeros(num * dim, dtype=wp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        inner.set_num_active_factors(inner.capacity)
        fb = pycunls.WeightedFactorBatch(inner, weight=2.0)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]

    def test_per_factor_weights(self):
        num = 20
        dim = 3
        obs = wp.zeros(num * dim, dtype=wp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        inner.set_num_active_factors(inner.capacity)
        weights = wp.ones(num, dtype=wp.float32)
        fb = pycunls.WeightedFactorBatch(inner, weights=weights)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]

    def test_positional_number_is_a_scalar_weight(self):
        # ``weights`` is keyword-only: a positional number never becomes a pointer.
        inner = pycunls.PriorVectorFactorBatch3(wp.zeros(3, dtype=wp.float32), 1)
        fb = pycunls.WeightedFactorBatch(inner, 2)
        assert fb.capacity == 1
        with pytest.raises(TypeError):
            pycunls.WeightedFactorBatch(inner, wp.ones(4, dtype=wp.float32))

    def test_wraps_se3_between(self):
        num = 10
        deltas = wp.zeros(num * 16, dtype=wp.float32)
        inner = pycunls.SE3BetweenFactorBatch(deltas, num)
        fb = pycunls.WeightedFactorBatch(inner, weight=0.5)
        assert fb.capacity == num
        assert fb.num_active_factors == 0
        fb.set_num_active_factors(num)  # forwarded to the wrapped batch
        assert inner.num_active_factors == num
        assert fb.num_active_factors == num
        assert fb.residuals_size == 6
        assert fb.state_sizes() == [6, 6]

    def test_wraps_information_prior_vector(self):
        num = 20
        dim = 3
        obs = wp.zeros(num * dim, dtype=wp.float32)
        base = pycunls.PriorVectorFactorBatch3(obs, num)
        base.set_num_active_factors(base.capacity)
        sqrt_info = wp.array(np.tile(np.eye(dim, dtype=np.float32).reshape(-1), num))
        info_inner = pycunls.InformationFactorBatch(base, sqrt_info)
        fb = pycunls.WeightedFactorBatch(info_inner, weight=1.5)
        assert fb.num_active_factors == num
        assert fb.residuals_size == dim
        assert fb.state_sizes() == [dim]


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
