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

Each test class constructs one factor type with dummy (zeroed) CuPy observation
data and verifies that ``num_factors``, ``residuals_size``, and
``state_block_sizes()`` report the expected values.  These are pure
construction/metadata tests — evaluation correctness is tested end-to-end in
``test_minimizer.py`` and ``test_warp_factor.py``.
"""

import cupy as cp
import pytest

import pycunls


class TestReprojectionFactorBatch:
    """2D reprojection factor: residual=2, states=[SE3(6), Point3(3)]."""
    def test_creation(self):
        num_obs = 100
        obs = cp.zeros(num_obs * 2, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(obs, num_obs)
        assert fb.num_factors == num_obs
        assert fb.residuals_size == 2
        assert fb.state_block_sizes() == [6, 3]

    def test_custom_z_threshold(self):
        obs = cp.zeros(20, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(obs, 10, z_threshold=0.1)
        assert fb.num_factors == 10


class TestPnPFactorBatch:
    """PnP factor: residual=2, single SE3(6); 3D points fixed at construction."""
    def test_creation(self):
        num_obs = 20
        obs = cp.zeros(num_obs * 2, dtype=cp.float32)
        pts = cp.zeros(num_obs * 3, dtype=cp.float32)
        fb = pycunls.PnPFactorBatch(obs, pts, num_obs)
        assert fb.num_factors == num_obs
        assert fb.residuals_size == 2
        assert fb.state_block_sizes() == [6]

    def test_creation_with_camera_from_rig(self):
        n = 8
        obs = cp.zeros(n * 2, dtype=cp.float32)
        rig = cp.zeros(n * 16, dtype=cp.float32)
        rig[0::16] = 1.0
        rig[5::16] = 1.0
        rig[10::16] = 1.0
        rig[15::16] = 1.0
        pts = cp.zeros(n * 3, dtype=cp.float32)
        fb = pycunls.PnPFactorBatch(obs, rig, pts, n, z_threshold=0.02)
        assert fb.num_factors == n
        assert fb.state_block_sizes() == [6]


class TestSE3BetweenFactorBatch:
    """SE(3) between factor: residual=6, states=[SE3(6), SE3(6)]."""
    def test_creation(self, cublas):
        num = 50
        deltas = cp.zeros(num * 16, dtype=cp.float32)
        fb = pycunls.SE3BetweenFactorBatch(deltas, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6, 6]


class TestSE3PriorFactorBatch:
    """SE(3) prior factor: residual=6, states=[SE3(6)]."""
    def test_creation(self):
        num = 10
        obs = cp.zeros(num * 16, dtype=cp.float32)
        fb = pycunls.SE3PriorFactorBatch(obs, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6]


class TestSO3PriorFactorBatch:
    """SO(3) prior factor: residual=3, states=[SO3(3)]."""
    def test_creation(self):
        num = 10
        obs = cp.zeros(num * 9, dtype=cp.float32)
        fb = pycunls.SO3PriorFactorBatch(obs, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 3
        assert fb.state_block_sizes() == [3]


class TestSO2PriorFactorBatch:
    """SO(2) prior factor: residual=1, states=[SO2(1)]."""
    def test_creation(self):
        num = 10
        obs = cp.zeros(num * 4, dtype=cp.float32)
        fb = pycunls.SO2PriorFactorBatch(obs, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 1
        assert fb.state_block_sizes() == [1]


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
        obs = cp.zeros(num * dim, dtype=cp.float32)
        fb = cls(obs, num)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]


class TestPointToPointFactorBatch:
    """Point-to-point ICP factor: residual = p - T*q, states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = cp.zeros(num * 3, dtype=cp.float32)
        q = cp.zeros(num * 3, dtype=cp.float32)
        fb = pycunls.PointToPointFactorBatch(p, q, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 3
        assert fb.state_block_sizes() == [6]


class TestPointToPlaneFactorBatch:
    """Point-to-plane ICP factor: residual = Nq'(p - T*q), states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = cp.zeros(num * 3, dtype=cp.float32)
        q = cp.zeros(num * 3, dtype=cp.float32)
        nq = cp.zeros(num * 3, dtype=cp.float32)
        fb = pycunls.PointToPlaneFactorBatch(p, q, nq, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 1
        assert fb.state_block_sizes() == [6]


class TestSymmetricPointToPlaneFactorBatch:
    """Symmetric point-to-plane ICP factor, states=[SE3(6)]."""
    def test_creation(self):
        num = 30
        p = cp.zeros(num * 3, dtype=cp.float32)
        q = cp.zeros(num * 3, dtype=cp.float32)
        np_ = cp.zeros(num * 3, dtype=cp.float32)
        nq = cp.zeros(num * 3, dtype=cp.float32)
        fb = pycunls.SymmetricPointToPlaneFactorBatch(p, q, np_, nq, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 1
        assert fb.state_block_sizes() == [6]


class TestInformationFactorBatch:
    """Wraps any factor and applies per-factor sqrt-information matrices."""
    def test_creation(self, cublas):
        num = 20
        dim = 3
        obs = cp.zeros(num * dim, dtype=cp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        sqrt_info = cp.eye(dim, dtype=cp.float32).reshape(-1)
        sqrt_info = cp.tile(sqrt_info, num)
        fb = pycunls.InformationFactorBatch(cublas, inner, sqrt_info)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]

    def test_wraps_se3_between(self, cublas):
        num = 10
        deltas = cp.zeros(num * 16, dtype=cp.float32)
        inner = pycunls.SE3BetweenFactorBatch(deltas, num)
        sqrt_info = cp.eye(6, dtype=cp.float32).reshape(-1)
        sqrt_info = cp.tile(sqrt_info, num)
        fb = pycunls.InformationFactorBatch(cublas, inner, sqrt_info)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6, 6]

    def test_wraps_weighted_prior_vector(self, cublas):
        num = 20
        dim = 3
        obs = cp.zeros(num * dim, dtype=cp.float32)
        base = pycunls.PriorVectorFactorBatch3(obs, num)
        weighted = pycunls.WeightedFactorBatch(base, weight=1.5)
        sqrt_info = cp.eye(dim, dtype=cp.float32).reshape(-1)
        sqrt_info = cp.tile(sqrt_info, num)
        fb = pycunls.InformationFactorBatch(cublas, weighted, sqrt_info)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]


class TestWeightedFactorBatch:
    """Wraps any factor and scales by uniform or per-factor weights."""
    def test_uniform_weight(self):
        num = 20
        dim = 3
        obs = cp.zeros(num * dim, dtype=cp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        fb = pycunls.WeightedFactorBatch(inner, weight=2.0)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]

    def test_per_factor_weights(self):
        num = 20
        dim = 3
        obs = cp.zeros(num * dim, dtype=cp.float32)
        inner = pycunls.PriorVectorFactorBatch3(obs, num)
        weights = cp.ones(num, dtype=cp.float32)
        fb = pycunls.WeightedFactorBatch(inner, weights=weights)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]

    def test_wraps_se3_between(self, cublas):
        num = 10
        deltas = cp.zeros(num * 16, dtype=cp.float32)
        inner = pycunls.SE3BetweenFactorBatch(deltas, num)
        fb = pycunls.WeightedFactorBatch(inner, weight=0.5)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6, 6]

    def test_wraps_information_prior_vector(self, cublas):
        num = 20
        dim = 3
        obs = cp.zeros(num * dim, dtype=cp.float32)
        base = pycunls.PriorVectorFactorBatch3(obs, num)
        sqrt_info = cp.eye(dim, dtype=cp.float32).reshape(-1)
        sqrt_info = cp.tile(sqrt_info, num)
        info_inner = pycunls.InformationFactorBatch(cublas, base, sqrt_info)
        fb = pycunls.WeightedFactorBatch(info_inner, weight=1.5)
        assert fb.num_factors == num
        assert fb.residuals_size == dim
        assert fb.state_block_sizes() == [dim]


class TestCustomFactorBatch:
    """Python-side trampoline for user-defined factors."""
    def test_creation(self):
        fb = pycunls.CustomFactorBatch(2, [6, 3], 100)
        assert fb.num_factors == 100
        assert fb.residuals_size == 2
        assert fb.state_block_sizes() == [6, 3]

    def test_evaluate_raises_without_override(self):
        fb = pycunls.CustomFactorBatch(1, [1], 10)
        with pytest.raises(RuntimeError, match="must be overridden"):
            fb.evaluate(0, 0, 0, 0)
