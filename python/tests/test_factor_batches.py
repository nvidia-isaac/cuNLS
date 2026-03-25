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
    def test_creation(self, cublas):
        num_obs = 100
        obs = cp.zeros(num_obs * 2, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(cublas, obs, num_obs)
        assert fb.num_factors == num_obs
        assert fb.residuals_size == 2
        assert fb.state_block_sizes() == [6, 3]

    def test_custom_z_threshold(self, cublas):
        obs = cp.zeros(20, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(cublas, obs, 10, z_threshold=0.1)
        assert fb.num_factors == 10


class TestSE3BetweenFactorBatch:
    """SE(3) between factor: residual=6, states=[SE3(6), SE3(6)]."""
    def test_creation(self, cublas):
        num = 50
        deltas = cp.zeros(num * 16, dtype=cp.float32)
        fb = pycunls.SE3BetweenFactorBatch(cublas, deltas, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6, 6]


class TestSE3PriorFactorBatch:
    """SE(3) prior factor: residual=6, states=[SE3(6)]."""
    def test_creation(self, cublas):
        num = 10
        obs = cp.zeros(num * 16, dtype=cp.float32)
        fb = pycunls.SE3PriorFactorBatch(cublas, obs, num)
        assert fb.num_factors == num
        assert fb.residuals_size == 6
        assert fb.state_block_sizes() == [6]


class TestSO3PriorFactorBatch:
    """SO(3) prior factor: residual=3, states=[SO3(3)]."""
    def test_creation(self, cublas):
        num = 10
        obs = cp.zeros(num * 9, dtype=cp.float32)
        fb = pycunls.SO3PriorFactorBatch(cublas, obs, num)
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
