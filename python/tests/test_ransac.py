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

"""Tests for the RANSAC minimizer bindings.

* PnP with gross outliers: RANSAC-GN / RANSAC-LM recover the pose and the
  inlier mask; plain LM does not.
* An always-on prior batch is accepted and never classified.
* Custom Python factor / state (CuPy kernels) that follow the item / replica
  contract work under RANSAC.
* Option validation errors surface as Python exceptions.
"""

import numpy as np
import cupy as cp
import pytest

import pycunls

TAU = 0.01


# ── PnP scene ───────────────────────────────────────────────────────────────

def _rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _pose(rx, ry, rz, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _rotation(rx, ry, rz)
    T[:3, 3] = t
    return T


def _pnp_scene(n, outlier_ratio, seed):
    rng = np.random.default_rng(seed)
    gt = _pose(0.1, -0.2, 0.05, [0.3, -0.2, 8.0])
    pts = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
    cam = pts @ gt[:3, :3].T + gt[:3, 3]
    obs = cam[:, :2] / cam[:, 2:3] + rng.normal(0, 3e-3, (n, 2))
    is_outlier = rng.uniform(size=n) < outlier_ratio
    obs[is_outlier] += rng.choice([-1, 1], (is_outlier.sum(), 2)) * rng.uniform(
        0.1, 0.4, (is_outlier.sum(), 2))
    init = _pose(0.15, -0.25, 0.1, [0.5, -0.4, 8.5])
    return gt, init, pts, obs.astype(np.float32), is_outlier


class _PnPProblem:
    def __init__(self, init, pts, obs, with_prior=False):
        self.cublas = pycunls.CublasHandle()
        self.pose_gpu = cp.asarray(init.reshape(-1))
        self.pts_gpu = cp.asarray(pts.reshape(-1))
        self.obs_gpu = cp.asarray(obs.reshape(-1))
        self.state = pycunls.SE3StateBatch(self.cublas, self.pose_gpu, 1)
        self.state.set_num_state_blocks(self.state.capacity, self.state.const_capacity)
        self.pnp = pycunls.PnPFactorBatch(self.obs_gpu, self.pts_gpu, len(pts))
        self.pnp.set_num_factors(self.pnp.capacity)
        self.problem = pycunls.Problem()
        self.problem.add_state_batch(self.state)
        ptr = self.state.state_block_device_ptr(0)
        self.problem.add_factor_batch(self.pnp, [ptr] * len(pts))
        if with_prior:
            self.prior_gpu = cp.asarray(init.reshape(-1))
            self.prior = pycunls.SE3PriorFactorBatch(self.prior_gpu, 1)
            self.prior.set_num_factors(self.prior.capacity)
            self.problem.add_factor_batch(self.prior, [ptr])

    def pose(self):
        return cp.asnumpy(self.pose_gpu).reshape(4, 4)


def _rot_err_deg(a, b):
    c = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _ransac_options(roles=None):
    o = pycunls.RansacMinimizerOptions()
    o.factor_batches = roles or [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, TAU)]
    o.seed = 3
    return o


# ── Tests ───────────────────────────────────────────────────────────────────

class TestRansacOptions:
    def test_defaults_and_assignment(self):
        o = pycunls.RansacMinimizerOptions()
        assert o.hypotheses_per_round == 256
        assert o.scoring_finalists == 4
        assert o.factor_batches == []
        o.factor_batches = [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.always_on)]
        assert o.factor_batches[0].role == pycunls.RansacRole.always_on

    def test_lm_base_options_are_modified_in_place(self):
        lm = pycunls.RansacLevenbergMarquardtMinimizerOptions()
        lm.base_options.max_rounds = 3
        assert lm.base_options.max_rounds == 3


class TestRansacPnP:
    @pytest.mark.parametrize("use_lm", [False, True])
    def test_rejects_outliers(self, stream, use_lm):
        gt, init, pts, obs, is_outlier = _pnp_scene(1000, 0.5, seed=1)
        p = _PnPProblem(init, pts, obs)
        if use_lm:
            lm = pycunls.RansacLevenbergMarquardtMinimizerOptions()
            lm.base_options = _ransac_options()
            minimizer = pycunls.RansacLevenbergMarquardtMinimizer(lm)
        else:
            minimizer = pycunls.RansacGaussNewtonMinimizer(_ransac_options())
        summary = minimizer.minimize(stream, p.problem)

        assert isinstance(summary, pycunls.RansacSummary)
        assert _rot_err_deg(p.pose(), gt) < 0.2
        mask = minimizer.inlier_mask(0)
        assert mask.dtype == np.uint8 and mask.shape == (1000,)
        assert int(np.sum((mask == 1) & is_outlier)) == 0
        assert int(np.sum((mask == 1) & ~is_outlier)) >= 0.97 * np.sum(~is_outlier)
        assert summary.num_inliers == int(mask.sum())

    def test_plain_lm_fails_on_same_data(self, stream):
        gt, init, pts, obs, _ = _pnp_scene(1000, 0.5, seed=1)
        p = _PnPProblem(init, pts, obs)
        pycunls.LevenbergMarquardtMinimizer().minimize(stream, p.problem)
        assert _rot_err_deg(p.pose(), gt) > 0.5

    def test_always_on_prior_is_not_classified(self, stream):
        gt, init, pts, obs, _ = _pnp_scene(600, 0.3, seed=2)
        p = _PnPProblem(init, pts, obs, with_prior=True)
        roles = [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, TAU),
                 pycunls.RansacFactorBatchOptions(pycunls.RansacRole.always_on)]
        o = _ransac_options(roles)
        o.score_always_on = False  # the prior sits at the (wrong) initial guess
        minimizer = pycunls.RansacGaussNewtonMinimizer(o)
        minimizer.minimize(stream, p.problem)
        assert minimizer.inlier_mask(0).shape == (600,)
        with pytest.raises(RuntimeError):
            minimizer.inlier_mask(1)

    def test_same_seed_is_reproducible(self, stream):
        _, init, pts, obs, _ = _pnp_scene(500, 0.4, seed=4)
        poses = []
        for _ in range(2):
            p = _PnPProblem(init, pts, obs)
            pycunls.RansacGaussNewtonMinimizer(_ransac_options()).minimize(stream, p.problem)
            poses.append(p.pose())
        assert np.array_equal(poses[0], poses[1])

    def test_errors(self, stream):
        _, init, pts, obs, _ = _pnp_scene(100, 0.0, seed=5)
        p = _PnPProblem(init, pts, obs)
        minimizer = pycunls.RansacGaussNewtonMinimizer(_ransac_options())
        with pytest.raises(RuntimeError):
            minimizer.inlier_mask(0)  # before any run
        with pytest.raises(RuntimeError):
            minimizer.inlier_mask(5)  # no such residual batch
        bad = _ransac_options()
        bad.factor_batches = []
        bad.hypotheses_per_round = 0
        with pytest.raises(ValueError):
            pycunls.RansacGaussNewtonMinimizer(bad).minimize(stream, p.problem)


# ── Custom factor and state under RANSAC ───────────────────────────────────
#
# Line fit y = a * x + b with a custom 2D state (Euclidean Plus written by
# hand) and a custom factor. The kernels follow the item / replica contract:
#   * evaluate: item t reads measurement f(t) = factor_ids[t] (or t % N),
#     its own state pointer state_pointers[t], and writes row t;
#   * plus: process num_replicas * num_blocks blocks.

class _StreamHandle:
    """Exposes a raw cudaStream_t handle through the CUDA stream protocol."""

    def __init__(self, handle):
        self.handle = handle

    def __cuda_stream__(self):
        return (0, self.handle)


def cupy_stream(handle):
    """CuPy stream wrapping cuNLS's cudaStream_t (use as a context manager)."""
    if hasattr(cp.cuda.Stream, "from_external"):  # CuPy >= 14
        return cp.cuda.Stream.from_external(_StreamHandle(handle))
    return cp.cuda.ExternalStream(handle)


_line_kernel = cp.RawKernel(r"""
extern "C" __global__
void line(const float* xs, const float* ys, const int* factor_ids, int num_factors,
          const unsigned long long* state_ptrs, float* res, float* jac, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  int f = factor_ids ? factor_ids[t] : t % num_factors;   // measurement of item t
  const float* ab = (const float*)state_ptrs[t];           // state of item t
  res[t] = ab[0] * xs[f] + ab[1] - ys[f];                  // row t
  if (jac) { jac[2 * t] = xs[f]; jac[2 * t + 1] = 1.f; }
}
""", "line")

_plus_kernel = cp.RawKernel(r"""
extern "C" __global__
void plus(const float* x, const float* d, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + d[i];
}
""", "plus")


class LineState(pycunls.CustomStateBatch):
    def __init__(self, data):
        super().__init__(data, 2, 2, 1)

    def plus(self, x_ptr, delta_ptr, out_ptr, stream_handle, num_replicas):
        n = 2 * self.num_state_blocks * num_replicas  # every replica, every float
        with cupy_stream(stream_handle):
            _plus_kernel(((n + 127) // 128,), (128,),
                         (cp.uint64(x_ptr), cp.uint64(delta_ptr), cp.uint64(out_ptr),
                          cp.int32(n)))


class LineFactor(pycunls.CustomFactorBatch):
    def __init__(self, xs, ys):
        super().__init__(1, [2], len(xs))
        self.xs, self.ys = xs, ys

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids  # always the actual item count in Python
        with cupy_stream(stream_handle):
            _line_kernel(((n + 127) // 128,), (128,),
                         (self.xs, self.ys, cp.uint64(factor_ids_ptr), cp.int32(self.num_factors),
                          cp.uint64(sp_ptr), cp.uint64(res_ptr), cp.uint64(jac_ptr),
                          cp.int32(n)))
        return True


class TestCustomTypesUnderRansac:
    def test_line_fit_with_outliers(self, stream):
        rng = np.random.default_rng(7)
        n = 400
        xs = rng.uniform(-5, 5, n).astype(np.float32)
        ys = (1.5 * xs - 2.0 + rng.normal(0, 0.01, n)).astype(np.float32)
        out = rng.uniform(size=n) < 0.6
        ys[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(1, 10, out.sum())

        ab = cp.zeros(2, dtype=cp.float32)
        state = LineState(ab)
        state.set_num_state_blocks(state.capacity, state.const_capacity)
        factor = LineFactor(cp.asarray(xs), cp.asarray(ys))
        factor.set_num_factors(factor.capacity)
        problem = pycunls.Problem()
        problem.add_state_batch(state)
        problem.add_factor_batch(factor, [state.state_block_device_ptr(0)] * n)

        o = pycunls.RansacMinimizerOptions()
        o.default_inlier_threshold = 0.05
        minimizer = pycunls.RansacGaussNewtonMinimizer(o)
        summary = minimizer.minimize(stream, problem)
        a, b = cp.asnumpy(ab)
        assert abs(a - 1.5) < 0.01 and abs(b + 2.0) < 0.01
        mask = minimizer.inlier_mask(0)
        assert int(np.sum((mask == 1) & out)) == 0
        assert summary.num_inliers >= 0.95 * np.sum(~out)
