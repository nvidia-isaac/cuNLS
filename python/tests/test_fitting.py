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

"""pycunls.fit: one-call model fitting (array and Warp models, batches, losses, bounds, starts)."""

import json

import numpy as np
import pytest
import warp as wp

import pycunls

wp.config.quiet = True


def decay(x, p):
    return p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2]


@wp.func
def decay_wp(x: float, p: wp.vec3) -> float:
    return p[0] * wp.exp(-p[1] * x) + p[2]


def line(x, p):
    return p[:, 0] * x + p[:, 1]


def _decay_data(seed=0, n=40, sigma=0.02):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n)
    return x, 2.0 * np.exp(-1.3 * x) + 0.5 + rng.normal(0, sigma, n)


def _reference(x, y, sigma):
    """Gauss-Newton in float64 on the host: the exact least-squares answer and its errors."""
    p = np.array([1.0, 1.0, 0.0])
    for _ in range(100):
        e = np.exp(-p[1] * x)
        r = (p[0] * e + p[2] - y) / sigma
        J = np.stack([e, -p[0] * x * e, np.ones_like(x)], 1) / sigma
        p = p - np.linalg.solve(J.T @ J, J.T @ r)
    cov = np.linalg.inv(J.T @ J)
    return p, np.sqrt(np.diag(cov))


@pytest.mark.parametrize("model", [decay, decay_wp], ids=["array", "warp"])
def test_single_fit_matches_the_least_squares_answer(model):
    x, y = _decay_data()
    res = pycunls.fit(model, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
    p_ref, err_ref = _reference(x, y, 0.02)
    assert res.converged and res.params.shape == (3,)
    np.testing.assert_allclose(res.params, p_ref, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(res.param_errors, err_ref, rtol=2e-2)
    assert 0.3 < res.reduced_chi2 < 2.0
    assert res.jacobian_mode == ("numeric" if model is decay else "autodiff")
    assert res.residuals.shape == (40,)


def test_batched_fits_are_one_solve_with_per_fit_results():
    rng = np.random.default_rng(1)
    truth = rng.uniform([1.0, 0.5, 0.0], [2.0, 2.0, 1.0], (500, 3))
    x = np.linspace(0, 4, 16)
    Y = truth[:, :1] * np.exp(-truth[:, 1:2] * x) + truth[:, 2:] + rng.normal(0, 1e-3, (500, 16))
    res = pycunls.fit(decay, x, Y, p0=[1.0, 1.0, 0.0], sigma=1e-3)
    assert res.params.shape == (500, 3) and res.converged.shape == (500,)
    assert res.all_converged
    assert np.abs(res.params - truth).max() < 0.05
    assert np.median(res.reduced_chi2) == pytest.approx(1.0, abs=0.3)


def test_per_fit_x_and_per_fit_p0():
    x, y = _decay_data()
    X = np.stack([x, x * 0.5])
    Y = np.stack([y, 2.0 * np.exp(-1.3 * X[1]) + 0.5])
    res = pycunls.fit(decay, X, Y, p0=[[1, 1, 0], [2, 1, 0.4]])
    assert res.params.shape == (2, 3)
    np.testing.assert_allclose(res.params[1], [2.0, 1.3, 0.5], atol=1e-3)


def test_nan_samples_are_skipped_for_ragged_batches():
    x, y = _decay_data()
    Y = np.stack([y, y])
    Y[1, 20:] = np.nan
    res = pycunls.fit(decay, x, Y, p0=[1, 1, 0])
    assert res.num_samples.tolist() == [40, 20]
    assert np.isnan(res.residuals[1, 25]) and np.isfinite(res.residuals[1, 5])
    assert res.all_converged


def test_robust_loss_ignores_outliers():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 300)
    y = 2.0 * x - 1.0 + rng.normal(0, 0.1, x.size)
    bad = rng.random(x.size) < 0.2
    y[bad] += rng.uniform(2, 10, bad.sum())
    plain = pycunls.fit(line, x, y, p0=[0, 0])
    robust = pycunls.fit(line, x, y, p0=[0, 0], sigma=0.1, loss="cauchy", loss_scale=2.0)
    assert abs(plain.params[1] + 1.0) > 0.3
    np.testing.assert_allclose(robust.params, [2.0, -1.0], atol=0.05)


def test_bounds_hold_and_the_fit_converges():
    x, y = _decay_data()
    res = pycunls.fit(decay, x, y, p0=[1, 1, 0], bounds=([0, 0, 0.6], [10, 10, 10]))
    assert res.converged
    assert res.params[2] == pytest.approx(0.6, abs=1e-6)


def test_starts_escape_a_local_minimum():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 100)
    y = 1.5 * np.sin(3.7 * x) + rng.normal(0, 0.01, x.size)

    def sine(x, p):
        return p[:, 0] * np.sin(p[:, 1] * x)

    single = pycunls.fit(sine, x, y, p0=[1, 1])
    assert abs(single.params[1] - 3.7) > 0.5  # a local minimum
    multi = pycunls.fit(sine, x, y, p0=[1, 1], bounds=([1, 0.5], [2, 5]), starts=32)
    np.testing.assert_allclose(multi.params, [1.5, 3.7], atol=1e-2)
    assert 0 <= multi.start_index < 32
    explicit = pycunls.fit(sine, x, y, p0=[1, 1], starts=[[1, 1], [1.4, 3.6]])
    assert explicit.start_index == 1


def test_vector_output_model():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 6, 50)
    pts = np.stack([1 + 2 * np.cos(t), -1 + 2 * np.sin(t)], 1) + rng.normal(0, 0.01, (50, 2))

    def circle(t, p):
        return np.stack([p[:, 0] + p[:, 2] * np.cos(t), p[:, 1] + p[:, 2] * np.sin(t)], axis=1)

    res = pycunls.fit(circle, t, pts, p0=[0, 0, 1])
    np.testing.assert_allclose(res.params, [1, -1, 2], atol=0.01)
    assert res.residuals.shape == (50, 2)


def test_wrong_model_shows_in_reduced_chi2():
    x, _ = _decay_data()
    rng = np.random.default_rng(5)
    y = 1.0 * np.exp(-0.5 * x) + 1.0 * np.exp(-3.0 * x) + rng.normal(0, 0.01, x.size)
    res = pycunls.fit(decay, x, y, p0=[1, 1, 0], sigma=0.01)
    assert res.converged and res.reduced_chi2 > 5


def test_report_and_to_dict():
    x, y = _decay_data()
    res = pycunls.fit(decay, x, y, p0=[1, 1, 0], sigma=0.02)
    assert res.report().startswith("FitResult: converged")
    d = res.to_dict()
    json.dumps(d)
    assert d["converged"] is True and len(d["params"]) == 3


def test_errors_explain_the_mistake():
    x, y = _decay_data()
    with pytest.raises(ValueError, match="mixes rows"):
        pycunls.fit(lambda x, p: p[:, 0] * x - np.mean(x * p[:, 1]), x, y, p0=[1, 1])
    with pytest.raises(ValueError, match="one prediction per row"):
        pycunls.fit(lambda x, p: np.sum(p), x, y, p0=[1, 1])
    with pytest.raises(ValueError, match="up to 12 parameters"):
        pycunls.fit(decay, x, y, p0=np.zeros(13))
    with pytest.raises(ValueError, match="starts=K needs finite bounds"):
        pycunls.fit(decay, x, y, p0=[1, 1, 0], starts=4)
    with pytest.raises(ValueError, match="loss must be one of"):
        pycunls.fit(decay, x, y, p0=[1, 1, 0], loss="l1")
