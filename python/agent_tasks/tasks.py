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

"""Agent tasks: a hidden simulator, the data an agent sees, and a checker.

A task is what an agent is asked to do with cuNLS: a statement (``prompt``), the data it gets
(``make(seed)`` returns the arrays and the hidden truth), and a ``check`` that scores its answer (a
JSON-like dict) against the truth. Each task has a ``reference`` solution written with the
documented API and a ``naive`` one that falls into the task's trap; the tests require that every
reference passes and every naive answer fails, so the suite measures what an agent has to know,
not what is easy.

Use it to evaluate changes to the library or its docs: give an agent the prompt and the data
directory, let it solve the task, and run ``python -m agent_tasks check TASK answer.json --seed
N``. The pass rate over agents and seeds, with and without a change, says whether it helps (see
``scripts/run_agent_eval.sh``).
"""

import dataclasses
from typing import Any, Callable, Dict, Tuple

import numpy as np

Arrays = Dict[str, np.ndarray]
Truth = Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Task:
    """``make(seed) -> (data, truth)``; ``check(answer, truth) -> (passed, message)``;
    ``reference(data) -> answer`` and ``naive(data) -> answer``. ``prompt`` states the goal and the
    answer format; the model family is named, the parameter values and the traps are not."""
    name: str
    prompt: str
    make: Callable[[int], Tuple[Arrays, Truth]]
    check: Callable[[Dict, Truth], Tuple[bool, str]]
    reference: Callable[[Arrays], Dict]
    naive: Callable[[Arrays], Dict]


# ---------------------------------------------------------------------------------------------
# Checking helpers
# ---------------------------------------------------------------------------------------------

def _vector(answer: Dict, key: str, shape) -> Tuple[Any, str]:
    """The answer's ``key`` as a float array of ``shape``, or a message saying what is wrong."""
    try:
        values = np.asarray(answer[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None, f"the answer needs '{key}': numbers of shape {shape}"
    if values.shape != tuple(shape):
        return None, f"'{key}' must have shape {tuple(shape)}, got {values.shape}"
    if not np.all(np.isfinite(values)):
        return None, f"'{key}' must be finite numbers"
    return values, ""


def _close(answer: Dict, truth: Truth, rtol: float, atol: float = 0.0) -> Tuple[bool, str]:
    values, error = _vector(answer, "params", np.shape(truth["params"]))
    if values is None:
        return False, error
    expected = np.asarray(truth["params"], dtype=np.float64)
    tol = atol + rtol * np.abs(expected)
    bad = np.abs(values - expected) > tol
    if bad.any():
        i = int(np.argmax(np.abs(values - expected) / np.maximum(tol, 1e-30)))
        return False, (f"params[{i}] = {values.flat[i]:.4g}, expected within "
                       f"{tol.flat[i]:.3g} of the truth")
    return True, "ok"


def _decay(x, a, b, c):
    return a * np.exp(-b * x) + c


# ---------------------------------------------------------------------------------------------
# decay_curve: basic fitting
# ---------------------------------------------------------------------------------------------

def _make_decay(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform([1.0, 0.5, 0.0], [3.0, 2.0, 1.0])
    x = np.linspace(0, 5, 60).astype(np.float32)
    y = (_decay(x, *p) + rng.normal(0, 0.02, x.size)).astype(np.float32)
    return {"x": x, "y": y}, {"params": p.tolist()}


def _reference_decay(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"], p0=[1.0, 1.0, 0.0], sigma=0.02)
    return {"params": res.params.tolist()}


def _naive_decay(data):
    # A log-linear fit that ignores the offset.
    y = np.clip(data["y"], 1e-3, None)
    b, log_a = np.polyfit(data["x"], np.log(y), 1)
    return {"params": [float(np.exp(log_a)), float(-b), 0.0]}


# ---------------------------------------------------------------------------------------------
# sine_frequency: a local minimum (multiple starts)
# ---------------------------------------------------------------------------------------------

def _make_sine(seed):
    rng = np.random.default_rng(seed)
    p = np.array([rng.uniform(1.0, 2.0), rng.uniform(2.5, 4.5)])
    x = np.linspace(0, 10, 200).astype(np.float32)
    y = (p[0] * np.sin(p[1] * x) + rng.normal(0, 0.01, x.size)).astype(np.float32)
    return {"x": x, "y": y}, {"params": p.tolist()}


def _reference_sine(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.sin(p[:, 1] * x), data["x"], data["y"],
                      p0=[1.5, 1.0], bounds=([1.0, 0.5], [2.0, 5.0]), starts=64)
    return {"params": res.params.tolist()}


def _naive_sine(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.sin(p[:, 1] * x), data["x"], data["y"],
                      p0=[1.5, 1.0])
    return {"params": res.params.tolist()}


# ---------------------------------------------------------------------------------------------
# per_pixel_decay: thousands of independent fits (batching)
# ---------------------------------------------------------------------------------------------

def _make_pixels(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform([1.0, 0.5, 0.0], [2.0, 2.0, 0.5], (3000, 3))
    x = np.linspace(0, 4, 16).astype(np.float32)
    y = p[:, :1] * np.exp(-p[:, 1:2] * x) + p[:, 2:] + rng.normal(0, 0.005, (3000, 16))
    return {"x": x, "y": y.astype(np.float32)}, {"params": p.tolist()}


def _check_pixels(answer, truth):
    expected = np.asarray(truth["params"])
    values, error = _vector(answer, "params", expected.shape)
    if values is None:
        return False, error
    good = np.all(np.abs(values - expected) <= 0.05 + 0.05 * np.abs(expected), axis=1)
    if good.mean() < 0.99:
        return False, f"{100 * good.mean():.1f}% of the pixels within tolerance (need 99%)"
    return True, "ok"


def _reference_pixels(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"], p0=[1.0, 1.0, 0.0])
    return {"params": res.params.tolist()}


def _naive_pixels(data):
    # One fit of the mean curve for every pixel.
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"].mean(axis=0), p0=[1.0, 1.0, 0.0])
    return {"params": np.tile(res.params, (data["y"].shape[0], 1)).tolist()}


# ---------------------------------------------------------------------------------------------
# line_with_outliers: robust loss
# ---------------------------------------------------------------------------------------------

def _make_outliers(seed):
    rng = np.random.default_rng(seed)
    p = np.array([rng.uniform(-3, 3), rng.uniform(-5, 5)])
    x = np.sort(rng.uniform(0, 10, 300)).astype(np.float32)
    y = p[0] * x + p[1] + rng.normal(0, 0.1, x.size)
    bad = rng.random(x.size) < 0.2
    y[bad] += rng.uniform(2, 20, bad.sum())
    return {"x": x, "y": y.astype(np.float32)}, {"params": p.tolist()}


def _reference_outliers(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * x + p[:, 1], data["x"], data["y"], p0=[0.0, 0.0],
                      sigma=0.1, loss="cauchy", loss_scale=2.0)
    return {"params": res.params.tolist()}


def _naive_outliers(data):
    m, q = np.polyfit(data["x"], data["y"], 1)
    return {"params": [float(m), float(q)]}


# ---------------------------------------------------------------------------------------------
# inadequate_model: residual statistics decide, not "the cost went down"
# ---------------------------------------------------------------------------------------------

def _make_adequacy(seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 5, 60).astype(np.float32)
    adequate = bool(rng.random() < 0.5)
    if adequate:
        y = _decay(x, *rng.uniform([1, 0.5, 0], [2, 2, 1]))
    else:
        y = (rng.uniform(0.8, 1.2) * np.exp(-rng.uniform(0.3, 0.6) * x)
             + rng.uniform(0.8, 1.2) * np.exp(-rng.uniform(2.5, 4.0) * x))
    y = (y + rng.normal(0, 0.01, x.size)).astype(np.float32)
    return {"x": x, "y": y}, {"adequate": adequate}


def _check_adequacy(answer, truth):
    value = answer.get("model_adequate") if isinstance(answer, dict) else None
    if not isinstance(value, bool):
        return False, "the answer needs 'model_adequate': true or false"
    if value != truth["adequate"]:
        return False, f"the model {'does' if truth['adequate'] else 'does not'} explain the data"
    return True, "ok"


def _reference_adequacy(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"], p0=[1.0, 1.0, 0.0], sigma=0.01)
    return {"model_adequate": bool(res.reduced_chi2 < 2.0)}


def _naive_adequacy(data):
    return {"model_adequate": True}  # "it converged, so it fits"


# ---------------------------------------------------------------------------------------------
# oscillator_ode: a model given as an ODE
# ---------------------------------------------------------------------------------------------

def _oscillator(t, c, k):
    w = np.sqrt(np.maximum(k - 0.25 * c * c, 1e-12))
    return np.exp(-0.5 * c * t) * (np.cos(w * t) + 0.5 * c / w * np.sin(w * t))


def _make_oscillator(seed):
    rng = np.random.default_rng(seed)
    c, k = rng.uniform(0.1, 0.5), rng.uniform(4.0, 10.0)
    t = (0.05 * np.arange(1, 201)).astype(np.float32)
    y = (_oscillator(t, c, k) + rng.normal(0, 0.01, t.size)).astype(np.float32)
    return {"y": y}, {"params": [c, k]}


def _reference_oscillator(data):
    import pycunls

    t = 0.05 * np.arange(1, data["y"].size + 1)

    def model(t, p):
        c, k = p[:, 0], p[:, 1]
        w = np.sqrt(np.maximum(k - 0.25 * c * c, 1e-12))
        return np.exp(-0.5 * c * t) * (np.cos(w * t) + 0.5 * c / w * np.sin(w * t))

    # The frequency makes the cost multi-modal: several starts in a plausible range.
    res = pycunls.fit(model, t, data["y"], p0=[0.3, 5.0], bounds=([0.0, 0.5], [2.0, 50.0]),
                      starts=64)
    return {"params": res.params.tolist()}


def _naive_oscillator(data):
    # The right model from a single start: the frequency lands in a local minimum.
    import pycunls

    t = 0.05 * np.arange(1, data["y"].size + 1)

    def model(t, p):
        c, k = p[:, 0], p[:, 1]
        w = np.sqrt(np.maximum(k - 0.25 * c * c, 1e-12))
        return np.exp(-0.5 * c * t) * (np.cos(w * t) + 0.5 * c / w * np.sin(w * t))

    res = pycunls.fit(model, t, data["y"], p0=[0.3, 1.0])
    return {"params": res.params.tolist()}


# ---------------------------------------------------------------------------------------------
# pose_graph: SE(3) layout, the between convention, and the gauge
# ---------------------------------------------------------------------------------------------

def _exp_se3(w, v):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        R, V = np.eye(3) + K, np.eye(3)
    else:
        R = np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K
        V = np.eye(3) + (1 - np.cos(th)) / th ** 2 * K + (th - np.sin(th)) / th ** 3 * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, V @ v
    return T


def _make_pose_graph(seed):
    rng = np.random.default_rng(seed)
    n = 60
    truth = [np.eye(4)]
    for _ in range(n - 1):
        truth.append(truth[-1] @ _exp_se3(rng.normal(0, 0.1, 3), np.array([1.0, 0, 0])
                                          + rng.normal(0, 0.1, 3)))
    edges = [(i, i + 1) for i in range(n - 1)]
    edges += [tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(30)]
    meas = []
    for i, j in edges:
        rel = np.linalg.inv(truth[i]) @ truth[j]
        meas.append(rel @ _exp_se3(rng.normal(0, 0.002, 3), rng.normal(0, 0.005, 3)))
    # Odometry with drift for the initial guess; pose 0 is exact.
    init = [truth[0]]
    for i in range(n - 1):
        init.append(init[-1] @ meas[i] @ _exp_se3(rng.normal(0, 0.02, 3), rng.normal(0, 0.05, 3)))
    data = {"initial_poses": np.stack(init).astype(np.float32),
            "edges": np.asarray(edges, np.int32),
            "relative_poses": np.stack(meas).astype(np.float32)}
    return data, {"positions": np.stack(truth)[:, :3, 3].tolist()}


def _check_pose_graph(answer, truth):
    expected = np.asarray(truth["positions"])
    values, error = _vector(answer, "poses", (expected.shape[0], 4, 4))
    if values is None:
        return False, error
    rmse = float(np.sqrt(np.mean(np.sum((values[:, :3, 3] - expected) ** 2, axis=1))))
    if rmse > 0.1:
        return False, f"position RMSE {rmse:.3f} (need <= 0.1; pose 0 is the fixed reference)"
    return True, f"ok (position RMSE {rmse:.4f})"


def _reference_pose_graph(data):
    import pycunls

    # cuNLS between factors measure Delta = T_j^-1 T_i, the inverse of T_i^-1 T_j; pose 0 fixes
    # the gauge.
    deltas = np.linalg.inv(data["relative_poses"]).astype(np.float32)
    result = pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=data["initial_poses"],
                           connect=data["edges"], constant=[0])
    assert result.converged, result.message
    return {"poses": result.states.tolist()}


def _naive_pose_graph(data):
    return {"poses": data["initial_poses"].tolist()}  # odometry only


# ---------------------------------------------------------------------------------------------
# bounded_fit: a physical constraint (bounds)
# ---------------------------------------------------------------------------------------------

def _unconstrained_decay_fit(x, y, p):
    """Gauss-Newton for a*exp(-b*x) + c in float64 (to check the trap when making the data)."""
    p = np.array(p, dtype=np.float64)
    with np.errstate(all="ignore"):
        for _ in range(50):
            e = np.exp(-p[1] * x)
            J = np.stack([e, -p[0] * x * e, np.ones_like(x)], 1)
            if not np.all(np.isfinite(J)):
                return np.full(3, np.nan)  # diverged
            try:
                step = np.linalg.lstsq(J, -(p[0] * e + p[2] - y), rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.full(3, np.nan)
            p += step
    return p


def _make_bounded(seed):
    rng = np.random.default_rng(seed)
    p = np.array([rng.uniform(0.8, 1.2), rng.uniform(0.6, 1.0), 0.0])
    x = np.linspace(0, 2, 10).astype(np.float32)
    # Draw noise until the unconstrained offset comes out clearly negative.
    for _ in range(1000):
        y = _decay(x, *p) + rng.normal(0, 0.05, x.size)
        free = _unconstrained_decay_fit(x.astype(np.float64), y, [1.0, 1.0, 0.1])
        if np.all(np.isfinite(free)) and free[2] < -0.03:
            break
    else:
        raise RuntimeError(f"bounded_fit: no noise draw for seed {seed} gives a negative "
                           "unconstrained offset; the task would have no trap")
    return {"x": x, "y": y.astype(np.float32)}, {"params": p.tolist()}


def _check_bounded(answer, truth):
    values, error = _vector(answer, "params", (3,))
    if values is None:
        return False, error
    if values[2] < 0:
        return False, f"the offset c = {values[2]:.4g} is negative; concentrations are >= 0"
    return _close(answer, truth, rtol=0.25, atol=0.05)


def _reference_bounded(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"], p0=[1.0, 1.0, 0.1], bounds=([0, 0, 0], [np.inf] * 3))
    return {"params": res.params.tolist()}


def _naive_bounded(data):
    import pycunls

    res = pycunls.fit(lambda x, p: p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2], data["x"],
                      data["y"], p0=[1.0, 1.0, 0.1])
    return {"params": res.params.tolist()}


TASKS = {t.name: t for t in [
    Task("decay_curve",
         "data.npz has x and y: y = a*exp(-b*x) + c plus Gaussian noise (sigma 0.02). Estimate "
         "[a, b, c]. Answer: {\"params\": [a, b, c]}.",
         _make_decay, lambda a, t: _close(a, t, rtol=0.03, atol=0.01), _reference_decay,
         _naive_decay),
    Task("sine_frequency",
         "data.npz has x and y: y = a*sin(w*x) plus noise (sigma 0.01), with a in [1, 2] and w in "
         "[0.5, 5]. Estimate [a, w]. Answer: {\"params\": [a, w]}.",
         _make_sine, lambda a, t: _close(a, t, rtol=0.01), _reference_sine, _naive_sine),
    Task("per_pixel_decay",
         "data.npz has x of shape [16] and y of shape [3000, 16]: every row is one pixel, y = "
         "a*exp(-b*x) + c plus noise (sigma 0.005). Estimate [a, b, c] for every pixel. Answer: "
         "{\"params\": [[a, b, c], ...]} with 3000 rows.",
         _make_pixels, _check_pixels, _reference_pixels, _naive_pixels),
    Task("line_with_outliers",
         "data.npz has x and y: y = m*x + q plus noise (sigma 0.1), but about 20% of the y are "
         "corrupted by large positive errors. Estimate [m, q] of the underlying line. Answer: "
         "{\"params\": [m, q]}.",
         _make_outliers, lambda a, t: _close(a, t, rtol=0.0, atol=0.05), _reference_outliers,
         _naive_outliers),
    Task("inadequate_model",
         "data.npz has x and y (noise sigma 0.01). Fit y = a*exp(-b*x) + c and decide whether this "
         "model explains the data. Answer: {\"model_adequate\": true or false}.",
         _make_adequacy, _check_adequacy, _reference_adequacy, _naive_adequacy),
    Task("oscillator_ode",
         "data.npz has y: the position of a damped oscillator x'' = -k*x - c*x' sampled every 0.05 "
         "time units from t = 0.05, with x(0) = 1 and x'(0) = 0, plus noise (sigma 0.01). "
         "Estimate [c, k]. Answer: {\"params\": [c, k]}.",
         _make_oscillator, lambda a, t: _close(a, t, rtol=0.05), _reference_oscillator,
         _naive_oscillator),
    Task("pose_graph",
         "data.npz has initial_poses [60, 4, 4] (world_from_body, odometry with drift), edges "
         "[89, 2] (pairs i, j) and relative_poses [89, 4, 4], the noisy measurements of "
         "inv(T_i) @ T_j for each edge. Pose 0 is exact. Optimize the pose graph with cuNLS. "
         "Answer: {\"poses\": [60 4x4 matrices]}.",
         _make_pose_graph, _check_pose_graph, _reference_pose_graph, _naive_pose_graph),
    Task("bounded_fit",
         "data.npz has x and y: a concentration y = a*exp(-b*x) + c measured with noise (sigma "
         "0.05). All of a, b, c are physically non-negative. Estimate [a, b, c]. Answer: "
         "{\"params\": [a, b, c]}.",
         _make_bounded, _check_bounded, _reference_bounded, _naive_bounded),
]}
