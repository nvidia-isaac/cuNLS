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

"""Model fitting in one call: :func:`fit` finds the parameters ``p`` of ``y ~ model(x, p)``.

One curve, or thousands of independent fits solved together on the GPU, with optional measurement
errors, robust losses, bounds and multiple starts. The result says per fit whether it converged
and how well the model explains the data (reduced chi-square, standard errors).

The model is a **Warp function** of one sample, ``@wp.func def model(x: float, p: wp.vec3) ->
float``: ``x`` is a scalar or a ``wp.vecN`` (one sample's inputs), ``p`` a ``wp.vecP`` (a scalar
for one parameter), and the result a scalar or a ``wp.vecM`` (one sample's prediction). The
Jacobian comes from Warp's automatic differentiation, checked against central differences at
``p0`` (``jacobian="auto"``). Warp differentiates loops only up to ``wp.config.max_unroll``
iterations (16 by default; e.g. a 32-step integrator): "auto" detects a wrong gradient and falls
back to central differences, ``jacobian="autodiff"`` does not.

Example::

    import numpy as np, warp as wp, pycunls

    @wp.func
    def decay(x: float, p: wp.vec3) -> float:   # y = a exp(-b x) + c
        return p[0] * wp.exp(-p[1] * x) + p[2]

    res = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
    print(res.report())                     # params, errors, reduced chi2, convergence
    res.params, res.param_errors, res.converged

    # 3000 independent fits (one per row of y) in one solve:
    res = pycunls.fit(decay, x, Y, p0=[1.0, 1.0, 0.0])   # Y: (3000, n); res.params: (3000, 3)

See ``python/examples/model_fitting.py`` for a tour.
"""


import collections
import dataclasses
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from pycunls import _pycunls_core as _core
from pycunls._arrays import to_numpy, warp_stream
from pycunls._summary import _plain

_VECTOR_SIZES = (1, 2, 3, 4, 6, 12)
_MAX_PARAMETERS = 12
_FD_STEP = 3e-4   # relative central-difference step in float32


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

def _length(t) -> int:
    """1 for a scalar Warp type, N for a wp.vecN, 0 for anything else."""
    return 1 if t in (float, wp.float32) else int(getattr(t, "_length_", 0))


def _is_warp_function(model) -> bool:
    # warp's Function class; its module moved between Warp versions.
    return type(model).__name__ == "Function" and type(model).__module__.startswith("warp")


class _Workspace:
    """Device buffers of one evaluation of ``rows`` samples (reused while large enough)."""

    def __init__(self, model: "_WarpModel", rows: int):
        m, num_params, device = model.m, model.num_params, model.device
        self.rows = rows
        self.sample = wp.empty(rows, dtype=wp.int32, device=device)
        self.params = wp.empty(rows, dtype=model.p_type, device=device, requires_grad=True)
        self.pred = wp.empty((rows, m), dtype=float, device=device, requires_grad=True)
        self.jac = wp.empty((rows, m, num_params), dtype=float, device=device)
        seeds = np.zeros((m, rows, m), np.float32)
        for r in range(m):
            seeds[r, :, r] = 1.0
        self.seeds = [wp.array(seeds[r], dtype=float, device=device) for r in range(m)]


class _WarpModel:
    """A ``@wp.func`` model(x, p) evaluated per sample, with autodiff or central-difference
    Jacobians. All kernels run on the device's current Warp stream."""

    def __init__(self, fn, jacobian: str, device):
        types = list(fn.input_types.values())
        if len(types) != 2:
            raise TypeError(f"fit: the Warp model takes {len(types)} arguments (expected 2: "
                            "model(x, p)). Fix: declare it as @wp.func def model(x: float, "
                            "p: wp.vec3) -> float.")
        self.fn = fn
        self.device = device
        self.x_type, self.p_type = types
        self.y_type = getattr(fn, "func", None) and fn.func.__annotations__.get("return")
        self.y_type = self.y_type or fn.value_type or float
        self.num_params = _length(self.p_type)
        self.x_dim = _length(self.x_type)
        self.m = _length(self.y_type)
        if not (self.num_params and self.x_dim and self.m):
            raise TypeError("fit: unsupported Warp model types (got x: "
                            f"{self.x_type}, p: {self.p_type}, result: {self.y_type}; expected "
                            "x float or wp.vecN, p wp.vecP (e.g. wp.vec3 or "
                            "wp.types.vector(length=5, dtype=float)), result float or wp.vecM). "
                            "Fix: annotate the arguments and the return type with those types.")
        self.mode = jacobian
        self._kernels = self._make_kernels()
        self._workspace: Optional[_Workspace] = None
        self._retired = []   # buffers replaced while kernels may still read them

    def _make_kernels(self):
        fn, XT, PT = self.fn, self.x_type, self.p_type
        P, M = self.num_params, self.m

        @wp.kernel(module="unique")
        def gather(pointers: wp.array(dtype=wp.uint64), ids: wp.array(dtype=wp.int32),
                   use_ids: int, num_factors: int, width: int, params: wp.array(dtype=PT),
                   sample: wp.array(dtype=wp.int32)):
            # Item t: the parameters its state pointer addresses, and its sample.
            t = wp.tid()
            a = wp.array(ptr=pointers[t], shape=(width,), dtype=float)
            if wp.static(P == 1):
                params[t] = a[0]
            else:
                v = PT()
                for k in range(P):
                    v[k] = a[k]
                params[t] = v
            if use_ids:
                sample[t] = ids[t]
            else:
                sample[t] = t % num_factors

        @wp.kernel(module="unique")
        def predict(x: wp.array(dtype=XT), sample: wp.array(dtype=wp.int32),
                    params: wp.array(dtype=PT), pred: wp.array2d(dtype=float)):
            t = wp.tid()
            y = fn(x[sample[t]], params[t])
            if wp.static(M == 1):
                pred[t, 0] = y
            else:
                for i in range(M):
                    pred[t, i] = y[i]

        @wp.kernel(module="unique")
        def central_differences(x: wp.array(dtype=XT), sample: wp.array(dtype=wp.int32),
                                params: wp.array(dtype=PT), rel: float,
                                jac: wp.array3d(dtype=float)):
            t = wp.tid()
            xs = x[sample[t]]
            p = params[t]
            for k in range(P):
                if wp.static(P == 1):
                    h = rel * wp.max(1.0, wp.abs(p))
                    d = (fn(xs, p + h) - fn(xs, p - h)) / (2.0 * h)
                else:
                    h = rel * wp.max(1.0, wp.abs(p[k]))
                    e = PT()
                    e[k] = h
                    d = (fn(xs, p + e) - fn(xs, p - e)) / (2.0 * h)
                if wp.static(M == 1):
                    jac[t, 0, k] = d
                else:
                    for i in range(M):
                        jac[t, i, k] = d[i]

        @wp.kernel(module="unique")
        def store_gradient(grad: wp.array(dtype=PT), row: int, jac: wp.array3d(dtype=float)):
            t = wp.tid()
            g = grad[t]
            if wp.static(P == 1):
                jac[t, row, 0] = g
            else:
                for k in range(P):
                    jac[t, row, k] = g[k]

        return dict(gather=gather, predict=predict, central_differences=central_differences,
                    store_gradient=store_gradient)

    def workspace(self, rows: int) -> _Workspace:
        """Buffers for ``rows`` samples; grows (never shrinks) and keeps replaced buffers alive."""
        if self._workspace is None or self._workspace.rows < rows:
            if self._workspace is not None:
                self._retired.append(self._workspace)
            self._workspace = _Workspace(self, rows)
        return self._workspace

    def gather(self, pointers: wp.array, ids: Optional[wp.array], num_factors: int, width: int,
               rows: int) -> _Workspace:
        """Reads each item's parameters through its state pointer and its sample index."""
        ws = self.workspace(rows)
        wp.launch(self._kernels["gather"], dim=rows, device=self.device,
                  inputs=[pointers, ids, int(ids is not None), num_factors, width, ws.params,
                          ws.sample])
        return ws

    def evaluate(self, x: wp.array, ws: _Workspace, rows: int, mode: Optional[str]) -> None:
        """``ws.pred`` (rows, m) and, unless ``mode`` is None, ``ws.jac`` (rows, m, P) at
        ``ws.params`` for samples ``ws.sample``; ``mode`` is "autodiff" or "numeric"."""
        k = self._kernels
        inputs = [x, ws.sample, ws.params]
        if mode != "autodiff":
            wp.launch(k["predict"], dim=rows, inputs=inputs + [ws.pred], device=self.device)
            if mode == "numeric":
                wp.launch(k["central_differences"], dim=rows, device=self.device,
                          inputs=inputs + [_FD_STEP, ws.jac])
            return
        tape = wp.Tape()
        with tape:
            wp.launch(k["predict"], dim=rows, inputs=inputs + [ws.pred], device=self.device)
        for r in range(self.m):
            tape.backward(grads={ws.pred: ws.seeds[r]})
            wp.launch(k["store_gradient"], dim=rows, inputs=[ws.params.grad, r, ws.jac],
                      device=self.device)
            tape.zero()

    def choose_jacobian(self, x: wp.array, p0_rows: np.ndarray) -> None:
        """``jacobian="auto"``: autodiff unless it disagrees with central differences at p0."""
        if self.mode != "auto":
            return
        rows = min(256, p0_rows.shape[0])
        ws = self.workspace(rows)
        wp.copy(ws.sample, wp.array(np.arange(rows, dtype=np.int32), device=self.device))
        wp.copy(ws.params, wp.array(np.ascontiguousarray(p0_rows[:rows], np.float32),
                                    dtype=self.p_type, device=self.device), count=rows)
        self.evaluate(x, ws, rows, "autodiff")
        auto = ws.jac.numpy()[:rows].reshape(rows, -1)
        self.evaluate(x, ws, rows, "numeric")
        numeric = ws.jac.numpy()[:rows].reshape(rows, -1)
        # Rows where the model is not finite (a start outside its domain) say nothing.
        finite = np.all(np.isfinite(auto) & np.isfinite(numeric), axis=1)
        if not finite.any():
            self.mode = "autodiff"
            return
        auto, numeric = auto[finite], numeric[finite]
        scale = max(1.0, float(np.max(np.abs(numeric))))
        err = float(np.max(np.abs(auto - numeric))) / scale
        if err > 1e-2:
            warnings.warn(f"Warp autodiff Jacobian differs from finite differences at p0 "
                          f"(relative error {err:.3g}); using finite differences. Check the "
                          "model for in-place writes or data-dependent loops.", stacklevel=3)
            self.mode = "numeric"
        else:
            self.mode = "autodiff"


def _make_model(model, jacobian: str, device) -> _WarpModel:
    if not _is_warp_function(model):
        raise TypeError(
            f"fit: model must be a Warp function (got {type(model).__name__}). Fix: write it "
            "for one sample, e.g. @wp.func def model(x: float, p: wp.vec3) -> float: return "
            "p[0] * wp.exp(-p[1] * x) + p[2].")
    if jacobian not in ("auto", "autodiff", "numeric"):
        raise ValueError(f"fit: jacobian must be 'auto', 'autodiff' or 'numeric' (got "
                         f"{jacobian!r}).")
    return _WarpModel(model, jacobian, device)


# ---------------------------------------------------------------------------------------------
# The factor: one residual per sample
# ---------------------------------------------------------------------------------------------

@wp.kernel
def _write_residuals(pred: wp.array2d(dtype=float), jac: wp.array3d(dtype=float),
                     sample: wp.array(dtype=wp.int32), y: wp.array2d(dtype=float),
                     w: wp.array2d(dtype=float), num_params: int, want_jacobian: int,
                     residuals: wp.array2d(dtype=float), jacobians: wp.array3d(dtype=float)):
    # residual = w * (pred - y); samples with w = 0 (NaN y) contribute nothing.
    t = wp.tid()
    s = sample[t]
    for i in range(pred.shape[1]):
        wi = w[s, i]
        r = float(0.0)
        if wi > 0.0:
            r = (pred[t, i] - y[s, i]) * wi
        residuals[t, i] = r
        if want_jacobian:
            for k in range(jacobians.shape[2]):
                v = float(0.0)
                if wi > 0.0 and k < num_params:
                    v = jac[t, i, k] * wi
                jacobians[t, i, k] = v


class _SampleFactors(_core.CustomFactorBatch):
    """residual_t = w_t * (model(x_t, p_fit(t)) - y_t) for every sample t."""

    def __init__(self, model: _WarpModel, x: wp.array, y: wp.array, w: wp.array,
                 num_params: int, width: int):
        super().__init__(model.m, [width], y.shape[0])
        self.model, self.x, self.y, self.w = model, x, y, w
        self.num_params, self.width = num_params, width
        self._streams: Dict[int, wp.Stream] = {}

    def _stream(self, handle: int) -> wp.Stream:
        stream = self._streams.get(handle)
        if stream is None:
            stream = self._streams[handle] = warp_stream(handle)
        return stream

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, ids_ptr, num_ids):
        n = num_ids if num_ids else self.num_active_factors
        m, width, device = self.model.m, self.width, self.model.device

        def wrap(ptr, dtype, shape):
            return wp.array(ptr=ptr, dtype=dtype, shape=shape, device=device, copy=False)

        with wp.ScopedStream(self._stream(stream_handle), sync_enter=False):
            # Item t evaluates factor ids[t], or t % N without ids (FactorBatch.Evaluate).
            ids = wrap(ids_ptr, wp.int32, (n,)) if ids_ptr else None
            ws = self.model.gather(wrap(sp_ptr, wp.uint64, (n,)), ids, self.num_active_factors,
                                   width, n)
            self.model.evaluate(self.x, ws, n, self.model.mode if jac_ptr else None)
            wp.launch(_write_residuals, dim=n, device=device,
                      inputs=[ws.pred, ws.jac, ws.sample, self.y, self.w, self.num_params,
                              int(jac_ptr != 0), wrap(res_ptr, float, (n, m)),
                              wrap(jac_ptr, float, (n, m, width)) if jac_ptr else None])
        return True


# ---------------------------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class FitResult:
    """Result of :func:`fit`. Arrays are NumPy; batched fits have a leading fit axis.

    Attributes:
        params: Best-fit parameters, ``(P,)`` or ``(B, P)``.
        param_errors: One-sigma standard errors from the Jacobian at the solution,
            ``sqrt(diag((J^T J)^-1))``, multiplied by ``sqrt(reduced_chi2)`` when ``sigma`` was not
            given (as ``scipy.optimize.curve_fit``). NaN when the parameters are not determined.
        covariance: Parameter covariance, ``(P, P)`` or ``(B, P, P)``, same scaling.
        converged: Whether each fit stopped at a convergence criterion (bool or ``(B,)``).
        reasons: :class:`pycunls.TerminationReason` name of each fit (str or ``(B,)``), or
            ``"NoObservations"`` for a fit whose samples are all NaN.
        cost: ``0.5 * sum(r^2)`` of the weighted residuals (before a robust loss), per fit;
            NaN when the model is not finite at the parameters.
        reduced_chi2: ``2 * cost / (num_samples * m - P)`` per fit. With the true ``sigma`` it is
            about 1 when the model explains the data; much larger means the model does not fit
            (or ``sigma`` is too small).
        num_samples: Samples used per fit (NaN ``y`` are skipped).
        residuals: Weighted residuals ``(model - y) / sigma``, shaped like ``y`` (NaN where
            ``y`` was NaN).
        message: The solver's message (why it stopped, what to try).
        summary: The full solver summary (:class:`pycunls.MinimizerSummary` or the augmented
            Lagrangian's with bounds).
        jacobian_mode: How the Jacobian was computed ("numeric" or "autodiff").
        start_index: With ``starts``: the start that won, per fit.
    """
    params: np.ndarray
    param_errors: np.ndarray
    covariance: np.ndarray
    converged: Union[bool, np.ndarray]
    reasons: Union[str, np.ndarray]
    cost: Union[float, np.ndarray]
    reduced_chi2: Union[float, np.ndarray]
    num_samples: Union[int, np.ndarray]
    residuals: np.ndarray
    message: str
    summary: Any
    jacobian_mode: str
    start_index: Optional[Union[int, np.ndarray]] = None

    @property
    def all_converged(self) -> bool:
        """True when every fit converged."""
        return bool(np.all(self.converged))

    def to_dict(self) -> Dict:
        """JSON-ready dict (lists instead of arrays, NaN as None; residuals and covariance left
        out)."""
        return {k: _plain(getattr(self, k)) for k in
                ("params", "param_errors", "converged", "reasons", "cost", "reduced_chi2",
                 "num_samples", "message", "jacobian_mode", "start_index")}

    def report(self, max_fits: int = 5) -> str:
        """Readable summary: convergence, parameters with errors, reduced chi-square."""
        batched = np.ndim(self.params) == 2
        lines = []
        conv = np.atleast_1d(self.converged)
        head = "converged" if conv.all() else f"{int((~conv).sum())} of {conv.size} NOT CONVERGED"
        lines.append(f"FitResult: {head} ({self.jacobian_mode} Jacobian)")
        lines.append(f"  {self.message}")
        params = np.atleast_2d(self.params)
        errors = np.atleast_2d(self.param_errors)
        chi2 = np.atleast_1d(self.reduced_chi2)
        reasons = np.atleast_1d(self.reasons)
        shown = range(min(max_fits, params.shape[0]))
        for b in shown:
            label = f"  fit {b}: " if batched else "  "
            values = ", ".join(f"{v:.6g} ± {e:.2g}" for v, e in zip(params[b], errors[b]))
            lines.append(f"{label}p = [{values}], reduced chi2 {chi2[b]:.3g}, {reasons[b]}")
        if batched and params.shape[0] > max_fits:
            lines.append(f"  ... {params.shape[0] - max_fits} more fits; reduced chi2 median "
                         f"{np.nanmedian(chi2):.3g}, max {np.nanmax(chi2):.3g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------------------------

def _loss(loss, scale: float):
    if loss is None or loss == "linear":
        return None
    if isinstance(loss, _core.LossFunctionBatch):
        return loss
    k = float(scale)
    table = {
        "huber": lambda: _core.HuberLossFunctionBatch(k),
        "cauchy": lambda: _core.CauchyLossFunctionBatch(k * k, 1.0 / (k * k)),
        "soft_l1": lambda: _core.SoftLOneLossFunctionBatch(k * k, 1.0 / (k * k)),
        "tukey": lambda: _core.TukeyLossFunctionBatch(k),
    }
    if loss not in table:
        raise ValueError(f"loss must be one of {sorted(table)}, None, or a LossFunctionBatch; "
                         f"got {loss!r}")
    return table[loss]()


def fit(model: Callable, x, y, p0, *, sigma=None, loss: Optional[Union[str, Any]] = None,
        loss_scale: float = 1.0, bounds: Optional[Tuple[Any, Any]] = None,
        starts: Optional[Union[int, Sequence]] = None, method: str = "lm",
        max_iterations: int = 100, jacobian: str = "auto", seed: int = 0,
        stream: Optional[_core.CudaStream] = None) -> FitResult:
    """Fits ``y ~ model(x, p)``: one fit, or many independent fits solved together.

    Shapes (``n`` samples, ``P`` parameters, ``m`` model outputs, ``B`` fits):

    * one fit: ``y`` is ``(n,)`` (or ``(n, m)``), ``x`` is ``(n,)`` or ``(n, d)``, ``p0`` is
      ``(P,)``;
    * ``B`` fits: ``y`` is ``(B, n)`` (or ``(B, n, m)``); ``x`` is shared, ``(n, ...)``, or per
      fit, ``(B, n, ...)``; ``p0`` is ``(P,)`` (same start for all) or ``(B, P)``. All fits are
      one solve on the GPU, and each converges on its own. Fits with fewer samples: pad ``y``
      with NaN (NaN samples are skipped; +-inf in ``y`` is an error).

    Args:
        model: Warp function ``@wp.func def model(x, p)`` of one sample: ``x`` a float or
            ``wp.vecN``, ``p`` a ``wp.vecP`` (a float for one parameter), returning a float or
            ``wp.vecM`` (see the module docs).
        x: Inputs (NumPy, lists or Warp arrays).
        y: Observations.
        p0: Initial parameters. A poor start can end in a local minimum: use ``starts``.
        sigma: Measurement standard deviation: a scalar or an array broadcastable to ``y``,
            finite and positive for every observed sample.
            Residuals are ``(model - y) / sigma``; then ``reduced_chi2`` is about 1 for a model
            that explains the data, and ``param_errors`` are absolute.
        loss: Robust loss for outliers: ``"huber"``, ``"cauchy"``, ``"soft_l1"``, ``"tukey"`` or a
            :class:`pycunls.LossFunctionBatch`. ``loss_scale`` is where it starts to down-weight,
            in units of the weighted residual (e.g. 2 = two sigma).
        loss_scale: See ``loss``.
        bounds: ``(lower, upper)``, each a scalar or ``(P,)`` or ``(B, P)``; ``±np.inf`` for
            unbounded. Solved by the augmented Lagrangian (projected) minimizer.
        starts: Multiple starts against local minima: an int ``K`` draws ``K`` starts uniformly
            within finite ``bounds`` (the first is ``p0``), or an array ``(K, P)`` of starts. Every
            start is solved (all in one solve); per fit, the start with the lowest objective (the
            robust one with ``loss``) wins; starts where the model is not finite never win.
        method: ``"lm"`` (Levenberg-Marquardt, default, robust) or ``"gn"`` (Gauss-Newton).
        max_iterations: Iteration cap of the solve.
        jacobian: ``"auto"`` (autodiff, checked against central differences at ``p0``),
            ``"autodiff"`` or ``"numeric"`` (central differences).
        seed: Seed of the random starts.
        stream: CUDA stream (a new one by default).

    Returns:
        :class:`FitResult`. Check ``result.converged`` and ``result.reduced_chi2``.

    Raises:
        TypeError: the model is not a ``@wp.func`` of two arguments with supported types.
        ValueError: inconsistent shapes or more than 12 parameters.
    """
    stream = stream if stream is not None else _core.CudaStream()
    device = wp.get_cuda_device()

    def _host(a, name):
        return to_numpy(a, np.float64, name, "fit")

    fitted = _make_model(model, jacobian, device)

    # ---- shapes ------------------------------------------------------------------------------
    y_in = _host(y, "y")
    x_in = _host(x, "x")
    p0_d = _host(p0, "p0")
    if p0_d.ndim not in (1, 2):
        raise ValueError(f"p0 must be (P,) or (B, P); got shape {p0_d.shape}")
    num_params = p0_d.shape[-1]
    if num_params > _MAX_PARAMETERS:
        raise ValueError(f"fit() supports up to {_MAX_PARAMETERS} parameters per fit, got "
                         f"{num_params}; build the problem with pycunls.Problem instead")
    if num_params != fitted.num_params:
        raise ValueError(f"p0 has {num_params} parameters but the Warp model's p is a vector "
                         f"of {fitted.num_params}")
    batched, num_fits, n = _layout(x_in, y_in, p0_d)
    y_d = (y_in if batched else y_in[None]).reshape(num_fits, n, -1)
    if x_in.shape[:2] == (num_fits, n) and batched and not (x_in.shape[0] == n and num_fits != n):
        x_d = x_in
    else:
        x_d = np.broadcast_to(x_in[None], (num_fits,) + x_in.shape)
    if p0_d.ndim == 1:
        p0_d = np.broadcast_to(p0_d, (num_fits, num_params))
    elif p0_d.shape[0] != num_fits:
        raise ValueError(f"p0 has {p0_d.shape[0]} rows but there are {num_fits} fits")
    if np.any(np.isinf(y_d)):
        raise ValueError("y contains +-inf; mark missing samples with NaN (they are skipped)")
    if sigma is None:
        sig_d = np.ones_like(y_d)
    else:
        sig_d = np.broadcast_to(_host(sigma, "sigma"), y_in.shape)
        sig_d = (sig_d if batched else sig_d[None]).reshape(y_d.shape)
        observed = ~np.isnan(y_d)
        if np.any(observed & ~(np.isfinite(sig_d) & (sig_d > 0))):
            raise ValueError("sigma must be finite and positive for every observed sample "
                             "(NaN samples of y are skipped)")

    # ---- starts ------------------------------------------------------------------------------
    lower = upper = None
    if bounds is not None:
        lower = np.broadcast_to(_host(bounds[0], "bounds[0]"), (num_fits, num_params))
        upper = np.broadcast_to(_host(bounds[1], "bounds[1]"), (num_fits, num_params))
        if np.any(lower > upper):
            raise ValueError("bounds: lower must not exceed upper")
    num_starts = 1
    if starts is not None:
        if isinstance(starts, (int, np.integer)):
            num_starts = int(starts)
            if lower is None or not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
                raise ValueError("starts=K needs finite bounds to draw starts from; pass bounds "
                                 "or an explicit (K, P) array of starts")
            rng = np.random.default_rng(seed)
            draws = lower[None] + (upper - lower)[None] * rng.random(
                (num_starts, num_fits, num_params))
            draws[0] = p0_d
            p0_all = draws
        else:
            s = _host(starts, "starts")
            if s.ndim != 2 or s.shape[1] != num_params:
                raise ValueError(f"starts must be an int or a (K, P) array; got shape {s.shape}")
            num_starts = s.shape[0]
            p0_all = np.broadcast_to(s[:, None, :], (num_starts, num_fits, num_params))
        total = num_starts * num_fits
        p0_flat = p0_all.reshape(total, num_params)

        def repeat(a):
            return np.broadcast_to(a[None], (num_starts,) + a.shape).reshape((total,) + a.shape[1:])

        x_rep, y_rep, sig_rep = repeat(x_d), repeat(y_d), repeat(sig_d)
        if lower is not None:
            lower, upper = repeat(lower), repeat(upper)
    else:
        total, p0_flat, x_rep, y_rep, sig_rep = num_fits, p0_d, x_d, y_d, sig_d

    # ---- per-sample arrays -------------------------------------------------------------------
    rows = total * n
    x_s = np.ascontiguousarray(x_rep, np.float32)
    if x_s.size != rows * fitted.x_dim:
        raise ValueError(f"x has {x_s.size // rows} values per sample but the Warp model's x "
                         f"has {fitted.x_dim}; x must be (n,) / (B, n) for a float x or "
                         "(n, d) / (B, n, d) for a wp.vecd x")
    scalar_x = fitted.x_type in (float, wp.float32)
    x_s = x_s.reshape(rows) if scalar_x else x_s.reshape(rows, fitted.x_dim)
    y_s = y_rep.reshape(rows, -1)
    if y_s.shape[1] != fitted.m:
        raise ValueError(f"the model returns {fitted.m} values per sample but y has "
                         f"{y_s.shape[1]}; y must be (n,) / (B, n) for scalar models or "
                         "(n, m) / (B, n, m)")
    m = fitted.m
    sig = sig_rep.reshape(rows, -1)
    valid = np.isfinite(y_s)
    w = np.where(valid, 1.0 / sig, 0.0).astype(np.float32)
    y_clean = np.where(valid, y_s, 0.0).astype(np.float32)

    # ---- problem -----------------------------------------------------------------------------
    width = next(k for k in _VECTOR_SIZES if k >= num_params)
    start = np.zeros((total, width), np.float32)
    start[:, :num_params] = p0_flat if lower is None else np.clip(p0_flat, lower, upper)
    solver_stream = warp_stream(stream)
    with wp.ScopedStream(solver_stream):
        x_dev = wp.array(x_s, dtype=fitted.x_type, device=device)
        fitted.choose_jacobian(x_dev, np.repeat(p0_flat, n, axis=0))
        params = wp.array(start, dtype=float, device=device)
        state_cls = getattr(_core, f"VectorStateBatch{width}")
        states = state_cls(params, total)
        states.set_num_active_states(total)
        if lower is not None:
            lo = np.full((total, width), -np.inf, np.float32)
            hi = np.full((total, width), np.inf, np.float32)
            lo[:, :num_params], hi[:, :num_params] = lower, upper
            lo, hi = wp.array(lo, device=device), wp.array(hi, device=device)
            states.set_bounds(lo, hi)
        factors = _SampleFactors(fitted, x_dev, wp.array(y_clean, device=device),
                                 wp.array(w, device=device), num_params, width)
        factors.set_num_active_factors(rows)
        fit_of_sample = wp.array(np.repeat(np.arange(total, dtype=np.int32), n), device=device)
        loss_batch = _loss(loss, loss_scale)
        problem = _core.Problem()
        problem.add_state_batch(states)
        problem.add_factor_batch(factors, [states], fit_of_sample, loss_batch)
        fit_ids = wp.array(np.arange(total, dtype=np.int32), device=device)
        if total > 1:
            problem.set_problem_partition(total, [fit_ids])
        wp.synchronize_stream(solver_stream)

    base = _core.MinimizerOptions()
    base.max_num_iterations = int(max_iterations)
    base.state_tolerance = 1e-14
    base.cost_tolerance = 1e-14
    if method == "lm":
        options = _core.LevenbergMarquardtMinimizerOptions()
        options.base_options = base
        options.relative_reduction_tolerance = 1e-7
        minimizer = _core.LevenbergMarquardtMinimizer(options)
    elif method == "gn":
        minimizer = _core.GaussNewtonMinimizer(base)
    else:
        raise ValueError(f"method must be 'lm' or 'gn', got {method!r}")
    if lower is not None:
        al = _core.AugmentedLagrangianMinimizer(minimizer)
        summary = al.minimize(stream, problem)
    else:
        summary = minimizer.minimize(stream, problem)

    # ---- results -----------------------------------------------------------------------------
    # Why each fit stopped (with bounds: the wrapped minimizer's last solve).
    codes = np.asarray(minimizer.subproblem_termination_reasons())
    with wp.ScopedStream(solver_stream):
        p_final = params.numpy()[:, :num_params].astype(np.float64)
        ws = fitted.workspace(rows)
        wp.copy(ws.sample, wp.array(np.arange(rows, dtype=np.int32), device=device))
        p_rows = np.ascontiguousarray(np.repeat(p_final, n, axis=0), np.float32)
        wp.copy(ws.params, wp.array(p_rows, dtype=fitted.p_type, device=device), count=rows)
        fitted.evaluate(x_dev, ws, rows, fitted.mode)
        pred = ws.pred.numpy()[:rows].astype(np.float64)
        jac = ws.jac.numpy()[:rows].astype(np.float64)
    # Skipped (NaN) samples contribute nothing; a non-finite prediction for an observed sample
    # stays non-finite, so a failed start can never look like a perfect fit.
    w64 = w.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        r_obs = np.where(valid, (pred - y_clean) * w64, 0.0)
        r = np.where(valid, r_obs, np.nan).reshape(total, n, m)
        rr = r_obs.reshape(total, n * m)
        cost = 0.5 * np.sum(rr * rr, axis=1)
    counts = np.sum(valid.reshape(total, n * m), axis=1)
    dof = counts - num_params
    chi2 = np.where(dof > 0, 2.0 * cost / np.maximum(dof, 1), np.nan)
    usable = np.all(np.isfinite(p_final), axis=1) & np.isfinite(cost)

    # Best start per fit, by the objective that was minimized (the robust one with a loss);
    # starts with non-finite parameters or cost cannot win.
    start_index = None
    if num_starts > 1:
        score = _objective(loss_batch, rr.reshape(total, n, m), cost, stream, device)
        c = np.where(usable & np.isfinite(score), score, np.inf).reshape(num_starts, num_fits)
        best = np.argmin(c, axis=0)
        pick = best * num_fits + np.arange(num_fits)
        start_index = best
    else:
        pick = np.arange(num_fits)
    jw = (jac * w64[:, :, None]).reshape(total, n * m, num_params)[pick]
    # Only finite Jacobians are inverted; the others get undefined (NaN) errors.
    ok = usable[pick] & np.all(np.isfinite(jw.reshape(jw.shape[0], -1)), axis=1)
    jw = np.where(ok[:, None, None], jw, 0.0)
    jtj = np.einsum("bki,bkj->bij", jw, jw)
    jtj = np.where(ok[:, None, None], jtj, np.eye(num_params)[None])
    cov = np.where(ok[:, None, None], _batched_pinv(jtj), np.nan)
    if sigma is None:
        cov = cov * chi2[pick][:, None, None]
    errors = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0, None))

    codes = codes[pick]
    names = [_reason_name(c) for c in codes]
    converged = np.array([_converged(c) for c in codes])
    # A fit without one observed sample determines nothing.
    empty = counts[pick] == 0
    for k in np.flatnonzero(empty):
        names[k] = "NoObservations"
    converged[empty | ~usable[pick]] = False
    reason_names = np.array(names)
    message = summary.message
    if num_starts > 1 or empty.any():
        message = _fit_message(names, converged, num_starts)
    out = dict(
        params=p_final[pick], param_errors=errors, covariance=cov, converged=converged,
        reasons=reason_names, cost=cost[pick], reduced_chi2=chi2[pick],
        num_samples=counts[pick] // m, residuals=r[pick],
    )
    if not batched:
        out = {k: v[0] for k, v in out.items()}
        out["converged"] = bool(out["converged"])
        out["reasons"] = str(out["reasons"])
        out["num_samples"] = int(out["num_samples"])
        out["cost"] = float(out["cost"])
        out["reduced_chi2"] = float(out["reduced_chi2"])
        if m == 1:
            out["residuals"] = out["residuals"][:, 0]
        if start_index is not None:
            start_index = int(start_index[0])
    elif m == 1:
        out["residuals"] = out["residuals"][..., 0]
    return FitResult(message=message, summary=summary, jacobian_mode=fitted.mode,
                     start_index=start_index, **out)


def _fit_message(names, converged, num_starts: int) -> str:
    """Outcome of the chosen start of each fit (the solver's message covers every start)."""
    total = len(names)
    starts = f" (best of {num_starts} starts each)" if num_starts > 1 else ""
    if converged.all():
        return f"All {total} fits converged{starts}."
    bad = collections.Counter(n for n, c in zip(names, converged) if not c)
    detail = ", ".join(f"{k}: {v}" for k, v in bad.most_common())
    fixes = []
    if "NoObservations" in bad:
        fixes.append("NoObservations: every sample of the fit is NaN")
    if "NonFiniteCost" in bad or "NonFiniteStep" in bad:
        fixes.append("non-finite: the model returns NaN/inf at those parameters (check its "
                     "domain, add bounds)")
    if "MaxIterations" in bad:
        fixes.append("MaxIterations: raise max_iterations or improve p0")
    return (f"{int((~converged).sum())} of {total} fits did not converge{starts} ({detail})." +
            (" " + "; ".join(fixes) + "." if fixes else ""))


def _objective(loss_batch, rr: np.ndarray, cost: np.ndarray, stream, device) -> np.ndarray:
    """Per-fit objective that the solve minimized: ``cost`` without a loss, else the sum of
    ``0.5 rho(|r_t|^2)`` over the samples (``rr`` is ``(fits, n, m)`` weighted residuals)."""
    if loss_batch is None:
        return cost
    sq = np.sum(rr * rr, axis=2).reshape(-1).astype(np.float32)
    with wp.ScopedStream(warp_stream(stream)):
        sq_dev = wp.array(sq, device=device)
        rho = wp.empty((sq.size, 3), dtype=float, device=device)
        loss_batch.evaluate(sq_dev, rho, sq.size, stream)
        values = rho.numpy()[:, 0]
    return 0.5 * np.sum(values.reshape(rr.shape[:2]).astype(np.float64), axis=1)


def _layout(x: np.ndarray, y: np.ndarray, p0: np.ndarray) -> Tuple[bool, int, int]:
    """(batched, number of fits B, samples per fit n) from the shapes (see fit())."""
    if p0.ndim == 2 and p0.shape[0] > 1:
        if y.ndim < 2 or y.shape[0] != p0.shape[0]:
            raise ValueError(f"p0 has {p0.shape[0]} rows (fits) but y has shape {y.shape}; "
                             "batched y must be (B, n) or (B, n, m)")
        return True, y.shape[0], y.shape[1]
    if y.ndim == 1:
        return False, 1, y.shape[0]
    if y.ndim == 3:
        return True, y.shape[0], y.shape[1]
    if y.ndim != 2:
        raise ValueError(f"y must be (n,), (n, m), (B, n) or (B, n, m); got shape {y.shape}")
    # (B, n) scalar fits or (n, m) one fit with m outputs: x decides.
    if x.shape[0] == y.shape[1] and x.shape[0] != y.shape[0]:
        return True, y.shape[0], y.shape[1]       # shared x of n samples
    if x.ndim >= 2 and x.shape[:2] == y.shape:
        return True, y.shape[0], y.shape[1]       # per-fit x
    if x.shape[0] == y.shape[0]:
        return False, 1, y.shape[0]               # one fit with vector outputs
    raise ValueError(f"cannot match x {x.shape} to y {y.shape}: use y (n,) with x (n, ...), or "
                     "y (B, n) with x (n, ...) shared or (B, n, ...) per fit")


def _batched_pinv(a: np.ndarray) -> np.ndarray:
    """Inverse of each symmetric (P, P) matrix; NaN where it is singular."""
    out = np.full_like(a, np.nan)
    try:
        inv = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        inv = np.stack([np.linalg.pinv(m) for m in a])
    ok = np.all(np.isfinite(inv.reshape(a.shape[0], -1)), axis=1)
    cond = np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(np.nan_to_num(inv), axis=(1, 2))
    ok &= cond < 1e12
    return np.where(ok[:, None, None], inv, out)


def _reason_name(code: int) -> str:
    try:
        return _core.TerminationReason(int(code)).name
    except ValueError:
        return "NotDetermined"


def _converged(code: int) -> bool:
    try:
        reason = _core.TerminationReason(int(code))
    except ValueError:
        return False
    return reason in (_core.TerminationReason.CostTolerance,
                      _core.TerminationReason.StateTolerance,
                      _core.TerminationReason.RelativeReductionTolerance,
                      _core.TerminationReason.NoCostDecrease,
                      _core.TerminationReason.NoFreeStates)
