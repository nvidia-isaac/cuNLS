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

The model is either

* an **array function** ``model(x, p)``: ``x`` holds one sample per row, ``p`` the parameters of
  that row's fit (shape ``(rows, P)``), and it returns one prediction per row (shape ``(rows,)``
  or ``(rows, m)``). Write it with NumPy functions (``np.exp(...)`` works on the CuPy arrays it
  receives) and no loops: row ``i`` of the output may depend only on row ``i`` of ``x`` and ``p``.
  The Jacobian is taken by central differences in float64. Easiest to write.
* a **Warp function** ``@wp.func def model(x: float, p: wp.vec3) -> float`` (``x`` a scalar or a
  ``wp.vecN``, ``p`` a ``wp.vecP``, the result a scalar or ``wp.vecM``). The Jacobian comes from
  Warp's automatic differentiation (checked against finite differences at ``p0``). Fastest for
  large problems.

Example::

    import numpy as np, pycunls

    def decay(x, p):                       # y = a exp(-b x) + c
        return p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2]

    res = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
    print(res.report())                     # params, errors, reduced chi2, convergence
    res.params, res.param_errors, res.converged

    # 3000 independent fits (one per row of y) in one solve:
    res = pycunls.fit(decay, x, Y, p0=[1.0, 1.0, 0.0])   # Y: (3000, n); res.params: (3000, 3)

See ``python/examples/model_fitting.py`` for a tour.
"""


import dataclasses
import math
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import cupy as cp
import numpy as np

from pycunls import _pycunls_core as _core

_VECTOR_SIZES = (1, 2, 3, 4, 6, 12)
_MAX_PARAMETERS = 12

_GATHER = cp.RawKernel(r"""
extern "C" __global__
void gather_params(const unsigned long long* pointers, float* out, long long items, int width) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < items * width) {
    const float* p = reinterpret_cast<const float*>(pointers[i / width]);
    out[i] = p[i % width];
  }
}
""", "gather_params")


def _cupy_stream(handle: int):
    class _Handle:
        def __cuda_stream__(self):
            return (0, handle)

    if hasattr(cp.cuda.Stream, "from_external"):
        return cp.cuda.Stream.from_external(_Handle())
    return cp.cuda.ExternalStream(handle)


def _device_view(ptr: int, shape: Tuple[int, ...], dtype=cp.float32) -> cp.ndarray:
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.UnownedMemory(ptr, size, None)
    return cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(mem, 0))


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

class _ArrayModel:
    """model(x_rows, p_rows) -> (rows,) or (rows, m); Jacobian by central differences."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.m = None
        self.float64 = True

    def predict(self, x: cp.ndarray, p: cp.ndarray) -> cp.ndarray:
        out = self.fn(x, p)
        out = cp.asarray(out)
        if out.ndim == 0 or out.shape[0] != p.shape[0]:
            raise ValueError(
                f"model(x, p) must return one prediction per row: got shape {out.shape} for "
                f"{p.shape[0]} rows. Use p[:, k] for parameter k (p has shape (rows, P)) and "
                "keep the row axis (no sum or mean over rows).")
        return out.reshape(out.shape[0], -1)

    def setup(self, x: cp.ndarray, p0_rows: cp.ndarray) -> None:
        rows = min(8, x.shape[0])
        out = self.predict(x[:rows], p0_rows[:rows])
        self.m = out.shape[1]
        self.float64 = out.dtype == cp.float64
        # Rows must be independent: permuting the rows permutes the output, and the first
        # rows give the same output alone (no sums or means over rows).
        if rows > 1:
            perm = cp.arange(rows)[::-1]
            swapped = self.predict(x[:rows][perm], p0_rows[:rows][perm])
            half = self.predict(x[:rows // 2], p0_rows[:rows // 2])
            if not (cp.allclose(swapped, out[perm], rtol=1e-5, atol=1e-6, equal_nan=True) and
                    cp.allclose(half, out[:rows // 2], rtol=1e-5, atol=1e-6, equal_nan=True)):
                raise ValueError(
                    "model(x, p) mixes rows: row i of the output must depend only on row i of x "
                    "and p (no sums, means or cumulative operations over the row axis).")

    def evaluate(self, x: cp.ndarray, p: cp.ndarray, want_jacobian: bool):
        """Predictions (rows, m) and, if asked, the Jacobian (rows, m, P), both float32."""
        p64 = p.astype(cp.float64)
        f = self.predict(x, p64)
        if not want_jacobian:
            return f.astype(cp.float32), None
        rel = 1e-6 if self.float64 else 3e-4
        jac = cp.empty((p.shape[0], self.m, p.shape[1]), cp.float32)
        for k in range(p.shape[1]):
            h = rel * cp.maximum(1.0, cp.abs(p64[:, k]))
            plus, minus = p64.copy(), p64.copy()
            plus[:, k] += h
            minus[:, k] -= h
            diff = (self.predict(x, plus) - self.predict(x, minus)) / (2.0 * h)[:, None]
            jac[:, :, k] = diff
        return f.astype(cp.float32), jac


class _WarpModel:
    """A @wp.func model(x, p) with Warp autodiff Jacobians (or central differences)."""

    def __init__(self, fn, jacobian: str):
        import warp as wp

        self.wp = wp
        self.fn = fn
        types = list(fn.input_types.values())
        if len(types) != 2:
            raise TypeError("a Warp model takes two arguments: model(x, p)")
        self.x_type, self.p_type = types
        self.y_type = fn.value_type if fn.value_type is not None else float
        def length(t):
            return 1 if t in (float, wp.float32) else int(getattr(t, "_length_", 0))

        self.num_params = length(self.p_type)
        self.x_dim = length(self.x_type)
        self.m = length(self.y_type)
        if not (self.num_params and self.x_dim and self.m):
            raise TypeError("Warp model types: x must be float or a wp.vecN, p a wp.vecP (e.g. "
                            "wp.vec3, or wp.types.vector(length=5, dtype=float)), and the "
                            "result float or a wp.vecM")
        self.mode = jacobian
        self.float64 = False
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        wp = self.wp
        fn, XT, PT, YT = self.fn, self.x_type, self.p_type, self.y_type

        @wp.kernel(module="unique")
        def predict(x: wp.array(dtype=XT), p: wp.array(dtype=PT), y: wp.array(dtype=YT)):
            t = wp.tid()
            y[t] = fn(x[t], p[t])

        return predict

    def setup(self, x: cp.ndarray, p0_rows: cp.ndarray) -> None:
        if p0_rows.shape[1] != self.num_params:
            raise ValueError(f"p0 has {p0_rows.shape[1]} parameters but the Warp model's p is "
                             f"a vector of {self.num_params}")
        if self.mode == "auto":
            rows = min(256, x.shape[0])
            _, auto = self._autodiff(x[:rows], p0_rows[:rows])
            fd = _ArrayModel(lambda xx, pp: self._predict(xx, pp.astype(cp.float32)))
            fd.m, fd.float64 = self.m, False
            _, numeric = fd.evaluate(x[:rows], p0_rows[:rows], True)
            scale = max(1.0, float(cp.max(cp.abs(numeric))))
            err = float(cp.max(cp.abs(auto - numeric))) / scale
            if not math.isfinite(err) or err > 1e-2:
                warnings.warn(f"Warp autodiff Jacobian differs from finite differences at p0 "
                              f"(relative error {err:.3g}); using finite differences. Check the "
                              "model for in-place writes or data-dependent loops.", stacklevel=3)
                self.mode = "numeric"
            else:
                self.mode = "autodiff"

    def _wp(self, a: cp.ndarray, dtype, requires_grad=False):
        return self.wp.array(ptr=a.data.ptr, dtype=dtype, shape=(a.shape[0],),
                             device=f"cuda:{a.device.id}", copy=False,
                             requires_grad=requires_grad)

    def _predict(self, x: cp.ndarray, p: cp.ndarray) -> cp.ndarray:
        x = cp.ascontiguousarray(x, dtype=cp.float32)
        p = cp.ascontiguousarray(p, dtype=cp.float32)
        out = cp.empty((p.shape[0], self.m), cp.float32)
        cp.cuda.get_current_stream().synchronize()
        device = f"cuda:{p.device.id}"
        self.wp.launch(self._kernel, dim=p.shape[0], inputs=[self._wp(x, self.x_type),
                       self._wp(p, self.p_type), self._wp(out, self.y_type)], device=device)
        self.wp.synchronize_device(device)
        return out

    def _autodiff(self, x: cp.ndarray, p: cp.ndarray):
        wp = self.wp
        x = cp.ascontiguousarray(x, dtype=cp.float32)
        p = cp.ascontiguousarray(p, dtype=cp.float32)
        n = p.shape[0]
        cp.cuda.get_current_stream().synchronize()
        xw = self._wp(x, self.x_type)
        pw = self._wp(p, self.p_type, requires_grad=True)
        out = cp.empty((n, self.m), cp.float32)
        yw = self._wp(out, self.y_type, requires_grad=True)
        jac = cp.empty((n, self.m, self.num_params), cp.float32)
        tape = wp.Tape()
        device = f"cuda:{p.device.id}"
        with tape:
            wp.launch(self._kernel, dim=n, inputs=[xw, pw, yw], device=device)
        for r in range(self.m):
            seed = cp.zeros((n, self.m), cp.float32)
            seed[:, r] = 1.0
            cp.cuda.get_current_stream().synchronize()
            tape.backward(grads={yw: self._wp(seed, self.y_type)})
            wp.synchronize_device(device)
            jac[:, r, :] = _device_view(pw.grad.ptr, (n, self.num_params))
            tape.zero()
        wp.synchronize_device(device)
        return out, jac

    def evaluate(self, x: cp.ndarray, p: cp.ndarray, want_jacobian: bool):
        if not want_jacobian:
            return self._predict(x, p), None
        if self.mode == "numeric":
            fd = _ArrayModel(lambda xx, pp: self._predict(xx, pp.astype(cp.float32)))
            fd.m, fd.float64 = self.m, False
            return fd.evaluate(x, p, True)
        return self._autodiff(x, p)


def _make_model(model, jacobian: str):
    # A @wp.func (warp's Function class; its module moved between Warp versions).
    is_warp = type(model).__name__ == "Function" and type(model).__module__.startswith("warp")
    if is_warp:
        if jacobian not in ("auto", "autodiff", "numeric"):
            raise ValueError("jacobian must be 'auto', 'autodiff' or 'numeric'")
        return _WarpModel(model, jacobian)
    if not callable(model):
        raise TypeError("model must be a function model(x, p) or a Warp @wp.func")
    return _ArrayModel(model)


# ---------------------------------------------------------------------------------------------
# The factor: one residual per sample
# ---------------------------------------------------------------------------------------------

class _SampleFactors(_core.CustomFactorBatch):
    """residual_t = w_t * (model(x_t, p_fit(t)) - y_t) for every sample t."""

    def __init__(self, model, x, y, w, num_params, width):
        super().__init__(model.m, [width], y.shape[0])
        self.model, self.x, self.y, self.w = model, x, y, w
        self.num_params, self.width = num_params, width

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, ids_ptr, num_ids):
        n = num_ids if num_ids else self.num_active_factors
        m, width, num_params = self.model.m, self.width, self.num_params
        with _cupy_stream(stream_handle):
            pointers = _device_view(sp_ptr, (n,), cp.uint64)
            params = cp.empty((n, width), cp.float32)
            threads = 256
            _GATHER(((n * width + threads - 1) // threads,), (threads,),
                    (pointers, params, np.int64(n), np.int32(width)))
            # Item t evaluates factor ids[t], or t % N without ids (FactorBatch.Evaluate).
            if ids_ptr:
                samples = _device_view(ids_ptr, (n,), cp.int32)
            elif n == self.num_active_factors:
                samples = slice(0, n)
            else:
                samples = cp.arange(n) % self.num_active_factors
            x, y, w = self.x[samples], self.y[samples], self.w[samples]
            pred, jac = self.model.evaluate(x, params[:, :num_params], jac_ptr != 0)
            res = _device_view(res_ptr, (n, m))
            res[...] = cp.where(w > 0, (pred - y) * w, 0.0)
            if jac_ptr:
                out = _device_view(jac_ptr, (n, m, width))
                out[...] = 0.0
                out[:, :, :num_params] = cp.where(w[:, :, None] > 0, jac * w[:, :, None], 0.0)
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
        reasons: :class:`pycunls.TerminationReason` name of each fit (str or ``(B,)``).
        cost: ``0.5 * sum(r^2)`` of the weighted residuals (before a robust loss), per fit.
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
        """JSON-ready dict (lists instead of arrays; residuals and covariance left out)."""
        def plain(v):
            if isinstance(v, np.ndarray):
                if v.dtype.kind != "f":
                    return v.tolist()
                # Strict JSON: NaN / inf become None.
                return np.where(np.isfinite(v), v, None).tolist()
            if isinstance(v, (np.floating, float)):
                return float(v) if math.isfinite(float(v)) else None
            if isinstance(v, np.generic):
                return v.item()
            return v
        return {k: plain(getattr(self, k)) for k in
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


def _as_device(a, name: str) -> cp.ndarray:
    try:
        return cp.asarray(a, dtype=cp.float64)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{name} must be array-like (NumPy, CuPy, list); got {type(a)}") from exc


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
      with NaN (NaN samples are skipped).

    Args:
        model: ``model(x, p)`` (array function or Warp ``@wp.func``; see the module docs).
        x: Inputs (NumPy, CuPy or lists).
        y: Observations.
        p0: Initial parameters. A poor start can end in a local minimum: use ``starts``.
        sigma: Measurement standard deviation: a scalar or an array broadcastable to ``y``.
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
            start is solved (all in one solve) and the lowest cost wins per fit.
        method: ``"lm"`` (Levenberg-Marquardt, default, robust) or ``"gn"`` (Gauss-Newton).
        max_iterations: Iteration cap of the solve.
        jacobian: Warp models: ``"auto"`` (autodiff, checked against finite differences at
            ``p0``), ``"autodiff"`` or ``"numeric"``. Array models always use central differences.
        seed: Seed of the random starts.
        stream: CUDA stream (a new one by default).

    Returns:
        :class:`FitResult`. Check ``result.converged`` and ``result.reduced_chi2``.

    Raises:
        ValueError: inconsistent shapes, more than 12 parameters, or a model that mixes rows.
    """
    stream = stream if stream is not None else _core.CudaStream()
    fitted = _make_model(model, jacobian)

    # ---- shapes ------------------------------------------------------------------------------
    y_in = _as_device(y, "y")
    x_in = _as_device(x, "x")
    p0_d = _as_device(p0, "p0")
    if p0_d.ndim not in (1, 2):
        raise ValueError(f"p0 must be (P,) or (B, P); got shape {p0_d.shape}")
    num_params = p0_d.shape[-1]
    if num_params > _MAX_PARAMETERS:
        raise ValueError(f"fit() supports up to {_MAX_PARAMETERS} parameters per fit, got "
                         f"{num_params}; build the problem with pycunls.Problem instead")
    batched, num_fits, n = _layout(x_in, y_in, p0_d)
    y_d = (y_in if batched else y_in[None]).reshape(num_fits, n, -1)
    if x_in.shape[:2] == (num_fits, n) and batched and not (x_in.shape[0] == n and num_fits != n):
        x_d = x_in
    else:
        x_d = cp.broadcast_to(x_in[None], (num_fits,) + x_in.shape)
    if p0_d.ndim == 1:
        p0_d = cp.broadcast_to(p0_d, (num_fits, num_params))
    elif p0_d.shape[0] != num_fits:
        raise ValueError(f"p0 has {p0_d.shape[0]} rows but there are {num_fits} fits")
    if sigma is None:
        sig_d = cp.ones_like(y_d)
    else:
        sig_d = cp.broadcast_to(_as_device(sigma, "sigma"), y_in.shape)
        sig_d = (sig_d if batched else sig_d[None]).reshape(y_d.shape)
        if bool(cp.any(sig_d <= 0)):
            raise ValueError("sigma must be positive")

    # ---- starts ------------------------------------------------------------------------------
    lower = upper = None
    if bounds is not None:
        lower = cp.broadcast_to(_as_device(bounds[0], "bounds[0]"), (num_fits, num_params))
        upper = cp.broadcast_to(_as_device(bounds[1], "bounds[1]"), (num_fits, num_params))
        if bool(cp.any(lower > upper)):
            raise ValueError("bounds: lower must not exceed upper")
    num_starts = 1
    if starts is not None:
        if isinstance(starts, (int, np.integer)):
            num_starts = int(starts)
            if lower is None or not (bool(cp.all(cp.isfinite(lower)))
                                     and bool(cp.all(cp.isfinite(upper)))):
                raise ValueError("starts=K needs finite bounds to draw starts from; pass bounds "
                                 "or an explicit (K, P) array of starts")
            rng = cp.random.default_rng(seed)
            draws = lower[None] + (upper - lower)[None] * rng.random(
                (num_starts, num_fits, num_params))
            draws[0] = p0_d
            p0_all = draws
        else:
            s = _as_device(starts, "starts")
            if s.ndim != 2 or s.shape[1] != num_params:
                raise ValueError(f"starts must be an int or a (K, P) array; got shape {s.shape}")
            num_starts = s.shape[0]
            p0_all = cp.broadcast_to(s[:, None, :], (num_starts, num_fits, num_params))
        total = num_starts * num_fits
        p0_flat = p0_all.reshape(total, num_params)
        x_rep = cp.broadcast_to(x_d[None], (num_starts,) + x_d.shape).reshape((total,) +
                                                                            x_d.shape[1:])
        y_rep = cp.broadcast_to(y_d[None], (num_starts,) + y_d.shape).reshape((total,) +
                                                                            y_d.shape[1:])
        sig_rep = cp.broadcast_to(sig_d[None], (num_starts,) + sig_d.shape).reshape(
            (total,) + sig_d.shape[1:])
        if lower is not None:
            lower = cp.broadcast_to(lower[None], (num_starts, num_fits, num_params)).reshape(
                total, num_params)
            upper = cp.broadcast_to(upper[None], (num_starts, num_fits, num_params)).reshape(
                total, num_params)
    else:
        total, p0_flat, x_rep, y_rep, sig_rep = num_fits, p0_d, x_d, y_d, sig_d

    # ---- per-sample arrays -------------------------------------------------------------------
    x_s = cp.ascontiguousarray(x_rep.reshape((total * n,) + x_rep.shape[2:]))
    p0_rows = cp.repeat(cp.ascontiguousarray(p0_flat), n, axis=0)
    fitted.setup(x_s if isinstance(fitted, _ArrayModel) else x_s.astype(cp.float32),
                 p0_rows if isinstance(fitted, _ArrayModel) else p0_rows.astype(cp.float32))
    m = fitted.m
    y_s = y_rep.reshape(total * n, -1)
    if y_s.shape[1] != m:
        raise ValueError(f"the model returns {m} values per sample but y has {y_s.shape[1]}; "
                         "y must be (n,) / (B, n) for scalar models or (n, m) / (B, n, m)")
    sig = sig_rep.reshape(total * n, -1)
    valid = cp.isfinite(y_s)
    w = cp.where(valid, 1.0 / sig, 0.0).astype(cp.float32)
    y_clean = cp.where(valid, y_s, 0.0).astype(cp.float32)
    if not isinstance(fitted, _ArrayModel):
        x_s = x_s.astype(cp.float32)

    # ---- problem -----------------------------------------------------------------------------
    width = next(k for k in _VECTOR_SIZES if k >= num_params)
    params = cp.zeros((total, width), cp.float32)
    params[:, :num_params] = p0_flat
    if lower is not None:
        params[:, :num_params] = cp.clip(params[:, :num_params], lower, upper)
    state_cls = getattr(_core, f"VectorStateBatch{width}")
    states = state_cls(params, total)
    states.set_num_active_states(total)
    lo = hi = None
    if lower is not None:
        lo = cp.full((total, width), -cp.inf, cp.float32)
        hi = cp.full((total, width), cp.inf, cp.float32)
        lo[:, :num_params], hi[:, :num_params] = lower, upper
        states.set_bounds(lo, hi)
    factors = _SampleFactors(fitted, x_s, y_clean, w, num_params, width)
    factors.set_num_active_factors(total * n)
    fit_of_sample = cp.repeat(cp.arange(total, dtype=cp.int32), n)
    loss_batch = _loss(loss, loss_scale)
    problem = _core.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(factors, [states], fit_of_sample, loss_batch)
    fit_ids = cp.arange(total, dtype=cp.int32)
    if total > 1:
        problem.set_problem_partition(total, [fit_ids])

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
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # ---- results -----------------------------------------------------------------------------
    # Why each fit stopped (with bounds: the wrapped minimizer's last solve).
    codes = np.asarray(minimizer.subproblem_termination_reasons())
    p_final = params[:, :num_params].astype(cp.float64)
    pred, jac = fitted.evaluate(x_s, cp.repeat(p_final, n, axis=0), True)
    r = cp.where(valid, (pred - y_clean) * w, cp.nan).reshape(total, n, m)
    jw = (jac * w[:, :, None]).reshape(total, n * m, num_params).astype(cp.float64)
    rr = cp.nan_to_num(r).reshape(total, n * m).astype(cp.float64)
    cost = 0.5 * cp.sum(rr * rr, axis=1)
    counts = cp.sum(valid.reshape(total, n * m), axis=1)
    dof = counts - num_params
    chi2 = cp.where(dof > 0, 2.0 * cost / cp.maximum(dof, 1), cp.nan)

    # Best start per fit.
    start_index = None
    if num_starts > 1:
        c = cost.reshape(num_starts, num_fits)
        c = cp.where(cp.isfinite(c), c, cp.inf)
        best = cp.argmin(c, axis=0)
        pick = best * num_fits + cp.arange(num_fits)
        start_index = cp.asnumpy(best)
    else:
        pick = cp.arange(num_fits)
    jtj = cp.einsum("bki,bkj->bij", jw[pick], jw[pick])
    cov = _batched_pinv(jtj)
    if sigma is None:
        cov = cov * chi2[pick][:, None, None]
    errors = cp.sqrt(cp.clip(cp.diagonal(cov, axis1=1, axis2=2), 0, None))

    codes = codes[cp.asnumpy(pick)]
    reason_names = np.array([_reason_name(c) for c in codes])
    converged = np.array([_converged(c) for c in codes])
    out = dict(
        params=cp.asnumpy(p_final[pick]), param_errors=cp.asnumpy(errors),
        covariance=cp.asnumpy(cov), converged=converged, reasons=reason_names,
        cost=cp.asnumpy(cost[pick]), reduced_chi2=cp.asnumpy(chi2[pick]),
        num_samples=cp.asnumpy(counts[pick] // m), residuals=cp.asnumpy(r[pick]),
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
    return FitResult(message=summary.message, summary=summary,
                     jacobian_mode=getattr(fitted, "mode", "numeric")
                     if not isinstance(fitted, _ArrayModel) else "numeric",
                     start_index=start_index, **out)


def _layout(x: cp.ndarray, y: cp.ndarray, p0: cp.ndarray) -> Tuple[bool, int, int]:
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


def _batched_pinv(a: cp.ndarray) -> cp.ndarray:
    """Inverse of each symmetric (P, P) matrix; NaN where it is singular."""
    out = cp.full_like(a, cp.nan)
    try:
        inv = cp.linalg.inv(a)
    except cp.linalg.LinAlgError:
        inv = cp.stack([cp.linalg.pinv(m) for m in a])
    ok = cp.all(cp.isfinite(inv.reshape(a.shape[0], -1)), axis=1)
    cond = cp.linalg.norm(a, axis=(1, 2)) * cp.linalg.norm(cp.nan_to_num(inv), axis=(1, 2))
    ok &= cond < 1e12
    return cp.where(ok[:, None, None], inv, out)


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
