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

"""NVIDIA Warp integration for pycunls custom factors and state batches.

Provides two convenience base classes:

* **WarpFactorBatch** — lets users implement custom factor evaluation
  (residuals + Jacobians) using Warp kernels.
* **WarpStateBatch** — lets users implement a custom manifold retraction
  (the *Plus* operation) using Warp kernels.

Both classes wrap the raw device pointers passed by the cuNLS optimizer into
``warp.array`` objects so that standard ``wp.launch()`` calls work out of
the box.

For factors over poses, points and rotations (custom kinematic objectives through
``pycunls.LinkFrameFactorBatch`` among them):

* device functions :func:`pose_at`, :func:`point_at`, :func:`rotation_at` read a state (pointer
  table entry ``t * K + k``: state k of item t) in the layout of ``SE3StateBatch``, ``VectorStateBatch3`` and ``SO3StateBatch`` (the layouts of the
  ``FrameView`` views), and :func:`jacobian_index` gives the position of a Jacobian entry;
* :class:`WarpPoseFactorBatch` is a factor over one SE(3) pose from a residual function alone:
  its Jacobian comes from Warp's automatic differentiation.

Requires ``warp-lang`` (``pip install warp-lang``).
"""


import typing
from typing import Any, List, NamedTuple, Sequence, Tuple, Union

import numpy as np

import warp as wp

from pycunls._pycunls_core import CustomFactorBatch, CustomStateBatch


class FactorItems(NamedTuple):
    """The arrays of one ``WarpFactorBatch.evaluate`` call, ready for a kernel (see
    :meth:`WarpFactorBatch.items`)."""
    num_items: int            #: items to evaluate (one thread each)
    ids: wp.array             #: int32, the factor (measurement) index of each item
    state_pointers: wp.array  #: uint64, item-major: state k of item t at t * K + k
    residuals: wp.array       #: float32, residual i of item t at t * residual_size + i
    jacobians: wp.array       #: float32 Jacobian blocks (jacobian_index), or a 1-float stand-in
    write_jacobians: int      #: 1 when Jacobians are requested, else 0 (do not write them)
    stream: wp.Stream         #: the stream cuNLS evaluates on


class WarpFactorBatch(CustomFactorBatch):
    """Base class for user-defined factors evaluated via Warp kernels.

    Subclasses override :meth:`evaluate`. The shortest form turns the raw call into Warp arrays
    with :meth:`items` and launches one thread per item with :meth:`launch`, which appends the
    standard arguments after the factor's own inputs::

        @wp.kernel
        def diff_kernel(measurements: wp.array(dtype=float),          # the factor's inputs ...
                        ids: wp.array(dtype=wp.int32),                  # ... then the standard
                        state_pointers: wp.array(dtype=wp.uint64),      #     arguments of launch()
                        residuals: wp.array(dtype=float), jacobians: wp.array(dtype=float),
                        write_jacobians: int):
            t = wp.tid()
            ...

        class DiffFactor(WarpFactorBatch):
            def evaluate(self, *args):
                return self.launch(diff_kernel, self.items(*args), [self.measurements])

    Parameters
    ----------
    residual_size : int
        Dimension of the residual vector per factor.
    state_sizes : list[int]
        Tangent dimensions of each state consumed by one factor.
    capacity : int
        Number of factors the measurement buffers hold. The batch starts
        with 0 active factors: call :meth:`set_num_active_factors` before solving.
        The active count is :attr:`num_active_factors`.
    device : str
        Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        residual_size: int,
        state_sizes: list[int],
        capacity: int,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(residual_size, state_sizes, capacity)
        self._device = device
        self._num_states = len(state_sizes)
        self._num_columns = int(sum(state_sizes))
        self._streams: dict = {}    # cudaStream_t handle -> wp.Stream (made once per handle)
        # Passed as `jacobians` when none are requested (kernels skip it: write_jacobians = 0).
        self._no_jacobians = wp.zeros(1, dtype=wp.float32, device=device)
        self._default_ids: dict[Tuple[int, int], wp.array] = {}
        # Active counts with cached default ids: the current one and the one
        # before it (whose arrays in-flight kernels may still read).
        self._default_ids_counts: List[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def wrap_array(
        self,
        ptr: int,
        dtype: Any,
        shape: Union[int, Tuple[int, ...]],
    ) -> wp.array:
        """Wrap a device pointer as a zero-copy ``warp.array``.

        Parameters
        ----------
        ptr : int
            Device pointer (as returned by the trampoline).
        dtype : warp dtype
            Element type, e.g. ``wp.float32``, ``wp.uint64``.
        shape : int or tuple[int, ...]
            Array shape.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return wp.array(ptr=ptr, dtype=dtype, shape=shape,
                        device=self._device, copy=False)

    def factor_ids(self, factor_ids_ptr: int, num_items: int) -> wp.array:
        """Factor (measurement) index of every item, as an ``int32`` array.

        Wraps ``factor_ids_ptr`` when it is non-null; otherwise returns
        ``t % num_active_factors`` for ``t < num_items`` (built once per item count
        and active factor count, and cached). Kernels can then always read
        ``ids[t]``. The cache keeps the arrays of the current and the previous
        active factor count only, so changing the count every frame does not
        accumulate arrays, while kernels still reading the previous count's
        arrays stay valid. Raises ``ValueError`` when ``num_items > 0`` but the
        batch has no active factors.
        """
        if factor_ids_ptr != 0:
            return self.wrap_array(factor_ids_ptr, wp.int32, num_items)
        num_factors = self.num_active_factors
        if num_factors == 0 and num_items > 0:
            raise ValueError(
                f"factor_ids: {num_items} items requested but the batch has 0 active "
                "factors; call set_num_active_factors() first")
        if not self._default_ids_counts or self._default_ids_counts[-1] != num_factors:
            if num_factors in self._default_ids_counts:
                self._default_ids_counts.remove(num_factors)
            self._default_ids_counts.append(num_factors)
            del self._default_ids_counts[:-2]
            self._default_ids = {key: ids for key, ids in self._default_ids.items()
                                 if key[1] in self._default_ids_counts}
        key = (num_items, num_factors)
        ids = self._default_ids.get(key)
        if ids is None:
            values = np.arange(num_items, dtype=np.int32)
            if num_factors > 0:
                values %= num_factors
            ids = wp.array(values, dtype=wp.int32, device=self._device)
            # The upload runs on Warp's stream and the caller's kernel on cuNLS's: finish it
            # (once per item count and active count, not per evaluation).
            wp.synchronize_stream(wp.get_stream(self._device))
            self._default_ids[key] = ids
        return ids

    def make_warp_stream(self, stream_handle: int) -> wp.Stream:
        """The ``warp.Stream`` wrapping an existing ``cudaStream_t`` (made once per handle and
        cached, so a callback does not build a new one at every evaluation).

        Parameters
        ----------
        stream_handle : int
            The raw ``cudaStream_t`` handle (as an integer).
        """
        stream = self._streams.get(stream_handle)
        if stream is None:
            stream = wp.Stream(self._device, cuda_stream=stream_handle)
            self._streams[stream_handle] = stream
        return stream

    def items(self, residuals_ptr: int, jacobians_ptr: int, state_pointers_ptr: int,
              stream_handle: int, factor_ids_ptr: int, num_factor_ids: int) -> FactorItems:
        """The arguments of :meth:`evaluate` as Warp arrays (see :class:`FactorItems`):
        ``state_pointers`` holds ``num_items * K`` entries (K = ``len(state_sizes)``),
        ``residuals`` ``num_items * residual_size`` floats, ``jacobians`` the row-major Jacobian
        blocks (``num_items * residual_size * sum(state_sizes)`` floats; index them with
        :func:`jacobian_index`) or, when none are requested, a 1-float stand-in with
        ``write_jacobians = 0``. Nothing is allocated except the default ids on a first use (see
        :meth:`factor_ids`)."""
        n, m = num_factor_ids, self.residuals_size
        write = 1 if jacobians_ptr != 0 else 0
        return FactorItems(
            n, self.factor_ids(factor_ids_ptr, n),
            self.wrap_array(state_pointers_ptr, wp.uint64, n * self._num_states),
            self.wrap_array(residuals_ptr, wp.float32, n * m),
            (self.wrap_array(jacobians_ptr, wp.float32, n * m * self._num_columns) if write
             else self._no_jacobians),
            write, self.make_warp_stream(stream_handle))

    def launch(self, kernel, items: FactorItems, inputs: Sequence = ()) -> bool:
        """Launches ``kernel`` with one thread per item on cuNLS's stream. Its arguments are
        ``inputs`` (the factor's own arrays and values) followed by ``ids, state_pointers,
        residuals, jacobians, write_jacobians`` from ``items``. Returns True, so that
        ``evaluate`` can ``return self.launch(...)``."""
        wp.launch(kernel, dim=items.num_items,
                  inputs=[*inputs, items.ids, items.state_pointers, items.residuals,
                          items.jacobians, items.write_jacobians],
                  stream=items.stream, device=self._device)
        return True

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        residuals_ptr: int,
        jacobians_ptr: int,
        state_pointers_ptr: int,
        stream_handle: int,
        factor_ids_ptr: int,
        num_factor_ids: int,
    ) -> bool:
        """Evaluate residuals and Jacobians of ``num_factor_ids`` items.

        An *item* is one factor of this batch evaluated against one set of
        states (the same contract as C++ ``FactorBatch::Evaluate``).
        The regular minimizers call this with
        ``num_factor_ids == num_active_factors``
        and ``factor_ids_ptr == 0``, i.e. item *t* is factor *t*. The RANSAC
        minimizers also evaluate the same factors against many state sets in
        one call. Launch one thread per item; for item *t*:

        * its factor (measurement) index is ``factor_ids[t]`` if
          ``factor_ids_ptr != 0``, otherwise ``t % num_active_factors``;
        * its *K* states are ``state_pointers[t * K + k]``;
        * it writes ``residual_size`` residuals at ``t * residual_size`` and,
          if requested, its row-major ``residual_size x sum(state_sizes)``
          Jacobian block at ``t * residual_size * sum(state_sizes)``.

        Override this method in your subclass. :meth:`items` converts the
        arguments into Warp arrays and :meth:`launch` launches a kernel with
        them (see the class docstring); for anything else, :meth:`wrap_array`
        wraps a raw device pointer and :meth:`make_warp_stream` gives the
        ``warp.Stream`` for ``wp.launch(..., stream=...)``.

        Parameters
        ----------
        residuals_ptr : int
            Device pointer to the output residuals
            (``num_factor_ids * residual_size`` floats).
        jacobians_ptr : int
            Device pointer to the output Jacobians
            (``num_factor_ids * residual_size * sum(state_sizes)`` floats).
            ``0`` (null) when only residuals are needed.
        state_pointers_ptr : int
            Device pointer to ``num_factor_ids * K`` ``float*`` state
            pointers, item-major.
        stream_handle : int
            ``cudaStream_t`` handle for asynchronous kernel launches.
        factor_ids_ptr : int
            Device pointer to ``num_factor_ids`` ``int32`` factor indices in
            ``[0, num_active_factors)``, or ``0`` for item *t* -> factor
            ``t % num_active_factors``.
        num_factor_ids : int
            Number of items to evaluate, always > 0 (unlike C++, where 0
            means ``num_active_factors``, the actual count is passed here).

        Returns
        -------
        bool
            ``True`` on success.
        """
        raise NotImplementedError(
            "WarpFactorBatch.evaluate() must be overridden in a subclass."
        )


# ======================================================================
# WarpStateBatch
# ======================================================================

class WarpStateBatch(CustomStateBatch):
    """Base class for user-defined state batches with a Warp-based Plus.

    Subclasses must override :meth:`plus` and use ``wp.launch()`` to
    compute the manifold retraction ``x_plus_delta = x (+) delta`` on the
    GPU.

    Parameters
    ----------
    data : DevicePointer
        GPU array (e.g. ``wp.array``) or raw int pointer to the contiguous buffer
        holding ``capacity * ambient_size`` floats.
    ambient_size : int
        Number of floats stored per state (storage dimension).
    tangent_size : int
        Number of floats per tangent-space update vector.
    capacity : int
        Number of states the buffer holds. The batch starts with 0
        active states: call :meth:`set_num_active_states` before solving.
    const_state_ids : DevicePointer, optional
        GPU array of ``int32`` indices of states held constant.
    const_capacity : int
        Number of entries the *const_state_ids* buffer holds.
    device : str
        Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        data: Any,
        ambient_size: int,
        tangent_size: int,
        capacity: int,
        const_state_ids: Any = None,
        const_capacity: int = 0,
        device: str = "cuda:0",
    ) -> None:
        if const_state_ids is not None:
            super().__init__(data, ambient_size, tangent_size, capacity,
                             const_state_ids, const_capacity)
        else:
            super().__init__(data, ambient_size, tangent_size, capacity)
        self._device = device
        self._streams: dict = {}    # cudaStream_t handle -> wp.Stream (made once per handle)

    # ------------------------------------------------------------------
    # Helpers (same API as WarpFactorBatch)
    # ------------------------------------------------------------------

    def wrap_array(
        self,
        ptr: int,
        dtype: Any,
        shape: Union[int, Tuple[int, ...]],
    ) -> wp.array:
        """Wrap a device pointer as a zero-copy ``warp.array``.

        Parameters
        ----------
        ptr : int
            Device pointer (as returned by the trampoline).
        dtype : warp dtype
            Element type, e.g. ``wp.float32``.
        shape : int or tuple[int, ...]
            Array shape.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return wp.array(ptr=ptr, dtype=dtype, shape=shape,
                        device=self._device, copy=False)

    def make_warp_stream(self, stream_handle: int) -> wp.Stream:
        """The ``warp.Stream`` wrapping an existing ``cudaStream_t`` (made once per handle and
        cached, so a callback does not build a new one at every evaluation).

        Parameters
        ----------
        stream_handle : int
            The raw ``cudaStream_t`` handle (as an integer).
        """
        stream = self._streams.get(stream_handle)
        if stream is None:
            stream = wp.Stream(self._device, cuda_stream=stream_handle)
            self._streams[stream_handle] = stream
        return stream

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def plus(
        self,
        x_ptr: int,
        delta_ptr: int,
        x_plus_delta_ptr: int,
        stream_handle: int,
        num_replicas: int,
    ) -> None:
        """Apply the manifold retraction: ``x_plus_delta = x (+) delta``.

        The arrays hold ``num_replicas`` contiguous copies of the batch
        (the same contract as C++ ``StateBatch::Plus``): with
        ``N = num_active_states`` (the active count), replica *r* is states
        ``[r * N, (r + 1) * N)``.
        The regular minimizers pass ``num_replicas == 1``; the RANSAC
        minimizers update one replica per hypothesis in a single call.
        Process all ``num_replicas * N`` states, each independently.

        Override this method in your subclass. Use :meth:`wrap_array` to
        convert the raw device pointers into ``warp.array`` objects and
        :meth:`make_warp_stream` to obtain a ``warp.Stream`` suitable for
        ``wp.launch(..., stream=...)``.

        Parameters
        ----------
        x_ptr : int
            Device pointer to the current state values
            (``num_replicas * N * ambient_size`` floats).
        delta_ptr : int
            Device pointer to the tangent-space update
            (``num_replicas * N * tangent_size`` floats).
        x_plus_delta_ptr : int
            Device pointer to the output buffer for the retracted state
            (``num_replicas * N * ambient_size`` floats); does not
            overlap the inputs.
        stream_handle : int
            ``cudaStream_t`` handle for asynchronous kernel launches.
        num_replicas : int
            Number of contiguous copies (>= 1).
        """
        raise NotImplementedError(
            "WarpStateBatch.plus() must be overridden in a subclass."
        )


# ======================================================================
# Device helpers for factors over poses, points and rotations
# ======================================================================

@wp.func
def pose_at(state_pointers: wp.array(dtype=wp.uint64), index: int) -> wp.transform:
    """The SE(3) pose ``state_pointers[index]`` points to (``SE3StateBatch`` layout, or a
    ``FrameView.Pose`` view: row-major 4x4 ``world_from_frame``), as a ``wp.transform``.
    ``index`` is a position in the item-major pointer table: state k of item t is at
    ``t * K + k`` for a factor with K states (``t`` when K = 1)."""
    X = wp.array(ptr=state_pointers[index], shape=(16,), dtype=float)
    R = wp.mat33(X[0], X[1], X[2], X[4], X[5], X[6], X[8], X[9], X[10])
    return wp.transform(wp.vec3(X[3], X[7], X[11]), wp.quat_from_matrix(R))


@wp.func
def point_at(state_pointers: wp.array(dtype=wp.uint64), index: int) -> wp.vec3:
    """The 3D point ``state_pointers[index]`` points to (``VectorStateBatch3`` layout, or a
    ``FrameView.Position`` view: the frame's origin in the world); ``index`` as in
    :func:`pose_at` (``t * K + k``)."""
    x = wp.array(ptr=state_pointers[index], shape=(3,), dtype=float)
    return wp.vec3(x[0], x[1], x[2])


@wp.func
def rotation_at(state_pointers: wp.array(dtype=wp.uint64), index: int) -> wp.mat33:
    """The rotation ``state_pointers[index]`` points to (``SO3StateBatch`` layout, or a
    ``FrameView.Orientation`` view: row-major 3x3 ``world_from_frame``); ``index`` as in
    :func:`pose_at` (``t * K + k``)."""
    R = wp.array(ptr=state_pointers[index], shape=(9,), dtype=float)
    return wp.mat33(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8])


@wp.func
def jacobian_index(item: int, row: int, column: int, num_rows: int, num_columns: int) -> int:
    """Position of Jacobian entry (row, column) of item ``item`` in the row-major
    ``num_rows x num_columns`` blocks of ``WarpFactorBatch.evaluate`` (``num_columns`` = the sum
    of ``state_sizes``; a state's columns start after those of the states before it)."""
    return (item * num_rows + row) * num_columns + column


# ======================================================================
# WarpPoseFactorBatch: a residual function of one pose, autodiff Jacobian
# ======================================================================

def _warp_length(t) -> int:
    """1 for a scalar Warp type, N for a wp.vecN, 0 for anything else."""
    return 1 if t in (float, wp.float32) else int(getattr(t, "_length_", 0))


class WarpPoseFactorBatch(WarpFactorBatch):
    """A factor over one SE(3) pose from a residual function alone; Warp differentiates it.

    The residual is a Warp function of the pose and of one per-factor data item::

        @wp.func
        def above_table(X: wp.transform, height: float) -> float:
            return height - wp.transform_get_translation(X)[2]   # <= 0: above the table

        factor = WarpPoseFactorBatch(above_table, wp.array(heights, dtype=float), capacity=n)

    ``X`` is the pose of the factor's state (``SE3StateBatch``, or a ``FrameView.Pose`` view of a
    ``LinkFrameFactorBatch``), ``data[f]`` the data item of factor ``f`` (any Warp type: a float,
    a vector, a struct), and the result a float or a ``wp.vecM`` (M residual rows).

    The Jacobian is with respect to the right perturbation ``X Exp([omega, v])`` (the convention
    of ``SE3StateBatch``): one forward pass, then one reverse (adjoint) pass per residual row.
    ``jacobian="auto"`` (default) checks it once, at the first evaluation with Jacobians, against
    central differences and falls back to them with a warning when they disagree (Warp unrolls
    loops only up to ``wp.config.max_unroll`` iterations); ``"autodiff"`` and ``"numeric"`` pick
    one. An analytic ``WarpFactorBatch`` is faster; use this for prototyping and for residuals
    whose derivatives are tedious.

    Parameters
    ----------
    residual : Warp function
        ``residual(X: wp.transform, data: T) -> float | wp.vecM``.
    data : warp.array
        ``capacity`` items of the residual's second argument type (kept alive).
    capacity : int
        Factors the data holds; the batch starts with 0 active factors.
    jacobian : str
        "auto", "autodiff" or "numeric".
    device : str
        Warp device.
    """

    def __init__(self, residual, data, capacity: int, jacobian: str = "auto",
                 device: str = "cuda:0") -> None:
        if type(residual).__name__ != "Function" or not type(residual).__module__.startswith(
                "warp"):
            raise TypeError("WarpPoseFactorBatch: residual is not a Warp function (got "
                            f"{type(residual).__name__}). Fix: decorate it with @wp.func.")
        types = list(residual.input_types.values())
        ret = None
        if getattr(residual, "func", None) is not None:
            try:   # resolves string annotations (`from __future__ import annotations`)
                ret = typing.get_type_hints(residual.func).get("return")
            except Exception:   # an annotation Python cannot evaluate: fall back to Warp's view
                ret = residual.func.__annotations__.get("return")
        ret = ret or residual.value_type or float
        m = _warp_length(ret)
        if len(types) != 2 or not wp.types.types_equal(types[0], wp.transform) or m == 0:
            raise TypeError("WarpPoseFactorBatch: the residual has the wrong signature (got "
                            f"arguments {types} and result {ret}, expected (X: wp.transform, data: "
                            "T) -> float or wp.vecM). Fix: declare it as @wp.func def r(X: "
                            "wp.transform, data: T) -> wp.vecM.")
        if jacobian not in ("auto", "autodiff", "numeric"):
            raise ValueError(f"WarpPoseFactorBatch: jacobian={jacobian!r} (expected 'auto', "
                             "'autodiff' or 'numeric'). Fix: pick one of them.")
        super().__init__(residual_size=m, state_sizes=[6], capacity=capacity, device=device)
        self.residual, self.data, self.mode, self.m = residual, data, jacobian, m
        self._kernels = self._make_kernels(residual, types[1], m)
        # Item count -> work arrays, for the current and the previous count only (in-flight
        # kernels of the previous evaluation may still read its arrays; older sets are released,
        # so a changing active count does not accumulate GPU memory).
        self._work = {}

    @staticmethod
    def _make_kernels(fn, DT, M):
        @wp.kernel(module="unique")
        def gather(pointers: wp.array(dtype=wp.uint64), poses: wp.array(dtype=wp.transform)):
            t = wp.tid()
            poses[t] = pose_at(pointers, t)

        @wp.kernel(module="unique")
        def evaluate(poses: wp.array(dtype=wp.transform), ids: wp.array(dtype=wp.int32),
                     data: wp.array(dtype=DT), delta: wp.array(dtype=wp.spatial_vector),
                     out: wp.array2d(dtype=float)):
            # r(X Exp(delta)); the step is exact to first order (all the Jacobian needs) and
            # accurate enough for central differences along one tangent axis at a time.
            t = wp.tid()
            d = delta[t]
            dq = wp.normalize(wp.quat(0.5 * d[0], 0.5 * d[1], 0.5 * d[2], 1.0))
            X = wp.transform_multiply(poses[t], wp.transform(wp.vec3(d[3], d[4], d[5]), dq))
            r = fn(X, data[ids[t]])
            if wp.static(M == 1):
                out[t, 0] = r
            else:
                for i in range(M):
                    out[t, i] = r[i]

        @wp.kernel(module="unique")
        def seed_row(row: int, seed: wp.array2d(dtype=float)):
            t = wp.tid()
            for i in range(M):
                seed[t, i] = wp.where(i == row, 1.0, 0.0)

        @wp.kernel(module="unique")
        def store_row(grad: wp.array(dtype=wp.spatial_vector), row: int,
                      jac: wp.array3d(dtype=float)):
            t = wp.tid()
            g = grad[t]
            for k in range(6):
                jac[t, row, k] = g[k]

        @wp.kernel(module="unique")
        def set_step(axis: int, h: float, delta: wp.array(dtype=wp.spatial_vector)):
            t = wp.tid()
            d = wp.spatial_vector()
            d[axis] = h
            delta[t] = d

        @wp.kernel(module="unique")
        def store_difference(plus: wp.array2d(dtype=float), minus: wp.array2d(dtype=float),
                             axis: int, h: float, jac: wp.array3d(dtype=float)):
            t = wp.tid()
            for i in range(M):
                jac[t, i, axis] = (plus[t, i] - minus[t, i]) / (2.0 * h)

        return dict(gather=gather, evaluate=evaluate, seed_row=seed_row, store_row=store_row,
                    set_step=set_step, store_difference=store_difference)

    def _buffers(self, n: int) -> dict:
        """Work arrays for n items, made once per item count; the arrays of the two most
        recently used counts are kept (kernels in flight may still read the previous ones)."""
        work = self._work.get(n)
        if work is not None:
            self._work[n] = self._work.pop(n)   # most recent last
        if work is None:
            d, m = self._device, self.m
            work = dict(poses=wp.empty(n, dtype=wp.transform, device=d),
                        delta=wp.zeros(n, dtype=wp.spatial_vector, device=d, requires_grad=True),
                        out=wp.zeros((n, m), dtype=float, device=d, requires_grad=True),
                        seed=wp.zeros((n, m), dtype=float, device=d),
                        plus=wp.zeros((n, m), dtype=float, device=d),
                        minus=wp.zeros((n, m), dtype=float, device=d),
                        step=wp.zeros(n, dtype=wp.spatial_vector, device=d))
            self._work[n] = work
            while len(self._work) > 2:
                del self._work[next(iter(self._work))]   # the least recently used count
        return work

    def _autodiff(self, w, n, ids, jac) -> None:
        k = self._kernels
        tape = wp.Tape()
        with tape:
            wp.launch(k["evaluate"], dim=n, inputs=[w["poses"], ids, self.data, w["delta"]],
                      outputs=[w["out"]], device=self._device)
        for row in range(self.m):
            wp.launch(k["seed_row"], dim=n, inputs=[row, w["seed"]], device=self._device)
            tape.backward(grads={w["out"]: w["seed"]})
            wp.launch(k["store_row"], dim=n, inputs=[w["delta"].grad, row, jac],
                      device=self._device)
            tape.zero()

    def _numeric(self, w, n, ids, jac, h: float = 1e-3) -> None:
        k = self._kernels
        for axis in range(6):
            for sign, out in ((1.0, w["plus"]), (-1.0, w["minus"])):
                wp.launch(k["set_step"], dim=n, inputs=[axis, sign * h, w["step"]],
                          device=self._device)
                wp.launch(k["evaluate"], dim=n, inputs=[w["poses"], ids, self.data, w["step"]],
                          outputs=[out], device=self._device)
            wp.launch(k["store_difference"], dim=n, inputs=[w["plus"], w["minus"], axis, h, jac],
                      device=self._device)

    def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
                 factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        k = self._kernels
        with wp.ScopedStream(self.make_warp_stream(stream_handle)):
            w = self._buffers(n)
            ids = self.factor_ids(factor_ids_ptr, n)
            wp.launch(k["gather"], dim=n,
                      inputs=[self.wrap_array(state_pointers_ptr, wp.uint64, n), w["poses"]],
                      device=self._device)
            residuals = self.wrap_array(residuals_ptr, wp.float32, (n, self.m))
            wp.launch(k["evaluate"], dim=n, inputs=[w["poses"], ids, self.data, w["delta"]],
                      outputs=[residuals], device=self._device)
            if jacobians_ptr == 0:
                return True
            jac = self.wrap_array(jacobians_ptr, wp.float32, (n, self.m, 6))
            if self.mode == "numeric":
                self._numeric(w, n, ids, jac)
                return True
            self._autodiff(w, n, ids, jac)
            if self.mode == "auto":
                self._check_autodiff(w, n, ids, jac)
        return True

    def _check_autodiff(self, w, n, ids, jac) -> None:
        """First evaluation with Jacobians in "auto" mode: autodiff against central differences
        (one host synchronization); on a mismatch, warn and use central differences from now on."""
        numeric = wp.zeros((n, self.m, 6), dtype=float, device=self._device)
        self._numeric(w, n, ids, numeric)
        a, b = jac.numpy(), numeric.numpy()
        error = float(np.max(np.abs(a - b) / (1.0 + np.abs(b)))) if a.size else 0.0
        if not np.isfinite(error) or error > 1e-2:
            import warnings
            warnings.warn(f"WarpPoseFactorBatch: the autodiff Jacobian of {self.residual.key} "
                          f"differs from central differences (relative error {error:.3g}); using "
                          "central differences from now on. Fix: check loops longer than "
                          "wp.config.max_unroll, or pass jacobian='numeric'.", RuntimeWarning)
            wp.copy(jac, numeric)
            self.mode = "numeric"
        else:
            self.mode = "autodiff"
