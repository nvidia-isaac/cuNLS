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

* :class:`WarpFunctionFactorBatch` is a factor from a residual function alone: a ``@wp.func`` of
  its states (poses, rotations, vectors) and of one per-factor data item. Its Jacobian comes from
  Warp's automatic differentiation, central differences, or a Jacobian function.
* :class:`WarpFactorBatch` is the base class for factors evaluated by your own Warp kernels
  (residuals and analytic Jacobians), the fastest form.
* :class:`WarpStateBatch` is the base class for a custom manifold update (the *Plus* operation)
  in Warp kernels.

The base classes wrap the raw device pointers the cuNLS minimizers pass into ``warp.array``
objects, so that standard ``wp.launch()`` calls work. For kernels over poses, points and
rotations (custom kinematic objectives through ``pycunls.LinkFrameFactorBatch`` among them), the
device functions :func:`pose_at`, :func:`point_at` and :func:`rotation_at` read a state (pointer
table entry ``t * K + k``: state k of item t) in the layout of ``SE3StateBatch``,
``VectorStateBatch3`` and ``SO3StateBatch`` (the layouts of the ``FrameView`` views), and
:func:`jacobian_index` gives the position of a Jacobian entry.

Requires ``warp-lang`` (``pip install warp-lang``).
"""


import collections
import itertools
import linecache
import warnings
from typing import Any, List, NamedTuple, Sequence, Tuple, Union

import numpy as np

import warp as wp

from pycunls._arrays import cached_stream, is_warp_function, warp_length, warp_result_type
from pycunls._pycunls_core import CustomFactorBatch, CustomStateBatch


def _cuda_device(device, owner: str) -> wp.Device:
    """The Warp device a batch lives on: ``device``, or Warp's current device when it is None
    (``wp.ScopedDevice`` included); it must be a CUDA device."""
    resolved = wp.get_device(device)
    if not resolved.is_cuda:
        raise ValueError(f"{owner}: the device must be a CUDA device (got {resolved}, expected "
                         "cuda:N). Fix: pass device='cuda:N', or create the batch under "
                         "wp.ScopedDevice('cuda:N').")
    return resolved


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
    device : str or wp.Device, optional
        The CUDA device the factor's data lives on. None (default): Warp's current device at
        construction (``wp.get_device()``, which follows ``wp.ScopedDevice``).
    """

    def __init__(
        self,
        residual_size: int,
        state_sizes: list[int],
        capacity: int,
        device: Any = None,
    ) -> None:
        super().__init__(residual_size, state_sizes, capacity)
        device = self._device = _cuda_device(device, type(self).__name__)
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
        return cached_stream(self._streams, stream_handle, self._device)

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
    device : str or wp.Device, optional
        The CUDA device of ``data``. None (default): ``data.device`` for a ``wp.array``, else
        Warp's current device at construction (``wp.get_device()``, which follows
        ``wp.ScopedDevice``).
    """

    def __init__(
        self,
        data: Any,
        ambient_size: int,
        tangent_size: int,
        capacity: int,
        const_state_ids: Any = None,
        const_capacity: int = 0,
        device: Any = None,
    ) -> None:
        if const_state_ids is not None:
            super().__init__(data, ambient_size, tangent_size, capacity,
                             const_state_ids, const_capacity)
        else:
            super().__init__(data, ambient_size, tangent_size, capacity)
        if device is None and isinstance(data, wp.array):
            device = data.device
        self._device = _cuda_device(device, type(self).__name__)
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
        return cached_stream(self._streams, stream_handle, self._device)

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
# WarpFunctionFactorBatch: a factor from a residual function
# ======================================================================

@wp.func
def _plus_se3(X: wp.transform, d: wp.spatial_vector) -> wp.transform:
    """X Exp([omega, v]) to first order in the step, and exact along one tangent axis but for the
    rotation angle 2 atan(h / 2) (all a Jacobian needs, central differences included)."""
    dq = wp.normalize(wp.quat(0.5 * d[0], 0.5 * d[1], 0.5 * d[2], 1.0))
    return wp.transform_multiply(X, wp.transform(wp.vec3(d[3], d[4], d[5]), dq))


@wp.func
def _plus_so3(q: wp.quat, d: wp.vec3) -> wp.quat:
    """q Exp(omega), to the same order as :func:`_plus_se3`."""
    return q * wp.normalize(wp.quat(0.5 * d[0], 0.5 * d[1], 0.5 * d[2], 1.0))


@wp.func
def _quat_at(state_pointers: wp.array(dtype=wp.uint64), index: int) -> wp.quat:
    """The rotation ``state_pointers[index]`` points to (``SO3StateBatch`` layout), as a
    quaternion."""
    return wp.quat_from_matrix(rotation_at(state_pointers, index))


class _Slot(NamedTuple):
    """How a residual argument type reads and moves its state (a row of the type table of
    :class:`WarpFunctionFactorBatch`)."""
    state: str    # the state batch class whose layout the slot reads (catalog name)
    tangent: int  # tangent size
    step: Any     # Warp type of one tangent step
    load: str     # source reading the state: {p} the pointer table, {i} its entry, {T} the type
    plus: str     # source of the moved state: {x} the state, {d} the step


def _slot(t) -> Union[_Slot, None]:
    """The :class:`_Slot` of a residual argument type, or None when it is not a state type."""
    if wp.types.types_equal(t, wp.transform):
        return _Slot("SE3StateBatch", 6, wp.spatial_vector, "pose_at({p}, {i})",
                     "_plus_se3({x}, {d})")
    if wp.types.types_equal(t, wp.quat):
        return _Slot("SO3StateBatch", 3, wp.vec3, "_quat_at({p}, {i})", "_plus_so3({x}, {d})")
    vector = wp.types.type_is_vector(t) and getattr(t, "_wp_scalar_type_", None) == wp.float32
    if t in (float, wp.float32) or vector:
        return _Slot("VectorStateBatch", warp_length(t), t,
                     "wp.array(ptr={p}[{i}], shape=(1,), dtype={T})[0]", "{x} + {d}")
    return None


_SOURCES = itertools.count()


def _kernels(namespace: dict, num_states: int, has_data: bool, M: int, J: int,
             slots: List[_Slot], analytic: bool, vector: bool) -> dict:
    """Compiles the kernels of one residual function (Warp kernels have a fixed argument list, so
    their source is generated for the function's states). The source is registered with
    ``linecache``, where Warp's code generation reads a function's source."""
    K = num_states
    offsets = list(itertools.accumulate([0] + [s.tangent for s in slots]))
    states = "".join(f"s{k}: wp.array(dtype=T{k}), " for k in range(K))
    data = "ids: wp.array(dtype=wp.int32), data: wp.array(dtype=DT), " if has_data else ""
    item = ", data[ids[t]]" if has_data else ""

    def call(args):
        return f"fn({', '.join(args)}{item})"

    def load(k):
        return slots[k].load.format(p="p", i=f"t * {K} + {k}", T=f"T{k}")

    def plus(k, step):
        return slots[k].plus.format(x=f"s{k}[t]", d=step)

    def unit(k):  # step of slot k along tangent column c (zero when c is another slot's)
        if slots[k].step in (float, wp.float32):
            return f"    e{k} = wp.where(c == {offsets[k]}, h, 0.0)\n"
        return (f"    e{k} = D{k}()\n    for j in range({slots[k].tangent}):\n"
                f"        e{k}[j] = wp.where(c == {offsets[k]} + j, h, 0.0)\n")

    def store(k):  # Jacobian columns of slot k from its gradient; the gradient is zeroed
        if slots[k].step in (float, wp.float32):
            body = f"    jac[t, row, {offsets[k]}] = a{k}[t, row]\n    a{k}[t, row] = 0.0\n"
        else:
            body = (f"    g{k} = a{k}[t, row]\n    for j in range({slots[k].tangent}):\n"
                    f"        jac[t, row, {offsets[k]} + j] = g{k}[j]\n    a{k}[t, row] = D{k}()\n")
        return body

    rows_of = "r[i]" if vector else "r"
    steps = "".join(f"d{k}: wp.array2d(dtype=D{k}), " for k in range(K))
    grads = "".join(f"a{k}: wp.array2d(dtype=D{k}), " for k in range(K))
    src = f'''
@wp.kernel(module="unique")
def residual(p: wp.array(dtype=wp.uint64), {data}out: wp.array2d(dtype=float)):
    t = wp.tid()
    r = {call([load(k) for k in range(K)])}
    for i in range({M}):
        out[t, i] = {rows_of}


@wp.kernel(module="unique")
def gather(p: wp.array(dtype=wp.uint64), {states.rstrip(", ")}):
    t = wp.tid()
{"".join(f"    s{k}[t] = {load(k)}{chr(10)}" for k in range(K))}

@wp.kernel(module="unique")
def rows(
    {states}{steps}{data}out: wp.array2d(dtype=float)):
    t, row = wp.tid()
    r = {call([plus(k, f"d{k}[t, row]") for k in range(K)])}
    out[t, row] = {"r[row]" if vector else "r"}


@wp.kernel(module="unique")
def store({grads}jac: wp.array3d(dtype=float)):
    t, row = wp.tid()
{"".join(store(k) for k in range(K))}

@wp.kernel(module="unique")
def numeric({states}{data}h: float, jac: wp.array3d(dtype=float)):
    t, c = wp.tid()
{"".join(unit(k) for k in range(K))}    rp = {call([plus(k, f"e{k}") for k in range(K)])}
    rm = {call([plus(k, f"-e{k}") for k in range(K)])}
    for i in range({M}):
        jac[t, i, c] = ({"rp[i]" if vector else "rp"} - {"rm[i]" if vector else "rm"}) / (2.0 * h)
'''
    if analytic:
        src += f'''

@wp.kernel(module="unique")
def analytic(p: wp.array(dtype=wp.uint64), {data}jac: wp.array3d(dtype=float)):
    t = wp.tid()
    m = jfn({", ".join(load(k) for k in range(K))}{item})
    for i in range({M}):
        for c in range({J}):
            jac[t, i, c] = m[i, c]
'''
    name = f"<pycunls.warp generated kernels {next(_SOURCES)}>"
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    exec(compile(src, name, "exec"), namespace)
    return {key: namespace[key] for key in ("residual", "gather", "rows", "store", "numeric",
                                            "analytic") if key in namespace}


class WarpFunctionFactorBatch(WarpFactorBatch):
    """A factor from a residual function alone: Warp differentiates it.

    The residual is a ``@wp.func`` of the factor's states, in slot order, and optionally of one
    per-factor data item (its last argument, ``data[f]`` for factor ``f``: any Warp type, a
    ``@wp.struct`` for several fields). It returns a ``float`` or a ``wp.vecM`` (M residual rows)::

        @wp.func
        def reprojection(T: wp.transform, p: wp.vec3, uv: wp.vec2) -> wp.vec2:
            q = wp.transform_point(wp.transform_inverse(T), p)   # T: world_from_camera
            return wp.vec2(q[0] / q[2], q[1] / q[2]) - uv

        factor = WarpFunctionFactorBatch(reprojection, data=observations, capacity=n)
        factor.set_num_active_factors(n)
        problem.add_factor_batch(factor, [poses, points], index_table)  # slots in argument order

    Each argument type decides the state the slot reads and the update the Jacobian is taken
    with respect to:

    * ``wp.transform``: an ``SE3StateBatch`` (or a ``FrameView.Pose`` view), tangent 6, update
      ``X Exp([omega, v])``;
    * ``wp.quat``: an ``SO3StateBatch`` (or ``FrameView.Orientation``), tangent 3, ``R Exp(omega)``;
    * ``float``: a ``VectorStateBatch`` of dim 1, ``x + d``;
    * ``wp.vecN`` (any N): a ``VectorStateBatch`` of dim N (or ``FrameView.Position`` for N = 3),
      ``x + d``.

    States without a Warp value type (SE(2), Sim(2/3), SL(4), articulation coordinates) need a
    :class:`WarpFactorBatch`. Link frames of an articulation are reached through
    ``LinkFrameFactorBatch`` views. :attr:`slot_states` names each slot's state batch class;
    ``ProblemBuilder`` and :func:`pycunls.diagnose` check the connected batches against it.

    ``jacobian`` chooses the Jacobian:

    * ``"auto"`` (default): Warp's reverse-mode automatic differentiation, checked once (at the
      first evaluation with Jacobians, one synchronization) against central differences; on a
      mismatch it warns and uses central differences from then on (Warp unrolls loops only up
      to ``wp.config.max_unroll`` iterations; reference entries that are not finite, e.g. a
      difference step outside the residual's domain, are not compared). Run one solve before
      capturing a CUDA graph, so that the check (it needs Jacobians: ``pycunls.evaluate`` does
      not run it) is not captured.
    * ``"autodiff"`` or ``"numeric"`` (central differences): one of them, without the check.
    * a ``@wp.func`` with the residual's arguments that returns a
      ``wp.types.matrix(shape=(M, J), dtype=float)``: the analytic Jacobian, J = the sum of the
      tangent sizes, columns in argument order with the conventions of the table.

    An evaluation is a few recorded launches on the minimizer's stream (re-recorded only for new
    buffers or item counts). An analytic :class:`WarpFactorBatch` kernel is still faster:
    prototype with this class, and write one where the evaluation time matters. Work arrays are
    kept for the last two item counts: a captured CUDA graph replays those of the counts it was
    captured with, so keep the active counts unchanged while a graph is in use (as
    ``reuse_structure`` requires anyway).

    Parameters
    ----------
    residual : Warp function
        ``residual(x_0, ..., x_{K-1}[, item]) -> float | wp.vecM``.
    data : warp.array, optional
        At least ``capacity`` items of the residual's last argument type (kept alive); None when
        the residual takes only states.
    capacity : int
        Factors the batch holds (keyword); it starts with 0 active factors.
    jacobian : str or Warp function
        ``"auto"``, ``"autodiff"``, ``"numeric"`` or a Jacobian function (see above).
    device : str or wp.Device, optional
        The CUDA device. None (default): ``data.device``, or Warp's current device at
        construction when there is no data.

    Attributes
    ----------
    slot_states : tuple of str
        The state batch class each slot reads (``"SE3StateBatch"``, ``"SO3StateBatch"`` or
        ``"VectorStateBatch"``).
    jacobian_mode : str
        The Jacobian in use: ``"auto"`` until the first check, then ``"autodiff"`` or
        ``"numeric"``; ``"analytic"`` for a Jacobian function.
    """

    _STEP = 1e-3  # central-difference step in tangent coordinates (float32)

    def __init__(self, residual, data=None, *, capacity: int, jacobian: Any = "auto",
                 device: Any = None) -> None:
        name = "WarpFunctionFactorBatch"
        if not is_warp_function(residual):
            raise TypeError(f"{name}: residual is not a Warp function (got "
                            f"{type(residual).__name__}). Fix: decorate it with @wp.func.")
        types = list(residual.input_types.values())
        has_data = data is not None
        state_types = types[:-1] if has_data else types
        slots = [_slot(t) for t in state_types]
        ret = warp_result_type(residual)
        M = warp_length(ret)
        if not state_types or None in slots or M == 0 or getattr(
                ret, "_wp_scalar_type_", wp.float32) != wp.float32:
            raise TypeError(
                f"{name}: the residual has the wrong signature (got arguments {types} and result "
                f"{ret}, expected states of type wp.transform, wp.quat, float or wp.vecN"
                f"{', then the data item type' if has_data else ''}, and a float or wp.vecM "
                "result). Fix: declare it as @wp.func def r(x: wp.transform, ..., item: T) -> "
                "wp.vecM, and pass data=None when it takes only states.")
        if has_data:
            item_type = {float: wp.float32, int: wp.int32, bool: wp.bool}.get(types[-1], types[-1])
            if not isinstance(data, wp.array) or not wp.types.types_equal(data.dtype, item_type):
                raise TypeError(f"{name}: data must be a wp.array of the residual's last argument "
                                f"type (got {getattr(data, 'dtype', type(data).__name__)}, "
                                f"expected {item_type}). Fix: wp.array(items, dtype=...).")
            if data.ndim != 1:
                raise ValueError(f"{name}: data must be a one-dimensional array of items (got shape "
                                 f"{data.shape}, expected (capacity,)). Fix: one item per factor; a "
                                 "@wp.struct or wp.vecN item for several values.")
            if data.shape[0] < capacity:
                raise ValueError(f"{name}: data holds fewer items than the capacity (got "
                                 f"{data.shape[0]}, expected >= {capacity}). Fix: one data item "
                                 "per factor.")
            if device is not None and wp.get_device(device) != data.device:
                raise ValueError(f"{name}: device differs from data.device (got {device}, "
                                 f"expected {data.device}). Fix: drop device=.")
            device = data.device
        J = sum(s.tangent for s in slots)
        if is_warp_function(jacobian):
            shape = getattr(warp_result_type(jacobian), "_shape_", None)
            given = list(jacobian.input_types.values())
            if (len(given) != len(types) or tuple(shape or ()) != (M, J) or
                    not all(wp.types.types_equal(a, b) for a, b in zip(given, types))):
                raise TypeError(f"{name}: the Jacobian function has the wrong signature (got "
                                f"arguments {given} and result shape {shape}, expected the "
                                f"residual's arguments and a ({M}, {J}) matrix). Fix: declare it "
                                f"as @wp.func returning wp.types.matrix(shape=({M}, {J}), "
                                "dtype=float).")
            mode = "analytic"
        elif jacobian in ("auto", "autodiff", "numeric"):
            mode = jacobian
        else:
            raise ValueError(f"{name}: jacobian={jacobian!r} (expected 'auto', 'autodiff', "
                             "'numeric' or a Jacobian function). Fix: pick one of them.")
        super().__init__(residual_size=M, state_sizes=[s.tangent for s in slots],
                         capacity=capacity, device=device)
        self.residual, self.data, self.jacobian_mode = residual, data, mode
        self.slot_states = tuple(s.state for s in slots)
        self._has_data, self._m = has_data, M
        self._state_types, self._step_types = state_types, [s.step for s in slots]
        namespace = {"wp": wp, "fn": residual, "jfn": jacobian if mode == "analytic" else None,
                     "DT": types[-1] if has_data else None, "pose_at": pose_at,
                     "_quat_at": _quat_at, "_plus_se3": _plus_se3, "_plus_so3": _plus_so3}
        for k, (t, s) in enumerate(zip(state_types, slots)):
            namespace[f"T{k}"], namespace[f"D{k}"] = t, s.step
        self._kernels = _kernels(namespace, len(slots), has_data, M, J, slots, mode == "analytic",
                                 wp.types.type_is_vector(ret))
        # Item count -> work arrays and recorded launches, for the current and the previous count
        # only (in-flight kernels of the previous evaluation may still read its arrays; older sets
        # are released, so a changing active count does not accumulate GPU memory).
        self._work = {}

    def _buffers(self, n: int) -> dict:
        """The work arrays and recorded launches of n items (made on first use)."""
        work = self._work.pop(n, None)
        if work is None:
            work = {"launches": collections.OrderedDict()}
        self._work[n] = work  # most recent last
        while len(self._work) > 2:
            del self._work[next(iter(self._work))]  # the least recently used count
        return work

    def _arrays(self, work: dict, n: int, autodiff: bool, stream: wp.Stream) -> dict:
        """Gathered states (and for autodiff zero steps, their gradients and the output seed),
        initialized on the evaluation's ``stream``, so its kernels see them filled."""
        d, m = self._device, self._m
        if "s" not in work:
            work["s"] = [wp.empty(n, dtype=t, device=d) for t in self._state_types]
        if autodiff and "d" not in work:
            with wp.ScopedStream(stream):
                work["d"] = [wp.zeros((n, m), dtype=t, device=d) for t in self._step_types]
                work["a"] = [wp.zeros((n, m), dtype=t, device=d) for t in self._step_types]
                work["ones"] = wp.ones((n, m), dtype=float, device=d)
        return work

    def _record(self, mode: str, n: int, ptrs: tuple, stream: wp.Stream,
                jac: Any = None) -> List[wp.Launch]:
        """The launches of one evaluation in ``mode`` ("residual", "autodiff", "numeric" or
        "analytic"), recorded once per item count and set of buffers (an evaluation then only
        replays them); ``jac`` replaces the Jacobian output (not cached)."""
        residuals_ptr, jacobians_ptr, pointers_ptr, ids_ptr = ptrs
        work, cache = self._buffers(n), jac is None
        # The default factor ids (null ids_ptr) depend on the active count.
        key = (mode, ptrs, self.num_active_factors if ids_ptr == 0 and self._has_data else 0)
        launches = work["launches"].get(key) if cache else None
        if launches is not None:
            work["launches"].move_to_end(key)
            return launches
        m, kern, dev = self._m, self._kernels, self._device
        p = self.wrap_array(pointers_ptr, wp.uint64, n * self._num_states)
        out = self.wrap_array(residuals_ptr, wp.float32, (n, m))
        item = ([self.factor_ids(ids_ptr, n), self.data] if self._has_data else [])
        if jac is None and mode != "residual":
            jac = self.wrap_array(jacobians_ptr, wp.float32, (n, m, self._num_columns))

        def rec(kernel, dim, inputs, outputs, **kw):
            return wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=dev,
                             record_cmd=True, **kw)

        launches = [rec(kern["residual"], n, [p, *item], [out])] if mode != "autodiff" else []
        if mode in ("autodiff", "numeric"):
            s = self._arrays(work, n, mode == "autodiff", stream)["s"]
            launches.append(rec(kern["gather"], n, [p], s))
        if mode == "autodiff":
            inputs = [*s, *work["d"], *item]
            launches += [
                rec(kern["rows"], (n, m), inputs, [out]),
                rec(kern["rows"], (n, m), inputs, [out], adjoint=True,
                    adj_inputs=[None] * len(s) + work["a"] + [None] * len(item),
                    adj_outputs=[work["ones"]]),
                rec(kern["store"], (n, m), work["a"], [jac])]
        elif mode == "numeric":
            launches.append(rec(kern["numeric"], (n, self._num_columns),
                                [*s, *item, self._STEP], [jac]))
        elif mode == "analytic":
            launches.append(rec(kern["analytic"], n, [p, *item], [jac]))
        if cache:
            work["launches"][key] = launches
            while len(work["launches"]) > 8:
                work["launches"].popitem(last=False)
        return launches

    def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
                 factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        stream = self.make_warp_stream(stream_handle)
        ptrs = (residuals_ptr, jacobians_ptr, state_pointers_ptr, factor_ids_ptr)
        mode = self.jacobian_mode if jacobians_ptr != 0 else "residual"
        for launch in self._record("autodiff" if mode == "auto" else mode, n, ptrs, stream):
            launch.launch(stream)
        if mode == "auto":
            self._check_autodiff(n, ptrs, stream)
        return True

    def _check_autodiff(self, n: int, ptrs: tuple, stream: wp.Stream) -> None:
        """First evaluation with Jacobians in "auto" mode: the autodiff Jacobian just written
        against central differences (one synchronization); on a mismatch, warn and use central
        differences from now on."""
        jac = self.wrap_array(ptrs[1], wp.float32, (n, self._m, self._num_columns))
        with wp.ScopedStream(stream):
            numeric = wp.zeros((n, self._m, self._num_columns), dtype=float, device=self._device)
            for launch in self._record("numeric", n, ptrs, stream, jac=numeric):
                launch.launch(stream)
            a, b = jac.numpy(), numeric.numpy()
            # Central differences can step outside the residual's domain: a non-finite reference
            # entry says nothing about the autodiff one; a non-finite autodiff entry against a
            # finite reference is wrong.
            finite_a, finite_b = np.isfinite(a), np.isfinite(b)
            both = finite_a & finite_b
            error = (float(np.max(np.abs(a[both] - b[both]) / (1.0 + np.abs(b[both]))))
                     if both.any() else 0.0)
            if np.any(finite_b & ~finite_a):
                error = float("inf")
            if error > 1e-2:
                warnings.warn(f"WarpFunctionFactorBatch: the autodiff Jacobian of "
                              f"{self.residual.key} differs from central differences (relative "
                              f"error {error:.3g}); using central differences from now on. Fix: "
                              "check loops longer than wp.config.max_unroll, or pass "
                              "jacobian='numeric'.", RuntimeWarning)
                wp.copy(jac, numeric)
                self.jacobian_mode = "numeric"
            else:
                self.jacobian_mode = "autodiff"
