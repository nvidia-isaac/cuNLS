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

Requires ``warp-lang`` (``pip install warp-lang``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "pycunls.warp requires the 'warp-lang' package. "
        "Install it with: pip install warp-lang"
    ) from exc

from pycunls._pycunls_core import CustomFactorBatch, CustomStateBatch


class WarpFactorBatch(CustomFactorBatch):
    """Base class for user-defined factors evaluated via Warp kernels.

    Subclasses must override :meth:`evaluate` and use ``wp.launch()`` to
    compute residuals and (optionally) Jacobians on the GPU.

    Parameters
    ----------
    residual_size : int
        Dimension of the residual vector per factor.
    state_block_sizes : list[int]
        Tangent dimensions of each state block consumed by one factor.
    capacity : int
        Number of factors the measurement buffers hold. The batch starts
        with 0 active factors: call :meth:`set_num_factors` before solving.
        The active count is :attr:`num_factors`.
    device : str
        Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        residual_size: int,
        state_block_sizes: list[int],
        capacity: int,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(residual_size, state_block_sizes, capacity)
        self._device = device
        self._default_ids: dict[Tuple[int, int], wp.array] = {}

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
        ``t % num_factors`` for ``t < num_items`` (built once per item count
        and active factor count, and cached). Kernels can then always read
        ``ids[t]``.
        """
        if factor_ids_ptr != 0:
            return self.wrap_array(factor_ids_ptr, wp.int32, num_items)
        key = (num_items, self.num_factors)
        ids = self._default_ids.get(key)
        if ids is None:
            values = np.arange(num_items, dtype=np.int32) % self.num_factors
            ids = wp.array(values, dtype=wp.int32, device=self._device)
            self._default_ids[key] = ids
        return ids

    def make_warp_stream(self, stream_handle: int) -> wp.Stream:
        """Create a ``warp.Stream`` that wraps an existing ``cudaStream_t``.

        Parameters
        ----------
        stream_handle : int
            The raw ``cudaStream_t`` handle (as an integer).
        """
        return wp.Stream(cuda_stream=stream_handle)

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
        state blocks (the same contract as C++ ``FactorBatch::Evaluate``).
        The regular minimizers call this with ``num_factor_ids == num_factors``
        and ``factor_ids_ptr == 0``, i.e. item *t* is factor *t*. The RANSAC
        minimizers also evaluate the same factors against many state sets in
        one call. Launch one thread per item; for item *t*:

        * its factor (measurement) index is ``factor_ids[t]`` if
          ``factor_ids_ptr != 0``, otherwise ``t % num_factors``;
        * its *K* state blocks are ``state_pointers[t * K + k]``;
        * it writes ``residual_size`` residuals at ``t * residual_size`` and,
          if requested, its row-major ``residual_size x sum(state_block_sizes)``
          Jacobian block at ``t * residual_size * sum(state_block_sizes)``.

        Override this method in your subclass.  Use :meth:`wrap_array` to
        convert the raw device pointers into ``warp.array`` objects and
        :meth:`make_warp_stream` to obtain a ``warp.Stream`` suitable for
        ``wp.launch(..., stream=...)``.

        Parameters
        ----------
        residuals_ptr : int
            Device pointer to the output residuals
            (``num_factor_ids * residual_size`` floats).
        jacobians_ptr : int
            Device pointer to the output Jacobians
            (``num_factor_ids * residual_size * sum(state_block_sizes)`` floats).
            ``0`` (null) when only residuals are needed.
        state_pointers_ptr : int
            Device pointer to ``num_factor_ids * K`` ``float*`` state block
            pointers, item-major.
        stream_handle : int
            ``cudaStream_t`` handle for asynchronous kernel launches.
        factor_ids_ptr : int
            Device pointer to ``num_factor_ids`` ``int32`` factor indices in
            ``[0, num_factors)``, or ``0`` for item *t* -> factor
            ``t % num_factors``.
        num_factor_ids : int
            Number of items to evaluate, always > 0 (unlike C++, where 0
            means ``num_factors``, the actual count is passed here).

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
        CuPy array (or raw int pointer) to the contiguous GPU buffer
        holding ``capacity * ambient_size`` floats.
    ambient_size : int
        Number of floats stored per state block (storage dimension).
    tangent_size : int
        Number of floats per tangent-space update vector.
    capacity : int
        Number of state blocks the buffer holds. The batch starts with 0
        active blocks: call :meth:`set_num_state_blocks` before solving.
    const_state_ids : DevicePointer, optional
        CuPy array of ``int32`` indices of blocks held constant.
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
        """Create a ``warp.Stream`` that wraps an existing ``cudaStream_t``.

        Parameters
        ----------
        stream_handle : int
            The raw ``cudaStream_t`` handle (as an integer).
        """
        return wp.Stream(cuda_stream=stream_handle)

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
        ``N = num_state_blocks`` (the active count), replica *r* is blocks ``[r * N, (r + 1) * N)``.
        The regular minimizers pass ``num_replicas == 1``; the RANSAC
        minimizers update one replica per hypothesis in a single call.
        Process all ``num_replicas * N`` blocks, each independently.

        Override this method in your subclass.  Use :meth:`wrap_array` to
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
