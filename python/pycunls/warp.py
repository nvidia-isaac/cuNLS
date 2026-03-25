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
    num_factors : int
        Number of factors in the batch.
    device : str
        Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        residual_size: int,
        state_block_sizes: list[int],
        num_factors: int,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(residual_size, state_block_sizes, num_factors)
        self._device = device

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
    ) -> bool:
        """Evaluate residuals and Jacobians for every factor in the batch.

        Override this method in your subclass.  Use :meth:`wrap_array` to
        convert the raw device pointers into ``warp.array`` objects and
        :meth:`make_warp_stream` to obtain a ``warp.Stream`` suitable for
        ``wp.launch(..., stream=...)``.

        Parameters
        ----------
        residuals_ptr : int
            Device pointer to the output residuals buffer
            (``num_factors * residual_size`` floats).
        jacobians_ptr : int
            Device pointer to the output Jacobians buffer.  May be ``0``
            (null) when the optimizer only needs residuals.
        state_pointers_ptr : int
            Device pointer to an array of ``float*`` state block pointers.
            For each factor *i* with *K* state blocks the layout is:
            ``state_pointers[i * K + k]`` points to state block *k*.
        stream_handle : int
            ``cudaStream_t`` handle for asynchronous kernel launches.

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
        holding ``num_blocks * ambient_size`` floats.
    ambient_size : int
        Number of floats stored per state block (storage dimension).
    tangent_size : int
        Number of floats per tangent-space update vector.
    num_blocks : int
        Number of state blocks in the batch.
    const_state_ids : DevicePointer, optional
        CuPy array of ``int32`` indices of blocks held constant.
    num_const_state_blocks : int
        Number of constant blocks (length of *const_state_ids*).
    device : str
        Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        data: Any,
        ambient_size: int,
        tangent_size: int,
        num_blocks: int,
        const_state_ids: Any = None,
        num_const_state_blocks: int = 0,
        device: str = "cuda:0",
    ) -> None:
        if const_state_ids is not None:
            super().__init__(data, ambient_size, tangent_size, num_blocks,
                             const_state_ids, num_const_state_blocks)
        else:
            super().__init__(data, ambient_size, tangent_size, num_blocks)
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
    ) -> None:
        """Apply the manifold retraction: ``x_plus_delta = x (+) delta``.

        Override this method in your subclass.  Use :meth:`wrap_array` to
        convert the raw device pointers into ``warp.array`` objects and
        :meth:`make_warp_stream` to obtain a ``warp.Stream`` suitable for
        ``wp.launch(..., stream=...)``.

        Parameters
        ----------
        x_ptr : int
            Device pointer to the current state values
            (``num_blocks * ambient_size`` floats).
        delta_ptr : int
            Device pointer to the tangent-space update
            (``num_blocks * tangent_size`` floats).
        x_plus_delta_ptr : int
            Device pointer to the output buffer for the retracted state
            (``num_blocks * ambient_size`` floats).
        stream_handle : int
            ``cudaStream_t`` handle for asynchronous kernel launches.
        """
        raise NotImplementedError(
            "WarpStateBatch.plus() must be overridden in a subclass."
        )
