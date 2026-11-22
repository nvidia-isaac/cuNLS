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

"""Array conversions, streams and Warp function types shared by the pure-Python layers."""

import typing

import numpy as np
import warp as wp


def interface_device(a):
    """The Warp device of an array exposing ``__cuda_array_interface__``, which names no device:
    from DLPack (``__dlpack_device__``, as CuPy and PyTorch provide), else None (Warp then
    assumes its current device when wrapping the array)."""
    dlpack_device = getattr(a, "__dlpack_device__", None)
    if dlpack_device is None:
        return None
    kind, ordinal = dlpack_device()
    return wp.get_device(f"cuda:{int(ordinal)}") if int(kind) in (2, 13) else None  # CUDA, managed


def to_numpy(a, dtype, name: str, owner: str) -> np.ndarray:
    """Host copy of an array-like as ``dtype``: NumPy, lists, Warp arrays, or any GPU array
    exposing ``__cuda_array_interface__`` (CuPy, PyTorch). ``owner`` names the caller in errors."""
    try:
        if isinstance(a, wp.array):
            a = a.numpy()
        elif hasattr(type(a), "__cuda_array_interface__"):
            view = wp.array(a, copy=False, device=interface_device(a))
            # Warp ignores the producer stream the interface names: order the copy after it
            # (None: no ordering needed; 1 and 2 are the legacy and per-thread default streams,
            # which are also valid stream handles).
            producer = a.__cuda_array_interface__.get("stream")
            if producer:
                wp.get_stream(view.device).wait_stream(
                    wp.Stream(view.device, cuda_stream=producer))
            a = view.numpy()
        return np.asarray(a, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{owner}: {name} must be array-like (got {type(a).__name__}, expected "
                        "a NumPy array, list, wp.array or another GPU array). Fix: pass "
                        f"np.asarray({name}).") from exc


def cuda_device(device, owner: str) -> wp.Device:
    """The CUDA device ``device`` names (``"cuda:N"``, ``"cuda"`` or a ``wp.Device``); None:
    Warp's current device (``wp.get_device()``, which follows ``wp.ScopedDevice``). ``owner``
    names the caller in the error."""
    try:
        resolved = wp.get_device(device)
    except ValueError:
        resolved = None
    if resolved is None or not resolved.is_cuda:
        raise ValueError(f"{owner}: the device must be a CUDA device of this machine (got "
                         f"{device!r}, expected 'cuda:N' or a CUDA wp.Device). Fix: pass "
                         "device='cuda:N', or create it under wp.ScopedDevice('cuda:N').")
    return resolved


def gpu_device(*arrays):
    """The Warp device of the first GPU array among ``arrays`` (a ``wp.array`` on a CUDA device,
    or any array exposing ``__cuda_array_interface__``: see :func:`interface_device`), or None
    when none is one."""
    for a in arrays:
        if isinstance(a, wp.array):
            if a.device.is_cuda:
                return a.device
        elif hasattr(type(a), "__cuda_array_interface__"):
            return interface_device(a) or wp.array(a, copy=False).device
    return None


def to_device(a, dtype=None, name: str = "array", owner: str = "pycunls") -> wp.array:
    """Contiguous device copy (``wp.array``) of an array-like, keeping its shape (and its dtype
    when ``dtype`` is None)."""
    host = np.ascontiguousarray(a if dtype is None else to_numpy(a, dtype, name, owner))
    return wp.array(host, dtype=wp.dtype_from_numpy(host.dtype), device=wp.get_cuda_device())


def warp_stream(stream) -> wp.Stream:
    """A ``wp.Stream`` for any stream pycunls accepts: a ``wp.Stream`` itself, a
    :class:`pycunls.CudaStream`, an object with the CUDA stream protocol ``__cuda_stream__``, or a
    raw ``cudaStream_t``; it does not own the stream."""
    if isinstance(stream, wp.Stream):
        return stream
    if isinstance(stream, int):
        handle = stream
    elif hasattr(stream, "get_stream"):
        handle = stream.get_stream()
    elif hasattr(stream, "cuda_stream"):
        handle = int(stream.cuda_stream)
    else:
        handle = int(stream.__cuda_stream__()[1])
    return wp.Stream(wp.get_cuda_device(), cuda_stream=handle)


def cached_stream(cache: dict, handle: int, device) -> wp.Stream:
    """The ``wp.Stream`` wrapping the raw ``cudaStream_t`` ``handle``, made once per handle and
    kept in ``cache``, so a callback does not build a new one at every evaluation."""
    stream = cache.get(handle)
    if stream is None:
        stream = cache[handle] = wp.Stream(device, cuda_stream=handle)
    return stream


def is_warp_function(obj) -> bool:
    """Whether ``obj`` is a ``@wp.func`` (Warp's Function class; its module moved between Warp
    versions)."""
    return type(obj).__name__ == "Function" and type(obj).__module__.startswith("warp")


def warp_length(t) -> int:
    """1 for a scalar Warp type, N for a wp.vecN, 0 for anything else."""
    return 1 if t in (float, wp.float32) else int(getattr(t, "_length_", 0))


def warp_result_type(fn):
    """The result type of the ``@wp.func`` ``fn``, also when its annotations are strings
    (``from __future__ import annotations``); float when it has none."""
    result = None
    if getattr(fn, "func", None) is not None:
        try:
            result = typing.get_type_hints(fn.func).get("return")
        except Exception:  # an annotation Python cannot evaluate: fall back to the raw one
            result = fn.func.__annotations__.get("return")
    return result or fn.value_type or float
