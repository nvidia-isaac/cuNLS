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

"""Array conversions and streams shared by the pure-Python layers (Warp-based)."""

import numpy as np
import warp as wp


def to_numpy(a, dtype, name: str, owner: str) -> np.ndarray:
    """Host copy of an array-like as ``dtype``: NumPy, lists, Warp arrays, or any GPU array
    exposing ``__cuda_array_interface__`` (CuPy, PyTorch). ``owner`` names the caller in errors."""
    try:
        if isinstance(a, wp.array):
            a = a.numpy()
        elif hasattr(type(a), "__cuda_array_interface__"):
            a = wp.array(a, copy=False).numpy()
        return np.asarray(a, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{owner}: {name} must be array-like (got {type(a).__name__}, expected "
                        "a NumPy array, list, wp.array or another GPU array). Fix: pass "
                        f"np.asarray({name}).") from exc


def to_device(a, dtype, name: str, owner: str) -> wp.array:
    """Contiguous device copy (``wp.array``) of an array-like, keeping its shape."""
    host = np.ascontiguousarray(to_numpy(a, dtype, name, owner))
    return wp.array(host, dtype=wp.dtype_from_numpy(host.dtype), device=wp.get_cuda_device())


def warp_stream(stream) -> wp.Stream:
    """A ``wp.Stream`` view of a :class:`pycunls.CudaStream` (or a raw ``cudaStream_t``); it does
    not own the stream."""
    handle = stream if isinstance(stream, int) else stream.get_stream()
    return wp.Stream(wp.get_cuda_device(), cuda_stream=handle)
