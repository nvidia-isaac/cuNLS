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

"""Small GPU helpers for custom pycunls factors written with CuPy or Warp."""

import cupy as cp
import numpy as np

_gather_kernel = cp.RawKernel(r"""
extern "C" __global__
void gather_floats(const unsigned long long* ptrs, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = reinterpret_cast<const float*>(ptrs[i])[0];
}
""", "gather_floats")


def gather_state_values(state_ptrs_ptr, count, stream_handle):
    """Reads one float from each of `count` device pointers, on cuNLS's stream.

    `state_ptrs_ptr` is the ``state_pointers`` argument of ``evaluate`` (a
    device array of ``float*``). Warp kernels cannot dereference such a table,
    so the values are first gathered into a contiguous CuPy array.
    """
    ptrs = cp.ndarray(
        shape=(count,), dtype=cp.uint64,
        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(state_ptrs_ptr, count * 8, None), 0))
    threads = 256
    with cupy_stream(stream_handle):
        out = cp.empty(count, dtype=cp.float32)
        _gather_kernel(((count + threads - 1) // threads,), (threads,),
                       (ptrs, out, np.int32(count)))
    return out


def gather_state_pairs(state_ptrs_ptr, num_items, stream_handle):
    """For factors reading two scalar states: returns (left, right) CuPy arrays
    with left[t] = *state_pointers[2t] and right[t] = *state_pointers[2t + 1]."""
    values = gather_state_values(state_ptrs_ptr, 2 * num_items, stream_handle)
    with cupy_stream(stream_handle):  # keep the copies ordered after the gather
        return values[0::2].copy(), values[1::2].copy()


class _StreamHandle:
    """Exposes a raw cudaStream_t handle through the CUDA stream protocol."""

    def __init__(self, handle):
        self.handle = handle

    def __cuda_stream__(self):
        return (0, self.handle)


def cupy_stream(handle):
    """CuPy stream wrapping a cuNLS ``stream_handle`` (use as a context manager)."""
    if hasattr(cp.cuda.Stream, "from_external"):  # CuPy >= 14
        return cp.cuda.Stream.from_external(_StreamHandle(handle))
    return cp.cuda.ExternalStream(handle)
