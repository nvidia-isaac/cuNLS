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

"""Shared pytest fixtures for all pycunls tests.

Fixtures
--------
stream   – A fresh CudaStream for each test; many pycunls operations require one. It is also
           Warp's current stream for the test, so ``wp.array`` uploads, ``.assign`` and
           ``.numpy()`` read-backs are ordered with the cuNLS work on it.
"""

import pytest
import warp as wp

import pycunls


@pytest.fixture
def stream():
    """Create a CUDA stream scoped to a single test and make it Warp's current stream."""
    s = pycunls.CudaStream()
    device = wp.get_cuda_device()
    previous = wp.get_stream(device)
    wp.set_stream(wp.Stream(device, cuda_stream=s.get_stream()), device)
    yield s
    wp.synchronize_device(device)
    wp.set_stream(previous, device)
