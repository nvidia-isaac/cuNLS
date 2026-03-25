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
stream   – A fresh CudaStream for each test; many pycunls operations require one.
cublas   – A cuBLAS handle needed by Lie-group state batches and some factors.
sync     – Yields, then synchronises the stream *after* the test body finishes.
           Use as ``usefixtures("sync")`` or request it directly when a test
           needs the GPU work to have completed before it reads results back.
"""

import pytest
import cupy as cp

import pycunls


@pytest.fixture
def stream():
    """Create a CUDA stream scoped to a single test."""
    return pycunls.CudaStream()


@pytest.fixture
def cublas():
    """Create a cuBLAS handle scoped to a single test."""
    return pycunls.CublasHandle()


@pytest.fixture
def sync(stream):
    """Yield, then synchronize the stream after the test body runs."""
    yield
    cp.cuda.runtime.streamSynchronize(stream.get_stream())
