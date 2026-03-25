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

"""Tests for loss function batch bindings.

Each loss function is purely a construction smoke-test — we verify the object
can be instantiated with its documented parameters.  Numeric correctness of
the loss kernels is exercised indirectly by ``test_minimizer.py``'s Huber
test case.
"""

import pycunls


class TestLossFunctions:
    """Instantiation smoke-tests for every robust loss function variant."""
    def test_trivial(self):
        loss = pycunls.TrivialLossFunctionBatch()
        assert loss is not None

    def test_huber(self):
        loss = pycunls.HuberLossFunctionBatch(1.345)
        assert loss is not None

    def test_cauchy(self):
        loss = pycunls.CauchyLossFunctionBatch(1.0, 1.0)
        assert loss is not None

    def test_arctan(self):
        loss = pycunls.ArctanLossFunctionBatch(1.0, 1.0)
        assert loss is not None

    def test_soft_l_one(self):
        loss = pycunls.SoftLOneLossFunctionBatch(1.0, 1.0)
        assert loss is not None

    def test_tolerant(self):
        loss = pycunls.TolerantLossFunctionBatch(0.5, 1.0)
        assert loss is not None

    def test_tukey(self):
        loss = pycunls.TukeyLossFunctionBatch(4.685)
        assert loss is not None
