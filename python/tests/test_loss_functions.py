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

"""Construction smoke tests for the loss function batch bindings: every robust loss can be
instantiated with its documented parameters. Numeric correctness of the loss kernels is exercised
indirectly by ``test_minimizer.py`` (Huber) and ``test_fitting.py`` (Cauchy, Huber).
"""

import pytest

import pycunls


@pytest.mark.parametrize("make", [
    lambda: pycunls.TrivialLossFunctionBatch(),
    lambda: pycunls.HuberLossFunctionBatch(1.345),
    lambda: pycunls.CauchyLossFunctionBatch(1.0, 1.0),
    lambda: pycunls.ArctanLossFunctionBatch(1.0, 1.0),
    lambda: pycunls.SoftLOneLossFunctionBatch(1.0, 1.0),
    lambda: pycunls.TolerantLossFunctionBatch(0.5, 1.0),
    lambda: pycunls.TukeyLossFunctionBatch(4.685),
    lambda: pycunls.ScaledLossFunctionBatch(pycunls.HuberLossFunctionBatch(1.0), 2.5),
], ids=["trivial", "huber", "cauchy", "arctan", "soft_l_one", "tolerant", "tukey", "scaled"])
def test_loss_function_constructs(make):
    assert isinstance(make(), pycunls.LossFunctionBatch)
