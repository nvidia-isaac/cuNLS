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

"""Shared helpers for the pycunls examples.

The examples keep every pycunls call inline; these modules hold everything
else: synthetic data (:mod:`datasets`), error metrics (:mod:`metrics`),
printing and checks (:mod:`report`), small GPU glue (:mod:`gpu`), SE(3)
math (:mod:`se3`), a point cloud fused from a TartanGround environment
(:mod:`tartan_map`) and the Rerun visualizations of the TartanGround
showcases (:mod:`tartan_vio_rerun`, :mod:`supermarket_drones_rerun`).
"""
