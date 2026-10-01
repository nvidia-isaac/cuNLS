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

"""Error metrics for the pycunls examples."""

import numpy as np


def mse(a, b):
    """Mean squared difference of two arrays."""
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def rotation_error_deg(a, b):
    """Angle of Ra^T Rb in degrees, via atan2 (accurate near 0, exactly 0 for a == b)."""
    m = a[:3, :3].astype(np.float64).T @ b[:3, :3].astype(np.float64)
    s = 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    c = 0.5 * (np.trace(m) - 1.0)
    return float(np.degrees(np.arctan2(np.linalg.norm(s), c)))


def translation_error(a, b):
    """|t_a - t_b|."""
    return float(np.linalg.norm(a[:3, 3].astype(np.float64) - b[:3, 3]))


def inlier_mask_stats(mask, is_outlier):
    """(true inliers kept, true inliers in total, outliers accepted) of an inlier mask."""
    kept = mask.astype(bool)
    return (int(np.sum(kept & ~is_outlier)), int(np.sum(~is_outlier)),
            int(np.sum(kept & is_outlier)))
