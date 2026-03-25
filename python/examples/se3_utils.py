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

"""SE(3) Lie group utilities for pycunls examples (CPU / NumPy).

All functions operate on NumPy float32 arrays to match cuNLS's single-precision
convention.  They are intentionally kept simple (no Lie-algebra library
dependencies) and are meant for test-data generation and ground-truth
comparison on the host — not for GPU computation.

Conventions
-----------
* A rigid-body transform ``T`` is a 4 x 4 homogeneous matrix stored in
  **row-major** order (same as NumPy's default C layout and cuNLS's on-device
  representation).
* A twist vector is ``[omega (3), rho (3)]``, where ``omega`` is the rotation
  axis-angle and ``rho`` is the translational velocity.
"""

import numpy as np


def skew(v):
    """Return the 3 x 3 skew-symmetric (hat) matrix of a 3-vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float32)


def twist_to_se3(twist):
    """SE(3) exponential map: 6-vector ``[omega, rho]`` → 4 x 4 matrix.

    Uses Rodrigues' formula for the rotation and the closed-form V matrix
    for the translational part of the exponential map.  When the rotation
    angle ``theta`` is near zero the first-order approximation is used.

    The V matrix formula (with ``K = skew(omega / theta)``):

        V = I + ((1 - cos θ) / θ) K + ((θ - sin θ) / θ) K²

    Note: because ``K`` is the skew of the *unit* rotation axis, the
    standard textbook expressions ``(1 - cos θ) / θ²`` and
    ``(θ - sin θ) / θ³`` are multiplied by ``θ`` from the normalisation
    factor in ``K = skew(omega / theta) = skew(axis) / 1``, yielding
    ``(1 - cos θ) / θ`` and ``(θ - sin θ) / θ``.
    """
    omega = twist[:3]
    rho = twist[3:]
    theta = np.linalg.norm(omega)
    T = np.eye(4, dtype=np.float32)
    if theta < 1e-8:
        T[:3, 3] = rho
        return T
    K = skew(omega / theta)
    R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    V = (np.eye(3, dtype=np.float32)
         + ((1 - np.cos(theta)) / theta) * K
         + ((theta - np.sin(theta)) / theta) * (K @ K))
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_inverse(T):
    """Invert a 4 x 4 SE(3) matrix.  Exploits R^{-1} = R^T for rotations."""
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_se3(A, B):
    """Compose two SE(3) transforms: ``A @ B``."""
    return (A @ B).astype(np.float32)


def project_normalized(T, point):
    """Project a 3D world point into normalised image coordinates via pose ``T``.

    Returns a 2-vector ``[x/z, y/z]`` in the camera frame.
    """
    p_cam = T[:3, :3] @ point + T[:3, 3]
    return np.array([p_cam[0] / p_cam[2], p_cam[1] / p_cam[2]], dtype=np.float32)


def compute_depth(T, point):
    """Return the z-coordinate (depth) of a 3D world point in camera frame."""
    return (T[:3, :3] @ point + T[:3, 3])[2]
