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

"""A colored point cloud of a TartanGround environment, fused from its depth images.

Every trajectory of the environment (``<env>/Data_*/P*``) contributes every
``frame_stride``-th frame of its front camera: the depth image is
back-projected through the camera's ground-truth pose and the points are
merged on a voxel grid (one point per voxel, the mean color). The world frame
is z-up: TartanAir's NED (x north, y east, z down) with y and z negated.
The recorded trajectories (the robot's floor paths) are returned as well.
The result is cached in an ``.npz`` next to nothing else: pass the path.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

FOCAL, CENTER = 320.0, 320.0  # lcam_front pinhole, 640 x 640
NED_TO_ZUP = np.array([1.0, -1.0, -1.0])


def _depth(path):
    """Planar depth [m]: float32 bytes stored in the PNG's channels in B, G, R, A order."""
    rgba = np.asarray(Image.open(path))
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]).view("<f4")[..., 0]


def _frame_points(seq, k, pose, pixel_stride, max_depth):
    """World points (z-up) and colors of one frame."""
    depth = _depth(os.path.join(seq, "depth_lcam_front", f"{k:06d}_lcam_front_depth.png"))
    rgb = np.asarray(Image.open(os.path.join(seq, "image_lcam_front", f"{k:06d}_lcam_front.png")))
    v, u = np.mgrid[0:640:pixel_stride, 0:640:pixel_stride]
    z = depth[v, u]
    ok = (z > 0.2) & (z < max_depth)
    u, v, z = u[ok], v[ok], z[ok]
    # OpenCV camera (x right, y down, z forward) -> body FRD (x forward, y right, z down).
    body = np.stack([z, (u - CENTER) / FOCAL * z, (v - CENTER) / FOCAL * z], 1)
    R = Rotation.from_quat(pose[3:]).as_matrix()
    world = (body @ R.T + pose[:3]) * NED_TO_ZUP
    return world.astype(np.float32), rgb[v, u, :3]


def build_map(env_dir, cache, frame_stride=8, pixel_stride=4, voxel=0.05, max_depth=12.0):
    """Returns (points [M, 3] float32, colors [M, 3] uint8, trajectories [list of [n, 3]])."""
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        return data["points"], data["colors"], list(data["trajectories"])
    sums, trajectories = {}, []
    sequences = sorted(glob.glob(os.path.join(env_dir, "Data_*", "P*")))
    for seq in sequences:
        poses = np.loadtxt(os.path.join(seq, "pose_lcam_front.txt"))
        trajectories.append((poses[:, :3] * NED_TO_ZUP).astype(np.float32))
        frames = range(0, len(poses), frame_stride)
        with ThreadPoolExecutor(16) as pool:
            parts = list(pool.map(
                lambda k: _frame_points(seq, k, poses[k], pixel_stride, max_depth), frames))
        pts = np.concatenate([p for p, _ in parts])
        col = np.concatenate([c for _, c in parts]).astype(np.float64)
        keys, inverse = np.unique(np.floor(pts / voxel).astype(np.int32), axis=0,
                                  return_inverse=True)
        inverse = inverse.reshape(-1)
        count = np.bincount(inverse)
        mean_p = np.stack([np.bincount(inverse, pts[:, i]) for i in range(3)], 1) / count[:, None]
        mean_c = np.stack([np.bincount(inverse, col[:, i]) for i in range(3)], 1) / count[:, None]
        for key, p, c, n in zip(map(tuple, keys), mean_p, mean_c, count):
            if key in sums:
                q, d, m = sums[key]
                sums[key] = ((q * m + p * n) / (m + n), (d * m + c * n) / (m + n), m + n)
            else:
                sums[key] = (p, c, n)
        print(f"  fused {os.path.relpath(seq, env_dir)}: {len(frames)} frames, "
              f"{len(sums)} voxels so far")
    points = np.array([v[0] for v in sums.values()], dtype=np.float32)
    colors = np.array([v[1] for v in sums.values()]).clip(0, 255).astype(np.uint8)
    traj = np.empty(len(trajectories), dtype=object)
    traj[:] = trajectories
    np.savez_compressed(cache, points=points, colors=colors, trajectories=traj)
    return points, colors, trajectories
