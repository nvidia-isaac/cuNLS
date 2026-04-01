/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cunls/common/types.h"

namespace examples {

using cunls::SE3Transform;
using cunls::Vector;

// Depth of a world point in a camera frame (positive means in front).
inline float ComputeDepth(const SE3Transform& pose, const Vector<3>& point) {
  float depth = pose[2 * 4 + 3];
  for (int j = 0; j < 3; ++j) {
    depth += pose[2 * 4 + j] * point[j];
  }
  return depth;
}

// Project a world point to normalized image coordinates through pose T:
//   p_cam = T * p_world,  obs = (p_cam.x / p_cam.z,  p_cam.y / p_cam.z)
inline Vector<2> ProjectNormalized(const SE3Transform& pose,
                                   const Vector<3>& point) {
  float point_cam[3];
  point_cam[0] = pose[3];
  point_cam[1] = pose[7];
  point_cam[2] = pose[11];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      point_cam[i] += pose[i * 4 + j] * point[j];
    }
  }

  Vector<2> obs;
  const float inv_z = 1.0f / point_cam[2];
  obs[0] = point_cam[0] * inv_z;
  obs[1] = point_cam[1] * inv_z;
  return obs;
}

}  // namespace examples
