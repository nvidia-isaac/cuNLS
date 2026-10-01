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

#include <cmath>
#include <cstdint>
#include <vector>

#include "cunls/common/types.h"
#include "utils/se3_utils.h"

namespace examples {

// Mean squared error between two equally-sized vectors of cunls::Vector<N>.
template <size_t N>
float ComputeVectorMSE(const std::vector<cunls::Vector<N>> &a,
                       const std::vector<cunls::Vector<N>> &b) {
  float mse = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      const float d = a[i][j] - b[i][j];
      mse += d * d;
    }
  }
  return mse / static_cast<float>(a.size());
}

// Mean squared Frobenius error between two equally-sized vectors of SE(3)
// transforms, measuring element-wise deviation of 4x4 matrices.
inline float ComputePoseMSE(const std::vector<SE3Transform> &a,
                            const std::vector<SE3Transform> &b) {
  float mse = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < 16; ++j) {
      const float d = a[i][j] - b[i][j];
      mse += d * d;
    }
  }
  return mse / static_cast<float>(a.size());
}

// Pose-chain validation metric.
// For each consecutive pair (T_i, T_{i+1}), compute
//   T_res = delta_i * T_i^{-1} * T_{i+1}
// and measure the mean squared distance to the identity matrix.
inline float ChainConstraintError(const std::vector<SE3Transform> &poses,
                                  const std::vector<SE3Transform> &deltas) {
  float mse = 0.0f;
  const size_t num_constraints = deltas.size();
  for (size_t i = 0; i < num_constraints; ++i) {
    const SE3Transform pose_inv = InverseSE3(poses[i]);
    const SE3Transform residual_tf = ComposeSE3(deltas[i], ComposeSE3(pose_inv, poses[i + 1]));

    float err = 0.0f;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        const float target = (r == c) ? 1.0f : 0.0f;
        const float diff = residual_tf[r * 4 + c] - target;
        err += diff * diff;
      }
    }
    mse += err;
  }
  return mse / static_cast<float>(num_constraints);
}

// Rotation angle (degrees) between two poses' rotations, as
// atan2(|axis part|, cos part) of R_a^T R_b: accurate for small angles and
// exactly 0 for a == b.
inline float RotationErrorDeg(const SE3Transform &a, const SE3Transform &b) {
  double m[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m[i][j] = 0.0;
      for (int k = 0; k < 3; ++k) m[i][j] += double(a[k * 4 + i]) * b[k * 4 + j];
    }
  }
  const double c = 0.5 * (m[0][0] + m[1][1] + m[2][2] - 1.0);
  const double sx = 0.5 * (m[2][1] - m[1][2]);
  const double sy = 0.5 * (m[0][2] - m[2][0]);
  const double sz = 0.5 * (m[1][0] - m[0][1]);
  return static_cast<float>(std::atan2(std::sqrt(sx * sx + sy * sy + sz * sz), c) * 180.0 / M_PI);
}

// Euclidean distance between two poses' translations.
inline float TranslationError(const SE3Transform &a, const SE3Transform &b) {
  return std::hypot(std::hypot(a[3] - b[3], a[7] - b[7]), a[11] - b[11]);
}

// An inlier mask (1 = inlier) compared with the generated ground truth.
struct InlierMaskStats {
  size_t true_inliers = 0;       // factors that really are inliers
  size_t kept_inliers = 0;       // true inliers the mask accepts
  size_t accepted_outliers = 0;  // outliers the mask wrongly accepts
};

inline InlierMaskStats CompareInlierMask(const std::vector<uint8_t> &mask,
                                         const std::vector<uint8_t> &is_outlier) {
  InlierMaskStats s;
  for (size_t i = 0; i < mask.size(); ++i) {
    s.true_inliers += !is_outlier[i];
    s.kept_inliers += mask[i] && !is_outlier[i];
    s.accepted_outliers += mask[i] && is_outlier[i];
  }
  return s;
}

}  // namespace examples
