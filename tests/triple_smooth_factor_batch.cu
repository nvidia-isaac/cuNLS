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

#include <cuda_runtime.h>

#include <cmath>

#include "tests/triple_smooth_factor_batch.h"

namespace cunls {
namespace test_utils {
namespace {
constexpr int kThreadsPerBlock = 128;
// Step of the central differences, in tangent units (radians, meters).
constexpr double kDifferenceStep = 1e-6;

// Rigid transform, double precision: p -> rotation * p + translation.
struct Pose {
  double rotation[9];
  double translation[3];
};

__device__ Pose LoadPose(const float* matrix) {
  Pose pose;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      pose.rotation[row * 3 + col] = matrix[row * 4 + col];
    }
    pose.translation[row] = matrix[row * 4 + 3];
  }
  return pose;
}

__device__ Pose Multiply(const Pose& lhs, const Pose& rhs) {
  Pose out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += lhs.rotation[row * 3 + k] * rhs.rotation[k * 3 + col];
      }
      out.rotation[row * 3 + col] = sum;
    }
    double sum = lhs.translation[row];
    for (int k = 0; k < 3; ++k) {
      sum += lhs.rotation[row * 3 + k] * rhs.translation[k];
    }
    out.translation[row] = sum;
  }
  return out;
}

__device__ Pose Inverse(const Pose& pose) {
  Pose out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.rotation[row * 3 + col] = pose.rotation[col * 3 + row];
    }
  }
  for (int row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
      sum -= out.rotation[row * 3 + k] * pose.translation[k];
    }
    out.translation[row] = sum;
  }
  return out;
}

// out = I + a * K + b * K^2 for the skew matrix K of `axis`.
__device__ void SkewSeries(const double* axis, double a, double b, double* out) {
  const double k[9] = {0.0, -axis[2], axis[1], axis[2], 0.0, -axis[0], -axis[1], axis[0], 0.0};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double square = 0.0;
      for (int i = 0; i < 3; ++i) {
        square += k[row * 3 + i] * k[i * 3 + col];
      }
      out[row * 3 + col] = (row == col ? 1.0 : 0.0) + a * k[row * 3 + col] + b * square;
    }
  }
}

// Rotation matrix of the rotation vector `w` (Rodrigues).
__device__ void ExpSO3(const double* w, double* rotation) {
  const double theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const double theta = sqrt(theta2);
  double a = 1.0 - theta2 / 6.0;
  double b = 0.5 - theta2 / 24.0;
  if (theta > 1e-4) {
    a = sin(theta) / theta;
    b = (1.0 - cos(theta)) / theta2;
  }
  SkewSeries(w, a, b, rotation);
}

// Rotation vector of a rotation matrix, through the quaternion (the branch of
// its largest component), so that it is exact for every angle.
__device__ void LogSO3(const double* r, double* w) {
  const double trace = r[0] + r[4] + r[8];
  double q[4];  // w, x, y, z
  if (trace > 0.0) {
    const double s = 2.0 * sqrt(trace + 1.0);
    q[0] = 0.25 * s;
    q[1] = (r[7] - r[5]) / s;
    q[2] = (r[2] - r[6]) / s;
    q[3] = (r[3] - r[1]) / s;
  } else if (r[0] >= r[4] && r[0] >= r[8]) {
    const double s = 2.0 * sqrt(1.0 + r[0] - r[4] - r[8]);
    q[0] = (r[7] - r[5]) / s;
    q[1] = 0.25 * s;
    q[2] = (r[1] + r[3]) / s;
    q[3] = (r[2] + r[6]) / s;
  } else if (r[4] >= r[8]) {
    const double s = 2.0 * sqrt(1.0 + r[4] - r[0] - r[8]);
    q[0] = (r[2] - r[6]) / s;
    q[1] = (r[1] + r[3]) / s;
    q[2] = 0.25 * s;
    q[3] = (r[5] + r[7]) / s;
  } else {
    const double s = 2.0 * sqrt(1.0 + r[8] - r[0] - r[4]);
    q[0] = (r[3] - r[1]) / s;
    q[1] = (r[2] + r[6]) / s;
    q[2] = (r[5] + r[7]) / s;
    q[3] = 0.25 * s;
  }
  if (q[0] < 0.0) {
    for (int i = 0; i < 4; ++i) {
      q[i] = -q[i];
    }
  }
  const double norm = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double scale = norm < 1e-12 ? 2.0 / q[0] : 2.0 * atan2(norm, q[0]) / norm;
  for (int i = 0; i < 3; ++i) {
    w[i] = scale * q[i + 1];
  }
}

// SE(3) exponential of the twist [w; rho]: translation = J_l(w) rho.
__device__ Pose ExpSE3(const double* twist) {
  Pose pose;
  ExpSO3(twist, pose.rotation);
  const double theta2 = twist[0] * twist[0] + twist[1] * twist[1] + twist[2] * twist[2];
  const double theta = sqrt(theta2);
  double a = 0.5 - theta2 / 24.0;
  double b = 1.0 / 6.0 - theta2 / 120.0;
  if (theta > 1e-4) {
    a = (1.0 - cos(theta)) / theta2;
    b = (theta - sin(theta)) / (theta2 * theta);
  }
  double left_jacobian[9];
  SkewSeries(twist, a, b, left_jacobian);
  for (int row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
      sum += left_jacobian[row * 3 + k] * twist[3 + k];
    }
    pose.translation[row] = sum;
  }
  return pose;
}

// SE(3) logarithm: [w; rho] with w = Log(R), rho = J_l(w)^-1 t.
__device__ void LogSE3(const Pose& pose, double* twist) {
  LogSO3(pose.rotation, twist);
  const double theta2 = twist[0] * twist[0] + twist[1] * twist[1] + twist[2] * twist[2];
  const double theta = sqrt(theta2);
  double b = 1.0 / 12.0 + theta2 / 720.0;
  if (theta > 1e-4) {
    b = (1.0 - 0.5 * theta * sin(theta) / (1.0 - cos(theta))) / theta2;
  }
  double inverse_jacobian[9];
  SkewSeries(twist, -0.5, b, inverse_jacobian);
  for (int row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
      sum += inverse_jacobian[row * 3 + k] * pose.translation[k];
    }
    twist[3 + row] = sum;
  }
}

// The residual of a triple smooth factor, see TripleSmoothFactorBatch.
__device__ void Residual(const Pose& first, const Pose& middle, const Pose& last, double ratio,
                         double* residual) {
  double twist[6];
  LogSE3(Multiply(Inverse(first), last), twist);
  for (int i = 0; i < 6; ++i) {
    twist[i] *= ratio;
  }
  const Pose error = Multiply(Inverse(middle), Multiply(first, ExpSE3(twist)));
  LogSO3(error.rotation, residual);
  for (int i = 0; i < 3; ++i) {
    residual[3 + i] = error.translation[i];
  }
}

__device__ int MeasurementIndex(int item, const int* factor_ids, int num_factors) {
  if (factor_ids != nullptr) {
    return factor_ids[item];
  }
  return item < num_factors ? item : item % num_factors;
}

__global__ void TripleSmoothKernel(float const* const* state_pointers, const float* ratios,
                                   size_t num_items, const int* factor_ids, int num_factors,
                                   float* residuals, float* jacobians) {
  const int item = threadIdx.x + blockIdx.x * blockDim.x;
  if (item >= static_cast<int>(num_items)) {
    return;
  }
  const double ratio = ratios[MeasurementIndex(item, factor_ids, num_factors)];
  Pose poses[3];
  for (int i = 0; i < 3; ++i) {
    poses[i] = LoadPose(state_pointers[3 * item + i]);
  }

  double residual[6];
  Residual(poses[0], poses[1], poses[2], ratio, residual);
  for (int i = 0; i < 6; ++i) {
    residuals[6 * item + i] = static_cast<float>(residual[i]);
  }
  if (jacobians == nullptr) {
    return;
  }

  // Column 6 * state + k: the residual's response to T_state * Exp(h e_k).
  float* jacobian = jacobians + static_cast<size_t>(item) * 6 * 18;
  for (int state = 0; state < 3; ++state) {
    for (int k = 0; k < 6; ++k) {
      double residuals_by_sign[2][6];
      for (int sign = 0; sign < 2; ++sign) {
        double twist[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        twist[k] = sign == 0 ? kDifferenceStep : -kDifferenceStep;
        Pose moved[3] = {poses[0], poses[1], poses[2]};
        moved[state] = Multiply(poses[state], ExpSE3(twist));
        Residual(moved[0], moved[1], moved[2], ratio, residuals_by_sign[sign]);
      }
      for (int row = 0; row < 6; ++row) {
        jacobian[row * 18 + 6 * state + k] = static_cast<float>(
            (residuals_by_sign[0][row] - residuals_by_sign[1][row]) / (2.0 * kDifferenceStep));
      }
    }
  }
}

}  // namespace

TripleSmoothFactorBatch::TripleSmoothFactorBatch(const float* ratios, size_t capacity)
    : SizedFactorBatch(capacity), ratios_(ratios) {}

bool TripleSmoothFactorBatch::Evaluate(float* residuals, float* jacobians,
                                       float const* const* state_pointers, cudaStream_t stream,
                                       const int* factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) return true;
  const size_t num_blocks = (num_items + kThreadsPerBlock - 1) / kThreadsPerBlock;
  TripleSmoothKernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      state_pointers, ratios_, num_items, factor_ids, static_cast<int>(NumActiveFactors()),
      residuals, jacobians);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace test_utils
}  // namespace cunls
