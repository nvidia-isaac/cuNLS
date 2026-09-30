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

#include <algorithm>
#include <cmath>

#include "cunls/common/helper.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace ransac_test {

namespace {

constexpr int kThreads = 256;

void Skew(const double w[3], double s[3][3]) {
  s[0][0] = 0;
  s[0][1] = -w[2];
  s[0][2] = w[1];
  s[1][0] = w[2];
  s[1][1] = 0;
  s[1][2] = -w[0];
  s[2][0] = -w[1];
  s[2][1] = w[0];
  s[2][2] = 0;
}

}  // namespace

SE3Transform Identity() {
  SE3Transform t{};
  for (int i = 0; i < 16; ++i) {
    t[i] = (i % 5 == 0) ? 1.f : 0.f;
  }
  return t;
}

SE3Transform ExpSE3(const std::array<double, 6> &twist) {
  const double w[3] = {twist[0], twist[1], twist[2]};
  const double v[3] = {twist[3], twist[4], twist[5]};
  const double theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const double theta = std::sqrt(theta2);
  double a, b, c;
  if (theta2 < 1e-12) {
    a = 1.0 - theta2 / 6.0;
    b = 0.5 - theta2 / 24.0;
    c = 1.0 / 6.0 - theta2 / 120.0;
  } else {
    a = std::sin(theta) / theta;
    b = (1.0 - std::cos(theta)) / theta2;
    c = (1.0 - a) / theta2;
  }
  double s[3][3];
  Skew(w, s);
  double s2[3][3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        s2[i][j] += s[i][k] * s[k][j];
      }
    }
  }
  SE3Transform out = Identity();
  double jv[3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double id = i == j ? 1.0 : 0.0;
      out[i * 4 + j] = static_cast<float>(id + a * s[i][j] + b * s2[i][j]);
      jv[i] += (id + b * s[i][j] + c * s2[i][j]) * v[j];
    }
    out[i * 4 + 3] = static_cast<float>(jv[i]);
  }
  return out;
}

SE3Transform Compose(const SE3Transform &a, const SE3Transform &b) {
  SE3Transform out{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) {
        s += static_cast<double>(a[i * 4 + k]) * b[k * 4 + j];
      }
      out[i * 4 + j] = static_cast<float>(s);
    }
  }
  return out;
}

SE3Transform Inverse(const SE3Transform &a) {
  SE3Transform out = Identity();
  for (int i = 0; i < 3; ++i) {
    double t = 0;
    for (int j = 0; j < 3; ++j) {
      out[i * 4 + j] = a[j * 4 + i];
      t -= static_cast<double>(a[j * 4 + i]) * a[j * 4 + 3];
    }
    out[i * 4 + 3] = static_cast<float>(t);
  }
  return out;
}

double RotationErrorDeg(const SE3Transform &a, const SE3Transform &b) {
  double trace = 0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      trace += static_cast<double>(a[i * 4 + k]) * b[i * 4 + k];
    }
  }
  const double c = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
  return std::acos(c) * 180.0 / M_PI;
}

double TranslationError(const SE3Transform &a, const SE3Transform &b) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    const double d = static_cast<double>(a[i * 4 + 3]) - b[i * 4 + 3];
    s += d * d;
  }
  return std::sqrt(s);
}

std::array<double, 3> Transform(const SE3Transform &pose, const std::array<double, 3> &x) {
  std::array<double, 3> y{};
  for (int i = 0; i < 3; ++i) {
    y[i] = pose[i * 4 + 3];
    for (int j = 0; j < 3; ++j) {
      y[i] += static_cast<double>(pose[i * 4 + j]) * x[j];
    }
  }
  return y;
}

std::array<double, 6> RandomTwist(std::mt19937 &rng, double rot, double trans) {
  std::normal_distribution<double> n(0.0, 1.0);
  std::array<double, 6> t{};
  double wn = 0, vn = 0;
  for (int i = 0; i < 3; ++i) {
    t[i] = n(rng);
    t[i + 3] = n(rng);
    wn += t[i] * t[i];
    vn += t[i + 3] * t[i + 3];
  }
  wn = std::sqrt(wn);
  vn = std::sqrt(vn);
  for (int i = 0; i < 3; ++i) {
    t[i] *= rot / std::max(wn, 1e-12);
    t[i + 3] *= trans / std::max(vn, 1e-12);
  }
  return t;
}

PnPScene MakePnPScene(size_t num_points, double outlier_ratio, double noise_sigma,
                      double min_outlier_error, uint32_t seed, const SE3Transform *world_to_cam) {
  std::mt19937 rng(seed);
  PnPScene scene;
  if (world_to_cam != nullptr) {
    scene.world_to_cam = *world_to_cam;
  } else {
    scene.world_to_cam = ExpSE3(RandomTwist(rng, 0.6, 1.5));
  }
  const SE3Transform cam_to_world = Inverse(scene.world_to_cam);
  std::uniform_real_distribution<double> fov(-0.6, 0.6);
  std::uniform_real_distribution<double> depth(2.0, 10.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, noise_sigma);
  scene.points_world.resize(num_points);
  scene.observations.resize(num_points);
  scene.is_outlier.assign(num_points, 0);
  const size_t num_outliers = static_cast<size_t>(std::llround(outlier_ratio * num_points));
  std::vector<size_t> order(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    order[i] = i;
  }
  std::shuffle(order.begin(), order.end(), rng);
  for (size_t k = 0; k < num_outliers; ++k) {
    scene.is_outlier[order[k]] = 1;
  }
  for (size_t i = 0; i < num_points; ++i) {
    const double z = depth(rng);
    const std::array<double, 3> pc = {fov(rng) * z, fov(rng) * z, z};
    const std::array<double, 3> pw = Transform(cam_to_world, pc);
    for (int k = 0; k < 3; ++k) {
      scene.points_world[i][k] = static_cast<float>(pw[k]);
    }
    // Project the float-rounded point so inliers are exact up to the noise.
    std::array<double, 3> pwf = {scene.points_world[i][0], scene.points_world[i][1],
                                 scene.points_world[i][2]};
    const std::array<double, 3> p = Transform(scene.world_to_cam, pwf);
    const double u = p[0] / p[2];
    const double v = p[1] / p[2];
    double ou = u + noise(rng);
    double ov = v + noise(rng);
    if (scene.is_outlier[i]) {
      do {
        ou = fov(rng);
        ov = fov(rng);
      } while (std::hypot(ou - u, ov - v) < min_outlier_error);
    }
    scene.observations[i][0] = static_cast<float>(ou);
    scene.observations[i][1] = static_cast<float>(ov);
  }
  return scene;
}

PnPScene MakeCoherentPnPScene(size_t num_points, double outlier_ratio, double noise_sigma,
                              double min_outlier_error, uint32_t seed,
                              const std::array<double, 6> &outlier_twist) {
  std::mt19937 rng(seed);
  PnPScene scene;
  scene.world_to_cam = ExpSE3(RandomTwist(rng, 0.6, 1.5));
  const SE3Transform wrong = Compose(scene.world_to_cam, ExpSE3(outlier_twist));
  const SE3Transform cam_to_world = Inverse(scene.world_to_cam);
  std::uniform_real_distribution<double> fov(-0.6, 0.6);
  std::uniform_real_distribution<double> depth(2.0, 10.0);
  std::normal_distribution<double> noise(0.0, noise_sigma);
  scene.points_world.resize(num_points);
  scene.observations.resize(num_points);
  scene.is_outlier.assign(num_points, 0);
  const size_t num_outliers = static_cast<size_t>(std::llround(outlier_ratio * num_points));
  std::vector<size_t> order(num_points);
  for (size_t i = 0; i < num_points; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  for (size_t k = 0; k < num_outliers; ++k) scene.is_outlier[order[k]] = 1;
  auto project = [](const SE3Transform &pose, const Vector<3> &x) {
    const std::array<double, 3> p = Transform(pose, {x[0], x[1], x[2]});
    return std::array<double, 3>{p[0] / p[2], p[1] / p[2], p[2]};
  };
  for (size_t i = 0; i < num_points; ++i) {
    for (int attempt = 0;; ++attempt) {
      const double z = depth(rng);
      const std::array<double, 3> pw = Transform(cam_to_world, {fov(rng) * z, fov(rng) * z, z});
      for (int k = 0; k < 3; ++k) scene.points_world[i][k] = static_cast<float>(pw[k]);
      const auto truth = project(scene.world_to_cam, scene.points_world[i]);
      const auto seen = scene.is_outlier[i] ? project(wrong, scene.points_world[i]) : truth;
      const double err = std::hypot(seen[0] - truth[0], seen[1] - truth[1]);
      if (!scene.is_outlier[i] || (seen[2] > 0.5 && err >= min_outlier_error) || attempt > 1000) {
        scene.observations[i][0] = static_cast<float>(seen[0] + noise(rng));
        scene.observations[i][1] = static_cast<float>(seen[1] + noise(rng));
        break;
      }
    }
  }
  return scene;
}

LinearScene MakeLinearScene(int dim, size_t num_points, double outlier_ratio, double noise_sigma,
                            double min_outlier_error, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, noise_sigma);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  LinearScene s;
  s.dim = dim;
  s.x_true.resize(dim);
  for (int k = 0; k < dim; ++k) {
    s.x_true[k] = static_cast<float>(n01(rng));
  }
  s.a.resize(num_points * dim);
  s.y.resize(num_points);
  s.is_outlier.assign(num_points, 0);
  const size_t num_outliers = static_cast<size_t>(std::llround(outlier_ratio * num_points));
  std::vector<size_t> order(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    order[i] = i;
  }
  std::shuffle(order.begin(), order.end(), rng);
  for (size_t k = 0; k < num_outliers; ++k) {
    s.is_outlier[order[k]] = 1;
  }
  for (size_t i = 0; i < num_points; ++i) {
    double y = 0;
    for (int k = 0; k < dim; ++k) {
      const float a = static_cast<float>(n01(rng));
      s.a[i * dim + k] = a;
      y += static_cast<double>(a) * s.x_true[k];
    }
    if (s.is_outlier[i]) {
      const double off = min_outlier_error + 10.0 * unit(rng);
      y += unit(rng) < 0.5 ? -off : off;
    } else {
      y += noise(rng);
    }
    s.y[i] = static_cast<float>(y);
  }
  return s;
}

// ----------------------------------------------------------------------------
// Kernels of the custom factors
// ----------------------------------------------------------------------------

namespace {

__global__ void LinearRegressionKernel(int dim, const float *a, const float *y, int n,
                                       float *residuals, float *jacobians,
                                       float const *const *state_pointers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float *x = state_pointers[i];
  const float *ai = a + static_cast<size_t>(i) * dim;
  float r = -y[i];
  for (int k = 0; k < dim; ++k) {
    r = fmaf(ai[k], x[k], r);
  }
  if (residuals != nullptr) {
    residuals[i] = r;
  }
  if (jacobians != nullptr) {
    for (int k = 0; k < dim; ++k) {
      jacobians[static_cast<size_t>(i) * dim + k] = ai[k];
    }
  }
}

__global__ void FocalPnPKernel(const Vector<2> *obs, const Vector<3> *points, int n,
                               float *residuals, float *jacobians,
                               float const *const *state_pointers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float *pose = state_pointers[2 * i];
  const float f = state_pointers[2 * i + 1][0];
  const Vector<3> &P = points[i];
  const float xc = pose[3] + pose[0] * P[0] + pose[1] * P[1] + pose[2] * P[2];
  const float yc = pose[7] + pose[4] * P[0] + pose[5] * P[1] + pose[6] * P[2];
  const float zc = pose[11] + pose[8] * P[0] + pose[9] * P[1] + pose[10] * P[2];
  float *r = residuals + 2 * i;
  if (zc < 1e-3f) {
    r[0] = 0.f;
    r[1] = 0.f;
    if (jacobians != nullptr) {
      for (int k = 0; k < 14; ++k) {
        jacobians[14 * i + k] = 0.f;
      }
    }
    return;
  }
  const float iz = 1.f / zc;
  const float u = xc * iz;
  const float v = yc * iz;
  r[0] = f * u - obs[i][0];
  r[1] = f * v - obs[i][1];
  if (jacobians == nullptr) {
    return;
  }
  // d(u, v)/d(world point), as in PnPFactorBatch; pose Jacobian for x (+) d = x exp(d).
  const float iz2 = iz * iz;
  float jp[2][3];
  for (int j = 0; j < 3; ++j) {
    jp[0][j] = pose[j] * iz - pose[8 + j] * iz2 * xc;
    jp[1][j] = pose[4 + j] * iz - pose[8 + j] * iz2 * yc;
  }
  float *J = jacobians + 14 * i;  // row-major 2 x (6 + 1)
  for (int row = 0; row < 2; ++row) {
    float *Jr = J + row * 7;
    Jr[0] = f * (P[1] * jp[row][2] - jp[row][1] * P[2]);
    Jr[1] = f * (P[2] * jp[row][0] - jp[row][2] * P[0]);
    Jr[2] = f * (P[0] * jp[row][1] - jp[row][0] * P[1]);
    Jr[3] = f * jp[row][0];
    Jr[4] = f * jp[row][1];
    Jr[5] = f * jp[row][2];
    Jr[6] = row == 0 ? u : v;
  }
}

}  // namespace

void LaunchLinearRegression(int dim, const float *a, const float *y, size_t num_factors,
                            float *residuals, float *jacobians,
                            float const *const *state_pointers, cudaStream_t stream) {
  if (num_factors == 0) {
    return;
  }
  const int blocks = static_cast<int>((num_factors + kThreads - 1) / kThreads);
  LinearRegressionKernel<<<blocks, kThreads, 0, stream>>>(dim, a, y, static_cast<int>(num_factors),
                                                          residuals, jacobians, state_pointers);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

bool FocalPnPFactorBatch::Evaluate(float *residuals, float *jacobians,
                                   float const *const *state_pointers,
                                   cudaStream_t stream) const {
  if (num_factors_ == 0) {
    return true;
  }
  const int blocks = static_cast<int>((num_factors_ + kThreads - 1) / kThreads);
  FocalPnPKernel<<<blocks, kThreads, 0, stream>>>(obs_, points_, static_cast<int>(num_factors_),
                                                  residuals, jacobians, state_pointers);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

bool SyncingFactorBatch::Evaluate(float *residuals, float *jacobians,
                                  float const *const *state_pointers,
                                  cudaStream_t stream) const {
  const bool ok = inner_->Evaluate(residuals, jacobians, state_pointers, stream);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return ok;
}

}  // namespace ransac_test
}  // namespace cunls
