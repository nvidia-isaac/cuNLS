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

/**
 * @file ransac_test_support.h
 * @brief Synthetic scenes, host SE(3) helpers and custom (user-style) factor
 * batches shared by the RANSAC minimizer tests and benchmarks.
 */

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {
namespace ransac_test {

// ----------------------------------------------------------------------------
// Host SE(3) helpers (row-major 4x4, x_cam = R x_world + t)
// ----------------------------------------------------------------------------

/** exp of a twist (w, v) with the same closed form as SE3StateBatch::Plus. */
SE3Transform ExpSE3(const std::array<double, 6> &twist);
SE3Transform Compose(const SE3Transform &a, const SE3Transform &b);
SE3Transform Inverse(const SE3Transform &a);
SE3Transform Identity();
/** Rotation angle of R_a R_b^T in degrees. */
double RotationErrorDeg(const SE3Transform &a, const SE3Transform &b);
/** |t_a - t_b|. */
double TranslationError(const SE3Transform &a, const SE3Transform &b);
/** Applies x_cam = R x + t. */
std::array<double, 3> Transform(const SE3Transform &pose, const std::array<double, 3> &x);

/** Random twist with rotation angle `rot` (rad) and translation norm `trans`. */
std::array<double, 6> RandomTwist(std::mt19937 &rng, double rot, double trans);

// ----------------------------------------------------------------------------
// Scenes
// ----------------------------------------------------------------------------

/** A camera observing 3D points; some observations replaced by gross outliers. */
struct PnPScene {
  SE3Transform world_to_cam;
  std::vector<Vector<3>> points_world;
  std::vector<Vector<2>> observations;  ///< Normalized image coordinates.
  std::vector<uint8_t> is_outlier;
};

/**
 * @param noise_sigma Gaussian noise on inlier observations (normalized units).
 * @param min_outlier_error Outliers are resampled until their error exceeds this.
 */
PnPScene MakePnPScene(size_t num_points, double outlier_ratio, double noise_sigma,
                      double min_outlier_error, uint32_t seed,
                      const SE3Transform *world_to_cam = nullptr);

/**
 * Like MakePnPScene, but outliers are coherent: they are projections of their
 * points through a second pose world_to_cam * exp(outlier_twist) (plus the
 * same noise), i.e. a competing, self-consistent wrong model. Outliers whose
 * error w.r.t. the true pose is below min_outlier_error are re-drawn.
 */
PnPScene MakeCoherentPnPScene(size_t num_points, double outlier_ratio, double noise_sigma,
                              double min_outlier_error, uint32_t seed,
                              const std::array<double, 6> &outlier_twist);

/** Linear model y = a^T x_true + noise, outliers have y offset by >= min_outlier_error. */
struct LinearScene {
  int dim = 0;
  std::vector<float> x_true;
  std::vector<float> a;  ///< N * dim
  std::vector<float> y;  ///< N
  std::vector<uint8_t> is_outlier;
};

LinearScene MakeLinearScene(int dim, size_t num_points, double outlier_ratio, double noise_sigma,
                            double min_outlier_error, uint32_t seed);

// ----------------------------------------------------------------------------
// Custom factor batches (implemented like a user would, only Evaluate()).
// ----------------------------------------------------------------------------

/** Evaluates `num_items` items (see FactorBatch::Evaluate) of the linear regression factor. */
void LaunchLinearRegression(int dim, const float *a, const float *y, size_t num_factors,
                            float *residuals, float *jacobians, float const *const *state_pointers,
                            const int *factor_ids, size_t num_items, cudaStream_t stream);

/** r_i = a_i^T x - y_i with a VectorStateBatch<Dim> state. */
template <int Dim>
class LinearRegressionFactorBatch : public SizedFactorBatch<1, Dim> {
 public:
  LinearRegressionFactorBatch(const float *a, const float *y, size_t capacity)
      : SizedFactorBatch<1, Dim>(capacity), a_(a), y_(y) {}
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override {
    const size_t n = this->NumActiveFactors();
    LaunchLinearRegression(Dim, a_, y_, n, residuals, jacobians, state_pointers, factor_ids,
                           num_factor_ids == 0 ? n : num_factor_ids, stream);
    return true;
  }

 private:
  const float *a_;
  const float *y_;
};

/**
 * Pinhole projection with unknown focal length: r = f * (X/Z, Y/Z) - obs_pixels.
 * States: SE3 pose (6) and a Vector<1> focal length (1).
 */
class FocalPnPFactorBatch : public SizedFactorBatch<2, 6, 1> {
 public:
  FocalPnPFactorBatch(const Vector<2> *obs_pixels, const Vector<3> *points_world, size_t capacity)
      : SizedFactorBatch(capacity), obs_(obs_pixels), points_(points_world) {}
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const Vector<2> *obs_;
  const Vector<3> *points_;
};

/**
 * Forwards to another factor batch but synchronizes the stream inside
 * Evaluate(): a custom factor doing host-side work must still give the same
 * results.
 */
class SyncingFactorBatch : public FactorBatch {
 public:
  explicit SyncingFactorBatch(FactorBatch *inner) : inner_(inner) {}
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;
  size_t ResidualsSize() const override { return inner_->ResidualsSize(); }
  std::vector<size_t> StateSizes() const override { return inner_->StateSizes(); }
  size_t NumActiveFactors() const override { return inner_->NumActiveFactors(); }
  size_t Capacity() const override { return inner_->Capacity(); }
  void SetNumActiveFactors(size_t num_active_factors) override {
    inner_->SetNumActiveFactors(num_active_factors);
  }

 private:
  FactorBatch *inner_;
};

}  // namespace ransac_test
}  // namespace cunls
