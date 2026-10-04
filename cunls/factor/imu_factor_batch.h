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

#include <cuda_runtime.h>

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Sensor model of an ImuFactorBatch: gravity, noise densities and the
 * IMU-body extrinsic.
 *
 * Noise densities are continuous-time: a sample over Δt has variance σ² / Δt
 * (white noise) and a random walk grows by σ² Δt.
 */
struct ImuParameters {
  /// World-frame gravity [m/s²]. Default: +Z up (ROS REP-103). A wrong
  /// direction (e.g. a Z-down / NED world) still converges, to a wrong answer.
  Vector<3> gravity = {0.f, 0.f, -9.80665f};
  // Noise defaults: the ADIS16448 of the EuRoC MAV dataset.
  float gyro_noise_density = 1.6968e-4f;  ///< σ_g [rad/s/√Hz]
  float accel_noise_density = 2.0e-3f;    ///< σ_a [m/s²/√Hz]
  /// σ_i [m/√s]: position noise of the Euler step (as GTSAM's integration
  /// covariance). Must be positive: with one sample, velocity and position
  /// errors are otherwise perfectly correlated.
  float integration_noise_density = 1e-4f;
  /// σ_bg [rad/s²/√Hz] and σ_ba [m/s³/√Hz]. Values this small make the bias
  /// rows very stiff against the rest of a problem (weight 1 / (σ_b √T), about
  /// 1e5 for the gyro over 0.2 s); in float32 the outer linear solve then
  /// loses the weak directions and Levenberg-Marquardt stalls. Loosen them
  /// (about 1e-2 / 1e-1) if it does.
  float gyro_bias_random_walk = 1.9393e-5f;
  float accel_bias_random_walk = 3.0e-3f;
  /// Pose of the IMU in the body (rig) frame, rig_from_imu; identity when
  /// the IMU is the body frame.
  SE3Transform body_from_imu = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
};

/**
 * @brief IMU factor between two keyframes, with the raw samples between them
 * marginalized inside the factor (Schur complement, recomputed every
 * evaluation; no preintegration cache, no reference bias).
 *
 * States (in order): pose X_a (SE3StateBatch, body_from_world: the same rig
 * pose as ReprojectionFactorBatch and PnPFactorBatch read, so the factors can
 * share it), velocity v_a (VectorStateBatch<3>), bias b_a = [b_g; b_a]
 * (VectorStateBatch<6>), and X_b, v_b, b_b at the next keyframe. v is the
 * world-frame velocity of the IMU (the body velocity when the IMU sits at the
 * body origin). The IMU's pose in the world is X⁻¹ body_from_imu.
 *
 * The N samples (ω̃_k, ã_k, Δt_k) between the keyframes define a chain of
 * Euler steps between N - 1 intermediate states x_k = (R_k, v_k, p_k) of the
 * IMU, with the bias of keyframe a:
 *
 * @verbatim
 *   R_{k+1} = R_k Exp((ω̃_k - b_g) Δt_k)
 *   v_{k+1} = v_k + g Δt_k + R_k (ã_k - b_a) Δt_k
 *   p_{k+1} = p_k + v_k Δt_k + ½ g Δt_k² + ½ R_k (ã_k - b_a) Δt_k²
 * @endverbatim
 *
 * each step's defect weighted by its noise (σ_g, σ_a, σ_i). Evaluate
 * linearizes the chain at the states integrated forward from keyframe a,
 * eliminates the intermediate states and returns the marginal factor on the
 * keyframes: its J^T J and J^T r are the Schur complement of the chain's
 * normal equations onto the keyframe states, and |r|² is the chain's
 * linearized cost minimized over the intermediate states. The elimination
 * runs as the covariance form of the Riccati recursion, Σ_{k+1} = Φ_k Σ_k
 * Φ_kᵀ + Q_k: same result as the information form, but the information form
 * loses about five digits in float32 over 200 samples. With
 * e = [Log(R̂ᵀ R_b); v_b - v̂; p_b - p̂] the defect of keyframe b (IMU frame)
 * against the prediction and Σ = L Lᵀ:
 *
 * @verbatim
 *   residual[0:9]   = L⁻¹ e
 *   residual[9:15]  = (b_b - b_a) / (σ_b √(Σ Δt_k))      (bias random walk)
 * @endverbatim
 *
 * The rank of the marginal is 15 (9 chain + 6 bias), which is why the
 * residual has 15 rows and not one per keyframe tangent dimension (30).
 * Analytic Jacobians (Σ and the whitening treated as constant, as usual for
 * Gauss-Newton). Each factor's chain runs on 1 to 32 threads: runs of samples
 * reduce to summaries that compose associatively (as preintegrated deltas
 * do), so small batches split chains over a warp and large batches give each
 * thread a whole chain.
 */
class ImuFactorBatch : public SizedFactorBatch<15, 6, 3, 6, 6, 3, 6> {
 public:
  /**
   * @param imu_samples Device array of 7 floats per sample, (ω_x, ω_y, ω_z,
   *        a_x, a_y, a_z, Δt): gyro [rad/s] and specific force [m/s²] in the
   *        IMU frame, measured at the start of the step, and the step
   *        duration [s]. Samples of all factors back to back.
   * @param sample_offsets Device array of capacity + 1 offsets (CSR row
   *        pointers): factor f uses samples [offsets[f], offsets[f + 1]); at
   *        least one, with positive Δt, the first starting at keyframe a and
   *        the last ending at keyframe b.
   * @param num_samples Number of samples imu_samples holds. Only sizes the
   *        work split: num_samples / capacity is taken as the typical chain
   *        length, and short chains in small batches are split over several
   *        threads.
   * @param parameters Gravity, noise densities, extrinsic.
   * @param capacity Number of factors (keyframe pairs) the buffers hold. The
   *        active count starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if a pointer is null or a noise density is
   *         not positive.
   */
  ImuFactorBatch(const float *imu_samples, const int *sample_offsets, size_t num_samples,
                 const ImuParameters &parameters, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

  const ImuParameters &Parameters() const { return parameters_; }

 private:
  const float *imu_samples_;
  const int *sample_offsets_;
  size_t num_samples_;
  ImuParameters parameters_;
};

}  // namespace cunls
