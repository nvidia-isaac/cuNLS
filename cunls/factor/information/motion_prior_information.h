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

#include "cunls/common/cublas_helper.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"

namespace cunls {

/**
 * @brief Computes per-factor square-root information matrices for the
 * constant-velocity (white-noise-on-acceleration / WNOA) motion prior, for
 * direct use with InformationFactorBatch<ConstantVelocityXxxFactorBatch>.
 *
 * Closed-form derivation: the paper's information matrix for a 2-block
 * state `[twist; vel]` factors as a
 * Kronecker product `Q(dt)^-1 = M(dt) (x) Qc^-1` of a 2x2 scalar matrix
 *
 *     M(dt) = [[12/dt^3, -6/dt^2], [-6/dt^2, 4/dt]]
 *
 * with the (assumed diagonal) continuous-time process-noise PSD `Qc` (one
 * entry per tangent DOF). Because `chol(A (x) B) = chol(A) (x) chol(B)` for
 * SPD `A`, `B`, and `M(dt)`'s Cholesky factor has the closed form
 *
 *     L_M(dt) = [[2*sqrt(3)/dt^1.5, 0], [-sqrt(3)/dt^0.5, 1/dt^0.5]]
 *
 * (verified: `L_M L_M^T == M`), the full sqrt-information matrix is
 * `L_M(dt) (x) diag(1/sqrt(qc))` — computed directly per factor with no
 * numerical Cholesky needed. Output is row-major, `(2*Dim) x (2*Dim)`,
 * matching `InformationFactorBatch<ConstantVelocityXxxFactorBatch<Dim>>`'s
 * expected `Matrix<2*Dim>` layout.
 *
 * @tparam Dim Pose tangent dimension (6 for SE(3), 3 for SE(2)/SO(3), 1 for
 * SO(2)), matching the wrapped ConstantVelocityXxxFactorBatch.
 * @param stream CUDA stream for asynchronous execution.
 * @param dt_ptr Device pointer to per-factor time deltas (t_{k+1} - t_k), at
 * least num_factors floats.
 * @param qc_diag_ptr Device pointer to the continuous-time process-noise PSD
 * diagonal, Dim floats. Constant across the batch (a property of the noise
 * process, not of any individual factor).
 * @param num_factors Number of factors.
 * @param sqrt_information_out Device pointer to output square-root
 * information matrices, row-major `(2*Dim) x (2*Dim)` per factor
 * (`num_factors * (2*Dim) * (2*Dim)` floats total).
 */
template <int Dim>
void ComputeConstantVelocitySqrtInformation(cudaStream_t stream, const float *dt_ptr,
                                            const float *qc_diag_ptr, size_t num_factors,
                                            float *sqrt_information_out);

/**
 * @brief Computes per-factor square-root information matrices for the
 * constant-acceleration (white-noise-on-jerk / WNOJ) motion prior, for
 * direct use with InformationFactorBatch<ConstantAccelerationXxxFactorBatch>.
 *
 * Same derivation as ComputeConstantVelocitySqrtInformation, but for the
 * 3-block state `[twist; vel; accel]`, whose information matrix factors as
 * `Q(dt)^-1 = M(dt) (x) Qc^-1` with the paper's 3x3
 *
 *     M(dt) = [[720/dt^5, -360/dt^4, 60/dt^3],
 *              [-360/dt^4, 192/dt^3, -36/dt^2],
 *              [60/dt^3, -36/dt^2, 9/dt]]
 *
 * whose Cholesky factor also has closed form:
 *
 *     l11 = 12*sqrt(5)/dt^2.5
 *     l21 = -6*sqrt(5)/dt^1.5,  l22 = 2*sqrt(3)/dt^1.5
 *     l31 = sqrt(5)/dt^0.5,     l32 = -sqrt(3)/dt^0.5,   l33 = 1/dt^0.5
 *
 * (verified: `L_M L_M^T == M`). Output is row-major, `(3*Dim) x (3*Dim)`,
 * matching `InformationFactorBatch<ConstantAccelerationXxxFactorBatch<Dim>>`.
 *
 * @tparam Dim Pose tangent dimension (6 for SE(3), 3 for SE(2)/SO(3), 1 for
 * SO(2)), matching the wrapped ConstantAccelerationXxxFactorBatch.
 * @param stream CUDA stream for asynchronous execution.
 * @param dt_ptr Device pointer to per-factor time deltas, at least
 * num_factors floats.
 * @param qc_diag_ptr Device pointer to the continuous-time process-noise PSD
 * diagonal, Dim floats.
 * @param num_factors Number of factors.
 * @param sqrt_information_out Device pointer to output square-root
 * information matrices, row-major `(3*Dim) x (3*Dim)` per factor.
 */
template <int Dim>
void ComputeConstantAccelerationSqrtInformation(cudaStream_t stream, const float *dt_ptr,
                                                const float *qc_diag_ptr, size_t num_factors,
                                                float *sqrt_information_out);

namespace detail {

/**
 * @brief Base-from-member helper: computes and owns the sqrt-information
 * buffer for a motion-prior factor.
 *
 * InformationFactorBatch<T> only stores a non-owning pointer to a
 * caller-managed sqrt-information buffer, so MotionPriorInformationFactorBatch
 * needs somewhere to both compute *and keep alive* that buffer before (and
 * for the lifetime of) the InformationFactorBatch<T> base. Listing this
 * struct as an earlier base class than InformationFactorBatch<T> guarantees
 * it is fully constructed first, so its `sqrt_information.data()` is valid
 * device memory by the time InformationFactorBatch<T>'s constructor runs
 * (the "base-from-member" idiom).
 */
template <class T, int Dim>
struct MotionPriorSqrtInformationStorage {
  static constexpr size_t kMultiplier = T::residual_size_ / Dim;
  static_assert(T::residual_size_ == kMultiplier * Dim && (kMultiplier == 2 || kMultiplier == 3),
                "MotionPriorInformationFactorBatch<T, Dim>: T's residual size must be 2*Dim "
                "(constant-velocity) or 3*Dim (constant-acceleration) factors.");
  using InformationMatrix = Matrix<T::residual_size_>;

  DeviceVector<InformationMatrix> sqrt_information;

  MotionPriorSqrtInformationStorage(cudaStream_t stream, const float *dt_ptr,
                                    const float *qc_diag_ptr, size_t num_factors)
      : sqrt_information(num_factors) {
    if constexpr (kMultiplier == 2) {
      ComputeConstantVelocitySqrtInformation<Dim>(
          stream, dt_ptr, qc_diag_ptr, num_factors,
          reinterpret_cast<float *>(sqrt_information.data()));
    } else {
      ComputeConstantAccelerationSqrtInformation<Dim>(
          stream, dt_ptr, qc_diag_ptr, num_factors,
          reinterpret_cast<float *>(sqrt_information.data()));
    }
    // Evaluate() may be called with a different stream than the one used
    // here to compute sqrt_information, so the buffer must be fully
    // populated (not just enqueued) before this constructor returns.
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
};

}  // namespace detail

/**
 * @brief Constant-velocity/-acceleration motion-prior factor batch with the
 * paper's closed-form process-noise covariance Q(dt)^{-1} fused directly
 * into the residual and Jacobian.
 *
 * Thin convenience wrapper: internally computes the square-root information
 * matrices with ComputeConstantVelocitySqrtInformation /
 * ComputeConstantAccelerationSqrtInformation (selected automatically from
 * T's residual size) and composes them with the generic
 * InformationFactorBatch<T>, so callers never construct or manage a
 * sqrt-information buffer, or need to know about the Kronecker-product
 * structure behind it, themselves. Prefer the named aliases below (e.g.
 * ConstantVelocityInformationSE3FactorBatch) over spelling out this
 * template directly.
 *
 * @tparam T One of the ConstantVelocityXxxFactorBatch /
 * ConstantAccelerationXxxFactorBatch classes.
 * @tparam Dim Pose tangent dimension T was instantiated for (6 for SE(3), 3
 * for SE(2)/SO(3), 1 for SO(2)).
 */
template <class T, int Dim>
class MotionPriorInformationFactorBatch : private detail::MotionPriorSqrtInformationStorage<T, Dim>,
                                          public InformationFactorBatch<T> {
  using Storage = detail::MotionPriorSqrtInformationStorage<T, Dim>;
  using Base = InformationFactorBatch<T>;

 public:
  /**
   * @brief Constructs the motion-prior factor batch with fused covariance
   * weighting.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param stream CUDA stream used to compute the sqrt-information matrices
   * up front (the factor's own Evaluate() still takes its stream as a
   * per-call argument, as usual).
   * @param dt_ptr Device pointer to per-factor time deltas (t_{k+1} - t_k).
   * Must remain valid for the lifetime of this object (also forwarded to
   * T's own constructor).
   * @param qc_diag_ptr Device pointer to the continuous-time process-noise
   * PSD diagonal (Dim floats), constant across the batch.
   * @param num_factors Number of factors in the batch.
   */
  MotionPriorInformationFactorBatch(cuBLASHandle &cublas_handle, cudaStream_t stream,
                                    const float *dt_ptr, const float *qc_diag_ptr,
                                    size_t num_factors)
      : Storage(stream, dt_ptr, qc_diag_ptr, num_factors),
        Base(cublas_handle,
             reinterpret_cast<const typename Base::InformationMatrix *>(
                 this->Storage::sqrt_information.data()),
             num_factors, dt_ptr, num_factors) {}
};

// Named aliases: prefer these over spelling out MotionPriorInformationFactorBatch<T, Dim>.
using ConstantVelocityInformationSE3FactorBatch =
    MotionPriorInformationFactorBatch<ConstantVelocitySE3FactorBatch, 6>;
using ConstantVelocityInformationSO3FactorBatch =
    MotionPriorInformationFactorBatch<ConstantVelocitySO3FactorBatch, 3>;
using ConstantVelocityInformationSE2FactorBatch =
    MotionPriorInformationFactorBatch<ConstantVelocitySE2FactorBatch, 3>;
using ConstantVelocityInformationSO2FactorBatch =
    MotionPriorInformationFactorBatch<ConstantVelocitySO2FactorBatch, 1>;
using ConstantAccelerationInformationSE3FactorBatch =
    MotionPriorInformationFactorBatch<ConstantAccelerationSE3FactorBatch, 6>;
using ConstantAccelerationInformationSO3FactorBatch =
    MotionPriorInformationFactorBatch<ConstantAccelerationSO3FactorBatch, 3>;
using ConstantAccelerationInformationSE2FactorBatch =
    MotionPriorInformationFactorBatch<ConstantAccelerationSE2FactorBatch, 3>;
using ConstantAccelerationInformationSO2FactorBatch =
    MotionPriorInformationFactorBatch<ConstantAccelerationSO2FactorBatch, 1>;

}  // namespace cunls
