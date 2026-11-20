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

namespace cunls {

// Batched SO(2), SE(2), SO(3) and SE(3) maps used by the built-in state and
// factor batches. Every function launches one kernel on `stream` (asynchronous)
// over `size` batch elements; inputs and outputs are device arrays, element i
// starts at `i * *_stride` floats, matrices are row-major with rows `*_pitch`
// floats apart. Tangent layouts: SO(2) [theta], SE(2) [v_x, v_y, theta],
// SO(3) [phi], SE(3) [phi, rho].

// ============================================================================
// SO(2) operations  (tangent dim 1, ambient 2x2)
// ============================================================================

/** @brief angles[i] = Log(R_i) = atan2(R_i[1][0], R_i[0][0]) for 2x2 rotations. */
void ComputeLogSO2(cudaStream_t stream, const float *rotations, size_t rotation_stride,
                   size_t angle_stride, size_t size, float *angles);

// ============================================================================
// SE(2) operations  (tangent dim 3, ambient 3x3)
// ============================================================================

/** @brief tangent[i] = Log(T_i) for 3x3 transforms (float32-safe near theta = 0). */
void ComputeLogSE2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                   size_t tangent_stride, size_t size, float *tangent);

/** @brief jacobians[i] = J_r^{-1}(xi_i) (3x3); J_l^{-1}(xi) = J_r^{-1}(-xi). */
void ComputeJacobianRightInverseSE2(cudaStream_t stream, const float *tangent,
                                    size_t tangent_stride, size_t jacobian_stride, size_t size,
                                    float *jacobians);

// ============================================================================
// SO(3) operations  (tangent dim 3, ambient 3x3)
// ============================================================================

/** @brief twist[i] = Log(R_i) for 3x3 rotations, accurate at every angle. */
void ComputeLogSO3(cudaStream_t stream, const float *rotation, const size_t rotation_pitch,
                   const size_t rotation_stride, const size_t twist_stride, size_t size,
                   float *twist);

/** @brief jacobian_inv[i] = J_r^{-1}(phi_i) = J_l^{-1}(-phi_i) of SO(3) (3x3). */
void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_inv_pitch,
                                    const size_t jacobian_inv_stride, size_t size,
                                    float *jacobian_inv);

// ============================================================================
// SE(3) operations  (tangent dim 6, ambient 4x4)
// ============================================================================

/** @brief transform[i] = Exp(xi_i) of SE(3), 4x4 with t = J_l(phi) rho. */
void ComputeExpSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                   const size_t transform_pitch, const size_t transform_stride, size_t size,
                   float *transform);

/** @brief twist[i] = Log(T_i) of SE(3) for 4x4 transforms. */
void ComputeLogSE3(cudaStream_t stream, const float *transform, const size_t transform_pitch,
                   const size_t transform_stride, const size_t twist_stride, size_t size,
                   float *twist);

/** @brief jacobian[i] = J_l(xi_i) of SE(3) (6x6): [[J_l(phi), 0], [Q(xi), J_l(phi)]]. */
void ComputeJacobianLeftSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                            const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                            float *jacobian);

/** @brief jacobian[i] = J_r^{-1}(xi_i) = J_l^{-1}(-xi_i) of SE(3) (6x6). */
void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size, float *jacobian);

}  // namespace cunls
