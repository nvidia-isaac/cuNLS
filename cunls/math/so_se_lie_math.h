/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// ============================================================================
// SO(2) operations  (tangent dim 1, ambient 2x2)
// ============================================================================

void ComputeExpSO2(cudaStream_t stream, const float* angles,
                   size_t angle_stride, size_t rotation_stride, size_t size,
                   float* rotations);

void ComputeLogSO2(cudaStream_t stream, const float* rotations,
                   size_t rotation_stride, size_t angle_stride, size_t size,
                   float* angles);

void ComputeTransposeSO2(cudaStream_t stream, const float* rotations,
                         size_t input_stride, size_t output_stride,
                         size_t size, float* transposed);

// ============================================================================
// SE(2) operations  (tangent dim 3, ambient 3x3)
// ============================================================================

void ComputeExpSE2(cudaStream_t stream, const float* tangent,
                   size_t tangent_stride, size_t transform_stride, size_t size,
                   float* transforms);

void ComputeLogSE2(cudaStream_t stream, const float* transforms,
                   size_t transform_stride, size_t tangent_stride, size_t size,
                   float* tangent);

void ComputeInverseSE2(cudaStream_t stream, const float* transforms,
                       size_t transform_stride, size_t inverse_stride,
                       size_t size, float* inverse_transforms);

void ComputeJacobianRightInverseSE2(cudaStream_t stream, const float* tangent,
                                    size_t tangent_stride,
                                    size_t jacobian_stride, size_t size,
                                    float* jacobians);

// ============================================================================
// SO(3) operations  (tangent dim 3, ambient 3x3)
// ============================================================================

void ComputeSkewSO3(cudaStream_t stream, const float* twist,
                    const size_t twist_stride, const size_t skew_pitch,
                    const size_t skew_stride, size_t size, float* skew);

void ComputeExpSO3(cudaStream_t stream, const float* twist,
                   const size_t twist_stride, const size_t rotation_pitch,
                   const size_t rotation_stride, size_t size, float* rotation);

void ComputeLogSO3(cudaStream_t stream, const float* rotation,
                   const size_t rotation_pitch, const size_t rotation_stride,
                   const size_t twist_stride, size_t size, float* twist);

void ComputeTransposeSO3(cudaStream_t stream, const float* rotation,
                         size_t input_stride, size_t output_stride,
                         size_t size, float* transposed);

void ComputeJacobianLeftSO3(cudaStream_t stream, const float* twist,
                             const size_t twist_stride,
                             const size_t jacobian_pitch,
                             const size_t jacobian_stride, size_t size,
                             float* jacobian);

void ComputeJacobianRightSO3(cudaStream_t stream, const float* twist,
                              const size_t twist_stride,
                              const size_t jacobian_pitch,
                              const size_t jacobian_stride, size_t size,
                              float* jacobian);

void ComputeJacobianLeftInverseSO3(cudaStream_t stream, const float* twist,
                                   const size_t twist_stride,
                                   const size_t jacobian_inv_pitch,
                                   const size_t jacobian_inv_stride,
                                   size_t size, float* jacobian_inv);

void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float* twist,
                                    const size_t twist_stride,
                                    const size_t jacobian_inv_pitch,
                                    const size_t jacobian_inv_stride,
                                    size_t size, float* jacobian_inv);

// ============================================================================
// SE(3) operations  (tangent dim 6, ambient 4x4)
// ============================================================================

void ComputeNegateMatrix(cudaStream_t stream, const float* matrix, size_t rows,
                         size_t cols, const size_t pitch, const size_t stride,
                         size_t size, float* negated_matrix);

void ComputeInverseSE3(cudaStream_t stream, const float* transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t inverse_pitch,
                       const size_t inverse_stride, size_t size,
                       float* inverse_transform);

void ComputeExpSE3(cudaStream_t stream, const float* twist,
                   const size_t twist_stride, const size_t transform_pitch,
                   const size_t transform_stride, size_t size,
                   float* transform);

void ComputeLogSE3(cudaStream_t stream, const float* transform,
                   const size_t transform_pitch,
                   const size_t transform_stride, const size_t twist_stride,
                   size_t size, float* twist);

void ComputeAdjointSE3(cudaStream_t stream, const float* transform,
                        const size_t transform_pitch,
                        const size_t transform_stride,
                        const size_t adjoint_pitch,
                        const size_t adjoint_stride, size_t size,
                        float* adjoint);

void ComputeInverseAdjointSE3(cudaStream_t stream, const float* transform,
                               const size_t transform_pitch,
                               const size_t transform_stride,
                               const size_t inv_adjoint_pitch,
                               const size_t inv_adjoint_stride, size_t size,
                               float* inv_adjoint);

void ComputeJacobianLeftSE3(cudaStream_t stream, const float* twist,
                             const size_t twist_stride,
                             const size_t jacobian_pitch,
                             const size_t jacobian_stride, size_t size,
                             float* jacobian);

void ComputeJacobianLeftInverseSE3(cudaStream_t stream, const float* twist,
                                   const size_t twist_stride,
                                   const size_t jacobian_pitch,
                                   const size_t jacobian_stride, size_t size,
                                   float* jacobian);

void ComputeJacobianRightSE3(cudaStream_t stream, const float* twist,
                              const size_t twist_stride,
                              const size_t jacobian_pitch,
                              const size_t jacobian_stride, size_t size,
                              float* jacobian);

void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float* twist,
                                    const size_t twist_stride,
                                    const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size,
                                    float* jacobian);

}  // namespace cunls
