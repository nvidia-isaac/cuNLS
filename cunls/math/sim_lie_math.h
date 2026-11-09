/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cunls {

// ============================================================================
// Sim(2) operations  (tangent dim 4, ambient 3x3)
// ============================================================================

void ComputeExpSim2(cudaStream_t stream, const float *tangent, size_t tangent_stride,
                    size_t transform_stride, size_t size, float *transforms);

void ComputeLogSim2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent);

void ComputeInverseSim2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                        size_t inverse_stride, size_t size, float *inverse_transforms);

void ComputeJacobianRightInverseSim2(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians);

// ============================================================================
// Sim(3) operations  (tangent dim 7, ambient 4x4)
// ============================================================================

void ComputeExpSim3(cudaStream_t stream, const float *tangent, size_t tangent_stride,
                    size_t transform_stride, size_t size, float *transforms);

void ComputeLogSim3(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent);

void ComputeInverseSim3(cudaStream_t stream, const float *transforms, size_t transform_stride,
                        size_t inverse_stride, size_t size, float *inverse_transforms);

void ComputeJacobianRightInverseSim3(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians);

void ComputeAdjointSim3(cudaStream_t stream, const float *transforms, size_t transform_stride,
                        float *adjoints, size_t adjoint_stride, size_t size);

}  // namespace cunls
