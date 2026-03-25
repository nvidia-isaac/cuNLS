/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cunls {

/**
 * @brief SL(4) Lie algebra / group operations (tangent dim 15, 4x4 ambient).
 *
 * Basis matches orthonormal decomposition
 * sl(4) = so(4) ⊕ sym_off(4) ⊕ diag_traceless(4).
 *
 * The exponential map uses matrix exp(Hat(xi)) followed by projection to det=1.
 * The logarithm uses matrix log with inverse scaling-and-squaring near
 * identity.
 *
 * @note Jacobian helpers for SL(4) use identity approximations where exact
 *       right Jacobians are omitted.
 */

void ComputeExpSL4(cudaStream_t stream, const float* twist,
                   const size_t twist_stride, const size_t transform_pitch,
                   const size_t transform_stride, size_t size,
                   float* transform);

void ComputeLogSL4(cudaStream_t stream, const float* transform,
                   const size_t transform_pitch, const size_t transform_stride,
                   const size_t twist_stride, size_t size, float* twist);

void ComputeInverseSL4(cudaStream_t stream, const float* transform,
                       const size_t transform_pitch, const size_t transform_stride,
                       const size_t inverse_pitch, const size_t inverse_stride,
                       size_t size, float* inverse_transform);

void ComputeAdjointSL4(cudaStream_t stream, const float* transform,
                       const size_t transform_pitch, const size_t transform_stride,
                       const size_t adjoint_pitch, const size_t adjoint_stride,
                       size_t size, float* adjoint);

void ComputeNegateMatrix15x15(cudaStream_t stream, const float* matrix,
                              const size_t pitch, const size_t stride, size_t size,
                              float* out);

void FillIdentity15x15(cudaStream_t stream, size_t size, float* matrices,
                       const size_t pitch, const size_t stride);

}  // namespace cunls
