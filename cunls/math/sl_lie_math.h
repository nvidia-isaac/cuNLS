/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cunls {

/**
 * @brief SL(4) Lie algebra / group operations (tangent dim 15, 4x4 ambient),
 * used by SL4StateBatch and the SL(4) prior and between factors.
 *
 * Basis matches orthonormal decomposition
 * sl(4) = so(4) ⊕ sym_off(4) ⊕ diag_traceless(4).
 *
 * Every function launches one kernel on `stream` (asynchronous) over `size`
 * batch elements; matrices are row-major with rows `*_pitch` floats apart,
 * element i at `i * *_stride` floats.
 */

/**
 * @brief transform[i] = Exp(Hat(xi_i)): matrix exponential followed by
 * projection to det = 1.
 */
void ComputeExpSL4(cudaStream_t stream, const float *twist, const size_t twist_stride,
                   const size_t transform_pitch, const size_t transform_stride, size_t size,
                   float *transform);

/**
 * @brief twist[i] = Vee(log(T_i)): matrix logarithm with inverse
 * scaling-and-squaring near the identity.
 */
void ComputeLogSL4(cudaStream_t stream, const float *transform, const size_t transform_pitch,
                   const size_t transform_stride, const size_t twist_stride, size_t size,
                   float *twist);

/**
 * @brief Writes 15x15 identity matrices: the approximate SL(4) right Jacobian
 * inverse of the prior factor (exact right Jacobians are omitted).
 */
void FillIdentity15x15(cudaStream_t stream, size_t size, float *matrices, const size_t pitch,
                       const size_t stride);

}  // namespace cunls
