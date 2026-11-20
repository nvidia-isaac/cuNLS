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
 * @brief twist[i] = Vee(log(T_i)): the principal matrix logarithm, by inverse
 * scaling-and-squaring (Denman-Beavers square roots until T is near the
 * identity, then a series).
 *
 * Supported domain: transforms with no eigenvalues on the closed negative real
 * axis (-inf, 0]. For others the square-root iteration breaks down and the
 * result is wrong, with no error reported: zero when an iterate turns singular
 * and the identity is substituted for the root (e.g. diag(-1, -1, 1, 1), a
 * half-turn in a plane), otherwise whatever the non-converging iteration leaves
 * (e.g. diag(-2, -0.5, 1, 1)).
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
