/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cunls {

// Batched Sim(2) and Sim(3) maps used by the built-in similarity state and
// factor batches. Every function launches one kernel on `stream` (asynchronous)
// over `size` batch elements; inputs and outputs are device arrays, element i
// starts at `i * *_stride` floats, matrices are row-major and contiguous.

// ============================================================================
// Sim(2) operations  (tangent dim 4, ambient 3x3)
// ============================================================================

/**
 * @brief tangent[i] = Log(T_i) = [u_x, u_y, theta, lambda] for 3x3 transforms
 * T = [R t; 0 0 1/s] (lambda = log s).
 */
void ComputeLogSim2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent);

/** @brief jacobians[i] = J_r^{-1}(xi_i) of Sim(2) (4x4, exact closed form). */
void ComputeJacobianRightInverseSim2(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians);

// ============================================================================
// Sim(3) operations  (tangent dim 7, ambient 4x4)
// ============================================================================

/**
 * @brief tangent[i] = Log(T_i) = [w, u, lambda] for 4x4 transforms
 * T = [R t; 0 1/s] (lambda = log s).
 */
void ComputeLogSim3(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent);

/** @brief jacobians[i] = J_r^{-1}(xi_i) of Sim(3) (7x7, Bernoulli series to order 6). */
void ComputeJacobianRightInverseSim3(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians);

}  // namespace cunls
