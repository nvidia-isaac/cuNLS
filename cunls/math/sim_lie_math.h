/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cunls {

// ============================================================================
// Sim(2) device helpers
// ============================================================================

/**
 * @brief Compute J_r^{-1}(xi) for Sim(2) into a local 4x4 array (row-major).
 *
 * xi = (u1, u2, w, lambda).  Uses the same closed-form as the batched
 * ComputeJacobianRightInverseSim2 kernel but is callable from device code.
 */
__device__ __forceinline__ void Sim2JrInv(float u1, float u2, float w,
                                          float lam, float* Jr) {
  float d2 = lam * lam + w * w;
  if (d2 < 1e-5f) {
    Jr[0]  = 1.0f + 0.5f * lam;  Jr[1]  = -0.5f * w;
    Jr[2]  = 0.5f * u2;          Jr[3]  = -0.5f * u1;
    Jr[4]  = 0.5f * w;           Jr[5]  = 1.0f + 0.5f * lam;
    Jr[6]  = -0.5f * u1;         Jr[7]  = -0.5f * u2;
    Jr[8]  = 0.0f;               Jr[9]  = 0.0f;
    Jr[10] = 1.0f;               Jr[11] = 0.0f;
    Jr[12] = 0.0f;               Jr[13] = 0.0f;
    Jr[14] = 0.0f;               Jr[15] = 1.0f;
    return;
  }

  float enl = expf(-lam);
  float cw = cosf(w), sw = sinf(w);
  float alpha = 1.0f - enl * cw;
  float beta  = enl * sw;

  float P = (lam * alpha + w * beta) / d2;
  float Q = (lam * beta  - w * alpha) / d2;
  float det_TL = P * P + Q * Q;
  float id = 1.0f / det_TL;
  float Ti00 =  P * id, Ti01 = Q * id;
  float Ti10 = -Q * id, Ti11 = P * id;

  float c_val = lam - alpha;
  float d_val = w   - beta;
  float d4 = d2 * d2;
  float e_val = (lam * lam - w * w) / d4;
  float f_val = -2.0f * lam * w / d4;

  float g = e_val * c_val - f_val * d_val;
  float h = e_val * d_val + f_val * c_val;

  float TR00 = -(g * u2 + h * u1);
  float TR01 =   g * u1 - h * u2;
  float TR10 =   g * u1 - h * u2;
  float TR11 =   g * u2 + h * u1;

  Jr[0]  = Ti00;  Jr[1]  = Ti01;
  Jr[2]  = -(Ti00 * TR00 + Ti01 * TR10);
  Jr[3]  = -(Ti00 * TR01 + Ti01 * TR11);
  Jr[4]  = Ti10;  Jr[5]  = Ti11;
  Jr[6]  = -(Ti10 * TR00 + Ti11 * TR10);
  Jr[7]  = -(Ti10 * TR01 + Ti11 * TR11);
  Jr[8]  = 0.0f;  Jr[9]  = 0.0f;  Jr[10] = 1.0f;  Jr[11] = 0.0f;
  Jr[12] = 0.0f;  Jr[13] = 0.0f;  Jr[14] = 0.0f;  Jr[15] = 1.0f;
}

// ============================================================================
// Sim(2) operations  (tangent dim 4, ambient 3x3)
// ============================================================================

void ComputeExpSim2(cudaStream_t stream, const float* tangent,
                    size_t tangent_stride, size_t transform_stride,
                    size_t size, float* transforms);

void ComputeLogSim2(cudaStream_t stream, const float* transforms,
                    size_t transform_stride, size_t tangent_stride,
                    size_t size, float* tangent);

void ComputeInverseSim2(cudaStream_t stream, const float* transforms,
                        size_t transform_stride, size_t inverse_stride,
                        size_t size, float* inverse_transforms);

void ComputeJacobianRightInverseSim2(cudaStream_t stream, const float* tangent,
                                     size_t tangent_stride,
                                     size_t jacobian_stride, size_t size,
                                     float* jacobians);

// ============================================================================
// Sim(3) operations  (tangent dim 7, ambient 4x4)
// ============================================================================

void ComputeExpSim3(cudaStream_t stream, const float* tangent,
                    size_t tangent_stride, size_t transform_stride,
                    size_t size, float* transforms);

void ComputeLogSim3(cudaStream_t stream, const float* transforms,
                    size_t transform_stride, size_t tangent_stride,
                    size_t size, float* tangent);

void ComputeInverseSim3(cudaStream_t stream, const float* transforms,
                        size_t transform_stride, size_t inverse_stride,
                        size_t size, float* inverse_transforms);

void ComputeJacobianRightInverseSim3(cudaStream_t stream, const float* tangent,
                                     size_t tangent_stride,
                                     size_t jacobian_stride, size_t size,
                                     float* jacobians);

void ComputeAdjointSim3(cudaStream_t stream, const float* transforms,
                         size_t transform_stride, float* adjoints,
                         size_t adjoint_stride, size_t size);

}  // namespace cunls
