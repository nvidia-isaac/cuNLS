/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 * xi = (u1, u2, w, lambda). Exact closed form (series near 0), float32-safe;
 * the batched ComputeJacobianRightInverseSim2 kernel calls it.
 */
__device__ __forceinline__ void Sim2JrInv(float u1, float u2, float w, float lam, float *Jr) {
  // ad(xi) = [[M, N], [0, 0]] with M = [[lam, -w], [w, lam]] (multiplication
  // by z = lam + i w) and N = [[u2, -u1], [-u1, -u2]], so
  // J_r^{-1} = sum_k B_k^+ / k! ad^k = [[f(M), g(M) N], [0, I]] with
  // f(z) = z / (1 - e^{-z}) and g(z) = (f(z) - 1) / z.
  float fr, fi, gr, gi;
  const float z2 = lam * lam + w * w;
  if (z2 < 0.25f) {
    // Bernoulli series: f = 1 + z/2 + z^2/12 - z^4/720 + z^6/30240 - z^8/1209600.
    const float zr2 = lam * lam - w * w, zi2 = 2.0f * lam * w;  // z^2
    // h(y) = 1/12 - y/720 + y^2/30240 - y^3/1209600 at y = z^2 (complex).
    float hr = -1.0f / 1209600.0f, hi = 0.0f;
    const float coeffs[3] = {1.0f / 30240.0f, -1.0f / 720.0f, 1.0f / 12.0f};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const float tr = hr * zr2 - hi * zi2 + coeffs[k];
      hi = hr * zi2 + hi * zr2;
      hr = tr;
    }
    // g = 1/2 + z h,  f = 1 + z g.
    gr = 0.5f + lam * hr - w * hi;
    gi = lam * hi + w * hr;
    fr = 1.0f + lam * gr - w * gi;
    fi = lam * gi + w * gr;
  } else {
    // 1 - e^{-z} = a + i b.
    const float e = expf(-lam);
    const float a = 1.0f - e * cosf(w);
    const float b = e * sinf(w);
    const float inv = 1.0f / (a * a + b * b);
    fr = (lam * a + w * b) * inv;
    fi = (w * a - lam * b) * inv;
    // g = (f - 1) / z.
    const float inv_z = 1.0f / z2;
    gr = ((fr - 1.0f) * lam + fi * w) * inv_z;
    gi = (fi * lam - (fr - 1.0f) * w) * inv_z;
  }
  // g(M) N with g(M) = [[gr, -gi], [gi, gr]].
  Jr[0] = fr;
  Jr[1] = -fi;
  Jr[2] = gr * u2 + gi * u1;
  Jr[3] = -gr * u1 + gi * u2;
  Jr[4] = fi;
  Jr[5] = fr;
  Jr[6] = gi * u2 - gr * u1;
  Jr[7] = -gi * u1 - gr * u2;
  Jr[8] = 0.0f;
  Jr[9] = 0.0f;
  Jr[10] = 1.0f;
  Jr[11] = 0.0f;
  Jr[12] = 0.0f;
  Jr[13] = 0.0f;
  Jr[14] = 0.0f;
  Jr[15] = 1.0f;
}

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
