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

// Pinhole projection through a world_from_rig pose state, shared by the
// reprojection and PnP factors (device code, internal to cunls/factor).
//
// T = (R, t) is world_from_rig (rows 0..2 of a row-major 4x4, pitch 4), E an
// optional camera_from_rig (same layout; nullptr = identity):
//
//   p_r = Rᵀ (P - t),   p_c = E p_r,   r = p_c.xy / p_c.z - obs
//
// The state is perturbed on the right, T Exp(δ), δ = [φ, ρ] in the rig frame:
// p_r ← Exp(δ)⁻¹ p_r ≈ p_r + [p_r]x φ - ρ. With A = J_c E_R (2x3, J_c =
// ∂(x/z, y/z)/∂p_c), row i of the Jacobian is
//
//   ∂r_i/∂φ = A_i × p_r,   ∂r_i/∂ρ = -A_i,   ∂r_i/∂P = A_i Rᵀ.
//
// The rotation columns scale with the depth p_r, not with the distance from
// the world origin; P - t is formed first, so far-away scenes keep their
// precision.

namespace cunls {
namespace projection {

/** p_r = Rᵀ (P - t) and p_c = E p_r (p_c = p_r without E). */
__device__ __forceinline__ void RigAndCameraPoint(const float *__restrict__ T,
                                                  const float *__restrict__ E, const float *P,
                                                  float *p_r, float *p_c) {
  const float d0 = P[0] - T[3], d1 = P[1] - T[7], d2 = P[2] - T[11];
#pragma unroll
  for (int j = 0; j < 3; ++j) p_r[j] = T[j] * d0 + T[4 + j] * d1 + T[8 + j] * d2;
  if (E != nullptr) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      p_c[i] = E[i * 4 + 3] + E[i * 4] * p_r[0] + E[i * 4 + 1] * p_r[1] + E[i * 4 + 2] * p_r[2];
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) p_c[i] = p_r[i];
  }
}

/**
 * Jacobian rows (see the file comment) for a valid projection (inv_z = 1 /
 * p_c.z): pose[i][0..5] = [A_i × p_r, -A_i]; point[i][0..2] = A_i Rᵀ when
 * `point` is not null.
 */
__device__ __forceinline__ void Jacobian(const float *__restrict__ T, const float *__restrict__ E,
                                         const float *p_r, const float *p_c, float inv_z,
                                         float (*pose)[6], float (*point)[3]) {
  const float u = p_c[0] * inv_z, v = p_c[1] * inv_z;
  // J_c = inv_z [[1, 0, -u], [0, 1, -v]].
  float A[2][3];
  if (E != nullptr) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      A[0][j] = inv_z * (E[j] - u * E[8 + j]);
      A[1][j] = inv_z * (E[4 + j] - v * E[8 + j]);
    }
  } else {
    A[0][0] = inv_z, A[0][1] = 0.f, A[0][2] = -u * inv_z;
    A[1][0] = 0.f, A[1][1] = inv_z, A[1][2] = -v * inv_z;
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float *a = A[i];
    pose[i][0] = a[1] * p_r[2] - a[2] * p_r[1];
    pose[i][1] = a[2] * p_r[0] - a[0] * p_r[2];
    pose[i][2] = a[0] * p_r[1] - a[1] * p_r[0];
    pose[i][3] = -a[0];
    pose[i][4] = -a[1];
    pose[i][5] = -a[2];
    if (point != nullptr) {
#pragma unroll
      for (int j = 0; j < 3; ++j)
        point[i][j] = a[0] * T[j * 4] + a[1] * T[j * 4 + 1] + a[2] * T[j * 4 + 2];
    }
  }
}

}  // namespace projection
}  // namespace cunls
