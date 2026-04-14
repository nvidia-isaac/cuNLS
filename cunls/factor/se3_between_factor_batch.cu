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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t block_size = 256;

/**
 * @brief Fused kernel: collect L/R SE3 transforms, compute T_left^{-1},
 *        then compute error = Delta * T_left^{-1} * T_right in one pass.
 *
 * Replaces: collect_poses_kernel + ComputeInverseSE3 + 2x cuBLAS SGEMM.
 * Fully unrolled with SE3 structure exploitation: last row is always [0 0 0 1],
 * reducing 128 FMAs to ~78 FMAs + additions.  Zero local-memory arrays.
 */
__global__ void collect_and_compute_se3_between_error_kernel(
    float const *const *state_pointers, const SE3Transform *deltas,
    size_t num_factors, SE3Transform *errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors)
    return;

  const float *__restrict__ L = state_pointers[2 * tid];
  const float *__restrict__ R = state_pointers[2 * tid + 1];
  const float *__restrict__ D = deltas[tid].data();
  float *__restrict__ out = errors[tid].data();

  // Load L rotation (3x3) and translation (3x1)
  const float l00 = L[0], l01 = L[1], l02 = L[2], l03 = L[3];
  const float l10 = L[4], l11 = L[5], l12 = L[6], l13 = L[7];
  const float l20 = L[8], l21 = L[9], l22 = L[10], l23 = L[11];

  // L_inv rotation = L_rot^T (transpose), L_inv translation = -L_rot^T * L_t
  // i00 = l00, i01 = l10, i02 = l20  (transposed)
  // i10 = l01, i11 = l11, i12 = l21
  // i20 = l02, i21 = l12, i22 = l22
  const float i03 = -(l00 * l03 + l10 * l13 + l20 * l23);
  const float i13 = -(l01 * l03 + l11 * l13 + l21 * l23);
  const float i23 = -(l02 * l03 + l12 * l13 + l22 * l23);

  // Load R (3x4 active part)
  const float r00 = R[0], r01 = R[1], r02 = R[2], r03 = R[3];
  const float r10 = R[4], r11 = R[5], r12 = R[6], r13 = R[7];
  const float r20 = R[8], r21 = R[9], r22 = R[10], r23 = R[11];

  // temp = L_inv * R  (only compute 3x4 active part; row 3 = [0 0 0 1])
  // Using transposed L indices directly
  const float t00 = l00 * r00 + l10 * r10 + l20 * r20;
  const float t01 = l00 * r01 + l10 * r11 + l20 * r21;
  const float t02 = l00 * r02 + l10 * r12 + l20 * r22;
  const float t03 = l00 * r03 + l10 * r13 + l20 * r23 + i03;

  const float t10 = l01 * r00 + l11 * r10 + l21 * r20;
  const float t11 = l01 * r01 + l11 * r11 + l21 * r21;
  const float t12 = l01 * r02 + l11 * r12 + l21 * r22;
  const float t13 = l01 * r03 + l11 * r13 + l21 * r23 + i13;

  const float t20 = l02 * r00 + l12 * r10 + l22 * r20;
  const float t21 = l02 * r01 + l12 * r11 + l22 * r21;
  const float t22 = l02 * r02 + l12 * r12 + l22 * r22;
  const float t23 = l02 * r03 + l12 * r13 + l22 * r23 + i23;

  // Load D (3x4 active part)
  const float d00 = D[0], d01 = D[1], d02 = D[2], d03 = D[3];
  const float d10 = D[4], d11 = D[5], d12 = D[6], d13 = D[7];
  const float d20 = D[8], d21 = D[9], d22 = D[10], d23 = D[11];

  // error = D * temp  (only compute 3x4 active part; row 3 = [0 0 0 1])
  out[0] = d00 * t00 + d01 * t10 + d02 * t20;
  out[1] = d00 * t01 + d01 * t11 + d02 * t21;
  out[2] = d00 * t02 + d01 * t12 + d02 * t22;
  out[3] = d00 * t03 + d01 * t13 + d02 * t23 + d03;

  out[4] = d10 * t00 + d11 * t10 + d12 * t20;
  out[5] = d10 * t01 + d11 * t11 + d12 * t21;
  out[6] = d10 * t02 + d11 * t12 + d12 * t22;
  out[7] = d10 * t03 + d11 * t13 + d12 * t23 + d13;

  out[8] = d20 * t00 + d21 * t10 + d22 * t20;
  out[9] = d20 * t01 + d21 * t11 + d22 * t21;
  out[10] = d20 * t02 + d21 * t12 + d22 * t22;
  out[11] = d20 * t03 + d21 * t13 + d22 * t23 + d23;

  out[12] = 0.0f;
  out[13] = 0.0f;
  out[14] = 0.0f;
  out[15] = 1.0f;
}

// ---------------------------------------------------------------------------
// Fused kernel: computes BOTH left and right SE3 Jacobians in one pass.
//
// Left  Jacobian (cols  0..5): J_left  = -Ad(Delta) * J_l^{-1}(twist)
// Right Jacobian (cols 6..11): J_right =  J_r^{-1}(twist)
//
// Cooperative design: 6 threads per factor. Each thread owns one output row
// of the 6x12 Jacobian. The 6 threads share J_so3[9] and Q[9] through
// shared memory, so no thread needs to materialize a full 6x6 matrix.
//
// Shared memory per factor: twist[6] + J_so3[9] + Q[9] + Jl_inv[36] = 60 floats
// Per-thread registers: ~6 (jl_row) + 6 (ad_row) + 6 (jr_row) + temps ≈ 40
//
// The left-Jacobian multiply (-Ad * Jl_inv) requires column access to Jl_inv,
// so Jl_inv rows are exchanged through shared memory.
//
// Right Jacobian uses the identity J_r_inv(xi) = J_l_inv(-xi):
//   SO(3): J_l_inv(-phi) = J_l_inv(phi)^T  (read J columns as rows from smem)
//   SE(3) lower-left block: -J^T * Q(-tw) * J^T
//   Q(-tw) is recomputed once in shared memory for the negated twist.
// ---------------------------------------------------------------------------

// Rodrigues formula for a single row of SO(3) Jacobian inverse.
// Computes row `r` of: k1*I + k2*[phi]_x + k3*phi*phi^T  (with k4 for
// small-angle)
__device__ __forceinline__ void rodrigues_row(const float *phi, int r, float k1,
                                              float k2, float k3, float k4,
                                              float *row3, float tol = 1e-5f) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  // Skew row r: row 0 = [0, -z, y], row 1 = [z, 0, -x], row 2 = [-y, x, 0]
  float sx, sy, sz;
  if (r == 0) {
    sx = 0.f;
    sy = -phi[2];
    sz = phi[1];
  } else if (r == 1) {
    sx = phi[2];
    sy = 0.f;
    sz = -phi[0];
  } else {
    sx = -phi[1];
    sy = phi[0];
    sz = 0.f;
  }
  if (theta < tol) {
    row3[0] = (r == 0 ? 1.f : 0.f) + k4 * sx;
    row3[1] = (r == 1 ? 1.f : 0.f) + k4 * sy;
    row3[2] = (r == 2 ? 1.f : 0.f) + k4 * sz;
    return;
  }
  float pr = phi[r];
  row3[0] = (r == 0 ? k1 : 0.f) + k2 * sx + k3 * pr * phi[0];
  row3[1] = (r == 1 ? k1 : 0.f) + k2 * sy + k3 * pr * phi[1];
  row3[2] = (r == 2 ? k1 : 0.f) + k2 * sz + k3 * pr * phi[2];
}

__device__ __forceinline__ void so3_jac_left_inv_row(const float *phi, int r,
                                                     float *row3) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  float th2 = theta * theta;
  float half = 0.5f * theta;
  float k1 = half / tanf(half);
  float k2 = -half / theta;
  float k3 = (1.0f - k1) / th2;
  rodrigues_row(phi, r, k1, k2, k3, -0.5f, row3);
}

// Compute full Q matrix (9 floats) for a given twist. Called once per factor
// by a designated thread and stored to shared memory for all 6 threads to read.
__device__ __forceinline__ void compute_Q_full(const float *tw, float *Q) {
  float phi_norm = norm3df(tw[0], tw[1], tw[2]);
  float A = 1.f / 6.f, B = 1.f / 24.f, C = 1.f / 120.f;
  constexpr float tol = 1e-5f;
  if (phi_norm > tol) {
    float s = sinf(phi_norm), c = cosf(phi_norm);
    float p2 = phi_norm * phi_norm, p3 = p2 * phi_norm;
    float p4 = p3 * phi_norm, p5 = p4 * phi_norm;
    A = (phi_norm - s) / p3;
    B = (p2 * 0.5f + c - 1.f) / p4;
    C = 0.5f * ((2.f + c) / p4 - 3.f * s / p5);
  }

  const float *phi = tw;
  const float *rho = tw + 3;
  float V[9], W[9];
  V[0] = 0.f;
  V[1] = -rho[2];
  V[2] = rho[1];
  V[3] = rho[2];
  V[4] = 0.f;
  V[5] = -rho[0];
  V[6] = -rho[1];
  V[7] = rho[0];
  V[8] = 0.f;
  W[0] = 0.f;
  W[1] = -phi[2];
  W[2] = phi[1];
  W[3] = phi[2];
  W[4] = 0.f;
  W[5] = -phi[0];
  W[6] = -phi[1];
  W[7] = phi[0];
  W[8] = 0.f;

#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] = 0.5f * V[i];

  // VW, WV (reuse temporaries)
  float VW[9], WV[9];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) {
      VW[i * 3 + j] =
          V[i * 3] * W[j] + V[i * 3 + 1] * W[3 + j] + V[i * 3 + 2] * W[6 + j];
      WV[i * 3 + j] =
          W[i * 3] * V[j] + W[i * 3 + 1] * V[3 + j] + W[i * 3 + 2] * V[6 + j];
    }
#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] += A * (VW[i] + WV[i]);

  // VWW = VW * W
  float VWW[9];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++)
      VWW[i * 3 + j] = VW[i * 3] * W[j] + VW[i * 3 + 1] * W[3 + j] +
                       VW[i * 3 + 2] * W[6 + j];

  // WWV = W * WV
  float WWV[9];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++)
      WWV[i * 3 + j] = W[i * 3] * WV[j] + W[i * 3 + 1] * WV[3 + j] +
                       W[i * 3 + 2] * WV[6 + j];

#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] += B * (VWW[i] + WWV[i]);

  // WVW = WV * W
  float WVW[9];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++)
      WVW[i * 3 + j] = WV[i * 3] * W[j] + WV[i * 3 + 1] * W[3 + j] +
                       WV[i * 3 + 2] * W[6 + j];

#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] += (A - 3.f * B) * WVW[i];

  // WVWW = WVW * W, WWVW = W * WVW
  float T[9];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) {
      T[i * 3 + j] = WVW[i * 3] * W[j] + WVW[i * 3 + 1] * W[3 + j] +
                     WVW[i * 3 + 2] * W[6 + j];
    }
#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] += C * T[i];

#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) {
      T[i * 3 + j] = W[i * 3] * WVW[j] + W[i * 3 + 1] * WVW[3 + j] +
                     W[i * 3 + 2] * WVW[6 + j];
    }
#pragma unroll
  for (int i = 0; i < 9; i++)
    Q[i] += C * T[i];
}

// 6 threads per factor. Each thread computes one row of the 6x12 Jacobian.
// Shared memory per factor: twist[6] + J_so3[9] + Q[9] + Jl_inv[36] = 60 floats
constexpr int kThreadsPerFactor = 6;
constexpr int kFactorsPerBlock = 32;
constexpr int kJacBlockSize = kFactorsPerBlock * kThreadsPerFactor; // 192
constexpr int kSmemPerFactor = 60;

__global__ void __launch_bounds__(kJacBlockSize, 5)
    se3_between_fused_jacobians_kernel(
        const float *__restrict__ residuals,
        const Matrix<6> *__restrict__ delta_adjoints, int num_factors,
        float *__restrict__ jacobians) {
  __shared__ float smem[kFactorsPerBlock * kSmemPerFactor];

  const int local_factor = threadIdx.x / kThreadsPerFactor;
  const int row = threadIdx.x % kThreadsPerFactor;
  const int global_factor = blockIdx.x * kFactorsPerBlock + local_factor;

  float *s_base = smem + local_factor * kSmemPerFactor;
  float *s_twist = s_base;   // [6]
  float *s_J = s_base + 6;   // [9]  -- J_so3 (3x3 row-major)
  float *s_Q = s_base + 15;  // [9]  -- Q (3x3 row-major)
  float *s_jl = s_base + 24; // [36] -- full J_l_inv (6x6) for column access

  const bool active = global_factor < num_factors;

  // --- Phase 1: cooperative load of twist ---
  if (active && row < 6)
    s_twist[row] = residuals[global_factor * 6 + row];
  __syncthreads();

  // --- Phase 2: threads 0-2 compute J_so3 rows, thread 3 computes Q ---
  float tw[6];
  if (active) {
#pragma unroll
    for (int i = 0; i < 6; i++)
      tw[i] = s_twist[i];

    if (row < 3) {
      float jr[3];
      so3_jac_left_inv_row(tw, row, jr);
      s_J[row * 3] = jr[0];
      s_J[row * 3 + 1] = jr[1];
      s_J[row * 3 + 2] = jr[2];
    }
    if (row == 3) {
      compute_Q_full(tw, s_Q);
    }
  }
  __syncthreads();

  // --- Phase 3: each thread computes its row of J_l_inv(twist) ---
  // J_l_inv = [ J_so3    |   0   ]
  //           [ -J*Q*J   |  J_so3]
  float jl_row[6] = {};
  if (active) {
    if (row < 3) {
      jl_row[0] = s_J[row * 3];
      jl_row[1] = s_J[row * 3 + 1];
      jl_row[2] = s_J[row * 3 + 2];
    } else {
      int r = row - 3;
      float Jr[3] = {s_J[r * 3], s_J[r * 3 + 1], s_J[r * 3 + 2]};

      // (J * Q * J)[r] = Jr . Q . J  (row-vector * matrix * matrix)
      float tmp[3];
#pragma unroll
      for (int j = 0; j < 3; j++)
        tmp[j] = Jr[0] * s_Q[j] + Jr[1] * s_Q[3 + j] + Jr[2] * s_Q[6 + j];

#pragma unroll
      for (int j = 0; j < 3; j++)
        jl_row[j] =
            -(tmp[0] * s_J[j] + tmp[1] * s_J[3 + j] + tmp[2] * s_J[6 + j]);

      jl_row[3] = Jr[0];
      jl_row[4] = Jr[1];
      jl_row[5] = Jr[2];
    }
  }

#pragma unroll
  for (int j = 0; j < 6; j++)
    s_jl[row * 6 + j] = jl_row[j];
  __syncthreads();

  // --- Phase 4: left output = -Ad[row] . Jl_inv (read Jl_inv columns from
  // smem) ---
  constexpr int jac_pitch = 12;
  if (active) {
    float *out = jacobians + global_factor * (6 * jac_pitch) + row * jac_pitch;

    const float *ad_src = delta_adjoints[global_factor].data();
    float ad_row[6];
#pragma unroll
    for (int i = 0; i < 6; i++)
      ad_row[i] = ad_src[row * 6 + i];

#pragma unroll
    for (int j = 0; j < 6; j++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 6; k++)
        s += ad_row[k] * s_jl[k * 6 + j];
      out[j] = -s;
    }

    // --- Phase 5: right Jacobian = J_r_inv(twist) = J_l_inv(-twist) ---
    // J_so3_l_inv(-phi) = J_so3_l_inv(phi)^T (skew-sym part negates →
    // transpose) Lower-left: -J^T * Q(-tw) * J^T
    if (row == 3) {
      float ntw[6] = {-tw[0], -tw[1], -tw[2], -tw[3], -tw[4], -tw[5]};
      compute_Q_full(ntw, s_Q);
    }
  }
  __syncthreads();

  if (active) {
    float *out = jacobians + global_factor * (6 * jac_pitch) + row * jac_pitch;

    float jr_row[6];
    if (row < 3) {
      jr_row[0] = s_J[row];
      jr_row[1] = s_J[3 + row];
      jr_row[2] = s_J[6 + row];
      jr_row[3] = 0.f;
      jr_row[4] = 0.f;
      jr_row[5] = 0.f;
    } else {
      int r = row - 3;
      float Jt_r[3] = {s_J[r], s_J[3 + r], s_J[6 + r]};

      float tmp[3];
#pragma unroll
      for (int j = 0; j < 3; j++)
        tmp[j] = Jt_r[0] * s_Q[j] + Jt_r[1] * s_Q[3 + j] + Jt_r[2] * s_Q[6 + j];

        // J^T[k][j] = J[j][k] = s_J[j*3+k]
#pragma unroll
      for (int j = 0; j < 3; j++)
        jr_row[j] = -(tmp[0] * s_J[j * 3] + tmp[1] * s_J[j * 3 + 1] +
                      tmp[2] * s_J[j * 3 + 2]);

      jr_row[3] = Jt_r[0];
      jr_row[4] = Jt_r[1];
      jr_row[5] = Jt_r[2];
    }

#pragma unroll
    for (int j = 0; j < 6; j++)
      out[6 + j] = jr_row[j];
  }
}

/**
 * @brief Constructs the SE3 between factor batch.
 *
 * Allocates device memory for intermediate results and precomputes the
 * adjoint matrices of the pose deltas, which are reused during every
 * Evaluate() call for Jacobian computation.
 *
 * @param cublas_handle Reference to an externally-owned cuBLAS handle.
 * @param pose_deltas_ptr    Device pointer to SE3 pose delta constraints.
 * @param num_factors Number of factors in the batch.
 */
SE3BetweenFactorBatch::SE3BetweenFactorBatch(
    const SE3Transform *pose_deltas_ptr, size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr), num_factors_(num_factors),
      delta_adjoints_(num_factors), poses_left_inverse_(num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

/**
 * @brief Evaluates residuals and optionally Jacobians for the SE3 between
 * factor.
 *
 * Computes residual = Log(Delta * T_left^{-1} * T_right) for each pair of
 * poses. If jacobians is non-null, also computes the left and right pose
 * Jacobians using the SE(3) left/right inverse Jacobians and the precomputed
 * delta adjoint matrices.
 *
 * @param residuals   Output device pointer for residuals (6 floats per factor).
 * @param jacobians   Output device pointer for Jacobians (6x12 floats per
 *                    factor), or nullptr to skip Jacobian computation.
 * @param state_pointers  Device pointer to state block pointers.
 * @param stream      CUDA stream for asynchronous execution.
 * @return true on success.
 */
bool SE3BetweenFactorBatch::Evaluate(float *residuals, float *jacobians,
                                     float const *const *state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + block_size - 1) / block_size;

  // Fused: collect L/R + compute Delta * L^{-1} * R in one kernel
  collect_and_compute_se3_between_error_kernel<<<num_blocks, block_size, 0,
                                                 stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  constexpr size_t twist_stride = 6;
  ComputeLogSE3(stream,
                reinterpret_cast<const float *>(poses_left_inverse_.data()),
                pitch, stride, twist_stride, num_factors, residuals);

  if (jacobians != nullptr) {
    size_t jac_blocks = (num_factors + kFactorsPerBlock - 1) / kFactorsPerBlock;
    se3_between_fused_jacobians_kernel<<<jac_blocks, kJacBlockSize, 0,
                                         stream>>>(
        residuals, delta_adjoints_.data(), num_factors, jacobians);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

/**
 * @brief Precomputes SE(3) adjoint matrices for all pose deltas.
 *
 * The adjoint of each delta transform is used during Jacobian computation
 * (specifically for the left pose Jacobian). This is called once during
 * construction and the results are cached in delta_adjoints_.
 *
 * @param stream CUDA stream for asynchronous execution.
 */
void SE3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  auto delta_ptr = reinterpret_cast<const float *>(pose_deltas_ptr_);
  auto delta_adjoints_ptr = reinterpret_cast<float *>(delta_adjoints_.data());

  constexpr size_t delta_pitch = 4;
  constexpr size_t delta_stride = 16;
  constexpr size_t delta_adjoint_pitch = 6;
  constexpr size_t delta_adjoint_stride = 36;

  ComputeAdjointSE3(stream, delta_ptr, delta_pitch, delta_stride,
                    delta_adjoint_pitch, delta_adjoint_stride, num_factors_,
                    delta_adjoints_ptr);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
