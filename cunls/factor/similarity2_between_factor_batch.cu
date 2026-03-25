/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity2_between_factor_batch.h"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// Sim(2) between factor layout constants
constexpr size_t kSim2TransformStride = 9;
constexpr size_t kSim2TangentStride = 4;

/**
 * @brief Gather left and right Sim(2) poses from state pointers.
 */
__global__ void collect_sim2_between_poses_kernel(
    float const* const* state_pointers, size_t num_factors,
    Matrix<3>* pose_left, Matrix<3>* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  pose_left[tid] = *reinterpret_cast<const Matrix<3>*>(state_pointers[2 * tid]);
  pose_right[tid] =
      *reinterpret_cast<const Matrix<3>*>(state_pointers[2 * tid + 1]);
}

/**
 * @brief Compute exact Jacobian blocks for the Sim(2) between factor.
 *
 * Residual r = Log(Delta * T_left^{-1} * T_right), tangent = [u1,u2,w,lambda].
 *
 * Right-perturbation Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)    (4x4, columns 0-3)
 *   H_right =  J_r^{-1}(r)                 (4x4, columns 4-7)
 *
 * where J_l^{-1}(xi) = J_r^{-1}(-xi), and Ad(Delta) is the Sim(2) adjoint:
 *   Ad = [[s*R, [s*ty,-s*tx]^T, [-s*tx,-s*ty]^T],
 *         [0 0, 1, 0],
 *         [0 0, 0, 1]]
 */
__global__ void sim2_between_jacobian_kernel(const float* residuals,
                                             const float* deltas,
                                             size_t delta_stride,
                                             float* jacobians,
                                             size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }

  const float* r = residuals + tid * 4;
  float u1 = r[0], u2 = r[1], w = r[2], lam = r[3];

  // Read Delta: [[c,-s,tx],[s,c,ty],[0,0,1/scale]]  (row-major 3x3)
  const float* D = deltas + tid * delta_stride;
  float dc = D[0], ds = D[3], dtx = D[2], dty = D[5];
  float scale = 1.0f / D[8];

  // Ad(Delta) 4x4
  float Ad[16];
  Ad[0]  = scale * dc;   Ad[1]  = -scale * ds;
  Ad[2]  = scale * dty;  Ad[3]  = -scale * dtx;
  Ad[4]  = scale * ds;   Ad[5]  = scale * dc;
  Ad[6]  = -scale * dtx; Ad[7]  = -scale * dty;
  Ad[8]  = 0.0f;         Ad[9]  = 0.0f;  Ad[10] = 1.0f; Ad[11] = 0.0f;
  Ad[12] = 0.0f;         Ad[13] = 0.0f;  Ad[14] = 0.0f; Ad[15] = 1.0f;

  float Jr[16], Jl[16];
  Sim2JrInv(u1, u2, w, lam, Jr);
  Sim2JrInv(-u1, -u2, -w, -lam, Jl);

  // Write 4x8 Jacobian: [H_left | H_right]
  float* J = jacobians + tid * 32;

  // H_left = -Jl * Ad  (4x4 * 4x4)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        s += Jl[i * 4 + k] * Ad[k * 4 + j];
      }
      J[i * 8 + j] = -s;
    }
  }

  // H_right = Jr
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      J[i * 8 + 4 + j] = Jr[i * 4 + j];
    }
  }
}

Similarity2BetweenFactorBatch::Similarity2BetweenFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<3>* pose_deltas_ptr,
    size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool Similarity2BetweenFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Step 1: Gather left and right poses
  collect_sim2_between_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute T_left^{-1}
  ComputeInverseSim2(stream,
                     reinterpret_cast<const float*>(poses_left_.data()),
                     kSim2TransformStride, kSim2TransformStride, num_factors,
                     reinterpret_cast<float*>(poses_left_inverse_.data()));

  // Step 3: Compute T_left^{-1} * T_right via batched GEMM
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_.data()), mat_size, stride,
      reinterpret_cast<const float*>(poses_left_inverse_.data()), mat_size,
      stride, &beta, reinterpret_cast<float*>(poses_left_.data()), mat_size,
      stride, num_factors));

  // Step 4: Multiply by delta^{-1}
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride, &beta,
      reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size, stride,
      num_factors));

  // Step 5: residual = Log(error)
  ComputeLogSim2(stream,
                 reinterpret_cast<const float*>(poses_left_inverse_.data()),
                 kSim2TransformStride, kSim2TangentStride, num_factors,
                 residuals);

  // Step 6: Jacobian = [-J_l^{-1}(r)*Ad(Delta) | J_r^{-1}(r)]
  if (jacobians != nullptr) {
    sim2_between_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        residuals, reinterpret_cast<const float*>(pose_deltas_ptr_),
        kSim2TransformStride, jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
