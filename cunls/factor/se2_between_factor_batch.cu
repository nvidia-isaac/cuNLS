/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// SE(2) between factor layout constants
constexpr size_t kSE2TransformStride = 9;
constexpr size_t kSE2TangentStride = 3;

/**
 * @brief Gather left and right SE(2) poses from state pointers.
 *
 * For a between factor connecting state i and j, state_pointers has
 * alternating left/right pairs: [left_0, right_0, left_1, right_1, ...].
 */
__global__ void collect_se2_poses_between_kernel(
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
 * @brief Compute exact Jacobian blocks for the SE(2) between factor.
 *
 * The residual is r = Log(Delta * T_left^{-1} * T_right).
 *
 * Right-perturbation Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)    (3x3, columns 0-2)
 *   H_right =  J_r^{-1}(r)                 (3x3, columns 3-5)
 *
 * where J_r^{-1} is the SE(2) LogmapDerivative, J_l^{-1}(xi) = J_r^{-1}(-xi),
 * and Ad(Delta) is the SE(2) adjoint of the measurement-inverse transform:
 *   Ad([[c,-s,tx],[s,c,ty],[0,0,1]]) = [[c,-s,ty],[s,c,-tx],[0,0,1]].
 *
 * @param residuals  Residual vectors (3 floats per factor)
 * @param deltas     Delta transforms (9 floats per factor, row-major 3x3)
 * @param delta_stride Stride between consecutive delta transforms
 * @param jacobians  Output 3x6 Jacobian matrices (18 floats per factor)
 * @param num_factors Number of factors in the batch
 */
__global__ void se2_between_jacobian_kernel(const float* residuals,
                                            const float* deltas,
                                            size_t delta_stride,
                                            float* jacobians,
                                            size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }

  const float* r = residuals + tid * 3;
  float v1 = r[0], v2 = r[1], alpha = r[2];

  // Read Delta transform: [[c,-s,tx],[s,c,ty],[0,0,1]] (row-major)
  const float* D = deltas + tid * delta_stride;
  float dc = D[0], ds = D[3], dtx = D[2], dty = D[5];

  // J_r^{-1}(r) and J_l^{-1}(r) = J_r^{-1}(-r)
  float Jl[9], Jr[9];

  if (fabsf(alpha) > 1e-3f) {
    float ai = 1.0f / alpha;
    float hch = 0.5f * sinf(alpha) / (1.0f - cosf(alpha));
    float ach = alpha * hch;
    float ha = 0.5f * alpha;

    Jr[0] = ach;
    Jr[1] = -ha;
    Jr[2] = v1 * ai - v1 * hch + 0.5f * v2;
    Jr[3] = ha;
    Jr[4] = ach;
    Jr[5] = v2 * ai - 0.5f * v1 - v2 * hch;
    Jr[6] = 0.0f;
    Jr[7] = 0.0f;
    Jr[8] = 1.0f;

    Jl[0] = ach;
    Jl[1] = ha;
    Jl[2] = v1 * ai - v1 * hch - 0.5f * v2;
    Jl[3] = -ha;
    Jl[4] = ach;
    Jl[5] = v2 * ai + 0.5f * v1 - v2 * hch;
    Jl[6] = 0.0f;
    Jl[7] = 0.0f;
    Jl[8] = 1.0f;
  } else {
    Jr[0] = 1.0f;
    Jr[1] = 0.0f;
    Jr[2] = 0.5f * v2;
    Jr[3] = 0.0f;
    Jr[4] = 1.0f;
    Jr[5] = -0.5f * v1;
    Jr[6] = 0.0f;
    Jr[7] = 0.0f;
    Jr[8] = 1.0f;

    Jl[0] = 1.0f;
    Jl[1] = 0.0f;
    Jl[2] = -0.5f * v2;
    Jl[3] = 0.0f;
    Jl[4] = 1.0f;
    Jl[5] = 0.5f * v1;
    Jl[6] = 0.0f;
    Jl[7] = 0.0f;
    Jl[8] = 1.0f;
  }

  // H_left = -J_l^{-1} * Ad(Delta)
  // Ad(Delta) = [[dc, -ds, dty], [ds, dc, -dtx], [0, 0, 1]]
  float* J = jacobians + tid * 18;

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    float a = Jl[i * 3 + 0];
    float b = Jl[i * 3 + 1];
    float c = Jl[i * 3 + 2];
    J[i * 6 + 0] = -(a * dc + b * ds);
    J[i * 6 + 1] = -(a * (-ds) + b * dc);
    J[i * 6 + 2] = -(a * dty + b * (-dtx) + c);
  }

  // H_right = J_r^{-1}(r)
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    J[i * 6 + 3] = Jr[i * 3 + 0];
    J[i * 6 + 4] = Jr[i * 3 + 1];
    J[i * 6 + 5] = Jr[i * 3 + 2];
  }
}

SE2BetweenFactorBatch::SE2BetweenFactorBatch(cuBLASHandle& cublas_handle,
                                             const Matrix<3>* pose_deltas_ptr,
                                             size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool SE2BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Step 1: Gather left and right poses from state pointers
  collect_se2_poses_between_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute T_left^{-1}
  ComputeInverseSE2(stream,
                    reinterpret_cast<const float*>(poses_left_.data()),
                    kSE2TransformStride, kSE2TransformStride, num_factors,
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

  // Step 4: Multiply by delta^{-1} to get error: (T_left^{-1}*T_right) * delta_inv
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride,
      &beta, reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size,
      stride, num_factors));

  // Step 5: residual = Log(error)
  ComputeLogSE2(stream,
                reinterpret_cast<const float*>(poses_left_inverse_.data()),
                kSE2TransformStride, kSE2TangentStride, num_factors,
                residuals);

  // Step 6: jacobians
  //   H_left  = -J_l^{-1}(r) * Ad(Delta)
  //   H_right =  J_r^{-1}(r)
  if (jacobians != nullptr) {
    se2_between_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        residuals, reinterpret_cast<const float*>(pose_deltas_ptr_),
        kSE2TransformStride, jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
