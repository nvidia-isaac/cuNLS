/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/math/dense_matrix_ops.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

constexpr size_t kSim3TransformStride = 16;
constexpr size_t kSim3TangentStride = 7;
constexpr size_t kSim3TangentDim = 7;
constexpr size_t kSim3AdjointStride = 49;
constexpr size_t kSim3JacobianPitch = 14;
constexpr size_t kSim3JacobianStride = 98;

/**
 * @brief Gather left and right Sim(3) poses from state pointers.
 */
__global__ void collect_sim3_between_poses_kernel(
    float const* const* state_pointers, size_t num_factors,
    Matrix<4>* pose_left, Matrix<4>* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  pose_left[tid] = *reinterpret_cast<const Matrix<4>*>(state_pointers[2 * tid]);
  pose_right[tid] =
      *reinterpret_cast<const Matrix<4>*>(state_pointers[2 * tid + 1]);
}

Similarity3BetweenFactorBatch::Similarity3BetweenFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<4>* pose_deltas_ptr,
    size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors),
      delta_adjoints_(kSim3AdjointStride * num_factors),
      jacobian_temp_(kSim3AdjointStride * num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void Similarity3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  size_t num_factors = NumFactors();
  ComputeAdjointSim3(stream,
                      reinterpret_cast<const float*>(pose_deltas_ptr_),
                      kSim3TransformStride, delta_adjoints_.data(),
                      kSim3AdjointStride, num_factors);
}

void Similarity3BetweenFactorBatch::ComputeLeftPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  size_t num_factors = NumFactors();

  // Negate residuals into temp storage (reuse poses_left_ which is free after
  // residual computation; 16 floats/factor > 7 floats needed).
  float* neg_res = reinterpret_cast<float*>(poses_left_.data());
  ComputeNegateMatrix(stream, residuals, 1, kSim3TangentDim,
                      kSim3TangentDim, kSim3TangentStride,
                      num_factors, neg_res);

  // J_r^{-1}(-r) = J_l^{-1}(r) into jacobian_temp_ (dense 7x7, stride 49)
  ComputeJacobianRightInverseSim3(stream, neg_res, kSim3TangentStride,
                                  kSim3AdjointStride, num_factors,
                                  jacobian_temp_.data());

  // Negate to get -J_l^{-1}(r)
  ComputeNegateMatrix(stream, jacobian_temp_.data(), kSim3TangentDim,
                      kSim3TangentDim, kSim3TangentDim, kSim3AdjointStride,
                      num_factors, jacobian_temp_.data());

  // cuBLAS GEMM: H_left = (-J_l^{-1}) * Ad(Delta)
  // Row-major C = B * A via column-major cuBLAS: C_col = A_col * B_col.
  // Output writes directly into the left 7x7 block of the 7x14 Jacobian
  // using ldc=14 (full row width).
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 7;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_adjoints_.data(), mat_size, kSim3AdjointStride,
      jacobian_temp_.data(), mat_size, kSim3AdjointStride,
      &beta,
      jacobians, kSim3JacobianPitch, kSim3JacobianStride,
      num_factors));
}

void Similarity3BetweenFactorBatch::ComputeRightPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  size_t num_factors = NumFactors();

  // J_r^{-1}(r) into jacobian_temp_ (dense 7x7, stride 49)
  ComputeJacobianRightInverseSim3(stream, residuals, kSim3TangentStride,
                                  kSim3AdjointStride, num_factors,
                                  jacobian_temp_.data());

  // Scatter into the right 7x7 block of each 7x14 Jacobian row
  ScatterToRightBlock(stream, jacobian_temp_.data(), kSim3TangentDim,
                      kSim3AdjointStride, jacobians + kSim3TangentDim,
                      kSim3JacobianPitch, kSim3JacobianStride, num_factors);
}

bool Similarity3BetweenFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Step 1: Gather left and right poses
  collect_sim3_between_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute T_left^{-1}
  ComputeInverseSim3(stream,
                     reinterpret_cast<const float*>(poses_left_.data()),
                     kSim3TransformStride, kSim3TransformStride, num_factors,
                     reinterpret_cast<float*>(poses_left_inverse_.data()));

  // Step 3: Compute T_left^{-1} * T_right via batched GEMM
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;
  constexpr int stride = 16;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_.data()), mat_size, stride,
      reinterpret_cast<const float*>(poses_left_inverse_.data()), mat_size,
      stride, &beta, reinterpret_cast<float*>(poses_left_.data()), mat_size,
      stride, num_factors));

  // Step 4: Multiply by Delta
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride, &beta,
      reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size, stride,
      num_factors));

  // Step 5: residual = Log(error)
  ComputeLogSim3(stream,
                 reinterpret_cast<const float*>(poses_left_inverse_.data()),
                 kSim3TransformStride, kSim3TangentStride, num_factors,
                 residuals);

  // Step 6: Jacobian = [-J_l^{-1}(r)*Ad(Delta) | J_r^{-1}(r)]
  if (jacobians != nullptr) {
    ComputeLeftPoseJacobian(stream, residuals, jacobians);
    ComputeRightPoseJacobian(stream, residuals, jacobians);
  }

  return true;
}

}  // namespace cunls
