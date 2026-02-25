/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/device_vector.h>

#include "cunls/common/helper.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"

namespace cunls {
namespace cg = cooperative_groups;
#define WARP_SIZE 32

constexpr size_t block_size = 256;  ///< Default thread block size for CUDA kernels.

/**
 * @brief Computes the Jacobian scaling factor alpha for robust loss correction.
 *
 * Given the squared residual norm and the loss function derivatives (rho),
 * computes alpha = (1 - sqrt(1 + 2 * sq_norm * rho.z / rho.y)) / sq_norm.
 * This factor is used to construct the Jacobian correction matrix.
 *
 * @param sq_norm Squared L2 norm of the residual vector.
 * @param rho Loss function derivatives: rho.x = rho(s), rho.y = rho'(s), rho.z = rho''(s).
 * @return The alpha scaling factor, or 0 if sq_norm == 0 or rho.z <= 0.
 */
__device__ float jacobian_scaling_alpha(float sq_norm, const float3& rho) {
  assert(sq_norm >= 0.0);
  if ((sq_norm == 0.0) || (rho.z <= 0.0)) {
    return 0.0;
  }

  assert(rho.y > 0.0);

  const float D = 1.0 + 2.0 * sq_norm * rho.z / rho.y;

  const float alpha = 1.0 - sqrtf(D);
  return alpha / sq_norm;
}

/**
 * @brief Computes the residual scaling factor for robust loss correction.
 *
 * Scales the residual so that its squared norm equals rho(||r||^2).
 * The scaling is sqrt(rho') / (1 - alpha), where alpha accounts for
 * the curvature of the loss function.
 *
 * @param sq_norm Squared L2 norm of the residual vector.
 * @param rho Loss function derivatives: rho.x = rho(s), rho.y = rho'(s), rho.z = rho''(s).
 * @return The residual scaling factor.
 */
__device__ float residual_scaling(float sq_norm, const float3& rho) {
  assert(sq_norm >= 0);
  float sqrt_rho1_ = sqrtf(rho.y);

  if ((sq_norm == 0.0) || (rho.z <= 0.0)) {
    return sqrt_rho1_;
  }
  assert(rho.y > 0);

  const float D = 1.0 + 2.0 * sq_norm * rho.z / rho.y;

  const float alpha = 1.0 - sqrtf(D);
  return sqrt_rho1_ / (1 - alpha);
}

/**
 * @brief CUDA kernel to scale residuals by the robust loss function correction.
 *
 * Each warp processes one residual vector, computing the residual scaling
 * factor from the squared error and loss derivatives, then multiplying
 * each element of the residual by that factor.
 *
 * @param[in,out] residuals Residual values (num_residuals * residual_dim).
 * @param squared_error Squared L2 norm per residual (num_residuals).
 * @param rho Loss function derivatives per residual (num_residuals).
 * @param num_residuals Number of residual vectors.
 * @param residual_dim Dimension of each residual vector.
 *
 * Grid: (num_residuals / warps_per_block), Block: block_size.
 * One warp per residual.
 */
__global__ void scale_residuals_kernel(float* residuals, float* squared_error,
                                       float3* rho, int num_residuals,
                                       int residual_dim) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);
  auto wid = tile.meta_group_rank();
  int idx = blockIdx.x * tile.meta_group_size() + wid;
  if (idx >= num_residuals) {
    return;
  }
  auto lane = tile.thread_rank();
  float scaling = 0;
  if (lane == 0) {
    scaling = residual_scaling(squared_error[idx], rho[idx]);
  }
  scaling = tile.shfl(scaling, 0);

  for (auto i = lane; i < residual_dim; i += WARP_SIZE) {
    residuals[idx * residual_dim + i] *= scaling;
  }
}

/**
 * @brief CUDA kernel to compute the squared L2 norm of each residual vector.
 *
 * Each warp cooperatively computes the sum of squares for one residual
 * using warp-level reduction.
 *
 * @param residuals Input residual values (num_residuals * residual_dim).
 * @param[out] squared_error Output squared norm per residual (num_residuals).
 * @param num_residuals Number of residual vectors.
 * @param residual_dim Dimension of each residual vector.
 *
 * Grid: (num_residuals / warps_per_block), Block: block_size.
 * One warp per residual.
 */
__global__ void square_error_kernel(float* residuals, float* squared_error,
                                    int num_residuals, int residual_dim) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);
  auto wid = tile.meta_group_rank();
  int idx = blockIdx.x * tile.meta_group_size() + wid;
  if (idx >= num_residuals) {
    return;
  }
  auto lane = tile.thread_rank();
  float sum = 0.f;
  for (auto i = lane; i < residual_dim; i += WARP_SIZE) {
    float r = residuals[idx * residual_dim + i];
    sum += r * r;
  }
  sum = cg::reduce(tile, sum, cg::plus<float>());
  if (lane == 0) {
    squared_error[idx] = sum;
  }
}

/**
 * @brief Computes the outer product v * v^T using warp shuffles.
 *
 * Each thread computes one row of the outer product matrix. Column tiles
 * are distributed across the warp using __shfl_sync for register-level
 * data sharing, avoiding shared memory overhead.
 *
 * @param v Input vector of length N.
 * @param[out] C Output N x N matrix (v * v^T), stored in row-major order.
 * @param N Length of the input vector.
 * @param tid Thread ID within the block (determines which row to compute).
 */
__device__ void outer_product(const float* __restrict__ v,
                              float* __restrict__ C, int N, int tid) {
  const unsigned fullMask = 0xffffffffu;
  const int lane = tid % WARP_SIZE;

  // Each thread corresponds to one row inside the warp tile:
  const int row = tid;
  // load v[row] into register (0 if out of bounds)
  const float v_row = (row < N) ? v[row] : 0.0f;

  for (int colBase = 0; colBase < N; colBase += WARP_SIZE) {
    // each lane loads one element of the column tile into a register
    const int colIdxForLane = colBase + lane;
    const float v_col_lane = (colIdxForLane < N) ? v[colIdxForLane] : 0.0f;

    for (int k = 0; k < WARP_SIZE; ++k) {
      const float val_col = __shfl_sync(fullMask, v_col_lane, k);
      const int col = colBase + k;

      if (row < N && col < N) {
        C[(size_t)row * N + col] = v_row * val_col;
      }
    }
  }
}

/**
 * @brief CUDA kernel to scale Jacobians by the robust loss function correction.
 *
 * Constructs the scaling matrix S = sqrt(rho') * (I - alpha * r * r^T) and
 * left-multiplies the Jacobian block by S for each residual. Uses shared
 * memory for the outer product and scaling matrix computation.
 *
 * @param residuals Residual values (num_residuals * residual_dim).
 * @param[in,out] jacobians Jacobian values, scaled in-place.
 * @param squared_error Squared L2 norm per residual.
 * @param rho_coeffs Loss function derivatives per residual.
 * @param num_residuals Number of residual vectors.
 * @param residual_dim Dimension of each residual vector.
 * @param num_cols Number of Jacobian columns (sum of state block sizes).
 *
 * Grid: (1, num_residuals), Block: (WARP_SIZE, WARP_SIZE).
 * Shared memory: (residual_dim^2 + residual_dim * WARP_SIZE) * sizeof(float).
 */
__global__ void scale_jacobians_kernel(float* residuals, float* jacobians,
                                       float* squared_error, float3* rho_coeffs,
                                       int num_residuals, int residual_dim,
                                       int num_cols) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);
  extern __shared__ float sh_scaling_matrix[];
  float* sh_temp = sh_scaling_matrix + residual_dim * residual_dim;

  int thread_id = block.thread_rank();

  float sqrt_rho1 = 0;
  float alpha = 0;

  {
    // Calculate scalar coefficients first.
    // First lane of each warp reads the data.
    // Consecutive warps may access same addresses, but since warps are
    // different, no uncoaleased access happens.
    if (tile.thread_rank() == 0) {
      float sq_error = squared_error[blockIdx.y];
      auto rho = rho_coeffs[blockIdx.y];

      sqrt_rho1 = sqrtf(rho.y);
      alpha = jacobian_scaling_alpha(sq_error, rho);
    }

    // Populate data across the warp
    sqrt_rho1 = tile.shfl(sqrt_rho1, 0);
    alpha = tile.shfl(alpha, 0);
  }

  // Calculate the outer product of the residual with all threads in the block
  outer_product(residuals + residual_dim * blockIdx.y, sh_scaling_matrix,
                residual_dim, thread_id);
  block.sync();

  // Scale outer product to obtain the scaling matix
  for (int row = threadIdx.y; row < residual_dim; row += blockDim.y) {
    for (int col = threadIdx.x; col < residual_dim; col += blockDim.x) {
      float one_on_diag = row == col ? 1.f : 0.f;
      sh_scaling_matrix[row * residual_dim + col] =
          sqrt_rho1 *
          (one_on_diag - alpha * sh_scaling_matrix[row * residual_dim + col]);
    }
  }
  block.sync();
  // // Multiply the jacobian matrix by the scaling matrix from the left
  int row_offset = residual_dim * blockIdx.y;

  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == WARP_SIZE);

  for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
    for (int row = threadIdx.y; row < residual_dim; row += blockDim.y) {
      float sum = 0;
      for (int k = 0; k < residual_dim; k++) {
        sum += sh_scaling_matrix[row * residual_dim + k] *
               jacobians[num_cols * (row_offset + k) + col];
      }
      sh_temp[WARP_SIZE * row + threadIdx.x] = sum;
    }
    block.sync();
    for (int row = threadIdx.y; row < residual_dim; row += blockDim.y) {
      jacobians[num_cols * (row_offset + row) + col] =
          sh_temp[WARP_SIZE * row + threadIdx.x];
    }
    block.sync();
  }
}

/**
 * @brief CUDA kernel to extract per-residual cost from loss function output.
 *
 * Computes cost[i] = 0.5 * rho[i].x, where rho.x is the loss function
 * value at the squared residual norm.
 *
 * @param[out] cost Output per-residual cost values.
 * @param rho Loss function derivatives per residual.
 * @param num_residuals Number of residuals.
 *
 * Grid: (num_residuals / block_size), Block: block_size.
 */
__global__ void extract_cost_kernel(float* cost, float3* rho,
                                    int num_residuals) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_residuals) {
    return;
  }
  cost[tid] = 0.5 * rho[tid].x;
}

/**
 * @brief Constructs a ResidualBatch from a factor batch and loss function.
 *
 * @param factor_batch Pointer to the factor batch.
 * @param loss_function Pointer to the loss function batch, or nullptr for trivial loss.
 */
ResidualBatch::ResidualBatch(FactorBatch* factor_batch,
                             LossFunctionBatch* loss_function)
    : factor_batch_(factor_batch), loss_function_(loss_function) {}

/**
 * @brief Evaluates residuals, Jacobians, and/or cost for the current states.
 *
 * Execution steps:
 * 1. Evaluate the factor batch to get raw residuals and Jacobians
 * 2. Compute squared error per residual (||r_i||^2)
 * 3. Evaluate the loss function: rho(||r_i||^2)
 * 4. If loss function is non-trivial, scale Jacobians and residuals
 * 5. If cost is requested, extract cost = 0.5 * rho(||r_i||^2)
 *
 * @param[out] cost Output per-residual costs, or nullptr if not needed.
 * @param[out] residuals Output residual values (must not be nullptr).
 * @param[out] jacobians Output Jacobian values, or nullptr if not needed.
 * @param state_pointers Device pointer array to state blocks.
 * @param stream CUDA stream for GPU operations.
 * @return True on success.
 */
bool ResidualBatch::Evaluate(float* cost, float* residuals, float* jacobians,
                             float const* const* state_pointers,
                             cudaStream_t stream) const {
  assert(residuals != nullptr);
  assert(state_pointers != nullptr);

  // Calculate residuals and jacobians
  factor_batch_->Evaluate(residuals, jacobians, state_pointers, stream);

  size_t num_residuals = factor_batch_->NumFactors();
  size_t residual_dim = factor_batch_->ResidualsSize();

  // allocating CUDA memory at runtime
  thrust::device_vector<float> squared_error_(num_residuals);
  thrust::device_vector<float3> robustifier_coeffs_(num_residuals);

  auto sq_err_ptr = thrust::raw_pointer_cast(squared_error_.data());
  auto rho_ptr = thrust::raw_pointer_cast(robustifier_coeffs_.data());

  {
    // Calculate squared error
    size_t num_warps_per_block = block_size / WARP_SIZE;
    size_t num_blocks =
        (num_residuals + num_warps_per_block - 1) / num_warps_per_block;

    square_error_kernel<<<num_blocks, block_size, 0, stream>>>(
        residuals, sq_err_ptr, num_residuals, residual_dim);

    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  {
    // Apply loss function
    if (loss_function_ != nullptr) {
      loss_function_->Evaluate(sq_err_ptr, rho_ptr, num_residuals, stream);
    } else {
      TrivialLossFunctionBatch trivial_loss;
      trivial_loss.Evaluate(sq_err_ptr, rho_ptr, num_residuals, stream);
    }
    if (jacobians != nullptr && loss_function_ != nullptr) {
      // Apply correction to jacobians
      dim3 blocks(1, num_residuals);
      dim3 threads(WARP_SIZE, WARP_SIZE);

      size_t sh_mem_size =
          (residual_dim * residual_dim + residual_dim * WARP_SIZE) *
          sizeof(float);

      int num_cols = 0;
      for (const auto& param_dim : factor_batch_->StateBlockSizes()) {
        num_cols += param_dim;
      }

      scale_jacobians_kernel<<<blocks, threads, sh_mem_size, stream>>>(
          residuals, jacobians, sq_err_ptr, rho_ptr, num_residuals,
          residual_dim, num_cols);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }

    if (residuals != nullptr && loss_function_ != nullptr) {
      // Apply correction to residuals
      size_t num_warps_per_block = block_size / WARP_SIZE;
      size_t num_blocks =
          (num_residuals + num_warps_per_block - 1) / num_warps_per_block;
      scale_residuals_kernel<<<num_blocks, block_size, 0, stream>>>(
          residuals, sq_err_ptr, rho_ptr, num_residuals, residual_dim);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
  }

  if (cost != nullptr) {
    // Calculate cost
    size_t num_blocks = (num_residuals + block_size - 1) / block_size;
    extract_cost_kernel<<<num_blocks, block_size, 0, stream>>>(cost, rho_ptr,
                                                               num_residuals);

    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}
}  // namespace cunls
