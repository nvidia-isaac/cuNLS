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

#include "cunls/common/helper.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"

namespace cunls {
namespace cg = cooperative_groups;
#define WARP_SIZE 32

constexpr size_t block_size = 256;  ///< Default thread block size for CUDA kernels.

namespace {

/** @brief Maps `workspace` to squared-error and rho pointers (layout matches `ResidualBatchWorkspaceSizeBytes`). */
void MapRobustWorkspace(float* workspace, size_t num_residuals, float** sq_err_out,
                        float3** rho_out) {
  *sq_err_out = workspace;
  const size_t sq_bytes = num_residuals * sizeof(float);
  const size_t align = alignof(float3);
  const size_t rho_byte_offset = (sq_bytes + align - 1u) / align * align;
  *rho_out = reinterpret_cast<float3*>(reinterpret_cast<uint8_t*>(workspace) +
                                        rho_byte_offset);
}

}  // namespace

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

// Shared memory layout for scale_jacobians_kernel: sh_r[residual_dim],
// sh_partial[WARP_SIZE][WARP_SIZE], sh_dot[WARP_SIZE].
#define SCALE_JACOB_SHMEM_R(residual_dim) (residual_dim)
#define SCALE_JACOB_SHMEM_PARTIAL (WARP_SIZE * WARP_SIZE)
#define SCALE_JACOB_SHMEM_DOT (WARP_SIZE)

/**
 * @brief CUDA kernel to scale Jacobians by the robust loss function correction.
 *
 * Uses S*v = sqrt(rho')*(v - alpha*(r'*v)*r) so the full scaling matrix S is
 * never stored. For each column J_col we compute dot = r'*J_col, then
 * (S*J)_col = sqrt_rho1*(J_col - alpha*dot*r). Reduces shared memory and
 * register pressure vs building S explicitly.
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
 * Shared memory: (residual_dim + WARP_SIZE*WARP_SIZE + WARP_SIZE) * sizeof(float).
 */
__global__ void scale_jacobians_kernel(float* residuals, float* jacobians,
                                       float* squared_error, float3* rho_coeffs,
                                       int num_residuals, int residual_dim,
                                       int num_cols) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);

  extern __shared__ float sh[];
  float* sh_r = sh;
  float* sh_partial = sh + SCALE_JACOB_SHMEM_R(residual_dim);
  float* sh_dot = sh_partial + SCALE_JACOB_SHMEM_PARTIAL;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_offset = residual_dim * blockIdx.y;
  const float* res_base = residuals + row_offset;

  // Lane 0 loads coefficients and broadcasts
  float sqrt_rho1 = 0.f;
  float alpha = 0.f;
  if (tile.thread_rank() == 0) {
    float sq_error = squared_error[blockIdx.y];
    float3 rho = rho_coeffs[blockIdx.y];
    sqrt_rho1 = sqrtf(rho.y);
    alpha = jacobian_scaling_alpha(sq_error, rho);
  }
  sqrt_rho1 = tile.shfl(sqrt_rho1, 0);
  alpha = tile.shfl(alpha, 0);

  // Load residual into shared memory (cooperative)
  for (int i = ty * blockDim.x + tx; i < residual_dim; i += blockDim.x * blockDim.y) {
    sh_r[i] = res_base[i];
  }
  block.sync();

  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == WARP_SIZE);

  // Process 32 columns per batch. S*J_col = sqrt_rho1*(J_col - alpha*(r'*J_col)*r).
  // Use fixed iteration count so all threads hit the same block.sync()s (avoids
  // deadlock when num_cols is not a multiple of WARP_SIZE).
  const int num_col_batches = (num_cols + blockDim.x - 1) / blockDim.x;
  for (int batch = 0; batch < num_col_batches; batch++) {
    const int col = batch * blockDim.x + tx;
    const bool active = (col < num_cols);

    // Partial dot: r'*J_col over rows this thread owns
    float partial = 0.f;
    if (active) {
      for (int row = ty; row < residual_dim; row += blockDim.y) {
        partial += sh_r[row] * jacobians[num_cols * (row_offset + row) + col];
      }
    }
    sh_partial[ty * WARP_SIZE + tx] = partial;
    block.sync();

    // Reduce partials to get dot = r'*J_col (one thread per column)
    if (ty == 0) {
      float dot = 0.f;
      for (int i = 0; i < WARP_SIZE; i++) {
        dot += sh_partial[i * WARP_SIZE + tx];
      }
      sh_dot[tx] = dot;
    }
    block.sync();

    float dot = sh_dot[tx];
    const float scale = sqrt_rho1 * alpha * dot;

    if (active) {
      for (int row = ty; row < residual_dim; row += blockDim.y) {
        int idx = num_cols * (row_offset + row) + col;
        jacobians[idx] = sqrt_rho1 * jacobians[idx] - scale * sh_r[row];
      }
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

bool ResidualBatch::Evaluate(cudaStream_t stream, float* workspace,
                             float* residuals, float const* const* state_pointers,
                             float* cost, float* jacobians) const {
  assert(residuals != nullptr);
  assert(state_pointers != nullptr);

  factor_batch_->Evaluate(residuals, jacobians, state_pointers, stream);

  size_t num_residuals = factor_batch_->NumFactors();
  size_t residual_dim = factor_batch_->ResidualsSize();

  if (num_residuals == 0) {
    return true;
  }
  assert(workspace != nullptr);

  float* sq_err_ptr = nullptr;
  float3* rho_ptr = nullptr;
  MapRobustWorkspace(workspace, num_residuals, &sq_err_ptr, &rho_ptr);

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
          (SCALE_JACOB_SHMEM_R(residual_dim) + SCALE_JACOB_SHMEM_PARTIAL +
           SCALE_JACOB_SHMEM_DOT) *
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
