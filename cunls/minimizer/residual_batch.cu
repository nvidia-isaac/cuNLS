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

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "cunls/common/helper.h"
#include "cunls/minimizer/residual_batch.h"

namespace cunls {

constexpr int kBlockSize = 256;

namespace {

/** The squared norms and rho triplets in the workspace (ResidualBatchWorkspaceSizeBytes). */
void MapRobustWorkspace(float *workspace, size_t num_residuals, float **sq_err_out,
                        float3 **rho_out) {
  *sq_err_out = workspace;
  const size_t rho_byte_offset =
      ResidualBatchWorkspaceSizeBytes(num_residuals) - num_residuals * sizeof(float3);
  *rho_out = reinterpret_cast<float3 *>(reinterpret_cast<uint8_t *>(workspace) + rho_byte_offset);
}

}  // namespace

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float jacobian_scaling_alpha(float sq_norm, const float3 &rho) {
  if ((sq_norm == 0.0f) || (rho.z <= 0.0f)) return 0.0f;
  const float D = 1.0f + 2.0f * sq_norm * rho.z / rho.y;
  return (1.0f - sqrtf(D)) / sq_norm;
}

__device__ __forceinline__ float residual_scaling(float sq_norm, const float3 &rho) {
  float sqrt_rho1 = sqrtf(rho.y);
  if ((sq_norm == 0.0f) || (rho.z <= 0.0f)) return sqrt_rho1;
  const float D = 1.0f + 2.0f * sq_norm * rho.z / rho.y;
  return sqrt_rho1 / (1.0f - (1.0f - sqrtf(D)));
}

// ============================================================================
// Kernel 1: Cost of the trivial loss
// ============================================================================
// No robust loss: residuals and Jacobians stay as they are and the cost is
// 0.5 * ||r||^2, one thread per factor.

__global__ void trivial_loss_cost_kernel(const float *__restrict__ residuals,
                                         float *__restrict__ cost, int num_residuals,
                                         int residual_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_residuals) return;

  const float *r = residuals + tid * residual_dim;
  float sum = 0.0f;
  for (int i = 0; i < residual_dim; ++i) {
    float v = r[i];
    sum += v * v;
  }
  cost[tid] = 0.5f * sum;
}

// ============================================================================
// Kernel 2: Squared-error (thread-per-factor, for small residual dims)
// ============================================================================
// Replaces the old warp-per-factor kernel with one thread per factor.
// For typical NLS problems (residual dim 1-15), this eliminates 95%+ of
// wasted warp lanes.

__global__ void square_error_thread_kernel(const float *__restrict__ residuals,
                                           float *__restrict__ squared_error, int num_residuals,
                                           int residual_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_residuals) return;

  const float *r = residuals + tid * residual_dim;
  float sum = 0.0f;
  for (int i = 0; i < residual_dim; ++i) {
    float v = r[i];
    sum += v * v;
  }
  squared_error[tid] = sum;
}

// ============================================================================
// Kernel 3: Scale residuals (thread-per-factor)
// ============================================================================
// One thread per factor; each thread computes the scaling factor once,
// then applies it to all residual_dim elements. Eliminates warp-shuffle
// overhead and wasted lanes for small residual dims.

__global__ void scale_residuals_thread_kernel(float *__restrict__ residuals,
                                              const float *__restrict__ squared_error,
                                              const float3 *__restrict__ rho, int num_residuals,
                                              int residual_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_residuals) return;

  float scaling = residual_scaling(squared_error[tid], rho[tid]);
  float *r = residuals + tid * residual_dim;
  for (int i = 0; i < residual_dim; ++i) {
    r[i] *= scaling;
  }
}

// ============================================================================
// Kernel 4: Scale Jacobians
// ============================================================================
// S*J_col = sqrt_rho1*(J_col - alpha*(r'*J_col)*r)
//
// One thread per (factor, Jacobian column). The columns of a factor are
// consecutive threads, so each Jacobian row is read and written coalesced and
// every lane is busy (a warp per factor left 23 of 32 lanes idle on a 2x9
// reprojection Jacobian). The residual and the loss coefficients of a factor
// are shared by its columns' threads (cache hits).
__global__ void scale_jacobians_kernel(const float *__restrict__ residuals,
                                       float *__restrict__ jacobians,
                                       const float *__restrict__ squared_error,
                                       const float3 *__restrict__ rho_coeffs, int num_residuals,
                                       int residual_dim, int num_cols) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(num_residuals) * num_cols) return;
  const int factor = static_cast<int>(i / num_cols);
  const int col = static_cast<int>(i - static_cast<long long>(factor) * num_cols);
  const float3 rh = rho_coeffs[factor];
  const float sqrt_rho1 = sqrtf(rh.y);
  const float alpha = jacobian_scaling_alpha(squared_error[factor], rh);
  const float *r = residuals + static_cast<long long>(factor) * residual_dim;
  float *J = jacobians + static_cast<long long>(factor) * residual_dim * num_cols + col;
  float dot = 0.0f;
  for (int row = 0; row < residual_dim; ++row) dot += r[row] * J[row * num_cols];
  const float scale = sqrt_rho1 * alpha * dot;
  for (int row = 0; row < residual_dim; ++row) {
    J[row * num_cols] = sqrt_rho1 * J[row * num_cols] - scale * r[row];
  }
}

// ============================================================================
// Kernel 5: Extract cost
// ============================================================================
__global__ void extract_cost_kernel(float *__restrict__ cost, const float3 *__restrict__ rho,
                                    int num_residuals) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_residuals) return;
  cost[tid] = 0.5f * rho[tid].x;
}

// ============================================================================
// ResidualBatch implementation
// ============================================================================

ResidualBatch::ResidualBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function)
    : factor_batch_(factor_batch), loss_function_(loss_function) {}

bool ResidualBatch::Evaluate(cudaStream_t stream, float *workspace, float *residuals,
                             float const *const *state_pointers, float *cost, float *jacobians,
                             const int *factor_ids, size_t num_factor_ids) const {
  const size_t items = num_factor_ids == 0 ? factor_batch_->NumActiveFactors() : num_factor_ids;

  // Return before the preconditions below: an empty batch has nothing to
  // evaluate, its buffers are legitimately null (a zero-size DeviceVector has
  // no allocation), and the factor kernels would be launched with a zero-size
  // grid.
  if (items == 0 || factor_batch_->NumActiveFactors() == 0) {
    return true;
  }

  assert(residuals != nullptr);
  assert(state_pointers != nullptr);
  assert(workspace != nullptr);

  if (!factor_batch_->Evaluate(residuals, jacobians, state_pointers, stream, factor_ids, items)) {
    return false;
  }
  return ApplyLossToItems(stream, workspace, residuals, cost, jacobians, items);
}

bool ResidualBatch::ApplyLoss(cudaStream_t stream, float *workspace, float *residuals, float *cost,
                              float *jacobians) const {
  return ApplyLossToItems(stream, workspace, residuals, cost, jacobians,
                          factor_batch_->NumActiveFactors());
}

bool ResidualBatch::ApplyLossToItems(cudaStream_t stream, float *workspace, float *residuals,
                                     float *cost, float *jacobians, size_t num_items) const {
  int num_residuals = static_cast<int>(num_items);
  int residual_dim = static_cast<int>(factor_batch_->ResidualsSize());

  if (num_residuals == 0) {
    return true;
  }

  assert(residuals != nullptr);
  assert(workspace != nullptr);

  int num_thread_blocks = (num_residuals + kBlockSize - 1) / kBlockSize;

  if (loss_function_ == nullptr) {
    // Trivial loss: nothing to scale, only the cost (if asked for).
    if (cost != nullptr) {
      trivial_loss_cost_kernel<<<num_thread_blocks, kBlockSize, 0, stream>>>(
          residuals, cost, num_residuals, residual_dim);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    return true;
  }

  float *sq_err_ptr = nullptr;
  float3 *rho_ptr = nullptr;
  MapRobustWorkspace(workspace, num_residuals, &sq_err_ptr, &rho_ptr);

  // Robust loss path: need squared errors for the loss function.
  square_error_thread_kernel<<<num_thread_blocks, kBlockSize, 0, stream>>>(
      residuals, sq_err_ptr, num_residuals, residual_dim);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  loss_function_->Evaluate(sq_err_ptr, rho_ptr, num_residuals, stream);

  if (jacobians != nullptr) {
    int num_cols = 0;
    for (const auto &d : factor_batch_->StateSizes()) num_cols += d;

    const size_t num_threads = static_cast<size_t>(num_residuals) * num_cols;
    if (num_threads > 0) {
      scale_jacobians_kernel<<<static_cast<unsigned>((num_threads + kBlockSize - 1) / kBlockSize),
                               kBlockSize, 0, stream>>>(residuals, jacobians, sq_err_ptr, rho_ptr,
                                                        num_residuals, residual_dim, num_cols);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
  }

  // Scale residuals
  scale_residuals_thread_kernel<<<num_thread_blocks, kBlockSize, 0, stream>>>(
      residuals, sq_err_ptr, rho_ptr, num_residuals, residual_dim);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  if (cost != nullptr) {
    extract_cost_kernel<<<num_thread_blocks, kBlockSize, 0, stream>>>(cost, rho_ptr, num_residuals);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

size_t EvaluateFactorCosts(cudaStream_t stream, const std::vector<ResidualBatch> &residual_batches,
                           const std::function<float const *const *(size_t)> &state_pointers,
                           dvector<float> &buffer) {
  size_t num_costs = 0, max_residuals = 0, max_workspace = 0;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    if (state_pointers(i) == nullptr) continue;
    const FactorBatch *factor_batch = residual_batches[i].GetFactorBatch();
    const size_t n = factor_batch->NumActiveFactors();
    num_costs += n;
    max_residuals = std::max(max_residuals, n * factor_batch->ResidualsSize());
    max_workspace = std::max(max_workspace, ResidualBatchWorkspaceNumFloats(n));
  }
  if (buffer.size() < num_costs + max_residuals + max_workspace) {
    buffer.resize(num_costs + max_residuals + max_workspace);
  }
  float *cost = buffer.data();
  float *residuals = cost + num_costs;
  float *workspace = residuals + max_residuals;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    float const *const *pointers = state_pointers(i);
    if (pointers == nullptr) continue;
    const ResidualBatch &rb = residual_batches[i];
    CheckEvaluate(rb.Evaluate(stream, workspace, residuals, pointers, cost, nullptr), i);
    cost += rb.GetFactorBatch()->NumActiveFactors();
  }
  return num_costs;
}

}  // namespace cunls
