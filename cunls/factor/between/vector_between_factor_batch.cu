/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/factor/between/vector_between_factor_batch.h"

namespace cunls {

namespace {

/**
 * Factor (measurement) index of evaluation item `item`: factor_ids[item], or
 * item modulo the batch size when factor_ids is null (see FactorBatch::Evaluate).
 */
__device__ __forceinline__ int FactorMeasurementIndex(int item, const int *factor_ids,
                                                      int num_factors) {
  if (factor_ids != nullptr) {
    return factor_ids[item];
  }
  return item < num_factors ? item : item % num_factors;
}

}  // namespace

constexpr int kVectorBetweenBlock = 256;

__global__ void vector_between_kernel(const float *deltas, float const *const *state_pointers,
                                      float *residuals, float *jacobians, int dim, int num_items,
                                      const int *factor_ids, int num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_items) {
    return;
  }

  const float *left = state_pointers[2 * tid];
  const float *right = state_pointers[2 * tid + 1];
  const float *d = deltas + FactorMeasurementIndex(tid, factor_ids, num_factors) * dim;
  float *res = residuals + tid * dim;
  for (int i = 0; i < dim; ++i) {
    res[i] = left[i] - right[i] - d[i];
  }

  if (jacobians == nullptr) {
    return;
  }

  float *J = jacobians + tid * dim * (2 * dim);
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < 2 * dim; ++c) {
      J[r * (2 * dim) + c] = 0.f;
    }
  }
  for (int i = 0; i < dim; ++i) {
    J[i * (2 * dim) + i] = 1.f;
    J[i * (2 * dim) + dim + i] = -1.f;
  }
}

void LaunchVectorBetweenFactorKernel(const float *deltas, float const *const *state_pointers,
                                     float *residuals, float *jacobians, int dim, int num_items,
                                     const int *factor_ids, int num_factors, cudaStream_t stream) {
  int nb = (num_items + kVectorBetweenBlock - 1) / kVectorBetweenBlock;
  vector_between_kernel<<<nb, kVectorBetweenBlock, 0, stream>>>(
      deltas, state_pointers, residuals, jacobians, dim, num_items, factor_ids, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
