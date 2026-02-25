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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/minimizer/jacobian_ops.h"

namespace cunls {

namespace {

/**
 * @brief Calculates the number of values in a Jacobian block for a cost
 * function.
 *
 * Each factor batch produces a dense Jacobian block of size
 * (num_factors * residual_dim) x (sum of state block sizes).
 *
 * @param factor_batch Pointer to the factor batch.
 * @return Total number of values in the Jacobian block.
 */
size_t CalculateJacobianBlockSize(const FactorBatch* factor_batch) {
  size_t num_factors = factor_batch->NumFactors();
  size_t num_rows = num_factors * factor_batch->ResidualsSize();

  auto block_sizes = factor_batch->StateBlockSizes();
  size_t num_cols = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);

  return num_rows * num_cols;
}

/**
 * @brief Calculates the total number of values in the full Jacobian.
 *
 * Sums the Jacobian block sizes across all factor batches.
 *
 * @param factor_batches Vector of factor batch pointers.
 * @return Total number of Jacobian values (including potential zeros for constant params).
 */
size_t CalculateJacobianNonZeros(
    const std::vector<FactorBatch*>& factor_batches) {
  size_t size = 0;
  for (const auto batch : factor_batches) {
    size += CalculateJacobianBlockSize(batch);
  }
  return size;
}

/**
 * @brief Holds metadata about a state batch for Jacobian column indexing.
 *
 * Stores the base device pointer, block dimensions, and a column-ID mapping
 * array used by the col_ids_kernel to compute Jacobian column indices.
 */
struct StateBatchDescription {
  /**
   * @brief Constructs from a state batch and its column-ID mapping.
   *
   * @param pbatch Pointer to the state batch.
   * @param p_col_mapping Device pointer to column-ID mapping array.
   */
  StateBatchDescription(const StateBatch* pbatch, const int* p_col_mapping)
      : ptr(pbatch->StateBlockDevicePtr(0)),
        tangent_dim(pbatch->TangentSize()),
        ambient_dim(pbatch->AmbientSize()),
        num_blocks(pbatch->NumStateBlocks()),
        col_ids(p_col_mapping) {}
  const float* ptr;       ///< Base device pointer to the first state block.
  size_t tangent_dim;     ///< Tangent (local) dimension of each state block.
  size_t ambient_dim;     ///< Ambient dimension of each state block.
  size_t num_blocks;      ///< Number of state blocks in the batch.
  const int* col_ids;     ///< Column-ID mapping: block index -> starting column in reduced system.
};

/**
 * @brief Holds metadata about a Jacobian block for a single factor batch.
 *
 * Describes the dimensions of the dense Jacobian sub-block produced by one
 * factor batch: num_rows x num_cols, where rows correspond to residuals
 * and columns correspond to state components.
 */
struct JacobianBlockDescription {
  /**
   * @brief Constructs from a factor batch.
   *
   * @param factor_batch Pointer to the factor batch.
   */
  JacobianBlockDescription(const FactorBatch* factor_batch)
      : residual_dim(factor_batch->ResidualsSize()) {
    size_t num_factors = factor_batch->NumFactors();
    num_rows = num_factors * factor_batch->ResidualsSize();

    auto block_sizes = factor_batch->StateBlockSizes();
    num_cols = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
    num_state_blocks_in_res = block_sizes.size();
  }
  size_t num_rows;          ///< Total number of rows (num_factors * residual_dim).
  size_t num_cols;  ///< Total number of columns (sum of state block sizes).
  size_t residual_dim;      ///< Dimension of a single residual vector.
  size_t num_state_blocks_in_res;  ///< Number of state blocks per residual.
};

/**
 * @brief CUDA kernel to assign row indices for a Jacobian block.
 *
 * Each thread writes the row index for one element of the dense Jacobian
 * block. The row index is offset by the number of rows from previously
 * processed factor batches.
 *
 * @param[out] start Output row index array (dense Jacobian block, row-major).
 * @param num_rows Number of rows in this Jacobian block.
 * @param num_cols Number of columns in this Jacobian block.
 * @param offset Row offset from previously processed factor batches.
 *
 * Grid: ((num_cols+7)/8, (num_rows+63)/64), Block: (8, 64).
 */
__global__ void row_ids_kernel(int* start, size_t num_rows, size_t num_cols,
                               int offset) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= num_cols || y >= num_rows) {
    return;
  }

  start[y * num_cols + x] = offset + y;
}

/**
 * @brief Fills the row index array for the full triplet Jacobian structure.
 *
 * Launches the row_ids_kernel for each factor batch, advancing
 * the output pointer and row offset across batches.
 *
 * @param stream CUDA stream for GPU operations.
 * @param factor_batches Vector of factor batch pointers.
 * @param[out] row_ids Output device vector of row indices.
 */
void FillRowIds(cudaStream_t stream,
                const std::vector<FactorBatch*>& factor_batches,
                dvector<int>& row_ids) {
  dim3 block(8, 64);

  int* ptr = row_ids.data();
  int offset = 0;
  for (auto batch : factor_batches) {
    JacobianBlockDescription j_desc(batch);
    dim3 grid((j_desc.num_cols + block.x - 1) / block.x,
              (j_desc.num_rows + block.y - 1) / block.y);

    row_ids_kernel<<<grid, block, 0, stream>>>(ptr, j_desc.num_rows,
                                               j_desc.num_cols, offset);

    THROW_ON_CUDA_ERROR(cudaGetLastError());

    ptr += j_desc.num_rows * j_desc.num_cols;
    offset += j_desc.num_rows;
  }
}

/**
 * @brief CUDA kernel to assign column indices for a Jacobian block.
 *
 * For each element in the dense Jacobian block, determines which state
 * block the column belongs to by matching the state pointer against the
 * state batch's base pointer. Constant state pointers receive col_id = -1.
 *
 * @param[out] start Output column index array (dense Jacobian block,
 * row-major).
 * @param state_pointers Device pointer array mapping factor instances to
 * state_pointers.
 * @param col_block_index Index of the current state block within the residual.
 * @param j_desc Jacobian block dimensions and metadata.
 * @param pb_desc State batch metadata with column-ID mapping.
 *
 * Grid: ((tangent_dim+7)/8, (num_rows+63)/64), Block: (8, 64). Max 512 threads.
 */
__launch_bounds__(512) __global__
    void col_ids_kernel(int* __restrict__ start,
                        float const* const* __restrict__ state_pointers,
                        int col_block_index, JacobianBlockDescription j_desc,
                        StateBatchDescription pb_desc) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  assert(col_block_index < j_desc.num_state_blocks_in_res);

  if (x >= pb_desc.tangent_dim || y >= j_desc.num_rows) {
    return;
  }

  int factor_id = y / j_desc.residual_dim;

  float const* ptr = state_pointers[factor_id * j_desc.num_state_blocks_in_res +
                                    col_block_index];

  int pidx = (ptr - pb_desc.ptr) / pb_desc.ambient_dim;

  bool valid = pidx >= 0 && pidx < pb_desc.num_blocks;
  int param_col_id = valid ? pb_desc.col_ids[pidx] : -1;

  if (param_col_id != -1) {
    start[y * j_desc.num_cols + x] = param_col_id + x;
  }
}

/**
 * @brief Creates a mapping from state block indices to column IDs in the
 * reduced system.
 *
 * For each state block in the batch, assigns a starting column index in
 * the reduced (non-constant) system. Constant state_pointers are marked with
 * -1. Uses exclusive scan to compute cumulative column offsets.
 *
 * @param stream CUDA stream for GPU operations.
 * @param offset Starting column index for this state batch.
 * @param state_batch The state batch.
 * @param[out] col_ids Device vector of column IDs (one per state block).
 */
void CreateStateColIdMapping(cudaStream_t stream, int offset,
                             const StateBatch* state_batch,
                             thrust::device_vector<int>& col_ids) {
  const int* cparam_ids = state_batch->ConstStateIds();
  size_t num_const_params = state_batch->NumConstStateBlocks();
  size_t tangent_dim = state_batch->TangentSize();
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  col_ids.resize(state_batch->NumStateBlocks());
  thrust::fill(stream_policy, col_ids.begin(), col_ids.end(),
               static_cast<int>(tangent_dim));

  if (cparam_ids != nullptr && num_const_params > 0) {
    auto zero_it = thrust::make_constant_iterator(0);
    thrust::device_ptr<const int> cparam_ids_ptr(cparam_ids);
    thrust::scatter(stream_policy, zero_it, zero_it + num_const_params,
                    cparam_ids_ptr, col_ids.begin());
    thrust::exclusive_scan(stream_policy, col_ids.begin(), col_ids.end(),
                           col_ids.begin(), offset);
    auto minus_one_it = thrust::make_constant_iterator(-1);
    thrust::scatter(stream_policy, minus_one_it,
                    minus_one_it + num_const_params, cparam_ids_ptr,
                    col_ids.begin());
  } else {
    thrust::exclusive_scan(stream_policy, col_ids.begin(), col_ids.end(),
                           col_ids.begin(), offset);
  }
}

/**
 * @brief Fills column indices for a Jacobian block corresponding to one factor
 * batch.
 *
 * Launches the col_ids_kernel for each state block referenced by the cost
 * function. Skips state blocks whose tangent dimension doesn't match the
 * expected block size.
 *
 * @param stream CUDA stream for GPU operations.
 * @param[out] start Pointer into the column index array for this Jacobian
 * block.
 * @param factor_batch The factor batch.
 * @param param_ptr Device state pointers for this factor batch.
 * @param pb_desc State batch description with column-ID mapping.
 */
void FillColIdsInJacobianBlock(cudaStream_t stream, int* start,
                               const FactorBatch* factor_batch,
                               const DeviceVector<float*>& param_ptr,
                               const StateBatchDescription& pb_desc) {
  auto block_sizes = factor_batch->StateBlockSizes();

  float const* const* state_pointers = param_ptr.data();

  JacobianBlockDescription j_desc(factor_batch);

  const dim3 block(8, 64);
  const dim3 grid((pb_desc.tangent_dim + block.x - 1) / block.x,
                  (j_desc.num_rows + block.y - 1) / block.y);

  int* col_ptr = start;
  for (size_t i = 0; i < block_sizes.size(); i++) {
    if (pb_desc.tangent_dim != block_sizes[i]) {
      // Mismatch in tangent dims, skip this col block
      col_ptr += block_sizes[i];
      continue;
    }

    col_ids_kernel<<<grid, block, 0, stream>>>(col_ptr, state_pointers, i,
                                               j_desc, pb_desc);

    THROW_ON_CUDA_ERROR(cudaGetLastError());

    col_ptr += block_sizes[i];
  }
}

/**
 * @brief Fills the column index array for the full triplet Jacobian structure.
 *
 * Iterates over state batches and factor batches to assign column
 * indices in the reduced linear system. Initializes all column IDs to -1
 * (invalid), then populates valid entries via FillColIdsInJacobianBlock.
 *
 * @param stream CUDA stream for GPU operations.
 * @param factor_batches Vector of factor batch pointers.
 * @param state_pointers Device state pointers per factor batch.
 * @param state_batches Vector of state batch pointers.
 * @param[out] col_ids Output device vector of column indices.
 */
void FillColIds(cudaStream_t stream,
                const std::vector<FactorBatch*>& factor_batches,
                const std::vector<DeviceVector<float*>>& state_pointers,
                const std::vector<StateBatch*>& state_batches,
                dvector<int>& col_ids) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  // Set all the column indices to be invalid, i.e. -1
  thrust::device_ptr<int> col_ids_ptr(col_ids.data());
  thrust::fill(stream_policy, col_ids_ptr, col_ids_ptr + col_ids.size(), -1);

  // Buffer to store the mapping between param pointers and
  // column ids
  thrust::device_vector<int> pbatch_cols;
  int last_col_id = 0;
  for (auto pbatch : state_batches) {
    // Create a mapping. Account for the number of already processed
    // state_pointers.
    CreateStateColIdMapping(stream, last_col_id, pbatch, pbatch_cols);

    StateBatchDescription pb_desc(
        pbatch, thrust::raw_pointer_cast(pbatch_cols.data()));

    // A pointer to the start of the jacobian block for each factor batch.
    int* start = col_ids.data();
    for (size_t j = 0; j < factor_batches.size(); j++) {
      auto batch = factor_batches[j];

      // Computes column indices for the jacobian block for this factor batch.
      FillColIdsInJacobianBlock(stream, start, batch, state_pointers[j],
                                pb_desc);

      // Update the pointer s.t in points to the next jacobian block
      start += CalculateJacobianBlockSize(batch);
    }

    // Update the last column index
    last_col_id += (pbatch->NumStateBlocks() - pbatch->NumConstStateBlocks()) *
                   pbatch->TangentSize();
  }
}

/**
 * @brief Builds a host-side mapping from state device pointers to column
 * indices.
 *
 * Creates a hash map that maps each non-constant state block's device
 * pointer to its starting column index in the reduced linear system.
 * Used by the CPU implementation of BuildTripletSparseStructureCPU.
 *
 * @param state_batches Vector of state batch pointers.
 * @param[out] mapping Output map from device pointer to column index.
 */
void BuildStateBlockMapping(const std::vector<StateBatch*>& state_batches,
                            std::unordered_map<float*, int>& mapping) {
  int last_col_id = 0;

  for (auto pbatch : state_batches) {
    const int* gpu_const_param_ids = pbatch->ConstStateIds();
    size_t num_const_params = pbatch->NumConstStateBlocks();

    std::unordered_set<int> const_param_ids_set;
    if (gpu_const_param_ids != nullptr && num_const_params > 0) {
      std::vector<int> cpu_const_param_ids(num_const_params);
      THROW_ON_CUDA_ERROR(cudaMemcpy(cpu_const_param_ids.data(), gpu_const_param_ids,
                                     num_const_params * sizeof(int),
                                     cudaMemcpyDeviceToHost));
      const_param_ids_set.insert(cpu_const_param_ids.begin(),
                                 cpu_const_param_ids.end());
    }

    for (size_t i = 0; i < pbatch->NumStateBlocks(); i++) {
      auto it = const_param_ids_set.find(i);
      if (it == const_param_ids_set.end()) {
        auto ptr = pbatch->StateBlockDevicePtr(i);
        mapping.insert({ptr, last_col_id});
        last_col_id += pbatch->TangentSize();
      }
    }
  }
}

}  // namespace

/**
 * @brief Builds the triplet (COO) sparse structure for the Jacobian on the GPU.
 *
 * Extracts factor batches from the problem's residual batches, calculates the
 * total Jacobian size, and fills row and column index arrays using CUDA
 * kernels. Column indices for constant states are set to -1.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param[out] structure Output triplet sparse structure (row_ids and col_ids).
 */
void BuildTripletSparseStructure(cudaStream_t stream, const Problem& problem,
                                 TripletSparseStructure& structure) {
  std::vector<FactorBatch*> factor_batches;
  factor_batches.reserve(problem.GetResidualBatches().size());

  for (const auto& rb : problem.GetResidualBatches()) {
    factor_batches.push_back(rb.GetFactorBatch());
  }

  // Calculate the total size of the jacobian
  size_t jacobian_size = CalculateJacobianNonZeros(factor_batches);
  structure.row_ids.resize(jacobian_size);
  structure.col_ids.resize(jacobian_size);

  // Create the triplet structure
  FillRowIds(stream, factor_batches, structure.row_ids);
  FillColIds(stream, factor_batches, problem.GetStatePointers(),
             problem.GetStateBatches(), structure.col_ids);
}

/**
 * @brief Builds the triplet (COO) sparse structure for the Jacobian on the CPU.
 *
 * Host-side implementation that copies state pointers to the CPU, builds
 * a pointer-to-column-index mapping, and iterates over all factor batches to
 * assign row and column indices. Constant state_pointers receive col_id = -1.
 * Results are uploaded to device vectors via host-to-device copy.
 *
 * @param problem The optimization problem.
 * @param[out] structure Output triplet sparse structure (device vectors).
 */
void BuildTripletSparseStructureCPU(const Problem& problem,
                                    TripletSparseStructure& structure) {
  const auto& state_pointers = problem.GetStatePointers();
  const auto& state_batches = problem.GetStateBatches();

  std::vector<FactorBatch*> factor_batches;
  factor_batches.reserve(problem.GetResidualBatches().size());

  for (const auto& rb : problem.GetResidualBatches()) {
    factor_batches.emplace_back(rb.GetFactorBatch());
  }

  size_t jacobian_size = CalculateJacobianNonZeros(factor_batches);

  std::vector<int> row_ids(jacobian_size);
  std::vector<int> col_ids(jacobian_size);

  std::vector<std::vector<float*>> state_pointers_cpu;

  for (const auto& pointers : state_pointers) {
    std::vector<float*> host_pointers(pointers.size());
    pointers.CopyToHost(host_pointers.data(), pointers.size());
    state_pointers_cpu.emplace_back(std::move(host_pointers));
  }

  std::unordered_map<float*, int> mapping;
  BuildStateBlockMapping(state_batches, mapping);

  size_t block_offset = 0;
  size_t row_offset = 0;
  for (int i = 0; i < factor_batches.size(); i++) {
    auto factor_batch = factor_batches[i];
    auto p_pointers = state_pointers_cpu[i];

    auto block_sizes = factor_batch->StateBlockSizes();
    size_t num_params = block_sizes.size();
    size_t res_dim = factor_batch->ResidualsSize();
    size_t num_rows = res_dim * factor_batch->NumFactors();
    size_t num_cols =
        std::accumulate(block_sizes.begin(), block_sizes.end(), 0);

    for (int row_id = 0; row_id < num_rows; row_id++) {
      int factor_idx = row_id / res_dim;
      for (int col_id = 0; col_id < num_cols; col_id++) {
        int idx = block_offset + row_id * num_cols + col_id;
        row_ids[idx] = row_offset + row_id;

        // State block index in the residual state block list
        int state_block_id = 0;

        // Index within the current state block (column in that block)
        int col_in_state_block = col_id;
        while (state_block_id < block_sizes.size() &&
               col_in_state_block >= block_sizes[state_block_id]) {
          col_in_state_block -= block_sizes[state_block_id];
          state_block_id++;
        }

        assert(state_block_id < block_sizes.size());

        // Extract the pointer to the state block w.r.t. Jacobian block
        float* state_pointer = p_pointers[factor_idx * num_params + state_block_id];

        auto it = mapping.find(state_pointer);

        if (it != mapping.end()) {
          // This is non constant state
          col_ids[idx] = it->second + col_in_state_block;
        } else {
          // This is constant state, set invalid column index
          col_ids[idx] = -1;
        }
      }
    }

    block_offset += num_rows * num_cols;
    row_offset += num_rows;
  }

  structure.row_ids.resize(row_ids.size());
  structure.row_ids.CopyFromHost(row_ids.data(), row_ids.size());

  structure.col_ids.resize(col_ids.size());
  structure.col_ids.CopyFromHost(col_ids.data(), col_ids.size());
}
}  // namespace cunls
