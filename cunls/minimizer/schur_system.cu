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

#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <stdexcept>
#include <string>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/minimizer/schur_system.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {
namespace {

using internal::schur::BatchGeometry;
using internal::schur::ColumnRange;

constexpr int kThreads = 256;
constexpr unsigned kFullMask = 0xffffffffu;
/// Pose-major pairs per CTA of the off-diagonal pass (long rows are split).
constexpr int kChunkPairs = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/** Bits needed for the values [0, n). */
int BitsFor(long long n) {
  int bits = 1;
  while ((1LL << bits) < n) ++bits;
  return bits;
}

/** Index of (j, k), j <= k, in the packed upper triangle of a d x d matrix. */
__host__ __device__ constexpr int Tri(int j, int k, int d) {
  return j * d - j * (j - 1) / 2 + (k - j);
}

/** The residual batch of global factor g: the last one starting at or before g. */
__device__ __forceinline__ int FindBatch(const BatchGeometry *batches, int num_batches, int g) {
  int lo = 0, hi = num_batches - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (batches[mid].factor_begin <= g) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/** The column range holding free column `col`. */
__device__ __forceinline__ const ColumnRange &FindRange(const ColumnRange *ranges, int num_ranges,
                                                        int col) {
  int r = 0;
  while (r + 1 < num_ranges && col >= ranges[r].end_col) ++r;
  return ranges[r];
}

/** Landmark or retained block id of free column `col` in `range`. */
__device__ __forceinline__ int IdOf(const ColumnRange &range, int col) {
  return range.first_id + (col - range.first_col) / range.tangent;
}

// ---------------------------------------------------------------------------
// CUB wrappers (setup only)
// ---------------------------------------------------------------------------

template <typename K, typename V>
void SortPairs(cudaStream_t stream, dvector<uint8_t> &temp, const K *keys_in, K *keys_out,
               const V *values_in, V *values_out, int n, int end_bit) {
  if (n == 0) return;
  size_t bytes = 0;
  THROW_ON_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_in, keys_out, values_in,
                                                      values_out, n, 0, end_bit, stream));
  if (temp.size() < bytes) temp.resize(bytes);
  THROW_ON_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(
      temp.data(), bytes, keys_in, keys_out, values_in, values_out, n, 0, end_bit, stream));
}

template <typename T>
void ExclusiveSum(cudaStream_t stream, dvector<uint8_t> &temp, const T *in, T *out, int n) {
  if (n == 0) return;
  size_t bytes = 0;
  THROW_ON_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, stream));
  if (temp.size() < bytes) temp.resize(bytes);
  THROW_ON_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(temp.data(), bytes, in, out, n, stream));
}

template <typename T>
void InclusiveSum(cudaStream_t stream, dvector<uint8_t> &temp, const T *in, T *out, int n) {
  if (n == 0) return;
  size_t bytes = 0;
  THROW_ON_CUDA_ERROR(cub::DeviceScan::InclusiveSum(nullptr, bytes, in, out, n, stream));
  if (temp.size() < bytes) temp.resize(bytes);
  THROW_ON_CUDA_ERROR(cub::DeviceScan::InclusiveSum(temp.data(), bytes, in, out, n, stream));
}

/** Reads one device value back (synchronizes the stream). */
template <typename T>
T ReadBack(cudaStream_t stream, const T *device_value) {
  T value{};
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&value, device_value, sizeof(T), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return value;
}

/** Fills `out[i] = i`. */
__global__ void IotaKernel(int n, int *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

void Iota(cudaStream_t stream, dvector<int> &out, int n) {
  out.resize(n);
  if (n == 0) return;
  IotaKernel<<<Blocks(n), kThreads, 0, stream>>>(n, out.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Symbolic phase
// ---------------------------------------------------------------------------

/** Error flags of the setup checks (SchurSystem::scratch.error[0]). */
enum SetupError : int { kTwoLandmarks = 1 };

/**
 * Per factor: its landmark (the single free slot in an eliminated batch),
 * the number of (landmark, retained slot) entries and of C^R items
 * (retained slot pairs: a < b for landmark factors, a <= b otherwise).
 * error[0] |= kTwoLandmarks for a factor with two free landmark slots;
 * error[1], error[2]: min and max tangent size of retained slots of landmark
 * factors.
 */
__global__ void ClassifyFactorsKernel(int num_factors, const BatchGeometry *batches,
                                      int num_batches, const int *slot_sizes,
                                      const int *factor_cols, const ColumnRange *ranges,
                                      int num_ranges, int *factor_landmark, int *factor_eslot,
                                      int *entry_count, int *rr_count, int *batch_has_landmarks,
                                      int *error) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_factors) return;
  const int batch_index = FindBatch(batches, num_batches, g);
  const BatchGeometry &batch = batches[batch_index];
  const int *cols = factor_cols + batch.col_offset + (g - batch.factor_begin) * batch.num_slots;
  int num_landmarks = 0, eslot = -1, landmark = -1, num_retained = 0;
  for (int s = 0; s < batch.num_slots; ++s) {
    const int col = cols[s];
    if (col < 0) continue;
    const ColumnRange &range = FindRange(ranges, num_ranges, col);
    if (range.eliminated) {
      ++num_landmarks;
      eslot = s;
      landmark = IdOf(range, col);
    } else {
      ++num_retained;
    }
  }
  if (num_landmarks > 1) atomicOr(error, kTwoLandmarks);
  const bool landmark_factor = num_landmarks == 1;
  factor_landmark[g] = landmark_factor ? landmark : -1;
  factor_eslot[g] = landmark_factor ? eslot : -1;
  entry_count[g] = landmark_factor ? num_retained : 0;
  rr_count[g] = landmark_factor ? num_retained * (num_retained - 1) / 2
                                : num_retained * (num_retained + 1) / 2;
  if (landmark_factor) {
    if (batch_has_landmarks[batch_index] == 0) atomicExch(batch_has_landmarks + batch_index, 1);
    for (int s = 0; s < batch.num_slots; ++s) {
      if (s == eslot || cols[s] < 0) continue;
      const int size = slot_sizes[batch.slot_begin + s];
      atomicMin(error + 1, size);
      atomicMax(error + 2, size);
    }
  }
}

/**
 * Writes each factor's entries (key: landmark << block_bits | block; value:
 * factor, slot) and C^R items (factor, packed slots, key of the S block).
 */
__global__ void EmitItemsKernel(int num_factors, const BatchGeometry *batches, int num_batches,
                                const int *factor_cols, const ColumnRange *ranges, int num_ranges,
                                const int *factor_landmark, const int *factor_eslot,
                                const int *entry_offsets, const int *rr_offsets, int block_bits,
                                unsigned long long *entry_keys, int2 *entry_values, int *rr_factor,
                                int *rr_slots, unsigned long long *rr_keys) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_factors) return;
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, g)];
  const int *cols = factor_cols + batch.col_offset + (g - batch.factor_begin) * batch.num_slots;
  const int landmark = factor_landmark[g];
  const int eslot = factor_eslot[g];
  int entry = entry_offsets[g];
  int rr = rr_offsets[g];
  for (int a = 0; a < batch.num_slots; ++a) {
    const int col_a = cols[a];
    if (col_a < 0 || a == eslot) continue;
    const int block_a = IdOf(FindRange(ranges, num_ranges, col_a), col_a);
    if (landmark >= 0) {
      entry_keys[entry] = (static_cast<unsigned long long>(landmark) << block_bits) |
                          static_cast<unsigned>(block_a);
      entry_values[entry] = make_int2(g, a);
      ++entry;
    }
    for (int b = landmark >= 0 ? a + 1 : a; b < batch.num_slots; ++b) {
      const int col_b = cols[b];
      if (col_b < 0 || b == eslot) continue;
      const int block_b = IdOf(FindRange(ranges, num_ranges, col_b), col_b);
      const int lo = min(block_a, block_b), hi = max(block_a, block_b);
      rr_factor[rr] = g;
      rr_slots[rr] = a | (b << 8) | ((block_a > block_b) << 16) | ((block_a == block_b) << 17);
      rr_keys[rr] = (static_cast<unsigned long long>(lo) << block_bits) | static_cast<unsigned>(hi);
      ++rr;
    }
  }
}

/** Unpacks the sorted entries and flags the first entry of every pair. */
__global__ void UnpackEntriesKernel(int num_entries, const unsigned long long *keys,
                                    const int2 *values, int *entry_factor, int *entry_slot,
                                    int *head) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_entries) return;
  entry_factor[e] = values[e].x;
  entry_slot[e] = values[e].y;
  head[e] = (e == 0 || keys[e] != keys[e - 1]) ? 1 : 0;
}

/** Writes the pairs from the heads of the sorted entries. */
__global__ void BuildPairsKernel(int num_entries, const unsigned long long *keys, const int *head,
                                 const int *pair_id_plus_one, int block_bits, int *pair_landmark,
                                 int *pair_block, int *pair_entry_offsets) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_entries) return;
  if (head[e]) {
    const int pair = pair_id_plus_one[e] - 1;
    pair_landmark[pair] = static_cast<int>(keys[e] >> block_bits);
    pair_block[pair] = static_cast<int>(keys[e] & ((1ull << block_bits) - 1));
    pair_entry_offsets[pair] = e;
  }
  if (e == num_entries - 1) pair_entry_offsets[pair_id_plus_one[e]] = num_entries;
}

/** landmark_pair_offsets from the landmark-sorted pairs (landmarks without pairs get empty ranges).
 */
__global__ void LandmarkOffsetsKernel(int num_pairs, int num_landmarks, const int *pair_landmark,
                                      int *landmark_pair_offsets) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k > num_pairs) return;
  const int prev = k == 0 ? -1 : pair_landmark[k - 1];
  const int cur = k == num_pairs ? num_landmarks : pair_landmark[k];
  for (int l = prev + 1; l <= cur; ++l) landmark_pair_offsets[l] = k;
}

/** Scatters the inverse permutation. */
__global__ void InvertPermutationKernel(int n, const int *perm, int *inverse) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) inverse[perm[i]] = i;
}

/** Off-diagonal clique products k (k - 1) / 2 of each landmark (index L: 0). */
__global__ void CliqueCountsKernel(int num_landmarks, const int *landmark_pair_offsets,
                                   long long *counts) {
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l > num_landmarks) return;
  long long k = l < num_landmarks ? landmark_pair_offsets[l + 1] - landmark_pair_offsets[l] : 0;
  counts[l] = k * (k - 1) / 2;
}

/** Keys of the S blocks of every clique product (a < b) of the pair's landmark, pair a. */
__global__ void CliqueKeysKernel(int num_pairs, const int *pair_landmark, const int *pair_block,
                                 const int *landmark_pair_offsets, const long long *clique_offsets,
                                 int block_bits, unsigned long long *keys, int *values) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= num_pairs) return;
  const int l = pair_landmark[pair];
  const int begin = landmark_pair_offsets[l];
  const long long k = landmark_pair_offsets[l + 1] - begin;
  const long long a = pair - begin;
  const unsigned long long p = static_cast<unsigned>(pair_block[pair]);
  long long index = clique_offsets[l] + a * k - a * (a + 1) / 2;
  for (long long b = a + 1; b < k; ++b, ++index) {
    keys[index] = (p << block_bits) | static_cast<unsigned>(pair_block[begin + b]);
    values[index] = static_cast<int>(index);
  }
}

/** Keys of the diagonal blocks and of the C^R items, after the clique keys. */
__global__ void OtherKeysKernel(int num_blocks, int num_rr, long long num_cliques,
                                const unsigned long long *rr_keys, int block_bits,
                                unsigned long long *keys, int *values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks + num_rr) return;
  const long long index = num_cliques + i;
  keys[index] = i < num_blocks
                    ? (static_cast<unsigned long long>(i) << block_bits) | static_cast<unsigned>(i)
                    : rr_keys[i - num_blocks];
  values[index] = static_cast<int>(index);
}

/** Head flags of the sorted keys (first of each distinct S block). */
__global__ void HeadFlagsKernel(long long n, const unsigned long long *keys, int *head) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  head[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1 : 0;
}

/**
 * Writes each distinct S block (row, col) and routes the block id of every
 * diagonal and C^R key back to its source.
 */
__global__ void AssignBlocksKernel(long long n, const unsigned long long *keys, const int *values,
                                   const int *head, const int *block_id_plus_one, int block_bits,
                                   long long num_cliques, int num_blocks, int *s_row, int *s_col,
                                   int *diag_block, int *rr_block) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const int block = block_id_plus_one[i] - 1;
  if (head[i]) {
    s_row[block] = static_cast<int>(keys[i] >> block_bits);
    s_col[block] = static_cast<int>(keys[i] & ((1ull << block_bits) - 1));
  }
  const long long v = values[i];
  if (v < num_cliques) {
    // Clique products find their block by search at run time.
  } else if (v < num_cliques + num_blocks) {
    diag_block[v - num_cliques] = block;
  } else {
    rr_block[v - num_cliques - num_blocks] = block;
  }
}

/** Value count of each S block, and the per-row upper and lower block counts. */
__global__ void BlockSizesKernel(int num_s_blocks, const int *s_row, const int *s_col,
                                 const int *block_dim, int *value_counts, int *upper_count,
                                 int *lower_count) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= num_s_blocks) return;
  const int p = s_row[j], q = s_col[j];
  value_counts[j] = block_dim[p] * block_dim[q];
  atomicAdd(upper_count + p, 1);
  if (p != q) atomicAdd(lower_count + q, 1);
}

/** Keys (col << bits | row) of the strictly upper blocks, to order the mirrored lower blocks. */
__global__ void LowerKeysKernel(int num_s_blocks, const int *s_row, const int *s_col,
                                int block_bits, unsigned long long *keys, int *values,
                                int *num_lower) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= num_s_blocks) return;
  const int p = s_row[j], q = s_col[j];
  if (p == q) return;
  const int slot = atomicAdd(num_lower, 1);
  keys[slot] = (static_cast<unsigned long long>(q) << block_bits) | static_cast<unsigned>(p);
  values[slot] = j;
}

/** Full block row offsets: the upper and the mirrored lower blocks of each row. */
__global__ void FullRowOffsetsKernel(int num_blocks, const int *upper_row_offsets,
                                     const int *lower_row_offsets, int *full_row_offsets) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p <= num_blocks) full_row_offsets[p] = upper_row_offsets[p] + lower_row_offsets[p];
}

/**
 * Places the upper blocks and the mirrored lower blocks (sorted by row, then
 * column) into the full block rows, lower part first, so columns ascend:
 * upper block i of row p lands at i + lower_row_offsets[p + 1], lower block t
 * of row q at t + upper_row_offsets[q].
 */
__global__ void PlaceFullBlocksKernel(int num_s_blocks, int num_lower, const int *s_row,
                                      const int *s_col, const unsigned long long *lower_keys,
                                      const int *lower_values, const int *upper_row_offsets,
                                      const int *lower_row_offsets, int block_bits, int *full_row,
                                      int *full_col, int *full_src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_s_blocks) {
    const int p = s_row[i];
    const int pos = i + lower_row_offsets[p + 1];
    full_row[pos] = p;
    full_col[pos] = s_col[i];
    full_src[pos] = i * 2;
  } else if (i < num_s_blocks + num_lower) {
    const int t = i - num_s_blocks;
    const int q = static_cast<int>(lower_keys[t] >> block_bits);
    const int p = static_cast<int>(lower_keys[t] & ((1ull << block_bits) - 1));
    const int pos = t + upper_row_offsets[q];
    full_row[pos] = q;
    full_col[pos] = p;
    full_src[pos] = lower_values[t] * 2 + 1;
  }
}

/** Width (columns) of each full block. */
__global__ void FullWidthsKernel(int num_full, const int *full_col, const int *block_dim,
                                 int *widths) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_full) widths[i] = block_dim[full_col[i]];
}

/** Scalar row length of each block row, and its nonzero count (rows x length). */
__global__ void RowLengthsKernel(int num_blocks, const int *full_row_offsets, const int *width_pos,
                                 const int *block_dim, int *row_length, int *row_nonzeros) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_blocks) return;
  const int length = width_pos[full_row_offsets[p + 1]] - width_pos[full_row_offsets[p]];
  row_length[p] = length;
  row_nonzeros[p] = length * block_dim[p];
}

/** CSR row offsets of S: rows of block p start at row_base[p] + r * row_length[p]. */
__global__ void CsrRowOffsetsKernel(int num_blocks, const int *block_dim, const int *block_col,
                                    const int *row_base, const int *row_length, int num_nonzeros,
                                    int num_cols, int *row_offsets) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_blocks) return;
  for (int r = 0; r < block_dim[p]; ++r) {
    row_offsets[block_col[p] + r] = row_base[p] + r * row_length[p];
  }
  if (p == num_blocks - 1) row_offsets[num_cols] = num_nonzeros;
}

/** Column ids of S and the position in the upper block storage each nonzero copies. */
__global__ void CsrFillKernel(int num_full, const int *full_row, const int *full_col,
                              const int *full_src, const int *full_row_offsets,
                              const int *width_pos, const int *block_dim, const int *block_col,
                              const int *row_base, const int *row_length,
                              const int *s_value_offsets, int *col_ids, int *csr_src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_full) return;
  const int p = full_row[i], q = full_col[i];
  const int dp = block_dim[p], dq = block_dim[q];
  const int col_in_row = width_pos[i] - width_pos[full_row_offsets[p]];
  const int src_block = full_src[i] >> 1;
  const bool mirrored = full_src[i] & 1;
  const int value0 = s_value_offsets[src_block];
  for (int r = 0; r < dp; ++r) {
    const int row_start = row_base[p] + r * row_length[p] + col_in_row;
    for (int c = 0; c < dq; ++c) {
      col_ids[row_start + c] = block_col[q] + c;
      int src;
      if (p == q) {
        src = value0 + min(r, c) * dp + max(r, c);  // diagonal blocks: upper triangle only
      } else if (mirrored) {
        src = value0 + c * dp + r;  // stored as block (q, p), dq x dp
      } else {
        src = value0 + r * dq + c;
      }
      csr_src[row_start + c] = src;
    }
  }
}

/** Retained blocks of one state batch: dims, columns of S and of the full system. */
__global__ void RetainedBlocksKernel(int num_states, int tangent, int first_block,
                                     int first_natural_col, int first_reduced_col, int *block_dim,
                                     int *block_col, int *block_natural_col) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_states) return;
  block_dim[first_block + i] = tangent;
  block_col[first_block + i] = first_reduced_col + i * tangent;
  block_natural_col[first_block + i] = first_natural_col + i * tangent;
}

/** Landmarks of one eliminated state batch: their first free column. */
__global__ void LandmarkColumnsKernel(int num_states, int tangent, int first_landmark,
                                      int first_natural_col, int *landmark_natural_col) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_states) landmark_natural_col[first_landmark + i] = first_natural_col + i * tangent;
}

/** Per column of S: its free column in the full system and its diagonal in the S values. */
__global__ void ColumnMapsKernel(int num_blocks, const int *block_dim, const int *block_col,
                                 const int *block_natural_col, const int *diag_block,
                                 const int *s_value_offsets, int *col_natural,
                                 int *col_diag_value) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_blocks) return;
  const int d = block_dim[p];
  const int value0 = s_value_offsets[diag_block[p]];
  for (int r = 0; r < d; ++r) {
    col_natural[block_col[p] + r] = block_natural_col[p] + r;
    col_diag_value[block_col[p] + r] = value0 + r * d + r;
  }
}

/** Per pose-major position: the pair's entry count. */
__global__ void PoseMajorPairsKernel(int num_pairs, const int *pm_pair,
                                     const int *pair_entry_offsets, int *pm_entry_count) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_pairs) return;
  const int pair = pm_pair[t];
  pm_entry_count[t] = pair_entry_offsets[pair + 1] - pair_entry_offsets[pair];
}

/**
 * Copies the entries of each pair to pose-major order, and gives each
 * landmark factor the first pose-major position of its pairs (its key in the
 * evaluation order).
 */
__global__ void PoseMajorEntriesKernel(int num_pairs, const int *pm_pair,
                                       const int *pair_entry_offsets, const int *pm_entry_offsets,
                                       const int *entry_factor, const int *entry_slot,
                                       int *pm_entry_factor, int *pm_entry_slot, int *factor_key) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_pairs) return;
  const int pair = pm_pair[t];
  int out = pm_entry_offsets[t];
  for (int e = pair_entry_offsets[pair]; e < pair_entry_offsets[pair + 1]; ++e, ++out) {
    pm_entry_factor[out] = entry_factor[e];
    pm_entry_slot[out] = entry_slot[e];
    atomicMin(factor_key + entry_factor[e], t);
  }
}

/**
 * Sort keys of the evaluation order: batch-major, then the factor's key
 * (landmark factors by their first pose-major pair, the others after them),
 * or the natural order for a batch that is not reordered.
 */
__global__ void OrderKeysKernel(int num_factors, const BatchGeometry *batches, int num_batches,
                                const int *reordered, const int *factor_key,
                                unsigned long long *keys, int *values) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_factors) return;
  const int b = FindBatch(batches, num_batches, g);
  const unsigned key = reordered[b] ? static_cast<unsigned>(factor_key[g])
                                    : static_cast<unsigned>(g - batches[b].factor_begin);
  keys[g] = (static_cast<unsigned long long>(b) << 32) | key;
  values[g] = g;
}

/**
 * The evaluation order (batch-local factor of each item), each factor's item,
 * and the slot columns of each item (FactorCols() in item order).
 */
__global__ void ItemsKernel(int num_factors, const BatchGeometry *batches, int num_batches,
                            const int *sorted_factors, const int *factor_cols, int *factor_item,
                            int *order, int *item_cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_factors) return;
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, i)];
  const int g = sorted_factors[i];
  const int item = i - batch.factor_begin, local = g - batch.factor_begin;
  factor_item[g] = item;
  order[i] = local;
  for (int s = 0; s < batch.num_slots; ++s) {
    item_cols[batch.col_offset + item * batch.num_slots + s] =
        factor_cols[batch.col_offset + local * batch.num_slots + s];
  }
}

/**
 * Where the pose-major entries' Jacobians and residuals are: the factor's
 * item start, and packed (retained slot column, landmark slot column, m, n).
 */
__global__ void EntryLocationsKernel(int num_entries, const int *pm_entry_factor,
                                     const int *pm_entry_slot, const BatchGeometry *batches,
                                     int num_batches, const int *slot_offsets,
                                     const int *factor_eslot, const int *factor_item,
                                     long long *jacobian_base, long long *residual_base,
                                     int *meta) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_entries) return;
  const int g = pm_entry_factor[e];
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, g)];
  const long long item = factor_item[g];
  const int m = batch.num_residuals, n = batch.tangent_dim;
  jacobian_base[e] = batch.jacobian_offset + item * m * n;
  residual_base[e] = batch.residual_offset + item * m;
  meta[e] = slot_offsets[batch.slot_begin + pm_entry_slot[e]] |
            (slot_offsets[batch.slot_begin + factor_eslot[g]] << 8) | (m << 16) | (n << 24);
}

// ---------------------------------------------------------------------------
// Numeric phase
// ---------------------------------------------------------------------------

/** Column scale, mask and damping (null: identity / none) of the free columns. */
struct Modifiers {
  const float *scale;
  const float *mask;
  const float *damping;
  __device__ __forceinline__ float Scale(int col) const { return scale ? scale[col] : 1.f; }
  __device__ __forceinline__ float Mask(int col) const { return mask ? mask[col] : 1.f; }
  __device__ __forceinline__ float Damping(int col) const { return damping ? damping[col] : 0.f; }
};

/**
 * Segments of equal keys over the lanes of a warp (consecutive lanes with the
 * same key; give inactive lanes distinct negative keys). All 32 lanes must
 * call it and Sum() together.
 */
struct WarpSegments {
  int start;  ///< First lane of this lane's segment.
  bool tail;  ///< This lane is the last of its segment (holds the sum).

  __device__ __forceinline__ explicit WarpSegments(int key) {
    const int lane = threadIdx.x & 31;
    const int prev = __shfl_up_sync(kFullMask, key, 1);
    const int next = __shfl_down_sync(kFullMask, key, 1);
    const unsigned heads = __ballot_sync(kFullMask, lane == 0 || prev != key);
    start = 31 - __clz(heads & (lane == 31 ? kFullMask : ((2u << lane) - 1u)));
    tail = lane == 31 || next != key;
  }

  /** Inclusive segmented sum: the tail lane gets the sum of its segment. */
  __device__ __forceinline__ float Sum(float v) const {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float u = __shfl_up_sync(kFullMask, v, d);
      if (lane - d >= start) v += u;
    }
    return v;
  }
};

/**
 * One thread per item (evaluation order), for the terms without a retained-state pair: the
 * landmark terms of a landmark factor (b_e, diag, D_i = J_eᵀJ_e, AoS upper
 * triangle; few factors per landmark, so plain atomics), and every term of a
 * factor between retained states only. The retained terms of landmark
 * factors are AssemblePairsKernel's.
 */
__global__ void AssembleFactorsKernel(int num_factors, const BatchGeometry *batches,
                                      int num_batches, const int *slot_sizes,
                                      const int *slot_offsets, const int *item_cols,
                                      const int *factor_landmark, const int *factor_eslot,
                                      const int *evaluation_order, int landmark_dim,
                                      const float *__restrict__ jacobians,
                                      const float *__restrict__ residuals, float *rhs,
                                      float *diagonal, float *landmark_hessian) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // item, so J and r stream
  if (i >= num_factors) return;
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, i)];
  const long long item = i - batch.factor_begin;
  const int local = evaluation_order[i];
  const int g = batch.factor_begin + local;
  const int m = batch.num_residuals, n = batch.tangent_dim;
  const float *J = jacobians + batch.jacobian_offset + item * m * n;
  const float *r = residuals + batch.residual_offset + item * m;
  const int *cols = item_cols + batch.col_offset + item * batch.num_slots;
  const int landmark = factor_landmark[g];
  for (int s = 0; s < batch.num_slots; ++s) {
    const int col = cols[s];
    if (col < 0 || (landmark >= 0 && s != factor_eslot[g])) continue;
    const int offset = slot_offsets[batch.slot_begin + s];
    for (int c = 0; c < slot_sizes[batch.slot_begin + s]; ++c) {
      float gradient = 0.f, norm = 0.f;
      for (int row = 0; row < m; ++row) {
        const float j = J[row * n + offset + c];
        gradient += j * r[row];
        norm += j * j;
      }
      atomicAdd(rhs + col + c, -gradient);
      atomicAdd(diagonal + col + c, norm);
    }
  }
  if (landmark < 0) return;
  const int offset = slot_offsets[batch.slot_begin + factor_eslot[g]];
  const int tri = landmark_dim * (landmark_dim + 1) / 2;
  for (int j = 0; j < landmark_dim; ++j) {
    for (int k = j; k < landmark_dim; ++k) {
      float sum = 0.f;
      for (int row = 0; row < m; ++row) sum += J[row * n + offset + j] * J[row * n + offset + k];
      atomicAdd(landmark_hessian + landmark * tri + Tri(j, k, landmark_dim), sum);
    }
  }
}

/**
 * One thread per pair, pose-major: the retained terms of the pair's landmark
 * factors (b_p = -Σ J_pᵀ r, diag J_pᵀJ_p), summed over each retained state's
 * run of pairs with WarpSegments, one atomic per state and warp.
 */
template <int DP>
__global__ void __launch_bounds__(kThreads)
    AssemblePairsKernel(int num_pairs, const int *__restrict__ pm_entry_offsets,
                        const long long *__restrict__ entry_jacobian,
                        const long long *__restrict__ entry_residual,
                        const int *__restrict__ entry_meta, const int *__restrict__ pm_pair,
                        const int *__restrict__ pair_block, const float *__restrict__ jacobians,
                        const float *__restrict__ residuals,
                        const int *__restrict__ block_natural_col, float *rhs, float *diagonal) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  float gradient[DP], norm[DP];
#pragma unroll
  for (int c = 0; c < DP; ++c) gradient[c] = norm[c] = 0.f;
  int p = -1 - static_cast<int>(threadIdx.x & 31);
  if (t < num_pairs) {
    p = pair_block[pm_pair[t]];
    for (int e = pm_entry_offsets[t]; e < pm_entry_offsets[t + 1]; ++e) {
      const int meta = entry_meta[e];
      const int m = (meta >> 16) & 0xff, n = meta >> 24;
      const float *Jp = jacobians + entry_jacobian[e] + (meta & 0xff);
      const float *r = residuals + entry_residual[e];
      for (int row = 0; row < m; ++row) {
        const float rr = r[row];
#pragma unroll
        for (int c = 0; c < DP; ++c) {
          const float j = Jp[row * n + c];
          gradient[c] += j * rr;
          norm[c] += j * j;
        }
      }
    }
  }
  const WarpSegments segments(p);
#pragma unroll
  for (int c = 0; c < DP; ++c) {
    gradient[c] = segments.Sum(gradient[c]);
    norm[c] = segments.Sum(norm[c]);
  }
  if (segments.tail && p >= 0) {
    const int col = block_natural_col[p];
#pragma unroll
    for (int c = 0; c < DP; ++c) {
      atomicAdd(rhs + col + c, -gradient[c]);
      atomicAdd(diagonal + col + c, norm[c]);
    }
  }
}

/** g = the retained part of the (scaled, masked) right-hand side. */
__global__ void GatherRhsKernel(int num_cols, const int *col_natural, const float *rhs,
                                float *reduced_rhs) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < num_cols) reduced_rhs[c] = rhs[col_natural[c]];
}

/**
 * One thread per C^R item: J_aᵀ J_b of its factor, scaled and masked, into
 * its block of S (transposed if the block is stored the other way; for two
 * slots on one state, as the symmetric sum; diagonal blocks keep only their
 * upper triangle).
 */
__global__ void RetainedProductsKernel(int num_rr, const int *rr_factor, const int *rr_slots,
                                       const int *rr_block, const BatchGeometry *batches,
                                       int num_batches, const int *slot_sizes,
                                       const int *slot_offsets, const int *factor_cols,
                                       const int *factor_item, const float *__restrict__ jacobians,
                                       const int *s_value_offsets, Modifiers mod, float *s_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rr) return;
  const int g = rr_factor[i];
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, g)];
  const int local = g - batch.factor_begin;
  const int m = batch.num_residuals, n = batch.tangent_dim;
  const float *J =
      jacobians + batch.jacobian_offset + static_cast<long long>(factor_item[g]) * m * n;
  const int packed = rr_slots[i];
  const int a = packed & 0xff, b = (packed >> 8) & 0xff;
  const bool transposed = (packed >> 16) & 1, same_state = (packed >> 17) & 1;
  const int da = slot_sizes[batch.slot_begin + a], db = slot_sizes[batch.slot_begin + b];
  const int oa = slot_offsets[batch.slot_begin + a], ob = slot_offsets[batch.slot_begin + b];
  const int *cols = factor_cols + batch.col_offset + local * batch.num_slots;
  const int col_a = cols[a], col_b = cols[b];
  float *block = s_values + s_value_offsets[rr_block[i]];
  for (int r = 0; r < da; ++r) {
    const float wr = mod.Scale(col_a + r) * mod.Mask(col_a + r);
    for (int c = 0; c < db; ++c) {
      if (a == b && c < r) continue;
      float sum = 0.f;
      for (int row = 0; row < m; ++row) sum += J[row * n + oa + r] * J[row * n + ob + c];
      const float v = sum * wr * mod.Scale(col_b + c) * mod.Mask(col_b + c);
      if (a == b) {
        atomicAdd(block + r * da + c, v);
      } else if (same_state) {
        atomicAdd(block + min(r, c) * da + max(r, c), r == c ? 2.f * v : v);
      } else if (transposed) {
        atomicAdd(block + c * da + r, v);
      } else {
        atomicAdd(block + r * db + c, v);
      }
    }
  }
}

/**
 * One stored row of V (DE floats), padded for one vector load or store: a
 * landmark of size 3 takes a float4.
 */
template <int DE>
struct VRow;
template <>
struct VRow<1> {
  static constexpr int kFloats = 1;
  __device__ static void Load(const float *p, float (&v)[1]) { v[0] = *p; }
  __device__ static void Store(float *p, const float (&v)[1]) { *p = v[0]; }
};
template <>
struct VRow<2> {
  static constexpr int kFloats = 2;
  __device__ static void Load(const float *p, float (&v)[2]) {
    const float2 x = *reinterpret_cast<const float2 *>(p);
    v[0] = x.x;
    v[1] = x.y;
  }
  __device__ static void Store(float *p, const float (&v)[2]) {
    *reinterpret_cast<float2 *>(p) = make_float2(v[0], v[1]);
  }
};
template <>
struct VRow<3> {
  static constexpr int kFloats = 4;
  __device__ static void Load(const float *p, float (&v)[3]) {
    const float4 x = *reinterpret_cast<const float4 *>(p);
    v[0] = x.x;
    v[1] = x.y;
    v[2] = x.z;
  }
  __device__ static void Store(float *p, const float (&v)[3]) {
    *reinterpret_cast<float4 *>(p) = make_float4(v[0], v[1], v[2], 0.f);
  }
};

/** In-place Cholesky of the DE x DE matrix A (lower triangle used); false if not positive. */
template <int DE>
__device__ __forceinline__ bool Cholesky(float (&A)[DE][DE]) {
  bool ok = true;
#pragma unroll
  for (int j = 0; j < DE; ++j) {
    float d = A[j][j];
#pragma unroll
    for (int k = 0; k < j; ++k) d -= A[j][k] * A[j][k];
    ok = ok && d > 0.f;
    const float l = sqrtf(fmaxf(d, 1e-30f));
    A[j][j] = l;
    const float inv = 1.f / l;
#pragma unroll
    for (int i = j + 1; i < DE; ++i) {
      float v = A[i][j];
#pragma unroll
      for (int k = 0; k < j; ++k) v -= A[i][k] * A[j][k];
      A[i][j] = v * inv;
    }
  }
  return ok;
}

/** Solves L x = b in place (L lower, from Cholesky). */
template <int DE>
__device__ __forceinline__ void ForwardSolve(const float (&L)[DE][DE], float (&x)[DE]) {
#pragma unroll
  for (int j = 0; j < DE; ++j) {
    float v = x[j];
#pragma unroll
    for (int k = 0; k < j; ++k) v -= L[j][k] * x[k];
    x[j] = v / L[j][j];
  }
}

/** Solves Lᵀ x = b in place. */
template <int DE>
__device__ __forceinline__ void BackwardSolve(const float (&L)[DE][DE], float (&x)[DE]) {
#pragma unroll
  for (int j = DE - 1; j >= 0; --j) {
    float v = x[j];
#pragma unroll
    for (int k = j + 1; k < DE; ++k) v -= L[k][j] * x[k];
    x[j] = v / L[j][j];
  }
}

/**
 * The working landmark block of landmark l: s_e D_i s_e, masked (unit
 * diagonal), plus the damping, factored in place. Its free columns start at
 * `col`. A block that is not positive definite (a landmark seen along one
 * ray, or with a zero Jacobian, e.g. behind every camera, which no damping of
 * the form λ diag(D_i) fixes) is factored again with a small diagonal shift,
 * relative to its largest diagonal entry: in the spirit of the pseudo-inverse
 * Ceres uses for rank-deficient blocks, the unobservable directions then get
 * (almost) no step instead of failing the solve. Returns false only if that
 * fails too (non-finite values).
 */
template <int DE>
__device__ __forceinline__ bool LandmarkFactor(const float *__restrict__ landmark_hessian,
                                               int num_landmarks, int l, int col,
                                               const Modifiers &mod, float (&L)[DE][DE]) {
  constexpr float kRelativeShift = 1e-4f;
  constexpr float kMinShift = 1e-20f;
  float D[DE][DE], w[DE];
#pragma unroll
  for (int j = 0; j < DE; ++j) w[j] = mod.Scale(col + j) * mod.Mask(col + j);
  float largest = 0.f;
#pragma unroll
  for (int j = 0; j < DE; ++j) {
#pragma unroll
    for (int k = j; k < DE; ++k) {
      const float v = landmark_hessian[l * (DE * (DE + 1) / 2) + Tri(j, k, DE)] * w[j] * w[k];
      D[j][k] = v;
      D[k][j] = v;
    }
    D[j][j] = mod.Mask(col + j) == 0.f ? 1.f : D[j][j] + mod.Damping(col + j);
    largest = fmaxf(largest, D[j][j]);
  }
#pragma unroll
  for (int j = 0; j < DE; ++j) {
#pragma unroll
    for (int k = 0; k < DE; ++k) L[j][k] = D[j][k];
  }
  if (Cholesky<DE>(L)) return true;
  const float shift = fmaxf(kRelativeShift * largest, kMinShift);
#pragma unroll
  for (int j = 0; j < DE; ++j) {
#pragma unroll
    for (int k = 0; k < DE; ++k) L[j][k] = D[j][k] + (j == k ? shift : 0.f);
  }
  return Cholesky<DE>(L);
}

/**
 * One thread per pair, in pose-major order: contracts the pair's factors into
 * W = J_pᵀJ_e and A = J_pᵀJ_p (scaled, masked), eliminates the landmark
 * (V = W L⁻ᵀ, h = L⁻¹ b_e), stores V, and adds the diagonal bracket
 * A - V Vᵀ and -V h of all pairs of one retained state with a
 * warp-segmented sum, one atomic per state and warp.
 */
template <int DE, int DP>
__global__ void __launch_bounds__(kThreads)
    EliminatePairsKernel(int num_pairs, const int *__restrict__ pm_entry_offsets,
                         const long long *__restrict__ entry_jacobian,
                         const int *__restrict__ entry_meta, const int *__restrict__ pm_pair,
                         const int *__restrict__ pair_landmark, const int *__restrict__ pair_block,
                         const float *__restrict__ jacobians,
                         const float *__restrict__ landmark_hessian, int num_landmarks,
                         const int *__restrict__ landmark_natural_col,
                         const int *__restrict__ block_natural_col,
                         const int *__restrict__ block_col, const int *__restrict__ diag_block,
                         const int *__restrict__ s_value_offsets, const float *__restrict__ rhs,
                         Modifiers mod, float *__restrict__ pair_v, float *s_values,
                         float *reduced_rhs, int *failure) {
  constexpr int kTri = DP * (DP + 1) / 2;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  float bracket[kTri];
  float gp[DP];
#pragma unroll
  for (int i = 0; i < kTri; ++i) bracket[i] = 0.f;
#pragma unroll
  for (int i = 0; i < DP; ++i) gp[i] = 0.f;
  int p = -1 - lane;  // inactive lanes: distinct, negative keys (WarpSegments)
  if (t < num_pairs) {
    const int pair = pm_pair[t];
    p = pair_block[pair];
    const int l = pair_landmark[pair];
    float W[DP][DE];
#pragma unroll
    for (int r = 0; r < DP; ++r) {
#pragma unroll
      for (int j = 0; j < DE; ++j) W[r][j] = 0.f;
    }
    for (int e = pm_entry_offsets[t]; e < pm_entry_offsets[t + 1]; ++e) {
      const int meta = entry_meta[e];
      const int m = (meta >> 16) & 0xff, n = meta >> 24;
      const float *Jp = jacobians + entry_jacobian[e] + (meta & 0xff);
      const float *Je = jacobians + entry_jacobian[e] + ((meta >> 8) & 0xff);
      for (int row = 0; row < m; ++row) {
        float jp[DP], je[DE];
#pragma unroll
        for (int c = 0; c < DP; ++c) jp[c] = Jp[row * n + c];
#pragma unroll
        for (int j = 0; j < DE; ++j) je[j] = Je[row * n + j];
#pragma unroll
        for (int r = 0; r < DP; ++r) {
#pragma unroll
          for (int j = 0; j < DE; ++j) W[r][j] += jp[r] * je[j];
#pragma unroll
          for (int c = r; c < DP; ++c) bracket[Tri(r, c, DP)] += jp[r] * jp[c];
        }
      }
    }
    const int pcol = block_natural_col[p];
    const int ecol = landmark_natural_col[l];
    float wp[DP], we[DE];
#pragma unroll
    for (int r = 0; r < DP; ++r) wp[r] = mod.Scale(pcol + r) * mod.Mask(pcol + r);
#pragma unroll
    for (int j = 0; j < DE; ++j) we[j] = mod.Scale(ecol + j) * mod.Mask(ecol + j);
    float L[DE][DE];
    const bool factored = LandmarkFactor<DE>(landmark_hessian, num_landmarks, l, ecol, mod, L);
    float h[DE];
#pragma unroll
    for (int j = 0; j < DE; ++j) h[j] = rhs[ecol + j];
    ForwardSolve<DE>(L, h);
    // A landmark block that cannot be factored even with the shift of
    // LandmarkFactor (non-finite values) poisons g, so the whole step is NaN
    // and rejected.
    if (!factored) {
      h[0] = CUDART_NAN_F;
      *failure = 1;
    }
    // V = W L⁻ᵀ, row by row (L v = w for each row w of W).
    float V[DP][DE];
#pragma unroll
    for (int r = 0; r < DP; ++r) {
#pragma unroll
      for (int j = 0; j < DE; ++j) V[r][j] = W[r][j] * wp[r] * we[j];
      ForwardSolve<DE>(L, V[r]);
      VRow<DE>::Store(pair_v + (static_cast<long long>(t) * DP + r) * VRow<DE>::kFloats, V[r]);
    }
#pragma unroll
    for (int r = 0; r < DP; ++r) {
#pragma unroll
      for (int c = r; c < DP; ++c) {
        float vv = 0.f;
#pragma unroll
        for (int j = 0; j < DE; ++j) vv += V[r][j] * V[c][j];
        bracket[Tri(r, c, DP)] = bracket[Tri(r, c, DP)] * wp[r] * wp[c] - vv;
      }
      float vh = 0.f;
#pragma unroll
      for (int j = 0; j < DE; ++j) vh += V[r][j] * h[j];
      gp[r] = -vh;
    }
  }
  // Sum over runs of equal p (pose-major: a run is one retained state).
  const WarpSegments segments(p);
#pragma unroll
  for (int i = 0; i < kTri; ++i) bracket[i] = segments.Sum(bracket[i]);
#pragma unroll
  for (int i = 0; i < DP; ++i) gp[i] = segments.Sum(gp[i]);
  if (segments.tail && p >= 0) {
    float *block = s_values + s_value_offsets[diag_block[p]];
#pragma unroll
    for (int r = 0; r < DP; ++r) {
#pragma unroll
      for (int c = r; c < DP; ++c) atomicAdd(block + r * DP + c, bracket[Tri(r, c, DP)]);
      atomicAdd(reduced_rhs + block_col[p] + r, gp[r]);
    }
  }
}

/**
 * The off-diagonal blocks of S, by block row: one CTA per chunk of the
 * pose-major pairs (p, i) of retained state p, one warp per pair at a time.
 * The warps of a CTA share row p's blocks and V_pi's landmarks in cache.
 * For each later pair (q, i) of landmark i (q > p: the landmark's pairs are
 * sorted by state), -V_pi V_qiᵀ is added to block (p, q). The products are
 * taken 32 at a time: each lane finds one product's block (binary search in
 * the row) and V_qi's position; then the lanes sweep the products' entries,
 * so each warp-wide atomic (a native global reduction) covers contiguous
 * values of one block. An entry costs two vector loads (rows of V_pi in shared
 * memory and of V_qi) and one reduction. Chunks bound the work per CTA however
 * long the tracks are.
 */
template <int DE, int DP>
__global__ void __launch_bounds__(kThreads)
    OffDiagonalRowsKernel(const int3 *__restrict__ chunks,
                          const int *__restrict__ upper_row_offsets, const int *__restrict__ s_col,
                          const int *__restrict__ s_value_offsets, const int *__restrict__ pm_pair,
                          const int *__restrict__ pair_landmark, const int *__restrict__ pair_block,
                          const int *__restrict__ pair_pm,
                          const int *__restrict__ landmark_pair_offsets,
                          const float *__restrict__ pair_v, float *s_values) {
  using Row = VRow<DE>;
  constexpr int kVFloats = DP * Row::kFloats;
  constexpr int kBlock = DP * DP;
  constexpr int kWarps = kThreads / 32;
  // Per product: first value of its block of S, pose-major position of (q, i).
  __shared__ int2 product[kWarps][32];
  __shared__ __align__(16) float vp[kWarps][kVFloats];  // V_pi
  const int3 chunk = chunks[blockIdx.x];  // retained state, pose-major pairs [begin, end)
  const int p = chunk.x;
  const int first_block = upper_row_offsets[p], end_block = upper_row_offsets[p + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int t = chunk.y + warp; t < chunk.z; t += kWarps) {
    const int pair = pm_pair[t];
    const int later = landmark_pair_offsets[pair_landmark[pair] + 1] - pair - 1;
    if (later <= 0) continue;
    __syncwarp();
    for (int k = lane; k < kVFloats; k += 32)
      vp[warp][k] = pair_v[static_cast<long long>(t) * kVFloats + k];
    for (int base = 0; base < later; base += 32) {
      const int count = min(32, later - base);
      __syncwarp();
      if (lane < count) {
        const int other = pair + 1 + base + lane;
        const int q = pair_block[other];
        int lo = first_block + 1, hi = end_block - 1;  // the row's first block is (p, p)
        while (lo < hi) {
          const int mid = (lo + hi) >> 1;
          if (s_col[mid] < q) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }
        product[warp][lane] = make_int2(s_value_offsets[lo], pair_pm[other]);
      }
      __syncwarp();
      // kUnroll entries per lane per step, their loads issued together: the
      // V_qi gathers are the latency this loop waits on.
      constexpr int kUnroll = 4;
      const int total = count * kBlock;
      for (int idx0 = lane; idx0 < total; idx0 += 32 * kUnroll) {
        float a[kUnroll][DE], b[kUnroll][DE];
        int target[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const int idx = min(idx0 + 32 * u, total - 1);
          const int k = idx / kBlock;
          const int e = idx - k * kBlock;
          const int r = e / DP, c = e - r * DP;
          const int2 info = product[warp][k];
          Row::Load(&vp[warp][r * Row::kFloats], a[u]);
          Row::Load(pair_v + (static_cast<long long>(info.y) * DP + c) * Row::kFloats, b[u]);
          target[u] = info.x + e;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          if (idx0 + 32 * u >= total) break;
          float v = 0.f;
#pragma unroll
          for (int j = 0; j < DE; ++j) v += a[u][j] * b[u][j];
          atomicAdd(s_values + target[u], -v);
        }
      }
    }
  }
}

/** Damping (or a unit diagonal for a masked column) on the diagonal of S. */
__global__ void FinalizeDiagonalKernel(int num_cols, const int *col_natural,
                                       const int *col_diag_value, Modifiers mod, float *s_values) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cols) return;
  const int col = col_natural[c];
  float &d = s_values[col_diag_value[c]];
  d = mod.Mask(col) == 0.f ? 1.f : d + mod.Damping(col);
}

/** Expands the upper block storage of S to its full scalar CSR. */
__global__ void ExpandKernel(int num_nonzeros, const int *__restrict__ csr_src,
                             const float *__restrict__ s_values, float *__restrict__ values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_nonzeros) values[i] = s_values[csr_src[i]];
}

/** step at the retained columns = x_r. */
__global__ void ScatterRetainedKernel(int num_cols, const int *col_natural,
                                      const float *reduced_step, float *step) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < num_cols) step[col_natural[c]] = reduced_step[c];
}

/** One thread per landmark: x_e = L⁻ᵀ (L⁻¹ b_e - Σ_p V_piᵀ x_p). */
template <int DE, int DP>
__global__ void BackSubstituteKernel(
    int num_landmarks, int num_pairs, const int *__restrict__ landmark_pair_offsets,
    const int *__restrict__ pair_block, const int *__restrict__ pair_pm,
    const float *__restrict__ landmark_hessian, const int *__restrict__ landmark_natural_col,
    const int *__restrict__ block_col, const float *__restrict__ pair_v,
    const float *__restrict__ reduced_step, const float *__restrict__ rhs, Modifiers mod,
    float *__restrict__ step, int *failure) {
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= num_landmarks) return;
  const int ecol = landmark_natural_col[l];
  float L[DE][DE];
  const bool factored = LandmarkFactor<DE>(landmark_hessian, num_landmarks, l, ecol, mod, L);
  float u[DE];
#pragma unroll
  for (int j = 0; j < DE; ++j) u[j] = factored ? rhs[ecol + j] : CUDART_NAN_F;
  if (!factored) *failure = 1;
  ForwardSolve<DE>(L, u);
  for (int pair = landmark_pair_offsets[l]; pair < landmark_pair_offsets[l + 1]; ++pair) {
    const long long pm = pair_pm[pair];
    const float *x = reduced_step + block_col[pair_block[pair]];
#pragma unroll
    for (int r = 0; r < DP; ++r) {
      const float xr = x[r];
      float v[DE];
      VRow<DE>::Load(pair_v + (pm * DP + r) * VRow<DE>::kFloats, v);
#pragma unroll
      for (int j = 0; j < DE; ++j) u[j] -= v[j] * xr;
    }
  }
  BackwardSolve<DE>(L, u);
#pragma unroll
  for (int j = 0; j < DE; ++j) step[ecol + j] = u[j];
}

/** One thread per item: ‖J_f δ_f‖² of its factor f, written at f. */
__global__ void FactorCurvatureKernel(int num_factors, const BatchGeometry *batches,
                                      int num_batches, const int *slot_sizes,
                                      const int *slot_offsets, const int *item_cols,
                                      const int *evaluation_order,
                                      const float *__restrict__ jacobians,
                                      const float *__restrict__ step, float *curvature) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // item, so J streams
  if (i >= num_factors) return;
  const BatchGeometry &batch = batches[FindBatch(batches, num_batches, i)];
  const int item = i - batch.factor_begin;
  const int g = batch.factor_begin + evaluation_order[i];
  const int m = batch.num_residuals, n = batch.tangent_dim;
  const float *J = jacobians + batch.jacobian_offset + static_cast<long long>(item) * m * n;
  const int *cols = item_cols + batch.col_offset + item * batch.num_slots;
  float sum = 0.f;
  for (int row = 0; row < m; ++row) {
    float v = 0.f;
    for (int s = 0; s < batch.num_slots; ++s) {
      const int col = cols[s];
      if (col < 0) continue;
      const int offset = slot_offsets[batch.slot_begin + s];
      for (int c = 0; c < slot_sizes[batch.slot_begin + s]; ++c) {
        v += J[row * n + offset + c] * step[col + c];
      }
    }
    sum += v * v;
  }
  curvature[g] = sum;
}

/**
 * Runs the statement(s) with the constants DE and DP bound to the supported (landmark,
 * retained) tangent sizes (see SupportedDims).
 */
#define SCHUR_DISPATCH_DIMS(de, dp, ...) \
  do {                                   \
    if ((de) == 3 && (dp) == 6) {        \
      constexpr int DE = 3, DP = 6;      \
      static_cast<void>(DE);             \
      static_cast<void>(DP);             \
      __VA_ARGS__;                       \
    } else if ((de) == 1 && (dp) == 6) { \
      constexpr int DE = 1, DP = 6;      \
      static_cast<void>(DE);             \
      static_cast<void>(DP);             \
      __VA_ARGS__;                       \
    } else if ((de) == 2 && (dp) == 6) { \
      constexpr int DE = 2, DP = 6;      \
      static_cast<void>(DE);             \
      static_cast<void>(DP);             \
      __VA_ARGS__;                       \
    } else if ((de) == 3 && (dp) == 9) { \
      constexpr int DE = 3, DP = 9;      \
      static_cast<void>(DE);             \
      static_cast<void>(DP);             \
      __VA_ARGS__;                       \
    } else if ((de) == 3 && (dp) == 3) { \
      constexpr int DE = 3, DP = 3;      \
      static_cast<void>(DE);             \
      static_cast<void>(DP);             \
      __VA_ARGS__;                       \
    }                                    \
  } while (0)

/** Whether (de, dp) is a supported combination (see DispatchDims). */
bool SupportedDims(int de, int dp) {
  return (dp == 6 && de >= 1 && de <= 3) || (de == 3 && (dp == 9 || dp == 3));
}

}  // namespace

SchurSystem::SchurSystem(SparseLinearSolverPtr solver, bool check_failures)
    : solver_(std::move(solver)), check_failures_(check_failures), failure_(0, 1) {}

/** Setup temporaries passed between the phases of SchurSystem::Initialize(). */
struct SchurSystem::SetupScratch {
  dvector<uint8_t> temp;                ///< CUB temporary storage.
  int block_bits = 0;                   ///< Key bits of a retained state.
  int landmark_bits = 0;                ///< Key bits of a landmark.
  dvector<int> entry_offsets;           ///< Per factor: its first entry.
  dvector<int> rr_offsets;              ///< Per factor: its first C^R item.
  dvector<int> batch_has_landmarks;     ///< Per residual batch: has landmark factors.
  dvector<unsigned long long> rr_keys;  ///< Per C^R item: the key of its block of S.
  dvector<int> entry_factor;            ///< Per entry (landmark-major): its factor.
  dvector<int> entry_slot;              ///< Per entry (landmark-major): its retained slot.
  dvector<int> pair_entry_offsets;      ///< Per pair: its entries [offset, next).
  // The full (both triangles) block rows of S.
  dvector<int> full_row_offsets;  ///< Per retained state: its full blocks [offset, next).
  dvector<int> full_row;          ///< Per full block: its block row.
  dvector<int> full_col;          ///< Per full block: its block column.
  dvector<int> full_src;          ///< Per full block: its upper block (transposed if lower).
  dvector<int> width_pos;         ///< Per full block: first column within its row.
  int num_s_blocks = 0;           ///< Blocks of S (upper triangle).
  int num_full = 0;               ///< Full blocks.
  int num_values = 0;             ///< Values of S (upper block triangle).
  int num_nonzeros = 0;           ///< Nonzeros of the scalar CSR of S.
  // Geometry and counts.
  dvector<ColumnRange> ranges;        ///< Per state batch with free columns.
  dvector<int> block_dim;             ///< Per retained state: tangent size.
  dvector<int> error;                 ///< Setup error flags (see ClassifyFactorsKernel).
  int num_entries = 0;                ///< (landmark factor, retained slot) entries.
  int num_rr = 0;                     ///< (factor, retained slot, retained slot) items of C^R.
  dvector<long long> clique_offsets;  ///< Per landmark: start of its clique products.
  long long num_cliques = 0;          ///< Σ_i k_i (k_i - 1) / 2 off-diagonal clique products.
};

void SchurSystem::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                             const std::vector<bool> &reorderable) {
  const auto &eliminated = problem.EliminatedStateBatches();
  if (eliminated.empty()) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: declare the state batches to eliminate with "
        "Problem::SetEliminatedStateBatches");
  }
  num_cols_ = num_cols;
  structure_.ResolveColumns(stream, problem, num_cols);
  SetupScratch scratch;
  SetUpGeometry(stream, problem, scratch);
  ClassifyFactors(stream, scratch);
  BuildPairs(stream, scratch);
  BuildEvaluationOrder(stream, reorderable, scratch);
  BuildReducedPattern(stream, scratch);
  BuildReducedCsr(stream, scratch);
  AllocateNumericBuffers(stream, scratch);
}

/** Residual batch geometry and slot tables; the landmarks and the retained states, numbered in
 * column order. */
void SchurSystem::SetUpGeometry(cudaStream_t stream, const Problem &problem,
                                SetupScratch &scratch) {
  const auto &eliminated = problem.EliminatedStateBatches();
  const auto &residual_batches = problem.GetResidualBatches();
  const auto &layout = structure_.Layout();
  std::vector<BatchGeometry> batches;
  std::vector<int> slot_sizes, slot_offsets;
  factors_.batch_offsets.clear();
  int factor_cursor = 0;
  for (size_t b = 0; b < residual_batches.size(); ++b) {
    const HessianBatchLayout &lb = layout[b];
    BatchGeometry geometry{};
    geometry.jacobian_offset = static_cast<long long>(lb.jacobian_offset);
    geometry.residual_offset = static_cast<long long>(lb.residual_offset);
    geometry.factor_begin = factor_cursor;
    geometry.num_residuals = lb.residual_dim;
    geometry.tangent_dim = lb.tangent_dim;
    geometry.num_slots = lb.num_blocks;
    geometry.slot_begin = static_cast<int>(slot_sizes.size());
    geometry.col_offset = static_cast<int>(lb.col_offset);
    int offset = 0;
    for (size_t size : residual_batches[b].GetFactorBatch()->StateSizes()) {
      slot_sizes.push_back(static_cast<int>(size));
      slot_offsets.push_back(offset);
      offset += static_cast<int>(size);
    }
    batches.push_back(geometry);
    factors_.batch_offsets.push_back(factor_cursor);
    factor_cursor += lb.num_factors;
  }
  factors_.landmark.resize(factor_cursor);
  factors_.landmark_slot.resize(factor_cursor);
  factors_.batches = dvector<BatchGeometry>(batches);
  factors_.slot_sizes = dvector<int>(slot_sizes);
  factors_.slot_offsets = dvector<int>(slot_offsets);

  // Column ranges: landmarks and retained blocks, numbered in column order.
  std::vector<ColumnRange> ranges;
  int col = 0, landmarks = 0, blocks = 0, reduced_col = 0;
  landmark_dim_ = 0;
  for (const StateBatch *batch : problem.GetStateBatches()) {
    const int num_free = static_cast<int>(batch->NumActiveStates() - batch->NumConstStates());
    const int tangent = static_cast<int>(batch->TangentSize());
    if (num_free == 0 || tangent == 0) continue;
    const bool is_eliminated =
        std::find(eliminated.begin(), eliminated.end(), batch) != eliminated.end();
    if (is_eliminated) {
      if (landmark_dim_ != 0 && landmark_dim_ != tangent) {
        throw std::invalid_argument(
            "LinearSystemType::SchurComplement: the eliminated state batches have different "
            "tangent sizes");
      }
      landmark_dim_ = tangent;
    }
    ranges.push_back({col, col + num_free * tangent, tangent, is_eliminated ? landmarks : blocks,
                      is_eliminated ? 1 : 0});
    col += num_free * tangent;
    if (is_eliminated) {
      landmarks += num_free;
    } else {
      blocks += num_free;
    }
  }
  if (landmark_dim_ > 3) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: eliminated states must have tangent size 1 to 3");
  }

  scratch.ranges = dvector<ColumnRange>(ranges);
  scratch.block_dim.resize(blocks);
  pattern_.state_col.resize(blocks);
  pattern_.state_natural_col.resize(blocks);
  pattern_.landmark_natural_col.resize(landmarks);
  for (const ColumnRange &range : ranges) {
    const int num_states = (range.end_col - range.first_col) / range.tangent;
    if (range.eliminated) {
      LandmarkColumnsKernel<<<Blocks(num_states), kThreads, 0, stream>>>(
          num_states, range.tangent, range.first_id, range.first_col,
          pattern_.landmark_natural_col.data());
    } else {
      RetainedBlocksKernel<<<Blocks(num_states), kThreads, 0, stream>>>(
          num_states, range.tangent, range.first_id, range.first_col, reduced_col,
          scratch.block_dim.data(), pattern_.state_col.data(), pattern_.state_natural_col.data());
      reduced_col += num_states * range.tangent;
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  num_retained_cols_ = reduced_col;
  if (NumLandmarks() == 0) {
    LogWarning(
        "LinearSystemType::SchurComplement: no free state to eliminate; solving the full "
        "system densely through the reduced system");
  }
}

/** Classifies the factors (landmark, entries, C^R items) and checks the requirements of the class
 * comment. */
void SchurSystem::ClassifyFactors(cudaStream_t stream, SetupScratch &scratch) {
  dvector<uint8_t> &temp = scratch.temp;
  const int F = NumFactors();
  const int *factor_cols = structure_.FactorCols().data();
  std::vector<int> error_init = {0, INT_MAX, 0};
  scratch.error = dvector<int>(error_init);
  dvector<int> entry_count(F + 1), rr_count(F + 1);
  dvector<int> &entry_offsets = scratch.entry_offsets, &rr_offsets = scratch.rr_offsets;
  entry_offsets.resize(F + 1);
  rr_offsets.resize(F + 1);
  dvector<int> &batch_has_landmarks = scratch.batch_has_landmarks;
  batch_has_landmarks = dvector<int>(0, std::max(NumBatches(), 1));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(entry_count.data(), 0, (F + 1) * sizeof(int), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(rr_count.data(), 0, (F + 1) * sizeof(int), stream));
  if (F > 0) {
    ClassifyFactorsKernel<<<Blocks(F), kThreads, 0, stream>>>(
        F, factors_.batches.data(), NumBatches(), factors_.slot_sizes.data(), factor_cols,
        scratch.ranges.data(), static_cast<int>(scratch.ranges.size()), factors_.landmark.data(),
        factors_.landmark_slot.data(), entry_count.data(), rr_count.data(),
        batch_has_landmarks.data(), scratch.error.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  ExclusiveSum(stream, temp, entry_count.data(), entry_offsets.data(), F + 1);
  ExclusiveSum(stream, temp, rr_count.data(), rr_offsets.data(), F + 1);
  std::vector<int> errors(3);
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(errors.data(), scratch.error.data(), 3 * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  scratch.num_entries = ReadBack(stream, entry_offsets.data() + F);
  scratch.num_rr = ReadBack(stream, rr_offsets.data() + F);
  if (errors[0] & kTwoLandmarks) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: a factor touches two states of the eliminated state "
        "batches; keep such states retained");
  }
  pair_block_dim_ = scratch.num_entries > 0 ? errors[2] : 6;
  if (scratch.num_entries > 0 && errors[1] != errors[2]) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: the states that share factors with landmarks must all "
        "have one tangent size");
  }
  if (NumLandmarks() > 0 && !SupportedDims(landmark_dim_, pair_block_dim_)) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: unsupported tangent sizes (landmarks " +
        std::to_string(landmark_dim_) + ", states observing them " +
        std::to_string(pair_block_dim_) + "); supported: 1-3 with 6, and 3 with 3 or 9");
  }
  scratch.block_bits = BitsFor(std::max(NumRetainedStates(), 2));
  scratch.landmark_bits = BitsFor(std::max(NumLandmarks(), 2));
  if (2 * scratch.block_bits > 64 || scratch.landmark_bits + scratch.block_bits > 64) {
    throw std::invalid_argument("LinearSystemType::SchurComplement: problem too large");
  }
}

/** The (retained state, landmark) pairs, landmark-major, and their pose-major order cut into chunks
 * for the off-diagonal pass. */
void SchurSystem::BuildPairs(cudaStream_t stream, SetupScratch &scratch) {
  // Entries, sorted by (landmark, retained state); stable, so by factor within.
  dvector<uint8_t> &temp = scratch.temp;
  const int F = NumFactors();
  const int *factor_cols = structure_.FactorCols().data();
  const int block_bits = scratch.block_bits, landmark_bits = scratch.landmark_bits;
  const dvector<int> &entry_offsets = scratch.entry_offsets, &rr_offsets = scratch.rr_offsets;
  const int E = scratch.num_entries, R = scratch.num_rr;
  dvector<unsigned long long> entry_keys(E), entry_keys_sorted(E);
  dvector<int2> entry_values(E), entry_values_sorted(E);
  pattern_.rr_factor.resize(R);
  pattern_.rr_slots.resize(R);
  dvector<unsigned long long> &rr_keys = scratch.rr_keys;
  rr_keys.resize(R);
  if (F > 0) {
    EmitItemsKernel<<<Blocks(F), kThreads, 0, stream>>>(
        F, factors_.batches.data(), NumBatches(), factor_cols, scratch.ranges.data(),
        static_cast<int>(scratch.ranges.size()), factors_.landmark.data(),
        factors_.landmark_slot.data(), entry_offsets.data(), rr_offsets.data(), block_bits,
        entry_keys.data(), entry_values.data(), pattern_.rr_factor.data(), pattern_.rr_slots.data(),
        rr_keys.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  SortPairs(stream, temp, entry_keys.data(), entry_keys_sorted.data(), entry_values.data(),
            entry_values_sorted.data(), E, landmark_bits + block_bits);

  // Pairs: runs of equal keys.
  dvector<int> &entry_factor = scratch.entry_factor, &entry_slot = scratch.entry_slot;
  entry_factor.resize(E);
  entry_slot.resize(E);
  dvector<int> head(E), pair_id(E);
  int P = 0;
  if (E > 0) {
    UnpackEntriesKernel<<<Blocks(E), kThreads, 0, stream>>>(
        E, entry_keys_sorted.data(), entry_values_sorted.data(), entry_factor.data(),
        entry_slot.data(), head.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    InclusiveSum(stream, temp, head.data(), pair_id.data(), E);
    P = ReadBack(stream, pair_id.data() + E - 1);
  }
  pairs_.landmark.resize(P);
  pairs_.state.resize(P);
  dvector<int> &pair_entry_offsets = scratch.pair_entry_offsets;
  pair_entry_offsets.resize(P + 1);
  pairs_.landmark_offsets.resize(NumLandmarks() + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(pairs_.landmark_offsets.data(), 0,
                                      (NumLandmarks() + 1) * sizeof(int), stream));
  if (E > 0) {
    BuildPairsKernel<<<Blocks(E), kThreads, 0, stream>>>(
        E, entry_keys_sorted.data(), head.data(), pair_id.data(), block_bits,
        pairs_.landmark.data(), pairs_.state.data(), pair_entry_offsets.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    LandmarkOffsetsKernel<<<Blocks(P + 1), kThreads, 0, stream>>>(
        P, NumLandmarks(), pairs_.landmark.data(), pairs_.landmark_offsets.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // Pose-major order of the pairs (stable: landmark-ascending within a state).
  dvector<int> pair_iota;
  Iota(stream, pair_iota, P);
  dvector<unsigned> pm_keys(P);
  pairs_.pose_major.resize(P);
  pairs_.pose_major_position.resize(P);
  SortPairs(stream, temp, reinterpret_cast<const unsigned *>(pairs_.state.data()), pm_keys.data(),
            pair_iota.data(), pairs_.pose_major.data(), P, block_bits);
  if (P > 0) {
    InvertPermutationKernel<<<Blocks(P), kThreads, 0, stream>>>(P, pairs_.pose_major.data(),
                                                                pairs_.pose_major_position.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // Pose-major runs of the retained states, cut into chunks for the
  // off-diagonal pass.
  {
    dvector<int> pm_row_offsets(NumRetainedStates() + 1);
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(pm_row_offsets.data(), 0, (NumRetainedStates() + 1) * sizeof(int), stream));
    if (P > 0) {
      LandmarkOffsetsKernel<<<Blocks(P + 1), kThreads, 0, stream>>>(
          P, NumRetainedStates(), reinterpret_cast<const int *>(pm_keys.data()),
          pm_row_offsets.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    std::vector<int> offsets(NumRetainedStates() + 1);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(offsets.data(), pm_row_offsets.data(),
                                        offsets.size() * sizeof(int), cudaMemcpyDeviceToHost,
                                        stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    std::vector<int3> chunks;
    for (int b = 0; b < NumRetainedStates(); ++b) {
      for (int t = offsets[b]; t < offsets[b + 1]; t += kChunkPairs) {
        chunks.push_back(make_int3(b, t, std::min(t + kChunkPairs, offsets[b + 1])));
      }
    }
    pairs_.chunks = dvector<int3>(chunks);
  }
}

/** The pose-major entries and the evaluation order that lays their Jacobians out in that order. */
void SchurSystem::BuildEvaluationOrder(cudaStream_t stream, const std::vector<bool> &reorderable,
                                       SetupScratch &scratch) {
  dvector<uint8_t> &temp = scratch.temp;
  const int F = NumFactors(), E = scratch.num_entries, P = NumPairs();
  const int *factor_cols = structure_.FactorCols().data();
  const dvector<int> &pair_entry_offsets = scratch.pair_entry_offsets;
  const dvector<int> &entry_factor = scratch.entry_factor, &entry_slot = scratch.entry_slot;
  const dvector<int> &batch_has_landmarks = scratch.batch_has_landmarks;
  pairs_.entry_offsets.resize(P + 1);
  {
    dvector<int> pm_entry_count(P + 1);
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(pm_entry_count.data(), 0, (P + 1) * sizeof(int), stream));
    dvector<int> pm_entry_factor(E), pm_entry_slot(E);
    dvector<int> factor_key(F);
    if (F > 0) {
      THROW_ON_CUDA_ERROR(cudaMemsetAsync(factor_key.data(), 0x7f, F * sizeof(int), stream));
    }
    if (P > 0) {
      PoseMajorPairsKernel<<<Blocks(P), kThreads, 0, stream>>>(
          P, pairs_.pose_major.data(), pair_entry_offsets.data(), pm_entry_count.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    ExclusiveSum(stream, temp, pm_entry_count.data(), pairs_.entry_offsets.data(), P + 1);
    if (P > 0) {
      PoseMajorEntriesKernel<<<Blocks(P), kThreads, 0, stream>>>(
          P, pairs_.pose_major.data(), pair_entry_offsets.data(), pairs_.entry_offsets.data(),
          entry_factor.data(), entry_slot.data(), pm_entry_factor.data(), pm_entry_slot.data(),
          factor_key.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    std::vector<int> has_landmarks(NumBatches());
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(has_landmarks.data(), batch_has_landmarks.data(),
                                        NumBatches() * sizeof(int), cudaMemcpyDeviceToHost,
                                        stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    factors_.reordered.assign(NumBatches(), false);
    std::vector<int> reordered(NumBatches());
    for (int b = 0; b < NumBatches(); ++b) {
      factors_.reordered[b] =
          has_landmarks[b] != 0 && b < static_cast<int>(reorderable.size()) && reorderable[b];
      reordered[b] = factors_.reordered[b] ? 1 : 0;
    }
    dvector<int> reordered_device(reordered);
    dvector<unsigned long long> order_keys(F), order_keys_sorted(F);
    dvector<int> order_values(F), order_values_sorted(F);
    factors_.item.resize(F);
    factors_.evaluation_order.resize(F);
    factors_.item_cols.resize(structure_.FactorCols().size());
    if (F > 0) {
      OrderKeysKernel<<<Blocks(F), kThreads, 0, stream>>>(
          F, factors_.batches.data(), NumBatches(), reordered_device.data(), factor_key.data(),
          order_keys.data(), order_values.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
      SortPairs(stream, temp, order_keys.data(), order_keys_sorted.data(), order_values.data(),
                order_values_sorted.data(), F, 32 + BitsFor(std::max(NumBatches(), 2)));
      ItemsKernel<<<Blocks(F), kThreads, 0, stream>>>(
          F, factors_.batches.data(), NumBatches(), order_values_sorted.data(), factor_cols,
          factors_.item.data(), factors_.evaluation_order.data(), factors_.item_cols.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    pairs_.entry_jacobian.resize(E);
    pairs_.entry_residual.resize(E);
    pairs_.entry_meta.resize(E);
    if (E > 0) {
      EntryLocationsKernel<<<Blocks(E), kThreads, 0, stream>>>(
          E, pm_entry_factor.data(), pm_entry_slot.data(), factors_.batches.data(), NumBatches(),
          factors_.slot_offsets.data(), factors_.landmark_slot.data(), factors_.item.data(),
          pairs_.entry_jacobian.data(), pairs_.entry_residual.data(), pairs_.entry_meta.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
}

/** The block pattern of S: clique products, C^R items and the diagonal; value offsets and the full
 * block rows. */
void SchurSystem::BuildReducedPattern(cudaStream_t stream, SetupScratch &scratch) {
  dvector<uint8_t> &temp = scratch.temp;
  const int P = NumPairs(), R = scratch.num_rr, block_bits = scratch.block_bits;
  const dvector<unsigned long long> &rr_keys = scratch.rr_keys;
  // Clique products and the block pattern of S (upper triangle).
  const int L = NumLandmarks();
  dvector<long long> clique_counts(L + 1);
  scratch.clique_offsets.resize(L + 1);
  CliqueCountsKernel<<<Blocks(L + 1), kThreads, 0, stream>>>(L, pairs_.landmark_offsets.data(),
                                                             clique_counts.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  ExclusiveSum(stream, temp, clique_counts.data(), scratch.clique_offsets.data(), L + 1);
  scratch.num_cliques = ReadBack(stream, scratch.clique_offsets.data() + L);
  const long long K = scratch.num_cliques + NumRetainedStates() + R;
  if (K >= INT_MAX) {
    throw std::invalid_argument(
        "LinearSystemType::SchurComplement: too many landmark observation pairs");
  }
  const int Ki = static_cast<int>(K);
  dvector<unsigned long long> keys(Ki), keys_sorted(Ki);
  dvector<int> values(Ki), values_sorted(Ki);
  if (P > 0) {
    CliqueKeysKernel<<<Blocks(P), kThreads, 0, stream>>>(
        P, pairs_.landmark.data(), pairs_.state.data(), pairs_.landmark_offsets.data(),
        scratch.clique_offsets.data(), block_bits, keys.data(), values.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  if (NumRetainedStates() + R > 0) {
    OtherKeysKernel<<<Blocks(NumRetainedStates() + R), kThreads, 0, stream>>>(
        NumRetainedStates(), R, scratch.num_cliques, rr_keys.data(), block_bits, keys.data(),
        values.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  SortPairs(stream, temp, keys.data(), keys_sorted.data(), values.data(), values_sorted.data(), Ki,
            2 * block_bits);
  dvector<int> key_head(Ki), block_id(Ki);
  int &num_s_blocks = scratch.num_s_blocks;
  if (Ki > 0) {
    HeadFlagsKernel<<<Blocks(Ki), kThreads, 0, stream>>>(Ki, keys_sorted.data(), key_head.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    InclusiveSum(stream, temp, key_head.data(), block_id.data(), Ki);
    num_s_blocks = ReadBack(stream, block_id.data() + Ki - 1);
  }
  dvector<int> s_row(num_s_blocks);
  pattern_.upper_col.resize(num_s_blocks);
  dvector<int> &s_col = pattern_.upper_col;
  pattern_.state_diagonal.resize(NumRetainedStates());
  pattern_.rr_block.resize(R);
  if (Ki > 0) {
    AssignBlocksKernel<<<Blocks(Ki), kThreads, 0, stream>>>(
        Ki, keys_sorted.data(), values_sorted.data(), key_head.data(), block_id.data(), block_bits,
        scratch.num_cliques, NumRetainedStates(), s_row.data(), s_col.data(),
        pattern_.state_diagonal.data(), pattern_.rr_block.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  // The key arrays are the largest setup buffers; release them early.
  keys = {};
  keys_sorted = {};
  values = {};
  values_sorted = {};
  key_head = {};
  block_id = {};

  // Value offsets of the S blocks; full (both triangles) block rows.
  const int NB = NumRetainedStates();
  dvector<int> value_counts(num_s_blocks + 1), upper_count(NB + 1), lower_count(NB + 1);
  pattern_.value_offsets.resize(num_s_blocks + 1);
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(value_counts.data(), 0, (num_s_blocks + 1) * sizeof(int), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(upper_count.data(), 0, (NB + 1) * sizeof(int), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(lower_count.data(), 0, (NB + 1) * sizeof(int), stream));
  dvector<int> num_lower_device(0, 1);
  dvector<unsigned long long> lower_keys(num_s_blocks), lower_keys_sorted(num_s_blocks);
  dvector<int> lower_values(num_s_blocks), lower_values_sorted(num_s_blocks);
  if (num_s_blocks > 0) {
    BlockSizesKernel<<<Blocks(num_s_blocks), kThreads, 0, stream>>>(
        num_s_blocks, s_row.data(), s_col.data(), scratch.block_dim.data(), value_counts.data(),
        upper_count.data(), lower_count.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    LowerKeysKernel<<<Blocks(num_s_blocks), kThreads, 0, stream>>>(
        num_s_blocks, s_row.data(), s_col.data(), block_bits, lower_keys.data(),
        lower_values.data(), num_lower_device.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  ExclusiveSum(stream, temp, value_counts.data(), pattern_.value_offsets.data(), num_s_blocks + 1);
  const int num_lower = ReadBack(stream, num_lower_device.data());
  scratch.num_values = ReadBack(stream, pattern_.value_offsets.data() + num_s_blocks);
  SortPairs(stream, temp, lower_keys.data(), lower_keys_sorted.data(), lower_values.data(),
            lower_values_sorted.data(), num_lower, 2 * block_bits);
  pattern_.upper_row_offsets.resize(NB + 1);
  dvector<int> &upper_row_offsets = pattern_.upper_row_offsets;
  dvector<int> lower_row_offsets(NB + 1);
  dvector<int> &full_row_offsets = scratch.full_row_offsets;
  full_row_offsets.resize(NB + 1);
  ExclusiveSum(stream, temp, upper_count.data(), upper_row_offsets.data(), NB + 1);
  ExclusiveSum(stream, temp, lower_count.data(), lower_row_offsets.data(), NB + 1);
  FullRowOffsetsKernel<<<Blocks(NB + 1), kThreads, 0, stream>>>(
      NB, upper_row_offsets.data(), lower_row_offsets.data(), full_row_offsets.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  const int num_full = scratch.num_full = num_s_blocks + num_lower;
  dvector<int> &full_row = scratch.full_row, &full_col = scratch.full_col;
  dvector<int> &full_src = scratch.full_src, &width_pos = scratch.width_pos;
  full_row.resize(num_full);
  full_col.resize(num_full);
  full_src.resize(num_full);
  width_pos.resize(num_full + 1);
  dvector<int> widths(num_full + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(widths.data(), 0, (num_full + 1) * sizeof(int), stream));
  if (num_full > 0) {
    PlaceFullBlocksKernel<<<Blocks(num_full), kThreads, 0, stream>>>(
        num_s_blocks, num_lower, s_row.data(), s_col.data(), lower_keys_sorted.data(),
        lower_values_sorted.data(), upper_row_offsets.data(), lower_row_offsets.data(), block_bits,
        full_row.data(), full_col.data(), full_src.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    FullWidthsKernel<<<Blocks(num_full), kThreads, 0, stream>>>(
        num_full, full_col.data(), scratch.block_dim.data(), widths.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  ExclusiveSum(stream, temp, widths.data(), width_pos.data(), num_full + 1);
}

/** The scalar CSR of S (both triangles) and the column maps. */
void SchurSystem::BuildReducedCsr(cudaStream_t stream, SetupScratch &scratch) {
  dvector<uint8_t> &temp = scratch.temp;
  const int NB = NumRetainedStates(), num_full = scratch.num_full;
  const dvector<int> &full_row_offsets = scratch.full_row_offsets;
  const dvector<int> &full_row = scratch.full_row, &full_col = scratch.full_col;
  const dvector<int> &full_src = scratch.full_src, &width_pos = scratch.width_pos;
  dvector<int> row_length(NB), row_nonzeros(NB + 1), row_base(NB + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(row_nonzeros.data(), 0, (NB + 1) * sizeof(int), stream));
  if (NB > 0) {
    RowLengthsKernel<<<Blocks(NB), kThreads, 0, stream>>>(
        NB, full_row_offsets.data(), width_pos.data(), scratch.block_dim.data(), row_length.data(),
        row_nonzeros.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  ExclusiveSum(stream, temp, row_nonzeros.data(), row_base.data(), NB + 1);
  const int num_nonzeros = scratch.num_nonzeros = ReadBack(stream, row_base.data() + NB);
  reduced_.row_offsets.resize(num_retained_cols_ + 1);
  reduced_.col_ids.resize(num_nonzeros);
  reduced_.values.resize(num_nonzeros);
  pattern_.csr_source.resize(num_nonzeros);
  if (NB > 0) {
    CsrRowOffsetsKernel<<<Blocks(NB), kThreads, 0, stream>>>(
        NB, scratch.block_dim.data(), pattern_.state_col.data(), row_base.data(), row_length.data(),
        num_nonzeros, num_retained_cols_, reduced_.row_offsets.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  if (num_full > 0) {
    CsrFillKernel<<<Blocks(num_full), kThreads, 0, stream>>>(
        num_full, full_row.data(), full_col.data(), full_src.data(), full_row_offsets.data(),
        width_pos.data(), scratch.block_dim.data(), pattern_.state_col.data(), row_base.data(),
        row_length.data(), pattern_.value_offsets.data(), reduced_.col_ids.data(),
        pattern_.csr_source.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  pattern_.col_natural.resize(num_retained_cols_);
  pattern_.col_diagonal.resize(num_retained_cols_);
  if (NB > 0) {
    ColumnMapsKernel<<<Blocks(NB), kThreads, 0, stream>>>(
        NB, scratch.block_dim.data(), pattern_.state_col.data(), pattern_.state_natural_col.data(),
        pattern_.state_diagonal.data(), pattern_.value_offsets.data(), pattern_.col_natural.data(),
        pattern_.col_diagonal.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

/** The per-solve buffers. */
void SchurSystem::AllocateNumericBuffers(cudaStream_t stream, const SetupScratch &scratch) {
  const int L = NumLandmarks(), P = NumPairs(), F = NumFactors();
  const int tri = landmark_dim_ * (landmark_dim_ + 1) / 2;
  hessian_diagonal_.resize(num_cols_);
  landmark_hessian_.resize(static_cast<size_t>(tri) * L);
  s_values_.resize(scratch.num_values);
  pair_v_.resize(static_cast<size_t>(P) * pair_block_dim_ *
                 (landmark_dim_ == 3 ? 4 : landmark_dim_));
  reduced_rhs_.resize(num_retained_cols_);
  reduced_step_.resize(num_retained_cols_);
  factor_curvature_.resize(F);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  LogMessage(
      "Schur complement: {} landmarks (dim {}), {} retained states ({} columns, dim {} with "
      "landmarks), {} pairs, {} clique products, S: {} blocks, {} nonzeros",
      L, landmark_dim_, NumRetainedStates(), num_retained_cols_, pair_block_dim_, P,
      scratch.num_cliques, scratch.num_s_blocks, scratch.num_nonzeros);
}

void SchurSystem::Assemble(cudaStream_t stream, const Problem &, const float *jacobians,
                           const float *residuals, dvector<float> &rhs) {
  jacobians_ = jacobians;
  rhs.resize(num_cols_);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(rhs.data(), 0, rhs.size() * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(hessian_diagonal_.data(), 0,
                                      hessian_diagonal_.size() * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(landmark_hessian_.data(), 0,
                                      landmark_hessian_.size() * sizeof(float), stream));
  if (NumFactors() == 0) return;
  AssembleFactorsKernel<<<Blocks(NumFactors()), kThreads, 0, stream>>>(
      NumFactors(), factors_.batches.data(), NumBatches(), factors_.slot_sizes.data(),
      factors_.slot_offsets.data(), factors_.item_cols.data(), factors_.landmark.data(),
      factors_.landmark_slot.data(), factors_.evaluation_order.data(), landmark_dim_, jacobians,
      residuals, rhs.data(), hessian_diagonal_.data(), landmark_hessian_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (NumPairs() == 0) return;
  SCHUR_DISPATCH_DIMS(landmark_dim_, pair_block_dim_, {
    AssemblePairsKernel<DP><<<Blocks(NumPairs()), kThreads, 0, stream>>>(
        NumPairs(), pairs_.entry_offsets.data(), pairs_.entry_jacobian.data(),
        pairs_.entry_residual.data(), pairs_.entry_meta.data(), pairs_.pose_major.data(),
        pairs_.state.data(), jacobians, residuals, pattern_.state_natural_col.data(), rhs.data(),
        hessian_diagonal_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  });
}

void SchurSystem::ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  diagonal.resize(hessian_diagonal_.size());
  if (diagonal.empty()) return;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(diagonal.data(), hessian_diagonal_.data(),
                                      diagonal.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                      stream));
}

void SchurSystem::FormReducedSystem(cudaStream_t stream, const dvector<float> &rhs) {
  const Modifiers mod{ColumnScale() ? ColumnScale()->data() : nullptr,
                      Mask() ? Mask()->data() : nullptr, Damping() ? Damping()->data() : nullptr};
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(s_values_.data(), 0, s_values_.size() * sizeof(float), stream));
  if (num_retained_cols_ > 0) {
    GatherRhsKernel<<<Blocks(num_retained_cols_), kThreads, 0, stream>>>(
        num_retained_cols_, pattern_.col_natural.data(), rhs.data(), reduced_rhs_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  const int *factor_cols = structure_.FactorCols().data();
  const int num_rr = static_cast<int>(pattern_.rr_factor.size());
  if (num_rr > 0) {
    RetainedProductsKernel<<<Blocks(num_rr), kThreads, 0, stream>>>(
        num_rr, pattern_.rr_factor.data(), pattern_.rr_slots.data(), pattern_.rr_block.data(),
        factors_.batches.data(), NumBatches(), factors_.slot_sizes.data(),
        factors_.slot_offsets.data(), factor_cols, factors_.item.data(), jacobians_,
        pattern_.value_offsets.data(), mod, s_values_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  if (NumPairs() > 0) {
    SCHUR_DISPATCH_DIMS(landmark_dim_, pair_block_dim_, {
      EliminatePairsKernel<DE, DP><<<Blocks(NumPairs()), kThreads, 0, stream>>>(
          NumPairs(), pairs_.entry_offsets.data(), pairs_.entry_jacobian.data(),
          pairs_.entry_meta.data(), pairs_.pose_major.data(), pairs_.landmark.data(),
          pairs_.state.data(), jacobians_, landmark_hessian_.data(), NumLandmarks(),
          pattern_.landmark_natural_col.data(), pattern_.state_natural_col.data(),
          pattern_.state_col.data(), pattern_.state_diagonal.data(), pattern_.value_offsets.data(),
          rhs.data(), mod, pair_v_.data(), s_values_.data(), reduced_rhs_.data(), failure_.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
      OffDiagonalRowsKernel<DE, DP>
          <<<static_cast<unsigned>(pairs_.chunks.size()), kThreads, 0, stream>>>(
              pairs_.chunks.data(), pattern_.upper_row_offsets.data(), pattern_.upper_col.data(),
              pattern_.value_offsets.data(), pairs_.pose_major.data(), pairs_.landmark.data(),
              pairs_.state.data(), pairs_.pose_major_position.data(),
              pairs_.landmark_offsets.data(), pair_v_.data(), s_values_.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    });
  }
  if (num_retained_cols_ > 0) {
    FinalizeDiagonalKernel<<<Blocks(num_retained_cols_), kThreads, 0, stream>>>(
        num_retained_cols_, pattern_.col_natural.data(), pattern_.col_diagonal.data(), mod,
        s_values_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  const int nnz = static_cast<int>(pattern_.csr_source.size());
  if (nnz > 0) {
    ExpandKernel<<<Blocks(nnz), kThreads, 0, stream>>>(nnz, pattern_.csr_source.data(),
                                                       s_values_.data(), reduced_.values.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

bool SchurSystem::PrepareSolver(cudaStream_t stream, const Problem &problem,
                                const dvector<float> &rhs, dvector<float> &) {
  FormReducedSystem(stream, rhs);
  return solver_->Initialize(stream, problem, reduced_, reduced_rhs_, reduced_step_);
}

bool SchurSystem::Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) {
  if (check_failures_) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(failure_.data(), 0, sizeof(int), stream));
  }
  FormReducedSystem(stream, rhs);
  if (!solver_->Solve(stream, reduced_, reduced_rhs_, reduced_step_)) return false;
  step.resize(num_cols_);
  if (num_retained_cols_ > 0) {
    ScatterRetainedKernel<<<Blocks(num_retained_cols_), kThreads, 0, stream>>>(
        num_retained_cols_, pattern_.col_natural.data(), reduced_step_.data(), step.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  if (NumLandmarks() > 0) {
    const Modifiers mod{ColumnScale() ? ColumnScale()->data() : nullptr,
                        Mask() ? Mask()->data() : nullptr, Damping() ? Damping()->data() : nullptr};
    SCHUR_DISPATCH_DIMS(landmark_dim_, pair_block_dim_, {
      BackSubstituteKernel<DE, DP><<<Blocks(NumLandmarks()), kThreads, 0, stream>>>(
          NumLandmarks(), NumPairs(), pairs_.landmark_offsets.data(), pairs_.state.data(),
          pairs_.pose_major_position.data(), landmark_hessian_.data(),
          pattern_.landmark_natural_col.data(), pattern_.state_col.data(), pair_v_.data(),
          reduced_step_.data(), rhs.data(), mod, step.data(), failure_.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    });
  }
  if (check_failures_ && ReadBack(stream, failure_.data()) != 0) {
    LogError("Schur complement: a landmark block could not be factored (non-finite values)");
    return false;
  }
  return true;
}

void SchurSystem::StepCurvature(cudaStream_t stream, const dvector<float> &step,
                                ProblemPartition &partition, float *out) const {
  if (NumFactors() > 0) {
    FactorCurvatureKernel<<<Blocks(NumFactors()), kThreads, 0, stream>>>(
        NumFactors(), factors_.batches.data(), NumBatches(), factors_.slot_sizes.data(),
        factors_.slot_offsets.data(), factors_.item_cols.data(), factors_.evaluation_order.data(),
        jacobians_, step.data(), factor_curvature_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  partition.SumFactorCosts(stream, factor_curvature_.data(), out);
}

const int *SchurSystem::EvaluationOrder(size_t batch) const {
  if (batch >= factors_.reordered.size() || !factors_.reordered[batch]) return nullptr;
  return factors_.evaluation_order.data() + factors_.batch_offsets[batch];
}

}  // namespace cunls
