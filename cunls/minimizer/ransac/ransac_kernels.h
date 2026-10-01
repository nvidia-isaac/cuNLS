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

#pragma once

/**
 * @file ransac_kernels.h
 * @brief Kernel launchers of the RANSAC minimizers. Internal: not part of the
 * public API (not included by cunls.h); declared here for the host components
 * and the tests.
 *
 * Vocabulary:
 *  - slot: one independent small problem (a hypothesis, a local-optimization
 *    candidate, or the final refinement) with its own state replicas.
 *  - item: one factor evaluation that contributes to a slot.
 *  - view: where a slot finds one residual batch's rows in an output buffer.
 *
 * Every reduction runs in a fixed order, so results are bitwise reproducible.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cunls {
namespace ransac_internal {

/** @brief Most state blocks a single factor may reference. */
constexpr int kMaxBlocksPerFactor = 8;

/** @brief Largest supported free tangent dimension (mirrors kMaxRansacTangentDim). */
constexpr int kMaxDim = 64;

/** @brief Shared-memory words of one block group of the normal-equation kernel. */
constexpr int kBlockGroupWords = 8192;

/**
 * @brief Shared-memory words the normal-equation kernel needs to stage one item:
 * m_max Jacobian rows padded to a multiple of 4 columns plus a residual each,
 * two index words, and the item's local block columns.
 */
__host__ __device__ inline int NormalEquationsItemWords(int m_max, int dim) {
  return m_max * (((dim + 3) & ~3) + 1) + 2 + kMaxBlocksPerFactor;
}

// ---------------------------------------------------------------------------
// Shared descriptors
// ---------------------------------------------------------------------------

/**
 * @brief How a slot reads one residual batch.
 *
 * kViewSamples: the slot's rows are its minimal sample, row t = sample entry t
 * (rows of sample entries from other batches are unused). kViewPerSlot: row f
 * = factor f. kViewPerSlotMasked: as kViewPerSlot, skipping factors whose mask
 * byte is 0. Rows of slot p start at p * stride.
 */
enum ViewKind : int { kViewSamples = 0, kViewPerSlot = 1, kViewPerSlotMasked = 2 };

/** @brief One residual batch as seen by the slots of a slot set. */
struct BatchView {
  int kind = kViewPerSlot;
  int m = 0;            ///< Residual dimension.
  int n = 0;            ///< Sum of the factor's block tangent sizes.
  int nb = 0;           ///< State blocks per factor.
  int num_factors = 0;  ///< Factors in the batch.
  int u_offset = -1;    ///< Offset in the concatenated sampled index; -1 if not sampled.
  int block_col_off[kMaxBlocksPerFactor] = {};  ///< Column offset of each block within n.
  int block_size[kMaxBlocksPerFactor] = {};     ///< Tangent size of each block.
  const int *local_col = nullptr;  ///< num_factors * nb local columns; -1 = constant block.
  const float *res = nullptr;      ///< Residual buffer base.
  const float *jac = nullptr;      ///< Jacobian buffer base (may be null for cost-only use).
  const float *cost = nullptr;     ///< Per-factor cost buffer base (may be null).
  size_t stride_res = 0;           ///< Floats between consecutive slots.
  size_t stride_jac = 0;           ///< Floats between consecutive slots.
  size_t stride_cost = 0;          ///< Floats between consecutive slots.
  float tau_sq = 0.f;              ///< Squared inlier threshold (sampled batches).
};

/** @brief Everything a slot needs to enumerate its items. */
struct SlotItems {
  const BatchView *views = nullptr;  ///< Device array.
  int num_views = 0;
  const int *samples = nullptr;   ///< num_slots * sample_size concatenated sampled indices.
  int sample_size = 0;            ///< Items taken from sample views (0 = none).
  const uint8_t *mask = nullptr;  ///< Per-slot inlier mask over the concatenated sampled index.
  int mask_stride = 0;            ///< Bytes between consecutive slots' masks.
  int items_per_slot = 0;         ///< sample_size + sum of per-slot view sizes.
  int m_max = 0;                  ///< Largest residual dimension among the views.
};

/** @brief One state batch, for building factor state-pointer tables. */
struct StateView {
  const float *base = nullptr;  ///< User storage (the initial guess / constants).
  float *rep_cur = nullptr;     ///< Current replicas (nullptr if not replicated).
  float *rep_cand = nullptr;    ///< Candidate replicas (nullptr if not replicated).
  int num_blocks = 0;
  int ambient = 0;
};

/** @brief Step policy of the accept kernel. */
struct DevicePolicy {
  int levenberg_marquardt = 0;
  float lambda_upscale = 1.f;
  float lambda_downscale = 1.f;
  float lambda_max = 0.f;
  float lambda_min = 0.f;
  float step_accept_threshold = 0.f;
  float lambda_downscale_threshold = 0.f;
  float cost_tolerance = 0.f;
  float state_tolerance = 0.f;
};

/** @brief Round statistics kept on the device, read once per round. */
struct DeviceStats {
  float best_score;      ///< Best score so far (lower is better).
  int best_inliers;      ///< Inliers of the best-so-far estimate.
  int improved;          ///< 1 if the last selection improved the best score.
  int best_slot;         ///< Slot that produced the improvement in the last selection.
  int valid_total;       ///< Valid hypotheses accumulated over rounds.
  float last_min_score;  ///< Best score within the last selection.
};

// ---------------------------------------------------------------------------
// Sampling (slot_kernels.cu)
// ---------------------------------------------------------------------------

/** @brief splitmix64 finalizer, used for counter-based keys. */
__host__ __device__ inline uint64_t Mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

/** @brief Key for (seed, round, stream id), e.g. the slot whose sample is drawn. */
__host__ __device__ inline uint64_t PermutationKey(uint64_t seed, uint64_t round, uint64_t id) {
  return Mix64(Mix64(Mix64(seed) ^ round) ^ (id * 0xD6E8FEB86659FD93ull));
}

/**
 * @brief Keyed bijection on [0, n) without sorting: a 4-round Feistel network
 * on the smallest even bit width covering n, cycle-walking back into range.
 */
__host__ __device__ inline uint32_t PermuteIndex(uint32_t index, uint32_t n, uint64_t key) {
  if (n <= 1) {
    return 0;
  }
  int bits = 2;
  while ((1ull << bits) < n) {
    bits += 2;
  }
  const int half = bits / 2;
  const uint32_t mask = (1u << half) - 1u;
  uint32_t x = index;
  do {
    uint32_t left = x >> half;
    uint32_t right = x & mask;
    for (uint32_t r = 0; r < 4; ++r) {
      const uint32_t f =
          static_cast<uint32_t>(Mix64(key ^ (static_cast<uint64_t>(r) << 56) ^ right)) & mask;
      const uint32_t next = left ^ f;
      left = right;
      right = next;
    }
    x = (left << half) | right;
  } while (x >= n);
  return x;
}

/**
 * @brief Minimal samples: slot p gets `sample_size` distinct indices into the
 * concatenated sampled factors, samples[p * sample_size + t] =
 * PermuteIndex(t, n, PermutationKey(seed, round, p)).
 */
void LaunchDrawSamples(cudaStream_t stream, int n, int sample_size, int num_slots, uint64_t seed,
                       uint64_t round, int *samples);

// ---------------------------------------------------------------------------
// State-pointer tables and replicas (slot_kernels.cu)
// ---------------------------------------------------------------------------

/**
 * @brief Tables and factor ids for evaluating the minimal samples of one
 * sampled batch: item p * sample_size + t is slot p's sample entry t. If that
 * entry belongs to this batch (u_offset <= u < u_offset + num_factors) the item
 * evaluates factor u - u_offset against replica p (current or candidate);
 * otherwise it evaluates factor 0 and its row is ignored.
 */
void LaunchSampleTables(cudaStream_t stream, const StateView *states, const int2 *blocks, int nb,
                        int num_factors, int u_offset, const int *samples, int sample_size,
                        int num_slots, bool candidate, float **table, int *factor_ids);

/**
 * @brief num_slots pointer tables: local slot p evaluates every factor against
 * replica `slot_offset + p`.
 */
void LaunchSlotTables(cudaStream_t stream, const StateView *states, const int2 *blocks,
                      int num_factors, int nb, int num_slots, int slot_offset, bool candidate,
                      float **tables);

/**
 * @brief Pointer tables for slots listed in slot_index: local slot p evaluates
 * every factor against replica slot_index[p].
 */
void LaunchIndexedSlotTables(cudaStream_t stream, const StateView *states, const int2 *blocks,
                             int num_factors, int nb, const int *slot_index, int num_slots,
                             float **tables);

/**
 * @brief Tables and factor ids for evaluating a factor subset: item
 * t = p * count + j evaluates factor ids[j] against replica slot_offset + p.
 */
void LaunchSubsetTables(cudaStream_t stream, const StateView *states, const int2 *blocks, int nb,
                        const int *ids, int count, int num_slots, int slot_offset, float **tables,
                        int *item_ids);

/** @brief ids[j] = the j-th element of a keyed permutation of [0, n), j < count. */
void LaunchPermutationPrefix(cudaStream_t stream, int n, int count, uint64_t key, int *ids);

/** @brief out[j * nb + b] = local_col[ids[j] * nb + b]. */
void LaunchGatherLocalColumns(cudaStream_t stream, const int *local_col, const int *ids, int count,
                              int nb, int *out);

/** @brief data[i] = value for i < n. */
void LaunchFill(cudaStream_t stream, float *data, size_t n, float value);

/**
 * @brief Expands per-slot local steps into one state batch's full tangent
 * layout (zeros for constant blocks).
 *
 * @param block_col num_blocks entries: local column of each block or -1.
 */
void LaunchScatterDelta(cudaStream_t stream, int num_slots, int dim, const float *delta,
                        int num_blocks, int tangent, const int *block_col, float *delta_full);

/** @brief cur[p] = cand[p] for the slots with accept[p] != 0. */
void LaunchCopyAccepted(cudaStream_t stream, int num_slots, size_t slot_floats, const int *accept,
                        const float *cand, float *cur);

/**
 * @brief Broadcasts one source replica: dst slot p = src slot s for every p in
 * [0, num_slots), where s = *src_slot (device) or 0 when src_slot is null.
 * When only_if is given and *only_if == 0, nothing is copied.
 */
void LaunchCopyReplicas(cudaStream_t stream, int num_slots, size_t slot_floats, const float *src,
                        const int *src_slot, const int *only_if, float *dst);

// ---------------------------------------------------------------------------
// Per-slot Gauss-Newton / LM step (normal_equations_kernels.cu,
// dense_solve_kernels.cu, slot_kernels.cu)
// ---------------------------------------------------------------------------

/** @brief Thread-group shape of the normal-equation and cost kernels. */
enum class SlotGroup {
  kAuto,   ///< Warp per slot for small systems, block per slot otherwise.
  kWarp,   ///< One warp per slot (few items, small D).
  kBlock,  ///< One 256-thread block per slot (many items or large D).
};

/**
 * @brief Scratch floats LaunchNormalEquations needs for these items: nonzero
 * when a slot has so many rows that it is split across several blocks.
 */
size_t NormalEquationsScratchFloats(const SlotItems &items, int num_slots, int dim,
                                    SlotGroup group = SlotGroup::kAuto);

/**
 * @brief Per-slot normal equations H = J^T J (dim x dim, full), g = -J^T r,
 * and the sum of item costs.
 *
 * @param scratch At least NormalEquationsScratchFloats() floats (may be null
 *        when that is 0).
 * @param group kWarp is honored only for systems small enough for a warp.
 */
void LaunchNormalEquations(cudaStream_t stream, const SlotItems &items, int num_slots, int dim,
                           float *hessian, float *gradient, float *cost, float *scratch,
                           SlotGroup group = SlotGroup::kAuto);

/** @brief Scratch floats LaunchSlotCost needs to split large slots across blocks. */
size_t SlotCostScratchFloats(const SlotItems &items, int num_slots);

/**
 * @brief Per-slot sum of item costs. Large slots are split across blocks when
 * scratch (SlotCostScratchFloats()) is given; the partials are added in order.
 */
void LaunchSlotCost(cudaStream_t stream, const SlotItems &items, int num_slots, float *cost,
                    float *scratch, SlotGroup group = SlotGroup::kAuto);

/** @brief Dense solvers, mirroring RansacLinearSolverType. */
enum SolverKind : int { kSolveCholesky = 0, kSolveLDLT = 1 };

/**
 * @brief Per-slot damped solve, one warp per slot.
 *
 * Solves (H + lambda * max(diag(H), 1e-6)) delta = g, or, when lambda is null,
 * Gauss-Newton with a relative diagonal floor of 1e-6. The system is
 * Jacobi-equilibrated first. kSolveCholesky fails on a non-positive pivot;
 * kSolveLDLT pivots on the diagonal and gives rank-deficient directions a zero
 * step. Inactive slots get a zero step. Also writes the predicted decrease
 * g^T delta - 0.5 delta^T H delta (undamped H), |delta|^2, and solve_ok.
 */
void LaunchSolve(cudaStream_t stream, int num_slots, int dim, SolverKind solver,
                 const float *hessian, const float *gradient, const float *lambda,
                 const int *active, float *delta, float *predicted, float *step_sq, int *solve_ok);

/**
 * @brief Accept / reject per slot; updates active flags and lambda.
 *
 * iterations[p] counts calls made while the slot was active. On a slot's
 * first such call, valid[p] (if given) is set to solve_ok[p].
 */
void LaunchAccept(cudaStream_t stream, int num_slots, const DevicePolicy &policy,
                  const float *cost_cur, const float *cost_cand, const float *predicted,
                  const float *step_sq, const int *solve_ok, int *active, float *lambda,
                  int *accept, int *valid, int *iterations, int *num_accepted);

/** @brief *count = number of slots with active[p] != 0 (one block, deterministic). */
void LaunchCountActive(cudaStream_t stream, int num_slots, const int *active, int *count);

/**
 * @brief Resets per-slot solver state: active = 1, lambda = initial, valid = 1
 * (if given), iteration and accept counters = 0.
 */
void LaunchResetSlots(cudaStream_t stream, int num_slots, float initial_lambda, int *active,
                      float *lambda, int *valid, int *iterations, int *num_accepted);

// ---------------------------------------------------------------------------
// Scoring and selection (scoring_kernels.cu)
// ---------------------------------------------------------------------------

/** @brief Scoring rules, mirroring RansacScoring. */
enum ScoringRule : int { kScoreMSAC = 0, kScoreInlierCount = 1 };

/** @brief What the score kernel reads. */
struct ScoreInputs {
  const BatchView *sampled = nullptr;  ///< Per-slot raw residual views of kSampled batches.
  int num_sampled = 0;
  const BatchView *always_on = nullptr;  ///< Per-slot cost views of kAlwaysOn batches.
  int num_always_on = 0;
  int total_sampled = 0;  ///< Concatenated sampled factor count.
  int rule = kScoreMSAC;
  int add_always_on = 0;
  /**
   * When set, a factor is an inlier only if its Jacobian has a non-zero entry
   * on a free block (sampled views must then carry jac / local_col). Guards
   * against factors that report a zero residual for configurations they
   * cannot evaluate, e.g. points behind the camera.
   */
  int require_informative = 0;
};

/** @brief Scratch floats LaunchScore needs for `num_slots` slots. */
size_t ScoreScratchFloats(int total_sampled, int num_slots);

/**
 * @brief Scores slots [0, num_slots). Sampled views are indexed by the local
 * slot, always-on views by `slot_offset + local slot`; outputs by
 * `slot_offset + local slot`. mask (optional) receives inlier bytes with
 * stride total_sampled per local slot. valid (optional, global index) forces
 * a +inf score when 0. Large slots are split across blocks; scratch holds
 * their partial sums (ScoreScratchFloats()). With slot_index, the global index
 * of local slot p is slot_index[p] instead of slot_offset + p.
 */
void LaunchScore(cudaStream_t stream, const ScoreInputs &inputs, int num_slots, int slot_offset,
                 const int *valid, float *score, int *inliers, uint8_t *mask, float *scratch,
                 const int *slot_index = nullptr);

/**
 * @brief Picks the top_m (<= 64) lowest scores (NaN = +inf, ties to the lower
 * index), updates the best-so-far statistics, and adds the number of valid
 * slots (if valid is given). stats->improved is 1 iff the minimum beats
 * stats->best_score.
 */
void LaunchSelect(cudaStream_t stream, const float *score, const int *inliers, const int *valid,
                  int num_slots, int top_m, int *top_indices, DeviceStats *stats);

/** @brief Initializes device statistics (best score = +inf). */
void LaunchInitStats(cudaStream_t stream, DeviceStats *stats);

}  // namespace ransac_internal
}  // namespace cunls
