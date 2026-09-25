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
#include <stdexcept>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/factor/factor_batch.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/numeric_diff_jacobian.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

// One thread per (factor, tangent-dof, residual-row) element. Reads the two
// (or one, for forward diff) perturbed residual evaluations for that dof and
// writes the finite-difference column directly into the dense per-factor
// Jacobian layout (`ResidualsSize() x sum(StateBlockSizes())`, row-major,
// per factor) that analytic Jacobians also use.
__global__ void NumericDiffColumnKernel(
    const float *__restrict__ perturbed_residuals, const float *__restrict__ baseline_residuals,
    const int *__restrict__ col_idx, const int *__restrict__ plus_slot,
    const int *__restrict__ minus_slot, const float *__restrict__ eps_arr, bool central, int F,
    int W, int residual_size, int total_cols, float *__restrict__ jacobian_out) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  long long total = static_cast<long long>(F) * W * residual_size;
  if (idx >= total) return;

  int r = static_cast<int>(idx % residual_size);
  long long tmp = idx / residual_size;
  int j = static_cast<int>(tmp % W);
  int f = static_cast<int>(tmp / W);

  int ps = plus_slot[j];
  float rp = perturbed_residuals[(static_cast<long long>(ps) * F + f) * residual_size + r];

  float rm;
  float denom;
  if (central) {
    int ms = minus_slot[j];
    rm = perturbed_residuals[(static_cast<long long>(ms) * F + f) * residual_size + r];
    denom = 2.0f * eps_arr[j];
  } else {
    rm = baseline_residuals[static_cast<long long>(f) * residual_size + r];
    denom = eps_arr[j];
  }

  float deriv = (rp - rm) / denom;
  jacobian_out[(static_cast<long long>(f) * residual_size + r) * total_cols + col_idx[j]] = deriv;
}

// Grows *ptr (a pinned host allocation) to at least `n` elements if needed,
// then returns it. Kept per-ComputeCache (never shared across residual
// batches) so that one batch's rebuild can never overwrite host memory a
// prior batch's still-in-flight async H2D upload is reading from.
template <typename T>
T *EnsurePinnedHost(T *&ptr, size_t &capacity, size_t n) {
  if (n > capacity) {
    if (ptr != nullptr) THROW_ON_CUDA_ERROR(cudaFreeHost(ptr));
    THROW_ON_CUDA_ERROR(cudaMallocHost(&ptr, n * sizeof(T)));
    capacity = n;
  }
  return ptr;
}

}  // namespace

NumericDiffJacobianBuilder::ComputeCache::~ComputeCache() {
  if (pinned_delta_host != nullptr) cudaFreeHost(pinned_delta_host);
  if (pinned_ptrs_host != nullptr) cudaFreeHost(pinned_ptrs_host);
  if (pinned_int_host != nullptr) cudaFreeHost(pinned_int_host);
  if (pinned_eps_host != nullptr) cudaFreeHost(pinned_eps_host);
}

NumericDiffJacobianBuilder::NumericDiffJacobianBuilder() {
  THROW_ON_CUDA_ERROR(cudaEventCreateWithFlags(&delta_ready_event_, cudaEventDisableTiming));
}

NumericDiffJacobianBuilder::~NumericDiffJacobianBuilder() {
  for (auto ev : pool_events_) {
    if (ev != nullptr) cudaEventDestroy(ev);
  }
  for (auto s : pool_streams_) {
    if (s != nullptr) cudaStreamDestroy(s);
  }
  if (delta_ready_event_ != nullptr) {
    cudaEventDestroy(delta_ready_event_);
  }
}

void NumericDiffJacobianBuilder::EnsureStreamPool(size_t num_streams) {
  while (pool_streams_.size() < num_streams) {
    cudaStream_t s;
    THROW_ON_CUDA_ERROR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    pool_streams_.push_back(s);
    cudaEvent_t ev;
    THROW_ON_CUDA_ERROR(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    pool_events_.push_back(ev);
  }
}

void NumericDiffJacobianBuilder::PrepareResidualBatch(const Problem &problem,
                                                      size_t residual_batch_index) {
  const auto &residual_batches = problem.GetResidualBatches();
  if (residual_batch_index >= residual_batches.size()) {
    throw std::runtime_error(
        "NumericDiffJacobianBuilder::PrepareResidualBatch: index out of range");
  }
  const auto &rb = residual_batches[residual_batch_index];
  const FactorBatch *factor_batch = rb.GetFactorBatch();
  const auto &state_batches = problem.GetStateBatches();
  const auto &host_ptrs = problem.GetStatePointers()[residual_batch_index];

  const size_t F = factor_batch->NumFactors();
  auto block_sizes = factor_batch->StateBlockSizes();
  const size_t P = block_sizes.size();

  // Address -> (state batch index, block index) lookup, built from the
  // problem's own state batches. This mirrors the pointer-identity walk
  // Problem::CheckGraphConnectivity already performs.
  std::unordered_map<const float *, std::pair<size_t, size_t>> addr_to_block;
  for (size_t bi = 0; bi < state_batches.size(); ++bi) {
    StateBatch *sb = state_batches[bi];
    const size_t n = sb->NumStateBlocks();
    for (size_t k = 0; k < n; ++k) {
      addr_to_block[sb->StateBlockDevicePtr(k)] = {bi, k};
    }
  }

  BatchPlan plan;
  plan.num_factors = F;
  plan.num_positions = P;
  plan.state_block_sizes = block_sizes;
  plan.col_offsets.resize(P);
  size_t running = 0;
  for (size_t b = 0; b < P; ++b) {
    plan.col_offsets[b] = running;
    running += block_sizes[b];
  }
  plan.owner_batch_index.assign(P, 0);
  plan.block_idx.assign(F * P, 0);

  if (F > 0 && P > 0) {
    for (size_t b = 0; b < P; ++b) {
      auto it = addr_to_block.find(host_ptrs[0 * P + b]);
      if (it == addr_to_block.end()) {
        throw std::runtime_error(
            "NumericDiffJacobianBuilder: factor state pointer is not owned by any registered "
            "StateBatch");
      }
      plan.owner_batch_index[b] = it->second.first;
    }
    for (size_t f = 0; f < F; ++f) {
      for (size_t b = 0; b < P; ++b) {
        auto it = addr_to_block.find(host_ptrs[f * P + b]);
        if (it == addr_to_block.end()) {
          throw std::runtime_error(
              "NumericDiffJacobianBuilder: factor state pointer is not owned by any registered "
              "StateBatch");
        }
        plan.block_idx[f * P + b] = it->second.second;
      }
    }
  }

  plans_[residual_batch_index] = std::move(plan);
  // Structure changed (or is being defined for the first time): any cached
  // slot layout / uploaded device scratch for this residual batch no longer
  // matches, so drop it and let the next Compute() rebuild from scratch.
  caches_.erase(residual_batch_index);
}

void NumericDiffJacobianBuilder::Compute(cudaStream_t stream, const Problem &problem,
                                         size_t residual_batch_index,
                                         const MinimizerState &minimizer_state,
                                         const float *baseline_residuals, float *jacobian_out,
                                         const NumericDiffOptions &options) {
  auto plan_it = plans_.find(residual_batch_index);
  if (plan_it == plans_.end()) {
    PrepareResidualBatch(problem, residual_batch_index);
    plan_it = plans_.find(residual_batch_index);
  }
  const BatchPlan &plan = plan_it->second;

  const auto &rb = problem.GetResidualBatches()[residual_batch_index];
  const FactorBatch *factor_batch = rb.GetFactorBatch();
  const size_t F = plan.num_factors;
  const size_t P = plan.num_positions;
  if (F == 0 || P == 0) return;

  const size_t residual_size = factor_batch->ResidualsSize();
  const size_t total_cols = plan.col_offsets.back() + plan.state_block_sizes.back();

  const auto &state_batches = problem.GetStateBatches();
  const auto &states = minimizer_state.GetStates();

  const bool central = (options.method == NumericDiffOptions::Method::kCentral);

  std::vector<size_t> tangent_size(P), ambient_size(P), num_state_blocks(P);
  size_t W = 0;
  for (size_t b = 0; b < P; ++b) {
    StateBatch *owner = state_batches[plan.owner_batch_index[b]];
    tangent_size[b] = owner->TangentSize();
    ambient_size[b] = owner->AmbientSize();
    num_state_blocks[b] = owner->NumStateBlocks();
    W += tangent_size[b];
  }
  if (W == 0) {
    // No optimizable tangent dof referenced by this factor batch (all
    // referenced blocks are zero-dimensional or constant); nothing to
    // differentiate. Leave jacobian_out untouched (callers should not read
    // it for constant-only groups anyway, mirroring analytic behavior).
    return;
  }

  const size_t S = central ? 2 * W : W;

  ComputeCache &cache = caches_[residual_batch_index];

  // ---- Current address fingerprint: the owning StateBatch data pointer
  // ---- per referenced position. If this matches what was uploaded last
  // ---- time (and the slot layout/options/sizes haven't changed), every
  // ---- host-built array this function would otherwise re-upload is
  // ---- byte-for-byte identical to what's already on the device -- it
  // ---- depends only on structure + options + these addresses, never on
  // ---- the state *values* -- so the rebuild below can be skipped
  // ---- entirely. ----
  std::vector<const float *> owner_data_ptr(P);
  for (size_t b = 0; b < P; ++b) owner_data_ptr[b] = states[plan.owner_batch_index[b]].data();

  bool needs_rebuild = !cache.uploaded || cache.central != central ||
                       cache.step_size != options.relative_step_size || cache.F != F ||
                       cache.P != P || cache.residual_size != residual_size ||
                       cache.total_cols != total_cols ||
                       cache.last_owner_data_ptr != owner_data_ptr;

  if (!needs_rebuild) {
    // x_plus_delta_scratch's own address can only change if it needed to
    // grow, which cannot happen without S/W (and thus the layout above)
    // also changing -- but check defensively anyway, it's one pointer.
    needs_rebuild = (cache.x_plus_delta_scratch.data() != cache.last_xpd_base);
  }

  cache.S = S;
  cache.W = W;
  cache.F = F;
  cache.P = P;
  cache.residual_size = residual_size;
  cache.total_cols = total_cols;

  if (needs_rebuild) {
    // ---- Slot layout: one (position b, tangent dof k, sign) per slot. ----
    cache.slot_b.assign(S, 0);
    cache.slot_k.assign(S, 0);
    cache.slot_owner.assign(S, 0);
    cache.slot_sign.assign(S, 0.0f);
    cache.delta_offset.assign(S, 0);
    cache.xpd_offset.assign(S, 0);
    cache.col_idx_h.assign(W, 0);
    cache.plus_slot_h.assign(W, 0);
    cache.minus_slot_h.assign(W, central ? 0 : -1);
    cache.eps_h.assign(W, options.relative_step_size);

    size_t delta_total = 0, xpd_total = 0, slot = 0, j = 0;
    for (size_t b = 0; b < P; ++b) {
      for (size_t k = 0; k < tangent_size[b]; ++k) {
        auto place_slot = [&](float sign) -> size_t {
          size_t s = slot++;
          cache.slot_b[s] = b;
          cache.slot_k[s] = k;
          cache.slot_owner[s] = plan.owner_batch_index[b];
          cache.slot_sign[s] = sign;
          cache.delta_offset[s] = delta_total;
          delta_total += num_state_blocks[b] * tangent_size[b];
          cache.xpd_offset[s] = xpd_total;
          xpd_total += num_state_blocks[b] * ambient_size[b];
          return s;
        };

        cache.plus_slot_h[j] = static_cast<int>(place_slot(1.0f));
        cache.minus_slot_h[j] = central ? static_cast<int>(place_slot(-1.0f)) : -1;
        cache.col_idx_h[j] = static_cast<int>(plan.col_offsets[b] + k);
        ++j;
      }
    }
    cache.delta_total = delta_total;
    cache.xpd_total = xpd_total;

    cache.delta_scratch.resize(delta_total);
    cache.x_plus_delta_scratch.resize(xpd_total);
    cache.perturbed_residuals.resize(S * F * residual_size);
    cache.state_pointer_scratch.resize(S * F * P);
    cache.col_idx_scratch.resize(W);
    cache.plus_slot_scratch.resize(W);
    cache.minus_slot_scratch.resize(W);
    cache.eps_scratch.resize(W);

    // ---- Build & upload the one-hot tangent deltas for every slot, via a
    // ---- pinned staging buffer (true async DMA, not an internally-staged
    // ---- copy through a driver bounce buffer). ----
    float *delta_host =
        EnsurePinnedHost(cache.pinned_delta_host, cache.pinned_delta_capacity, delta_total);
    std::fill(delta_host, delta_host + delta_total, 0.0f);
    for (size_t s = 0; s < S; ++s) {
      const size_t b = cache.slot_b[s];
      const size_t k = cache.slot_k[s];
      float *delta_block = delta_host + cache.delta_offset[s];
      for (size_t f = 0; f < F; ++f) {
        const size_t blk = plan.block_idx[f * P + b];
        delta_block[blk * tangent_size[b] + k] = cache.slot_sign[s] * options.relative_step_size;
      }
    }
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.delta_scratch.data(), delta_host,
                                        delta_total * sizeof(float), cudaMemcpyHostToDevice,
                                        stream));
    THROW_ON_CUDA_ERROR(cudaEventRecord(delta_ready_event_, stream));

    // ---- Plus() calls, spread across a small stream pool so independent
    // ---- perturbations overlap instead of serializing on one stream.
    // ---- IMPORTANT: several shipped StateBatch::Plus implementations (e.g.
    // ---- SO3StateBatch, SE3StateBatch) reuse `mutable` internal scratch
    // ---- buffers across calls and are therefore not safe to invoke
    // ---- concurrently on the same owner from different streams. Slots are
    // ---- assigned to pool streams by *owner batch index* (not by slot
    // ---- index), so every Plus() call against a given StateBatch lands on
    // ---- the same stream and is naturally serialized in issue order, while
    // ---- distinct owner batches can still overlap on different streams. ----
    constexpr size_t kStreamPoolSize = 8;
    EnsureStreamPool(kStreamPoolSize);
    for (size_t s = 0; s < S; ++s) {
      cudaStream_t ps = pool_streams_[cache.slot_owner[s] % pool_streams_.size()];
      THROW_ON_CUDA_ERROR(cudaStreamWaitEvent(ps, delta_ready_event_, 0));
      StateBatch *owner = state_batches[cache.slot_owner[s]];
      const float *x = states[cache.slot_owner[s]].data();
      const float *delta = cache.delta_scratch.data() + cache.delta_offset[s];
      float *xpd = cache.x_plus_delta_scratch.data() + cache.xpd_offset[s];
      owner->Plus(x, delta, xpd, ps);
    }
    // Join: main stream waits for every pool stream before reading the
    // perturbed buffers they wrote (harmless no-op wait for pool streams
    // that received no work this call).
    for (size_t i = 0; i < pool_streams_.size(); ++i) {
      THROW_ON_CUDA_ERROR(cudaEventRecord(pool_events_[i], pool_streams_[i]));
      THROW_ON_CUDA_ERROR(cudaStreamWaitEvent(stream, pool_events_[i], 0));
    }

    // ---- Build replicated state-pointer table (S * F * P) on the host and
    // ---- upload once (pinned staging). Baseline (unperturbed) positions
    // ---- reuse the current minimizer-state block pointers; the perturbed
    // ---- position for a slot's own (b) reads from that slot's Plus()
    // ---- output. This table is purely address-based (no state values), so
    // ---- it stays valid -- and is never rebuilt -- for as long as those
    // ---- addresses don't move. ----
    std::vector<const float *> baseline_ptr(F * P);
    for (size_t f = 0; f < F; ++f) {
      for (size_t b = 0; b < P; ++b) {
        const size_t owner_idx = plan.owner_batch_index[b];
        const size_t blk = plan.block_idx[f * P + b];
        baseline_ptr[f * P + b] = states[owner_idx].data() + blk * ambient_size[b];
      }
    }
    const float **ptrs_host =
        EnsurePinnedHost(cache.pinned_ptrs_host, cache.pinned_ptrs_capacity, S * F * P);
    for (size_t s = 0; s < S; ++s) {
      const size_t b = cache.slot_b[s];
      const float *xpd_base = cache.x_plus_delta_scratch.data() + cache.xpd_offset[s];
      for (size_t f = 0; f < F; ++f) {
        for (size_t bb = 0; bb < P; ++bb) {
          if (bb == b) {
            const size_t blk = plan.block_idx[f * P + bb];
            ptrs_host[(s * F + f) * P + bb] = xpd_base + blk * ambient_size[b];
          } else {
            ptrs_host[(s * F + f) * P + bb] = baseline_ptr[f * P + bb];
          }
        }
      }
    }
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.state_pointer_scratch.data(), ptrs_host,
                                        S * F * P * sizeof(const float *), cudaMemcpyHostToDevice,
                                        stream));

    // ---- Differencing kernel's static index/epsilon arrays: also
    // ---- structure-only, uploaded once via one merged pinned staging
    // ---- buffer (col_idx | plus_slot | minus_slot back-to-back) instead of
    // ---- three separate transfers. ----
    int *int_host = EnsurePinnedHost(cache.pinned_int_host, cache.pinned_int_capacity, 3 * W);
    std::copy(cache.col_idx_h.begin(), cache.col_idx_h.end(), int_host);
    std::copy(cache.plus_slot_h.begin(), cache.plus_slot_h.end(), int_host + W);
    std::copy(cache.minus_slot_h.begin(), cache.minus_slot_h.end(), int_host + 2 * W);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.col_idx_scratch.data(), int_host, W * sizeof(int),
                                        cudaMemcpyHostToDevice, stream));
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.plus_slot_scratch.data(), int_host + W,
                                        W * sizeof(int), cudaMemcpyHostToDevice, stream));
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.minus_slot_scratch.data(), int_host + 2 * W,
                                        W * sizeof(int), cudaMemcpyHostToDevice, stream));

    float *eps_host = EnsurePinnedHost(cache.pinned_eps_host, cache.pinned_eps_capacity, W);
    std::copy(cache.eps_h.begin(), cache.eps_h.end(), eps_host);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cache.eps_scratch.data(), eps_host, W * sizeof(float),
                                        cudaMemcpyHostToDevice, stream));

    cache.uploaded = true;
    cache.central = central;
    cache.step_size = options.relative_step_size;
    cache.last_owner_data_ptr = owner_data_ptr;
    cache.last_xpd_base = cache.x_plus_delta_scratch.data();
  } else {
    // ---- Fast path: structure, options, and all relevant addresses are
    // ---- unchanged since the last call, so `delta`, the replicated
    // ---- state-pointer table, and the differencing kernel's index/epsilon
    // ---- arrays are already correct on the device -- only the actual
    // ---- perturbed evaluations (which read the *current* state values)
    // ---- need to happen. No H2D copies, no stream-pool synchronization. ----
    for (size_t s = 0; s < S; ++s) {
      StateBatch *owner = state_batches[cache.slot_owner[s]];
      const float *x = states[cache.slot_owner[s]].data();
      const float *delta = cache.delta_scratch.data() + cache.delta_offset[s];
      float *xpd = cache.x_plus_delta_scratch.data() + cache.xpd_offset[s];
      owner->Plus(x, delta, xpd, stream);
    }
  }

  // ---- Residual-only evaluate, once per slot, queued back-to-back on
  // ---- `stream` with no synchronization in between (see the class-level
  // ---- comment in the header for why this cannot be collapsed into a
  // ---- single launch for arbitrary, unmodified shipped factor batches). ----
  for (size_t s = 0; s < S; ++s) {
    float *residuals_out = cache.perturbed_residuals.data() + s * F * residual_size;
    const float *const *ptrs = cache.state_pointer_scratch.data() + s * F * P;
    factor_batch->Evaluate(residuals_out, nullptr, ptrs, stream);
  }

  // ---- Differencing kernel: one launch, fully data-parallel over
  // ---- F * W * ResidualsSize() elements. ----
  const long long total_elements = static_cast<long long>(F) * W * residual_size;
  const int num_blocks = static_cast<int>((total_elements + kBlockSize - 1) / kBlockSize);
  NumericDiffColumnKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      cache.perturbed_residuals.data(), baseline_residuals, cache.col_idx_scratch.data(),
      cache.plus_slot_scratch.data(), cache.minus_slot_scratch.data(), cache.eps_scratch.data(),
      central, static_cast<int>(F), static_cast<int>(W), static_cast<int>(residual_size),
      static_cast<int>(total_cols), jacobian_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
