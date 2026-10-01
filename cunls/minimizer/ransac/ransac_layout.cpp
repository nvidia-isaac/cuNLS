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

#include "cunls/minimizer/ransac/ransac_layout.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"

namespace cunls {
namespace ransac_internal {

namespace {

/** Host copy of a state batch's constant flags. */
std::vector<int> ConstantFlags(const StateBatch &batch, int num_blocks) {
  std::vector<int> flags(num_blocks, 0);
  const size_t count = batch.NumConstStateBlocks();
  if (count == 0 || batch.ConstStateIds() == nullptr) {
    return flags;
  }
  std::vector<int> ids(count);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(ids.data(), batch.ConstStateIds(), count * sizeof(int), cudaMemcpyDeviceToHost));
  for (int id : ids) {
    if (id >= 0 && id < num_blocks) {
      flags[id] = 1;
    }
  }
  return flags;
}

std::string Str(size_t v) { return std::to_string(v); }

}  // namespace

void FailConfiguration(const std::string &message) {
  LogError("{}", message);
  throw std::invalid_argument(message);
}

void RansacLayout::Build(const Problem &problem, const RansacMinimizerOptions &options) {
  if (!problem.CheckConsistency()) {
    FailConfiguration(
        "RANSAC: Problem::CheckConsistency() failed; fix the problem before calling Minimize");
  }
  BuildStates(problem);
  BuildResiduals(problem, options);
}

void RansacLayout::BuildStates(const Problem &problem) {
  const auto &batches = problem.GetStateBatches();
  states_.resize(batches.size());  // resized, not cleared: device buffers keep capacity
  dim_ = 0;
  std::ostringstream breakdown;
  for (size_t j = 0; j < batches.size(); ++j) {
    const int free_blocks = BuildState(j, batches[j]);
    if (free_blocks > 0) {
      breakdown << "\n  state batch " << j << ": " << free_blocks << " free block(s) x tangent "
                << states_[j].tangent << " = " << free_blocks * states_[j].tangent;
    }
  }
  if (dim_ == 0) {
    FailConfiguration(
        "RANSAC: the problem has no free state blocks (free tangent dimension D = 0)");
  }
  if (dim_ > kMaxRansacTangentDim) {
    FailConfiguration("RANSAC: free tangent dimension D = " + Str(dim_) + " exceeds " +
                      Str(kMaxRansacTangentDim) + "; breakdown:" + breakdown.str() +
                      "\nMark states constant, or use GaussNewtonMinimizer / "
                      "LevenbergMarquardtMinimizer with a robust loss for large problems.");
  }
}

int RansacLayout::BuildState(size_t index, StateBatch *batch) {
  StateLayout &s = states_[index];
  s.batch = batch;
  s.num_blocks = static_cast<int>(batch->NumStateBlocks());
  s.ambient = static_cast<int>(batch->AmbientSize());
  s.tangent = static_cast<int>(batch->TangentSize());
  s.slot_floats = static_cast<size_t>(s.num_blocks) * s.ambient;
  s.slot_tangent = static_cast<size_t>(s.num_blocks) * s.tangent;
  const std::vector<int> constant = ConstantFlags(*batch, s.num_blocks);
  s.block_col_host.assign(s.num_blocks, -1);
  int free_blocks = 0;
  for (int k = 0; k < s.num_blocks; ++k) {
    if (!constant[k]) {
      s.block_col_host[k] = dim_;
      dim_ += s.tangent;
      ++free_blocks;
    }
  }
  s.replicated = free_blocks > 0;
  s.block_col.resize(s.num_blocks);
  s.block_col.CopyFromHost(s.block_col_host.data(), s.num_blocks);
  if (free_blocks > 0 && free_blocks * 2 < s.num_blocks) {
    LogMessage(
        "RANSAC: state batch {} is copied per hypothesis but only {} of its {} blocks "
        "are free; keep constant blocks in their own state batch to save memory",
        index, free_blocks, s.num_blocks);
  }
  return free_blocks;
}

void RansacLayout::BuildResiduals(const Problem &problem, const RansacMinimizerOptions &options) {
  const auto &batches = problem.GetResidualBatches();
  if (!options.factor_batches.empty() && options.factor_batches.size() != batches.size()) {
    FailConfiguration("RANSAC: options.factor_batches has " + Str(options.factor_batches.size()) +
                      " entries but the problem has " + Str(batches.size()) +
                      " residual batches (leave it empty to sample every batch)");
  }
  residuals_.resize(batches.size());
  total_sampled_ = 0;
  m_max_ = 1;
  max_factors_ = 1;
  int m_min_sampled = 1 << 30;
  for (size_t b = 0; b < batches.size(); ++b) {
    const RansacFactorBatchOptions role =
        options.factor_batches.empty()
            ? RansacFactorBatchOptions{RansacRole::kSampled, options.default_inlier_threshold}
            : options.factor_batches[b];
    BuildResidual(problem, b, role);
    const ResidualLayout &r = residuals_[b];
    if (r.sampled && r.num_factors > 0) {
      m_min_sampled = std::min(m_min_sampled, r.m);
    }
  }
  if (total_sampled_ == 0) {
    FailConfiguration(
        "RANSAC: no kSampled factors; at least one residual batch must be kSampled and non-empty");
  }
  // The normal-equation kernel must stage at least one item in a block group.
  if (NormalEquationsItemWords(m_max_, dim_) > kBlockGroupWords) {
    FailConfiguration("RANSAC: residual dimension " + Str(m_max_) +
                      " is too large for free tangent dimension " + Str(dim_));
  }
  ChooseSampleSize(options, m_min_sampled);
}

void RansacLayout::BuildResidual(const Problem &problem, size_t index,
                                 const RansacFactorBatchOptions &role) {
  ResidualLayout &r = residuals_[index];
  r.residual_batch = &problem.GetResidualBatches()[index];
  r.factor = r.residual_batch->GetFactorBatch();
  if (problem.JacobianModeFor(index, JacobianMode::kAnalytic) != JacobianMode::kAnalytic) {
    FailConfiguration(
        "RANSAC: residual batch " + Str(index) +
        " requests numeric Jacobians, which the RANSAC minimizers do not support yet");
  }
  r.sampled = role.role == RansacRole::kSampled;
  r.tau = role.inlier_threshold;
  if (r.sampled && !(r.tau > 0.f)) {
    FailConfiguration("RANSAC: inlier_threshold of residual batch " + Str(index) + " must be > 0");
  }
  r.m = static_cast<int>(r.factor->ResidualsSize());
  r.num_factors = static_cast<int>(r.factor->NumFactors());
  const auto sizes = r.factor->StateBlockSizes();
  r.nb = static_cast<int>(sizes.size());
  if (r.nb > kMaxBlocksPerFactor) {
    FailConfiguration("RANSAC: residual batch " + Str(index) + " references " + Str(r.nb) +
                      " state blocks per factor; at most " + Str(kMaxBlocksPerFactor) +
                      " are supported");
  }
  r.block_off.assign(r.nb, 0);
  r.block_size.assign(r.nb, 0);
  r.n = 0;
  for (int k = 0; k < r.nb; ++k) {
    r.block_off[k] = r.n;
    r.block_size[k] = static_cast<int>(sizes[k]);
    r.n += r.block_size[k];
  }
  m_max_ = std::max(m_max_, r.m);
  max_factors_ = std::max(max_factors_, r.num_factors);
  r.u_offset = r.sampled ? total_sampled_ : -1;
  total_sampled_ += r.sampled ? r.num_factors : 0;
  ResolveBlocks(r, index, problem.HostStatePointers(index));
}

void RansacLayout::ResolveBlocks(ResidualLayout &r, size_t index,
                                 const std::vector<float *> &pointers) {
  std::vector<int2> blocks(pointers.size());
  std::vector<int> local(pointers.size(), -1);
  for (size_t e = 0; e < pointers.size(); ++e) {
    const int slot_block = static_cast<int>(e % r.nb);
    bool found = false;
    for (size_t j = 0; j < states_.size() && !found; ++j) {
      const StateLayout &s = states_[j];
      const ptrdiff_t diff = pointers[e] - s.batch->StateBlockDevicePtr(0);
      if (diff < 0 || diff >= static_cast<ptrdiff_t>(s.slot_floats) || diff % s.ambient != 0) {
        continue;
      }
      if (s.tangent != r.block_size[slot_block]) {
        FailConfiguration("RANSAC: residual batch " + Str(index) + " block " + Str(slot_block) +
                          " has size " + Str(r.block_size[slot_block]) + " but state batch " +
                          Str(j) + " has tangent size " + Str(s.tangent));
      }
      const int k = static_cast<int>(diff / s.ambient);
      blocks[e] = make_int2(static_cast<int>(j), k);
      local[e] = s.block_col_host[k];
      found = true;
    }
    if (!found) {
      FailConfiguration("RANSAC: residual batch " + Str(index) +
                        " references a state block that belongs to no registered state batch");
    }
  }
  r.blocks.resize(blocks.size());
  r.blocks.CopyFromHost(blocks.data(), blocks.size());
  r.local_col.resize(local.size());
  r.local_col.CopyFromHost(local.data(), local.size());
  r.x0_table.resize(pointers.size());
  r.x0_table.CopyFromHost(const_cast<float *const *>(pointers.data()), pointers.size());
}

void RansacLayout::ChooseSampleSize(const RansacMinimizerOptions &options, int m_min_sampled) {
  sample_size_ = options.sample_size > 0 ? static_cast<int>(options.sample_size)
                                         : std::max(1, (dim_ + m_min_sampled - 1) / m_min_sampled);
  if (sample_size_ > total_sampled_) {
    FailConfiguration("RANSAC: sample size " + Str(sample_size_) + " exceeds the " +
                      Str(total_sampled_) + " kSampled factors");
  }
}

}  // namespace ransac_internal
}  // namespace cunls
