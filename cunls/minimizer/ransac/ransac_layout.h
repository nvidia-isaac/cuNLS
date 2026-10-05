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
 * @file ransac_layout.h
 * @brief What the RANSAC minimizers need to know about a Problem: which
 * states are free and where their columns go, how every factor slot maps to
 * (state batch, state), and which residual batches are sampled.
 */

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

struct RansacFactorBatchOptions;
struct RansacMinimizerOptions;
namespace ransac_internal {

/** @brief One state batch. */
struct StateLayout {
  StateBatch *batch = nullptr;
  int num_blocks = 0;
  int ambient = 0;
  int tangent = 0;
  bool replicated = false;          ///< Has a free state, so every slot gets a copy.
  size_t slot_floats = 0;           ///< num_blocks * ambient: one replica.
  size_t slot_tangent = 0;          ///< num_blocks * tangent: one replica's step.
  std::vector<int> block_col_host;  ///< Local column of each state, -1 if constant.
  dvector<int> block_col;           ///< Device copy of block_col_host.
};

/** @brief One residual batch. */
struct ResidualLayout {
  const ResidualBatch *residual_batch = nullptr;
  FactorBatch *factor = nullptr;
  bool sampled = false;  ///< RansacRole::kSampled.
  float tau = 0.f;       ///< Inlier threshold (sampled only).
  int m = 0;             ///< Residual dimension.
  int n = 0;             ///< Sum of state tangent sizes.
  int nb = 0;            ///< States per factor.
  int num_factors = 0;
  int u_offset = -1;  ///< Offset in the concatenated sampled index (sampled only).
  std::vector<int> block_off;
  std::vector<int> block_size;
  dvector<int2> blocks;       ///< (state batch, state) per (factor, state slot).
  dvector<int> local_col;     ///< Local column per (factor, state slot), -1 if constant.
  dvector<float *> x0_table;  ///< The problem's own pointer list (initial guess).
};

/**
 * @brief Validated description of a Problem for RANSAC.
 *
 * Build() throws std::invalid_argument with an actionable message for every
 * unsupported configuration. Rebuilding keeps device capacity.
 */
class RansacLayout {
 public:
  void Build(const Problem &problem, const RansacMinimizerOptions &options);

  const std::vector<StateLayout> &states() const { return states_; }
  const std::vector<ResidualLayout> &residuals() const { return residuals_; }
  int dim() const { return dim_; }                      ///< Free tangent dimension D.
  int total_sampled() const { return total_sampled_; }  ///< Sampled factors, all batches.
  int sample_size() const { return sample_size_; }      ///< Factors per minimal sample.
  int m_max() const { return m_max_; }                  ///< Largest residual dimension.
  int max_factors() const { return max_factors_; }      ///< Largest batch size.

 private:
  void BuildStates(const Problem &problem);
  /** Fills states_[index]; returns its number of free states. */
  int BuildState(size_t index, StateBatch *batch);
  void BuildResiduals(const Problem &problem, const RansacMinimizerOptions &options);
  void BuildResidual(const Problem &problem, size_t index, const RansacFactorBatchOptions &role);
  void ResolveBlocks(ResidualLayout &r, size_t index, const std::vector<float *> &pointers);
  void ChooseSampleSize(const RansacMinimizerOptions &options, int m_min_sampled);

  std::vector<StateLayout> states_;
  std::vector<ResidualLayout> residuals_;
  int dim_ = 0;
  int total_sampled_ = 0;
  int sample_size_ = 0;
  int m_max_ = 1;
  int max_factors_ = 1;
};

/** @brief Throws std::invalid_argument(message) after logging it. */
[[noreturn]] void FailConfiguration(const std::string &message);

}  // namespace ransac_internal
}  // namespace cunls
