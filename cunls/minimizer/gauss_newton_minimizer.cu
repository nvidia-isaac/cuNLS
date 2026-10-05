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

#include "cunls/common/helper.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;

/** The rule of the class comment, per active subproblem. */
__global__ void classify_kernel(size_t num_problems, const float *cost, const float *new_cost,
                                const float *step_squared, const int *active, float state_tolerance,
                                float cost_tolerance, int *reject, int *converged) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems || !active[p]) return;
  const float quality = new_cost[p] / cost[p];
  // Written so that a NaN quality (non-finite cost) rejects.
  const bool lowered = quality < 1.f;
  reject[p] = !lowered;
  converged[p] = step_squared[p] < state_tolerance || new_cost[p] < cost_tolerance || !lowered;
}

}  // namespace

GaussNewtonMinimizer::GaussNewtonMinimizer(const MinimizerOptions &options) : Minimizer(options) {}

void GaussNewtonMinimizer::ClassifySteps(cudaStream_t stream, const NormalEquations &,
                                         const dvector<float> &, ProblemPartition &partition) {
  const size_t n = partition.NumProblems();
  classify_kernel<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      n, partition.Cost(), partition.NewCost(), partition.StepSquared(), partition.Active(),
      Options().state_tolerance, Options().cost_tolerance, partition.Reject(),
      partition.Converged());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
