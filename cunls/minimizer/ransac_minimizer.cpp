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

#include "cunls/minimizer/ransac_minimizer.h"

#include "cunls/minimizer/ransac/ransac_context.h"

namespace cunls {

namespace {

ransac_internal::SolverSettings MakeSettings(const RansacMinimizerOptions &options,
                                             bool levenberg_marquardt, float initial_lambda,
                                             float up, float down, float lambda_max,
                                             float lambda_min, float accept, float down_threshold) {
  ransac_internal::SolverSettings s;
  s.policy.levenberg_marquardt = levenberg_marquardt ? 1 : 0;
  s.policy.lambda_upscale = up;
  s.policy.lambda_downscale = down;
  s.policy.lambda_max = lambda_max;
  s.policy.lambda_min = lambda_min;
  s.policy.step_accept_threshold = accept;
  s.policy.lambda_downscale_threshold = down_threshold;
  s.policy.cost_tolerance = options.cost_tolerance;
  s.policy.state_tolerance = options.state_tolerance;
  s.initial_lambda = initial_lambda;
  s.solver = options.linear_solver == RansacLinearSolverType::kCholesky
                 ? ransac_internal::kSolveCholesky
                 : ransac_internal::kSolveLDLT;
  return s;
}

}  // namespace

RansacGaussNewtonMinimizer::RansacGaussNewtonMinimizer(const RansacMinimizerOptions &options)
    : RansacGaussNewtonMinimizer(options, StepPolicy{}) {}

RansacGaussNewtonMinimizer::RansacGaussNewtonMinimizer(const RansacMinimizerOptions &options,
                                                       const StepPolicy &p)
    : context_(std::make_unique<ransac_internal::RansacContext>(
          options, MakeSettings(options, p.levenberg_marquardt, p.initial_lambda, p.lambda_upscale,
                                p.lambda_downscale, p.lambda_max, p.lambda_min,
                                p.step_accept_threshold, p.lambda_downscale_threshold))) {}

RansacGaussNewtonMinimizer::~RansacGaussNewtonMinimizer() = default;

RansacSummary RansacGaussNewtonMinimizer::Minimize(cudaStream_t stream, Problem &problem) {
  return context_->Minimize(stream, problem);
}

const uint8_t *RansacGaussNewtonMinimizer::InlierMask(size_t residual_batch_index) const {
  return context_->InlierMask(residual_batch_index);
}

size_t RansacGaussNewtonMinimizer::InlierMaskSize(size_t residual_batch_index) const {
  return context_->InlierMaskSize(residual_batch_index);
}

RansacLevenbergMarquardtMinimizer::RansacLevenbergMarquardtMinimizer(
    const RansacLevenbergMarquardtMinimizerOptions &o)
    : RansacGaussNewtonMinimizer(
          o.base_options,
          StepPolicy{true, o.initial_lambda, o.lambda_upscale, o.lambda_downscale, o.lambda_max,
                     o.lambda_min, o.step_accept_threshold, o.lambda_downscale_threshold}) {}

}  // namespace cunls
