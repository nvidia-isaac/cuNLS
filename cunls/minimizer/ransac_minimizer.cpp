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

ransac_internal::SolverKind Solver(const RansacMinimizerOptions &options) {
  return options.linear_solver == RansacLinearSolverType::kCholesky
             ? ransac_internal::kSolveCholesky
             : ransac_internal::kSolveLDLT;
}

/** Gauss-Newton hypotheses: the convergence tolerances only. */
ransac_internal::SolverSettings GaussNewtonSettings(const RansacMinimizerOptions &options) {
  ransac_internal::SolverSettings s;
  s.policy.cost_tolerance = options.cost_tolerance;
  s.policy.state_tolerance = options.state_tolerance;
  s.solver = Solver(options);
  return s;
}

/** Levenberg-Marquardt hypotheses: the tolerances and the damping rule. */
ransac_internal::SolverSettings LevenbergMarquardtSettings(
    const RansacLevenbergMarquardtMinimizerOptions &o) {
  ransac_internal::SolverSettings s = GaussNewtonSettings(o.base_options);
  s.policy.levenberg_marquardt = 1;
  s.policy.lambda_upscale = o.lambda_upscale;
  s.policy.lambda_downscale = o.lambda_downscale;
  s.policy.lambda_max = o.lambda_max;
  s.policy.lambda_min = o.lambda_min;
  s.policy.step_accept_threshold = o.step_accept_threshold;
  s.policy.lambda_downscale_threshold = o.lambda_downscale_threshold;
  s.initial_lambda = o.initial_lambda;
  return s;
}

}  // namespace

RansacMinimizer::RansacMinimizer(const RansacMinimizerOptions &options,
                                 const ransac_internal::SolverSettings &settings)
    : options_(options), settings_(settings) {}

RansacSummary RansacMinimizer::Minimize(cudaStream_t stream, Problem &problem) {
  return ransac_internal::RunRansac(stream, problem, options_, settings_, context_);
}

const uint8_t *RansacMinimizer::InlierMask(size_t residual_batch_index) const {
  return ransac_internal::InlierMask(context_, residual_batch_index);
}

size_t RansacMinimizer::InlierMaskSize(size_t residual_batch_index) const {
  return ransac_internal::InlierMaskSize(context_, residual_batch_index);
}

RansacGaussNewtonMinimizer::RansacGaussNewtonMinimizer(const RansacMinimizerOptions &options)
    : RansacMinimizer(options, GaussNewtonSettings(options)) {}

RansacLevenbergMarquardtMinimizer::RansacLevenbergMarquardtMinimizer(
    const RansacLevenbergMarquardtMinimizerOptions &options)
    : RansacMinimizer(options.base_options, LevenbergMarquardtSettings(options)) {}

}  // namespace cunls
