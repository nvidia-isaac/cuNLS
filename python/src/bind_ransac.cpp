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

// Bindings for the RANSAC minimizers (the RansacMinimizer base with
// RansacGaussNewtonMinimizer and RansacLevenbergMarquardtMinimizer), their
// options and summary.
//
// Usage mirrors the regular minimizers: build an ordinary Problem, construct a
// minimizer from its options, call minimize(stream, problem). The estimate is
// written back into the problem's state batches; inlier_mask(i) returns the
// classification of residual batch i as a numpy uint8 array.

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"

namespace {

using cunls::LogError;  // used by THROW_ON_CUDA_ERROR
using cunls::RansacFactorBatchOptions;
using cunls::RansacGaussNewtonMinimizer;
using cunls::RansacLevenbergMarquardtMinimizerOptions;
using cunls::RansacMinimizer;
using cunls::RansacMinimizerOptions;
using cunls::RansacSummary;

using MaskArray = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>;

/**
 * Copies the device inlier mask of residual batch `index` into a numpy array,
 * sized by the factor count recorded in the last minimize().
 */
MaskArray InlierMask(const RansacMinimizer &self, size_t index) {
  const uint8_t *device = self.InlierMask(index);
  if (device == nullptr) {
    throw std::runtime_error(
        "inlier_mask: no mask for residual batch " + std::to_string(index) +
        " (index out of range, an always_on batch, or minimize() has not run)");
  }
  const size_t n = self.InlierMaskSize(index);
  auto *host = new uint8_t[n];
  nb::capsule owner(host, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemcpy(host, device, n, cudaMemcpyDeviceToHost));
  }
  return MaskArray(host, {n}, owner);
}

RansacSummary Minimize(RansacMinimizer &self, cunls::CudaStream &stream, cunls::Problem &problem) {
  nb::gil_scoped_release release;  // custom Python factors re-acquire it
  return self.Minimize(stream.GetStream(), problem);
}

void BindOptions(nb::module_ &m) {
  nb::enum_<cunls::RansacRole>(m, "RansacRole", "Role of a residual batch in RANSAC.")
      .value("sampled", cunls::RansacRole::kSampled,
             "Data factors that may be outliers: sampled and classified.")
      .value("always_on", cunls::RansacRole::kAlwaysOn,
             "Trusted factors (priors): in every solve, never classified.");

  nb::enum_<cunls::RansacScoring>(m, "RansacScoring", "Hypothesis scoring rule.")
      .value("msac", cunls::RansacScoring::kMSAC, "Sum of min(|r|^2, tau^2).")
      .value("inlier_count", cunls::RansacScoring::kInlierCount,
             "Number of inliers (ties broken by MSAC).");

  nb::enum_<cunls::RansacLinearSolverType>(m, "RansacLinearSolverType",
                                           "Dense per-hypothesis solver.")
      .value("cholesky", cunls::RansacLinearSolverType::kCholesky)
      .value("ldlt", cunls::RansacLinearSolverType::kLDLT);

  nb::class_<RansacFactorBatchOptions>(m, "RansacFactorBatchOptions",
                                       "RANSAC configuration of one residual batch.")
      .def(
          "__init__",
          [](RansacFactorBatchOptions *self, cunls::RansacRole role, float inlier_threshold) {
            new (self) RansacFactorBatchOptions{role, inlier_threshold};
          },
          nb::arg("role") = cunls::RansacRole::kSampled, nb::arg("inlier_threshold") = 1.0f)
      .def_rw("role", &RansacFactorBatchOptions::role)
      .def_rw("inlier_threshold", &RansacFactorBatchOptions::inlier_threshold,
              "Inlier iff |r| <= inlier_threshold (raw residual norm).");

  nb::class_<RansacMinimizerOptions>(m, "RansacMinimizerOptions",
                                     "Options common to all RANSAC minimizers.")
      .def(nb::init<>())
      .def_rw("hypotheses_per_round", &RansacMinimizerOptions::hypotheses_per_round)
      .def_rw("max_rounds", &RansacMinimizerOptions::max_rounds)
      .def_rw("sample_size", &RansacMinimizerOptions::sample_size,
              "Factors per minimal sample; 0 = ceil(D / m_min).")
      .def_rw("confidence", &RansacMinimizerOptions::confidence)
      .def_rw("early_stop_inlier_ratio", &RansacMinimizerOptions::early_stop_inlier_ratio)
      .def_rw("seed", &RansacMinimizerOptions::seed)
      .def_rw("factor_batches", &RansacMinimizerOptions::factor_batches,
              "List of RansacFactorBatchOptions, one per residual batch (in the order the "
              "batches were added). Assign a whole list: appending to the returned copy has "
              "no effect. Empty = every batch sampled with default_inlier_threshold.")
      .def_rw("default_inlier_threshold", &RansacMinimizerOptions::default_inlier_threshold)
      .def_rw("scoring", &RansacMinimizerOptions::scoring)
      .def_rw("score_always_on", &RansacMinimizerOptions::score_always_on)
      .def_rw("require_informative_inliers", &RansacMinimizerOptions::require_informative_inliers)
      .def_rw("scoring_memory_budget_bytes", &RansacMinimizerOptions::scoring_memory_budget_bytes)
      .def_rw("scoring_subset_size", &RansacMinimizerOptions::scoring_subset_size)
      .def_rw("scoring_finalists", &RansacMinimizerOptions::scoring_finalists)
      .def_rw("hypothesis_iterations", &RansacMinimizerOptions::hypothesis_iterations)
      .def_rw("final_iterations", &RansacMinimizerOptions::final_iterations)
      .def_rw("state_tolerance", &RansacMinimizerOptions::state_tolerance)
      .def_rw("cost_tolerance", &RansacMinimizerOptions::cost_tolerance)
      .def_rw("linear_solver", &RansacMinimizerOptions::linear_solver);

  nb::class_<RansacLevenbergMarquardtMinimizerOptions>(
      m, "RansacLevenbergMarquardtMinimizerOptions",
      "Options of RansacLevenbergMarquardtMinimizer.")
      .def(nb::init<>())
      .def_rw("base_options", &RansacLevenbergMarquardtMinimizerOptions::base_options)
      .def_rw("initial_lambda", &RansacLevenbergMarquardtMinimizerOptions::initial_lambda)
      .def_rw("lambda_upscale", &RansacLevenbergMarquardtMinimizerOptions::lambda_upscale)
      .def_rw("lambda_downscale", &RansacLevenbergMarquardtMinimizerOptions::lambda_downscale)
      .def_rw("lambda_max", &RansacLevenbergMarquardtMinimizerOptions::lambda_max)
      .def_rw("lambda_min", &RansacLevenbergMarquardtMinimizerOptions::lambda_min)
      .def_rw("step_accept_threshold",
              &RansacLevenbergMarquardtMinimizerOptions::step_accept_threshold)
      .def_rw("lambda_downscale_threshold",
              &RansacLevenbergMarquardtMinimizerOptions::lambda_downscale_threshold);

  nb::class_<RansacSummary, cunls::MinimizerSummary>(m, "RansacSummary",
                                                     "Result of a RANSAC minimization.")
      .def_ro("num_rounds", &RansacSummary::num_rounds)
      .def_ro("num_hypotheses", &RansacSummary::num_hypotheses)
      .def_ro("num_valid_hypotheses", &RansacSummary::num_valid_hypotheses)
      .def_ro("num_inliers", &RansacSummary::num_inliers)
      .def_ro("inlier_ratio", &RansacSummary::inlier_ratio)
      .def_ro("best_score", &RansacSummary::best_score)
      .def_ro("refinement_reverted", &RansacSummary::refinement_reverted)
      .def("__repr__", [](const RansacSummary &s) {
        return "RansacSummary(rounds=" + std::to_string(s.num_rounds) +
               ", inliers=" + std::to_string(s.num_inliers) +
               ", inlier_ratio=" + std::to_string(s.inlier_ratio) +
               ", final_cost=" + std::to_string(s.final_cost) + ")";
      });
}

}  // namespace

void bind_ransac(nb::module_ &m) {
  BindOptions(m);

  nb::class_<RansacMinimizer>(
      m, "RansacMinimizer",
      "Common base of RansacGaussNewtonMinimizer and RansacLevenbergMarquardtMinimizer: "
      "RANSAC over an ordinary Problem. Not constructible; accepts either. The total free "
      "tangent dimension must be <= 64.")
      .def("minimize", &Minimize, nb::arg("stream"), nb::arg("problem"),
           "Run RANSAC; the estimate is written into the problem's state batches. "
           "Returns a RansacSummary.")
      .def("inlier_mask", &InlierMask, nb::arg("residual_batch_index"),
           "Inlier mask (numpy uint8, 1 = inlier) of a sampled residual batch, for the "
           "problem of the last minimize(). Raises RuntimeError for an out-of-range index, "
           "an always_on batch, or before any run.")
      .def_prop_ro(
          "options", [](const RansacMinimizer &self) { return self.Options(); },
          "Options common to all RANSAC minimizers, as constructed (a copy).");

  nb::class_<RansacGaussNewtonMinimizer, RansacMinimizer>(
      m, "RansacGaussNewtonMinimizer",
      "RANSAC with Gauss-Newton hypotheses and refinement: a step is taken if it lowers the "
      "cost; a hypothesis stops at the first step that does not.")
      .def(nb::init<const RansacMinimizerOptions &>(),
           nb::arg("options") = RansacMinimizerOptions());

  nb::class_<cunls::RansacLevenbergMarquardtMinimizer, RansacMinimizer>(
      m, "RansacLevenbergMarquardtMinimizer",
      "RANSAC with Levenberg-Marquardt hypotheses and refinement; each hypothesis carries its "
      "own damping.")
      .def(nb::init<const RansacLevenbergMarquardtMinimizerOptions &>(),
           nb::arg("options") = RansacLevenbergMarquardtMinimizerOptions());
}
