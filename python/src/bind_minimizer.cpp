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

// Bindings for the Gauss-Newton and Levenberg-Marquardt minimizers.
//
// Both minimizers follow the same pattern:
//   1. Construct with an options struct (MinimizerOptions or LM-specific).
//   2. Call minimize(stream, problem) -> MinimizerSummary.
//
// The minimize() wrapper releases the GIL before entering the C++ solver
// so that other Python threads (or async tasks) are not blocked during what
// can be a long-running GPU computation.  If the Problem contains a
// CustomFactorBatch whose evaluate() calls back into Python, the trampoline
// in PyFactorBatch re-acquires the GIL for the duration of that callback.

#include "bindings.h"

#include "cunls/common/cuda_stream.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"

void bind_minimizer(nb::module_ &m) {
  nb::class_<cunls::GaussNewtonMinimizer>(
      m, "GaussNewtonMinimizer",
      "Gauss-Newton minimizer for nonlinear least-squares problems.")
      .def(nb::init<const cunls::MinimizerOptions &>(),
           nb::arg("options") = cunls::MinimizerOptions())
      .def(
          "minimize",
          [](cunls::GaussNewtonMinimizer &self, cunls::CudaStream &stream,
             cunls::Problem &problem) {
            cunls::MinimizerSummary summary;
            {
              nb::gil_scoped_release release;
              summary = self.Minimize(stream.GetStream(), problem);
            }
            return summary;
          },
          nb::arg("stream"), nb::arg("problem"),
          "Run the Gauss-Newton optimizer. Returns a MinimizerSummary.");

  nb::class_<cunls::LevenbergMarquardtMinimizer, cunls::GaussNewtonMinimizer>(
      m, "LevenbergMarquardtMinimizer",
      "Levenberg-Marquardt minimizer (damped Gauss-Newton).")
      .def(nb::init<const cunls::LevenbergMarquardtMinimizerOptions &>(),
           nb::arg("options") = cunls::LevenbergMarquardtMinimizerOptions())
      .def(
          "minimize",
          [](cunls::LevenbergMarquardtMinimizer &self,
             cunls::CudaStream &stream, cunls::Problem &problem) {
            cunls::MinimizerSummary summary;
            {
              nb::gil_scoped_release release;
              summary = self.Minimize(stream.GetStream(), problem);
            }
            return summary;
          },
          nb::arg("stream"), nb::arg("problem"),
          "Run the Levenberg-Marquardt optimizer. Returns a MinimizerSummary.");
}
