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

// Bindings for the minimizers: the Minimizer base with GaussNewtonMinimizer
// and LevenbergMarquardtMinimizer, and the augmented Lagrangian around them.
//
// Both minimizers follow the same pattern:
//   1. Construct with an options struct (MinimizerOptions or LM-specific).
//   2. Call minimize(stream, problem) -> MinimizerSummary (on the base).
//
// The minimize() wrapper releases the GIL before entering the C++ solver
// so that other Python threads (or async tasks) are not blocked during what
// can be a long-running GPU computation.  If the Problem contains a
// CustomFactorBatch whose evaluate() calls back into Python, the trampoline
// in PyFactorBatch re-acquires the GIL for the duration of that callback.

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/problem.h"

using cunls::LogError;  // used by THROW_ON_CUDA_ERROR

void bind_minimizer(nb::module_ &m) {
  nb::class_<cunls::Minimizer>(
      m, "Minimizer",
      "Common base of GaussNewtonMinimizer and LevenbergMarquardtMinimizer. Not "
      "constructible; accepts either (e.g. AugmentedLagrangianMinimizer's inner minimizer).")
      .def(
          "minimize",
          [](cunls::Minimizer &self, cunls::CudaStream &stream, cunls::Problem &problem) {
            cunls::MinimizerSummary summary;
            {
              nb::gil_scoped_release release;
              summary = self.Minimize(stream.GetStream(), problem);
            }
            return summary;
          },
          nb::arg("stream"), nb::arg("problem"),
          "Minimize the problem's cost from its current states; the states are updated in "
          "place. Returns a MinimizerSummary.")
      .def(
          "subproblem_termination_reasons",
          [](const cunls::Minimizer &self) {
            const size_t n = self.NumSubproblems();
            auto *host = new int[n > 0 ? n : 1];
            nb::capsule owner(host, [](void *p) noexcept { delete[] static_cast<int *>(p); });
            if (n > 0 && self.SubproblemTerminationReasons() != nullptr) {
              THROW_ON_CUDA_ERROR(cudaMemcpy(host, self.SubproblemTerminationReasons(),
                                             n * sizeof(int), cudaMemcpyDeviceToHost));
            }
            return nb::ndarray<nb::numpy, int, nb::ndim<1>>(host, {n}, owner);
          },
          "The last minimize() call's TerminationReason value of every subproblem "
          "(Problem.set_problem_partition), as a numpy int array: e.g. "
          "np.flatnonzero(r >= int(TerminationReason.MaxIterations)) lists the subproblems "
          "that did not converge.")
      .def_prop_ro(
          "options", [](const cunls::Minimizer &self) { return self.Options(); },
          "Options the minimizer runs with (a copy; for Levenberg-Marquardt the base options "
          "with max_consecutive_rejected_steps widened by the damping's escalation room).");

  nb::class_<cunls::GaussNewtonMinimizer, cunls::Minimizer>(
      m, "GaussNewtonMinimizer",
      "Gauss-Newton: solves the undamped normal equations and takes every step that lowers "
      "the cost. Needs every gauge fixed (a singular system fails); prefer "
      "LevenbergMarquardtMinimizer unless the problem is well constrained. Construct with "
      "options: MinimizerOptions.")
      .def(nb::init<const cunls::MinimizerOptions &>(),
           nb::arg("options") = cunls::MinimizerOptions());

  nb::class_<cunls::LevenbergMarquardtMinimizer, cunls::Minimizer>(
      m, "LevenbergMarquardtMinimizer",
      "Levenberg-Marquardt: Gauss-Newton with an adaptive damping per subproblem; the robust "
      "default. Construct with options: LevenbergMarquardtMinimizerOptions (its base_options "
      "holds the common MinimizerOptions).")
      .def(nb::init<const cunls::LevenbergMarquardtMinimizerOptions &>(),
           nb::arg("options") = cunls::LevenbergMarquardtMinimizerOptions());

  // --- Augmented Lagrangian outer loop for problems with constraint batches ---
  nb::class_<cunls::AugmentedLagrangianMinimizerOptions>(
      m, "AugmentedLagrangianMinimizerOptions", "Options of the augmented Lagrangian outer loop.")
      .def(nb::init<>())
      .def_rw("constraint_tolerance",
              &cunls::AugmentedLagrangianMinimizerOptions::constraint_tolerance,
              "Feasibility: |c| <= tol (equalities), c <= tol (inequalities). Default: 1e-4.")
      .def_rw("max_outer_iterations",
              &cunls::AugmentedLagrangianMinimizerOptions::max_outer_iterations,
              "Maximum number of multiplier updates. Default: 20.")
      .def_rw("inner_iterations", &cunls::AugmentedLagrangianMinimizerOptions::inner_iterations,
              "Iteration cap of the inner solves while infeasible. Default: 5.")
      .def_rw("final_inner_iterations",
              &cunls::AugmentedLagrangianMinimizerOptions::final_inner_iterations,
              "Iteration cap of the inner solves once feasible; 0: the inner minimizer's "
              "max_num_iterations. Default: 0.")
      .def_rw("inner_line_search_steps",
              &cunls::AugmentedLagrangianMinimizerOptions::inner_line_search_steps,
              "Line-search steps of the inner solves. Default: 10.")
      .def_rw("initial_penalty", &cunls::AugmentedLagrangianMinimizerOptions::initial_penalty,
              "Initial penalty rho0. Default: 10.")
      .def_rw("penalty_increase", &cunls::AugmentedLagrangianMinimizerOptions::penalty_increase,
              "Penalty growth factor beta. Default: 10.")
      .def_rw("max_penalty", &cunls::AugmentedLagrangianMinimizerOptions::max_penalty,
              "Penalty cap. Default: 1e8.")
      .def_rw("violation_decrease", &cunls::AugmentedLagrangianMinimizerOptions::violation_decrease,
              "Required violation decrease eta before the penalty grows. Default: 0.25.")
      .def_rw("warm_start", &cunls::AugmentedLagrangianMinimizerOptions::warm_start,
              "Start from the previous call's multipliers and penalties. Default: False.")
      .def_rw("reuse_structure", &cunls::AugmentedLagrangianMinimizerOptions::reuse_structure,
              "The problem's structure is unchanged since the previous call (same batches, "
              "connectivity, active/constant counts, partition): skip the structure setup. "
              "Default: False.")
      .def_rw("max_bound_refinements",
              &cunls::AugmentedLagrangianMinimizerOptions::max_bound_refinements,
              "Box-bounded states: at most this many extra solves per inner iteration holding "
              "the free components the step would push through their bound (exactly this many "
              "with real_time). Default: 3.")
      .def_rw("real_time", &cunls::AugmentedLagrangianMinimizerOptions::real_time,
              "Fixed budget without host synchronization (one read-back at the end): exactly "
              "max_outer_iterations outer iterations of inner_iterations inner iterations "
              "(the last final_inner_iterations if > 0). Costs in the summary are NaN. "
              "Default: False.");

  nb::enum_<cunls::AugmentedLagrangianMinimizerStatus>(
      m, "AugmentedLagrangianMinimizerStatus",
      "Outcome of AugmentedLagrangianMinimizer.minimize(); the summary's message explains it.")
      .value("Converged", cunls::AugmentedLagrangianMinimizerStatus::kConverged,
             "Every subproblem is feasible and stationary.")
      .value("MaxOuterIterations", cunls::AugmentedLagrangianMinimizerStatus::kMaxOuterIterations,
             "Some subproblem ran out of outer iterations (raise max_outer_iterations).")
      .value("MaxPenalty", cunls::AugmentedLagrangianMinimizerStatus::kMaxPenalty,
             "Some subproblem cannot reduce its violation at max_penalty (likely infeasible).")
      .value("NotDetermined", cunls::AugmentedLagrangianMinimizerStatus::kNotDetermined,
             "Convergence was not checked: a real-time call without constraint batches reads "
             "nothing back. converged() is False.")
      .value("InnerSolveFailed", cunls::AugmentedLagrangianMinimizerStatus::kInnerSolveFailed,
             "The inner solve of some subproblem stopped on a non-finite cost or step (a factor "
             "returned NaN or Inf, or the system is singular), feasible or not.");

  nb::class_<cunls::AugmentedLagrangianMinimizerSummary>(
      m, "AugmentedLagrangianMinimizerSummary", "Summary of a constrained minimization run.")
      .def_ro("status", &cunls::AugmentedLagrangianMinimizerSummary::status,
              "AugmentedLagrangianMinimizerStatus: Converged, MaxOuterIterations, MaxPenalty, "
              "InnerSolveFailed or NotDetermined.")
      .def_ro("outer_iterations", &cunls::AugmentedLagrangianMinimizerSummary::outer_iterations,
              "Outer (multiplier update) iterations performed.")
      .def_ro("inner_iterations", &cunls::AugmentedLagrangianMinimizerSummary::inner_iterations,
              "Inner minimizer iterations, summed over the outer iterations.")
      .def_ro("max_violation", &cunls::AugmentedLagrangianMinimizerSummary::max_violation,
              "Largest constraint violation over all rows and subproblems (compare with "
              "constraint_tolerance).")
      .def_ro("initial_cost", &cunls::AugmentedLagrangianMinimizerSummary::initial_cost,
              "Objective (cost of the non-constraint factors) before the solve.")
      .def_ro("final_cost", &cunls::AugmentedLagrangianMinimizerSummary::final_cost,
              "Objective after the solve (NaN in real time).")
      .def_ro("num_problems", &cunls::AugmentedLagrangianMinimizerSummary::num_problems,
              "Number of subproblems (Problem.set_problem_partition; 1 without).")
      .def_ro("num_converged", &cunls::AugmentedLagrangianMinimizerSummary::num_converged,
              "Subproblems that are feasible and stationary.")
      .def_ro("num_max_penalty", &cunls::AugmentedLagrangianMinimizerSummary::num_max_penalty,
              "Subproblems stopped at the penalty cap (constraints likely infeasible).")
      .def_ro("num_inner_failed", &cunls::AugmentedLagrangianMinimizerSummary::num_inner_failed,
              "Subproblems whose inner solve stopped on a non-finite cost or step.")
      .def_ro("message", &cunls::AugmentedLagrangianMinimizerSummary::message,
              "The outcome and, when not converged, the likely causes and what to try.")
      .def_ro("solve_time_ms", &cunls::AugmentedLagrangianMinimizerSummary::solve_time_ms,
              "Wall time of the minimize() call in milliseconds.")
      .def("converged", &cunls::AugmentedLagrangianMinimizerSummary::Converged,
           "status == AugmentedLagrangianMinimizerStatus.Converged.")
      .def("__repr__", [](const cunls::AugmentedLagrangianMinimizerSummary &s) {
        return "AugmentedLagrangianMinimizerSummary(outer_iterations=" +
               std::to_string(s.outer_iterations) +
               ", inner_iterations=" + std::to_string(s.inner_iterations) +
               ", max_violation=" + std::to_string(s.max_violation) +
               ", final_cost=" + std::to_string(s.final_cost) +
               ", converged=" + std::to_string(s.num_converged) + "/" +
               std::to_string(s.num_problems) + ")";
      });

  nb::class_<cunls::AugmentedLagrangianMinimizer>(
      m, "AugmentedLagrangianMinimizer",
      "Augmented Lagrangian solver for problems with constraint factor batches "
      "(ConstraintFactorBatch, BoundFactorBatchN), around a Gauss-Newton or "
      "Levenberg-Marquardt minimizer (minimizer), with options: "
      "AugmentedLagrangianMinimizerOptions. Also the solver for box-bounded states "
      "(VectorStateBatch.set_bounds). Without constraints or bounds it is the wrapped "
      "minimizer.")
      .def(nb::init<cunls::Minimizer &, const cunls::AugmentedLagrangianMinimizerOptions &>(),
           nb::arg("minimizer"), nb::arg("options") = cunls::AugmentedLagrangianMinimizerOptions(),
           nb::keep_alive<1, 2>())
      .def(
          "minimize",
          [](cunls::AugmentedLagrangianMinimizer &self, cunls::CudaStream &stream,
             cunls::Problem &problem) {
            cunls::AugmentedLagrangianMinimizerSummary summary;
            {
              nb::gil_scoped_release release;
              summary = self.Minimize(stream.GetStream(), problem);
            }
            return summary;
          },
          nb::arg("stream"), nb::arg("problem"),
          "Minimize the objective subject to the constraint batches. Returns a "
          "AugmentedLagrangianMinimizerSummary.")
      .def_prop_rw(
          "options", [](const cunls::AugmentedLagrangianMinimizer &self) { return self.Options(); },
          [](cunls::AugmentedLagrangianMinimizer &self,
             const cunls::AugmentedLagrangianMinimizerOptions &options) {
            self.SetOptions(options);
          },
          "Options of the following calls (a copy; assign a modified one). Assigning keeps "
          "the warm-start state.");
}
