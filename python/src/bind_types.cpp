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

// Bindings for utility types, enumerations, and option/summary structs.
//
// This translation unit also defines extract_device_ptr(), the central helper
// that converts Python-side GPU memory handles into raw C pointers.  Every
// other bind_*.cpp file relies on it for constructor arguments that accept
// CuPy arrays or plain integer addresses.

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"

// Python arrays to raw device pointers. inspect_device_array accepts a plain
// int (a device pointer, unchecked), an object with __cuda_array_interface__
// (dtype and C-contiguity checked, element count recorded) or with .data.ptr;
// it rejects host (NumPy) arrays. The extract_device_ptr overloads wrap it, the
// five-argument one also checks the element count.
namespace {

/** numpy-style dtype name of a __cuda_array_interface__ typestr ("<f4" -> "float32"). */
std::string DtypeName(const std::string &typestr) {
  if (typestr.size() < 3) return typestr;
  const char kind = typestr[1];
  const std::string bytes = typestr.substr(2);
  const int bits = 8 * std::atoi(bytes.c_str());
  if (kind == 'f') return "float" + std::to_string(bits);
  if (kind == 'i') return "int" + std::to_string(bits);
  if (kind == 'u') return "uint" + std::to_string(bits);
  if (kind == 'b') return "bool";
  return typestr;
}

}  // namespace

DeviceArray inspect_device_array(nb::handle obj, const char *dtype, const char *name) {
  DeviceArray out;
  if (nb::isinstance<nb::int_>(obj)) {  // a raw device pointer: nothing to check
    out.ptr = nb::cast<uintptr_t>(obj);
    return out;
  }
  if (nb::hasattr(obj, "__cuda_array_interface__")) {
    const nb::dict cai = nb::cast<nb::dict>(obj.attr("__cuda_array_interface__"));
    const std::string actual = DtypeName(nb::cast<std::string>(cai["typestr"]));
    if (dtype != nullptr && actual != dtype) {
      throw nb::type_error((std::string(name) + " must have dtype " + dtype + ", got " + actual +
                            ". Fix: convert it, e.g. cupy.asarray(" + name + ", dtype=cupy." +
                            dtype + ").")
                               .c_str());
    }
    const nb::tuple shape = nb::cast<nb::tuple>(cai["shape"]);
    size_t count = 1;
    for (nb::handle extent : shape) count *= nb::cast<size_t>(extent);
    if (cai.contains("strides") && !cai["strides"].is_none()) {
      const nb::tuple strides = nb::cast<nb::tuple>(cai["strides"]);
      const std::string typestr = nb::cast<std::string>(cai["typestr"]);
      size_t expected = static_cast<size_t>(std::atoi(typestr.substr(2).c_str()));
      for (size_t i = shape.size(); i-- > 0;) {
        const size_t extent = nb::cast<size_t>(shape[i]);
        if (extent > 1 && nb::cast<int64_t>(strides[i]) != static_cast<int64_t>(expected)) {
          throw nb::value_error((std::string(name) +
                                 " must be C-contiguous (a row-major array without gaps); got "
                                 "a strided view. Fix: pass cupy.ascontiguousarray(" +
                                 name + ").")
                                    .c_str());
        }
        expected *= extent;
      }
    }
    out.ptr = nb::cast<uintptr_t>(nb::cast<nb::tuple>(cai["data"])[0]);
    out.num_elements = count;
    out.size_known = true;
    return out;
  }
  if (nb::hasattr(obj, "__array_interface__")) {
    throw nb::type_error((std::string(name) +
                          " is a host (NumPy) array, but cuNLS reads GPU memory. Fix: pass "
                          "cupy.asarray(" +
                          name + ")" + (dtype ? std::string(" with dtype ") + dtype : "") +
                          " and keep it alive while it is used.")
                             .c_str());
  }
  if (nb::hasattr(obj, "data") && nb::hasattr(obj.attr("data"), "ptr")) {
    out.ptr = nb::cast<uintptr_t>(obj.attr("data").attr("ptr"));
    return out;
  }
  throw nb::type_error((std::string(name) +
                        " must be a GPU array (cupy.ndarray or any object "
                        "with __cuda_array_interface__) or an int device "
                        "pointer; got " +
                        nb::cast<std::string>(nb::str(obj.type())))
                           .c_str());
}

uintptr_t extract_device_ptr(nb::handle obj) {
  return inspect_device_array(obj, nullptr, "array").ptr;
}

uintptr_t extract_device_ptr(nb::handle obj, const char *dtype, const char *name) {
  return inspect_device_array(obj, dtype, name).ptr;
}

uintptr_t extract_device_ptr(nb::handle obj, const char *dtype, const char *name,
                             size_t min_elements, const char *what) {
  const DeviceArray a = inspect_device_array(obj, dtype, name);
  if (a.size_known && a.num_elements < min_elements) {
    throw nb::value_error((std::string(name) + " has " + std::to_string(a.num_elements) + " " +
                           dtype + " values, but " + std::to_string(min_elements) +
                           " are needed (" + what +
                           "). Fix: pass a larger array or a smaller "
                           "capacity.")
                              .c_str());
  }
  return a.ptr;
}

void bind_types(nb::module_ &m) {
  // --- CUDA stream wrapper ---

  nb::class_<cunls::CudaStream>(
      m, "CudaStream",
      "A CUDA stream for cuNLS work (created on construction, destroyed with the object). "
      "sync_on_destroy: synchronize before destroying. minimize() synchronizes it before "
      "returning; use cupy.cuda.Stream.from_external(stream) to order CuPy work with it.")
      .def(nb::init<bool>(), nb::arg("sync_on_destroy") = false)
      .def(
          "get_stream",
          [](cunls::CudaStream &self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.GetStream());
          },
          "Returns the underlying cudaStream_t as an integer handle.")
      .def(
          "__cuda_stream__",
          [](cunls::CudaStream &self) {
            return nb::make_tuple(0, reinterpret_cast<uintptr_t>(self.GetStream()));
          },
          "CUDA stream protocol (version 0, handle): cupy.cuda.Stream.from_external(stream) "
          "and other libraries can enqueue work on this stream.");

  // --- Logging ---

  nb::enum_<cunls::Verbosity>(m, "Verbosity", nb::is_arithmetic(),
                              "Log levels, from least to most verbose.")
      .value("Silent", cunls::Verbosity::Silent, "Nothing is logged.")
      .value("Error", cunls::Verbosity::Error, "Errors only (they are also raised).")
      .value("Warning", cunls::Verbosity::Warning,
             "Errors and warnings (e.g. a failed linear solve that was rejected).")
      .value("Message", cunls::Verbosity::Message,
             "Also one line per minimizer iteration and the final termination message.")
      .value("Debug", cunls::Verbosity::Debug, "Everything, including internal details.");

  m.def(
      "set_log_verbosity", [](cunls::Verbosity verbosity) { cunls::SetLoggerOptions(verbosity); },
      nb::arg("verbosity"),
      "Sets the library log level. The default is Silent, or the level named by the "
      "environment variable CUNLS_LOG_LEVEL (silent, error, warning, message, debug). Message "
      "logs every minimizer iteration and why the solve stopped.");

  m.def(
      "_set_log_callback",
      [](nb::object callback) {
        if (callback.is_none()) {
          cunls::SetLogCallback(nullptr);
          return;
        }
        // Owned by the C++ callback; released when the callback is replaced.
        std::shared_ptr<nb::object> held(new nb::object(callback), [](nb::object *o) {
          if (!Py_IsInitialized()) return;  // interpreter gone: leak rather than crash
          nb::gil_scoped_acquire gil;
          delete o;
        });
        cunls::SetLogCallback([held](cunls::Verbosity verbosity, std::string_view msg) {
          if (!Py_IsInitialized()) return;
          nb::gil_scoped_acquire gil;
          try {
            (*held)(static_cast<int>(verbosity), nb::str(msg.data(), msg.size()));
          } catch (...) {
            // A logging failure must not break the solve.
          }
        });
      },
      nb::arg("callback").none(),
      "Internal: route log messages to callback(level: int, message: str) (None restores "
      "the console). Use pycunls.log_to_python() instead.");

  // --- Enumerations for solver/multiplier strategy selection ---

  nb::enum_<cunls::SparseLinearSolverType>(
      m, "SparseLinearSolverType",
      "Solver of the assembled system (MinimizerOptions.sparse_linear_solver_type; not used "
      "with LinearSystemType.SchurComplement). Default: BlockSparseCholesky.")
      .value("cuDSS", cunls::SparseLinearSolverType::cuDSS,
             "NVIDIA cuDSS sparse direct solver (optional dependency, loaded at run time).")
      .value("DenseLDLT", cunls::SparseLinearSolverType::DenseLDLT,
             "Dense pivoted LDLT (small systems).")
      .value("DenseCholesky", cunls::SparseLinearSolverType::DenseCholesky,
             "Dense cuSOLVER Cholesky (small SPD systems).")
      .value("DenseQR", cunls::SparseLinearSolverType::DenseQR,
             "Dense cuSOLVER QR (small non-singular systems).")
      .value("BlockSparsePCG", cunls::SparseLinearSolverType::BlockSparsePCG,
             "Block-Jacobi preconditioned conjugate gradient (inexact steps); for graphs with "
             "many long-range random edges.")
      .value("BlockTridiagonal", cunls::SparseLinearSolverType::BlockTridiagonal,
             "Block-tridiagonal Cholesky of stage-ordered problems (Problem.set_state_stages).")
      .value("BlockSparseCholesky", cunls::SparseLinearSolverType::BlockSparseCholesky,
             "Supernodal sparse Cholesky on the GPU (default; exact steps, no dependencies).");

  nb::enum_<cunls::LinearSystemType>(m, "LinearSystemType",
                                     "How each iteration's linear system is formed "
                                     "(MinimizerOptions.linear_system).")
      .value("Assembled", cunls::LinearSystemType::Assembled,
             "The full normal equations, solved with sparse_linear_solver_type.")
      .value("SchurComplement", cunls::LinearSystemType::SchurComplement,
             "Eliminate the state batches of Problem.set_eliminated_state_batches (the "
             "landmarks of a bundle adjustment) and solve the reduced system with the "
             "built-in supernodal Cholesky (BlockSparseCholesky; sparse_linear_solver_type "
             "is not used).");

  nb::enum_<cunls::ColumnScaling>(
      m, "ColumnScaling",
      "Diagonal scaling of the normal equations: none, or hessian_diagonal (S = diag(H)^-1/2, "
      "i.e. 1 / |J column|), which helps when parameters have very different scales.")
      .value("none", cunls::ColumnScaling::None, "No scaling.")
      .value("hessian_diagonal", cunls::ColumnScaling::HessianDiagonal,
             "Scale each column by 1 / sqrt(H_jj) (with a floor).");

  nb::enum_<cunls::JacobianMode>(
      m, "JacobianMode",
      "Selects how a factor batch's Jacobian is obtained: analytic (the "
      "factor's own hand-derived Evaluate() output) or numeric (finite "
      "differences on the manifold tangent space of each referenced state "
      "block, via Problem.add_factor_batch's jacobian_mode_override or "
      "MinimizerOptions.jacobian_mode).")
      .value("analytic", cunls::JacobianMode::kAnalytic,
             "The factor's own hand-derived Jacobians from Evaluate() (default).")
      .value("numeric", cunls::JacobianMode::kNumeric,
             "Finite differences on the tangent space of each referenced state.");

  nb::enum_<cunls::NumericDiffOptions::Method>(
      m, "NumericDiffMethod",
      "Finite-difference scheme used when a factor batch resolves to "
      "JacobianMode.numeric.")
      .value("forward", cunls::NumericDiffOptions::Method::kForward,
             "One-sided: (f(x+eps) - f(x)) / eps. Cheaper, less accurate.")
      .value("central", cunls::NumericDiffOptions::Method::kCentral,
             "Two-sided: (f(x+eps) - f(x-eps)) / (2*eps). Default.");

  nb::class_<cunls::NumericDiffOptions>(
      m, "NumericDiffOptions", "Tuning knobs for numeric (finite-difference) Jacobian computation.")
      .def(nb::init<>())
      .def_rw("method", &cunls::NumericDiffOptions::method,
              "Finite-difference stencil (NumericDiffMethod).")
      .def_rw("relative_step_size", &cunls::NumericDiffOptions::relative_step_size,
              "Per-tangent-coordinate perturbation step size. Default: 1e-4.");

  // --- Minimizer configuration structs ---
  // All fields are read/write so users can tune convergence behaviour
  // from Python before passing the options to a minimizer constructor.

  nb::class_<cunls::MinimizerOptions>(
      m, "MinimizerOptions",
      "Options common to GaussNewtonMinimizer and LevenbergMarquardtMinimizer.")
      .def(nb::init<>())
      .def_rw("max_num_iterations", &cunls::MinimizerOptions::max_num_iterations,
              "Maximum iterations of one minimize() call. Default: 50.")
      .def_rw("state_tolerance", &cunls::MinimizerOptions::state_tolerance,
              "Converged when the squared step norm |dx|^2 falls below this (absolute, tangent "
              "units). Default: 1e-6.")
      .def_rw("cost_tolerance", &cunls::MinimizerOptions::cost_tolerance,
              "Converged when the cost falls below this (absolute). Lower it when the final cost "
              "is tiny (exact data, small residual units). Default: 1e-6.")
      .def_rw("max_consecutive_rejected_steps",
              &cunls::MinimizerOptions::max_consecutive_rejected_steps,
              "Stop a subproblem after this many rejected steps in a row (Levenberg-Marquardt: at "
              "full damping); 0 disables. Default: 5.")
      .def_rw(
          "max_line_search_steps", &cunls::MinimizerOptions::max_line_search_steps,
          "Backtracking line search: a step that does not decrease the cost is halved "
          "along the same direction up to this many times; the first shorter step that "
          "decreases the cost is taken (any decreasing step is taken, and LM keeps its damping). "
          "0 (default) disables it.")
      .def_rw("reuse_structure", &cunls::MinimizerOptions::reuse_structure,
              "The problem's structure is unchanged since this minimizer's previous minimize() "
              "on it (same batches, connectivity, active/constant counts, partition): later "
              "calls skip the structure setup. A size change falls back to the full setup. "
              "Default: False.")
      .def_rw("sparse_linear_solver_type", &cunls::MinimizerOptions::sparse_linear_solver_type,
              "Linear solver of each step (SparseLinearSolverType). Default: BlockSparseCholesky "
              "(exact, no dependencies); BlockSparsePCG for graphs with many long-range edges; "
              "cuDSS optional; dense solvers for small dense problems.")
      .def_rw("linear_system", &cunls::MinimizerOptions::linear_system,
              "How each iteration's linear system is formed: LinearSystemType.Assembled "
              "(default) or SchurComplement (needs Problem.set_eliminated_state_batches).")
      .def_rw("column_scaling", &cunls::MinimizerOptions::column_scaling,
              "Diagonal scaling of the normal equations (ColumnScaling): hessian_diagonal helps "
              "when parameters have very different scales. Default: none.")
      .def_rw("jacobian_mode", &cunls::MinimizerOptions::jacobian_mode,
              "Global default JacobianMode for every factor batch, unless "
              "overridden per group via Problem.add_factor_batch's "
              "jacobian_mode_override. Default: JacobianMode.analytic.")
      .def_rw("numeric_diff_options", &cunls::MinimizerOptions::numeric_diff_options,
              "Tuning knobs used whenever a factor batch is evaluated with "
              "JacobianMode.numeric; ignored otherwise.")
      .def_rw("disable_safety_checks", &cunls::MinimizerOptions::disable_safety_checks,
              "When False, the linear solvers check their factorization (BlockSparseCholesky "
              "and Schur pivot flags, cuSOLVER devInfo, QR rank, LDLT pivots; one read-back "
              "each). A failed solve is then a rejected step for Levenberg-Marquardt (more "
              "damping) and a RuntimeError for Gauss-Newton. When True (default) these checks "
              "are skipped for speed: a singular system can give a non-finite step, which is "
              "rejected (summary reason NonFiniteStep). Default: True.");

  nb::enum_<cunls::TerminationReason>(
      m, "TerminationReason", nb::is_arithmetic(),
      "Why a minimizer stopped iterating a (sub)problem, ordered by severity. Convergence: "
      "CostTolerance, StateTolerance, RelativeReductionTolerance (Levenberg-Marquardt), "
      "NoCostDecrease (Gauss-Newton: a step no longer lowered the cost). Not converged: "
      "MaxIterations, MaxRejectedSteps, NonFiniteStep (a failed linear solve or NaN at the "
      "trial states), NonFiniteCost (NaN/Inf initial cost). Also NoFreeStates (all states "
      "constant) and NotDetermined (real-time solves).")
      .value("NotDetermined", cunls::TerminationReason::NotDetermined,
             "Real-time (fixed-iteration) solve: not read back.")
      .value("NoFreeStates", cunls::TerminationReason::NoFreeStates,
             "Every state is constant: nothing to optimize.")
      .value("CostTolerance", cunls::TerminationReason::CostTolerance,
             "Converged: the cost fell below cost_tolerance.")
      .value("StateTolerance", cunls::TerminationReason::StateTolerance,
             "Converged: the squared step norm fell below state_tolerance.")
      .value("RelativeReductionTolerance", cunls::TerminationReason::RelativeReductionTolerance,
             "Converged (Levenberg-Marquardt): the predicted relative cost reduction fell below "
             "relative_reduction_tolerance.")
      .value("NoCostDecrease", cunls::TerminationReason::NoCostDecrease,
             "Converged (Gauss-Newton): a full step no longer lowered the cost.")
      .value("MaxIterations", cunls::TerminationReason::MaxIterations,
             "Not converged: max_num_iterations was reached.")
      .value("MaxRejectedSteps", cunls::TerminationReason::MaxRejectedSteps,
             "Not converged: max_consecutive_rejected_steps steps were rejected in a row.")
      .value("NonFiniteStep", cunls::TerminationReason::NonFiniteStep,
             "Not converged: the last trial steps were not finite (a failed linear solve or NaN "
             "at the trial states).")
      .value("NonFiniteCost", cunls::TerminationReason::NonFiniteCost,
             "Not started: the initial cost is NaN or infinite.");

  nb::class_<cunls::MinimizerSummary>(
      m, "MinimizerSummary",
      "Result of Minimizer.minimize(): iterations, costs, why the solve stopped "
      "(termination_reason, message) and the wall time. Check converged() first; message "
      "says what went wrong and what to try. to_dict() gives a JSON-ready dict, report() a "
      "multi-line text.")
      .def_ro("num_iterations", &cunls::MinimizerSummary::num_iterations,
              "Iterations performed (each builds and solves one linear system).")
      .def_ro("initial_cost", &cunls::MinimizerSummary::initial_cost,
              "Total cost 1/2 sum rho(|r|^2) of the states passed in.")
      .def_ro("final_cost", &cunls::MinimizerSummary::final_cost,
              "Total cost of the states written back to the problem.")
      .def_ro("iteration_costs", &cunls::MinimizerSummary::iteration_costs,
              "Total cost at the start of each iteration.")
      .def_ro("termination_reason", &cunls::MinimizerSummary::termination_reason,
              "TerminationReason: why the solve stopped (with a partition, the most severe "
              "reason of any subproblem).")
      .def_ro("num_problems_by_reason", &cunls::MinimizerSummary::num_problems_by_reason,
              "Number of subproblems per TerminationReason value (index = int(reason)); empty "
              "for real-time solves.")
      .def_ro("message", &cunls::MinimizerSummary::message,
              "Why the solve stopped and, when it did not converge, the likely causes and what "
              "to try.")
      .def_ro("num_free_parameters", &cunls::MinimizerSummary::num_free_parameters,
              "Number of free (optimized) tangent-space parameters.")
      .def_ro("solve_time_ms", &cunls::MinimizerSummary::solve_time_ms,
              "Wall time of the minimize() call in milliseconds.")
      .def("converged", &cunls::MinimizerSummary::Converged,
           "True when every subproblem stopped at a convergence criterion (or had nothing to "
           "optimize); False at the iteration cap, after repeated rejected steps, or with a "
           "non-finite cost.")
      .def("__repr__", [](const cunls::MinimizerSummary &s) {
        char costs[96];
        std::snprintf(costs, sizeof(costs), "cost %.4g -> %.4g, %.3g ms", s.initial_cost,
                      s.final_cost, s.solve_time_ms);
        return std::string("MinimizerSummary(converged=") + (s.Converged() ? "True" : "False") +
               ", reason=" + cunls::ToString(s.termination_reason) +
               ", iterations=" + std::to_string(s.num_iterations) + ", " + costs + ")";
      });

  nb::class_<cunls::LevenbergMarquardtMinimizerOptions>(
      m, "LevenbergMarquardtMinimizerOptions", "Options for the Levenberg-Marquardt minimizer.")
      .def(nb::init<>())
      .def_rw("base_options", &cunls::LevenbergMarquardtMinimizerOptions::base_options,
              "Options common to all minimizers (MinimizerOptions). Set fields on it, e.g. "
              "options.base_options.max_num_iterations = 100.")
      .def_rw("initial_lambda", &cunls::LevenbergMarquardtMinimizerOptions::initial_lambda,
              "Initial damping lambda; larger is more conservative (closer to gradient descent). 0 "
              "(Gauss-Newton steps) is allowed. Default: 1e-3.")
      .def_rw("relative_reduction_tolerance",
              &cunls::LevenbergMarquardtMinimizerOptions::relative_reduction_tolerance,
              "Converged when the cost reduction the linear model predicts, relative to the cost, "
              "falls below this. Default: 1e-6.")
      .def_rw("lambda_upscale", &cunls::LevenbergMarquardtMinimizerOptions::lambda_upscale,
              "Damping growth on a rejected step (> 1); the k-th rejection in a row multiplies "
              "lambda by lambda_upscale * 2^(k-1). Default: 2.")
      .def_rw("lambda_downscale", &cunls::LevenbergMarquardtMinimizerOptions::lambda_downscale,
              "Damping factor on a very successful step (rho > lambda_downscale_threshold). "
              "Default: 0.5.")
      .def_rw("lambda_max", &cunls::LevenbergMarquardtMinimizerOptions::lambda_max,
              "Largest damping (>= lambda_min). Default: 1e6.")
      .def_rw("lambda_min", &cunls::LevenbergMarquardtMinimizerOptions::lambda_min,
              "Smallest damping (> 0). Default: 1e-6.")
      .def_rw("min_diagonal", &cunls::LevenbergMarquardtMinimizerOptions::min_diagonal,
              "Floor of the diagonal the damping scales, lambda * max(diag(H), min_diagonal), "
              "so unobservable unknowns (zero Jacobian columns) get a zero step instead of a "
              "singular system. Default: 1e-6.")
      .def_rw("step_accept_threshold",
              &cunls::LevenbergMarquardtMinimizerOptions::step_accept_threshold,
              "Accept a step when rho = actual / predicted cost reduction >= this. Default: 0.25.")
      .def_rw("lambda_downscale_threshold",
              &cunls::LevenbergMarquardtMinimizerOptions::lambda_downscale_threshold,
              "Lower the damping when rho > this. Default: 0.75.");
}
