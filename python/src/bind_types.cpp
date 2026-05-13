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

#include "bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/sparse_matrix_multiplier.h"

// Convert a Python object to a raw device pointer (uintptr_t).
//
// Two conventions are supported so that callers can pass either:
//   1. A plain Python int (already the numeric pointer value), or
//   2. Any object with a .data.ptr attribute — the protocol used by
//      cupy.ndarray, which exposes the GPU pointer through its memory
//      descriptor.
uintptr_t extract_device_ptr(nb::handle obj) {
  if (nb::isinstance<nb::int_>(obj))
    return nb::cast<uintptr_t>(obj);
  return nb::cast<uintptr_t>(obj.attr("data").attr("ptr"));
}

void bind_types(nb::module_ &m) {
  // --- CUDA stream / cuBLAS handle wrappers ---

  nb::class_<cunls::CudaStream>(m, "CudaStream",
                                "RAII wrapper for a CUDA stream.")
      .def(nb::init<bool>(), nb::arg("sync_on_destroy") = false)
      .def(
          "get_stream",
          [](cunls::CudaStream &self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.GetStream());
          },
          "Returns the underlying cudaStream_t as an integer handle.");

  nb::class_<cunls::cuBLASHandle>(m, "CublasHandle",
                                  "RAII wrapper for a cuBLAS handle.")
      .def(nb::init<>());

  // --- Enumerations for solver/multiplier strategy selection ---

  nb::enum_<cunls::SparseLinearSolverType>(m, "SparseLinearSolverType")
      .value("cuDSS", cunls::SparseLinearSolverType::cuDSS)
      .value("DenseLDLT", cunls::SparseLinearSolverType::DenseLDLT)
      .value("DenseCholesky", cunls::SparseLinearSolverType::DenseCholesky)
      .value("DenseQR", cunls::SparseLinearSolverType::DenseQR)
      .value("BlockSparsePCG", cunls::SparseLinearSolverType::BlockSparsePCG);

  nb::enum_<cunls::SparseMatrixMultiplierType>(m, "SparseMatrixMultiplierType")
      .value("cuSPARSE", cunls::SparseMatrixMultiplierType::cuSPARSE)
      .value("Fast", cunls::SparseMatrixMultiplierType::Fast);

  nb::enum_<cunls::ColumnScaling>(m, "ColumnScaling")
      .value("none", cunls::ColumnScaling::None)
      .value("hessian_diagonal", cunls::ColumnScaling::HessianDiagonal)
      .value("jacobian_column_norm", cunls::ColumnScaling::JacobianColumnNorm);

  // --- Minimizer configuration structs ---
  // All fields are read/write so users can tune convergence behaviour
  // from Python before passing the options to a minimizer constructor.

  nb::class_<cunls::MinimizerOptions>(
      m, "MinimizerOptions",
      "Options for Gauss-Newton and Levenberg-Marquardt minimizers.")
      .def(nb::init<>())
      .def_rw("max_num_iterations",
              &cunls::MinimizerOptions::max_num_iterations)
      .def_rw("state_tolerance", &cunls::MinimizerOptions::state_tolerance)
      .def_rw("cost_tolerance", &cunls::MinimizerOptions::cost_tolerance)
      .def_rw("max_consecutive_rejected_steps",
              &cunls::MinimizerOptions::max_consecutive_rejected_steps)
      .def_rw("sparse_linear_solver_type",
              &cunls::MinimizerOptions::sparse_linear_solver_type)
      .def_rw("sparse_square_multiplier_type",
              &cunls::MinimizerOptions::sparse_square_multiplier_type)
      .def_rw("column_scaling", &cunls::MinimizerOptions::column_scaling)
      .def_rw("disable_safety_checks",
              &cunls::MinimizerOptions::disable_safety_checks,
              "When False, the minimizer enables all optional runtime "
              "validation.  Currently this covers post-factorization "
              "checks in the linear solver: pivot/diagonal checks "
              "(LDLT), devInfo inspection (Cholesky/QR).  Future "
              "versions may add further checks (e.g. NaN/Inf "
              "detection, cost-increase guards).  Failures are "
              "reported via Solve() returning False and a LogError "
              "diagnostic; the minimizer then raises RuntimeError.\n\n"
              "Set to True only for well-conditioned, pre-validated "
              "systems where the extra device-to-host copy and stream "
              "sync are a measurable bottleneck. With checks disabled, "
              "singular or ill-conditioned matrices may produce silently "
              "incorrect results.");

  nb::class_<cunls::MinimizerSummary>(m, "MinimizerSummary",
                                      "Summary of a minimization run.")
      .def_ro("num_iterations", &cunls::MinimizerSummary::num_iterations)
      .def_ro("initial_cost", &cunls::MinimizerSummary::initial_cost)
      .def_ro("final_cost", &cunls::MinimizerSummary::final_cost)
      .def_ro("iteration_costs", &cunls::MinimizerSummary::iteration_costs)
      .def("__repr__", [](const cunls::MinimizerSummary &s) {
        return "MinimizerSummary(iterations=" +
               std::to_string(s.num_iterations) +
               ", initial_cost=" + std::to_string(s.initial_cost) +
               ", final_cost=" + std::to_string(s.final_cost) + ")";
      });

  nb::class_<cunls::LevenbergMarquardtMinimizerOptions>(
      m, "LevenbergMarquardtMinimizerOptions",
      "Options for the Levenberg-Marquardt minimizer.")
      .def(nb::init<>())
      .def_rw("base_options",
              &cunls::LevenbergMarquardtMinimizerOptions::base_options)
      .def_rw("initial_lambda",
              &cunls::LevenbergMarquardtMinimizerOptions::initial_lambda)
      .def_rw("lambda_upscale",
              &cunls::LevenbergMarquardtMinimizerOptions::lambda_upscale)
      .def_rw("lambda_downscale",
              &cunls::LevenbergMarquardtMinimizerOptions::lambda_downscale)
      .def_rw("lambda_max",
              &cunls::LevenbergMarquardtMinimizerOptions::lambda_max)
      .def_rw("lambda_min",
              &cunls::LevenbergMarquardtMinimizerOptions::lambda_min)
      .def_rw("step_accept_threshold",
              &cunls::LevenbergMarquardtMinimizerOptions::step_accept_threshold)
      .def_rw("lambda_downscale_threshold",
              &cunls::LevenbergMarquardtMinimizerOptions::
                  lambda_downscale_threshold);
}
