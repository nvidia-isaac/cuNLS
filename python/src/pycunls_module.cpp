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

// Entry point for the _pycunls_core native extension module.
//
// This file defines the NB_MODULE macro that nanobind expands into the
// CPython module initialisation function.  Each bind_*() call registers a
// group of cuNLS C++ types with the module; the grouping mirrors the cuNLS
// library's own header layout (state/, factor/, robustifier/, minimizer/).

#include "bindings.h"

NB_MODULE(_pycunls_core, m) {
  m.doc() =
      "Python bindings for cuNLS: CUDA-accelerated nonlinear least "
      "squares solver";

  // pycunls is a CUDA 13 build: an older driver cannot run it (no driver at all, 0, imports as
  // before so that docs builds work). NB_MODULE raises the exception as ImportError.
  int driver = 0;
  if (cudaDriverGetVersion(&driver) == cudaSuccess && driver > 0 && driver < 13000) {
    const std::string version =
        std::to_string(driver / 1000) + "." + std::to_string(driver % 1000 / 10);
    throw std::runtime_error("pycunls: the driver supports CUDA " + version +
                             " but pycunls is built with CUDA 13 (got " + version +
                             ", expected 13.0 or newer). Fix: install an R580 or newer driver; on "
                             "CUDA 12 systems (Jetson Orin) use the C++ library.");
  }

  bind_types(m);         // CudaStream, enums, MinimizerOptions, etc.
  bind_state(m);         // StateBatch subclasses (VectorStateBatch, SE3StateBatch, ...)
  bind_factor(m);        // FactorBatch subclasses + CustomFactorBatch trampoline
  bind_articulation(m);  // ArticulationTopology, ArticulationStateBatch, kinematic factors
  bind_loss(m);          // Robust loss functions (Huber, Cauchy, Tukey, ...)
  bind_minimizer(m);     // Minimizer, GaussNewtonMinimizer, LevenbergMarquardtMinimizer, AL
  bind_problem(m);       // Problem (assembles states, factors, and losses)
  bind_ransac(m);        // RansacMinimizer and its Gauss-Newton / Levenberg-Marquardt variants
}
