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
  m.doc() = "Python bindings for cuNLS: CUDA-accelerated nonlinear least "
            "squares solver";

  bind_types(m); // CudaStream, CublasHandle, enums, MinimizerOptions, etc.
  bind_state(m); // StateBatch subclasses (VectorStateBatch, SE3StateBatch, ...)
  bind_factor(m);    // FactorBatch subclasses + CustomFactorBatch trampoline
  bind_loss(m);      // Robust loss functions (Huber, Cauchy, Tukey, ...)
  bind_minimizer(m); // GaussNewtonMinimizer, LevenbergMarquardtMinimizer
  bind_problem(m);   // Problem (assembles states, factors, and losses)
}
