/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

// Shared header for all pycunls nanobind binding translation units.
//
// Each bind_*() function registers one logical group of cuNLS types with
// the nanobind module.  They are called sequentially from pycunls_module.cpp.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Extract a raw GPU device pointer from a Python object.
// Accepts either a plain int (already a pointer) or any object whose
// .data.ptr attribute yields one (e.g. a cupy.ndarray).
uintptr_t extract_device_ptr(nb::handle obj);

// --- Binding registration entry points (one per translation unit) ----------

void bind_types(nb::module_& m);     // CudaStream, CublasHandle, enums, options
void bind_state(nb::module_& m);     // StateBatch hierarchy (vector, Lie groups)
void bind_factor(nb::module_& m);    // FactorBatch hierarchy + CustomFactorBatch
void bind_loss(nb::module_& m);      // LossFunctionBatch hierarchy
void bind_minimizer(nb::module_& m); // GaussNewton / LevenbergMarquardt
void bind_problem(nb::module_& m);   // Problem (assembles states + factors)
