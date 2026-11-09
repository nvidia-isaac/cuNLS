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

#pragma once

// Shared header for all pycunls nanobind binding translation units.
//
// Each bind_*() function registers one logical group of cuNLS types with
// the nanobind module.  They are called sequentially from pycunls_module.cpp.

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string>

namespace nb = nanobind;

// Extract a raw GPU device pointer from a Python object.
// Accepts either a plain int (already a pointer) or any object whose
// .data.ptr attribute yields one (e.g. a cupy.ndarray).
uintptr_t extract_device_ptr(nb::handle obj);

// Same, and when obj is an array (has .dtype) checks that its dtype is
// `dtype` (e.g. "float32"), raising TypeError naming `name` otherwise. Plain
// ints (raw pointers) pass unchecked.
uintptr_t extract_device_ptr(nb::handle obj, const char *dtype, const char *name);

// Same, and when the array's size is known (__cuda_array_interface__) checks
// that it holds at least `min_elements` values, raising ValueError that names
// `name` and explains the count with `what` (e.g. "capacity 4 x 16 floats").
uintptr_t extract_device_ptr(nb::handle obj, const char *dtype, const char *name,
                             size_t min_elements, const char *what);

// A device array as seen through __cuda_array_interface__ (or a raw int
// pointer, whose size is unknown).
struct DeviceArray {
  uintptr_t ptr = 0;
  size_t num_elements = 0;
  bool size_known = false;
};

// Checks and reads a device array argument: an int pointer passes unchecked;
// an object with __cuda_array_interface__ (CuPy, PyTorch, Numba, Warp) must
// have dtype `dtype` (when not null) and be C-contiguous; a NumPy array raises
// TypeError explaining that GPU memory is needed. Errors name `name`.
DeviceArray inspect_device_array(nb::handle obj, const char *dtype, const char *name);

// count * per_item, raising ValueError (naming `name`) when the product overflows size_t
// (e.g. a huge capacity that would otherwise wrap to a small required size).
inline size_t checked_size(size_t count, size_t per_item, const char *name) {
  if (per_item != 0 && count > SIZE_MAX / per_item) {
    throw nb::value_error((std::string(name) + ": capacity " + std::to_string(count) + " x " +
                           std::to_string(per_item) +
                           " values overflows size_t. Fix: pass the real capacity.")
                              .c_str());
  }
  return count * per_item;
}

// Typed device array of `count` items of T (each sizeof(T) / 4 float32
// values), checked like extract_device_ptr with a minimum size.
template <typename T>
const T *device_items(nb::handle obj, const char *name, size_t count) {
  static_assert(sizeof(T) % sizeof(float) == 0, "T must be made of 32-bit values");
  const size_t per_item = sizeof(T) / sizeof(float);
  const std::string what =
      "capacity " + std::to_string(count) + " x " + std::to_string(per_item) + " float32 per item";
  return reinterpret_cast<const T *>(
      extract_device_ptr(obj, "float32", name, checked_size(count, per_item, name), what.c_str()));
}

// Device float32 array of `count` items of `per_item` values each ("<per_item>
// float32 per <item>" in the error), checked like extract_device_ptr with a
// minimum size.
inline const float *device_floats(nb::handle obj, const char *name, size_t count, size_t per_item,
                                  const char *item) {
  const std::string what =
      std::to_string(count) + " x " + std::to_string(per_item) + " float32 per " + item;
  return reinterpret_cast<const float *>(
      extract_device_ptr(obj, "float32", name, checked_size(count, per_item, name), what.c_str()));
}

// Device int32 array of at least `count` values (`what` explains the count).
inline const int *device_ints(nb::handle obj, const char *name, size_t count, const char *what) {
  return reinterpret_cast<const int *>(extract_device_ptr(obj, "int32", name, count, what));
}

// --- Binding registration entry points (one per translation unit) ----------

void bind_types(nb::module_ &m);      // CudaStream, enums, options
void bind_state(nb::module_ &m);      // StateBatch hierarchy (vector, Lie groups)
void bind_factor(nb::module_ &m);     // FactorBatch hierarchy + CustomFactorBatch
void bind_loss(nb::module_ &m);       // LossFunctionBatch hierarchy
void bind_minimizer(nb::module_ &m);  // GaussNewton / LevenbergMarquardt
void bind_ransac(nb::module_ &m);     // RANSAC minimizers, options, summary
void bind_problem(nb::module_ &m);    // Problem (assembles states + factors)
