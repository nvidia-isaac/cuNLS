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

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "cunls/state/vector_state_batch.h"

namespace nb = nanobind;

// Makes `device` current for the scope (when >= 0 and not current already) and restores the
// caller's device at its end: a native call runs on the device of the data it touches, whatever
// device the Python caller made current. Warp uses each device's primary context, the one the
// runtime selects. Use it around every call that launches work on a problem or a batch.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    if (device < 0) return;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current == device) return;
    if (cudaSetDevice(device) == cudaSuccess) previous_ = current;
  }
  ~DeviceScope() {
    if (previous_ >= 0) cudaSetDevice(previous_);
  }
  DeviceScope(const DeviceScope &) = delete;
  DeviceScope &operator=(const DeviceScope &) = delete;

 private:
  int previous_ = -1;  ///< The caller's device to restore; -1: nothing was switched.
};

// The device of a device pointer (cudaPointerGetAttributes), or -1 for anything else.
inline int device_of(const void *ptr) {
  cudaPointerAttributes attributes{};
  if (ptr == nullptr || cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
    (void)cudaGetLastError();
    return -1;
  }
  return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged
             ? attributes.device
             : -1;
}

// Extracts a raw GPU device pointer from a Python object: a plain int (already a pointer) or a
// device array (__cuda_array_interface__); when obj is an array, checks that its dtype is
// `dtype` (e.g. "float32"), raising TypeError naming `name` otherwise. Plain ints (raw pointers)
// pass unchecked.
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
  size_t ndim = 0;         // dimensions of the shape (when size_known)
  size_t last_extent = 0;  // its trailing extent (when ndim >= 2): e.g. the n of (N, n) or vecN
};

// Checks and reads a device array argument: an int pointer passes unchecked;
// an object with __cuda_array_interface__ (Warp, PyTorch, CuPy, Numba) must
// have dtype `dtype` (when not null) and be C-contiguous; a NumPy array raises
// TypeError explaining that GPU memory is needed. Errors name `name`.
DeviceArray inspect_device_array(nb::handle obj, const char *dtype, const char *name);

// The size of a runtime-sized vector batch (VectorStateBatch, PriorVectorFactorBatch, ...):
// `dim` when given (the array's size check then applies to capacity x dim values), else the
// trailing extent of `data`, a (N, n) array or an array of wp.vecN. Raises ValueError naming
// `cls` when neither gives it or dim < 1.
int vector_dim(nb::handle data, nb::handle dim, const char *cls, const char *name);

// A 1-D numpy array holding a host copy of the `n` values at `device` (zeros when `device` is
// null). Synchronous: the copy waits for the device.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> device_to_numpy(const T *device, size_t n) {
  auto *host = new T[n > 0 ? n : 1]();
  nb::capsule owner(host, [](void *p) noexcept { delete[] static_cast<T *>(p); });
  if (n > 0 && device != nullptr) {
    const cudaError_t status = cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {  // `owner` frees host
      throw std::runtime_error(std::string("device_to_numpy: cudaMemcpy failed (") +
                               cudaGetErrorString(status) + "). Fix: check the CUDA device.");
    }
  }
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(host, {n}, owner);
}

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

/**
 * Keeps the bounds arrays of state batch `self` alive while they are its bounds: one
 * attribute holds the current pair (None after removal), so replaced arrays are released
 * instead of accumulating for the batch's lifetime. Queued GPU work that still reads replaced
 * arrays needs them kept alive by the caller (or the stream synchronized) until it finishes.
 */
template <typename Batch>
inline void hold_bounds(Batch &self, nb::handle lower, nb::handle upper) {
  nb::setattr(nb::find(&self), "_bounds_arrays",
              lower.is_none() ? nb::none() : nb::object(nb::make_tuple(lower, upper)));
}

// Device float32 array of `count` items of `per_item` values each ("<per_item>
// float32 per <item>" in the error), checked like extract_device_ptr with a
// minimum size. The size check is best-effort: it applies when the size is
// known (__cuda_array_interface__); a raw int pointer or a .data.ptr-only object
// passes unchecked.
inline const float *device_floats(nb::handle obj, const char *name, size_t count, size_t per_item,
                                  const char *item) {
  const std::string what =
      std::to_string(count) + " x " + std::to_string(per_item) + " float32 per " + item;
  return reinterpret_cast<const float *>(
      extract_device_ptr(obj, "float32", name, checked_size(count, per_item, name), what.c_str()));
}

// The optional const_state_ids argument of a state batch constructor: nullptr for None (only with
// const_capacity 0: no ids to read), else a device int32 array of at least `const_capacity` ids.
inline const int *const_state_ids(nb::handle ids, size_t const_capacity) {
  if (ids.is_none()) {
    if (const_capacity != 0) {
      throw nb::value_error(("const_state_ids is None but const_capacity is " +
                             std::to_string(const_capacity) +
                             " (expected 0 without ids). Fix: pass the constant state ids, or "
                             "const_capacity=0.")
                                .c_str());
    }
    return nullptr;
  }
  return reinterpret_cast<const int *>(
      extract_device_ptr(ids, "int32", "const_state_ids", const_capacity, "const_capacity ids"));
}

// Device int32 array of at least `count` values (`what` explains the count); the
// size check is best-effort, as for device_floats.
inline const int *device_ints(nb::handle obj, const char *name, size_t count, const char *what) {
  return reinterpret_cast<const int *>(extract_device_ptr(obj, "int32", name, count, what));
}

/**
 * Binds `set_bounds(lower, upper)` (device float32 arrays of capacity x tangent_size, or None,
 * None to remove them; one None alone raises ValueError naming `cls_name`) and `has_bounds` on a
 * box-bounded state batch class, with `doc` for set_bounds.
 */
template <typename Class, typename... Extra>
void bind_box_bounds(nb::class_<Class, Extra...> &cls, const char *cls_name, const char *doc) {
  cls.def(
         "set_bounds",
         [cls_name](Class &self, nb::handle lower, nb::handle upper) {
           if (lower.is_none() && upper.is_none()) {
             self.SetBounds(nullptr, nullptr);
             hold_bounds(self, lower, upper);
             return;
           }
           if (lower.is_none() || upper.is_none()) {
             throw std::invalid_argument(
                 std::string(cls_name) +
                 ".set_bounds: got one bound and None (expected both arrays, or None, None to "
                 "remove the bounds). Fix: pass -inf / +inf arrays for an unbounded side.");
           }
           self.SetBounds(
               device_floats(lower, "lower", self.Capacity(), self.TangentSize(), "state"),
               device_floats(upper, "upper", self.Capacity(), self.TangentSize(), "state"));
           hold_bounds(self, lower, upper);
         },
         nb::arg("lower").none(), nb::arg("upper").none(), doc)
      .def_prop_ro(
          "has_bounds", [](const Class &self) { return cunls::HasBoxBounds(&self); },
          "Whether the batch has box bounds (solve such a problem with "
          "AugmentedLagrangianMinimizer).");
}

// --- Binding registration entry points (one per translation unit) ----------

void bind_types(nb::module_ &m);         // CudaStream, enums, options
void bind_state(nb::module_ &m);         // StateBatch hierarchy (vector, Lie groups)
void bind_articulation(nb::module_ &m);  // ArticulationTopology, state and kinematic factors
void bind_factor(nb::module_ &m);        // FactorBatch hierarchy + CustomFactorBatch
void bind_loss(nb::module_ &m);          // LossFunctionBatch hierarchy
void bind_minimizer(nb::module_ &m);     // GaussNewton / LevenbergMarquardt
void bind_ransac(nb::module_ &m);        // RANSAC minimizers, options, summary
void bind_problem(nb::module_ &m);       // Problem (assembles states + factors)
