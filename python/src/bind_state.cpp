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

// Bindings for all StateBatch subclasses.
//
// cuNLS represents optimisation variables as "state batches": contiguous GPU
// arrays that hold N states of a single geometric type.  Each batch knows its
// manifold structure (ambient dimension, tangent dimension, retraction/Plus).
//
// Two families exist:
//   - VectorStateBatch<Dim>  — Euclidean R^n where Plus is simple addition.
//   - Manifold batches       — Lie-group types (SE3, SO3, SO2, SE2, Sim2, Sim3,
//                              SL4) whose ambient/tangent dimensions differ.
//
// Both families expose the same Python interface (two constructor overloads,
// state_device_ptr(), and read-only num/tangent/ambient properties).
// The template helpers below factor out the repetitive nanobind boilerplate.

#include <nanobind/ndarray.h>

#include <stdexcept>
#include <string>

#include "bindings.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sized_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace {

// Python trampoline for cunls::StateBatch.
//
// Implements all StateBatch pure-virtual methods in C++, except Plus()
// which acquires the GIL and delegates to Python's ``plus()`` method.
// This is the state-batch counterpart of PyFactorBatch in bind_factor.cpp.
//
// Storage layout mirrors SizedStateBatch: a contiguous GPU buffer of
// capacity * ambient_size floats, with optional const-state indices. Like the
// built-in batches it starts with 0 active states (see set_num_active_states).
class PyStateBatch : public cunls::StateBatch {
 public:
  const float *ptr_;
  size_t ambient_size_;
  size_t tangent_size_;
  size_t num_active_states_;
  const int *const_ids_;
  size_t num_const_;
  size_t capacity_;        // states the buffer holds
  size_t const_capacity_;  // entries the constant-id buffer holds

  PyStateBatch(uintptr_t data_ptr, size_t ambient_size, size_t tangent_size, size_t capacity,
               uintptr_t const_ids_ptr, size_t const_capacity)
      : ptr_(reinterpret_cast<const float *>(data_ptr)),
        ambient_size_(ambient_size),
        tangent_size_(tangent_size),
        num_active_states_(0),
        const_ids_(reinterpret_cast<const int *>(const_ids_ptr)),
        num_const_(0),
        capacity_(capacity),
        const_capacity_(const_capacity) {}

  size_t TangentSize() const override { return tangent_size_; }
  size_t AmbientSize() const override { return ambient_size_; }
  size_t NumActiveStates() const override { return num_active_states_; }

  float *StateDevicePtr(size_t idx) override {
    if (idx >= capacity_) return nullptr;
    return const_cast<float *>(ptr_ + idx * ambient_size_);
  }

  const float *StateDevicePtr(size_t idx) const override {
    if (idx >= capacity_) return nullptr;
    return ptr_ + idx * ambient_size_;
  }

  const int *ConstStateIds() const override { return const_ids_; }
  size_t NumConstStates() const override { return num_const_; }
  size_t Capacity() const override { return capacity_; }
  size_t ConstCapacity() const override { return const_capacity_; }

  // Active sizes for buffers reused across solves (see StateBatch::SetNumActiveStates).
  void SetNumActiveStates(size_t num_active_states, size_t num_const = 0) override {
    if (num_active_states > capacity_ || num_const > const_capacity_) {
      throw std::invalid_argument("set_num_active_states: size exceeds the capacity");
    }
    num_active_states_ = num_active_states;
    num_const_ = num_const;
  }

  // Manifold retraction — forwards to Python ``plus(x, delta, out, stream,
  // num_replicas)`` on the subclass (see StateBatch::Plus for the replica
  // layout). The GIL must be re-acquired because the C++ minimizer releases
  // it before entering its iteration loop.
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override {
    nb::gil_scoped_acquire gil;
    nb::object self_obj = nb::find(this);
    self_obj.attr("plus")(reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(delta),
                          reinterpret_cast<uintptr_t>(x_plus_delta),
                          reinterpret_cast<uintptr_t>(stream), num_replicas);
  }
};

// Floats stored per state of a SizedStateBatch subclass.
template <int A, int T>
constexpr size_t AmbientOf(const cunls::SizedStateBatch<A, T> *) {
  return static_cast<size_t>(A);
}

// The state array of a batch of `capacity` states, checked to hold
// capacity x AmbientSize float32 values (e.g. 16 per SE3 state: a row-major 4x4).
template <typename Class>
const float *StateData(nb::handle data, size_t capacity) {
  constexpr size_t ambient = AmbientOf(static_cast<const Class *>(nullptr));
  const std::string what = "capacity " + std::to_string(capacity) + " states x " +
                           std::to_string(ambient) + " float32 per state";
  return reinterpret_cast<const float *>(extract_device_ptr(
      data, "float32", "data", checked_size(capacity, ambient, "data"), what.c_str()));
}

// Checked data array of a CustomStateBatch: capacity x ambient_size float32 values.
uintptr_t CustomStateData(nb::handle data, size_t ambient_size, size_t capacity) {
  const std::string what = "capacity " + std::to_string(capacity) + " states x " +
                           std::to_string(ambient_size) + " float32 per state";
  return extract_device_ptr(data, "float32", "data", checked_size(capacity, ambient_size, "data"),
                            what.c_str());
}

// Register a VectorStateBatch<Dim> with two constructor overloads:
//   1. (data, capacity)                                  — all states are variable
//   2. (data, capacity, const_state_ids, const_capacity) — some states are held constant
// Both start with 0 active states: call set_num_active_states() before solving.
//
// nb::keep_alive<1, N> prevents the Python objects that own the GPU memory
// (e.g. CuPy arrays passed as `data`) from being garbage-collected while the
// StateBatch is alive.
template <int Dim>
void bind_vector_state_batch(nb::module_ &m, const char *name) {
  using Class = cunls::VectorStateBatch<Dim>;
  nb::class_<Class, cunls::StateBatch>(
      m, name, "Euclidean vector state batch (Plus = element-wise addition).")
      .def(
          "__init__",
          [](Class *self, nb::handle data, size_t capacity) {
            auto ptr = StateData<Class>(data, capacity);
            new (self) Class(ptr, capacity);
          },
          nb::arg("data"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](Class *self, nb::handle data, size_t capacity, nb::handle const_ids,
             size_t const_capacity) {
            auto ptr = StateData<Class>(data, capacity);
            auto cids = reinterpret_cast<const int *>(extract_device_ptr(
                const_ids, "int32", "const_ids", const_capacity, "const_capacity ids"));
            new (self) Class(ptr, capacity, cids, const_capacity);
          },
          nb::arg("data"), nb::arg("capacity"), nb::arg("const_state_ids"),
          nb::arg("const_capacity"), nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>())
      .def(
          "set_bounds",
          [](Class &self, nb::handle lower, nb::handle upper) {
            if (lower.is_none() && upper.is_none()) {
              self.SetBounds(nullptr, nullptr);
              return;
            }
            if (lower.is_none() || upper.is_none()) {
              throw std::invalid_argument(
                  "VectorStateBatch.set_bounds: got one bound and None (expected both arrays, or "
                  "None, None to remove the bounds). Fix: pass -inf / +inf arrays for an "
                  "unbounded side.");
            }
            self.SetBounds(device_floats(lower, "lower", self.Capacity(), self.TangentSize(), "state"),
                           device_floats(upper, "upper", self.Capacity(), self.TangentSize(), "state"));
          },
          nb::arg("lower").none(), nb::arg("upper").none(), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(),
          "Box bounds lower <= x <= upper per component (device float32 arrays of "
          "capacity * dim; ±inf: unbounded). Bounds are a constraint: solve the problem "
          "with AugmentedLagrangianMinimizer, which keeps the iterates in the box "
          "(projected Gauss-Newton); the other minimizers reject it. None, None removes them.")
      .def_prop_ro(
          "has_bounds", [](const Class &self) { return cunls::HasBoxBounds(&self); },
          "Whether the batch has box bounds (solve such a problem with "
          "AugmentedLagrangianMinimizer).");
}

// Register a Lie-group state batch (SE3, SO3, ...).  Same two-overload
// pattern as the vector variant.
template <typename Class>
void bind_manifold_state_batch(nb::class_<Class, cunls::StateBatch> &cls) {
  cls.def(
         "__init__",
         [](Class *self, nb::handle data, size_t capacity) {
           auto ptr = StateData<Class>(data, capacity);
           new (self) Class(ptr, capacity);
         },
         nb::arg("data"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](Class *self, nb::handle data, size_t capacity, nb::handle const_ids,
             size_t const_capacity) {
            auto ptr = StateData<Class>(data, capacity);
            auto cids = reinterpret_cast<const int *>(extract_device_ptr(
                const_ids, "int32", "const_ids", const_capacity, "const_capacity ids"));
            new (self) Class(ptr, capacity, cids, const_capacity);
          },
          nb::arg("data"), nb::arg("capacity"), nb::arg("const_state_ids"),
          nb::arg("const_capacity"), nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>());
}

}  // namespace

using cunls::LogError;  // used by THROW_ON_CUDA_ERROR

void bind_state(nb::module_ &m) {
  nb::class_<cunls::StateBatch>(m, "StateBatch",
                                "Abstract base class for batched states on a manifold.")
      .def_prop_ro("capacity", &cunls::StateBatch::Capacity,
                   "States the state buffer holds (the constructor's capacity).")
      .def_prop_ro("const_capacity", &cunls::StateBatch::ConstCapacity,
                   "Entries the constant-id buffer holds (the constructor's const_capacity).")
      .def_prop_ro("num_const_states", &cunls::StateBatch::NumConstStates,
                   "Active constant-id count: the first num_const_states entries of the "
                   "constant-id buffer are held constant (0 until set_num_active_states).")
      .def("set_num_active_states", &cunls::StateBatch::SetNumActiveStates,
           nb::arg("num_active_states"), nb::arg("num_const_states") = 0,
           "Sets the active state count (the first num_active_states states of the buffer) and "
           "the active constant-id count (num_const_states). Batches start with 0 active states: "
           "call this before the first solve, and again whenever the sizes change. Host-only; "
           "takes effect at the next minimize(). Raises ValueError above the capacity.")
      .def_prop_ro("num_active_states", &cunls::StateBatch::NumActiveStates,
                   "Active state count (set_num_active_states).")
      .def_prop_ro("tangent_size", &cunls::StateBatch::TangentSize,
                   "Degrees of freedom of one state (size of its update).")
      .def_prop_ro("ambient_size", &cunls::StateBatch::AmbientSize,
                   "float32 values stored per state (e.g. 16 for SE3: a row-major 4x4).")
      .def(
          "state_device_ptr",
          [](cunls::StateBatch &self, size_t idx) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.StateDevicePtr(idx));
          },
          nb::arg("index"), "Device address of state `index` (an int).")
      .def(
          "constant_state_ids",
          [](const cunls::StateBatch &self) {
            const size_t n = self.NumConstStates();
            auto *host = new int[n > 0 ? n : 1];
            nb::capsule owner(host, [](void *p) noexcept { delete[] static_cast<int *>(p); });
            if (n > 0) {
              THROW_ON_CUDA_ERROR(
                  cudaMemcpy(host, self.ConstStateIds(), n * sizeof(int), cudaMemcpyDeviceToHost));
            }
            return nb::ndarray<nb::numpy, int, nb::ndim<1>>(host, {n}, owner);
          },
          "The active constant state ids (num_const_states entries) as a numpy int32 array.")
      .def(
          "apply_plus",
          [](cunls::StateBatch &self, nb::handle x, nb::handle delta, nb::handle out,
             cunls::CudaStream &stream) {
            const size_t n = self.NumActiveStates();
            auto *px = reinterpret_cast<const float *>(extract_device_ptr(
                x, "float32", "x", n * self.AmbientSize(), "num_active_states x ambient_size"));
            auto *pd = reinterpret_cast<const float *>(
                extract_device_ptr(delta, "float32", "delta", n * self.TangentSize(),
                                   "num_active_states x tangent_size"));
            auto *po = reinterpret_cast<float *>(extract_device_ptr(
                out, "float32", "out", n * self.AmbientSize(), "num_active_states x ambient_size"));
            nb::gil_scoped_release release;
            self.Plus(px, pd, po, stream.GetStream());
          },
          nb::arg("x"), nb::arg("delta"), nb::arg("out"), nb::arg("stream"),
          "out = x (+) delta for the num_active_states states of the given arrays (the "
          "manifold update the minimizers use; asynchronous on stream). x, out: float32 device "
          "arrays of num_active_states x ambient_size; delta: num_active_states x tangent_size. "
          "Used by pycunls.check_jacobians.");

  bind_vector_state_batch<1>(m, "VectorStateBatch1");
  bind_vector_state_batch<2>(m, "VectorStateBatch2");
  bind_vector_state_batch<3>(m, "VectorStateBatch3");
  bind_vector_state_batch<4>(m, "VectorStateBatch4");
  bind_vector_state_batch<6>(m, "VectorStateBatch6");
  bind_vector_state_batch<12>(m, "VectorStateBatch12");

  {
    auto cls = nb::class_<cunls::SE3StateBatch, cunls::StateBatch>(
        m, "SE3StateBatch", "SE(3) state batch. Ambient=16 (4x4 matrix), Tangent=6.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::SO3StateBatch, cunls::StateBatch>(
        m, "SO3StateBatch", "SO(3) state batch. Ambient=9 (3x3 matrix), Tangent=3.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::SO2StateBatch, cunls::StateBatch>(
        m, "SO2StateBatch", "SO(2) state batch. Ambient=4 (2x2 matrix), Tangent=1.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::SE2StateBatch, cunls::StateBatch>(
        m, "SE2StateBatch", "SE(2) state batch. Ambient=9 (3x3 matrix), Tangent=3.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::Similarity2StateBatch, cunls::StateBatch>(
        m, "Similarity2StateBatch", "2D similarity state batch. Ambient=9, Tangent=4.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::Similarity3StateBatch, cunls::StateBatch>(
        m, "Similarity3StateBatch", "3D similarity state batch. Ambient=16, Tangent=7.");
    bind_manifold_state_batch(cls);
  }
  {
    auto cls = nb::class_<cunls::SL4StateBatch, cunls::StateBatch>(
        m, "SL4StateBatch", "SL(4) state batch. Ambient=16 (4x4), Tangent=15.");
    bind_manifold_state_batch(cls);
  }

  // --- Custom state batch trampoline ---
  nb::class_<PyStateBatch, cunls::StateBatch>(
      m, "CustomStateBatch",
      "Base class for user-defined state batches. Override plus() in "
      "Python.\n\n"
      "The plus() method implements the manifold retraction:\n"
      "  x_plus_delta = x (+) delta\n"
      "All pointers are passed as integer handles.\n\n"
      "Constructor: data (float32 GPU array of capacity x ambient_size), ambient_size (floats "
      "stored per state), tangent_size (degrees of freedom), capacity (states the array "
      "holds), and optionally const_state_ids (int32 GPU array) with const_capacity (entries "
      "it holds).")
      .def(
          "__init__",
          [](PyStateBatch *self, nb::handle data, size_t ambient_size, size_t tangent_size,
             size_t capacity) {
            auto ptr = CustomStateData(data, ambient_size, capacity);
            new (self) PyStateBatch(ptr, ambient_size, tangent_size, capacity, 0, 0);
          },
          nb::arg("data"), nb::arg("ambient_size"), nb::arg("tangent_size"), nb::arg("capacity"),
          nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](PyStateBatch *self, nb::handle data, size_t ambient_size, size_t tangent_size,
             size_t capacity, nb::handle const_ids, size_t const_capacity) {
            auto ptr = CustomStateData(data, ambient_size, capacity);
            auto cids = extract_device_ptr(const_ids, "int32", "const_ids", const_capacity,
                                           "const_capacity ids");
            new (self)
                PyStateBatch(ptr, ambient_size, tangent_size, capacity, cids, const_capacity);
          },
          nb::arg("data"), nb::arg("ambient_size"), nb::arg("tangent_size"), nb::arg("capacity"),
          nb::arg("const_state_ids"), nb::arg("const_capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 6>())
      .def(
          "plus",
          [](PyStateBatch &, uintptr_t, uintptr_t, uintptr_t, uintptr_t, size_t) {
            throw std::runtime_error("CustomStateBatch.plus() must be overridden in a subclass.");
          },
          nb::arg("x_ptr"), nb::arg("delta_ptr"), nb::arg("x_plus_delta_ptr"),
          nb::arg("stream_handle"), nb::arg("num_replicas"),
          "Override in a subclass: x_plus_delta = x (+) delta for num_replicas x "
          "num_active_states states (int device pointers; enqueue on stream_handle). See "
          "pycunls.warp.WarpStateBatch for a Warp-based base class.");
}
