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

// Bindings for all StateBatch subclasses.
//
// cuNLS represents optimisation variables as "state batches": contiguous GPU
// arrays that hold N blocks of a single geometric type.  Each batch knows its
// manifold structure (ambient dimension, tangent dimension, retraction/Plus).
//
// Two families exist:
//   - VectorStateBatch<Dim>  — Euclidean R^n where Plus is simple addition.
//   - Manifold batches       — Lie-group types (SE3, SO3, SO2, SE2, Sim2, Sim3, SL4)
//                              that require a cuBLAS handle for their retraction
//                              and whose ambient/tangent dimensions differ.
//
// Both families expose the same Python interface (two constructor overloads,
// state_block_device_ptr(), and read-only num/tangent/ambient properties).
// The template helpers below factor out the repetitive nanobind boilerplate.

#include "bindings.h"

#include "cunls/state/vector_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"

namespace {

// Python trampoline for cunls::StateBatch.
//
// Implements all StateBatch pure-virtual methods in C++, except Plus()
// which acquires the GIL and delegates to Python's ``plus()`` method.
// This is the state-batch counterpart of PyFactorBatch in bind_factor.cpp.
//
// Storage layout mirrors SizedStateBatch: a contiguous GPU buffer of
// num_blocks * ambient_size floats, with optional const-state indices.
class PyStateBatch : public cunls::StateBatch {
 public:
    const float* ptr_;
    size_t ambient_size_;
    size_t tangent_size_;
    size_t num_blocks_;
    const int* const_ids_;
    size_t num_const_;

    PyStateBatch(uintptr_t data_ptr, size_t ambient_size, size_t tangent_size,
                 size_t num_blocks, uintptr_t const_ids_ptr, size_t num_const)
        : ptr_(reinterpret_cast<const float*>(data_ptr)),
          ambient_size_(ambient_size),
          tangent_size_(tangent_size),
          num_blocks_(num_blocks),
          const_ids_(reinterpret_cast<const int*>(const_ids_ptr)),
          num_const_(num_const) {}

    size_t TangentSize() const override { return tangent_size_; }
    size_t AmbientSize() const override { return ambient_size_; }
    size_t NumStateBlocks() const override { return num_blocks_; }

    float* StateBlockDevicePtr(size_t idx) override {
        if (idx >= num_blocks_) return nullptr;
        return const_cast<float*>(ptr_ + idx * ambient_size_);
    }

    const float* StateBlockDevicePtr(size_t idx) const override {
        if (idx >= num_blocks_) return nullptr;
        return ptr_ + idx * ambient_size_;
    }

    const int* ConstStateIds() const override { return const_ids_; }
    size_t NumConstStateBlocks() const override { return num_const_; }

    // Manifold retraction — forwards to Python ``plus()`` on the subclass.
    // The GIL must be re-acquired because the C++ minimizer releases it
    // before entering its iteration loop.
    void Plus(const float* x, const float* delta, float* x_plus_delta,
              cudaStream_t stream) override {
        nb::gil_scoped_acquire gil;
        nb::object self_obj = nb::find(this);
        self_obj.attr("plus")(
            reinterpret_cast<uintptr_t>(x),
            reinterpret_cast<uintptr_t>(delta),
            reinterpret_cast<uintptr_t>(x_plus_delta),
            reinterpret_cast<uintptr_t>(stream));
    }
};

// Register a VectorStateBatch<Dim> with two constructor overloads:
//   1. (data, num_blocks)                     — all blocks are variable
//   2. (data, num_blocks, const_ids, num_const) — some blocks are held constant
//
// nb::keep_alive<1, N> prevents the Python objects that own the GPU memory
// (e.g. CuPy arrays passed as `data`) from being garbage-collected while the
// StateBatch is alive.
template <int Dim>
void bind_vector_state_batch(nb::module_& m, const char* name) {
    using Class = cunls::VectorStateBatch<Dim>;
    nb::class_<Class, cunls::StateBatch>(m, name,
        "Euclidean vector state batch (Plus = element-wise addition).")
        .def("__init__", [](Class* self, nb::handle data, size_t num_blocks) {
            auto ptr = reinterpret_cast<const float*>(extract_device_ptr(data));
            new (self) Class(ptr, num_blocks);
        }, nb::arg("data"), nb::arg("num_blocks"),
           nb::keep_alive<1, 2>())
        .def("__init__", [](Class* self, nb::handle data, size_t num_blocks,
                            nb::handle const_ids, size_t num_const) {
            auto ptr = reinterpret_cast<const float*>(extract_device_ptr(data));
            auto cids = reinterpret_cast<const int*>(extract_device_ptr(const_ids));
            new (self) Class(ptr, num_blocks, cids, num_const);
        }, nb::arg("data"), nb::arg("num_blocks"),
           nb::arg("const_state_ids"), nb::arg("num_const_state_blocks"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>())
        .def("state_block_device_ptr", [](Class& self, size_t idx) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.StateBlockDevicePtr(idx));
        }, nb::arg("index"))
        .def_prop_ro("num_state_blocks", &Class::NumStateBlocks)
        .def_prop_ro("tangent_size", &Class::TangentSize)
        .def_prop_ro("ambient_size", &Class::AmbientSize);
}

// Register a Lie-group state batch (SE3, SO3, ...).  Same two-overload
// pattern as the vector variant, but the first constructor argument is always
// a CublasHandle required by the manifold retraction kernels.
template <typename Class>
void bind_manifold_state_batch(nb::class_<Class, cunls::StateBatch>& cls) {
    cls.def("__init__", [](Class* self, cunls::cuBLASHandle& cublas,
                           nb::handle data, size_t num_blocks) {
            auto ptr = reinterpret_cast<const float*>(extract_device_ptr(data));
            new (self) Class(cublas, ptr, num_blocks);
        }, nb::arg("cublas_handle"), nb::arg("data"), nb::arg("num_blocks"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("__init__", [](Class* self, cunls::cuBLASHandle& cublas,
                            nb::handle data, size_t num_blocks,
                            nb::handle const_ids, size_t num_const) {
            auto ptr = reinterpret_cast<const float*>(extract_device_ptr(data));
            auto cids = reinterpret_cast<const int*>(extract_device_ptr(const_ids));
            new (self) Class(cublas, ptr, num_blocks, cids, num_const);
        }, nb::arg("cublas_handle"), nb::arg("data"), nb::arg("num_blocks"),
           nb::arg("const_state_ids"), nb::arg("num_const_state_blocks"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 5>())
        .def("state_block_device_ptr", [](Class& self, size_t idx) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.StateBlockDevicePtr(idx));
        }, nb::arg("index"))
        .def_prop_ro("num_state_blocks", &Class::NumStateBlocks)
        .def_prop_ro("tangent_size", &Class::TangentSize)
        .def_prop_ro("ambient_size", &Class::AmbientSize);
}

}  // namespace

void bind_state(nb::module_& m) {
    nb::class_<cunls::StateBatch>(m, "StateBatch",
        "Abstract base class for batched state blocks on a manifold.");

    bind_vector_state_batch<1>(m, "VectorStateBatch1");
    bind_vector_state_batch<2>(m, "VectorStateBatch2");
    bind_vector_state_batch<3>(m, "VectorStateBatch3");
    bind_vector_state_batch<6>(m, "VectorStateBatch6");

    {
        auto cls = nb::class_<cunls::SE3StateBatch, cunls::StateBatch>(m,
            "SE3StateBatch",
            "SE(3) state batch. Ambient=16 (4x4 matrix), Tangent=6.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::SO3StateBatch, cunls::StateBatch>(m,
            "SO3StateBatch",
            "SO(3) state batch. Ambient=9 (3x3 matrix), Tangent=3.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::SO2StateBatch, cunls::StateBatch>(m,
            "SO2StateBatch",
            "SO(2) state batch. Ambient=4 (2x2 matrix), Tangent=1.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::SE2StateBatch, cunls::StateBatch>(m,
            "SE2StateBatch",
            "SE(2) state batch. Ambient=9 (3x3 matrix), Tangent=3.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::Similarity2StateBatch, cunls::StateBatch>(m,
            "Similarity2StateBatch",
            "2D similarity state batch. Ambient=9, Tangent=4.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::Similarity3StateBatch, cunls::StateBatch>(m,
            "Similarity3StateBatch",
            "3D similarity state batch. Ambient=16, Tangent=7.");
        bind_manifold_state_batch(cls);
    }
    {
        auto cls = nb::class_<cunls::SL4StateBatch, cunls::StateBatch>(m,
            "SL4StateBatch",
            "SL(4) state batch. Ambient=16 (4x4), Tangent=15.");
        bind_manifold_state_batch(cls);
    }

    // --- Custom state batch trampoline ---
    nb::class_<PyStateBatch, cunls::StateBatch>(m, "CustomStateBatch",
        "Base class for user-defined state batches. Override plus() in Python.\n\n"
        "The plus() method implements the manifold retraction:\n"
        "  x_plus_delta = x (+) delta\n"
        "All pointers are passed as integer handles.")
        .def("__init__", [](PyStateBatch* self, nb::handle data,
                            size_t ambient_size, size_t tangent_size,
                            size_t num_blocks) {
            auto ptr = extract_device_ptr(data);
            new (self) PyStateBatch(ptr, ambient_size, tangent_size,
                                    num_blocks, 0, 0);
        }, nb::arg("data"), nb::arg("ambient_size"), nb::arg("tangent_size"),
           nb::arg("num_blocks"),
           nb::keep_alive<1, 2>())
        .def("__init__", [](PyStateBatch* self, nb::handle data,
                            size_t ambient_size, size_t tangent_size,
                            size_t num_blocks,
                            nb::handle const_ids, size_t num_const) {
            auto ptr = extract_device_ptr(data);
            auto cids = extract_device_ptr(const_ids);
            new (self) PyStateBatch(ptr, ambient_size, tangent_size,
                                    num_blocks, cids, num_const);
        }, nb::arg("data"), nb::arg("ambient_size"), nb::arg("tangent_size"),
           nb::arg("num_blocks"),
           nb::arg("const_state_ids"), nb::arg("num_const_state_blocks"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 6>())
        .def("plus", [](PyStateBatch&, uintptr_t, uintptr_t,
                        uintptr_t, uintptr_t) {
            throw std::runtime_error(
                "CustomStateBatch.plus() must be overridden in a subclass.");
        }, nb::arg("x_ptr"), nb::arg("delta_ptr"),
           nb::arg("x_plus_delta_ptr"), nb::arg("stream_handle"))
        .def("state_block_device_ptr", [](PyStateBatch& self,
                                          size_t idx) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.StateBlockDevicePtr(idx));
        }, nb::arg("index"))
        .def_prop_ro("num_state_blocks", &PyStateBatch::NumStateBlocks)
        .def_prop_ro("tangent_size", &PyStateBatch::TangentSize)
        .def_prop_ro("ambient_size", &PyStateBatch::AmbientSize);
}
