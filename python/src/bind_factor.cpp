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

// Bindings for all FactorBatch subclasses and the CustomFactorBatch trampoline.
//
// A FactorBatch represents a batch of N residual functions that share the same
// structure (residual dimension, number and sizes of states).  cuNLS
// ships many built-in factor types (reprojection, SE3 between, ICP variants,
// priors); each is bound below as a nanobind class that inherits FactorBatch.
//
// User-defined factors are supported via the "trampoline" pattern:
// PyFactorBatch is a C++ class that inherits cunls::FactorBatch and overrides
// Evaluate() to call back into a Python method.  On the Python side it is
// exposed as CustomFactorBatch; users subclass it and implement evaluate().

#include <nanobind/stl/vector.h>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/clearance/se2_disk_clearance_factor_batch.h"
#include "cunls/factor/clearance/se3_sphere_clearance_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/dynamics/quadrotor_factor_batch.h"
#include "cunls/factor/dynamics/quadruped_factor_batch.h"
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
#include "cunls/factor/dynamics/se2_kinematic_bicycle_factor_batch.h"
#include "cunls/factor/dynamics/se2_kinematics_factor_batch.h"
#include "cunls/factor/dynamics/se3_differential_drive_factor_batch.h"
#include "cunls/factor/dynamics/se3_kinematic_bicycle_factor_batch.h"
#include "cunls/factor/dynamics/se3_kinematics_factor_batch.h"
#include "cunls/factor/dynamics/so3_kinematics_factor_batch.h"
#include "cunls/factor/halfspace_factor_batch.h"
#include "cunls/factor/imu_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/prior/so2_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/sized_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "py_factor_wrappers.h"

namespace {

// Template helper to bind PriorVectorFactorBatch<Dim> for a given dimension.
template <int Dim>
void bind_prior_vector_factor(nb::module_ &m, const char *name) {
  using Class = cunls::PriorVectorFactorBatch<Dim>;
  nb::class_<Class, cunls::FactorBatch>(
      m, name,
      "Prior factor for Dim-dimensional vectors: residual = state - "
      "observation.")
      .def(
          "__init__",
          [](Class *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::Vector<Dim> *>(extract_device_ptr(observations));
            new (self) Class(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &Class::NumActiveFactors)
      .def_prop_ro("residuals_size", &Class::ResidualsSize)
      .def("state_sizes", &Class::StateSizes);
}

template <int Dim>
void bind_vector_between_factor(nb::module_ &m, const char *name) {
  using Class = cunls::VectorBetweenFactorBatch<Dim>;
  nb::class_<Class, cunls::FactorBatch>(
      m, name, "Between factor on Euclidean vectors: residual = left - right - delta.")
      .def(
          "__init__",
          [](Class *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::Vector<Dim> *>(extract_device_ptr(deltas));
            new (self) Class(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &Class::NumActiveFactors)
      .def_prop_ro("residuals_size", &Class::ResidualsSize)
      .def("state_sizes", &Class::StateSizes);
}

template <int Dim>
void bind_bound_factor(nb::module_ &m, const char *name) {
  using Class = cunls::BoundFactorBatch<Dim>;
  nb::class_<Class, cunls::ConstraintFactorBatchBase>(
      m, name,
      "Box constraints lower <= x <= upper on the components of a vector state (an "
      "inequality constraint batch: add it to the problem directly and solve with "
      "AugmentedLagrangianMinimizer).\n\n"
      "Parameters\n"
      "----------\n"
      "lower, upper : DevicePointer\n"
      "    Device buffers of capacity * Dim floats (per factor and component); -inf / +inf\n"
      "    leave a side unbounded.\n"
      "capacity : int\n"
      "    Number of factors the buffers hold.\n"
      "scale : float\n"
      "    Positive row scale.")
      .def(
          "__init__",
          [](Class *self, nb::handle lower, nb::handle upper, size_t capacity, float scale) {
            new (self)
                Class(reinterpret_cast<const float *>(extract_device_ptr(lower)),
                      reinterpret_cast<const float *>(extract_device_ptr(upper)), capacity, scale);
          },
          nb::arg("lower"), nb::arg("upper"), nb::arg("capacity"), nb::arg("scale") = 1.f,
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &Class::NumActiveFactors)
      .def_prop_ro("residuals_size", &Class::ResidualsSize)
      .def("state_sizes", &Class::StateSizes);
}

template <int Dim>
void bind_halfspace_factor(nb::module_ &m, const char *name) {
  using Class = cunls::HalfspaceFactorBatch<Dim>;
  nb::class_<Class, cunls::FactorBatch>(
      m, name,
      "Signed halfspace value r = a^T x - b of a vector state (per-factor a, b). Wrap it in "
      "ConstraintFactorBatch(..., ConstraintKind.Inequality) for a^T x <= b.\n\n"
      "Parameters\n"
      "----------\n"
      "normals : DevicePointer\n"
      "    Device buffer of capacity * Dim floats (a).\n"
      "offsets : DevicePointer\n"
      "    Device buffer of capacity floats (b).\n"
      "capacity : int\n"
      "    Number of factors the buffers hold.")
      .def(
          "__init__",
          [](Class *self, nb::handle normals, nb::handle offsets, size_t capacity) {
            new (self)
                Class(reinterpret_cast<const float *>(extract_device_ptr(normals)),
                      reinterpret_cast<const float *>(extract_device_ptr(offsets)), capacity);
          },
          nb::arg("normals"), nb::arg("offsets"), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &Class::NumActiveFactors)
      .def_prop_ro("residuals_size", &Class::ResidualsSize)
      .def("state_sizes", &Class::StateSizes);
}

// Lie-group kinematics factors: (time_steps, capacity) constructors.
template <class Class>
void bind_kinematics_factor(nb::module_ &m, const char *name, const char *doc) {
  nb::class_<Class, cunls::FactorBatch>(m, name, doc)
      .def(
          "__init__",
          [](Class *self, nb::handle time_steps, size_t capacity) {
            new (self)
                Class(reinterpret_cast<const float *>(extract_device_ptr(time_steps)), capacity);
          },
          nb::arg("time_steps"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &Class::NumActiveFactors)
      .def_prop_ro("residuals_size", &Class::ResidualsSize)
      .def("state_sizes", &Class::StateSizes);
}

}  // namespace

void bind_factor(nb::module_ &m) {
  nb::class_<cunls::FactorBatch>(m, "FactorBatch", "Abstract base class for batched factors.")
      .def_prop_ro("capacity", &cunls::FactorBatch::Capacity,
                   "Factors the measurement buffers hold (the constructor's capacity).")
      .def("set_num_active_factors", &cunls::FactorBatch::SetNumActiveFactors,
           nb::arg("num_active_factors"),
           "Sets the active factor count (the first num_active_factors measurements are used). "
           "Factor batches start with 0 active factors: call this before the first solve, and "
           "again whenever the size changes. Host-only; takes effect at the next minimize(). "
           "Raises ValueError above the capacity.");

  // --- Custom factor trampoline ---
  nb::class_<PyFactorBatch, cunls::FactorBatch>(
      m, "CustomFactorBatch", "Base class for user-defined factors. Override evaluate() in Python.")
      .def(nb::init<size_t, std::vector<size_t>, size_t>(), nb::arg("residual_size"),
           nb::arg("state_sizes"), nb::arg("capacity"))
      .def(
          "evaluate",
          [](PyFactorBatch &, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
             size_t) -> bool {
            throw std::runtime_error(
                "CustomFactorBatch.evaluate() must be "
                "overridden in a subclass.");
          },
          nb::arg("residuals_ptr"), nb::arg("jacobians_ptr"), nb::arg("state_pointers_ptr"),
          nb::arg("stream_handle"), nb::arg("factor_ids_ptr"), nb::arg("num_factor_ids"))
      .def_prop_ro("num_active_factors", &PyFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &PyFactorBatch::ResidualsSize)
      .def("state_sizes", &PyFactorBatch::StateSizes);

  // --- Reprojection ---
  nb::class_<cunls::ReprojectionFactorBatch, cunls::FactorBatch>(
      m, "ReprojectionFactorBatch",
      "Batched 2D reprojection factor. Residual=2, States=[SE3(6), Point(3)].\n"
      "Observations must be in normalized image coordinates (K^-1 applied).")
      .def(
          "__init__",
          [](cunls::ReprojectionFactorBatch *self, nb::handle observations, size_t capacity,
             float z_threshold) {
            auto ptr = reinterpret_cast<const cunls::Vector<2> *>(extract_device_ptr(observations));
            new (self) cunls::ReprojectionFactorBatch(ptr, capacity, z_threshold);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::arg("z_threshold") = 1e-3f,
          nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::ReprojectionFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::ReprojectionFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::ReprojectionFactorBatch::StateSizes);

  nb::class_<cunls::PnPFactorBatch, cunls::FactorBatch>(
      m, "PnPFactorBatch",
      "Batched PnP reprojection: fixed 3D points, pose-only Jacobian.\n"
      "Residual=2, States=[SE3(6)]. Observations in normalized image coords.\n"
      "3D points are passed at construction (device); not optimized.")
      .def(
          "__init__",
          [](cunls::PnPFactorBatch *self, nb::handle observations, nb::handle points_world,
             size_t capacity, float z_threshold) {
            auto obs_ptr =
                reinterpret_cast<const cunls::Vector<2> *>(extract_device_ptr(observations));
            auto p_ptr =
                reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(points_world));
            new (self) cunls::PnPFactorBatch(obs_ptr, p_ptr, capacity, z_threshold);
          },
          nb::arg("observations"), nb::arg("points_world"), nb::arg("capacity"),
          nb::arg("z_threshold") = 1e-3f, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def(
          "__init__",
          [](cunls::PnPFactorBatch *self, nb::handle observations, nb::handle poses_camera_from_rig,
             nb::handle points_world, size_t capacity, float z_threshold) {
            auto obs_ptr =
                reinterpret_cast<const cunls::Vector<2> *>(extract_device_ptr(observations));
            auto rig_ptr = reinterpret_cast<const cunls::SE3Transform *>(
                extract_device_ptr(poses_camera_from_rig));
            auto p_ptr =
                reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(points_world));
            new (self) cunls::PnPFactorBatch(obs_ptr, rig_ptr, p_ptr, capacity, z_threshold);
          },
          nb::arg("observations"), nb::arg("poses_camera_from_rig"), nb::arg("points_world"),
          nb::arg("capacity"), nb::arg("z_threshold") = 1e-3f, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def_prop_ro("num_active_factors", &cunls::PnPFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::PnPFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::PnPFactorBatch::StateSizes);

  // --- SE3 Between ---
  nb::class_<cunls::SE3BetweenFactorBatch, cunls::FactorBatch>(
      m, "SE3BetweenFactorBatch",
      "Batched SE(3) between factor. Residual=6, States=[SE3(6), SE3(6)].")
      .def(
          "__init__",
          [](cunls::SE3BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::SE3Transform *>(extract_device_ptr(deltas));
            new (self) cunls::SE3BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE3BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE3BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE3BetweenFactorBatch::StateSizes);

  // --- SE2 Between ---
  nb::class_<cunls::SE2BetweenFactorBatch, cunls::FactorBatch>(
      m, "SE2BetweenFactorBatch",
      "Batched SE(2) between factor. Residual=3, States=[SE2(3), SE2(3)].")
      .def(
          "__init__",
          [](cunls::SE2BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::SE2Transform *>(extract_device_ptr(deltas));
            new (self) cunls::SE2BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE2BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE2BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE2BetweenFactorBatch::StateSizes);

  // --- SO2 Between ---
  nb::class_<cunls::SO2BetweenFactorBatch, cunls::FactorBatch>(
      m, "SO2BetweenFactorBatch",
      "Batched SO(2) between factor. Residual=1, States=[SO2(1), SO2(1)].")
      .def(
          "__init__",
          [](cunls::SO2BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::SO2Rotation *>(extract_device_ptr(deltas));
            new (self) cunls::SO2BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SO2BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SO2BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SO2BetweenFactorBatch::StateSizes);

  // --- SO3 Between ---
  nb::class_<cunls::SO3BetweenFactorBatch, cunls::FactorBatch>(
      m, "SO3BetweenFactorBatch",
      "Batched SO(3) between factor. Residual=3, States=[SO3(3), SO3(3)].")
      .def(
          "__init__",
          [](cunls::SO3BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::SO3Rotation *>(extract_device_ptr(deltas));
            new (self) cunls::SO3BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SO3BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SO3BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SO3BetweenFactorBatch::StateSizes);

  // --- Similarity2 Between ---
  nb::class_<cunls::Similarity2BetweenFactorBatch, cunls::FactorBatch>(
      m, "Similarity2BetweenFactorBatch",
      "Batched Sim(2) between factor. Residual=4, States=[Sim2(4), Sim2(4)].")
      .def(
          "__init__",
          [](cunls::Similarity2BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::Similarity2Transform *>(extract_device_ptr(deltas));
            new (self) cunls::Similarity2BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::Similarity2BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::Similarity2BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::Similarity2BetweenFactorBatch::StateSizes);

  // --- Similarity3 Between ---
  nb::class_<cunls::Similarity3BetweenFactorBatch, cunls::FactorBatch>(
      m, "Similarity3BetweenFactorBatch",
      "Batched Sim(3) between factor. Residual=7, States=[Sim3(7), Sim3(7)].")
      .def(
          "__init__",
          [](cunls::Similarity3BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::Similarity3Transform *>(extract_device_ptr(deltas));
            new (self) cunls::Similarity3BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::Similarity3BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::Similarity3BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::Similarity3BetweenFactorBatch::StateSizes);

  // --- SL4 Between ---
  nb::class_<cunls::SL4BetweenFactorBatch, cunls::FactorBatch>(
      m, "SL4BetweenFactorBatch",
      "Batched SL(4) between factor. Residual=15, States=[SL4(15), SL4(15)].")
      .def(
          "__init__",
          [](cunls::SL4BetweenFactorBatch *self, nb::handle deltas, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::SL4Transform *>(extract_device_ptr(deltas));
            new (self) cunls::SL4BetweenFactorBatch(ptr, capacity);
          },
          nb::arg("deltas"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SL4BetweenFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SL4BetweenFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SL4BetweenFactorBatch::StateSizes);

  // --- SE3 Prior ---
  nb::class_<cunls::SE3PriorFactorBatch, cunls::FactorBatch>(
      m, "SE3PriorFactorBatch", "Batched SE(3) prior factor. Residual=6, States=[SE3(6)].")
      .def(
          "__init__",
          [](cunls::SE3PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::SE3Transform *>(extract_device_ptr(observations));
            new (self) cunls::SE3PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE3PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE3PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE3PriorFactorBatch::StateSizes);

  // --- SL4 Prior ---
  nb::class_<cunls::SL4PriorFactorBatch, cunls::FactorBatch>(
      m, "SL4PriorFactorBatch", "Batched SL(4) prior factor. Residual=15, States=[SL4(15)].")
      .def(
          "__init__",
          [](cunls::SL4PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::SL4Transform *>(extract_device_ptr(observations));
            new (self) cunls::SL4PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SL4PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SL4PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SL4PriorFactorBatch::StateSizes);

  // --- SO3 Prior ---
  nb::class_<cunls::SO3PriorFactorBatch, cunls::FactorBatch>(
      m, "SO3PriorFactorBatch", "Batched SO(3) prior factor. Residual=3, States=[SO3(3)].")
      .def(
          "__init__",
          [](cunls::SO3PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::SO3Rotation *>(extract_device_ptr(observations));
            new (self) cunls::SO3PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SO3PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SO3PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SO3PriorFactorBatch::StateSizes);

  // --- SO2 Prior ---
  nb::class_<cunls::SO2PriorFactorBatch, cunls::FactorBatch>(
      m, "SO2PriorFactorBatch", "Batched SO(2) prior factor. Residual=1, States=[SO2(1)].")
      .def(
          "__init__",
          [](cunls::SO2PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::SO2Rotation *>(extract_device_ptr(observations));
            new (self) cunls::SO2PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SO2PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SO2PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SO2PriorFactorBatch::StateSizes);

  // --- SE2 Prior ---
  nb::class_<cunls::SE2PriorFactorBatch, cunls::FactorBatch>(
      m, "SE2PriorFactorBatch", "Batched SE(2) prior factor. Residual=3, States=[SE2(3)].")
      .def(
          "__init__",
          [](cunls::SE2PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr =
                reinterpret_cast<const cunls::SE2Transform *>(extract_device_ptr(observations));
            new (self) cunls::SE2PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE2PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE2PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE2PriorFactorBatch::StateSizes);

  // --- Similarity2 Prior ---
  nb::class_<cunls::Similarity2PriorFactorBatch, cunls::FactorBatch>(
      m, "Similarity2PriorFactorBatch",
      "Batched Sim(2) prior factor. Residual=4, States=[Similarity2(4)].")
      .def(
          "__init__",
          [](cunls::Similarity2PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::Similarity2Transform *>(
                extract_device_ptr(observations));
            new (self) cunls::Similarity2PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::Similarity2PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::Similarity2PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::Similarity2PriorFactorBatch::StateSizes);

  // --- Similarity3 Prior ---
  nb::class_<cunls::Similarity3PriorFactorBatch, cunls::FactorBatch>(
      m, "Similarity3PriorFactorBatch",
      "Batched Sim(3) prior factor. Residual=7, States=[Similarity3(7)].")
      .def(
          "__init__",
          [](cunls::Similarity3PriorFactorBatch *self, nb::handle observations, size_t capacity) {
            auto ptr = reinterpret_cast<const cunls::Similarity3Transform *>(
                extract_device_ptr(observations));
            new (self) cunls::Similarity3PriorFactorBatch(ptr, capacity);
          },
          nb::arg("observations"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::Similarity3PriorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::Similarity3PriorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::Similarity3PriorFactorBatch::StateSizes);

  // --- Prior Vector Factors ---
  bind_prior_vector_factor<1>(m, "PriorVectorFactorBatch1");
  bind_prior_vector_factor<2>(m, "PriorVectorFactorBatch2");
  bind_prior_vector_factor<3>(m, "PriorVectorFactorBatch3");
  bind_prior_vector_factor<4>(m, "PriorVectorFactorBatch4");
  bind_prior_vector_factor<6>(m, "PriorVectorFactorBatch6");
  bind_prior_vector_factor<12>(m, "PriorVectorFactorBatch12");

  bind_vector_between_factor<1>(m, "VectorBetweenFactorBatch1");
  bind_vector_between_factor<2>(m, "VectorBetweenFactorBatch2");
  bind_vector_between_factor<3>(m, "VectorBetweenFactorBatch3");
  bind_vector_between_factor<4>(m, "VectorBetweenFactorBatch4");
  bind_vector_between_factor<6>(m, "VectorBetweenFactorBatch6");
  bind_vector_between_factor<12>(m, "VectorBetweenFactorBatch12");

  // --- Point-to-Point ---
  nb::class_<cunls::PointToPointFactorBatch, cunls::FactorBatch>(
      m, "PointToPointFactorBatch",
      "Batched point-to-point factor: residual = p - T*q. Residual=3, "
      "States=[SE3(6)].")
      .def(
          "__init__",
          [](cunls::PointToPointFactorBatch *self, nb::handle p_obs, nb::handle q_obs,
             size_t capacity) {
            auto p = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(q_obs));
            new (self) cunls::PointToPointFactorBatch(p, q, capacity);
          },
          nb::arg("p_observations"), nb::arg("q_observations"), nb::arg("capacity"),
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &cunls::PointToPointFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::PointToPointFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::PointToPointFactorBatch::StateSizes);

  // --- Point-to-Plane ---
  nb::class_<cunls::PointToPlaneFactorBatch, cunls::FactorBatch>(
      m, "PointToPlaneFactorBatch",
      "Batched point-to-plane factor: residual = Nq^T*(p - T*q). Residual=1, "
      "States=[SE3(6)].")
      .def(
          "__init__",
          [](cunls::PointToPlaneFactorBatch *self, nb::handle p_obs, nb::handle q_obs,
             nb::handle nq_obs, size_t capacity) {
            auto p = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(q_obs));
            auto nq = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(nq_obs));
            new (self) cunls::PointToPlaneFactorBatch(p, q, nq, capacity);
          },
          nb::arg("p_observations"), nb::arg("q_observations"), nb::arg("nq_observations"),
          nb::arg("capacity"), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
          nb::keep_alive<1, 4>())
      .def_prop_ro("num_active_factors", &cunls::PointToPlaneFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::PointToPlaneFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::PointToPlaneFactorBatch::StateSizes);

  // --- Symmetric Point-to-Plane ---
  nb::class_<cunls::SymmetricPointToPlaneFactorBatch, cunls::FactorBatch>(
      m, "SymmetricPointToPlaneFactorBatch",
      "Batched symmetric point-to-plane factor. Residual=1, States=[SE3(6)].")
      .def(
          "__init__",
          [](cunls::SymmetricPointToPlaneFactorBatch *self, nb::handle p_obs, nb::handle q_obs,
             nb::handle np_obs, nb::handle nq_obs, size_t capacity) {
            auto p = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(q_obs));
            auto np = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(np_obs));
            auto nq = reinterpret_cast<const cunls::Vector<3> *>(extract_device_ptr(nq_obs));
            new (self) cunls::SymmetricPointToPlaneFactorBatch(p, q, np, nq, capacity);
          },
          nb::arg("p_observations"), nb::arg("q_observations"), nb::arg("np_observations"),
          nb::arg("nq_observations"), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>())
      .def_prop_ro("num_active_factors", &cunls::SymmetricPointToPlaneFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SymmetricPointToPlaneFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SymmetricPointToPlaneFactorBatch::StateSizes);

  // --- InformationFactorBatch (polymorphic wrapper) ---
  nb::class_<PyInformationFactorBatch, cunls::FactorBatch>(
      m, "InformationFactorBatch",
      "Wraps any factor batch and applies per-factor sqrt-information "
      "matrices.\n\n"
      "Residuals and Jacobians are left-multiplied by the corresponding\n"
      "square-root information matrix: r' = Omega^{1/2} r, J' = Omega^{1/2} "
      "J.\n\n"
      "Parameters\n"
      "----------\n"
      "inner_factor : FactorBatch\n"
      "    The factor batch to wrap.\n"
      "sqrt_information_matrices : DevicePointer\n"
      "    Device buffer with one square-root information matrix per inner "
      "factor\n"
      "    (``inner_factor.capacity`` matrices), each residual_size x "
      "residual_size,\n"
      "    stored contiguously in row-major order.")
      .def(
          "__init__",
          [](PyInformationFactorBatch *self, cunls::FactorBatch *inner, nb::handle sqrt_info) {
            auto ptr = reinterpret_cast<const float *>(extract_device_ptr(sqrt_info));
            new (self) PyInformationFactorBatch(inner, ptr);
          },
          nb::arg("inner_factor"), nb::arg("sqrt_information_matrices"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &PyInformationFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &PyInformationFactorBatch::ResidualsSize)
      .def("state_sizes", &PyInformationFactorBatch::StateSizes);

  // --- WeightedFactorBatch (polymorphic wrapper) ---
  nb::class_<PyWeightedFactorBatch, cunls::FactorBatch>(
      m, "WeightedFactorBatch",
      "Wraps any factor batch and scales residuals/Jacobians by a weight.\n\n"
      "Supports two modes:\n"
      "  1. Uniform weight (float): every factor is scaled equally.\n"
      "  2. Per-factor weights (DevicePointer): each factor gets its own "
      "weight.\n\n"
      "Parameters\n"
      "----------\n"
      "inner_factor : FactorBatch\n"
      "    The factor batch to wrap.\n"
      "weight : float, optional\n"
      "    Uniform scalar weight applied to all factors.\n"
      "weights : DevicePointer, optional, keyword-only\n"
      "    Device buffer with ``inner_factor.capacity`` floats (one per "
      "factor).\n\n"
      "Exactly one of ``weight`` or ``weights`` must be provided.")
      .def(
          "__init__",
          [](PyWeightedFactorBatch *self, cunls::FactorBatch *inner, float weight) {
            new (self) PyWeightedFactorBatch(inner, weight);
          },
          nb::arg("inner_factor"), nb::arg("weight"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](PyWeightedFactorBatch *self, cunls::FactorBatch *inner, nb::handle weights) {
            auto ptr = reinterpret_cast<const float *>(extract_device_ptr(weights));
            new (self) PyWeightedFactorBatch(inner, ptr);
          },
          // Keyword-only, so a positional number always selects the scalar `weight`
          // overload instead of being read as a device pointer.
          nb::arg("inner_factor"), nb::kw_only(), nb::arg("weights"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &PyWeightedFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &PyWeightedFactorBatch::ResidualsSize)
      .def("state_sizes", &PyWeightedFactorBatch::StateSizes);

  // --- Constraints (augmented Lagrangian; solved by AugmentedLagrangianMinimizer) ---
  nb::enum_<cunls::ConstraintKind>(m, "ConstraintKind", "Kind of a constraint row.")
      .value("Equality", cunls::ConstraintKind::kEquality, "c(x) = 0")
      .value("Inequality", cunls::ConstraintKind::kInequality, "c(x) <= 0");

  nb::class_<cunls::ConstraintFactorBatchBase, cunls::FactorBatch>(
      m, "ConstraintFactorBatchBase",
      "A factor batch whose rows are constraints. Evaluates the augmented Lagrangian "
      "residuals of its multipliers (one per row) and penalties (one per factor); "
      "AugmentedLagrangianMinimizer updates both.")
      .def_prop_ro("kind", &cunls::ConstraintFactorBatchBase::Kind)
      .def_prop_ro("scale", &cunls::ConstraintFactorBatchBase::Scale)
      .def_prop_ro(
          "multipliers_ptr",
          [](cunls::ConstraintFactorBatchBase &self) {
            return reinterpret_cast<uintptr_t>(self.Multipliers());
          },
          "Device pointer to capacity * residuals_size float multipliers (row r of factor f "
          "at f * residuals_size + r).")
      .def_prop_ro(
          "penalties_ptr",
          [](cunls::ConstraintFactorBatchBase &self) {
            return reinterpret_cast<uintptr_t>(self.Penalties());
          },
          "Device pointer to capacity float penalties, one per factor.")
      .def(
          "reset_multipliers",
          [](cunls::ConstraintFactorBatchBase &self, cunls::CudaStream &stream) {
            self.ResetMultipliers(stream.GetStream());
          },
          nb::arg("stream"), "Sets every multiplier to 0.")
      .def(
          "set_penalty",
          [](cunls::ConstraintFactorBatchBase &self, float penalty, cunls::CudaStream &stream) {
            self.SetPenalty(penalty, stream.GetStream());
          },
          nb::arg("penalty"), nb::arg("stream"), "Sets the penalty of every factor.");

  nb::class_<cunls::ConstraintFactorBatch, cunls::ConstraintFactorBatchBase>(
      m, "ConstraintFactorBatch",
      "Turns any factor batch into constraint rows: every residual row r of the wrapped "
      "batch becomes scale * r(x) = 0 (Equality) or scale * r(x) <= 0 (Inequality).\n\n"
      "Parameters\n"
      "----------\n"
      "inner_factor : FactorBatch\n"
      "    The factor batch whose residuals are the constraint rows.\n"
      "kind : ConstraintKind\n"
      "    Equality or Inequality.\n"
      "scale : float\n"
      "    Positive row scale, so that the constraint tolerance means the same in meters,\n"
      "    radians or newtons.")
      .def(nb::init<cunls::FactorBatch *, cunls::ConstraintKind, float>(), nb::arg("inner_factor"),
           nb::arg("kind"), nb::arg("scale") = 1.f, nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::ConstraintFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::ConstraintFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::ConstraintFactorBatch::StateSizes);

  bind_bound_factor<1>(m, "BoundFactorBatch1");
  bind_bound_factor<2>(m, "BoundFactorBatch2");
  bind_bound_factor<3>(m, "BoundFactorBatch3");
  bind_bound_factor<4>(m, "BoundFactorBatch4");
  bind_bound_factor<6>(m, "BoundFactorBatch6");
  bind_bound_factor<12>(m, "BoundFactorBatch12");

  bind_halfspace_factor<1>(m, "HalfspaceFactorBatch1");
  bind_halfspace_factor<2>(m, "HalfspaceFactorBatch2");
  bind_halfspace_factor<3>(m, "HalfspaceFactorBatch3");
  bind_halfspace_factor<6>(m, "HalfspaceFactorBatch6");

  // --- Dynamics (between consecutive trajectory steps) ---
  nb::class_<cunls::SE2DifferentialDriveFactorBatch, cunls::FactorBatch>(
      m, "SE2DifferentialDriveFactorBatch",
      "Two driven wheels plus casters (NVIDIA Carter), kinematic. States: pose T_k "
      "(SE2StateBatch), wheel speeds (omega_L, omega_R) [rad/s] (VectorStateBatch2), pose "
      "T_{k+1}. Residual Log((T_k Exp(dt_k xi(u_k)))^-1 T_{k+1}) with the body twist "
      "xi = [r (omega_R + omega_L) / 2, 0, r (omega_R - omega_L) / b]; exact for wheel speeds "
      "held over the step. Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations dt_k [s].\n"
      "wheel_radius, track_width : float\n"
      "    r and b [m].\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffer holds.")
      .def(
          "__init__",
          [](cunls::SE2DifferentialDriveFactorBatch *self, nb::handle time_steps,
             float wheel_radius, float track_width, size_t capacity) {
            new (self) cunls::SE2DifferentialDriveFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)), wheel_radius,
                track_width, capacity);
          },
          nb::arg("time_steps"), nb::arg("wheel_radius"), nb::arg("track_width"),
          nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE2DifferentialDriveFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE2DifferentialDriveFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE2DifferentialDriveFactorBatch::StateSizes);

  nb::class_<cunls::SE2KinematicBicycleFactorBatch, cunls::FactorBatch>(
      m, "SE2KinematicBicycleFactorBatch",
      "Car with Ackermann steering (kinematic bicycle, rear axle). States: pose T_k "
      "(SE2StateBatch), z_k = (v, delta) (VectorStateBatch2), control u_k = (a, delta rate) "
      "(VectorStateBatch2), pose T_{k+1}, z_{k+1}. Euler step: residual "
      "[Log((T_k Exp(dt_k xi(z_k)))^-1 T_{k+1}); z_{k+1} - z_k - dt_k u_k] with "
      "xi = [v, 0, v tan(delta) / L]. Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations dt_k [s].\n"
      "wheelbase : float\n"
      "    L [m].\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffer holds.")
      .def(
          "__init__",
          [](cunls::SE2KinematicBicycleFactorBatch *self, nb::handle time_steps, float wheelbase,
             size_t capacity) {
            new (self) cunls::SE2KinematicBicycleFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)), wheelbase,
                capacity);
          },
          nb::arg("time_steps"), nb::arg("wheelbase"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE2KinematicBicycleFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE2KinematicBicycleFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE2KinematicBicycleFactorBatch::StateSizes);

  nb::class_<cunls::SE3DifferentialDriveFactorBatch, cunls::FactorBatch>(
      m, "SE3DifferentialDriveFactorBatch",
      "Carter driving on non-planar terrain. States: pose T_k (SE3StateBatch; x forward, z up), "
      "wheel speeds (VectorStateBatch2), pose T_{k+1}. Residual: rows 2..5 (yaw, forward, "
      "lateral, vertical) of Log((T_k Exp(dt_k xi))^-1 T_{k+1}) with xi = [0, 0, omega, v, 0, "
      "0]; roll and pitch are left to terrain factors. Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations [s].\n"
      "wheel_radius, track_width : float\n"
      "    r and b [m].\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffer holds.")
      .def(
          "__init__",
          [](cunls::SE3DifferentialDriveFactorBatch *self, nb::handle time_steps,
             float wheel_radius, float track_width, size_t capacity) {
            new (self) cunls::SE3DifferentialDriveFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)), wheel_radius,
                track_width, capacity);
          },
          nb::arg("time_steps"), nb::arg("wheel_radius"), nb::arg("track_width"),
          nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE3DifferentialDriveFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE3DifferentialDriveFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE3DifferentialDriveFactorBatch::StateSizes);

  nb::class_<cunls::SE3KinematicBicycleFactorBatch, cunls::FactorBatch>(
      m, "SE3KinematicBicycleFactorBatch",
      "Car (kinematic bicycle) on non-planar roads. States: pose T_k (SE3StateBatch), (v, delta), "
      "(a, delta rate), pose T_{k+1}, (v, delta). Residual: rows 2..5 of "
      "Log((T_k Exp(dt_k xi))^-1 T_{k+1}) with xi = [0, 0, v tan(delta) / L, v, 0, 0], and "
      "z_{k+1} - z_k - dt_k u_k; roll and pitch are left to terrain factors.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations [s].\n"
      "wheelbase : float\n"
      "    L [m].\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffer holds.")
      .def(
          "__init__",
          [](cunls::SE3KinematicBicycleFactorBatch *self, nb::handle time_steps, float wheelbase,
             size_t capacity) {
            new (self) cunls::SE3KinematicBicycleFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)), wheelbase,
                capacity);
          },
          nb::arg("time_steps"), nb::arg("wheelbase"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE3KinematicBicycleFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE3KinematicBicycleFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE3KinematicBicycleFactorBatch::StateSizes);

  bind_kinematics_factor<cunls::SE2KinematicsFactorBatch>(
      m, "SE2KinematicsFactorBatch",
      "X_{k+1} = X_k Exp(dt_k xi_k) on SE(2) with the body twist [v_x, v_y, omega] as control. "
      "States: X_k (SE2StateBatch), xi_k (VectorStateBatch3), X_{k+1}. Residual "
      "Log((X_k Exp(dt xi))^-1 X_{k+1}); exact for a constant twist. time_steps: device buffer "
      "of capacity step durations.");
  bind_kinematics_factor<cunls::SO3KinematicsFactorBatch>(
      m, "SO3KinematicsFactorBatch",
      "R_{k+1} = R_k Exp(dt_k omega_k) on SO(3) with the body rate as control. States: R_k "
      "(SO3StateBatch), omega_k (VectorStateBatch3), R_{k+1}. time_steps: device buffer of "
      "capacity step durations.");
  bind_kinematics_factor<cunls::SE3KinematicsFactorBatch>(
      m, "SE3KinematicsFactorBatch",
      "T_{k+1} = T_k Exp(dt_k xi_k) on SE(3) with the body twist [omega, v] as control. States: "
      "T_k (SE3StateBatch), xi_k (VectorStateBatch6), T_{k+1}. time_steps: device buffer of "
      "capacity step durations.");

  // --- Quadrotor and quadruped (single rigid body) ---
  nb::class_<cunls::QuadrotorParameters>(
      m, "QuadrotorParameters",
      "Quadrotor in X configuration (body x forward, y left, z up). Rotor i at a (s_x, s_y), "
      "a = arm_length / sqrt(2), (s_x, s_y) = (+1, -1), (-1, +1), (+1, +1), (-1, -1) for i = 0..3; "
      "rotors 0, 1 spin counter-clockwise (yaw torque -k_m f_i), rotors 2, 3 clockwise "
      "(+k_m f_i).")
      .def(nb::init<>())
      .def_rw("mass", &cunls::QuadrotorParameters::mass)
      .def_prop_rw(
          "inertia",
          [](const cunls::QuadrotorParameters &p) {
            return std::vector<float>{p.inertia[0], p.inertia[1], p.inertia[2]};
          },
          [](cunls::QuadrotorParameters &p, const std::vector<float> &j) {
            if (j.size() != 3)
              throw std::invalid_argument("inertia needs 3 values (J_x, J_y, J_z)");
            for (int i = 0; i < 3; ++i) p.inertia[i] = j[i];
          },
          "Diagonal body inertia (J_x, J_y, J_z) [kg m^2].")
      .def_rw("arm_length", &cunls::QuadrotorParameters::arm_length)
      .def_rw("torque_coefficient", &cunls::QuadrotorParameters::torque_coefficient)
      .def_rw("linear_drag", &cunls::QuadrotorParameters::linear_drag)
      .def_rw("gravity", &cunls::QuadrotorParameters::gravity);

  nb::class_<cunls::QuadrotorFactorBatch, cunls::FactorBatch>(
      m, "QuadrotorFactorBatch",
      "Quadrotor rigid-body dynamics with rotor thrusts as controls (Euler). States: pose T_k "
      "(SE3StateBatch), world velocity v_k (VectorStateBatch3), body rates omega_k "
      "(VectorStateBatch3), rotor thrusts f_k (VectorStateBatch4), T_{k+1}, v_{k+1}, "
      "omega_{k+1}. Residual (12): pose defect, velocity and rate rows. Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations [s].\n"
      "parameters : QuadrotorParameters\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffer holds.")
      .def(
          "__init__",
          [](cunls::QuadrotorFactorBatch *self, nb::handle time_steps,
             const cunls::QuadrotorParameters &parameters, size_t capacity) {
            new (self) cunls::QuadrotorFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)), parameters,
                capacity);
          },
          nb::arg("time_steps"), nb::arg("parameters"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::QuadrotorFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::QuadrotorFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::QuadrotorFactorBatch::StateSizes);

  nb::class_<cunls::QuadrupedParameters>(m, "QuadrupedParameters",
                                         "Single-rigid-body quadruped: mass, inertia, gravity.")
      .def(nb::init<>())
      .def_rw("mass", &cunls::QuadrupedParameters::mass)
      .def_prop_rw(
          "inertia",
          [](const cunls::QuadrupedParameters &p) {
            return std::vector<float>{p.inertia[0], p.inertia[1], p.inertia[2]};
          },
          [](cunls::QuadrupedParameters &p, const std::vector<float> &j) {
            if (j.size() != 3)
              throw std::invalid_argument("inertia needs 3 values (J_x, J_y, J_z)");
            for (int i = 0; i < 3; ++i) p.inertia[i] = j[i];
          },
          "Diagonal body inertia (J_x, J_y, J_z) [kg m^2].")
      .def_rw("gravity", &cunls::QuadrupedParameters::gravity);

  nb::class_<cunls::QuadrupedFactorBatch, cunls::FactorBatch>(
      m, "QuadrupedFactorBatch",
      "Quadruped base as a single rigid body driven by four foot forces (Euler). States: base "
      "pose T_k (SE3StateBatch), world velocity v_k, body rates omega_k, foot forces F_k "
      "(VectorStateBatch12, world frame), T_{k+1}, v_{k+1}, omega_{k+1}. Per-step inputs: contact "
      "flags (4 per factor) and world foot positions (12 per factor). Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "time_steps : DevicePointer\n"
      "    Device buffer of ``capacity`` step durations [s].\n"
      "contacts : DevicePointer\n"
      "    4 floats per factor: 1 for a foot in stance, 0 in swing.\n"
      "foot_positions : DevicePointer\n"
      "    12 floats per factor: world positions of feet 0..3.\n"
      "parameters : QuadrupedParameters\n"
      "capacity : int\n"
      "    Number of factors (steps) the buffers hold.")
      .def(
          "__init__",
          [](cunls::QuadrupedFactorBatch *self, nb::handle time_steps, nb::handle contacts,
             nb::handle foot_positions, const cunls::QuadrupedParameters &parameters,
             size_t capacity) {
            new (self) cunls::QuadrupedFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(time_steps)),
                reinterpret_cast<const float *>(extract_device_ptr(contacts)),
                reinterpret_cast<const float *>(extract_device_ptr(foot_positions)), parameters,
                capacity);
          },
          nb::arg("time_steps"), nb::arg("contacts"), nb::arg("foot_positions"),
          nb::arg("parameters"), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def_prop_ro("num_active_factors", &cunls::QuadrupedFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::QuadrupedFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::QuadrupedFactorBatch::StateSizes);

  // --- IMU (raw samples between keyframes marginalized inside the factor) ---
  nb::class_<cunls::ImuParameters>(
      m, "ImuParameters",
      "Sensor model of ImuFactorBatch. Noise densities are continuous-time (defaults: the "
      "ADIS16448 of the EuRoC MAV dataset); gravity defaults to (0, 0, -9.80665), +Z up; "
      "body_from_imu is the row-major 4x4 pose of the IMU in the body (rig) frame, rig_from_imu "
      "(identity).")
      .def(nb::init<>())
      .def_prop_rw(
          "gravity",
          [](const cunls::ImuParameters &p) {
            return std::vector<float>(p.gravity, p.gravity + 3);
          },
          [](cunls::ImuParameters &p, const std::vector<float> &g) {
            if (g.size() != 3) throw std::invalid_argument("gravity needs 3 values");
            for (int i = 0; i < 3; ++i) p.gravity[i] = g[i];
          },
          "World-frame gravity [m/s^2].")
      .def_rw("gyro_noise_density", &cunls::ImuParameters::gyro_noise_density)
      .def_rw("accel_noise_density", &cunls::ImuParameters::accel_noise_density)
      .def_rw("integration_noise_density", &cunls::ImuParameters::integration_noise_density)
      .def_rw("gyro_bias_random_walk", &cunls::ImuParameters::gyro_bias_random_walk)
      .def_rw("accel_bias_random_walk", &cunls::ImuParameters::accel_bias_random_walk)
      .def_prop_rw(
          "body_from_imu",
          [](const cunls::ImuParameters &p) {
            return std::vector<float>(p.body_from_imu, p.body_from_imu + 16);
          },
          [](cunls::ImuParameters &p, const std::vector<float> &T) {
            if (T.size() != 16)
              throw std::invalid_argument("body_from_imu needs 16 values (row-major 4x4)");
            for (int i = 0; i < 16; ++i) p.body_from_imu[i] = T[i];
          },
          "Pose of the IMU in the body frame, row-major 4x4 (16 floats).");

  nb::class_<cunls::ImuFactorBatch, cunls::FactorBatch>(
      m, "ImuFactorBatch",
      "IMU factor between two keyframes; the raw samples between them define a chain of Euler "
      "steps whose intermediate states are marginalized inside the factor (Schur complement, "
      "recomputed every evaluation). States: X_a (SE3StateBatch, rig_from_world as in "
      "ReprojectionFactorBatch / PnPFactorBatch), v_a (VectorStateBatch3, world velocity of the "
      "IMU), b_a = [b_g; b_a] (VectorStateBatch6), X_b, v_b, b_b. Residual (15): "
      "the whitened defect of keyframe b against the prediction (9) and the bias random walk "
      "(6). Analytic Jacobians.\n\n"
      "Parameters\n"
      "----------\n"
      "imu_samples : DevicePointer\n"
      "    7 floats per sample (gyro xyz [rad/s], specific force xyz [m/s^2], dt [s]), IMU frame, "
      "all factors back to back.\n"
      "sample_offsets : DevicePointer\n"
      "    int32, capacity + 1 CSR offsets: factor f uses samples [offsets[f], offsets[f + 1]) "
      "(at least one).\n"
      "num_samples : int\n"
      "    Number of samples imu_samples holds; only sizes the work split (num_samples / "
      "capacity is taken as the typical chain length).\n"
      "parameters : ImuParameters\n"
      "capacity : int\n"
      "    Number of factors (keyframe pairs) the buffers hold.")
      .def(
          "__init__",
          [](cunls::ImuFactorBatch *self, nb::handle imu_samples, nb::handle sample_offsets,
             size_t num_samples, const cunls::ImuParameters &parameters, size_t capacity) {
            new (self) cunls::ImuFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(imu_samples)),
                reinterpret_cast<const int *>(extract_device_ptr(sample_offsets)), num_samples,
                parameters, capacity);
          },
          nb::arg("imu_samples"), nb::arg("sample_offsets"), nb::arg("num_samples"),
          nb::arg("parameters"), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>())
      .def_prop_ro("num_active_factors", &cunls::ImuFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::ImuFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::ImuFactorBatch::StateSizes);

  // --- Obstacle clearance (constraint functions; wrap as inequalities) ---
  nb::class_<cunls::SE2DiskClearanceFactorBatch, cunls::FactorBatch>(
      m, "SE2DiskClearanceFactorBatch",
      "Clearance c = (radius + margin) - |p - center| between an SE(2) pose's origin and a disk "
      "(feasible: c <= 0); wrap in ConstraintFactorBatch(..., ConstraintKind.Inequality). "
      "obstacles: device buffer of 3 floats per factor (center x, y, radius).")
      .def(
          "__init__",
          [](cunls::SE2DiskClearanceFactorBatch *self, nb::handle obstacles, float margin,
             size_t capacity) {
            new (self) cunls::SE2DiskClearanceFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(obstacles)), margin, capacity);
          },
          nb::arg("obstacles"), nb::arg("margin"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE2DiskClearanceFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE2DiskClearanceFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE2DiskClearanceFactorBatch::StateSizes);

  nb::class_<cunls::SE3SphereClearanceFactorBatch, cunls::FactorBatch>(
      m, "SE3SphereClearanceFactorBatch",
      "Clearance c = (radius + margin) - |p - center| between an SE(3) pose's origin and a "
      "sphere (feasible: c <= 0); wrap in ConstraintFactorBatch(..., ConstraintKind.Inequality). "
      "obstacles: device buffer of 4 floats per factor (center x, y, z, radius).")
      .def(
          "__init__",
          [](cunls::SE3SphereClearanceFactorBatch *self, nb::handle obstacles, float margin,
             size_t capacity) {
            new (self) cunls::SE3SphereClearanceFactorBatch(
                reinterpret_cast<const float *>(extract_device_ptr(obstacles)), margin, capacity);
          },
          nb::arg("obstacles"), nb::arg("margin"), nb::arg("capacity"), nb::keep_alive<1, 2>())
      .def_prop_ro("num_active_factors", &cunls::SE3SphereClearanceFactorBatch::NumActiveFactors)
      .def_prop_ro("residuals_size", &cunls::SE3SphereClearanceFactorBatch::ResidualsSize)
      .def("state_sizes", &cunls::SE3SphereClearanceFactorBatch::StateSizes);
}
