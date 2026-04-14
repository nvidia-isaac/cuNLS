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

// Bindings for all FactorBatch subclasses and the CustomFactorBatch trampoline.
//
// A FactorBatch represents a batch of N residual functions that share the same
// structure (residual dimension, number and sizes of state blocks).  cuNLS
// ships many built-in factor types (reprojection, SE3 between, ICP variants,
// priors); each is bound below as a nanobind class that inherits FactorBatch.
//
// User-defined factors are supported via the "trampoline" pattern:
// PyFactorBatch is a C++ class that inherits cunls::FactorBatch and overrides
// Evaluate() to call back into a Python method.  On the Python side it is
// exposed as CustomFactorBatch; users subclass it and implement evaluate().

#include "bindings.h"

#include "py_factor_wrappers.h"

#include <nanobind/stl/vector.h>

#include "cunls/factor/sized_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/factor/similarity2_between_factor_batch.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/factor/sl4_between_factor_batch.h"
#include "cunls/factor/sl4_prior_factor_batch.h"
#include "cunls/factor/so2_between_factor_batch.h"
#include "cunls/factor/so3_between_factor_batch.h"
#include "cunls/factor/vector_between_factor_batch.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/factor/so2_prior_factor_batch.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "cunls/common/types.h"

namespace {

// Template helper to bind PriorVectorFactorBatch<Dim> for a given dimension.
template <int Dim>
void bind_prior_vector_factor(nb::module_& m, const char* name) {
    using Class = cunls::PriorVectorFactorBatch<Dim>;
    nb::class_<Class, cunls::FactorBatch>(m, name,
        "Prior factor for Dim-dimensional vectors: residual = state - observation.")
        .def("__init__", [](Class* self, nb::handle observations, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Vector<Dim>*>(
                extract_device_ptr(observations));
            new (self) Class(ptr, num_factors);
        }, nb::arg("observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &Class::NumFactors)
        .def_prop_ro("residuals_size", &Class::ResidualsSize)
        .def("state_block_sizes", &Class::StateBlockSizes);
}

template <int Dim>
void bind_vector_between_factor(nb::module_& m, const char* name) {
    using Class = cunls::VectorBetweenFactorBatch<Dim>;
    nb::class_<Class, cunls::FactorBatch>(m, name,
        "Between factor on Euclidean vectors: residual = left - right - delta.")
        .def("__init__", [](Class* self, nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Vector<Dim>*>(
                extract_device_ptr(deltas));
            new (self) Class(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &Class::NumFactors)
        .def_prop_ro("residuals_size", &Class::ResidualsSize)
        .def("state_block_sizes", &Class::StateBlockSizes);
}

}  // namespace

void bind_factor(nb::module_& m) {
    nb::class_<cunls::FactorBatch>(m, "FactorBatch",
        "Abstract base class for batched factors.");

    // --- Custom factor trampoline ---
    nb::class_<PyFactorBatch, cunls::FactorBatch>(m, "CustomFactorBatch",
        "Base class for user-defined factors. Override evaluate() in Python.")
        .def(nb::init<size_t, std::vector<size_t>, size_t>(),
             nb::arg("residual_size"), nb::arg("state_block_sizes"),
             nb::arg("num_factors"))
        .def("evaluate", [](PyFactorBatch&, uintptr_t, uintptr_t,
                            uintptr_t, uintptr_t) -> bool {
            throw std::runtime_error(
                "CustomFactorBatch.evaluate() must be overridden in a subclass.");
        }, nb::arg("residuals_ptr"), nb::arg("jacobians_ptr"),
           nb::arg("state_pointers_ptr"), nb::arg("stream_handle"))
        .def_prop_ro("num_factors", &PyFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &PyFactorBatch::ResidualsSize)
        .def("state_block_sizes", &PyFactorBatch::StateBlockSizes);

    // --- Reprojection ---
    nb::class_<cunls::ReprojectionFactorBatch, cunls::FactorBatch>(m,
        "ReprojectionFactorBatch",
        "Batched 2D reprojection factor. Residual=2, States=[SE3(6), Point(3)].\n"
        "Observations must be in normalized image coordinates (K^-1 applied).")
        .def("__init__", [](cunls::ReprojectionFactorBatch* self,
                            nb::handle observations, size_t num_obs,
                            float z_threshold) {
            auto ptr = reinterpret_cast<const cunls::Vector<2>*>(
                extract_device_ptr(observations));
            new (self) cunls::ReprojectionFactorBatch(ptr, num_obs, z_threshold);
        }, nb::arg("observations"), nb::arg("num_observations"),
           nb::arg("z_threshold") = 1e-3f, nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::ReprojectionFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::ReprojectionFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::ReprojectionFactorBatch::StateBlockSizes);

    nb::class_<cunls::PnPFactorBatch, cunls::FactorBatch>(m, "PnPFactorBatch",
        "Batched PnP reprojection: fixed 3D points, pose-only Jacobian.\n"
        "Residual=2, States=[SE3(6)]. Observations in normalized image coords.\n"
        "3D points are passed at construction (device); not optimized.")
        .def("__init__", [](cunls::PnPFactorBatch* self, nb::handle observations,
                            nb::handle points_world, size_t num_obs,
                            float z_threshold) {
            auto obs_ptr = reinterpret_cast<const cunls::Vector<2>*>(
                extract_device_ptr(observations));
            auto p_ptr = reinterpret_cast<const cunls::Vector<3>*>(
                extract_device_ptr(points_world));
            new (self) cunls::PnPFactorBatch(obs_ptr, p_ptr, num_obs,
                                            z_threshold);
        }, nb::arg("observations"), nb::arg("points_world"),
           nb::arg("num_observations"), nb::arg("z_threshold") = 1e-3f,
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("__init__", [](cunls::PnPFactorBatch* self, nb::handle observations,
                            nb::handle poses_camera_from_rig,
                            nb::handle points_world, size_t num_obs,
                            float z_threshold) {
            auto obs_ptr = reinterpret_cast<const cunls::Vector<2>*>(
                extract_device_ptr(observations));
            auto rig_ptr = reinterpret_cast<const cunls::SE3Transform*>(
                extract_device_ptr(poses_camera_from_rig));
            auto p_ptr = reinterpret_cast<const cunls::Vector<3>*>(
                extract_device_ptr(points_world));
            new (self) cunls::PnPFactorBatch(obs_ptr, rig_ptr, p_ptr, num_obs,
                                            z_threshold);
        }, nb::arg("observations"), nb::arg("poses_camera_from_rig"),
           nb::arg("points_world"), nb::arg("num_observations"),
           nb::arg("z_threshold") = 1e-3f,
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>())
        .def_prop_ro("num_factors", &cunls::PnPFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::PnPFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::PnPFactorBatch::StateBlockSizes);

    // --- SE3 Between ---
    nb::class_<cunls::SE3BetweenFactorBatch, cunls::FactorBatch>(m,
        "SE3BetweenFactorBatch",
        "Batched SE(3) between factor. Residual=6, States=[SE3(6), SE3(6)].")
        .def("__init__", [](cunls::SE3BetweenFactorBatch* self,
                            cunls::cuBLASHandle& cublas,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::SE3Transform*>(
                extract_device_ptr(deltas));
            new (self) cunls::SE3BetweenFactorBatch(cublas, ptr, num_factors);
        }, nb::arg("cublas_handle"), nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def_prop_ro("num_factors", &cunls::SE3BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SE3BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SE3BetweenFactorBatch::StateBlockSizes);

    // --- SE2 Between ---
    nb::class_<cunls::SE2BetweenFactorBatch, cunls::FactorBatch>(m,
        "SE2BetweenFactorBatch",
        "Batched SE(2) between factor. Residual=3, States=[SE2(3), SE2(3)].")
        .def("__init__", [](cunls::SE2BetweenFactorBatch* self,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<3>*>(
                extract_device_ptr(deltas));
            new (self) cunls::SE2BetweenFactorBatch(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SE2BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SE2BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SE2BetweenFactorBatch::StateBlockSizes);

    // --- SO2 Between ---
    nb::class_<cunls::SO2BetweenFactorBatch, cunls::FactorBatch>(m,
        "SO2BetweenFactorBatch",
        "Batched SO(2) between factor. Residual=1, States=[SO2(1), SO2(1)].")
        .def("__init__", [](cunls::SO2BetweenFactorBatch* self,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<2>*>(
                extract_device_ptr(deltas));
            new (self) cunls::SO2BetweenFactorBatch(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SO2BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SO2BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SO2BetweenFactorBatch::StateBlockSizes);

    // --- SO3 Between ---
    nb::class_<cunls::SO3BetweenFactorBatch, cunls::FactorBatch>(m,
        "SO3BetweenFactorBatch",
        "Batched SO(3) between factor. Residual=3, States=[SO3(3), SO3(3)].")
        .def("__init__", [](cunls::SO3BetweenFactorBatch* self,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<3>*>(
                extract_device_ptr(deltas));
            new (self) cunls::SO3BetweenFactorBatch(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SO3BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SO3BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SO3BetweenFactorBatch::StateBlockSizes);

    // --- Similarity2 Between ---
    nb::class_<cunls::Similarity2BetweenFactorBatch, cunls::FactorBatch>(m,
        "Similarity2BetweenFactorBatch",
        "Batched Sim(2) between factor. Residual=4, States=[Sim2(4), Sim2(4)].")
        .def("__init__", [](cunls::Similarity2BetweenFactorBatch* self,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<3>*>(
                extract_device_ptr(deltas));
            new (self) cunls::Similarity2BetweenFactorBatch(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::Similarity2BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::Similarity2BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::Similarity2BetweenFactorBatch::StateBlockSizes);

    // --- Similarity3 Between ---
    nb::class_<cunls::Similarity3BetweenFactorBatch, cunls::FactorBatch>(m,
        "Similarity3BetweenFactorBatch",
        "Batched Sim(3) between factor. Residual=7, States=[Sim3(7), Sim3(7)].")
        .def("__init__", [](cunls::Similarity3BetweenFactorBatch* self,
                            cunls::cuBLASHandle& cublas,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<4>*>(
                extract_device_ptr(deltas));
            new (self) cunls::Similarity3BetweenFactorBatch(cublas, ptr, num_factors);
        }, nb::arg("cublas_handle"), nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def_prop_ro("num_factors", &cunls::Similarity3BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::Similarity3BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::Similarity3BetweenFactorBatch::StateBlockSizes);

    // --- SL4 Between ---
    nb::class_<cunls::SL4BetweenFactorBatch, cunls::FactorBatch>(m,
        "SL4BetweenFactorBatch",
        "Batched SL(4) between factor. Residual=15, States=[SL4(15), SL4(15)].")
        .def("__init__", [](cunls::SL4BetweenFactorBatch* self,
                            nb::handle deltas, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::SL4Transform*>(
                extract_device_ptr(deltas));
            new (self) cunls::SL4BetweenFactorBatch(ptr, num_factors);
        }, nb::arg("deltas"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SL4BetweenFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SL4BetweenFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SL4BetweenFactorBatch::StateBlockSizes);

    // --- SE3 Prior ---
    nb::class_<cunls::SE3PriorFactorBatch, cunls::FactorBatch>(m,
        "SE3PriorFactorBatch",
        "Batched SE(3) prior factor. Residual=6, States=[SE3(6)].")
        .def("__init__", [](cunls::SE3PriorFactorBatch* self,
                            nb::handle observations, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::SE3Transform*>(
                extract_device_ptr(observations));
            new (self) cunls::SE3PriorFactorBatch(ptr, num_factors);
        }, nb::arg("observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SE3PriorFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SE3PriorFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SE3PriorFactorBatch::StateBlockSizes);

    // --- SL4 Prior ---
    nb::class_<cunls::SL4PriorFactorBatch, cunls::FactorBatch>(m,
        "SL4PriorFactorBatch",
        "Batched SL(4) prior factor. Residual=15, States=[SL4(15)].")
        .def("__init__", [](cunls::SL4PriorFactorBatch* self,
                            nb::handle observations, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::SL4Transform*>(
                extract_device_ptr(observations));
            new (self) cunls::SL4PriorFactorBatch(ptr, num_factors);
        }, nb::arg("observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SL4PriorFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SL4PriorFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SL4PriorFactorBatch::StateBlockSizes);

    // --- SO3 Prior ---
    nb::class_<cunls::SO3PriorFactorBatch, cunls::FactorBatch>(m,
        "SO3PriorFactorBatch",
        "Batched SO(3) prior factor. Residual=3, States=[SO3(3)].")
        .def("__init__", [](cunls::SO3PriorFactorBatch* self,
                            nb::handle observations, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<3>*>(
                extract_device_ptr(observations));
            new (self) cunls::SO3PriorFactorBatch(ptr, num_factors);
        }, nb::arg("observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SO3PriorFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SO3PriorFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SO3PriorFactorBatch::StateBlockSizes);

    // --- SO2 Prior ---
    nb::class_<cunls::SO2PriorFactorBatch, cunls::FactorBatch>(m,
        "SO2PriorFactorBatch",
        "Batched SO(2) prior factor. Residual=1, States=[SO2(1)].")
        .def("__init__", [](cunls::SO2PriorFactorBatch* self,
                            nb::handle observations, size_t num_factors) {
            auto ptr = reinterpret_cast<const cunls::Matrix<2>*>(
                extract_device_ptr(observations));
            new (self) cunls::SO2PriorFactorBatch(ptr, num_factors);
        }, nb::arg("observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>())
        .def_prop_ro("num_factors", &cunls::SO2PriorFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SO2PriorFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SO2PriorFactorBatch::StateBlockSizes);

    // --- Prior Vector Factors ---
    bind_prior_vector_factor<1>(m, "PriorVectorFactorBatch1");
    bind_prior_vector_factor<2>(m, "PriorVectorFactorBatch2");
    bind_prior_vector_factor<3>(m, "PriorVectorFactorBatch3");
    bind_prior_vector_factor<6>(m, "PriorVectorFactorBatch6");

    bind_vector_between_factor<1>(m, "VectorBetweenFactorBatch1");
    bind_vector_between_factor<2>(m, "VectorBetweenFactorBatch2");
    bind_vector_between_factor<3>(m, "VectorBetweenFactorBatch3");
    bind_vector_between_factor<6>(m, "VectorBetweenFactorBatch6");

    // --- Point-to-Point ---
    nb::class_<cunls::PointToPointFactorBatch, cunls::FactorBatch>(m,
        "PointToPointFactorBatch",
        "Batched point-to-point factor: residual = p - T*q. Residual=3, States=[SE3(6)].")
        .def("__init__", [](cunls::PointToPointFactorBatch* self,
                            nb::handle p_obs, nb::handle q_obs,
                            size_t num_factors) {
            auto p = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(q_obs));
            new (self) cunls::PointToPointFactorBatch(p, q, num_factors);
        }, nb::arg("p_observations"), nb::arg("q_observations"),
           nb::arg("num_factors"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def_prop_ro("num_factors", &cunls::PointToPointFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::PointToPointFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::PointToPointFactorBatch::StateBlockSizes);

    // --- Point-to-Plane ---
    nb::class_<cunls::PointToPlaneFactorBatch, cunls::FactorBatch>(m,
        "PointToPlaneFactorBatch",
        "Batched point-to-plane factor: residual = Nq^T*(p - T*q). Residual=1, States=[SE3(6)].")
        .def("__init__", [](cunls::PointToPlaneFactorBatch* self,
                            nb::handle p_obs, nb::handle q_obs,
                            nb::handle nq_obs, size_t num_factors) {
            auto p = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(q_obs));
            auto nq = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(nq_obs));
            new (self) cunls::PointToPlaneFactorBatch(p, q, nq, num_factors);
        }, nb::arg("p_observations"), nb::arg("q_observations"),
           nb::arg("nq_observations"), nb::arg("num_factors"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
        .def_prop_ro("num_factors", &cunls::PointToPlaneFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::PointToPlaneFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::PointToPlaneFactorBatch::StateBlockSizes);

    // --- Symmetric Point-to-Plane ---
    nb::class_<cunls::SymmetricPointToPlaneFactorBatch, cunls::FactorBatch>(m,
        "SymmetricPointToPlaneFactorBatch",
        "Batched symmetric point-to-plane factor. Residual=1, States=[SE3(6)].")
        .def("__init__", [](cunls::SymmetricPointToPlaneFactorBatch* self,
                            nb::handle p_obs, nb::handle q_obs,
                            nb::handle np_obs, nb::handle nq_obs,
                            size_t num_factors) {
            auto p = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(p_obs));
            auto q = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(q_obs));
            auto np = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(np_obs));
            auto nq = reinterpret_cast<const cunls::Vector<3>*>(extract_device_ptr(nq_obs));
            new (self) cunls::SymmetricPointToPlaneFactorBatch(p, q, np, nq, num_factors);
        }, nb::arg("p_observations"), nb::arg("q_observations"),
           nb::arg("np_observations"), nb::arg("nq_observations"),
           nb::arg("num_factors"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>())
        .def_prop_ro("num_factors", &cunls::SymmetricPointToPlaneFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &cunls::SymmetricPointToPlaneFactorBatch::ResidualsSize)
        .def("state_block_sizes", &cunls::SymmetricPointToPlaneFactorBatch::StateBlockSizes);

    // --- InformationFactorBatch (polymorphic wrapper) ---
    nb::class_<PyInformationFactorBatch, cunls::FactorBatch>(m,
        "InformationFactorBatch",
        "Wraps any factor batch and applies per-factor sqrt-information matrices.\n\n"
        "Residuals and Jacobians are left-multiplied by the corresponding\n"
        "square-root information matrix: r' = Omega^{1/2} r, J' = Omega^{1/2} J.\n\n"
        "Parameters\n"
        "----------\n"
        "cublas_handle : CublasHandle\n"
        "    Shared cuBLAS handle.\n"
        "inner_factor : FactorBatch\n"
        "    The factor batch to wrap.\n"
        "sqrt_information_matrices : DevicePointer\n"
        "    Device buffer with one square-root information matrix per inner factor\n"
        "    (``inner_factor.num_factors`` matrices), each residual_size x residual_size,\n"
        "    stored contiguously in row-major order.")
        .def("__init__", [](PyInformationFactorBatch* self,
                            cunls::cuBLASHandle& cublas,
                            cunls::FactorBatch* inner,
                            nb::handle sqrt_info) {
            auto ptr = reinterpret_cast<const float*>(
                extract_device_ptr(sqrt_info));
            new (self) PyInformationFactorBatch(cublas, inner, ptr);
        }, nb::arg("cublas_handle"), nb::arg("inner_factor"),
           nb::arg("sqrt_information_matrices"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>())
        .def_prop_ro("num_factors", &PyInformationFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &PyInformationFactorBatch::ResidualsSize)
        .def("state_block_sizes", &PyInformationFactorBatch::StateBlockSizes);

    // --- WeightedFactorBatch (polymorphic wrapper) ---
    nb::class_<PyWeightedFactorBatch, cunls::FactorBatch>(m,
        "WeightedFactorBatch",
        "Wraps any factor batch and scales residuals/Jacobians by a weight.\n\n"
        "Supports two modes:\n"
        "  1. Uniform weight (float): every factor is scaled equally.\n"
        "  2. Per-factor weights (DevicePointer): each factor gets its own weight.\n\n"
        "Parameters\n"
        "----------\n"
        "inner_factor : FactorBatch\n"
        "    The factor batch to wrap.\n"
        "weight : float, optional\n"
        "    Uniform scalar weight applied to all factors.\n"
        "weights : DevicePointer, optional\n"
        "    Device buffer with ``inner_factor.num_factors`` floats (one per factor).\n\n"
        "Exactly one of ``weight`` or ``weights`` must be provided.")
        .def("__init__", [](PyWeightedFactorBatch* self,
                            cunls::FactorBatch* inner,
                            float weight) {
            new (self) PyWeightedFactorBatch(inner, weight);
        }, nb::arg("inner_factor"), nb::arg("weight"),
           nb::keep_alive<1, 2>())
        .def("__init__", [](PyWeightedFactorBatch* self,
                            cunls::FactorBatch* inner,
                            nb::handle weights) {
            auto ptr = reinterpret_cast<const float*>(
                extract_device_ptr(weights));
            new (self) PyWeightedFactorBatch(inner, ptr);
        }, nb::arg("inner_factor"), nb::arg("weights"),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def_prop_ro("num_factors", &PyWeightedFactorBatch::NumFactors)
        .def_prop_ro("residuals_size", &PyWeightedFactorBatch::ResidualsSize)
        .def("state_block_sizes", &PyWeightedFactorBatch::StateBlockSizes);
}
