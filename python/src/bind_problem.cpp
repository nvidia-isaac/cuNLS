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

// Bindings for the Problem class — the top-level container that assembles
// state batches, factor batches, and (optionally) loss functions into an
// optimisation problem that a minimizer can solve.
//
// Key design notes:
//
//   nb::keep_alive<1, N>  ensures the Python objects passed to add_*() are
//   prevented from garbage-collection as long as the Problem itself is alive.
//   Without this, a CuPy array backing a StateBatch could be freed while the
//   C++ optimizer still holds a raw pointer to its GPU memory.
//
//   state_pointers are received as std::vector<uintptr_t> and reinterpret-
//   cast to float* because nanobind cannot automatically convert a Python
//   list[int] to std::vector<float*>.
//
//   Device connectivity tables are taken as
//   DevicePointer objects (a CuPy array or an int): a uint64 table of state
//   pointers (keyword-only `state_pointer_table`, so a CuPy array is never
//   mistaken for a host list), or an int32 table of state indices together
//   with the state batch of each factor slot.

#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/problem.h"

// cunls::Problem contains DeviceVector members which are move-only, but the
// compiler-generated trait reports it as copy-constructible.  Explicitly tell
// nanobind it is not, so it does not try to synthesise a copy constructor.
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template <>
struct is_copy_constructible<cunls::Problem> : std::false_type {};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

void bind_problem(nb::module_ &m) {
  nb::class_<cunls::Problem>(m, "Problem",
                             "Defines a nonlinear least-squares problem from "
                             "state and factor batches.")
      .def("__init__", [](cunls::Problem *self) { new (self) cunls::Problem(); })
      .def("add_state_batch", &cunls::Problem::AddStateBatch, nb::arg("state_batch"),
           nb::keep_alive<1, 2>(), "Register a state batch with the problem.")
      // Overload without a loss function (defaults to trivial/identity loss).
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch,
             const std::vector<uintptr_t> &state_ptrs,
             std::optional<cunls::JacobianMode> jacobian_mode_override) {
            std::vector<float *> ptrs(state_ptrs.size());
            for (size_t i = 0; i < state_ptrs.size(); ++i)
              ptrs[i] = reinterpret_cast<float *>(state_ptrs[i]);
            self.AddFactorBatch(factor_batch, ptrs, jacobian_mode_override);
          },
          nb::arg("factor_batch"), nb::arg("state_pointers"),
          nb::arg("jacobian_mode_override") = std::nullopt, nb::keep_alive<1, 2>(),
          "Add a factor batch with its state pointer connectivity. "
          "jacobian_mode_override, when set, forces this factor batch to "
          "always use the given JacobianMode regardless of the minimizer's "
          "MinimizerOptions.jacobian_mode default.")
      // Overload with an explicit robust loss function.
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch, cunls::LossFunctionBatch *loss,
             const std::vector<uintptr_t> &state_ptrs,
             std::optional<cunls::JacobianMode> jacobian_mode_override) {
            std::vector<float *> ptrs(state_ptrs.size());
            for (size_t i = 0; i < state_ptrs.size(); ++i)
              ptrs[i] = reinterpret_cast<float *>(state_ptrs[i]);
            self.AddFactorBatch(factor_batch, loss, ptrs, jacobian_mode_override);
          },
          nb::arg("factor_batch"), nb::arg("loss_function"), nb::arg("state_pointers"),
          nb::arg("jacobian_mode_override") = std::nullopt, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(),
          "Add a factor batch with a loss function and state pointer "
          "connectivity. jacobian_mode_override, when set, forces this "
          "factor batch to always use the given JacobianMode regardless of "
          "the minimizer's MinimizerOptions.jacobian_mode default.")
      // Device table of state pointers.
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch, nb::handle table,
             cunls::LossFunctionBatch *loss,
             std::optional<cunls::JacobianMode> jacobian_mode_override) {
            self.AddFactorBatch(factor_batch, loss,
                                reinterpret_cast<float *const *>(
                                    extract_device_ptr(table, "uint64", "state_pointer_table")),
                                jacobian_mode_override);
          },
          nb::arg("factor_batch"), nb::kw_only(), nb::arg("state_pointer_table"),
          nb::arg("loss_function").none() = nullptr,
          nb::arg("jacobian_mode_override") = std::nullopt, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
          "Add a factor batch whose connectivity is a device table of state pointers "
          "(uint64, capacity * B entries; entry f * B + b points at the state that factor f "
          "reads in slot b). Bound once; rewrite its contents between solves.")
      // Device table of state indices.
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch,
             const std::vector<cunls::StateBatch *> &slot_state_batches, nb::handle indices,
             cunls::LossFunctionBatch *loss,
             std::optional<cunls::JacobianMode> jacobian_mode_override) {
            self.AddFactorBatch(factor_batch, loss, slot_state_batches,
                                reinterpret_cast<const int *>(
                                    extract_device_ptr(indices, "int32", "state_indices")),
                                jacobian_mode_override);
          },
          nb::arg("factor_batch"), nb::arg("slot_state_batches"), nb::arg("state_indices"),
          nb::arg("loss_function").none() = nullptr,
          nb::arg("jacobian_mode_override") = std::nullopt, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          "Add a factor batch whose connectivity is a device table of state indices "
          "(int32, capacity * B entries): factor f reads state state_indices[f * B + b] of "
          "slot_state_batches[b]. Bound once; rewrite its contents between solves.")
      .def(
          "set_state_pointers",
          [](cunls::Problem &self, size_t residual_batch_index,
             const std::vector<uintptr_t> &state_ptrs) {
            std::vector<float *> ptrs(state_ptrs.size());
            for (size_t i = 0; i < state_ptrs.size(); ++i)
              ptrs[i] = reinterpret_cast<float *>(state_ptrs[i]);
            self.SetStatePointers(residual_batch_index, ptrs);
          },
          nb::arg("residual_batch_index"), nb::arg("state_pointers"),
          "Replace the host-list connectivity of a residual batch "
          "(num_active_factors * B pointers).")
      .def(
          "validate",
          [](const cunls::Problem &self, cunls::CudaStream &stream) {
            return self.Validate(stream.GetStream());
          },
          nb::arg("stream"),
          "GPU check of every active connection (for connectivity rewritten on the device). "
          "Synchronizes the stream; logs the first failure.")
      .def("check_consistency", &cunls::Problem::CheckConsistency,
           "Validate that all state batches and factor batches are consistent.")
      .def(
          "set_problem_partition",
          [](cunls::Problem &self, size_t num_problems, const std::vector<nb::handle> &ids) {
            std::vector<const int *> ptrs;
            for (const auto &h : ids) {
              ptrs.push_back(reinterpret_cast<const int *>(
                  extract_device_ptr(h, "int32", "state_problem_ids")));
            }
            self.SetProblemPartition(num_problems, ptrs);
          },
          nb::arg("num_problems"), nb::arg("state_problem_ids"), nb::keep_alive<1, 3>(),
          "Declare the problem a batch of independent subproblems: state s of state batch b "
          "belongs to subproblem state_problem_ids[b][s] (device int32 arrays, one per state "
          "batch). Every factor must stay within one subproblem. The minimizers then accept, "
          "damp and stop each subproblem on its own. num_problems <= 1 clears the partition.")
      .def(
          "set_state_stages",
          [](cunls::Problem &self, const std::vector<nb::handle> &stages) {
            std::vector<const int *> ptrs;
            for (const auto &h : stages) {
              ptrs.push_back(
                  reinterpret_cast<const int *>(extract_device_ptr(h, "int32", "state_stages")));
            }
            self.SetStateStages(ptrs);
          },
          nb::arg("state_stages"), nb::keep_alive<1, 2>(),
          "Declare a time ordering of the states for the BlockTridiagonal linear solver: state "
          "s of state batch b belongs to stage state_stages[b][s] (device int32 arrays, one per "
          "state batch). Factors may only connect states of the same or adjacent stages within "
          "a subproblem. An empty list clears the stages.")
      .def_prop_ro("num_problems", &cunls::Problem::NumProblems);
}
