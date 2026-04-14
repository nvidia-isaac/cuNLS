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

#include "bindings.h"

#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "cunls/common/device_vector.h"
#include "cunls/minimizer/problem.h"

// cunls::Problem contains DeviceVector members which are move-only, but the
// compiler-generated trait reports it as copy-constructible.  Explicitly tell
// nanobind it is not, so it does not try to synthesise a copy constructor.
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template <> struct is_copy_constructible<cunls::Problem> : std::false_type {};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

void bind_problem(nb::module_ &m) {
  nb::class_<cunls::Problem>(m, "Problem",
                             "Defines a nonlinear least-squares problem from "
                             "state and factor batches.")
      .def("__init__",
           [](cunls::Problem *self) { new (self) cunls::Problem(); })
      .def("add_state_batch", &cunls::Problem::AddStateBatch,
           nb::arg("state_batch"), nb::keep_alive<1, 2>(),
           "Register a state batch with the problem.")
      // Overload without a loss function (defaults to trivial/identity loss).
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch,
             const std::vector<uintptr_t> &state_ptrs) {
            std::vector<float *> ptrs(state_ptrs.size());
            for (size_t i = 0; i < state_ptrs.size(); ++i)
              ptrs[i] = reinterpret_cast<float *>(state_ptrs[i]);
            self.AddFactorBatch(factor_batch, ptrs);
          },
          nb::arg("factor_batch"), nb::arg("state_pointers"),
          nb::keep_alive<1, 2>(),
          "Add a factor batch with its state pointer connectivity.")
      // Overload with an explicit robust loss function.
      .def(
          "add_factor_batch",
          [](cunls::Problem &self, cunls::FactorBatch *factor_batch,
             cunls::LossFunctionBatch *loss,
             const std::vector<uintptr_t> &state_ptrs) {
            std::vector<float *> ptrs(state_ptrs.size());
            for (size_t i = 0; i < state_ptrs.size(); ++i)
              ptrs[i] = reinterpret_cast<float *>(state_ptrs[i]);
            self.AddFactorBatch(factor_batch, loss, ptrs);
          },
          nb::arg("factor_batch"), nb::arg("loss_function"),
          nb::arg("state_pointers"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(),
          "Add a factor batch with a loss function and state pointer "
          "connectivity.")
      .def(
          "check_consistency", &cunls::Problem::CheckConsistency,
          "Validate that all state batches and factor batches are consistent.");
}
