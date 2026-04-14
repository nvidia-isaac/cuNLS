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

// Bindings for robust loss (robustifier) function batches.
//
// A LossFunctionBatch applies an element-wise scalar function rho(s) to
// squared residuals, down-weighting outliers.  The base class is abstract;
// each concrete subclass wraps a different robust kernel.  They are
// stateless after construction (parameters are baked in) and are passed to
// Problem.add_factor_batch() alongside a FactorBatch.

#include "bindings.h"
#include "py_factor_wrappers.h"

#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"

void bind_loss(nb::module_ &m) {
  nb::class_<cunls::LossFunctionBatch>(
      m, "LossFunctionBatch",
      "Abstract base class for batched robust loss functions.");

  nb::class_<cunls::TrivialLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TrivialLossFunctionBatch",
      "Identity loss: rho(s)=s. Equivalent to standard least-squares.")
      .def(nb::init<>());

  nb::class_<cunls::HuberLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "HuberLossFunctionBatch",
      "Huber loss: quadratic for small residuals, linear for large.")
      .def(nb::init<float>(), nb::arg("delta"));

  nb::class_<cunls::CauchyLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "CauchyLossFunctionBatch", "Cauchy (Lorentzian) robust loss function.")
      .def(nb::init<float, float>(), nb::arg("b"), nb::arg("c"));

  nb::class_<cunls::ArctanLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "ArctanLossFunctionBatch", "Arctan robust loss function.")
      .def(nb::init<float, float>(), nb::arg("a"), nb::arg("b"));

  nb::class_<cunls::SoftLOneLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "SoftLOneLossFunctionBatch", "Soft L1 robust loss function.")
      .def(nb::init<float, float>(), nb::arg("b"), nb::arg("c"));

  nb::class_<cunls::TolerantLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TolerantLossFunctionBatch", "Tolerant robust loss function.")
      .def(nb::init<float, float>(), nb::arg("a"), nb::arg("b"));

  nb::class_<cunls::TukeyLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TukeyLossFunctionBatch", "Tukey's biweight robust loss function.")
      .def(nb::init<float>(), nb::arg("a"));

  nb::class_<PyScaledLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "ScaledLossFunctionBatch",
      "Scales the output of another loss function by a positive scalar a: "
      "rho_scaled(s) = a * f(s).")
      .def(nb::init<cunls::LossFunctionBatch *, float>(),
           nb::arg("loss_function"), nb::arg("a"), nb::keep_alive<1, 2>());
}
