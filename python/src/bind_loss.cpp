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
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"
#include "py_factor_wrappers.h"

void bind_loss(nb::module_ &m) {
  nb::class_<cunls::LossFunctionBatch>(m, "LossFunctionBatch",
                                       "Abstract base class for batched robust loss functions.");

  nb::class_<cunls::TrivialLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TrivialLossFunctionBatch",
      "Identity loss: rho(s)=s. Equivalent to standard least-squares.")
      .def(nb::init<>());

  nb::class_<cunls::HuberLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "HuberLossFunctionBatch",
      "Huber loss on s = |r|^2: rho(s) = s for |r| <= delta, 2 delta |r| - delta^2 beyond. "
      "delta: residual norm (residual units) where it starts to down-weight, e.g. 2-3 sigma "
      "of the inlier noise. Pass it to Problem.add_factor_batch(factors, loss, pointers).")
      .def(nb::init<float>(), nb::arg("delta"));

  nb::class_<cunls::CauchyLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "CauchyLossFunctionBatch",
      "Cauchy (Lorentzian) loss rho(s) = b log(1 + c s), s = |r|^2. With a scale k (residual "
      "units): b = k^2, c = 1 / k^2. Strong outlier suppression; pycunls.fit(loss='cauchy', "
      "loss_scale=k) builds it.")
      .def(nb::init<float, float>(), nb::arg("b"), nb::arg("c"));

  nb::class_<cunls::ArctanLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "ArctanLossFunctionBatch",
      "Arctan loss rho(s) = a atan(s / a), s = |r|^2 (b is the derivative's curvature "
      "parameter, usually 1 / a^2): bounded by a pi / 2.")
      .def(nb::init<float, float>(), nb::arg("a"), nb::arg("b"));

  nb::class_<cunls::SoftLOneLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "SoftLOneLossFunctionBatch",
      "Soft L1 loss rho(s) = 2 b (sqrt(1 + c s) - 1), s = |r|^2: quadratic near 0, linear "
      "far out. With a scale k: b = k^2, c = 1 / k^2.")
      .def(nb::init<float, float>(), nb::arg("b"), nb::arg("c"));

  nb::class_<cunls::TolerantLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TolerantLossFunctionBatch",
      "Tolerant loss (as Ceres TolerantLoss): about 0 for s = |r|^2 well below a, linear in s "
      "well above; b sets the width of the transition.")
      .def(nb::init<float, float>(), nb::arg("a"), nb::arg("b"));

  nb::class_<cunls::TukeyLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "TukeyLossFunctionBatch",
      "Tukey biweight loss: residuals with |r| > a (residual units) get zero weight (a "
      "redescending loss: needs a good initial guess).")
      .def(nb::init<float>(), nb::arg("a"));

  nb::class_<PyScaledLossFunctionBatch, cunls::LossFunctionBatch>(
      m, "ScaledLossFunctionBatch",
      "Scales the output of another loss function (loss_function) by a positive scalar a: "
      "rho_scaled(s) = a * f(s).")
      .def(nb::init<cunls::LossFunctionBatch *, float>(), nb::arg("loss_function"), nb::arg("a"),
           nb::keep_alive<1, 2>());
}
