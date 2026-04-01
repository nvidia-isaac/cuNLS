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

/**
 * @file cunls.h
 * @brief Main umbrella header for the cuNLS library.
 *
 * Including this header provides access to all public cuNLS components:
 * factors, minimizers, state blocks, and robustifier loss functions.
 *
 * API terminology: optimization variables are referred to as **states** (and
 * state blocks). Factor batches consume state pointers and produce residuals
 * and Jacobians; minimizers update states via state batch operations.
 */

#pragma once

#include "cunls/factor/information_factor_batch.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/factor/similarity2_between_factor_batch.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/factor/sl4_between_factor_batch.h"
#include "cunls/factor/sl4_prior_factor_batch.h"
#include "cunls/factor/so2_between_factor_batch.h"
#include "cunls/factor/so2_prior_factor_batch.h"
#include "cunls/factor/so3_between_factor_batch.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/factor/vector_between_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
