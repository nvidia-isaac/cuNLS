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

#include "cunls/common/manifold.h"
#include "cunls/factor/between/between_factor_batch.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/information/motion_prior_information.h"
#include "cunls/factor/motion/constant_acceleration_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/prior/prior_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/prior/so2_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/scaled_loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
