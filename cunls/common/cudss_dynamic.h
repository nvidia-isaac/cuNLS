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

#pragma once

// cuDSS is an optional runtime dependency: cunls never links against
// libcudss.so. Instead this header exposes a table of function pointers,
// populated by dlopen()/dlsym() the first time cuDSS is actually needed
// (i.e. when SparseLinearSolverType::cuDSS is used). Using cunls without the
// cuDSS solver never touches libcudss.so at all.
//
// <cudss.h> is included only for type/enum declarations here; decltype() is
// an unevaluated context, so referencing cuDSS functions to derive their
// pointer types does not require the symbols to be resolvable at link time.
#include <cudss.h>

namespace cunls {

/**
 * @brief Table of cuDSS entry points resolved dynamically from libcudss.so.
 *
 * Member types mirror the declarations in the cuDSS header selected at
 * compile time, so this table automatically matches whichever cuDSS API
 * generation (<=0.7.x or >=0.8) cunls was built against.
 */
struct CudssApi {
  decltype(&::cudssCreate) Create = nullptr;
  decltype(&::cudssDestroy) Destroy = nullptr;
  decltype(&::cudssSetStream) SetStream = nullptr;
  decltype(&::cudssSetDeviceMemHandler) SetDeviceMemHandler = nullptr;
  decltype(&::cudssSetThreadingLayer) SetThreadingLayer = nullptr;
  decltype(&::cudssConfigCreate) ConfigCreate = nullptr;
  decltype(&::cudssConfigDestroy) ConfigDestroy = nullptr;
  decltype(&::cudssConfigSet) ConfigSet = nullptr;
  decltype(&::cudssDataCreate) DataCreate = nullptr;
  decltype(&::cudssDataDestroy) DataDestroy = nullptr;
  decltype(&::cudssMatrixCreateCsr) MatrixCreateCsr = nullptr;
  decltype(&::cudssMatrixCreateDn) MatrixCreateDn = nullptr;
  decltype(&::cudssMatrixDestroy) MatrixDestroy = nullptr;
  decltype(&::cudssExecute) Execute = nullptr;
};

/**
 * @brief Checks whether libcudss.so can be found and loaded, without
 * throwing.
 *
 * Performs (and caches) the same dlopen() probe as GetCudssApi(), but
 * reports the result instead of raising an exception. Useful for callers
 * that want to detect cuDSS availability up front (e.g. to choose a
 * fallback solver or skip a test).
 *
 * @return true if libcudss.so was found and every required symbol resolved.
 */
bool cuDSSIsAvailable();

/**
 * @brief Returns the lazily dlopen()'d cuDSS function table.
 *
 * On first call, attempts to dlopen("libcudss.so") (and versioned
 * fallbacks). The result is cached for the lifetime of the process.
 *
 * @return A reference to the populated API table.
 * @throws std::runtime_error if libcudss.so could not be found/loaded, with
 *         a message explaining that cuDSS ships separately from cunls and
 *         how to make it available (add its lib/ directory to
 *         LD_LIBRARY_PATH).
 */
const CudssApi &GetCudssApi();

}  // namespace cunls
