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

#include "cunls/common/cudss_dynamic.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace cunls {

namespace {

constexpr const char *kCudssSonameCandidates[] = {"libcudss.so.0", "libcudss.so"};

constexpr const char *kCudssMissingMessage =
    "cuNLS: the cuDSS-based sparse linear solver (SparseLinearSolverType::cuDSS) "
    "requires libcudss.so, which could not be found. cuDSS ships as a separate "
    "binary from cunls: download the matching cuDSS release from "
    "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/ (or "
    "install your platform's cudss package), then add the directory containing "
    "libcudss.so to LD_LIBRARY_PATH before creating a cuDSS solver. Other cunls "
    "solvers (DenseLDLT, DenseCholesky, DenseQR, BlockSparsePCG) do not require "
    "cuDSS and are unaffected.";

/**
 * @brief Resolves a cuDSS symbol, throwing a descriptive error if missing.
 *
 * A missing symbol (as opposed to a missing library) means a libcudss.so was
 * found but is a version incompatible with the cuDSS headers cunls was
 * built against.
 */
template <typename FnPtr>
FnPtr LoadCudssSymbol(void *library_handle, const char *symbol_name) {
  dlerror();  // Clear any pending error.
  void *symbol = dlsym(library_handle, symbol_name);
  const char *dlsym_error = dlerror();
  if (dlsym_error != nullptr || symbol == nullptr) {
    throw std::runtime_error(
        std::string("cuNLS: found libcudss.so but it is missing the symbol '") + symbol_name +
        "'. The loaded cuDSS library is likely a version incompatible with the "
        "cuDSS headers cunls was built against. (" +
        (dlsym_error != nullptr ? dlsym_error : "symbol not found") + ")");
  }
  return reinterpret_cast<FnPtr>(symbol);
}

/**
 * @brief Owns the dlopen() handle and the resolved cuDSS API table.
 */
struct CudssLoader {
  void *library_handle = nullptr;
  CudssApi api;
  bool available = false;
  std::string error_message;
};

void *OpenCudssLibrary() {
  for (const char *soname : kCudssSonameCandidates) {
    void *handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      return handle;
    }
  }
  return nullptr;
}

/**
 * @brief Verifies the loaded libcudss.so reports the same MAJOR.MINOR
 * version as the cuDSS headers cunls was compiled against.
 *
 * A mismatched version is the same failure mode as a missing symbol (an ABI
 * cunls cannot safely call into) but may not always manifest as one, so it
 * is checked explicitly.
 */
void CheckCudssVersion(const CudssApi &api) {
  int runtime_major = 0;
  int runtime_minor = 0;
  if (api.GetProperty(MAJOR_VERSION, &runtime_major) != CUDSS_STATUS_SUCCESS ||
      api.GetProperty(MINOR_VERSION, &runtime_minor) != CUDSS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "cuNLS: found libcudss.so but could not query its version via "
        "cudssGetProperty(). The loaded cuDSS library is likely a version "
        "incompatible with the cuDSS headers cunls was built against.");
  }

  if (runtime_major != CUDSS_VERSION_MAJOR || runtime_minor != CUDSS_VERSION_MINOR) {
    throw std::runtime_error(
        "cuNLS: found libcudss.so version " + std::to_string(runtime_major) + "." +
        std::to_string(runtime_minor) + ", but cunls was built against cuDSS " +
        std::to_string(CUDSS_VERSION_MAJOR) + "." + std::to_string(CUDSS_VERSION_MINOR) +
        " headers. Point LD_LIBRARY_PATH at a matching cuDSS " +
        std::to_string(CUDSS_VERSION_MAJOR) + "." + std::to_string(CUDSS_VERSION_MINOR) +
        ".x release instead.");
  }
}

CudssLoader LoadCudss() {
  CudssLoader loader;
  loader.library_handle = OpenCudssLibrary();
  if (loader.library_handle == nullptr) {
    loader.error_message = kCudssMissingMessage;
    return loader;
  }

  try {
    auto &api = loader.api;
    api.GetProperty =
        LoadCudssSymbol<decltype(api.GetProperty)>(loader.library_handle, "cudssGetProperty");
    api.Create = LoadCudssSymbol<decltype(api.Create)>(loader.library_handle, "cudssCreate");
    api.Destroy = LoadCudssSymbol<decltype(api.Destroy)>(loader.library_handle, "cudssDestroy");
    api.SetStream =
        LoadCudssSymbol<decltype(api.SetStream)>(loader.library_handle, "cudssSetStream");
    api.SetDeviceMemHandler = LoadCudssSymbol<decltype(api.SetDeviceMemHandler)>(
        loader.library_handle, "cudssSetDeviceMemHandler");
    api.SetThreadingLayer = LoadCudssSymbol<decltype(api.SetThreadingLayer)>(
        loader.library_handle, "cudssSetThreadingLayer");
    api.ConfigCreate =
        LoadCudssSymbol<decltype(api.ConfigCreate)>(loader.library_handle, "cudssConfigCreate");
    api.ConfigDestroy =
        LoadCudssSymbol<decltype(api.ConfigDestroy)>(loader.library_handle, "cudssConfigDestroy");
    api.ConfigSet =
        LoadCudssSymbol<decltype(api.ConfigSet)>(loader.library_handle, "cudssConfigSet");
    api.DataCreate =
        LoadCudssSymbol<decltype(api.DataCreate)>(loader.library_handle, "cudssDataCreate");
    api.DataDestroy =
        LoadCudssSymbol<decltype(api.DataDestroy)>(loader.library_handle, "cudssDataDestroy");
    api.MatrixCreateCsr = LoadCudssSymbol<decltype(api.MatrixCreateCsr)>(loader.library_handle,
                                                                         "cudssMatrixCreateCsr");
    api.MatrixCreateDn =
        LoadCudssSymbol<decltype(api.MatrixCreateDn)>(loader.library_handle, "cudssMatrixCreateDn");
    api.MatrixDestroy =
        LoadCudssSymbol<decltype(api.MatrixDestroy)>(loader.library_handle, "cudssMatrixDestroy");
    api.Execute = LoadCudssSymbol<decltype(api.Execute)>(loader.library_handle, "cudssExecute");

    CheckCudssVersion(api);

    loader.available = true;
  } catch (const std::exception &error) {
    loader.error_message = error.what();
    dlclose(loader.library_handle);
    loader.library_handle = nullptr;
    loader.api = CudssApi{};
    loader.available = false;
  }

  return loader;
}

const CudssLoader &GetCudssLoader() {
  static const CudssLoader loader = LoadCudss();
  return loader;
}

}  // namespace

bool cuDSSIsAvailable() { return GetCudssLoader().available; }

const CudssApi &GetCudssApi() {
  const CudssLoader &loader = GetCudssLoader();
  if (!loader.available) {
    throw std::runtime_error(loader.error_message);
  }
  return loader.api;
}

}  // namespace cunls
