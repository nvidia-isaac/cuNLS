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

#include "cunls/common/profiler.h"

#ifdef ENABLE_PROFILING
#include <random>

#include <nvtx3/nvtx3.hpp>
#endif

namespace cunls::profiler {
#ifdef ENABLE_PROFILING

namespace internal {
/**
 * @brief Constructs a domain range with specified attributes
 * @param handle The NVTX domain handle to associate this range with
 * @param name The name to display in the profiler for this range
 * @param color The color for this range (ARGB format, default is 0)
 */
DomainRange::DomainRange(void* handle, const std::string& name, uint32_t color)
    : handle_(handle), name_(name) {
  nvtxMessageValue_t message_;
  message_.ascii = name_.c_str();
  nvtxEventAttributes_t eventAttrib = {NVTX_VERSION,
                                       NVTX_EVENT_ATTRIB_STRUCT_SIZE,
                                       0,
                                       NVTX_COLOR_ARGB,
                                       color,
                                       NVTX_PAYLOAD_UNKNOWN,
                                       0,
                                       {},
                                       NVTX_MESSAGE_TYPE_ASCII,
                                       message_};
  nvtxDomainRangePushEx((nvtxDomainHandle_t)handle_, &eventAttrib);
}

/**
 * @brief Destructor that pops the range from the domain stack
 */
DomainRange::~DomainRange() {
  if (handle_) {
    nvtxDomainRangePop((nvtxDomainHandle_t)handle_);
  }
}
}  // namespace internal

ScopedRange::ScopedRange(const std::string& name) : name_(name) {
  nvtxRangePushA(name_.c_str());
};

/**
 * @brief Destructor that pops the range from the NVTX stack
 */
ScopedRange::~ScopedRange() { nvtxRangePop(); }

/**
 * @brief Constructs a new profiling domain with a random color
 * @param name The name of the domain (visible in profiling tools)
 */
Domain::Domain(const std::string& name)
    : name_(name), handle_(nvtxDomainCreateA(name_.c_str())) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> distrib(0, 0xFFFFFF);
  color_ = distrib(gen);
}

/**
 * @brief Destructor that destroys the NVTX domain
 */
Domain::~Domain() { nvtxDomainDestroy((nvtxDomainHandle_t)handle_); }

/**
 * @brief Creates a new profiling range within this domain
 * @param name The name for the range (visible in profiling tools)
 * @return A DomainRange object that will automatically close when destroyed
 */
internal::DomainRange Domain::CreateDomainRange(const std::string& name) const {
  return {handle_, name, color_};
}

#else
namespace internal {
/**
 * @brief Constructs a domain range with specified attributes
 * @param handle The NVTX domain handle to associate this range with
 * @param name The name to display in the profiler for this range
 * @param color The color for this range (ARGB format, default is 0)
 */
DomainRange::DomainRange(void* handle, const std::string& name,
                         uint32_t color) {}

/**
 * @brief Destructor that pops the range from the domain stack
 */
DomainRange::~DomainRange() {}
}  // namespace internal

ScopedRange::ScopedRange(const std::string& name) {}

/**
 * @brief Destructor that pops the range from the NVTX stack
 */
ScopedRange::~ScopedRange() {}

/**
 * @brief Constructs a new profiling domain with a random color
 * @param name The name of the domain (visible in profiling tools)
 */
Domain::Domain(const std::string& name) {}

/**
 * @brief Destructor that destroys the NVTX domain
 */
Domain::~Domain() {}

/**
 * @brief Creates a new profiling range within this domain
 * @param name The name for the range (visible in profiling tools)
 * @return A DomainRange object that will automatically close when destroyed
 */
internal::DomainRange Domain::CreateDomainRange(const std::string& name) const {
  return {nullptr, name, 0};
}

#endif
}  // namespace cunls::profiler
