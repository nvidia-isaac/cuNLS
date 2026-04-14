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
#include <cstdint>
#include <string>

namespace cunls::profiler {

/**
 * @brief RAII wrapper for an NVTX range used for profiling.
 *
 * Pushes an NVTX range on construction and pops it on destruction,
 * enabling automatic scope-based profiling annotation. Non-copyable
 * and non-movable to ensure one-to-one scope-to-range mapping.
 */
class ScopedRange {
public:
  ScopedRange(const ScopedRange &) = delete;
  ScopedRange &operator=(const ScopedRange &) = delete;
  ScopedRange(ScopedRange &&) = delete;
  ScopedRange &operator=(ScopedRange &&) = delete;

  /**
   * @brief Constructs a ScopedRange and pushes an NVTX range.
   *
   * @param name Human-readable label for the profiling range.
   */
  ScopedRange(const std::string &name);

  /**
   * @brief Destructor that pops the NVTX range.
   */
  ~ScopedRange();

private:
  std::string name_; ///< Label for the profiling range.
};

namespace internal {

/**
 * @brief RAII wrapper for a domain-scoped NVTX range.
 *
 * Starts a colored NVTX range within a specific domain on construction
 * and ends it on destruction. Used internally by Domain::CreateDomainRange.
 */
class DomainRange {
public:
  DomainRange(const DomainRange &) = delete;
  DomainRange &operator=(const DomainRange &) = delete;
  DomainRange(DomainRange &&) = delete;
  DomainRange &operator=(DomainRange &&) = delete;

  /**
   * @brief Constructs a DomainRange and starts an NVTX range within the domain.
   *
   * @param handle Opaque pointer to the NVTX domain handle.
   * @param name   Human-readable label for the profiling range.
   * @param color  ARGB color for the range visualization (0 = default).
   */
  DomainRange(void *handle, const std::string &name, uint32_t color = 0);

  /**
   * @brief Destructor that ends the NVTX domain range.
   */
  ~DomainRange();

private:
  void *handle_ = nullptr; ///< Handle to the NVTX domain.
  std::string name_;       ///< Copy of the name string to ensure lifetime.
};

} // namespace internal

/**
 * @brief NVTX profiling domain with automatic color cycling.
 *
 * Represents a named NVTX domain that groups related profiling ranges.
 * Each domain assigns incrementing colors to its ranges for visual
 * distinction in profiling tools such as Nsight Systems.
 *
 * Non-copyable and non-movable to ensure unique domain ownership.
 */
class Domain {
public:
  Domain(const Domain &) = delete;
  Domain &operator=(const Domain &) = delete;
  Domain(Domain &&) = delete;
  Domain &operator=(Domain &&) = delete;

  /**
   * @brief Constructs a named NVTX domain.
   *
   * @param name Human-readable name for the profiling domain.
   */
  Domain(const std::string &name);

  /**
   * @brief Destroys the NVTX domain and releases its handle.
   */
  ~Domain();

  /**
   * @brief Creates a scoped NVTX range within this domain.
   *
   * The returned DomainRange object automatically starts the range on
   * construction and ends it on destruction.
   *
   * @param name Label for the range within this domain.
   * @return DomainRange RAII object that manages the range lifetime.
   */
  internal::DomainRange CreateDomainRange(const std::string &name) const;

private:
  std::string name_;       ///< Copy of the name string to ensure lifetime.
  void *handle_ = nullptr; ///< Handle to the NVTX domain.
  uint32_t color_ = 0;     ///< Current color counter for range cycling.
};
} // namespace cunls::profiler
