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

#include <cuda_runtime.h>

namespace cunls {

/**
 * @brief RAII wrapper for a CUDA stream.
 *
 * Manages the lifecycle of a cudaStream_t, creating it on construction
 * and destroying it on destruction. Optionally synchronizes the stream
 * before destruction to ensure all queued work completes.
 *
 * Non-copyable to prevent accidental sharing of stream ownership.
 */
class CudaStream {
public:
  /**
   * @brief Constructs a new CUDA stream.
   *
   * @param sync_on_destroy If true, the stream will be synchronized
   *        (cudaStreamSynchronize) before being destroyed.
   */
  CudaStream(bool sync_on_destroy = false);

  /**
   * @brief Destroys the CUDA stream.
   *
   * If sync_on_destroy was set to true at construction, synchronizes
   * the stream before destruction to ensure all pending work completes.
   */
  ~CudaStream();

  /**
   * @brief Returns a reference to the underlying CUDA stream handle.
   *
   * @return Reference to the cudaStream_t handle.
   */
  cudaStream_t &GetStream();

private:
  cudaStream_t stream;   ///< The underlying CUDA stream handle.
  bool sync_on_destroy_; ///< Whether to synchronize on destruction.
};
} // namespace cunls
