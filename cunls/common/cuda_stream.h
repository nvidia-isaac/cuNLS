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
 * before destruction to ensure all queued work completes. The second
 * constructor wraps a stream created elsewhere (e.g. by another library's
 * stream object) without owning it.
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
   * @brief Wraps an existing stream without owning it: work goes to `external`,
   * which is neither synchronized nor destroyed with this object and must
   * outlive it.
   */
  explicit CudaStream(cudaStream_t external);

  /**
   * @brief Destroys the CUDA stream (an owned one only).
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
  cudaStream_t stream;    ///< The underlying CUDA stream handle.
  bool sync_on_destroy_;  ///< Whether to synchronize on destruction.
  bool owned_ = true;     ///< Created here (destroyed with the object), not wrapped.
};

/**
 * @brief Base of the library handle wrappers (cuBLASHandle, cuSPARSEHandle,
 * cuSolverHandle, cuDSSHandle): one handle bound to one CUDA stream at a time.
 *
 * The first GetHandle() creates the handle for its stream; a call with another
 * stream destroys it and creates a new one. The handle is owned: destroyed with
 * the object, and the object is non-copyable.
 */
class StreamBoundHandle {
 public:
  /** @brief Creates a library handle bound to `stream`; throws on a library error. */
  using CreateFn = void *(*)(cudaStream_t stream);

  /**
   * @brief Destroys a library handle: throws on a library error when `may_throw`,
   * else only logs it (the destructor).
   */
  using DestroyFn = void (*)(void *handle, bool may_throw);

  StreamBoundHandle(const StreamBoundHandle &) = delete;
  StreamBoundHandle &operator=(const StreamBoundHandle &) = delete;

  /** @brief Destroys the handle, if one was created (errors are logged, not thrown). */
  ~StreamBoundHandle();

  /**
   * @brief The handle bound to @p stream: the existing one when @p stream is the
   * stream of the previous call, else a new one (the old one is destroyed).
   *
   * @param stream CUDA stream to bind the handle to (must not be null).
   * @return An opaque pointer to the library handle.
   * @throws std::invalid_argument if @p stream is null.
   */
  void *GetHandle(cudaStream_t stream);

 protected:
  /**
   * @brief No handle yet; GetHandle() creates one.
   *
   * @param name Class name used in error messages (e.g. "cuBLASHandle").
   * @param create Creates a handle bound to a stream.
   * @param destroy Destroys a handle.
   */
  StreamBoundHandle(const char *name, CreateFn create, DestroyFn destroy);

 private:
  const char *name_;               ///< Class name for error messages.
  CreateFn create_;                ///< Creates a handle bound to a stream.
  DestroyFn destroy_;              ///< Destroys a handle.
  cudaStream_t stream_ = nullptr;  ///< Stream the current handle is bound to.
  void *handle_ = nullptr;         ///< The library handle, or null before the first GetHandle().
};
}  // namespace cunls
