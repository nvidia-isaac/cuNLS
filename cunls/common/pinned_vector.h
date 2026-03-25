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

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

#include "helper.h"

namespace cunls {

/**
 * @brief RAII container for CUDA page-locked (pinned) host memory.
 *
 * Pinned (page-locked) host memory is required for fully asynchronous
 * device-to-host and host-to-device transfers via cudaMemcpyAsync.
 * Regular (pageable) memory may cause the CUDA driver to fall back to a
 * synchronous copy, defeating the purpose of async transfers.
 *
 * This class mirrors the interface of DeviceVector but allocates on the
 * host side with cudaMallocHost / cudaFreeHost.  Typical use cases include
 * small status or flag buffers that are written on the GPU (into a
 * companion dvector) and then copied back to this pinned buffer in a
 * single cudaMemcpyAsync call.
 *
 * Copy construction and copy assignment are deleted; move semantics
 * transfer ownership of the underlying allocation.
 *
 * @tparam T Element type. Must be trivially copyable.
 */
template <typename T>
class PinnedVector {
 public:
  /** @brief Default constructor. Creates an empty vector with no allocation. */
  PinnedVector() : data_(nullptr), size_(0), capacity_(0) {}

  /**
   * @brief Constructs a PinnedVector with the specified number of elements.
   *
   * Allocates page-locked host memory via cudaMallocHost.  The contents of
   * the allocated memory are uninitialized.
   *
   * @param num_elements Number of elements to allocate space for.
   */
  explicit PinnedVector(size_t num_elements)
      : data_(nullptr), size_(num_elements), capacity_(num_elements) {
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMallocHost(&data_, num_elements * sizeof(T)));
    }
  }

  /** @brief Destructor. Releases page-locked memory via cudaFreeHost. */
  ~PinnedVector() {
    if (data_) {
      WARN_ON_CUDA_ERROR(cudaFreeHost(data_));
      data_ = nullptr;
    }
  }

  /** @brief Deleted copy constructor. */
  PinnedVector(const PinnedVector&) = delete;
  /** @brief Deleted copy assignment operator. */
  PinnedVector& operator=(const PinnedVector&) = delete;

  /**
   * @brief Move constructor. Transfers ownership of pinned memory.
   * @param other The vector to move from (left empty after the call).
   */
  PinnedVector(PinnedVector&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  /**
   * @brief Move assignment operator. Transfers ownership of pinned memory.
   *
   * Any previously held allocation is freed before taking ownership.
   *
   * @param other The vector to move from (left empty after the call).
   * @return Reference to this vector.
   */
  PinnedVector& operator=(PinnedVector&& other) noexcept {
    if (this != &other) {
      if (data_) {
        WARN_ON_CUDA_ERROR(cudaFreeHost(data_));
      }
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  /** @brief Returns a pointer to the pinned host memory. */
  T* data() noexcept { return data_; }
  /** @brief Returns a const pointer to the pinned host memory. */
  const T* data() const noexcept { return data_; }
  /** @brief Returns the number of elements currently stored. */
  size_t size() const noexcept { return size_; }
  /** @brief Returns the allocated capacity in number of elements. */
  size_t capacity() const noexcept { return capacity_; }
  /** @brief Checks if the vector is empty. */
  bool empty() const noexcept { return size_ == 0; }

  /** @brief Element access (no bounds checking). */
  T& operator[](size_t i) { return data_[i]; }
  /** @brief Const element access (no bounds checking). */
  const T& operator[](size_t i) const { return data_[i]; }

  /**
   * @brief Resizes the vector to the specified size.
   *
   * If @p new_size fits within the current capacity the size is updated
   * without any allocation.  Otherwise a new pinned allocation is made;
   * the old one is freed after optionally copying existing data.
   *
   * @param new_size New size in number of elements.
   * @param preserve_data If true, existing data is copied to the new
   *                      allocation via std::memcpy (host-to-host).
   */
  void resize(size_t new_size, bool preserve_data = false) {
    if (new_size <= capacity_) {
      size_ = new_size;
      return;
    }

    T* new_data = nullptr;
    if (new_size > 0) {
      THROW_ON_CUDA_ERROR(cudaMallocHost(&new_data, new_size * sizeof(T)));
    }

    if (preserve_data && data_ != nullptr && size_ > 0) {
      std::memcpy(new_data, data_, size_ * sizeof(T));
    }

    if (data_ != nullptr) {
      WARN_ON_CUDA_ERROR(cudaFreeHost(data_));
    }

    data_ = new_data;
    capacity_ = new_size;
    size_ = new_size;
  }

 private:
  T* data_;         ///< Pointer to page-locked host memory.
  size_t size_;     ///< Number of elements stored.
  size_t capacity_; ///< Number of elements allocated.
};

}  // namespace cunls
