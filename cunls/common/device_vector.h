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
#include <stdexcept>
#include <vector>

#include "cuda_stream.h"
#include "helper.h"

namespace cunls {

/**
 * @brief RAII container for managing CUDA device memory.
 *
 * DeviceVector provides automatic memory management for GPU memory with
 * host-to-device and device-to-host copy operations (both sync and async).
 *
 * @tparam T The element type stored in the vector. Must be trivially copyable.
 */
template <typename T>
class DeviceVector {
 public:
  /**
   * @brief Default constructor. Creates an empty vector with no allocated
   * memory.
   */
  DeviceVector() : data_(nullptr), size_(0), capacity_(0) {}

  /**
   * @brief Constructs a DeviceVector with the specified number of elements.
   *
   * @param num_elements Number of elements to allocate space for.
   */
  explicit DeviceVector(size_t num_elements)
      : data_(nullptr), size_(num_elements), capacity_(num_elements) {
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&data_, num_elements * sizeof(T)));
    }
  }

  /**
   * @brief Constructs a DeviceVector filled with a specified value.
   *
   * Allocates device memory and fills all elements with the given value.
   *
   * @param fill_value The value to fill all elements with.
   * @param num_elements Number of elements to allocate and fill.
   */
  DeviceVector(const T& fill_value, size_t num_elements)
      : data_(nullptr), size_(num_elements), capacity_(num_elements) {
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&data_, num_elements * sizeof(T)));
      std::vector<T> host_data(num_elements, fill_value);
      THROW_ON_CUDA_ERROR(cudaMemcpy(data_, host_data.data(), num_elements * sizeof(T),
                                     cudaMemcpyHostToDevice));
    }
  }

  DeviceVector(const T& fill_value, size_t num_elements, cudaStream_t stream)
      : data_(nullptr), size_(num_elements), capacity_(num_elements) {
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&data_, num_elements * sizeof(T)));
      T* pinned = nullptr;
      THROW_ON_CUDA_ERROR(cudaMallocHost(&pinned, num_elements * sizeof(T)));
      std::fill(pinned, pinned + num_elements, fill_value);
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(data_, pinned,
                                          num_elements * sizeof(T),
                                          cudaMemcpyHostToDevice, stream));
      THROW_ON_CUDA_ERROR(cudaLaunchHostFunc(stream,
          [](void* p) { cudaFreeHost(p); }, pinned));
    }
  }

  /**
   * @brief Constructs a DeviceVector from a std::vector, copying data to device.
   *
   * Allocates device memory and synchronously copies all elements from the
   * host vector to the device.
   *
   * @param host_vector The host vector to copy from.
   */
  explicit DeviceVector(const std::vector<T>& host_vector)
      : data_(nullptr), size_(host_vector.size()), capacity_(host_vector.size()) {
    if (size_ > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&data_, size_ * sizeof(T)));
      THROW_ON_CUDA_ERROR(cudaMemcpy(data_, host_vector.data(), size_ * sizeof(T),
                                     cudaMemcpyHostToDevice));
    }
  }

  DeviceVector(const std::vector<T>& host_vector, cudaStream_t stream)
      : data_(nullptr), size_(host_vector.size()), capacity_(host_vector.size()) {
    if (size_ > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&data_, size_ * sizeof(T)));
      T* pinned = nullptr;
      THROW_ON_CUDA_ERROR(cudaMallocHost(&pinned, size_ * sizeof(T)));
      std::copy(host_vector.begin(), host_vector.end(), pinned);
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(data_, pinned, size_ * sizeof(T),
                                          cudaMemcpyHostToDevice, stream));
      THROW_ON_CUDA_ERROR(cudaLaunchHostFunc(stream,
          [](void* p) { cudaFreeHost(p); }, pinned));
    }
  }

  /**
   * @brief Destructor. Frees allocated device memory.
   */
  ~DeviceVector() {
    if (data_ != nullptr) {
      WARN_ON_CUDA_ERROR(cudaFree(data_));
      data_ = nullptr;
    }
  }

  // Delete copy constructor and copy assignment
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;

  /**
   * @brief Move constructor. Transfers ownership of device memory.
   *
   * @param other The vector to move from.
   */
  DeviceVector(DeviceVector&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  /**
   * @brief Move assignment operator. Transfers ownership of device memory.
   *
   * @param other The vector to move from.
   * @return Reference to this vector.
   */
  DeviceVector& operator=(DeviceVector&& other) noexcept {
    if (this != &other) {
      // Free existing memory
      if (data_ != nullptr) {
        WARN_ON_CUDA_ERROR(cudaFree(data_));
      }

      // Transfer ownership
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;

      // Reset other
      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  /**
   * @brief Returns a pointer to the device memory.
   *
   * @return Pointer to device memory, or nullptr if empty.
   */
  T* data() noexcept { return data_; }

  /**
   * @brief Returns a const pointer to the device memory.
   *
   * @return Const pointer to device memory, or nullptr if empty.
   */
  const T* data() const noexcept { return data_; }

  /**
   * @brief Returns the number of elements currently stored.
   *
   * @return Number of elements.
   */
  size_t size() const noexcept { return size_; }

  /**
   * @brief Returns the number of elements that can be stored without
   * reallocation.
   *
   * @return Capacity in number of elements.
   */
  size_t capacity() const noexcept { return capacity_; }

  /**
   * @brief Checks if the vector is empty.
   *
   * @return true if size is 0, false otherwise.
   */
  bool empty() const noexcept { return size_ == 0; }

  /**
   * @brief Synchronously copies data from host to device.
   *
   * @param src Pointer to host memory.
   * @param num_elements Number of elements to copy.
   * @throws std::runtime_error if num_elements exceeds capacity.
   */
  void CopyFromHost(const T* src, size_t num_elements) {
    if (num_elements > capacity_) {
      throw std::runtime_error(
          "CopyFromHost: num_elements exceeds allocated capacity");
    }
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(
          cudaMemcpy(data_, src, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }
    size_ = num_elements;
  }

  /**
   * @brief Synchronously copies data from device to host.
   *
   * @param dst Pointer to host memory.
   * @param num_elements Number of elements to copy.
   * @throws std::runtime_error if num_elements exceeds size.
   */
  void CopyToHost(T* dst, size_t num_elements) const {
    if (num_elements > size_) {
      throw std::runtime_error("CopyToHost: num_elements exceeds stored size");
    }
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(
          cudaMemcpy(dst, data_, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  /**
   * @brief Asynchronously copies data from host to device.
   *
   * @param src Pointer to host memory (should be pinned for true async).
   * @param num_elements Number of elements to copy.
   * @param stream CUDA stream for the async operation.
   * @throws std::runtime_error if num_elements exceeds capacity.
   */
  void CopyFromHostAsync(const T* src, size_t num_elements, CudaStream& stream) {
    if (num_elements > capacity_) {
      throw std::runtime_error(
          "CopyFromHostAsync: num_elements exceeds allocated capacity");
    }
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(data_, src, num_elements * sizeof(T),
                                          cudaMemcpyHostToDevice,
                                          stream.GetStream()));
    }
    size_ = num_elements;
  }

  /**
   * @brief Asynchronously copies data from device to host.
   *
   * @param dst Pointer to host memory (should be pinned for true async).
   * @param num_elements Number of elements to copy.
   * @param stream CUDA stream for the async operation.
   * @throws std::runtime_error if num_elements exceeds size.
   */
  void CopyToHostAsync(T* dst, size_t num_elements, CudaStream& stream) const {
    if (num_elements > size_) {
      throw std::runtime_error(
          "CopyToHostAsync: num_elements exceeds stored size");
    }
    if (num_elements > 0) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dst, data_, num_elements * sizeof(T),
                                          cudaMemcpyDeviceToHost,
                                          stream.GetStream()));
    }
  }

  /**
   * @brief Resizes the vector to the specified size.
   *
   * If new_size > capacity, reallocates memory. If preserve_data is true,
   * existing data is copied to the new allocation.
   *
   * @param new_size New size in number of elements.
   * @param preserve_data If true, preserves existing data during reallocation.
   */
  void resize(size_t new_size, bool preserve_data = false) {
    if (new_size <= capacity_) {
      size_ = new_size;
      return;
    }

    // Need to allocate new memory
    T* new_data = nullptr;
    if (new_size > 0) {
      THROW_ON_CUDA_ERROR(cudaMalloc(&new_data, new_size * sizeof(T)));
    }

    // Copy existing data if requested
    if (preserve_data && data_ != nullptr && size_ > 0) {
      THROW_ON_CUDA_ERROR(cudaMemcpy(new_data, data_, size_ * sizeof(T),
                                     cudaMemcpyDeviceToDevice));
    }

    // Free old memory
    if (data_ != nullptr) {
      WARN_ON_CUDA_ERROR(cudaFree(data_));
    }

    data_ = new_data;
    capacity_ = new_size;
    size_ = new_size;
  }

  /**
   * @brief Reserves capacity without changing size.
   *
   * If new_capacity > current capacity, reallocates memory and preserves
   * existing data.
   *
   * @param new_capacity New capacity in number of elements.
   */
  void reserve(size_t new_capacity) {
    if (new_capacity <= capacity_) {
      return;
    }

    T* new_data = nullptr;
    THROW_ON_CUDA_ERROR(cudaMalloc(&new_data, new_capacity * sizeof(T)));

    // Copy existing data
    if (data_ != nullptr && size_ > 0) {
      THROW_ON_CUDA_ERROR(cudaMemcpy(new_data, data_, size_ * sizeof(T),
                                     cudaMemcpyDeviceToDevice));
    }

    // Free old memory
    if (data_ != nullptr) {
      WARN_ON_CUDA_ERROR(cudaFree(data_));
    }

    data_ = new_data;
    capacity_ = new_capacity;
  }

  /**
   * @brief Clears the vector, setting size to 0 but keeping capacity.
   */
  void clear() noexcept { size_ = 0; }

  /**
   * @brief Reduces capacity to match size, freeing unused memory.
   */
  void shrink_to_fit() {
    if (capacity_ == size_) {
      return;
    }

    if (size_ == 0) {
      // Free all memory
      if (data_ != nullptr) {
        WARN_ON_CUDA_ERROR(cudaFree(data_));
        data_ = nullptr;
      }
      capacity_ = 0;
      return;
    }

    // Allocate new smaller buffer
    T* new_data = nullptr;
    THROW_ON_CUDA_ERROR(cudaMalloc(&new_data, size_ * sizeof(T)));

    // Copy data
    THROW_ON_CUDA_ERROR(
        cudaMemcpy(new_data, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));

    // Free old memory
    WARN_ON_CUDA_ERROR(cudaFree(data_));

    data_ = new_data;
    capacity_ = size_;
  }

 private:
  T* data_;         ///< Pointer to device memory
  size_t size_;     ///< Number of elements stored
  size_t capacity_; ///< Number of elements allocated
};

}  // namespace cunls
