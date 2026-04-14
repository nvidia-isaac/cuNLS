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

#include "cunls/common/cuda_stream.h"

#include "cunls/common/helper.h"

namespace cunls {

/**
 * @brief Constructs a CudaStream and creates the underlying CUDA stream.
 *
 * Creates a new stream. Throws
 * on any CUDA error.
 *
 * @param sync_on_destroy If true, the stream will be synchronized before
 *                        destruction.
 */
CudaStream::CudaStream(bool sync_on_destroy)
    : sync_on_destroy_(sync_on_destroy) {
  THROW_ON_CUDA_ERROR(cudaStreamCreate(&stream));
}

/**
 * @brief Destroys the CudaStream and releases the underlying CUDA stream.
 *
 * If sync_on_destroy_ is true, synchronizes the stream first to ensure
 * all enqueued operations complete. Uses WARN macros to avoid throwing
 * from a destructor.
 */
CudaStream::~CudaStream() {
  if (sync_on_destroy_) {
    WARN_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
  WARN_ON_CUDA_ERROR(cudaStreamDestroy(stream));
}

/** @brief Returns a reference to the underlying cudaStream_t handle. */
cudaStream_t &CudaStream::GetStream() { return stream; }
} // namespace cunls
