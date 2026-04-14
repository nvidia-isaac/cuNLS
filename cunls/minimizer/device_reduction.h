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

#include <cstddef>

namespace cunls {

/**
 * Async sum reduction: d_output[0] = sum(input[0..n-1]).
 * Result stays in device memory; caller is responsible for D2H copy and sync.
 *
 * @param d_partials Scratch buffer with at least ReducePartialCount(n) floats.
 */
void ReduceSumToDevice(cudaStream_t stream, const float *input, size_t n,
                       float *d_output, float *d_partials);

/**
 * Async dot product: d_output[0] = sum(a[i]*b[i]).
 * Result stays in device memory.
 */
void DotProductToDevice(cudaStream_t stream, const float *a, const float *b,
                        size_t n, float *d_output, float *d_partials);

/**
 * Async weighted dot product: d_output[0] = sum(a[i]*w[i]*b[i]).
 * Result stays in device memory.
 */
void WeightedDotProductToDevice(cudaStream_t stream, const float *a,
                                const float *w, const float *b, size_t n,
                                float *d_output, float *d_partials);

/**
 * Returns the number of floats needed for the partials scratch buffer
 * given an input of size n.
 */
size_t ReducePartialCount(size_t n);

} // namespace cunls
