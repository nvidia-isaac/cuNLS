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

#include <string>

#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Deterministic `y = H x` for a bundle-adjustment Hessian.
 *
 * `H` is a full symmetric matrix with `6 * num_cameras` pose rows followed by
 * 3 rows per point, stored as scalar CSR or as 3x3 block storage.  Each point
 * couples only to cameras (its rows hold 6-wide camera blocks followed by its
 * own 3x3 block) and each pose row lists its pose columns first.  The product
 * reads every camera-point block once, from the point rows, and accumulates
 * the pose rows in fixed point, so the result is bit-identical run to run.
 * Values of magnitude 2^76 or more after scaling, and non-finite values, send
 * the pose rows of that call to a plain scalar product.  A pose row below
 * 2^23 after scaling (2^-17 of the largest sampled term, e.g. a camera whose
 * terms are far below the sampled scale or all rounded to zero) is
 * recomputed with the same scalar product.
 *
 * Supports up to 1379 cameras (the pose rows' accumulator table lives in
 * shared memory) and 524287 points.  Requires sm_80 or newer and cooperative
 * launch.  Every call is one cooperative launch over all SMs; its counters
 * live in the instance's own buffer, so calls on one instance are
 * stream-ordered and instances are independent.
 */
class BaHessianProduct {
 public:
  /**
   * @brief Validates the sparsity pattern and builds the structure tables.
   *
   * @param row_offsets Row offsets (block-row offsets for block storage).
   * @param col_ids     Column (block-column) indices.
   * @param block_size  1 for scalar CSR, 3 for 3x3 block storage.
   * @return false when the device, the sizes or the pattern do not fit
   *         (@ref FailureReason says which); @ref Multiply must not be
   *         called then.
   */
  bool Build(cudaStream_t stream, const int *row_offsets, const int *col_ids, int block_size,
             int num_cameras, int num_points);

  /**
   * @brief `y = H x` for a matrix with the pattern given to @ref Build.
   *
   * @return false, with `y` not written, when the cooperative launch does
   *         not fit on the device (e.g. a reduced SM share under MPS);
   *         @ref Ready is false from then on.  Other CUDA errors throw.
   */
  bool Multiply(cudaStream_t stream, const int *row_offsets, const int *col_ids,
                const float *values, const float *x, float *y);

  /** @brief Whether the last @ref Build succeeded and no @ref Multiply has failed since. */
  bool Ready() const { return ready_; }

  /** @brief Why the product is not ready (empty when it is). */
  const std::string &FailureReason() const { return reason_; }

  /**
   * @brief Whether the last @ref Multiply took the scalar fallback.
   *        Synchronizes the device; for tests and diagnostics.
   */
  bool LastMultiplyUsedFallback() const;

 private:
  bool ready_ = false;
  std::string reason_;  ///< see FailureReason
  int block_size_ = 1;
  int num_pose_rows_ = 0;
  int num_points_ = 0;
  int num_sms_ = 0;
  long long point_base_ = 0;                  ///< first value of the point rows
  dvector<int> pair_ptr_;                     ///< [P + 1] cumulative camera blocks per point
  dvector<int> pair_body_;                    ///< camera of each (point, camera) block
  dvector<int> cc_row_end_;                   ///< end of each pose (block) row's pose columns
  dvector<unsigned long long> check_;         ///< pattern violations, pose-row point entries
  dvector<unsigned long long> accumulators_;  ///< [2 n_c + 3] pose-row sums, call counters
};

}  // namespace cunls
