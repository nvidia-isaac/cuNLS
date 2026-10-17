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

#include "cunls/minimizer/linear_system.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

void LinearSystem::ExtractScaledHessianDiagonal(cudaStream_t stream,
                                                dvector<float> &diagonal) const {
  ExtractHessianDiagonal(stream, diagonal);
  if (column_scale_ == nullptr || diagonal.empty()) return;
  ElementwiseMultiplyInPlace(stream, diagonal.data(), column_scale_->data(), diagonal.size());
  ElementwiseMultiplyInPlace(stream, diagonal.data(), column_scale_->data(), diagonal.size());
}

}  // namespace cunls
