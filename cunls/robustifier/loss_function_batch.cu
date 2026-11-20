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

// The built-in robust losses: one kernel template evaluates any of them, each
// loss is a functor computing (rho(s), rho'(s), rho''(s)) for one squared
// residual s.

#include <cuda/std/limits>
#include <sstream>
#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/scaled_loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"

namespace cunls {
namespace {

constexpr int kBlockSize = 256;

/** Smallest positive rho'(s): a loss never down-weights a residual to exactly zero. */
constexpr float kMinWeight = cuda::std::numeric_limits<float>::min();

/**
 * out[i] = (rho(s), rho'(s), rho''(s)) of s = s[i] for every residual, one thread each. `Rho`
 * writes the three components into its float3 argument (stores per branch, as written,
 * keep the registers low).
 */
template <typename Rho>
__global__ void LossKernel(Rho rho, const float *__restrict__ s, float3 *__restrict__ out,
                           int num_losses) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_losses) return;
  rho(s[i], out[i]);
}

/** Launches LossKernel on @p stream; nothing for num_losses <= 0. */
template <typename Rho>
void LaunchLoss(const Rho &rho, const float *s, float3 *out, int num_losses, cudaStream_t stream) {
  if (num_losses <= 0) return;
  const int blocks = (num_losses + kBlockSize - 1) / kBlockSize;
  LossKernel<<<blocks, kBlockSize, 0, stream>>>(rho, s, out, num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

struct TrivialRho {
  __device__ void operator()(float s, float3 &rho) const { rho = {s, 1.0f, 0.0f}; }
};

struct HuberRho {
  float delta;
  __device__ void operator()(float s, float3 &rho) const {
    const float delta_squared = delta * delta;
    if (s > delta_squared) {  // outlier
      const float r = sqrtf(s);
      rho.x = 2.0 * delta * r - delta_squared;
      rho.y = fmaxf(kMinWeight, delta / r);
      rho.z = -rho.y / (2.0 * s);
    } else {
      rho = {s, 1.0f, 0.0f};
    }
  }
};

struct CauchyRho {
  float b, c;
  __device__ void operator()(float s, float3 &rho) const {
    const float sum = 1.0f + s * c;
    const float inv = 1.0f / sum;
    rho.x = b * logf(sum);
    rho.y = fmaxf(kMinWeight, inv);
    rho.z = -c * (inv * inv);
  }
};

struct ArctanRho {
  float a, b;
  __device__ void operator()(float s, float3 &rho) const {
    const float sum = 1.0f + s * s * b;
    const float inv = 1.0f / sum;
    rho.x = a * atanf(s / a);
    rho.y = fmaxf(kMinWeight, inv);
    rho.z = -2.0f * s * b * (inv * inv);
  }
};

struct SoftLOneRho {
  float b, c;
  __device__ void operator()(float s, float3 &rho) const {
    const float sum = 1.0f + s * c;
    const float root = sqrtf(sum);
    rho.x = 2.0f * b * (root - 1.0f);
    rho.y = fmaxf(kMinWeight, 1.0f / root);
    rho.z = -(c * rho.y) / (2.0f * sum);
  }
};

struct TolerantRho {
  /** x = (s - a) / b above log(2^53): 1 + e^x rounds to e^x, so rho is linear in s. */
  static constexpr float kLog2Pow53 = 36.7f;
  float a, b, c;
  __device__ void operator()(float s, float3 &rho) const {
    const float x = (s - a) / b;
    if (x > kLog2Pow53) {
      rho.x = s - a - c;
      rho.y = 1.0f;
      rho.z = 0.0f;
    } else {
      const float e_x = expf(x);
      const float one_plus_e_x = 1.0f + e_x;
      rho.x = b * logf(one_plus_e_x) - c;
      rho.y = fmaxf(kMinWeight, e_x / one_plus_e_x);
      rho.z = 0.5f / (b * (1.0f + coshf(x)));
    }
  }
};

struct TukeyRho {
  float a_squared;
  __device__ void operator()(float s, float3 &rho) const {
    if (s <= a_squared) {
      const float value = 1.0f - s / a_squared;
      const float value_sq = value * value;
      rho.x = (a_squared / 3.0f) * (1.0f - value_sq * value);
      rho.y = value_sq;
      rho.z = -2.0f / a_squared * value;
    } else {
      rho.x = a_squared / 3.0f;
      rho.y = 0.0f;
      rho.z = 0.0f;
    }
  }
};

/** out[i] *= a (every component), one thread per residual. */
__global__ void ScaleKernel(float a, float3 *__restrict__ out, int num_losses) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_losses) return;
  float3 &rho = out[i];
  rho.x *= a;
  rho.y *= a;
  rho.z *= a;
}

}  // namespace

void TrivialLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                        cudaStream_t stream) const {
  LaunchLoss(TrivialRho{}, s, out, num_losses, stream);
}

HuberLossFunctionBatch::HuberLossFunctionBatch(float delta) : delta_(delta) {}

void HuberLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                      cudaStream_t stream) const {
  LaunchLoss(HuberRho{delta_}, s, out, num_losses, stream);
}

CauchyLossFunctionBatch::CauchyLossFunctionBatch(float b, float c) : b_(b), c_(c) {}

void CauchyLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                       cudaStream_t stream) const {
  LaunchLoss(CauchyRho{b_, c_}, s, out, num_losses, stream);
}

ArctanLossFunctionBatch::ArctanLossFunctionBatch(float a, float b) : a_(a), b_(b) {}

void ArctanLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                       cudaStream_t stream) const {
  LaunchLoss(ArctanRho{a_, b_}, s, out, num_losses, stream);
}

SoftLOneLossFunctionBatch::SoftLOneLossFunctionBatch(float b, float c) : b_(b), c_(c) {}

void SoftLOneLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                         cudaStream_t stream) const {
  LaunchLoss(SoftLOneRho{b_, c_}, s, out, num_losses, stream);
}

TolerantLossFunctionBatch::TolerantLossFunctionBatch(float a, float b)
    : a_(a), b_(b), c_(b * logf(1.0f + expf(-a / b))) {}

void TolerantLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                         cudaStream_t stream) const {
  LaunchLoss(TolerantRho{a_, b_, c_}, s, out, num_losses, stream);
}

TukeyLossFunctionBatch::TukeyLossFunctionBatch(float a) : a_squared_(a * a) {}

void TukeyLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                      cudaStream_t stream) const {
  LaunchLoss(TukeyRho{a_squared_}, s, out, num_losses, stream);
}

ScaledLossFunctionBatch::ScaledLossFunctionBatch(const LossFunctionBatch &loss_function, float a)
    : loss_function_(loss_function), a_(a) {
  if (!(a_ > 0.0f)) {
    std::ostringstream msg;
    msg << "ScaledLossFunctionBatch::ScaledLossFunctionBatch: the scale factor is not positive "
           "(got a = "
        << a_ << ", expected a > 0). Fix: pass a positive weight for the scaled loss.";
    throw std::invalid_argument(msg.str());
  }
}

void ScaledLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                       cudaStream_t stream) const {
  if (num_losses <= 0) return;
  loss_function_.Evaluate(s, out, num_losses, stream);
  ScaleKernel<<<(num_losses + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(a_, out,
                                                                                     num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
