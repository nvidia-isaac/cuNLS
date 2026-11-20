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

/** @file lie_math_test.cu
 *  @brief Tests for Lie group math operations (SE3 left Jacobian and its
 * inverse, SO3 log at the edge cases).
 */

#include <gtest/gtest.h>
#include <thrust/device_ptr.h>

#include <cmath>
#include <random>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

/**
 * @brief CUDA kernel to multiply two square matrices: C = A * B
 *
 * Multiplies two NxN matrices stored in row-major format.
 * Matrices are stored with given pitch (stride between rows) and stride
 * (stride between matrices in the batch).
 *
 * @tparam N Matrix dimension (e.g., 3 for 3x3, 6 for 6x6)
 * @param A First matrix (NxN, row-major)
 * @param A_pitch Pitch of matrix A (stride between rows)
 * @param A_stride Stride between matrices in A
 * @param B Second matrix (NxN, row-major)
 * @param B_pitch Pitch of matrix B (stride between rows)
 * @param B_stride Stride between matrices in B
 * @param C Output matrix (NxN, row-major)
 * @param C_pitch Pitch of matrix C (stride between rows)
 * @param C_stride Stride between matrices in C
 * @param size Number of matrix pairs to multiply
 */
template <int N>
__global__ void matmul_batch_kernel(const float *A, const size_t A_pitch, const size_t A_stride,
                                    const float *B, const size_t B_pitch, const size_t B_stride,
                                    float *C, const size_t C_pitch, const size_t C_stride,
                                    size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  const float *A_ptr = A + tid * A_stride;
  const float *B_ptr = B + tid * B_stride;
  float *C_ptr = C + tid * C_stride;

  // Compute C = A * B (row-major)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += A_ptr[i * A_pitch + k] * B_ptr[k * B_pitch + j];
      }
      C_ptr[i * C_pitch + j] = sum;
    }
  }
}

/**
 * @brief Test fixture for SO3 and SE3 Lie group operations.
 *
 * Sets up random SE3 twists and 6x6 matrix buffers for the SE(3) Jacobian
 * test; the SO(3) log tests build their own inputs.
 */
class LieMathTest : public ::testing::Test {
 public:
  void SetUp() override {
    se3_twists_.resize(num_twists_);
    // 6x6 matrices (SE3 jacobians and their products)
    matrices_6x6_a_.resize(num_twists_ * 36);
    matrices_6x6_c_.resize(num_twists_ * 36);
    matrices_6x6_products_.resize(num_twists_ * 36);

    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> twist_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> translation_dist(-1.0f, 1.0f);

    // Initialize random SE3 twists (6D vectors: 3 rotation + 3 translation)
    for (size_t i = 0; i < num_twists_; i++) {
      Vector<6> &twist = se3_twists_[i];
      twist[0] = twist_dist(rng);        // rotation x
      twist[1] = twist_dist(rng);        // rotation y
      twist[2] = twist_dist(rng);        // rotation z
      twist[3] = translation_dist(rng);  // translation x
      twist[4] = translation_dist(rng);  // translation y
      twist[5] = translation_dist(rng);  // translation z
    }
    se3_twists_device_ = DeviceVector<Vector<6>>(se3_twists_);
  }

  const size_t num_twists_ = 10000;
  const uint32_t fixed_seed_ = 42;
  std::vector<Vector<6>> se3_twists_;
  DeviceVector<Vector<6>> se3_twists_device_;
  DeviceVector<float> matrices_6x6_a_;
  DeviceVector<float> matrices_6x6_c_;
  DeviceVector<float> matrices_6x6_products_;

  profiler::Domain profiler_domain_{"LieMathTest"};
};

/**
 * @brief Tests that J_l(twist) (ComputeJacobianLeftSE3) times its inverse
 * J_l^{-1}(twist) = J_r^{-1}(-twist) (ComputeJacobianRightInverseSE3) is the
 * identity matrix.
 *
 * This test verifies the mathematical property that multiplying the left
 * Jacobian of SE(3) with its inverse should result in the identity matrix
 * (within numerical tolerance). This validates the correctness of both
 * operations and their inverse relationship.
 *
 * Test procedure:
 * 1. Start with random SE3 twists (6D vectors: 3 rotation + 3 translation)
 * 2. Apply ComputeJacobianLeftSE3: J = J_l(twist) to get Jacobian matrices
 * 3. Apply ComputeJacobianRightInverseSE3 to -twist: J_inv = J_l^{-1}(twist)
 * 4. Multiply J * J_inv
 * 5. Verify result ≈ identity matrix
 */
TEST_F(LieMathTest, JacobianLeftSE3ThenInverse) {
  auto test_range = this->profiler_domain_.CreateDomainRange("JacobianLeftSE3ThenInverseTest");

  // Get device pointers (reusing 6x6 matrices)
  const float *se3_twists_ptr = reinterpret_cast<const float *>(se3_twists_device_.data());
  float *se3_jacobians_left_ptr = reinterpret_cast<float *>(matrices_6x6_a_.data());
  float *se3_jacobians_left_inverse_ptr = reinterpret_cast<float *>(matrices_6x6_c_.data());
  float *se3_jacobian_products_ptr = reinterpret_cast<float *>(matrices_6x6_products_.data());

  // J_l^{-1}(twist) = J_r^{-1}(-twist).
  std::vector<Vector<6>> negated_twists(se3_twists_);
  for (auto &twist : negated_twists)
    for (float &v : twist) v = -v;
  DeviceVector<Vector<6>> negated_twists_device(negated_twists);
  const float *negated_twists_ptr = reinterpret_cast<const float *>(negated_twists_device.data());

  CudaStream stream;

  const size_t twist_stride = 6;
  const size_t jacobian_pitch = 6;
  const size_t jacobian_stride = 36;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeJacobianLeftSE3(stream.GetStream(), se3_twists_ptr, twist_stride, jacobian_pitch,
                           jacobian_stride, this->num_twists_, se3_jacobians_left_ptr);
    ComputeJacobianRightInverseSE3(stream.GetStream(), negated_twists_ptr, twist_stride,
                                   jacobian_pitch, jacobian_stride, this->num_twists_,
                                   se3_jacobians_left_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("JacobianLeftSE3");
    // Apply JacobianLeftSE3: J = J_l(twist)
    ComputeJacobianLeftSE3(stream.GetStream(), se3_twists_ptr,
                           6,  // twist_stride = 6
                           jacobian_pitch, jacobian_stride, this->num_twists_,
                           se3_jacobians_left_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that jacobians are NOT all identity matrices (ComputeJacobianLeftSE3
  // actually changed something)
  {
    std::vector<float> host_se3_jacobians_left(matrices_6x6_a_.size());
    matrices_6x6_a_.CopyToHost(host_se3_jacobians_left.data(), host_se3_jacobians_left.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 6; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(host_se3_jacobians_left[i * 36 + row * 6 + col] - expected) > 1e-5f) {
            found_non_identity = true;
            break;
          }
        }
        if (found_non_identity) break;
      }
    }
    ASSERT_TRUE(found_non_identity);
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("JacobianLeftInverseSE3");
    // J_inv = J_l^{-1}(twist) = J_r^{-1}(-twist)
    ComputeJacobianRightInverseSE3(stream.GetStream(), negated_twists_ptr, twist_stride,
                                   jacobian_pitch, jacobian_stride, this->num_twists_,
                                   se3_jacobians_left_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("MatrixMultiply");
    // Multiply J * J_inv
    constexpr size_t block_size = 256;
    size_t num_blocks = (this->num_twists_ + block_size - 1) / block_size;
    matmul_batch_kernel<6><<<num_blocks, block_size, 0, stream.GetStream()>>>(
        se3_jacobians_left_ptr, jacobian_pitch,
        jacobian_stride,  // A: pitch=6, stride=36
        se3_jacobians_left_inverse_ptr, jacobian_pitch,
        jacobian_stride,  // B: pitch=6, stride=36
        se3_jacobian_products_ptr, jacobian_pitch,
        jacobian_stride,  // C: pitch=6, stride=36
        this->num_twists_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<float> host_se3_jacobian_products(matrices_6x6_products_.size());
  matrices_6x6_products_.CopyToHost(host_se3_jacobian_products.data(),
                                    host_se3_jacobian_products.size());

  // Verify that products are identity matrices (J * J_inv = I)
  for (size_t i = 0; i < this->num_twists_; i++) {
    for (int row = 0; row < 6; row++) {
      for (int col = 0; col < 6; col++) {
        ASSERT_NEAR(host_se3_jacobian_products[i * 36 + row * 6 + col], (row == col) ? 1.0f : 0.0f,
                    1e-4f);
      }
    }
  }
}

/**
 * @brief Tests LogSO3 on rotations near and at 180 degrees.
 *
 * Verifies that:
 * 1. Exact 180° rotation about each axis produces a twist with magnitude π.
 * 2. Near-180° rotations produce finite (non-NaN) twists.
 * 3. ExpSO3(LogSO3(R)) ≈ R for these edge-case rotations.
 */
TEST_F(LieMathTest, LogSO3_180DegreeRotation) {
  // Build rotation matrices for 180° about x, y, z axes and several
  // near-180° angles.
  constexpr float pi = static_cast<float>(M_PI);
  const float angles[] = {pi, pi - 1e-4f, pi - 1e-3f, pi - 1e-2f};
  const float axes[][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.5774f, 0.5774f, 0.5774f}};
  const size_t num_cases = sizeof(angles) / sizeof(angles[0]) * sizeof(axes) / sizeof(axes[0]);

  std::vector<float> rotations_host(num_cases * 9);
  size_t idx = 0;
  for (float angle : angles) {
    for (const auto &ax : axes) {
      float nx = ax[0], ny = ax[1], nz = ax[2];
      float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len;
      ny /= len;
      nz /= len;
      float c = std::cos(angle), s = std::sin(angle);
      float t = 1.0f - c;
      float *R = rotations_host.data() + idx * 9;
      R[0] = t * nx * nx + c;
      R[1] = t * nx * ny - s * nz;
      R[2] = t * nx * nz + s * ny;
      R[3] = t * nx * ny + s * nz;
      R[4] = t * ny * ny + c;
      R[5] = t * ny * nz - s * nx;
      R[6] = t * nx * nz - s * ny;
      R[7] = t * ny * nz + s * nx;
      R[8] = t * nz * nz + c;
      idx++;
    }
  }

  DeviceVector<float> rotations_device(rotations_host);
  DeviceVector<float> twists_out(num_cases * 3);
  DeviceVector<float> rotations_roundtrip(num_cases * 9);

  CudaStream stream;
  ComputeLogSO3(stream.GetStream(), rotations_device.data(), 3, 9, 3, num_cases, twists_out.data());
  // Exp of the logs: SO3StateBatch::Plus from the identity.
  std::vector<float> identity_host(num_cases * 9, 0.f);
  for (size_t i = 0; i < num_cases; i++) {
    identity_host[i * 9] = identity_host[i * 9 + 4] = identity_host[i * 9 + 8] = 1.f;
  }
  DeviceVector<float> identity(identity_host);
  SO3StateBatch states(identity.data(), num_cases);
  states.SetNumActiveStates(num_cases);
  states.Plus(identity.data(), twists_out.data(), rotations_roundtrip.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> twists_host(num_cases * 3);
  twists_out.CopyToHost(twists_host.data(), twists_host.size());
  std::vector<float> rt_host(num_cases * 9);
  rotations_roundtrip.CopyToHost(rt_host.data(), rt_host.size());

  idx = 0;
  for (float angle : angles) {
    for (size_t a = 0; a < sizeof(axes) / sizeof(axes[0]); a++) {
      SCOPED_TRACE("angle=" + std::to_string(angle) + " axis_idx=" + std::to_string(a));
      const float *tw = twists_host.data() + idx * 3;
      float mag = std::sqrt(tw[0] * tw[0] + tw[1] * tw[1] + tw[2] * tw[2]);
      EXPECT_TRUE(std::isfinite(tw[0]));
      EXPECT_TRUE(std::isfinite(tw[1]));
      EXPECT_TRUE(std::isfinite(tw[2]));
      float angle_tol = (angle > 3.0f) ? 0.1f : 5e-3f;
      EXPECT_NEAR(mag, angle, angle_tol);

      const float *R_orig = rotations_host.data() + idx * 9;
      const float *R_rt = rt_host.data() + idx * 9;
      float rt_tol = (angle > 3.0f) ? 0.1f : 5e-3f;
      for (int k = 0; k < 9; k++) {
        EXPECT_NEAR(R_orig[k], R_rt[k], rt_tol);
      }
      idx++;
    }
  }
}

/**
 * @brief Tests LogSO3 on rotation matrices with trace slightly outside [-1,3]
 * due to simulated floating-point drift.
 */
TEST_F(LieMathTest, LogSO3_TraceOutOfRange) {
  // Identity with trace nudged slightly above 3
  std::vector<float> rotations_host = {
      1.0f + 1e-6f,
      0.0f,
      0.0f,
      0.0f,
      1.0f + 1e-6f,
      0.0f,
      0.0f,
      0.0f,
      1.0f + 1e-6f,
      // 180° about x with trace nudged below -1
      1.0f,
      0.0f,
      0.0f,
      0.0f,
      -1.0f - 1e-6f,
      0.0f,
      0.0f,
      0.0f,
      -1.0f - 1e-6f,
  };
  const size_t num_cases = 2;

  DeviceVector<float> rotations_device(rotations_host);
  DeviceVector<float> twists_out(num_cases * 3);

  CudaStream stream;
  ComputeLogSO3(stream.GetStream(), rotations_device.data(), 3, 9, 3, num_cases, twists_out.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> twists_host(num_cases * 3);
  twists_out.CopyToHost(twists_host.data(), twists_host.size());

  for (size_t i = 0; i < num_cases; i++) {
    SCOPED_TRACE("case " + std::to_string(i));
    for (int j = 0; j < 3; j++) {
      EXPECT_TRUE(std::isfinite(twists_host[i * 3 + j])) << "NaN/Inf in twist[" << j << "]";
    }
  }
}

}  // namespace cunls
