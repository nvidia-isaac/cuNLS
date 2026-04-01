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

/** @file lie_math_test.cu
 *  @brief Tests for Lie group math operations (SO3/SE3 exp, log, adjoint, Jacobian, inverse).
 */

#include <gtest/gtest.h>
#include <thrust/device_ptr.h>

#include <cmath>
#include <random>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"

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
__global__ void matmul_batch_kernel(const float* A, const size_t A_pitch,
                                    const size_t A_stride, const float* B,
                                    const size_t B_pitch, const size_t B_stride,
                                    float* C, const size_t C_pitch,
                                    const size_t C_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  const float* A_ptr = A + tid * A_stride;
  const float* B_ptr = B + tid * B_stride;
  float* C_ptr = C + tid * C_stride;

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
 * Sets up test data including random twist vectors for testing ComputeExpSO3/ComputeLogSO3
 * and ComputeExpSE3/ComputeLogSE3 operations.
 */
class LieMathTest : public ::testing::Test {
 public:
  void SetUp() override {
    // SO3 setup
    twists_.resize(num_twists_);
    twists_device_.resize(num_twists_);
    recovered_twists_.resize(num_twists_);

    // SE3 setup
    se3_twists_.resize(num_twists_);
    se3_twists_device_.resize(num_twists_);
    transforms_.resize(num_twists_ * 16);  // 16 floats per 4x4 transform matrix
    transforms_inverse_.resize(
        num_twists_ * 16);  // 16 floats per 4x4 inverse transform matrix
    transforms_products_.resize(num_twists_ *
                                16);  // 16 floats per 4x4 product matrix
    recovered_se3_twists_.resize(num_twists_);

    // 3x3 matrices (reused for SO3 rotations and SO3 jacobians)
    matrices_3x3_a_.resize(num_twists_ * 9);
    matrices_3x3_b_.resize(num_twists_ * 9);
    matrices_3x3_products_.resize(num_twists_ * 9);

    // 6x6 matrices (reused for SE3 adjoints and SE3 jacobians)
    matrices_6x6_a_.resize(num_twists_ * 36);
    matrices_6x6_b_.resize(num_twists_ * 36);
    matrices_6x6_c_.resize(num_twists_ * 36);
    matrices_6x6_products_.resize(num_twists_ * 36);

    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> twist_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> translation_dist(-1.0f, 1.0f);

    // Initialize random SO3 twists (3D vectors)
    for (size_t i = 0; i < num_twists_; i++) {
      Vector<3>& twist = twists_[i];
      twist[0] = twist_dist(rng);  // rotation x
      twist[1] = twist_dist(rng);  // rotation y
      twist[2] = twist_dist(rng);  // rotation z
    }

    // Initialize random SE3 twists (6D vectors: 3 rotation + 3 translation)
    for (size_t i = 0; i < num_twists_; i++) {
      Vector<6>& twist = se3_twists_[i];
      twist[0] = twist_dist(rng);        // rotation x
      twist[1] = twist_dist(rng);        // rotation y
      twist[2] = twist_dist(rng);        // rotation z
      twist[3] = translation_dist(rng);  // translation x
      twist[4] = translation_dist(rng);  // translation y
      twist[5] = translation_dist(rng);  // translation z
    }

    // Copy to device
    twists_device_ = DeviceVector<Vector<3>>(twists_);
    se3_twists_device_ = DeviceVector<Vector<6>>(se3_twists_);
  }

  const size_t num_twists_ = 10000;
  const uint32_t fixed_seed_ = 42;
  // SO3 data
  std::vector<Vector<3>> twists_;
  DeviceVector<Vector<3>> twists_device_;
  DeviceVector<Vector<3>> recovered_twists_;
  // SE3 data
  std::vector<Vector<6>> se3_twists_;
  DeviceVector<Vector<6>> se3_twists_device_;
  DeviceVector<float>
      transforms_;  // 4x4 matrices stored as 16 floats each
  DeviceVector<float>
      transforms_inverse_;  // 4x4 matrices stored as 16 floats each
  DeviceVector<float>
      transforms_products_;  // 4x4 matrices stored as 16 floats each
  DeviceVector<Vector<6>> recovered_se3_twists_;
  // 3x3 matrices (reused for SO3 rotations and SO3 jacobians)
  DeviceVector<float> matrices_3x3_a_;
  DeviceVector<float> matrices_3x3_b_;
  DeviceVector<float> matrices_3x3_products_;
  // 6x6 matrices (reused for SE3 adjoints and SE3 jacobians)
  DeviceVector<float> matrices_6x6_a_;
  DeviceVector<float> matrices_6x6_b_;
  DeviceVector<float> matrices_6x6_c_;
  DeviceVector<float> matrices_6x6_products_;

  profiler::Domain profiler_domain_{"LieMathTest"};
};

/**
 * @brief Tests that ComputeExpSO3 followed by ComputeLogSO3 returns to the original twist.
 *
 * This test verifies the mathematical property that applying ComputeExpSO3 with a
 * twist and then ComputeLogSO3 should result in the original twist (within numerical
 * tolerance). This validates the correctness of both operations and their
 * inverse relationship.
 *
 * Test procedure:
 * 1. Start with random twists (3D vectors)
 * 2. Apply ComputeExpSO3: R = Exp(twist) to get rotation matrices
 * 3. Apply ComputeLogSO3: recovered_twist = Log(R)
 * 4. Verify recovered_twist ≈ original twist
 */
TEST_F(LieMathTest, ExpThenLog) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ExpThenLogTest");

  // Get device pointers
  const float* twists_ptr = reinterpret_cast<const float*>(
      twists_device_.data());
  float* rotations_ptr = reinterpret_cast<float*>(
      matrices_3x3_a_.data());
  float* recovered_twists_ptr = reinterpret_cast<float*>(
      recovered_twists_.data());

  CudaStream stream;

  const size_t twist_stride = 3;
  const size_t rotation_pitch = 3;
  const size_t rotation_stride = 9;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeExpSO3(stream.GetStream(), twists_ptr, twist_stride, rotation_pitch,
           rotation_stride, this->num_twists_, rotations_ptr);
    ComputeLogSO3(stream.GetStream(), rotations_ptr, rotation_pitch, rotation_stride,
           twist_stride, this->num_twists_, recovered_twists_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("ExpSO3");
    // Apply ExpSO3: R = Exp(twist)
    ComputeExpSO3(stream.GetStream(), twists_ptr, twist_stride, rotation_pitch,
           rotation_stride, this->num_twists_, rotations_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that rotations are NOT all identity matrices (ComputeExpSO3 actually
  // changed something)
  {
    std::vector<float> host_rotations(matrices_3x3_a_.size());
    matrices_3x3_a_.CopyToHost(host_rotations.data(), host_rotations.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      const float* rotation = host_rotations.data() + i * 9;
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(rotation[row * 3 + col] - expected) > 1e-5f) {
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
    auto range = this->profiler_domain_.CreateDomainRange("LogSO3");
    // Apply LogSO3: recovered_twist = Log(R)
    // This should result in recovered_twist ≈ original twist
    ComputeLogSO3(stream.GetStream(), rotations_ptr, rotation_pitch, rotation_stride,
           twist_stride, this->num_twists_, recovered_twists_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<Vector<3>> host_recovered_twists(recovered_twists_.size());
  recovered_twists_.CopyToHost(host_recovered_twists.data(), host_recovered_twists.size());

  // Verify that recovered twists match original twists (Exp then Log cancels
  // out)
  for (size_t i = 0; i < this->num_twists_; i++) {
    const Vector<3>& original = twists_[i];
    const Vector<3>& recovered = host_recovered_twists[i];

    for (int j = 0; j < 3; j++) {
      ASSERT_NEAR(original[j], recovered[j], 1e-4f);
    }
  }
}

/**
 * @brief Tests that ComputeExpSE3 followed by ComputeLogSE3 returns to the original twist.
 *
 * This test verifies the mathematical property that applying ComputeExpSE3 with a
 * twist and then ComputeLogSE3 should result in the original twist (within numerical
 * tolerance). This validates the correctness of both operations and their
 * inverse relationship.
 *
 * Test procedure:
 * 1. Start with random twists (6D vectors: 3 rotation + 3 translation)
 * 2. Apply ComputeExpSE3: T = Exp(twist) to get transformation matrices
 * 3. Apply ComputeLogSE3: recovered_twist = Log(T)
 * 4. Verify recovered_twist ≈ original twist
 */
TEST_F(LieMathTest, ExpSE3ThenLogSE3) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("ExpSE3ThenLogSE3Test");

  // Get device pointers
  const float* se3_twists_ptr = reinterpret_cast<const float*>(
      se3_twists_device_.data());
  float* transforms_ptr =
      reinterpret_cast<float*>(transforms_.data());
  float* recovered_se3_twists_ptr = reinterpret_cast<float*>(
      recovered_se3_twists_.data());

  CudaStream stream;

  const size_t twist_stride = 6;
  const size_t transform_pitch = 4;
  const size_t transform_stride = 16;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, twist_stride, transform_pitch,
           transform_stride, this->num_twists_, transforms_ptr);
    ComputeLogSE3(stream.GetStream(), transforms_ptr, transform_pitch,
           transform_stride, twist_stride, this->num_twists_,
           recovered_se3_twists_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("ExpSE3");
    // Apply ExpSE3: T = Exp(twist)
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, twist_stride, transform_pitch,
           transform_stride, this->num_twists_, transforms_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that transforms are NOT all identity matrices (ComputeExpSE3 actually
  // changed something)
  {
    std::vector<float> host_transforms(transforms_.size());
    transforms_.CopyToHost(host_transforms.data(), host_transforms.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      const float* transform = host_transforms.data() + i * 16;
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(transform[row * 4 + col] - expected) > 1e-5f) {
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
    auto range = this->profiler_domain_.CreateDomainRange("LogSE3");
    // Apply LogSE3: recovered_twist = Log(T)
    // This should result in recovered_twist ≈ original twist
    ComputeLogSE3(stream.GetStream(), transforms_ptr, transform_pitch,
           transform_stride, twist_stride, this->num_twists_,
           recovered_se3_twists_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<Vector<6>> host_recovered_se3_twists(recovered_se3_twists_.size());
  recovered_se3_twists_.CopyToHost(host_recovered_se3_twists.data(), host_recovered_se3_twists.size());

  // Verify that recovered twists match original twists (Exp then Log cancels
  // out)
  for (size_t i = 0; i < this->num_twists_; i++) {
    const Vector<6>& original = se3_twists_[i];
    const Vector<6>& recovered = host_recovered_se3_twists[i];

    for (int j = 0; j < 6; j++) {
      ASSERT_NEAR(original[j], recovered[j], 1e-4f);
    }
  }
}

/**
 * @brief Tests that ComputeJacobianLeftInverseSO3 after ComputeJacobianLeftSO3 results in
 * identity matrix.
 *
 * This test verifies the mathematical property that multiplying the left
 * Jacobian of SO(3) with its inverse should result in the identity matrix
 * (within numerical tolerance). This validates the correctness of both
 * operations and their inverse relationship.
 *
 * Test procedure:
 * 1. Start with random twists (3D vectors)
 * 2. Apply ComputeJacobianLeftSO3: J = J_l(twist) to get Jacobian matrices
 * 3. Apply ComputeJacobianLeftInverseSO3: J_inv = J_l^{-1}(twist) to get inverse
 *    Jacobian matrices
 * 4. Multiply J * J_inv
 * 5. Verify result ≈ identity matrix
 */
TEST_F(LieMathTest, JacobianLeftSO3ThenInverse) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "JacobianLeftSO3ThenInverseTest");

  // Get device pointers
  const float* twists_ptr = reinterpret_cast<const float*>(
      twists_device_.data());
  float* jacobians_left_ptr = reinterpret_cast<float*>(
      matrices_3x3_a_.data());
  float* jacobians_left_inverse_ptr = reinterpret_cast<float*>(
      matrices_3x3_b_.data());
  float* jacobian_products_ptr = reinterpret_cast<float*>(
      matrices_3x3_products_.data());

  CudaStream stream;

  const size_t twist_stride = 3;
  const size_t jacobian_pitch = 3;
  const size_t jacobian_stride = 9;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeJacobianLeftSO3(stream.GetStream(), twists_ptr, twist_stride,
                    jacobian_pitch, jacobian_stride, this->num_twists_,
                    jacobians_left_ptr);
    ComputeJacobianLeftInverseSO3(stream.GetStream(), twists_ptr, twist_stride,
                           jacobian_pitch, jacobian_stride, this->num_twists_,
                           jacobians_left_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("JacobianLeftSO3");
    // Apply JacobianLeftSO3: J = J_l(twist)
    ComputeJacobianLeftSO3(stream.GetStream(), twists_ptr, twist_stride,
                    jacobian_pitch, jacobian_stride, this->num_twists_,
                    jacobians_left_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that jacobians are NOT all identity matrices (ComputeJacobianLeftSO3
  // actually changed something)
  {
    std::vector<float> host_jacobians_left(matrices_3x3_a_.size());
    matrices_3x3_a_.CopyToHost(host_jacobians_left.data(), host_jacobians_left.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(host_jacobians_left[i* 9 + row * 3 + col] - expected) > 1e-5f) {
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
    auto range =
        this->profiler_domain_.CreateDomainRange("JacobianLeftInverseSO3");
    // Apply JacobianLeftInverseSO3: J_inv = J_l^{-1}(twist)
    ComputeJacobianLeftInverseSO3(stream.GetStream(), twists_ptr, twist_stride,
                           jacobian_pitch, jacobian_stride, this->num_twists_,
                           jacobians_left_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that inverse jacobians are NOT all identity matrices
  // (ComputeJacobianLeftInverseSO3 actually changed something)
  {
    std::vector<float> host_jacobians_left_inverse(matrices_3x3_b_.size());
    matrices_3x3_b_.CopyToHost(host_jacobians_left_inverse.data(), host_jacobians_left_inverse.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(host_jacobians_left_inverse[i* 9 + row * 3 + col] - expected) >
              1e-5f) {
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
    auto range = this->profiler_domain_.CreateDomainRange("MatrixMultiply");
    // Multiply J * J_inv
    constexpr size_t block_size = 256;
    size_t num_blocks = (this->num_twists_ + block_size - 1) / block_size;
    matmul_batch_kernel<3><<<num_blocks, block_size, 0, stream.GetStream()>>>(
        jacobians_left_ptr, jacobian_pitch,
        jacobian_stride,  // A: pitch=3, stride=9
        jacobians_left_inverse_ptr, jacobian_pitch,
        jacobian_stride,  // B: pitch=3, stride=9
        jacobian_products_ptr, jacobian_pitch,
        jacobian_stride,  // C: pitch=3, stride=9
        this->num_twists_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<float> host_jacobian_products(matrices_3x3_products_.size());
  matrices_3x3_products_.CopyToHost(host_jacobian_products.data(), host_jacobian_products.size());

  // Verify that products are identity matrices (J * J_inv = I)
  for (size_t i = 0; i < this->num_twists_; i++) {
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        ASSERT_NEAR(host_jacobian_products[i* 9 + row * 3 + col],
                    (row == col) ? 1.0f : 0.0f, 1e-4f);
      }
    }
  }
}

/**
 * @brief Tests that ComputeInverseAdjointSE3 after ComputeAdjointSE3 results in identity
 * matrix.
 *
 * This test verifies the mathematical property that multiplying the adjoint of
 * SE(3) with its inverse should result in the identity matrix (within numerical
 * tolerance). This validates the correctness of both operations and their
 * inverse relationship.
 *
 * Test procedure:
 * 1. Start with random SE3 twists (6D vectors)
 * 2. Apply ComputeExpSE3: T = Exp(twist) to get transformation matrices
 * 3. Apply ComputeAdjointSE3: Adj = Adjoint(T) to get adjoint matrices
 * 4. Apply ComputeInverseAdjointSE3: Adj_inv = Adjoint^{-1}(T) to get inverse
 *    adjoint matrices
 * 5. Multiply Adj * Adj_inv
 * 6. Verify result ≈ identity matrix
 */
TEST_F(LieMathTest, AdjointSE3ThenInverse) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("AdjointSE3ThenInverseTest");

  // First compute SE3 transforms from twists
  const float* se3_twists_ptr = reinterpret_cast<const float*>(
      se3_twists_device_.data());
  float* transforms_ptr =
      reinterpret_cast<float*>(transforms_.data());

  // Get device pointers for adjoints (reusing 6x6 matrices)
  float* adjoints_ptr = reinterpret_cast<float*>(
      matrices_6x6_a_.data());
  float* adjoints_inverse_ptr = reinterpret_cast<float*>(
      matrices_6x6_b_.data());
  float* adjoint_products_ptr = reinterpret_cast<float*>(
      matrices_6x6_products_.data());

  CudaStream stream;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, 6,  // twist_stride = 6
           4,                                      // transform_pitch = 4
           16,                                     // transform_stride = 16
           this->num_twists_, transforms_ptr);
    ComputeAdjointSE3(stream.GetStream(), transforms_ptr, 4,  // transform_pitch = 4
               16,                                     // transform_stride = 16
               6,                                      // adjoint_pitch = 6
               36,                                     // adjoint_stride = 36
               this->num_twists_, adjoints_ptr);
    ComputeInverseAdjointSE3(stream.GetStream(), transforms_ptr,
                      4,   // transform_pitch = 4
                      16,  // transform_stride = 16
                      6,   // inv_adjoint_pitch = 6
                      36,  // inv_adjoint_stride = 36
                      this->num_twists_, adjoints_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("ExpSE3");
    // Compute SE3 transforms: T = Exp(twist)
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, 6,  // twist_stride = 6
           4,                                      // transform_pitch = 4
           16,                                     // transform_stride = 16
           this->num_twists_, transforms_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("AdjointSE3");
    // Apply AdjointSE3: Adj = Adjoint(T)
    ComputeAdjointSE3(stream.GetStream(), transforms_ptr, 4,  // transform_pitch = 4
               16,                                     // transform_stride = 16
               6,                                      // adjoint_pitch = 6
               36,                                     // adjoint_stride = 36
               this->num_twists_, adjoints_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that adjoints are NOT all identity matrices (ComputeAdjointSE3 actually
  // changed something)
  {
    std::vector<float> host_adjoints(matrices_6x6_a_.size());
    matrices_6x6_a_.CopyToHost(host_adjoints.data(), host_adjoints.size());
    bool found_non_identity = false;
    for (size_t i = 0; i < this->num_twists_ && !found_non_identity; i++) {
      const float* adjoint = host_adjoints.data() + i * 36;
      for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 6; col++) {
          float expected = (row == col) ? 1.0f : 0.0f;
          if (std::abs(adjoint[row * 6 + col] - expected) > 1e-5f) {
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
    auto range = this->profiler_domain_.CreateDomainRange("InverseAdjointSE3");
    // Apply InverseAdjointSE3: Adj_inv = Adjoint^{-1}(T)
    ComputeInverseAdjointSE3(stream.GetStream(), transforms_ptr,
                      4,   // transform_pitch = 4
                      16,  // transform_stride = 16
                      6,   // inv_adjoint_pitch = 6
                      36,  // inv_adjoint_stride = 36
                      this->num_twists_, adjoints_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("MatrixMultiply");
    // Multiply Adj * Adj_inv
    constexpr size_t block_size = 256;
    size_t num_blocks = (this->num_twists_ + block_size - 1) / block_size;
    matmul_batch_kernel<6><<<num_blocks, block_size, 0, stream.GetStream()>>>(
        adjoints_ptr, 6, 36,          // A: pitch=6, stride=36
        adjoints_inverse_ptr, 6, 36,  // B: pitch=6, stride=36
        adjoint_products_ptr, 6, 36,  // C: pitch=6, stride=36
        this->num_twists_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<float> host_adjoint_products(matrices_6x6_products_.size());
  matrices_6x6_products_.CopyToHost(host_adjoint_products.data(), host_adjoint_products.size());

  // Verify that products are identity matrices (Adj * Adj_inv = I)
  for (size_t i = 0; i < this->num_twists_; i++) {
    const float* product = host_adjoint_products.data() + i * 36;

    for (int row = 0; row < 6; row++) {
      for (int col = 0; col < 6; col++) {
        float expected = (row == col) ? 1.0f : 0.0f;
        float actual = product[row * 6 + col];
        ASSERT_NEAR(expected, actual, 1e-4f);
      }
    }
  }
}

/**
 * @brief Tests that ComputeJacobianLeftInverseSE3 after ComputeJacobianLeftSE3 results in
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
 * 3. Apply ComputeJacobianLeftInverseSE3: J_inv = J_l^{-1}(twist) to get inverse
 *    Jacobian matrices
 * 4. Multiply J * J_inv
 * 5. Verify result ≈ identity matrix
 */
TEST_F(LieMathTest, JacobianLeftSE3ThenInverse) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "JacobianLeftSE3ThenInverseTest");

  // Get device pointers (reusing 6x6 matrices)
  const float* se3_twists_ptr = reinterpret_cast<const float*>(
      se3_twists_device_.data());
  float* se3_jacobians_left_ptr = reinterpret_cast<float*>(
      matrices_6x6_a_.data());
  float* se3_jacobians_left_inverse_ptr = reinterpret_cast<float*>(
      matrices_6x6_c_.data());
  float* se3_jacobian_products_ptr = reinterpret_cast<float*>(
      matrices_6x6_products_.data());

  CudaStream stream;

  const size_t twist_stride = 6;
  const size_t jacobian_pitch = 6;
  const size_t jacobian_stride = 36;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeJacobianLeftSE3(stream.GetStream(), se3_twists_ptr, twist_stride,
                    jacobian_pitch, jacobian_stride, this->num_twists_,
                    se3_jacobians_left_ptr);
    ComputeJacobianLeftInverseSE3(stream.GetStream(), se3_twists_ptr, twist_stride,
                           jacobian_pitch, jacobian_stride, this->num_twists_,
                           se3_jacobians_left_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("JacobianLeftSE3");
    // Apply JacobianLeftSE3: J = J_l(twist)
    ComputeJacobianLeftSE3(stream.GetStream(), se3_twists_ptr, 6,  // twist_stride = 6
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
          if (std::abs(host_se3_jacobians_left[i * 36 + row * 6 + col] - expected) >
              1e-5f) {
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
    auto range =
        this->profiler_domain_.CreateDomainRange("JacobianLeftInverseSE3");
    // Apply JacobianLeftInverseSE3: J_inv = J_l^{-1}(twist)
    ComputeJacobianLeftInverseSE3(stream.GetStream(), se3_twists_ptr, twist_stride,
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
  matrices_6x6_products_.CopyToHost(host_se3_jacobian_products.data(), host_se3_jacobian_products.size());

  // Verify that products are identity matrices (J * J_inv = I)
  for (size_t i = 0; i < this->num_twists_; i++) {
    for (int row = 0; row < 6; row++) {
      for (int col = 0; col < 6; col++) {
        ASSERT_NEAR(host_se3_jacobian_products[i * 36 + row * 6 + col],
                    (row == col) ? 1.0f : 0.0f, 1e-4f);
      }
    }
  }
}

/**
 * @brief Tests that ComputeInverseSE3 returns the inverse of a transform.
 *
 * This test verifies the mathematical property that multiplying an SE(3)
 * transformation matrix with its inverse should result in the identity matrix
 * (within numerical tolerance). This validates the correctness of the
 * ComputeInverseSE3 operation.
 *
 * Test procedure:
 * 1. Start with random SE3 twists (6D vectors: 3 rotation + 3 translation)
 * 2. Apply ComputeExpSE3: T = Exp(twist) to get transformation matrices
 * 3. Apply ComputeInverseSE3: T_inv = T^{-1} to get inverse transformation matrices
 * 4. Multiply T * T_inv
 * 5. Verify result ≈ identity matrix
 */
TEST_F(LieMathTest, InverseSE3) {
  auto test_range = this->profiler_domain_.CreateDomainRange("InverseSE3Test");

  // First compute SE3 transforms from twists
  const float* se3_twists_ptr = reinterpret_cast<const float*>(
      se3_twists_device_.data());
  float* transforms_ptr =
      reinterpret_cast<float*>(transforms_.data());
  float* transforms_inverse_ptr = reinterpret_cast<float*>(
      transforms_inverse_.data());
  float* transforms_products_ptr = reinterpret_cast<float*>(
      transforms_products_.data());

  CudaStream stream;

  const size_t transform_pitch = 4;
  const size_t transform_stride = 16;

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, 6, transform_pitch,
           transform_stride, this->num_twists_, transforms_ptr);
    ComputeInverseSE3(stream.GetStream(), transforms_ptr, transform_pitch,
               transform_stride, transform_pitch, transform_stride,
               this->num_twists_, transforms_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("ExpSE3");
    // Compute SE3 transforms: T = Exp(twist)
    ComputeExpSE3(stream.GetStream(), se3_twists_ptr, 6, transform_pitch,
           transform_stride, this->num_twists_, transforms_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("InverseSE3");
    // Apply InverseSE3: T_inv = T^{-1}
    ComputeInverseSE3(stream.GetStream(), transforms_ptr, transform_pitch,
               transform_stride, transform_pitch, transform_stride,
               this->num_twists_, transforms_inverse_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("MatrixMultiply");
    // Multiply T * T_inv
    constexpr size_t block_size = 256;
    size_t num_blocks = (this->num_twists_ + block_size - 1) / block_size;
    matmul_batch_kernel<4><<<num_blocks, block_size, 0, stream.GetStream()>>>(
        transforms_ptr, transform_pitch,
        transform_stride,  // A: pitch=4, stride=16
        transforms_inverse_ptr, transform_pitch,
        transform_stride,  // B: pitch=4, stride=16
        transforms_products_ptr, transform_pitch,
        transform_stride,  // C: pitch=4, stride=16
        this->num_twists_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for verification
  std::vector<float> host_transforms_products(transforms_products_.size());
  transforms_products_.CopyToHost(host_transforms_products.data(), host_transforms_products.size());

  // Verify that products are identity matrices (T * T_inv = I)
  for (size_t i = 0; i < this->num_twists_; i++) {
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        ASSERT_NEAR((row == col) ? 1.0f : 0.0f,
                    host_transforms_products[i * 16 + row * 4 + col], 1e-4f);
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
  const float axes[][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                           {0.5774f, 0.5774f, 0.5774f}};
  const size_t num_cases = sizeof(angles) / sizeof(angles[0]) *
                           sizeof(axes) / sizeof(axes[0]);

  std::vector<float> rotations_host(num_cases * 9);
  size_t idx = 0;
  for (float angle : angles) {
    for (const auto& ax : axes) {
      float nx = ax[0], ny = ax[1], nz = ax[2];
      float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len; ny /= len; nz /= len;
      float c = std::cos(angle), s = std::sin(angle);
      float t = 1.0f - c;
      float* R = rotations_host.data() + idx * 9;
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
  ComputeLogSO3(stream.GetStream(),
                rotations_device.data(), 3, 9, 3, num_cases,
                twists_out.data());
  ComputeExpSO3(stream.GetStream(),
                twists_out.data(), 3, 3, 9, num_cases,
                rotations_roundtrip.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> twists_host(num_cases * 3);
  twists_out.CopyToHost(twists_host.data(), twists_host.size());
  std::vector<float> rt_host(num_cases * 9);
  rotations_roundtrip.CopyToHost(rt_host.data(), rt_host.size());

  idx = 0;
  for (float angle : angles) {
    for (size_t a = 0; a < sizeof(axes) / sizeof(axes[0]); a++) {
      SCOPED_TRACE("angle=" + std::to_string(angle) +
                   " axis_idx=" + std::to_string(a));
      const float* tw = twists_host.data() + idx * 3;
      float mag = std::sqrt(tw[0] * tw[0] + tw[1] * tw[1] + tw[2] * tw[2]);
      EXPECT_TRUE(std::isfinite(tw[0]));
      EXPECT_TRUE(std::isfinite(tw[1]));
      EXPECT_TRUE(std::isfinite(tw[2]));
      float angle_tol = (angle > 3.0f) ? 0.1f : 5e-3f;
      EXPECT_NEAR(mag, angle, angle_tol);

      const float* R_orig = rotations_host.data() + idx * 9;
      const float* R_rt = rt_host.data() + idx * 9;
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
      1.0f + 1e-6f, 0.0f, 0.0f,
      0.0f, 1.0f + 1e-6f, 0.0f,
      0.0f, 0.0f, 1.0f + 1e-6f,
      // 180° about x with trace nudged below -1
      1.0f, 0.0f, 0.0f,
      0.0f, -1.0f - 1e-6f, 0.0f,
      0.0f, 0.0f, -1.0f - 1e-6f,
  };
  const size_t num_cases = 2;

  DeviceVector<float> rotations_device(rotations_host);
  DeviceVector<float> twists_out(num_cases * 3);

  CudaStream stream;
  ComputeLogSO3(stream.GetStream(),
                rotations_device.data(), 3, 9, 3, num_cases,
                twists_out.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> twists_host(num_cases * 3);
  twists_out.CopyToHost(twists_host.data(), twists_host.size());

  for (size_t i = 0; i < num_cases; i++) {
    SCOPED_TRACE("case " + std::to_string(i));
    for (int j = 0; j < 3; j++) {
      EXPECT_TRUE(std::isfinite(twists_host[i * 3 + j]))
          << "NaN/Inf in twist[" << j << "]";
    }
  }
}

}  // namespace cunls
