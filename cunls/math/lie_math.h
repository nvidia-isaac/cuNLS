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

namespace cunls {

/**
 * @brief Computes the skew-symmetric matrix from a 3D twist vector for SO(3).
 *
 * For a twist vector [w_x, w_y, w_z], computes the skew-symmetric matrix:
 * [0, -w_z,  w_y]
 * [w_z,  0, -w_x]
 * [-w_y, w_x,  0]
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param skew_pitch Pitch (stride between rows) of the skew matrices
 * @param skew_stride Stride between skew matrices in the batch
 * @param size Number of twist vectors to process
 * @param skew Output skew-symmetric matrices (3x3, device pointer)
 */
void ComputeSkewSO3(cudaStream_t stream, const float* twist, const size_t twist_stride,
             const size_t skew_pitch, const size_t skew_stride, size_t size,
             float* skew);

/**
 * @brief Negates a batch of matrices element-wise.
 *
 * Computes negated_matrix = -matrix for each matrix in the batch.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param matrix Input matrices (device pointer)
 * @param rows Number of rows in each matrix
 * @param cols Number of columns in each matrix
 * @param pitch Pitch (stride between rows) of the matrices
 * @param stride Stride between matrices in the batch
 * @param size Number of matrices to process
 * @param negated_matrix Output negated matrices (device pointer)
 */
void ComputeNegateMatrix(cudaStream_t stream, const float* matrix, size_t rows,
                  size_t cols, const size_t pitch, const size_t stride,
                  size_t size, float* negated_matrix);

/**
 * @brief Computes the inverse of SE(3) transformation matrices.
 *
 * For an SE(3) transform T = [R t; 0 1], computes T^{-1} = [R^T -R^T*t; 0 1].
 *
 * @param stream CUDA stream for asynchronous execution
 * @param transform Input transformation matrices (4x4, device pointer, row-major)
 * @param transform_pitch Pitch (stride between rows) of the transform matrices
 * @param transform_stride Stride between transform matrices in the batch
 * @param inverse_pitch Pitch (stride between rows) of the inverse matrices
 * @param inverse_stride Stride between inverse matrices in the batch
 * @param size Number of transforms to process
 * @param inverse_transform Output inverse transformation matrices (4x4, device pointer)
 */
void ComputeInverseSE3(cudaStream_t stream, const float* transform,
                const size_t transform_pitch, const size_t transform_stride,
                const size_t inverse_pitch, const size_t inverse_stride,
                size_t size, float* inverse_transform);

/**
 * @brief Computes the exponential map for SO(3): R = Exp(phi).
 *
 * Maps a 3D twist vector phi to a rotation matrix R using Rodrigues' formula.
 * This is the exponential map from the Lie algebra so(3) to the Lie group SO(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param rotation_pitch Pitch (stride between rows) of the rotation matrices
 * @param rotation_stride Stride between rotation matrices in the batch
 * @param size Number of twists to process
 * @param rotation Output rotation matrices (3x3, device pointer)
 */
void ComputeExpSO3(cudaStream_t stream, const float* twist, const size_t twist_stride,
            const size_t rotation_pitch, const size_t rotation_stride,
            size_t size, float* rotation);

/**
 * @brief Computes the logarithm map for SO(3): phi = Log(R).
 *
 * Maps a rotation matrix R to a 3D twist vector phi.
 * This is the inverse of the exponential map from SO(3) to so(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param rotation Input rotation matrices (3x3, device pointer)
 * @param rotation_pitch Pitch (stride between rows) of the rotation matrices
 * @param rotation_stride Stride between rotation matrices in the batch
 * @param twist_stride Stride between twist vectors in the batch
 * @param size Number of rotations to process
 * @param twist Output twist vectors (3D, device pointer)
 */
void ComputeLogSO3(cudaStream_t stream, const float* rotation,
            const size_t rotation_pitch, const size_t rotation_stride,
            const size_t twist_stride, size_t size, float* twist);

/**
 * @brief Computes the left Jacobian of SO(3) at a given twist.
 *
 * The left Jacobian J_l(phi) relates left-invariant vector fields on SO(3).
 * It satisfies: Exp(phi + dphi) ≈ Exp(phi) * Exp(J_l(phi)^{-1} * dphi)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the Jacobian matrices
 * @param jacobian_stride Stride between Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output left Jacobian matrices (3x3, device pointer)
 */
void ComputeJacobianLeftSO3(cudaStream_t stream, const float* twist,
                     const size_t twist_stride, const size_t jacobian_pitch,
                     const size_t jacobian_stride, size_t size,
                     float* jacobian);

/**
 * @brief Computes the right Jacobian of SO(3) at a given twist.
 *
 * The right Jacobian J_r(phi) relates right-invariant vector fields on SO(3).
 * It satisfies: Exp(phi + dphi) ≈ Exp(J_r(phi)^{-1} * dphi) * Exp(phi)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the Jacobian matrices
 * @param jacobian_stride Stride between Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output right Jacobian matrices (3x3, device pointer)
 */
void ComputeJacobianRightSO3(cudaStream_t stream, const float* twist,
                      const size_t twist_stride, const size_t jacobian_pitch,
                      const size_t jacobian_stride, size_t size,
                      float* jacobian);

/**
 * @brief Computes the inverse of the left Jacobian of SO(3) at a given twist.
 *
 * Computes J_l(phi)^{-1}, the inverse of the left Jacobian.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_inv_pitch Pitch (stride between rows) of the inverse Jacobian matrices
 * @param jacobian_inv_stride Stride between inverse Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian_inv Output inverse left Jacobian matrices (3x3, device pointer)
 */
void ComputeJacobianLeftInverseSO3(cudaStream_t stream, const float* twist,
                            const size_t twist_stride,
                            const size_t jacobian_inv_pitch,
                            const size_t jacobian_inv_stride, size_t size,
                            float* jacobian_inv);

/**
 * @brief Computes the inverse of the right Jacobian of SO(3) at a given twist.
 *
 * Computes J_r(phi)^{-1}, the inverse of the right Jacobian.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_inv_pitch Pitch (stride between rows) of the inverse Jacobian matrices
 * @param jacobian_inv_stride Stride between inverse Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian_inv Output inverse right Jacobian matrices (3x3, device pointer)
 */
void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float* twist,
                             const size_t twist_stride,
                             const size_t jacobian_inv_pitch,
                             const size_t jacobian_inv_stride, size_t size,
                             float* jacobian_inv);

/**
 * @brief Computes the exponential map for SE(3): T = Exp(xi).
 *
 * Maps a 6D twist vector xi = [phi, rho] (3D rotation + 3D translation) to an
 * SE(3) transformation matrix T. This is the exponential map from se(3) to SE(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (6D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param transform_pitch Pitch (stride between rows) of the transform matrices
 * @param transform_stride Stride between transform matrices in the batch
 * @param size Number of twists to process
 * @param transform Output transformation matrices (4x4, device pointer, row-major)
 */
void ComputeExpSE3(cudaStream_t stream, const float* twist, const size_t twist_stride,
            const size_t transform_pitch, const size_t transform_stride,
            size_t size, float* transform);

/**
 * @brief Computes the logarithm map for SE(3): xi = Log(T).
 *
 * Maps an SE(3) transformation matrix T to a 6D twist vector xi = [phi, rho].
 * This is the inverse of the exponential map from SE(3) to se(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param transform Input transformation matrices (4x4, device pointer, row-major)
 * @param transform_pitch Pitch (stride between rows) of the transform matrices
 * @param transform_stride Stride between transform matrices in the batch
 * @param twist_stride Stride between twist vectors in the batch
 * @param size Number of transforms to process
 * @param twist Output twist vectors (6D, device pointer)
 */
void ComputeLogSE3(cudaStream_t stream, const float* transform,
            const size_t transform_pitch, const size_t transform_stride,
            const size_t twist_stride, size_t size, float* twist);

/**
 * @brief Computes the adjoint representation of SE(3) at a given transform.
 *
 * The adjoint Ad_T maps twists from the body frame to the world frame.
 * For T = [R t; 0 1], Ad_T = [R 0; [t]_× R R] where [t]_× is the skew-symmetric
 * matrix of t.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param transform Input transformation matrices (4x4, device pointer, row-major)
 * @param transform_pitch Pitch (stride between rows) of the transform matrices
 * @param transform_stride Stride between transform matrices in the batch
 * @param adjoint_pitch Pitch (stride between rows) of the adjoint matrices
 * @param adjoint_stride Stride between adjoint matrices in the batch
 * @param size Number of transforms to process
 * @param adjoint Output adjoint matrices (6x6, device pointer)
 */
void ComputeAdjointSE3(cudaStream_t stream, const float* transform,
                const size_t transform_pitch, const size_t transform_stride,
                const size_t adjoint_pitch, const size_t adjoint_stride,
                size_t size, float* adjoint);

/**
 * @brief Computes the inverse adjoint representation of SE(3) at a given transform.
 *
 * Computes Ad_T^{-1}, the inverse of the adjoint representation.
 * This maps twists from the world frame to the body frame.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param transform Input transformation matrices (4x4, device pointer, row-major)
 * @param transform_pitch Pitch (stride between rows) of the transform matrices
 * @param transform_stride Stride between transform matrices in the batch
 * @param inv_adjoint_pitch Pitch (stride between rows) of the inverse adjoint matrices
 * @param inv_adjoint_stride Stride between inverse adjoint matrices in the batch
 * @param size Number of transforms to process
 * @param inv_adjoint Output inverse adjoint matrices (6x6, device pointer)
 */
void ComputeInverseAdjointSE3(cudaStream_t stream, const float* transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t inv_adjoint_pitch,
                       const size_t inv_adjoint_stride, size_t size,
                       float* inv_adjoint);

/**
 * @brief Computes the left Jacobian of SE(3) at a given twist.
 *
 * The left Jacobian J_l(xi) relates left-invariant vector fields on SE(3).
 * It satisfies: Exp(xi + dxi) ≈ Exp(xi) * Exp(J_l(xi)^{-1} * dxi)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (6D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the Jacobian matrices
 * @param jacobian_stride Stride between Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output left Jacobian matrices (6x6, device pointer)
 */
void ComputeJacobianLeftSE3(cudaStream_t stream, const float* twist,
                     const size_t twist_stride, const size_t jacobian_pitch,
                     const size_t jacobian_stride, size_t size,
                     float* jacobian);

/**
 * @brief Computes the inverse of the left Jacobian of SE(3) at a given twist.
 *
 * Computes J_l(xi)^{-1}, the inverse of the left Jacobian of SE(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (6D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the inverse Jacobian matrices
 * @param jacobian_stride Stride between inverse Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output inverse left Jacobian matrices (6x6, device pointer)
 */
void ComputeJacobianLeftInverseSE3(cudaStream_t stream, const float* twist,
                            const size_t twist_stride,
                            const size_t jacobian_pitch,
                            const size_t jacobian_stride, size_t size,
                            float* jacobian);

/**
 * @brief Computes the right Jacobian of SE(3) at a given twist.
 *
 * The right Jacobian J_r(xi) relates right-invariant vector fields on SE(3).
 * It satisfies: Exp(xi + dxi) ≈ Exp(J_r(xi)^{-1} * dxi) * Exp(xi)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (6D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the Jacobian matrices
 * @param jacobian_stride Stride between Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output right Jacobian matrices (6x6, device pointer)
 */
void ComputeJacobianRightSE3(cudaStream_t stream, const float* twist,
                      const size_t twist_stride, const size_t jacobian_pitch,
                      const size_t jacobian_stride, size_t size,
                      float* jacobian);

/**
 * @brief Computes the inverse of the right Jacobian of SE(3) at a given twist.
 *
 * Computes J_r(xi)^{-1}, the inverse of the right Jacobian of SE(3).
 *
 * @param stream CUDA stream for asynchronous execution
 * @param twist Input twist vectors (6D, device pointer)
 * @param twist_stride Stride between twist vectors in the batch
 * @param jacobian_pitch Pitch (stride between rows) of the inverse Jacobian matrices
 * @param jacobian_stride Stride between inverse Jacobian matrices in the batch
 * @param size Number of twists to process
 * @param jacobian Output inverse right Jacobian matrices (6x6, device pointer)
 */
void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float* twist,
                             const size_t twist_stride,
                             const size_t jacobian_pitch,
                             const size_t jacobian_stride, size_t size,
                             float* jacobian);

}  // namespace cunls
