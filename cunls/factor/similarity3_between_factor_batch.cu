/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/math/dense_matrix_ops.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

constexpr size_t kSim3TransformStride = 16;
constexpr size_t kSim3TangentStride = 7;
constexpr size_t kSim3TangentDim = 7;
constexpr size_t kSim3AdjointStride = 49;
constexpr size_t kSim3JacobianPitch = 14;
constexpr size_t kSim3JacobianStride = 98;

/**
 * @brief Fused kernel: collect L/R Sim(3), compute (L^{-1} * R) * Delta.
 *
 * Full 4x4 row-major multiply (128 FMAs for 2 matmuls). Inline Sim(3) inverse.
 * Replaces: collect + InverseSim3 + 2x cuBLAS 4x4 GEMM.
 */
__global__ void collect_and_compute_sim3_between_error_kernel(
    float const* const* state_pointers, const Matrix<4>* deltas,
    size_t num_factors, Matrix<4>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  // Load L: 4x4, bottom-right = 1/s
  const float l00=L[0],l01=L[1],l02=L[2],l03=L[3];
  const float l10=L[4],l11=L[5],l12=L[6],l13=L[7];
  const float l20=L[8],l21=L[9],l22=L[10],l23=L[11];
  const float l_inv_s=L[15];
  const float s = 1.0f / l_inv_s;

  // L^{-1}: rotation part = R^T/s^2 (where sR = upper-left 3x3 of L)
  // det(sR) = s^3*det(R) = s^3, so (sR)^{-1} = adj(sR)/s^3
  // For orthogonal R: (sR)^{-1} = R^T/s
  const float inv_s = l_inv_s;
  const float i00=l00*inv_s, i01=l10*inv_s, i02=l20*inv_s;
  const float i10=l01*inv_s, i11=l11*inv_s, i12=l21*inv_s;
  const float i20=l02*inv_s, i21=l12*inv_s, i22=l22*inv_s;
  const float i03 = -(i00*l03 + i01*l13 + i02*l23);
  const float i13 = -(i10*l03 + i11*l13 + i12*l23);
  const float i23 = -(i20*l03 + i21*l13 + i22*l23);
  const float i33 = s;

  // Load R
  const float r0=R[0],r1=R[1],r2=R[2],r3=R[3];
  const float r4=R[4],r5=R[5],r6=R[6],r7=R[7];
  const float r8=R[8],r9=R[9],r10=R[10],r11=R[11];
  const float r12=R[12],r13=R[13],r14=R[14],r15=R[15];

  // temp = L^{-1} * R  (full 4x4)
  const float t00 = i00*r0+i01*r4+i02*r8+i03*r12;
  const float t01 = i00*r1+i01*r5+i02*r9+i03*r13;
  const float t02 = i00*r2+i01*r6+i02*r10+i03*r14;
  const float t03 = i00*r3+i01*r7+i02*r11+i03*r15;
  const float t10 = i10*r0+i11*r4+i12*r8+i13*r12;
  const float t11 = i10*r1+i11*r5+i12*r9+i13*r13;
  const float t12 = i10*r2+i11*r6+i12*r10+i13*r14;
  const float t13 = i10*r3+i11*r7+i12*r11+i13*r15;
  const float t20 = i20*r0+i21*r4+i22*r8+i23*r12;
  const float t21 = i20*r1+i21*r5+i22*r9+i23*r13;
  const float t22 = i20*r2+i21*r6+i22*r10+i23*r14;
  const float t23 = i20*r3+i21*r7+i22*r11+i23*r15;
  const float t30 = i33*r12;
  const float t31 = i33*r13;
  const float t32 = i33*r14;
  const float t33 = i33*r15;

  // error = D * temp  (row-major: D_row * temp_row)
  // cuBLAS col-major convention: C_col = A_col * B_col => C_row = B_row * A_row
  const float d0=D[0],d1=D[1],d2=D[2],d3=D[3];
  const float d4=D[4],d5=D[5],d6=D[6],d7=D[7];
  const float d8=D[8],d9=D[9],d10=D[10],d11=D[11];
  const float d12=D[12],d13=D[13],d14=D[14],d15=D[15];

  out[0]  = d0*t00+d1*t10+d2*t20+d3*t30;
  out[1]  = d0*t01+d1*t11+d2*t21+d3*t31;
  out[2]  = d0*t02+d1*t12+d2*t22+d3*t32;
  out[3]  = d0*t03+d1*t13+d2*t23+d3*t33;
  out[4]  = d4*t00+d5*t10+d6*t20+d7*t30;
  out[5]  = d4*t01+d5*t11+d6*t21+d7*t31;
  out[6]  = d4*t02+d5*t12+d6*t22+d7*t32;
  out[7]  = d4*t03+d5*t13+d6*t23+d7*t33;
  out[8]  = d8*t00+d9*t10+d10*t20+d11*t30;
  out[9]  = d8*t01+d9*t11+d10*t21+d11*t31;
  out[10] = d8*t02+d9*t12+d10*t22+d11*t32;
  out[11] = d8*t03+d9*t13+d10*t23+d11*t33;
  out[12] = d12*t00+d13*t10+d14*t20+d15*t30;
  out[13] = d12*t01+d13*t11+d14*t21+d15*t31;
  out[14] = d12*t02+d13*t12+d14*t22+d15*t32;
  out[15] = d12*t03+d13*t13+d14*t23+d15*t33;
}

// Device function: compute J_r^{-1}(xi) for Sim(3), 7x7, row-major into J[49].
// Duplicates the logic from jacobian_right_inverse_sim3_kernel to allow inlining.
__device__ __forceinline__ void sim3_jr_inv(const float* xi, float* J) {
  float w[3] = {xi[0], xi[1], xi[2]};
  float u[3] = {xi[3], xi[4], xi[5]};
  float lam = xi[6];

  float theta2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];

  if (theta2 + lam*lam < 1e-6f) {
    for (int i = 0; i < 49; ++i) J[i] = 0.0f;
    J[0]=1.f; J[8]=1.f; J[16]=1.f; J[24]=1.f; J[32]=1.f; J[40]=1.f; J[48]=1.f;
    J[1]  += -0.5f*w[2];  J[2]  +=  0.5f*w[1];
    J[7]  +=  0.5f*w[2];  J[9]  += -0.5f*w[0];
    J[14] += -0.5f*w[1];  J[15] +=  0.5f*w[0];
    J[24] += 0.5f*lam;    J[25] += -0.5f*w[2];  J[26] +=  0.5f*w[1];
    J[31] += 0.5f*w[2];   J[32] += 0.5f*lam;    J[33] += -0.5f*w[0];
    J[38] += -0.5f*w[1];  J[39] +=  0.5f*w[0];  J[40] += 0.5f*lam;
    J[21] += -0.5f*u[2];  J[22] +=  0.5f*u[1];
    J[28] +=  0.5f*u[2];  J[30] += -0.5f*u[0];
    J[35] += -0.5f*u[1];  J[36] +=  0.5f*u[0];
    J[27] = -0.5f*u[0];  J[34] = -0.5f*u[1];  J[41] = -0.5f*u[2];
    return;
  }

  float theta = sqrtf(theta2);
  float a0 = 1.0f, a1 = 0.5f, a2;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    a2 = 1.0f / theta2 - (1.0f + ct) / (2.0f * theta * st);
  } else {
    a2 = 1.0f / 12.0f + theta2 / 720.0f;
  }

  // W = skew(w), W^2 = W*W
  float W[9], W2[9];
  W[0]=0; W[1]=-w[2]; W[2]=w[1]; W[3]=w[2]; W[4]=0; W[5]=-w[0]; W[6]=-w[1]; W[7]=w[0]; W[8]=0;
#pragma unroll
  for (int r = 0; r < 3; ++r)
#pragma unroll
    for (int c = 0; c < 3; ++c)
      W2[r*3+c] = W[r*3]*W[c] + W[r*3+1]*W[3+c] + W[r*3+2]*W[6+c];

  // U = skew(u), UW = U*W, UW2 = U*W2
  float U_mat[9], UW[9], UW2[9];
  U_mat[0]=0; U_mat[1]=-u[2]; U_mat[2]=u[1]; U_mat[3]=u[2]; U_mat[4]=0; U_mat[5]=-u[0]; U_mat[6]=-u[1]; U_mat[7]=u[0]; U_mat[8]=0;
#pragma unroll
  for (int r = 0; r < 3; ++r)
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      UW[r*3+c] = U_mat[r*3]*W[c] + U_mat[r*3+1]*W[3+c] + U_mat[r*3+2]*W[6+c];
      UW2[r*3+c] = U_mat[r*3]*W2[c] + U_mat[r*3+1]*W2[3+c] + U_mat[r*3+2]*W2[6+c];
    }

  float wu_cross[3] = {w[1]*u[2]-w[2]*u[1], w[2]*u[0]-w[0]*u[2], w[0]*u[1]-w[1]*u[0]};
  float dot_wu = w[0]*u[0] + w[1]*u[1] + w[2]*u[2];
  float W2u[3] = {w[0]*dot_wu-theta2*u[0], w[1]*dot_wu-theta2*u[1], w[2]*dot_wu-theta2*u[2]};

  constexpr float c1=0.5f, c2=1.f/12.f, c4=-1.f/720.f, c6=1.f/30240.f;
  constexpr float d0v=0.5f, d1v=1.f/12.f, d3v=-1.f/720.f, d5v=1.f/30240.f;

  float ma=lam, mb=1.0f, mc=0.0f;
  float sa=0.0f, sb=0.0f, sc=0.0f;

  float Bn[9], SB[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) { Bn[i] = U_mat[i]; SB[i] = 0.0f; }

  sa += c1*ma; sb += c1*mb; sc += c1*mc;
#pragma unroll
  for (int i = 0; i < 9; ++i) SB[i] += c1*Bn[i];

  { float na=lam*ma, nb2=lam*mb+ma-theta2*mc, nc=lam*mc+mb; ma=na; mb=nb2; mc=nc; }

  // Bn iteration for n=2
  { float WB[9];
#pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
      WB[r*3+c] = W[r*3]*Bn[c] + W[r*3+1]*Bn[3+c] + W[r*3+2]*Bn[6+c];
#pragma unroll
    for (int i=0;i<9;++i) Bn[i] = UW[i] + lam*Bn[i] + WB[i]; }

  sa += c2*ma; sb += c2*mb; sc += c2*mc;
#pragma unroll
  for (int i=0;i<9;++i) SB[i] += c2*Bn[i];

  // n=3
  { float na=lam*ma, nb2=lam*mb+ma-theta2*mc, nc=lam*mc+mb; ma=na; mb=nb2; mc=nc; }
  { float WB[9];
#pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
      WB[r*3+c] = W[r*3]*Bn[c] + W[r*3+1]*Bn[3+c] + W[r*3+2]*Bn[6+c];
#pragma unroll
    for (int i=0;i<9;++i) Bn[i] = UW2[i] + lam*Bn[i] + WB[i]; }

  // n=4
  { float na=lam*ma, nb2=lam*mb+ma-theta2*mc, nc=lam*mc+mb; ma=na; mb=nb2; mc=nc; }
  { float UWn[9], WB[9];
#pragma unroll
    for (int i=0;i<9;++i) UWn[i] = -theta2*UW[i];
#pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
      WB[r*3+c] = W[r*3]*Bn[c] + W[r*3+1]*Bn[3+c] + W[r*3+2]*Bn[6+c];
#pragma unroll
    for (int i=0;i<9;++i) Bn[i] = UWn[i] + lam*Bn[i] + WB[i]; }

  sa += c4*ma; sb += c4*mb; sc += c4*mc;
#pragma unroll
  for (int i=0;i<9;++i) SB[i] += c4*Bn[i];

  // n=5
  { float na=lam*ma, nb2=lam*mb+ma-theta2*mc, nc=lam*mc+mb; ma=na; mb=nb2; mc=nc; }
  { float UWn[9], WB[9];
#pragma unroll
    for (int i=0;i<9;++i) UWn[i] = -theta2*UW2[i];
#pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
      WB[r*3+c] = W[r*3]*Bn[c] + W[r*3+1]*Bn[3+c] + W[r*3+2]*Bn[6+c];
#pragma unroll
    for (int i=0;i<9;++i) Bn[i] = UWn[i] + lam*Bn[i] + WB[i]; }

  // n=6
  { float na=lam*ma, nb2=lam*mb+ma-theta2*mc, nc=lam*mc+mb; ma=na; mb=nb2; mc=nc; }
  { float UWn[9], WB[9]; float t4=theta2*theta2;
#pragma unroll
    for (int i=0;i<9;++i) UWn[i] = t4*U_mat[i];
#pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
      WB[r*3+c] = W[r*3]*Bn[c] + W[r*3+1]*Bn[3+c] + W[r*3+2]*Bn[6+c];
#pragma unroll
    for (int i=0;i<9;++i) Bn[i] = UWn[i] + lam*Bn[i] + WB[i]; }

  sa += c6*ma; sb += c6*mb; sc += c6*mc;
#pragma unroll
  for (int i=0;i<9;++i) SB[i] += c6*Bn[i];

  // Assemble 7x7 output
  for (int i=0;i<49;++i) J[i]=0.0f;

  // Block(0,0): SO(3) J_r^{-1}
#pragma unroll
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
    float val = a1*W[r*3+c] + a2*W2[r*3+c];
    if (r==c) val += a0;
    J[r*7+c] = val;
  }
  // Block(1,1): I + sa + sb*W + sc*W2
#pragma unroll
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
    float val = sb*W[r*3+c] + sc*W2[r*3+c];
    if (r==c) val += 1.0f + sa;
    J[(r+3)*7+(c+3)] = val;
  }
  // Block(1,0): SB
#pragma unroll
  for (int r=0;r<3;++r) for (int c=0;c<3;++c)
    J[(r+3)*7+c] = SB[r*3+c];

  // Block(1,2): column 6, computed from M^n series applied to u
  float ma_s[6], mb_s[6], mc_s[6];
  ma_s[0]=1.f; mb_s[0]=0.f; mc_s[0]=0.f;
  ma_s[1]=lam; mb_s[1]=1.f; mc_s[1]=0.f;
#pragma unroll
  for (int n=1;n<5;++n) {
    ma_s[n+1] = lam*ma_s[n];
    mb_s[n+1] = lam*mb_s[n] + ma_s[n] - theta2*mc_s[n];
    mc_s[n+1] = lam*mc_s[n] + mb_s[n];
  }
  float sd_a = d0v*ma_s[0]+d1v*ma_s[1]+d3v*ma_s[3]+d5v*ma_s[5];
  float sd_b = d0v*mb_s[0]+d1v*mb_s[1]+d3v*mb_s[3]+d5v*mb_s[5];
  float sd_c = d0v*mc_s[0]+d1v*mc_s[1]+d3v*mc_s[3]+d5v*mc_s[5];

  J[3*7+6] = -(sd_a*u[0] + sd_b*wu_cross[0] + sd_c*W2u[0]);
  J[4*7+6] = -(sd_a*u[1] + sd_b*wu_cross[1] + sd_c*W2u[1]);
  J[5*7+6] = -(sd_a*u[2] + sd_b*wu_cross[2] + sd_c*W2u[2]);
  J[48] = 1.0f;
}

// Fused kernel: computes the full 7x14 Jacobian for Sim(3) between factor.
// Left block (cols 0-6):  H_left  = -J_l^{-1}(r) * Ad(Delta) = -J_r^{-1}(-r) * Ad
// Right block (cols 7-13): H_right =  J_r^{-1}(r)
// Eliminates 6 separate kernel launches + cuBLAS GEMM.
__global__ void __launch_bounds__(128, 2)
    sim3_between_fused_jacobians_kernel(
    const float* __restrict__ residuals,
    const float* __restrict__ delta_adjoints,
    int num_factors,
    float* __restrict__ jacobians) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;

  const float* r = residuals + tid * kSim3TangentStride;

  // Compute J_r^{-1}(r) for right block
  float Jr[49];
  sim3_jr_inv(r, Jr);

  // Compute J_r^{-1}(-r) = J_l^{-1}(r) for left block
  float neg_r[7];
#pragma unroll
  for (int i = 0; i < 7; ++i) neg_r[i] = -r[i];
  float Jl[49];
  sim3_jr_inv(neg_r, Jl);

  // Load Ad(Delta) 7x7
  const float* Ad = delta_adjoints + tid * kSim3AdjointStride;

  float* out = jacobians + tid * kSim3JacobianStride;

  // Write both blocks row by row
#pragma unroll
  for (int row = 0; row < 7; ++row) {
    // Left block: -(Jl * Ad)[row][col] = -sum_k Jl[row][k] * Ad[k][col]
#pragma unroll
    for (int col = 0; col < 7; ++col) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 7; ++k) {
        s += Jl[row*7+k] * Ad[k*7+col];
      }
      out[row * kSim3JacobianPitch + col] = -s;
    }
    // Right block: Jr[row][col]
#pragma unroll
    for (int col = 0; col < 7; ++col) {
      out[row * kSim3JacobianPitch + 7 + col] = Jr[row*7+col];
    }
  }
}

Similarity3BetweenFactorBatch::Similarity3BetweenFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<4>* pose_deltas_ptr,
    size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors),
      delta_adjoints_(kSim3AdjointStride * num_factors),
      jacobian_temp_(kSim3AdjointStride * num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void Similarity3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  size_t num_factors = NumFactors();
  ComputeAdjointSim3(stream,
                      reinterpret_cast<const float*>(pose_deltas_ptr_),
                      kSim3TransformStride, delta_adjoints_.data(),
                      kSim3AdjointStride, num_factors);
}

constexpr size_t kSim3FusedBlockSize = 128;

bool Similarity3BetweenFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  collect_and_compute_sim3_between_error_kernel<<<num_blocks, kBlockSize, 0,
                                                  stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  ComputeLogSim3(stream,
                 reinterpret_cast<const float*>(poses_left_inverse_.data()),
                 kSim3TransformStride, kSim3TangentStride, num_factors,
                 residuals);

  if (jacobians != nullptr) {
    size_t fused_blocks = (num_factors + kSim3FusedBlockSize - 1) / kSim3FusedBlockSize;
    sim3_between_fused_jacobians_kernel<<<fused_blocks, kSim3FusedBlockSize, 0,
                                          stream>>>(
        residuals, delta_adjoints_.data(), num_factors, jacobians);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
