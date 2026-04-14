/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

// SL(4) Lie algebra / group operations on GPU.
//
// The Special Linear group SL(4,R) consists of real 4×4 matrices with
// determinant 1.  Its Lie algebra sl(4) is the space of 4×4 traceless
// matrices (15-dimensional).
//
// We use the orthonormal basis:
//
//   sl(4) = so(4) ⊕ sym_off(4) ⊕ diag₀(4)
//
// A tangent vector xi ∈ R^15 is ordered:
//   [r₀₁, r₀₂, r₀₃, r₁₂, r₁₃, r₂₃,      ← skew-symmetric rotations
//    s₀₁, s₀₂, s₀₃, s₁₂, s₁₃, s₂₃,      ← symmetric off-diagonal shears
//    h₁, h₂, h₃]                            ← traceless diagonal scalings
//
// The Hat map (xi → 4×4 matrix) uses scaling factors 1/√2, 1/√6, 1/√12
// to make the basis orthonormal under the Frobenius inner product.
//
// The exponential map uses matrix-level exp with scaling-and-squaring,
// followed by projection to det=1.  The logarithm uses inverse
// scaling-and-squaring with Denman–Beavers matrix square root.

#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/math/sl_lie_math.h"

namespace cunls {

namespace {

// Orthonormal basis scaling constants: 1/√2, 1/√6, 1/√12.
constexpr float kInvSqrt2 = 0.7071067811865475244f;
constexpr float kInvSqrt6 = 0.4082482904638630164f;
constexpr float kInvSqrt12 = 0.2886751345948128823f;
constexpr size_t kBlockSize = 256;

// ============================================================================
// 4×4 Matrix Utilities
// ============================================================================

// C = A * B  (row-major 4×4).
__device__ void Mat4Mul(const float *__restrict__ A,
                        const float *__restrict__ B, float *__restrict__ C) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        s += A[i * 4 + k] * B[k * 4 + j];
      }
      C[i * 4 + j] = s;
    }
  }
}

// B = A  (16 floats).
__device__ void Mat4Copy(const float *A, float *B) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    B[i] = A[i];
  }
}

// Mean absolute value of all 16 entries.  A cheap proxy for the spectral
// radius, used to decide how many times to halve A in scaling-and-squaring.
__device__ float Mat4MeanAbs(const float *A) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    s += fabsf(A[i]);
  }
  return s * 0.0625f; // 1/16
}

// Whether flat index i is on the 4×4 diagonal (row == col).
__device__ __forceinline__ bool IsDiag4(int i) { return (i & 3) == (i >> 2); }

// Frobenius norm of A − I, computed without materializing I in registers.
__device__ float Mat4FrobeniusDiffFromIdentity(const float *A) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float d = A[i] - (IsDiag4(i) ? 1.f : 0.f);
    s += d * d;
  }
  return sqrtf(s);
}

// ============================================================================
// 4×4 Determinant and Inverse
//
// Both use the Laplace expansion by 2×2 sub-determinants ("minors") formed
// from the top two rows {0,1} and the bottom two rows {2,3}.
//
// For each pair of column indices (a,b) with a < b in {0,1,2,3}:
//   s_k = M[0,a]*M[1,b] − M[0,b]*M[1,a]   (top rows, 6 values)
//   c_k = M[2,a]*M[3,b] − M[2,b]*M[3,a]   (bottom rows, 6 values)
//
// where k enumerates the 6 column pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
//
// The determinant is then:
//   det(M) = s0*c5 − s1*c4 + s2*c3 + s3*c2 − s4*c1 + s5*c0
//
// The inverse uses M^{-1} = adj(M)/det(M), where every adjugate entry can
// be expressed purely in terms of the 12 sub-determinants {s_k, c_k} and
// individual matrix elements.  This avoids redundant products: the original
// cofactor expansion recomputes the same 2×2 minors across multiple entries.
// ============================================================================

// det(M) for a row-major 4×4 matrix.  Uses 6 sub-determinants from rows {2,3}
// (c-minors) combined with a cofactor expansion along row 0.
__device__ float Det4(const float *M) {
  // 2×2 sub-determinants from rows 2 and 3.
  float c0 = M[8] * M[13] - M[9] * M[12];   // cols (0,1)
  float c1 = M[8] * M[14] - M[10] * M[12];  // cols (0,2)
  float c2 = M[8] * M[15] - M[11] * M[12];  // cols (0,3)
  float c3 = M[9] * M[14] - M[10] * M[13];  // cols (1,2)
  float c4 = M[9] * M[15] - M[11] * M[13];  // cols (1,3)
  float c5 = M[10] * M[15] - M[11] * M[14]; // cols (2,3)

  // Cofactor expansion along row 0.  Each 3×3 cofactor factors into
  // a dot product of one row-1 element with the c-minors.
  return M[0] * (M[5] * c5 - M[6] * c4 + M[7] * c3) -
         M[1] * (M[4] * c5 - M[6] * c2 + M[7] * c1) +
         M[2] * (M[4] * c4 - M[5] * c2 + M[7] * c0) -
         M[3] * (M[4] * c3 - M[5] * c1 + M[6] * c0);
}

// M^{-1} via adjugate / determinant.  Returns false if M is singular.
//
// Each entry of adj(M) is a signed 3×3 cofactor.  By expressing every
// cofactor through the 12 precomputed 2×2 sub-determinants (s_k from
// rows 0,1 and c_k from rows 2,3), each adjugate entry becomes a 3-term
// dot product — no redundant 2×2 products across the 16 entries.
__device__ bool Inv4(const float *M, float *R) {
  // 2×2 sub-determinants from rows {0,1}: s_k = M[0,a]*M[1,b] − M[0,b]*M[1,a].
  float s0 = M[0] * M[5] - M[1] * M[4];
  float s1 = M[0] * M[6] - M[2] * M[4];
  float s2 = M[0] * M[7] - M[3] * M[4];
  float s3 = M[1] * M[6] - M[2] * M[5];
  float s4 = M[1] * M[7] - M[3] * M[5];
  float s5 = M[2] * M[7] - M[3] * M[6];

  // 2×2 sub-determinants from rows {2,3}: c_k = M[2,a]*M[3,b] − M[2,b]*M[3,a].
  float c0 = M[8] * M[13] - M[9] * M[12];
  float c1 = M[8] * M[14] - M[10] * M[12];
  float c2 = M[8] * M[15] - M[11] * M[12];
  float c3 = M[9] * M[14] - M[10] * M[13];
  float c4 = M[9] * M[15] - M[11] * M[13];
  float c5 = M[10] * M[15] - M[11] * M[14];

  float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  if (fabsf(det) < 1e-7f) {
    return false;
  }
  float d = 1.0f / det;

  // adj(M) = transpose of the cofactor matrix.  Row i of adj uses:
  //   - c-minors with row-1 elements for rows 0,1 of the result
  //   - s-minors with row-{2,3} elements for rows 2,3 of the result
  R[0] = d * (M[5] * c5 - M[6] * c4 + M[7] * c3);
  R[1] = d * (-M[1] * c5 + M[2] * c4 - M[3] * c3);
  R[2] = d * (M[13] * s5 - M[14] * s4 + M[15] * s3);
  R[3] = d * (-M[9] * s5 + M[10] * s4 - M[11] * s3);

  R[4] = d * (-M[4] * c5 + M[6] * c2 - M[7] * c1);
  R[5] = d * (M[0] * c5 - M[2] * c2 + M[3] * c1);
  R[6] = d * (-M[12] * s5 + M[14] * s2 - M[15] * s1);
  R[7] = d * (M[8] * s5 - M[10] * s2 + M[11] * s1);

  R[8] = d * (M[4] * c4 - M[5] * c2 + M[7] * c0);
  R[9] = d * (-M[0] * c4 + M[1] * c2 - M[3] * c0);
  R[10] = d * (M[12] * s4 - M[13] * s2 + M[15] * s0);
  R[11] = d * (-M[8] * s4 + M[9] * s2 - M[11] * s0);

  R[12] = d * (-M[4] * c3 + M[5] * c1 - M[6] * c0);
  R[13] = d * (M[0] * c3 - M[1] * c1 + M[2] * c0);
  R[14] = d * (-M[12] * s3 + M[13] * s1 - M[14] * s0);
  R[15] = d * (M[8] * s3 - M[9] * s1 + M[10] * s0);
  return true;
}

// ============================================================================
// SL(4) Lie Algebra: Hat and Vee Maps
//
// Hat:  R^15 → sl(4)  (tangent vector → traceless 4×4 matrix)
// Vee:  sl(4) → R^15  (inverse of Hat)
//
// For each off-diagonal pair (i,j) with i < j (6 pairs), indexed by k:
//   A[i,j] = (1/√2)(xi[k] + xi[k+6])     rotation + shear
//   A[j,i] = (1/√2)(xi[k+6] − xi[k])     shear − rotation
//
// The diagonal is spanned by 3 traceless orthonormal vectors:
//   H₁ = (1/√2) diag(1,−1, 0, 0)
//   H₂ = (1/√6) diag(1, 1,−2, 0)
//   H₃ = (1/√12)diag(1, 1, 1,−3)
// ============================================================================

// Hat map: xi ∈ R^15 → A ∈ sl(4) ⊂ R^{4×4}.
__device__ void SL4Hat(const float *xi, float *A) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    A[i] = 0.f;
  }

  // Off-diagonal entries.  Pair ordering: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
  // Upper-triangle flat index:  1,  2,  3,  6,  7, 11
  // Lower-triangle flat index:  4,  8, 12,  9, 13, 14
  constexpr int kUpper[] = {1, 2, 3, 6, 7, 11};
  constexpr int kLower[] = {4, 8, 12, 9, 13, 14};
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    float r = kInvSqrt2 * xi[k];
    float s = kInvSqrt2 * xi[k + 6];
    A[kUpper[k]] = r + s;
    A[kLower[k]] = s - r;
  }

  // Traceless diagonal: sum of h_i * H_i basis matrices.
  float h1 = kInvSqrt2 * xi[12];
  float h2 = kInvSqrt6 * xi[13];
  float h3 = kInvSqrt12 * xi[14];
  A[0] = h1 + h2 + h3;
  A[5] = -h1 + h2 + h3;
  A[10] = -2.f * h2 + h3;
  A[15] = -3.f * h3;
}

// Vee map: A ∈ sl(4) → xi ∈ R^15  (inverse of Hat).
//
// From Hat we have:
//   A[i,j] + A[j,i] = (2/√2) xi[k+6]  →  xi[k+6] = (1/√2)(A[i,j] + A[j,i])
//   A[i,j] − A[j,i] = (2/√2) xi[k]    →  xi[k]   = (1/√2)(A[i,j] − A[j,i])
//
// The diagonal components are recovered by inverting the H-basis expansion.
__device__ void SL4Vee(const float *M, float *xi) {
  constexpr int kUpper[] = {1, 2, 3, 6, 7, 11};
  constexpr int kLower[] = {4, 8, 12, 9, 13, 14};
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    float u = M[kUpper[k]];
    float l = M[kLower[k]];
    xi[k] = kInvSqrt2 * (u - l);
    xi[k + 6] = kInvSqrt2 * (u + l);
  }
  // Invert the diagonal basis: H = [h1 h2 h3] * [H₁; H₂; H₃].
  // xi[12] = ⟨H, H₁⟩ = (1/√2)(A[0,0] − A[1,1])
  // xi[13] = ⟨H, H₂⟩ = (1/√6)(A[0,0] + A[1,1] − 2 A[2,2])
  // xi[14] = ⟨H, H₃⟩ = (1/√12)(A[0,0] + A[1,1] + A[2,2] − 3 A[3,3])
  xi[12] = kInvSqrt2 * (M[0] - M[5]);
  xi[13] = kInvSqrt6 * (M[0] + M[5] - 2.f * M[10]);
  xi[14] = kInvSqrt12 * (M[0] + M[5] + M[10] - 3.f * M[15]);
}

// ============================================================================
// Matrix Exponential and Logarithm
//
// exp(A):
//   Scaling: halve A until ‖A‖ < 0.5, counting s halvings.
//   Taylor via Horner:  exp(B) = I + B(I + B/2(I + B/3(… + B/N)))
//   Squaring: square the result s times to recover exp(A) = exp(B)^{2^s}.
//
// log(T):
//   Inverse scaling: repeatedly take matrix square root of T until T ≈ I.
//   Series via Horner:  log(I+X) = X · P(X) where
//     P(X) = 1 − X/2 + X²/3 − …  evaluated in Horner form.
//   Scale: multiply by 2^s to compensate the s square-roots.
//
// The Horner form reduces register pressure compared to maintaining separate
// accumulator and power arrays (48 vs 64 floats of live state).
// ============================================================================

// exp(A) via Taylor with scaling-and-squaring.  Order 20 gives ~7 digits of
// accuracy for ‖A‖ ≤ 0.5, sufficient for float32 work.
__device__ void Mat4Exp(const float *A, float *E) {
  float B[16];
  Mat4Copy(A, B);
  int s = 0;
  float norm = Mat4MeanAbs(B);
  while (norm > 0.5f && s < 32) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      B[i] *= 0.5f;
    }
    norm *= 0.5f;
    ++s;
  }

  // Horner evaluation of I + B + B²/2! + … + B^N/N!
  //   Q = I + B/N   (innermost term)
  //   for k = N−1 down to 1:  Q ← I + (B*Q)/k
  constexpr int kOrder = 20;
  float Q[16];
  float inv_n = 1.f / static_cast<float>(kOrder);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    Q[i] = inv_n * B[i];
  }
  Q[0] += 1.f;
  Q[5] += 1.f;
  Q[10] += 1.f;
  Q[15] += 1.f;

  for (int k = kOrder - 1; k >= 1; --k) {
    float T[16];
    Mat4Mul(B, Q, T);
    float inv_k = 1.f / static_cast<float>(k);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      Q[i] = inv_k * T[i];
    }
    Q[0] += 1.f;
    Q[5] += 1.f;
    Q[10] += 1.f;
    Q[15] += 1.f;
  }

  // Repeated squaring: exp(A) = exp(B)^{2^s}.
  Mat4Copy(Q, E);
  for (int i = 0; i < s; ++i) {
    float T[16];
    Mat4Mul(E, E, T);
    Mat4Copy(T, E);
  }
}

// Matrix square root via Denman–Beavers (product form of Newton's method):
//   X_{k+1} = (X_k + A * X_k^{-1}) / 2
//
// Converges quadratically to √A when A has no eigenvalues on R⁻.
// Terminates early when ‖X_{k+1} − X_k‖_F < ε.
__device__ void Mat4SqrtNewton(const float *A, float *S) {
  float X[16];
  Mat4Copy(A, X);
  float T1[16], T2[16];
  for (int it = 0; it < 32; ++it) {
    if (!Inv4(X, T1)) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        S[i] = IsDiag4(i) ? 1.f : 0.f;
      return;
    }
    Mat4Mul(A, T1, T2); // T2 = A * X^{-1}
    float diff_sq = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float xn = 0.5f * (X[i] + T2[i]);
      float d = X[i] - xn;
      diff_sq += d * d;
      X[i] = xn;
    }
    if (sqrtf(diff_sq) < 1e-6f) {
      Mat4Copy(X, S);
      return;
    }
  }
  Mat4Copy(X, S);
}

// log(T) via inverse scaling-and-squaring.
//
//   1. Repeatedly take √T until T ≈ I (‖T − I‖_F < 0.45), count s roots.
//   2. Let X = T − I.  Evaluate log(I+X) = X · P(X) via Horner where
//        P(X) = 1 − X/2 + X²/3 − … + (−1)^{N−1} X^{N−1}/N
//   3. Multiply by 2^s.
__device__ void Mat4Log(const float *T, float *L) {
  float B[16], P[16];
  Mat4Copy(T, B);
  int s = 0;
  while (Mat4FrobeniusDiffFromIdentity(B) > 0.45f && s < 32) {
    Mat4SqrtNewton(B, P);
    Mat4Copy(P, B);
    ++s;
  }

  // B ← B − I  (reuse B as the deviation from identity)
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (IsDiag4(i))
      B[i] -= 1.f;
  }

  // Horner evaluation of P(B) = sum_{k=0}^{N-1} c_k B^k
  //   where c_k = (−1)^k / (k+1).
  //
  // Start with the innermost coefficient:  P = c_{N-1} I
  // Then for k = N−2 down to 0:  P ← c_k I + B * P
  constexpr int kLogOrder = 24;
  float c = ((kLogOrder - 1) % 2 == 0) ? 1.f / static_cast<float>(kLogOrder)
                                       : -1.f / static_cast<float>(kLogOrder);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    P[i] = IsDiag4(i) ? c : 0.f;
  }

  float W[16];
  for (int k = kLogOrder - 2; k >= 0; --k) {
    Mat4Mul(B, P, W);
    c = (k % 2 == 0) ? 1.f / static_cast<float>(k + 1)
                     : -1.f / static_cast<float>(k + 1);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      P[i] = W[i] + (IsDiag4(i) ? c : 0.f);
    }
  }

  // L = B * P(B), then scale by 2^s.
  Mat4Mul(B, P, L);
  float scale = ldexpf(1.0f, s);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    L[i] *= scale;
  }
}

// ============================================================================
// SL(4) Group Operations
// ============================================================================

// Project a 4×4 matrix to SL(4) by normalizing: T ← T / det(T)^{1/4}.
__device__ void ProjectToSL4(float *M) {
  float d = Det4(M);
  if (!(d > 0.f) || !isfinite(d)) {
    return;
  }
  float sc = powf(d, -0.25f);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    M[i] *= sc;
  }
}

// Nonzero entries of column `col` of the 16×15 orthonormal VEC_TO_ALG matrix.
// Writes up to four (row, value) pairs; returns the number of nonzeros (2–4).
//
// The matrix encodes the 15 orthonormal basis elements of sl(4):
//   cols 0–5:   (E_{ij} − E_{ji})/√2   (skew-symmetric rotations)
//   cols 6–11:  (E_{ij} + E_{ji})/√2   (symmetric off-diagonal shears)
//   cols 12–14: traceless diagonal vectors H₁, H₂, H₃
__device__ int VecToAlgColumnSupport(int col, int *rows, float *vals) {
  constexpr int kPairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  if (col < 6) {
    int p = col;
    int r1 = kPairs[p][0] * 4 + kPairs[p][1];
    int r2 = kPairs[p][1] * 4 + kPairs[p][0];
    rows[0] = r1;
    vals[0] = kInvSqrt2;
    rows[1] = r2;
    vals[1] = -kInvSqrt2;
    return 2;
  }
  if (col < 12) {
    int p = col - 6;
    int r1 = kPairs[p][0] * 4 + kPairs[p][1];
    int r2 = kPairs[p][1] * 4 + kPairs[p][0];
    rows[0] = r1;
    vals[0] = kInvSqrt2;
    rows[1] = r2;
    vals[1] = kInvSqrt2;
    return 2;
  }
  if (col == 12) {
    rows[0] = 0;
    vals[0] = kInvSqrt2;
    rows[1] = 5;
    vals[1] = -kInvSqrt2;
    return 2;
  }
  if (col == 13) {
    rows[0] = 0;
    vals[0] = kInvSqrt6;
    rows[1] = 5;
    vals[1] = kInvSqrt6;
    rows[2] = 10;
    vals[2] = -2.f * kInvSqrt6;
    return 3;
  }
  rows[0] = 0;
  vals[0] = kInvSqrt12;
  rows[1] = 5;
  vals[1] = kInvSqrt12;
  rows[2] = 10;
  vals[2] = kInvSqrt12;
  rows[3] = 15;
  vals[3] = -3.f * kInvSqrt12;
  return 4;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// xi → T = ProjectToSL4(exp(Hat(xi))).
__global__ void ExpSL4Kernel(const float *twist, size_t twist_stride,
                             size_t transform_pitch, size_t transform_stride,
                             size_t size, float *transform) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(size)) {
    return;
  }
  const float *xi = twist + tid * twist_stride;
  float *out = transform + tid * transform_stride;

  float H[16];
  SL4Hat(xi, H);
  float E[16];
  Mat4Exp(H, E);
  ProjectToSL4(E);

#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      out[r * transform_pitch + c] = E[r * 4 + c];
    }
  }
}

// T → xi = Vee(log(T)).
__global__ void LogSL4Kernel(const float *transform, size_t transform_pitch,
                             size_t transform_stride, size_t twist_stride,
                             size_t size, float *twist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(size)) {
    return;
  }
  const float *M = transform + tid * transform_stride;
  float *xi = twist + tid * twist_stride;

  float Mrow[16];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      Mrow[r * 4 + c] = M[r * transform_pitch + c];
    }
  }
  float L[16];
  Mat4Log(Mrow, L);
  SL4Vee(L, xi);
}

// T → T^{-1}.
__global__ void InverseSL4Kernel(const float *transform, size_t transform_pitch,
                                 size_t transform_stride, size_t inverse_pitch,
                                 size_t inverse_stride, size_t size,
                                 float *inverse_transform) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(size)) {
    return;
  }
  const float *M = transform + tid * transform_stride;
  float *Mi = inverse_transform + tid * inverse_stride;

  float Mrow[16];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      Mrow[r * 4 + c] = M[r * transform_pitch + c];
    }
  }
  float I[16];
  if (!Inv4(Mrow, I)) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        Mi[r * inverse_pitch + c] = 0.f;
      }
    }
    return;
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      Mi[r * inverse_pitch + c] = I[r * 4 + c];
    }
  }
}

// T → Ad(T) ∈ R^{15×15}.
//
// Warp-cooperative kernel: one warp (32 lanes) per batch element.
// Shared memory holds the 15 precomputed basis column supports (block-wide),
// plus per-warp copies of T[16] and T^{-1}[16].  The 225 output entries are
// striped across 32 lanes (~7 entries each).
constexpr int kAdjWarpsPerBlock = 8;
constexpr int kAdjBlockSize = kAdjWarpsPerBlock * 32;

__global__ void AdjointSL4Kernel(const float *transform, size_t transform_pitch,
                                 size_t transform_stride, size_t adjoint_pitch,
                                 size_t adjoint_stride, size_t size,
                                 float *adjoint) {
  __shared__ int s_basis_nnz[15];
  __shared__ int s_basis_rows[15][4];
  __shared__ float s_basis_vals[15][4];
  __shared__ float s_T[kAdjWarpsPerBlock][16];
  __shared__ float s_Tinv[kAdjWarpsPerBlock][16];
  __shared__ int s_ok[kAdjWarpsPerBlock];

  if (threadIdx.x < 15) {
    s_basis_nnz[threadIdx.x] = VecToAlgColumnSupport(
        threadIdx.x, s_basis_rows[threadIdx.x], s_basis_vals[threadIdx.x]);
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int local_warp = threadIdx.x >> 5;
  const int warp_id = local_warp + blockIdx.x * kAdjWarpsPerBlock;
  if (warp_id >= static_cast<int>(size))
    return;

  const float *M = transform + warp_id * transform_stride;
  float *Ad = adjoint + warp_id * adjoint_stride;

  if (lane < 16) {
    s_T[local_warp][lane] = M[(lane >> 2) * transform_pitch + (lane & 3)];
  }
  __syncwarp();

  if (lane == 0) {
    s_ok[local_warp] = Inv4(s_T[local_warp], s_Tinv[local_warp]) ? 1 : 0;
  }
  __syncwarp();

  if (!s_ok[local_warp]) {
    for (int idx = lane; idx < 225; idx += 32) {
      int r = idx / 15;
      int c = idx - r * 15;
      Ad[r * adjoint_pitch + c] = (r == c) ? 1.f : 0.f;
    }
    return;
  }

  const float *__restrict__ wT = s_T[local_warp];
  const float *__restrict__ wTi = s_Tinv[local_warp];

  for (int idx = lane; idx < 225; idx += 32) {
    int i = idx / 15;
    int j = idx - i * 15;

    int ni = s_basis_nnz[i];
    int nj = s_basis_nnz[j];
    const int *ri = s_basis_rows[i];
    const float *vi = s_basis_vals[i];
    const int *rj = s_basis_rows[j];
    const float *vj = s_basis_vals[j];

    float s = 0.f;
#pragma unroll 4
    for (int ia = 0; ia < ni; ++ia) {
      int a = ri[ia];
      float va = vi[ia];
      float t_row0 = va * wT[(a >> 2) * 4];
      float t_row1 = va * wT[(a >> 2) * 4 + 1];
      float t_row2 = va * wT[(a >> 2) * 4 + 2];
      float t_row3 = va * wT[(a >> 2) * 4 + 3];
      int mi = a & 3;
#pragma unroll 4
      for (int jb = 0; jb < nj; ++jb) {
        int t = rj[jb];
        int bj = t >> 2;
        float tinv = wTi[(t & 3) * 4 + mi];
        float tij;
        if (bj == 0)
          tij = t_row0;
        else if (bj == 1)
          tij = t_row1;
        else if (bj == 2)
          tij = t_row2;
        else
          tij = t_row3;
        s += tij * tinv * vj[jb];
      }
    }
    Ad[i * adjoint_pitch + j] = s;
  }
}

// Element-wise negation of 15×15 matrices.
__global__ void Negate15x15Kernel(const float *in_mat, size_t pitch,
                                  size_t stride, size_t size, float *out_mat) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(size)) {
    return;
  }
  const float *A = in_mat + tid * stride;
  float *B = out_mat + tid * stride;
#pragma unroll
  for (int r = 0; r < 15; ++r) {
#pragma unroll
    for (int c = 0; c < 15; ++c) {
      B[r * pitch + c] = -A[r * pitch + c];
    }
  }
}

// Fill 15×15 identity matrices.
__global__ void Identity15x15Kernel(size_t size, float *matrices, size_t pitch,
                                    size_t stride) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(size)) {
    return;
  }
  float *M = matrices + tid * stride;
#pragma unroll
  for (int r = 0; r < 15; ++r) {
#pragma unroll
    for (int c = 0; c < 15; ++c) {
      M[r * pitch + c] = (r == c) ? 1.f : 0.f;
    }
  }
}

} // namespace

// ============================================================================
// Host API
// ============================================================================

void ComputeExpSL4(cudaStream_t stream, const float *twist,
                   const size_t twist_stride, const size_t transform_pitch,
                   const size_t transform_stride, size_t size,
                   float *transform) {
  size_t nb = (size + kBlockSize - 1) / kBlockSize;
  ExpSL4Kernel<<<nb, kBlockSize, 0, stream>>>(
      twist, twist_stride, transform_pitch, transform_stride, size, transform);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSL4(cudaStream_t stream, const float *transform,
                   const size_t transform_pitch, const size_t transform_stride,
                   const size_t twist_stride, size_t size, float *twist) {
  size_t nb = (size + kBlockSize - 1) / kBlockSize;
  LogSL4Kernel<<<nb, kBlockSize, 0, stream>>>(
      transform, transform_pitch, transform_stride, twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeInverseSL4(cudaStream_t stream, const float *transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t inverse_pitch, const size_t inverse_stride,
                       size_t size, float *inverse_transform) {
  size_t nb = (size + kBlockSize - 1) / kBlockSize;
  InverseSL4Kernel<<<nb, kBlockSize, 0, stream>>>(
      transform, transform_pitch, transform_stride, inverse_pitch,
      inverse_stride, size, inverse_transform);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeAdjointSL4(cudaStream_t stream, const float *transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t adjoint_pitch, const size_t adjoint_stride,
                       size_t size, float *adjoint) {
  size_t nb = (size + kAdjWarpsPerBlock - 1) / kAdjWarpsPerBlock;
  AdjointSL4Kernel<<<nb, kAdjBlockSize, 0, stream>>>(
      transform, transform_pitch, transform_stride, adjoint_pitch,
      adjoint_stride, size, adjoint);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeNegateMatrix15x15(cudaStream_t stream, const float *matrix,
                              const size_t pitch, const size_t stride,
                              size_t size, float *out) {
  size_t nb = (size + kBlockSize - 1) / kBlockSize;
  Negate15x15Kernel<<<nb, kBlockSize, 0, stream>>>(matrix, pitch, stride, size,
                                                   out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void FillIdentity15x15(cudaStream_t stream, size_t size, float *matrices,
                       const size_t pitch, const size_t stride) {
  size_t nb = (size + kBlockSize - 1) / kBlockSize;
  Identity15x15Kernel<<<nb, kBlockSize, 0, stream>>>(size, matrices, pitch,
                                                     stride);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
