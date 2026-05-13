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

#include <utility>
#include <vector>

#include "cunls/common/cusparse_helper.h"
#include "cunls/linear_solver/csr_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Configuration options for @ref BlockSparsePCGSolver.
 *
 * Two block-Jacobi layouts are supported:
 *  - **Uniform** (default): every diagonal tile is @ref block_size × @ref
 *    block_size.  Requires @c block_size to divide the matrix dimension.
 *  - **Variable** (preferred for SBA-like problems): @ref block_layout
 *    describes a sequence of contiguous segments
 *    `[(count_0, size_0), (count_1, size_1), ...]`, each contributing
 *    `count_i * size_i` rows to the matrix.  When non-empty,
 *    @c block_layout overrides @c block_size.  Use this when the matrix
 *    contains tiles of different sizes (e.g. SBA: 6×6 pose tiles followed
 *    by 3×3 landmark tiles).
 *
 * All other fields are convergence / numerical knobs for the inner
 * Preconditioned Conjugate Gradient loop.  See @ref BlockSparsePCGSolver
 * for the math.
 */
struct BlockSparsePCGOptions {
  /**
   * @brief Uniform diagonal tile size (rows = cols), used when
   *        @ref block_layout is empty.
   *
   * Must divide the matrix dimension.  `block_size = 1` reduces the
   * preconditioner to scalar Jacobi (M = diag(H)^{-1}).
   */
  int block_size = 6;

  /**
   * @brief Heterogeneous block-diagonal layout for the preconditioner.
   *
   * When non-empty, each element `(count_i, size_i)` describes a
   * contiguous segment of `count_i` consecutive diagonal blocks of
   * dimension `size_i × size_i`.  Segments are concatenated in order:
   * segment 0 covers rows `[0, count_0 * size_0)`, segment 1 covers
   * the next `count_1 * size_1` rows, etc.  The sum
   * `Σ count_i * size_i` must equal the matrix dimension.
   *
   * For a Gauss-Newton Hessian, the natural layout follows the order of
   * @ref Problem state batches with the tangent dimension of each batch
   * as the segment block size.  @ref GaussNewtonMinimizer derives this
   * layout automatically from the problem when the active solver is a
   * @ref BlockSparsePCGSolver.
   */
  std::vector<std::pair<int, int>> block_layout;

  /** @brief Maximum number of PCG iterations per @ref Solve call. */
  int max_iterations = 200;

  /**
   * @brief Relative-residual convergence threshold.
   *
   * The iteration stops when `||r_k|| <= relative_tolerance * ||b||`.
   * The squared residual is polled on the GPU every
   * `check_period` iterations to avoid one host sync per step.
   */
  float relative_tolerance = 1e-3f;

  /**
   * @brief Absolute-residual convergence threshold.
   *
   * The iteration also stops when `||r_k|| <= absolute_tolerance`.
   * The effective stopping rule is
   * `||r||^2 <= max(absolute_tolerance^2, relative_tolerance^2 * ||b||^2)`.
   */
  float absolute_tolerance = 1e-30f;

  /**
   * @brief Floor on |D_{kk}| during the per-block LDLT pivoting.
   *
   * Any pivot smaller in magnitude is replaced by @c pivot_floor with
   * the original sign.  Keeps the preconditioner invertible on
   * near-singular diagonal tiles (e.g. an unconstrained landmark);
   * does not affect correctness of the outer iteration because the
   * preconditioner is just a convergence accelerator.
   */
  float pivot_floor = 1e-12f;

  /**
   * @brief Number of PCG iterations between host-side convergence
   *        polls.  Higher values reduce host syncs but may do up to
   *        `check_period - 1` extra iterations after convergence.
   */
  int check_period = 4;
};

/**
 * @brief Block-Jacobi preconditioned conjugate gradient solver for
 *        symmetric positive (semi-)definite CSR systems.
 *
 * Solves `H x = b` with the standard PCG recurrence
 * (Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., §9.2):
 * @code{.unparsed}
 *   r_0 = b - H x_0,  z_0 = M^{-1} r_0,  p_0 = z_0,  rz_0 = <r_0, z_0>
 *   for k = 0, 1, ...
 *     q_k     = H p_k
 *     alpha_k = rz_k / <p_k, q_k>
 *     x_{k+1} = x_k + alpha_k p_k
 *     r_{k+1} = r_k - alpha_k q_k
 *     z_{k+1} = M^{-1} r_{k+1}
 *     rz_{k+1}= <r_{k+1}, z_{k+1}>
 *     beta_k  = rz_{k+1} / rz_k
 *     p_{k+1} = z_{k+1} + beta_k p_k
 * @endcode
 * where `M` is the block-Jacobi preconditioner formed from the dense
 * diagonal tiles of `H`.  Each diagonal tile is factored independently
 * via in-shared-memory LDLT (`H_d = L D L^T`); applying `M^{-1}` is a
 * batched triangular solve plus diagonal scaling.
 *
 * The CG search direction satisfies `<p_i, q_j> = 0` for `i != j`, which
 * makes alpha and beta computable in closed form from the recurrence's
 * own inner products and removes the host's reorientation work that
 * dominates many iterative solver implementations.
 *
 * Implementation notes (see the `.cu` file for derivations):
 *  - SpMV is delegated to cuSPARSE (`cusparseSpMV` with the default
 *    algorithm and an up-front `preprocess` pass) — the same matrix
 *    structure is reused across all PCG steps of one @ref Solve and
 *    typically across multiple @ref Solve calls inside a single
 *    Levenberg-Marquardt @ref Minimize.
 *  - All scalar quantities (`alpha`, `beta`, `<p, q>`, `<r, r>`,
 *    `<r, z>`) live on the device for the whole inner loop.  Only the
 *    residual norm is copied back to the host, and only once every
 *    `check_period` iterations (default 4).
 *  - The block-Jacobi factor `M` is rebuilt from the current `H` values
 *    on every @ref Solve (cheap: one CTA per tile reads a few floats
 *    and does a small LDLT entirely in shared memory).
 */
class BlockSparsePCGSolver : public CSRSparseLinearSolver {
public:
  /**
   * @brief Constructs the solver with the given options.
   *
   * Validates the uniform `block_size` (must be in `[1, 16]`) and the
   * `block_layout` per-segment sizes (same range).  Allocation of
   * device buffers is deferred to @ref Initialize.
   *
   * @param options Convergence and preconditioner-layout knobs.
   */
  explicit BlockSparsePCGSolver(BlockSparsePCGOptions options = {});

  /** @brief Releases owned resources. */
  ~BlockSparsePCGSolver() override;

  /**
   * @brief Prepares working buffers and the cuSPARSE SpMV plan for a
   *        system of the given size.
   *
   * Performs symbolic setup only — no PCG steps run here.  Must be
   * called once before the first @ref Solve.  Subsequent @ref Solve
   * calls may invoke this implicitly when the matrix dimension changes.
   *
   * When @p problem has registered state batches, the block-Jacobi
   * layout is derived automatically from
   * `problem.GetStateBatches()`: each batch contributes one layout
   * segment with `size = TangentSize()` and
   * `count = NumStateBlocks() - NumConstStateBlocks()`.  Consecutive
   * segments of equal size are merged so the dispatch loop only sees
   * distinct-size groups.  An explicit layout previously set via
   * @ref SetBlockLayout takes precedence; passing an empty problem
   * (default-constructed) reverts to the uniform
   * @ref BlockSparsePCGOptions::block_size.
   *
   * @param stream     CUDA stream used for all device work.
   * @param problem    The originating optimization problem.  Used to
   *                   derive the block-Jacobi preconditioner layout
   *                   when @ref BlockSparsePCGOptions::block_layout is
   *                   empty and the caller hasn't explicitly invoked
   *                   @ref SetBlockLayout.
   * @param spd_matrix Coefficient matrix `H` in CSR format.  Only its
   *                   sparsity pattern is examined here; values are read
   *                   on every @ref Solve.
   * @param rhs        Right-hand side vector `b`.  Used only to validate
   *                   dimensions; never read here.
   * @param result     Solution vector `x`.  Used only to validate
   *                   dimensions; never written here.
   * @return true on success, false on dimension mismatch or invalid
   *         layout.
   */
  bool Initialize(cudaStream_t stream, const Problem &problem,
                  const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  /**
   * @brief Runs the PCG loop on `H x = b`, writing into @p result.
   *
   * Recomputes the preconditioner from the current values of `H`, runs
   * up to @ref BlockSparsePCGOptions::max_iterations iterations of PCG,
   * and stops when the residual norm satisfies the tolerance rule
   * documented on @ref BlockSparsePCGOptions::relative_tolerance.
   *
   * @param stream     CUDA stream used for all device work.
   * @param spd_matrix Coefficient matrix `H` in CSR format (same
   *                   sparsity pattern as in @ref Initialize, but
   *                   values are read fresh on every call).
   * @param rhs        Right-hand side `b`.
   * @param result     Output vector `x`.  Caller-allocated.  The
   *                   solver initializes `x_0 = 0` (a zero warm start is
   *                   the right choice for Gauss-Newton's per-step
   *                   reset).
   * @return true on success, false on dimension mismatch.
   */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
             const dvector<float> &rhs, dvector<float> &result) final;

  /**
   * @brief Number of PCG iterations consumed by the most recent
   *        @ref Solve call.  Useful for convergence diagnostics.
   */
  int LastIterations() const { return last_iterations_; }

private:
  // ------------------------------------------------------------------
  // Layout helpers
  // ------------------------------------------------------------------

  /**
   * @brief Populates @ref segment_* device arrays from the current
   *        layout (either @c options_.block_layout or a single
   *        uniform segment derived from @c options_.block_size).
   *
   * Computes per-segment offsets into the factor buffer and the
   * required preconditioner-factor storage size.  Called by
   * @ref Initialize whenever the matrix dimension or layout changes.
   *
   * @param matrix_dim Total matrix size N.
   * @return false if the layout doesn't tile N exactly, true otherwise.
   */
  bool BuildSegmentTables(int matrix_dim);

  BlockSparsePCGOptions options_;
  /** Cached host copy of `options_.block_layout`, normalized to a
   *  single uniform segment when the user didn't supply one. */
  std::vector<std::pair<int, int>> layout_;

  /** Total matrix dimension N (cached from the last @ref Initialize). */
  int matrix_size_ = 0;
  /** Total number of diagonal tiles across all segments. */
  int total_blocks_ = 0;
  /** Total entries in @ref precond_factors_. */
  size_t total_factor_floats_ = 0;

  /** Per-segment data, kept on the host for the launch loop. */
  struct Segment {
    int block_size;      ///< side length of each tile (rows = cols)
    int num_blocks;      ///< number of tiles in this segment
    int row_start;       ///< first matrix row covered by this segment
    int factor_offset;   ///< first index in @ref precond_factors_
    int block_row_start; ///< first block index in the global tile order
  };
  std::vector<Segment> segments_;

  /**
   * @brief LDLT factors of every diagonal tile, packed contiguously.
   *
   * Tile @c b of segment @c s lives at `precond_factors_[segment_factor_offset(s) +
   * b * size_s * size_s ..]` in row-major order.  Lower triangle holds
   * `L` with unit diagonal (implicit); the stored diagonal holds `D`;
   * upper triangle is unused.
   */
  dvector<float> precond_factors_;

  // ------------------------------------------------------------------
  // PCG scratch
  // ------------------------------------------------------------------
  dvector<float> r_;  ///< residual `r_k`
  dvector<float> z_;  ///< preconditioned residual `z_k = M^{-1} r_k`
  dvector<float> p_;  ///< search direction `p_k`
  dvector<float> Ap_; ///< `H p_k` (the SpMV output)

  /** Device-resident scalar slots: alpha, beta, <p,Ap>, rz_old, rz_new,
   *  ||r||^2, ||b||^2.  Layout is fixed in the .cu file. */
  dvector<float> d_scratch_;

  // ------------------------------------------------------------------
  // cuSPARSE SpMV state
  // ------------------------------------------------------------------
  cuSPARSEHandle cusparse_handle_;
  cuSPARSEMatrixDescription mat_desc_;
  dvector<uint8_t> spmv_buffer_; ///< work buffer for cuSPARSE SpMV

  /** Iteration count reported by the last @ref Solve. */
  int last_iterations_ = 0;
};

} // namespace cunls
