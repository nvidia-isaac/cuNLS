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

/**
 * @file block_hessian_assembler_test.cpp
 * @brief Correctness and performance tests for Hessian assembly and storage.
 *
 * Three suites:
 *  - HessianStructureTest checks the GPU-derived sparsity pattern against a
 *    naive CPU oracle,
 *  - BlockHessianAssemblerTest checks the assembled H and rhs against a CPU
 *    oracle that contracts the same per-factor Jacobians,
 *  - HessianStorageTest checks scalar CSR and block BSR against each other.
 *
 * The oracles share no code with the GPU paths they validate, which matters:
 * both storage layouts run through one assembler, so comparing them against
 * each other cannot catch an error common to both.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/factor/vector_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/bsr_matrix.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/bsr_expansion.h"
#include "tests/utils.h"

namespace cunls {
namespace {

// ============================================================================
// Harness
// ============================================================================

/**
 * @brief Exposes one BuildSystem call and its outputs.
 *
 * BuildSystem is protected on GaussNewtonMinimizer, and hessian_/lhs_work_
 * live alongside it, so a thin subclass is the least invasive way to compare
 * the two assembly paths on identical input.
 */
class SystemBuilder : public GaussNewtonMinimizer {
 public:
  /**
   * @brief Builds with block storage (the default for a block-capable solver).
   */
  SystemBuilder() : GaussNewtonMinimizer(MakeOptions(SparseLinearSolverType::BlockSparsePCG)) {}

  explicit SystemBuilder(const MinimizerOptions &options) : GaussNewtonMinimizer(options) {}

  /**
   * @brief Builds with scalar CSR storage.
   *
   * There is no switch for this: storage is chosen from the problem's tangent
   * dimensions and the solver's capability.  Selecting a CSR-only backend is
   * how a caller actually ends up on the scalar path, so that is what the
   * comparisons here exercise.  Only assembly is compared, never the solve, so
   * the backend choice does not otherwise affect the result.
   */
  static SystemBuilder Scalar() {
    return SystemBuilder(MakeOptions(SparseLinearSolverType::cuDSS));
  }

  /** @brief Runs Initialize + one BuildSystem and syncs. */
  void Build(cudaStream_t stream, Problem &problem) {
    Initialize(stream, problem);
    current_state_.Recreate(stream, problem);
    BuildSystem(stream, problem, current_state_);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  const CSRSparseMatrix &HessianAsCSR(cudaStream_t stream) {
    if (!normal_equations_.UsesBlockStorage()) {
      return normal_equations_.LhsCSR();
    }
    test_utils::ExpandBSRToCSR(stream, normal_equations_.LhsBSR(), csr_mirror_, expand_scratch_);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    return csr_mirror_;
  }

  const dvector<float> &Rhs() const { return rhs_work_; }
  const dvector<float> &Residuals() const { return residuals_; }
  const PerFactorJacobians &FactorJacobians() const { return factor_jacobians_; }
  /** @brief step^T H step against the undamped Hessian, as LM computes it. */
  float WeightedSquaredStep(cudaStream_t stream, const dvector<float> &step) {
    d_scalars_.resize(1);
    d_reduce_partials_.resize(ReducePartialCount(step.size()));
    normal_equations_.WeightedSquaredStepAsync(stream, cusparse_handle_.GetHandle(stream), step,
                                               d_scalars_.data(), d_reduce_partials_.data(),
                                               buffer_);
    float out = 0.f;
    THROW_ON_CUDA_ERROR(
        cudaMemcpyAsync(&out, d_scalars_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    return out;
  }
  NormalEquations &Equations() { return normal_equations_; }
  bool UsesBlockStorage() const { return normal_equations_.UsesBlockStorage(); }

 private:
  static MinimizerOptions MakeOptions(SparseLinearSolverType solver) {
    MinimizerOptions options;
    options.sparse_linear_solver_type = solver;
    return options;
  }

  CSRSparseMatrix csr_mirror_;
  dvector<int> expand_scratch_;
};

/** @brief Host-side snapshot of an assembled system. */
struct SystemSnapshot {
  std::vector<int> row_offsets;
  std::vector<int> col_ids;
  std::vector<float> values;
  std::vector<float> rhs;
};

SystemSnapshot Snapshot(SystemBuilder &builder, cudaStream_t stream) {
  const CSRSparseMatrix &h = builder.HessianAsCSR(stream);
  SystemSnapshot out;
  out.row_offsets.resize(h.row_offsets.size());
  out.col_ids.resize(h.col_ids.size());
  out.values.resize(h.values.size());
  out.rhs.resize(builder.Rhs().size());
  h.row_offsets.CopyToHost(out.row_offsets.data(), out.row_offsets.size());
  h.col_ids.CopyToHost(out.col_ids.data(), out.col_ids.size());
  h.values.CopyToHost(out.values.data(), out.values.size());
  builder.Rhs().CopyToHost(out.rhs.data(), out.rhs.size());
  return out;
}

/** @brief Largest magnitude in a vector, used to set a relative tolerance. */
float MaxAbs(const std::vector<float> &v) {
  float m = 0.f;
  for (float x : v) {
    m = std::max(m, std::fabs(x));
  }
  return m;
}

// ============================================================================
// Problem builders
// ============================================================================

/** @brief Owns the device data behind a Problem for the lifetime of a test. */
struct VectorChainProblem {
  static constexpr int kDim = 4;

  std::vector<float> host_states;
  std::vector<Vector<kDim>> host_priors;
  dvector<float> states;
  dvector<Vector<kDim>> deltas;
  dvector<Vector<kDim>> priors;
  dvector<int> const_ids;
  std::unique_ptr<VectorStateBatch<kDim>> state_batch;
  std::unique_ptr<VectorBetweenFactorBatch<kDim>> between_batch;
  std::unique_ptr<PriorVectorFactorBatch<kDim>> prior_batch;
  std::vector<float *> between_pointers;
  std::vector<float *> prior_pointers;
  Problem problem;
};

/**
 * @brief Chain of `num_blocks` vector states with between + prior factors.
 *
 * @param num_blocks Number of state blocks.
 * @param constant_indices State blocks held constant (may be empty).
 * @param repeat_block When true, every between factor references the same
 *        state block on both sides, exercising the accumulate-twice path.
 * @param stride Gap between the two blocks a between factor joins; larger
 *        values keep the block and factor counts but change the connectivity.
 */
std::unique_ptr<VectorChainProblem> MakeVectorChain(int num_blocks,
                                                    const std::vector<int> &constant_indices,
                                                    bool repeat_block = false, int stride = 1) {
  constexpr int kDim = VectorChainProblem::kDim;
  auto data = std::make_unique<VectorChainProblem>();
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> &host_states = data->host_states;
  host_states.resize(num_blocks * kDim);
  for (float &x : host_states) {
    x = dist(rng);
  }
  data->states.resize(host_states.size());
  data->states.CopyFromHost(host_states.data(), host_states.size());

  const int num_between = num_blocks - 1;
  std::vector<Vector<kDim>> host_deltas(num_between);
  for (auto &d : host_deltas) {
    for (int k = 0; k < kDim; k++) {
      d[k] = dist(rng);
    }
  }
  data->deltas.resize(host_deltas.size());
  data->deltas.CopyFromHost(host_deltas.data(), host_deltas.size());

  std::vector<Vector<kDim>> &host_priors = data->host_priors;
  host_priors.resize(num_blocks);
  for (auto &p : host_priors) {
    for (int k = 0; k < kDim; k++) {
      p[k] = dist(rng);
    }
  }
  data->priors.resize(host_priors.size());
  data->priors.CopyFromHost(host_priors.data(), host_priors.size());

  if (constant_indices.empty()) {
    data->state_batch = std::make_unique<VectorStateBatch<kDim>>(data->states.data(), num_blocks);
  } else {
    data->const_ids.resize(constant_indices.size());
    data->const_ids.CopyFromHost(constant_indices.data(), constant_indices.size());
    data->state_batch = std::make_unique<VectorStateBatch<kDim>>(
        data->states.data(), num_blocks, data->const_ids.data(), constant_indices.size());
  }

  data->between_batch =
      std::make_unique<VectorBetweenFactorBatch<kDim>>(data->deltas.data(), num_between);
  data->prior_batch =
      std::make_unique<PriorVectorFactorBatch<kDim>>(data->priors.data(), num_blocks);

  for (int i = 0; i < num_between; i++) {
    float *left = data->states.data() + static_cast<size_t>(i) * kDim;
    const int right_index = repeat_block ? i : (i + stride) % num_blocks;
    float *right = data->states.data() + static_cast<size_t>(right_index) * kDim;
    data->between_pointers.push_back(left);
    data->between_pointers.push_back(right);
  }
  for (int i = 0; i < num_blocks; i++) {
    data->prior_pointers.push_back(data->states.data() + static_cast<size_t>(i) * kDim);
  }

  data->problem.AddStateBatch(data->state_batch.get());
  data->problem.AddFactorBatch(data->between_batch.get(), data->between_pointers);
  data->problem.AddFactorBatch(data->prior_batch.get(), data->prior_pointers);
  return data;
}

/** @brief Owns the device data behind a mixed-tangent-dimension Problem. */
struct MixedDimProblem {
  dvector<float> states_a;
  dvector<float> states_b;
  dvector<Vector<3>> priors_a;
  dvector<Vector<6>> priors_b;
  std::unique_ptr<VectorStateBatch<3>> batch_a;
  std::unique_ptr<VectorStateBatch<6>> batch_b;
  std::unique_ptr<PriorVectorFactorBatch<3>> prior_a;
  std::unique_ptr<PriorVectorFactorBatch<6>> prior_b;
  std::vector<float *> pointers_a;
  std::vector<float *> pointers_b;
  Problem problem;
};

/** @brief Two state batches with different tangent dims in one problem. */
std::unique_ptr<MixedDimProblem> MakeMixedDim(int count) {
  auto data = std::make_unique<MixedDimProblem>();
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> host_a(count * 3);
  std::vector<float> host_b(count * 6);
  for (float &x : host_a) {
    x = dist(rng);
  }
  for (float &x : host_b) {
    x = dist(rng);
  }
  data->states_a.resize(host_a.size());
  data->states_a.CopyFromHost(host_a.data(), host_a.size());
  data->states_b.resize(host_b.size());
  data->states_b.CopyFromHost(host_b.data(), host_b.size());

  std::vector<Vector<3>> pa(count);
  std::vector<Vector<6>> pb(count);
  for (int i = 0; i < count; i++) {
    for (int k = 0; k < 3; k++) {
      pa[i][k] = dist(rng);
    }
    for (int k = 0; k < 6; k++) {
      pb[i][k] = dist(rng);
    }
  }
  data->priors_a.resize(pa.size());
  data->priors_a.CopyFromHost(pa.data(), pa.size());
  data->priors_b.resize(pb.size());
  data->priors_b.CopyFromHost(pb.data(), pb.size());

  data->batch_a = std::make_unique<VectorStateBatch<3>>(data->states_a.data(), count);
  data->batch_b = std::make_unique<VectorStateBatch<6>>(data->states_b.data(), count);
  data->prior_a = std::make_unique<PriorVectorFactorBatch<3>>(data->priors_a.data(), count);
  data->prior_b = std::make_unique<PriorVectorFactorBatch<6>>(data->priors_b.data(), count);

  for (int i = 0; i < count; i++) {
    data->pointers_a.push_back(data->states_a.data() + static_cast<size_t>(i) * 3);
    data->pointers_b.push_back(data->states_b.data() + static_cast<size_t>(i) * 6);
  }

  data->problem.AddStateBatch(data->batch_a.get());
  data->problem.AddStateBatch(data->batch_b.get());
  data->problem.AddFactorBatch(data->prior_a.get(), data->pointers_a);
  data->problem.AddFactorBatch(data->prior_b.get(), data->pointers_b);
  return data;
}

/** @brief Owns the device data behind an SE3 pose-graph Problem. */
struct PoseGraphProblem {
  cuBLASHandle cublas_handle;
  dvector<SE3Transform> poses;
  dvector<SE3Transform> deltas;
  dvector<int> const_ids;
  std::unique_ptr<SE3StateBatch> pose_batch;
  std::unique_ptr<SE3BetweenFactorBatch> between_batch;
  std::vector<float *> pointers;
  Problem problem;
};

/** @brief Random SE3 poses joined by consecutive between factors. */
std::unique_ptr<PoseGraphProblem> MakePoseGraph(int num_poses, bool fix_first_pose) {
  auto data = std::make_unique<PoseGraphProblem>();
  CudaStream stream;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> rot(-0.4f, 0.4f);
  std::uniform_real_distribution<float> trans(-2.f, 2.f);

  auto random_transforms = [&](int count, dvector<SE3Transform> &out) {
    std::vector<Vector<6>> twists(count);
    for (auto &t : twists) {
      for (int k = 0; k < 3; k++) {
        t[k] = rot(rng);
      }
      for (int k = 3; k < 6; k++) {
        t[k] = trans(rng);
      }
    }
    dvector<Vector<6>> twists_device(count);
    twists_device.CopyFromHost(twists.data(), twists.size());
    out.resize(count);
    ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twists_device.data()), 6, 4,
                  16, count, reinterpret_cast<float *>(out.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  };

  random_transforms(num_poses, data->poses);
  random_transforms(num_poses - 1, data->deltas);

  if (fix_first_pose) {
    std::vector<int> ids{0};
    data->const_ids.resize(1);
    data->const_ids.CopyFromHost(ids.data(), 1);
    data->pose_batch = std::make_unique<SE3StateBatch>(
        data->cublas_handle, reinterpret_cast<const float *>(data->poses.data()), num_poses,
        data->const_ids.data(), 1);
  } else {
    data->pose_batch = std::make_unique<SE3StateBatch>(
        data->cublas_handle, reinterpret_cast<const float *>(data->poses.data()), num_poses);
  }

  data->between_batch = std::make_unique<SE3BetweenFactorBatch>(data->deltas.data(), num_poses - 1);

  auto *base = reinterpret_cast<float *>(data->poses.data());
  for (int i = 0; i + 1 < num_poses; i++) {
    data->pointers.push_back(base + static_cast<size_t>(i) * 16);
    data->pointers.push_back(base + static_cast<size_t>(i + 1) * 16);
  }

  data->problem.AddStateBatch(data->pose_batch.get());
  data->problem.AddFactorBatch(data->between_batch.get(), data->pointers);
  return data;
}

/** @brief Owns the device data behind a small bundle-adjustment Problem. */
struct BundleProblem {
  cuBLASHandle cublas_handle;
  dvector<SE3Transform> poses;
  dvector<float> points;
  dvector<Vector<2>> observations;
  std::unique_ptr<SE3StateBatch> pose_batch;
  std::unique_ptr<VectorStateBatch<3>> point_batch;
  std::unique_ptr<ReprojectionFactorBatch> reprojection_batch;
  std::unique_ptr<HuberLossFunctionBatch> huber;
  std::vector<float *> pointers;
  Problem problem;
};

/**
 * @brief Reprojection factors over 6-dof poses and 3-dof points.
 *
 * Mixes tangent dims *inside* a single factor (6 + 3), which the vector
 * problems above cannot exercise.
 */
std::unique_ptr<BundleProblem> MakeBundle(int num_poses, int num_points, bool robust_loss) {
  auto data = std::make_unique<BundleProblem>();
  CudaStream stream;
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> small(-0.15f, 0.15f);
  std::uniform_real_distribution<float> obs_noise(-0.02f, 0.02f);

  std::vector<Vector<6>> twists(num_poses);
  for (auto &t : twists) {
    for (int k = 0; k < 6; k++) {
      t[k] = small(rng);
    }
  }
  dvector<Vector<6>> twists_device(num_poses);
  twists_device.CopyFromHost(twists.data(), twists.size());
  data->poses.resize(num_poses);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twists_device.data()), 6, 4, 16,
                num_poses, reinterpret_cast<float *>(data->poses.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Points sit well in front of the cameras so every projection is valid.
  std::vector<float> host_points(num_points * 3);
  std::uniform_real_distribution<float> lateral(-1.f, 1.f);
  std::uniform_real_distribution<float> depth(4.f, 8.f);
  for (int i = 0; i < num_points; i++) {
    host_points[i * 3 + 0] = lateral(rng);
    host_points[i * 3 + 1] = lateral(rng);
    host_points[i * 3 + 2] = depth(rng);
  }
  data->points.resize(host_points.size());
  data->points.CopyFromHost(host_points.data(), host_points.size());

  std::vector<Vector<2>> host_obs;
  auto *pose_base = reinterpret_cast<float *>(data->poses.data());
  for (int p = 0; p < num_points; p++) {
    for (int c = 0; c < num_poses; c++) {
      float x = host_points[p * 3 + 0];
      float y = host_points[p * 3 + 1];
      float z = host_points[p * 3 + 2];
      host_obs.push_back({x / z + obs_noise(rng), y / z + obs_noise(rng)});
      data->pointers.push_back(pose_base + static_cast<size_t>(c) * 16);
      data->pointers.push_back(data->points.data() + static_cast<size_t>(p) * 3);
    }
  }
  data->observations.resize(host_obs.size());
  data->observations.CopyFromHost(host_obs.data(), host_obs.size());

  data->pose_batch = std::make_unique<SE3StateBatch>(
      data->cublas_handle, reinterpret_cast<const float *>(data->poses.data()), num_poses);
  data->point_batch = std::make_unique<VectorStateBatch<3>>(data->points.data(), num_points);
  data->reprojection_batch =
      std::make_unique<ReprojectionFactorBatch>(data->observations.data(), host_obs.size());

  data->problem.AddStateBatch(data->pose_batch.get());
  data->problem.AddStateBatch(data->point_batch.get());
  if (robust_loss) {
    data->huber = std::make_unique<HuberLossFunctionBatch>(0.02f);
    data->problem.AddFactorBatch(data->reprojection_batch.get(), data->huber.get(), data->pointers);
  } else {
    data->problem.AddFactorBatch(data->reprojection_batch.get(), data->pointers);
  }
  return data;
}

// ============================================================================
// Structure oracle
// ============================================================================

/** @brief Host-computed CSR sparsity pattern. */
struct ReferenceStructure {
  std::vector<int> row_offsets;
  std::vector<int> col_ids;
  /// Per residual batch: the global column of each (factor, block) slot, -1
  /// when the block is a constant state.
  std::vector<std::vector<int>> factor_columns;
  int num_cols = 0;
};

/**
 * @brief Independent CPU reference for the Hessian sparsity pattern.
 *
 * Both assembly paths now share HessianStructureBuilder, so comparing them
 * against each other can no longer catch a structure bug.  This is a
 * deliberately naive transcription of the definition — resolve every factor's
 * state pointers to global columns, collect the distinct block pairs in an
 * ordered set, expand each tile — so it shares no code with the GPU sort and
 * segmentation it validates.
 */
ReferenceStructure BuildReferenceStructure(const Problem &problem) {
  struct BatchDesc {
    const float *base = nullptr;
    int ambient = 0;
    int tangent = 0;
    int num_blocks = 0;
    std::vector<int> col_of_block;
  };

  std::vector<BatchDesc> descs;
  int last_col = 0;
  for (const auto *state_batch : problem.GetStateBatches()) {
    BatchDesc d;
    d.base = state_batch->StateBlockDevicePtr(0);
    d.ambient = static_cast<int>(state_batch->AmbientSize());
    d.tangent = static_cast<int>(state_batch->TangentSize());
    d.num_blocks = static_cast<int>(state_batch->NumStateBlocks());

    std::vector<int> const_ids(state_batch->NumConstStateBlocks());
    if (!const_ids.empty()) {
      THROW_ON_CUDA_ERROR(cudaMemcpy(const_ids.data(), state_batch->ConstStateIds(),
                                     const_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<bool> is_const(d.num_blocks, false);
    for (int id : const_ids) {
      if (id >= 0 && id < d.num_blocks) {
        is_const[id] = true;
      }
    }

    d.col_of_block.assign(d.num_blocks, -1);
    for (int j = 0; j < d.num_blocks; j++) {
      if (!is_const[j]) {
        d.col_of_block[j] = last_col;
        last_col += d.tangent;
      }
    }
    descs.push_back(std::move(d));
  }

  ReferenceStructure out;
  out.num_cols = last_col;

  std::vector<int> tangent_at_col(last_col, 0);
  for (const BatchDesc &d : descs) {
    for (int j = 0; j < d.num_blocks; j++) {
      if (d.col_of_block[j] >= 0) {
        for (int k = 0; k < d.tangent; k++) {
          tangent_at_col[d.col_of_block[j] + k] = d.tangent;
        }
      }
    }
  }

  // Mirrors the tangent-dim guard the resolve kernel applies.
  auto resolve = [&](const float *ptr, int declared_tangent) -> int {
    for (const BatchDesc &d : descs) {
      ptrdiff_t diff = ptr - d.base;
      if (diff >= 0 && diff < static_cast<ptrdiff_t>(d.num_blocks) * d.ambient) {
        if (d.tangent != declared_tangent) {
          return -1;
        }
        return d.col_of_block[diff / d.ambient];
      }
    }
    return -1;
  };

  std::set<std::pair<int, int>> pairs;
  const auto &residual_batches = problem.GetResidualBatches();
  const auto &state_pointers = problem.GetStatePointers();
  out.factor_columns.resize(residual_batches.size());
  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto *factor_batch = residual_batches[i].GetFactorBatch();
    auto block_sizes = factor_batch->StateBlockSizes();
    const size_t nb = block_sizes.size();
    for (size_t f = 0; f < factor_batch->NumFactors(); f++) {
      std::vector<int> cols;
      for (size_t b = 0; b < nb; b++) {
        int col = resolve(state_pointers[i][f * nb + b], static_cast<int>(block_sizes[b]));
        out.factor_columns[i].push_back(col);
        if (col >= 0) {
          cols.push_back(col);
        }
      }
      for (int a : cols) {
        for (int b : cols) {
          pairs.emplace(a, b);
        }
      }
    }
  }

  // Expand: within a block row the tiles sit side by side in column order.
  std::vector<int> row_counts(last_col, 0);
  std::vector<int> write_offsets;
  write_offsets.reserve(pairs.size());
  int prev_row = -1;
  int cursor = 0;
  for (const auto &p : pairs) {
    if (p.first != prev_row) {
      cursor = 0;
      prev_row = p.first;
    }
    write_offsets.push_back(cursor);
    cursor += tangent_at_col[p.second];
    for (int i = 0; i < tangent_at_col[p.first]; i++) {
      row_counts[p.first + i] += tangent_at_col[p.second];
    }
  }

  out.row_offsets.assign(last_col + 1, 0);
  for (int r = 0; r < last_col; r++) {
    out.row_offsets[r + 1] = out.row_offsets[r] + row_counts[r];
  }
  out.col_ids.assign(out.row_offsets.back(), -1);

  size_t k = 0;
  for (const auto &p : pairs) {
    const int height = tangent_at_col[p.first];
    const int width = tangent_at_col[p.second];
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        out.col_ids[out.row_offsets[p.first + i] + write_offsets[k] + j] = p.second + j;
      }
    }
    k++;
  }
  return out;
}

/**
 * @brief Host reference for the assembled normal equations.
 *
 * Contracts the very same per-factor Jacobian blocks the GPU kernel reads, but
 * with plain nested loops into a dense-per-row map, so it shares no addressing
 * or accumulation logic with the code it checks.  Constant state blocks are
 * dropped exactly as the kernel drops them.
 *
 * @param problem The optimization problem.
 * @param jacobians Per-factor dense Jacobian blocks, read back from the device.
 * @param residuals Residual vector, read back from the device.
 * @param structure Sparsity pattern the values must be laid out against.
 * @param[out] values CSR values of H.
 * @param[out] rhs Right-hand side -J^T r.
 */
void ComputeReferenceSystem(const Problem &problem, const std::vector<float> &jacobians,
                            const std::vector<float> &residuals,
                            const ReferenceStructure &structure, std::vector<float> &values,
                            std::vector<float> &rhs) {
  values.assign(structure.col_ids.size(), 0.f);
  rhs.assign(structure.num_cols, 0.f);

  // (row, col) -> index into values, built once from the reference pattern.
  std::map<std::pair<int, int>, size_t> slot;
  for (int row = 0; row + 1 < static_cast<int>(structure.row_offsets.size()); row++) {
    for (int k = structure.row_offsets[row]; k < structure.row_offsets[row + 1]; k++) {
      slot[{row, structure.col_ids[k]}] = static_cast<size_t>(k);
    }
  }

  size_t jacobian_cursor = 0;
  size_t residual_cursor = 0;
  const auto &residual_batches = problem.GetResidualBatches();
  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto *factor_batch = residual_batches[i].GetFactorBatch();
    const auto block_sizes = factor_batch->StateBlockSizes();
    const size_t num_blocks = block_sizes.size();
    const size_t residual_dim = factor_batch->ResidualsSize();
    const size_t tangent_dim = std::accumulate(block_sizes.begin(), block_sizes.end(), size_t(0));

    const std::vector<int> &columns = structure.factor_columns[i];
    for (size_t f = 0; f < factor_batch->NumFactors(); f++) {
      const float *J = jacobians.data() + jacobian_cursor + f * residual_dim * tangent_dim;
      const float *r = residuals.data() + residual_cursor + f * residual_dim;

      // Local tangent index -> global column, or -1 for a constant block.
      std::vector<int> global(tangent_dim, -1);
      size_t cursor = 0;
      for (size_t b = 0; b < num_blocks; b++) {
        const int base = columns[f * num_blocks + b];
        for (size_t k = 0; k < block_sizes[b]; k++, cursor++) {
          global[cursor] = base < 0 ? -1 : base + static_cast<int>(k);
        }
      }

      for (size_t p = 0; p < tangent_dim; p++) {
        if (global[p] < 0) {
          continue;
        }
        double b_acc = 0.0;
        for (size_t k = 0; k < residual_dim; k++) {
          b_acc += static_cast<double>(J[k * tangent_dim + p]) * r[k];
        }
        rhs[global[p]] -= static_cast<float>(b_acc);

        for (size_t q = 0; q < tangent_dim; q++) {
          if (global[q] < 0) {
            continue;
          }
          double h_acc = 0.0;
          for (size_t k = 0; k < residual_dim; k++) {
            h_acc += static_cast<double>(J[k * tangent_dim + p]) * J[k * tangent_dim + q];
          }
          values[slot.at({global[p], global[q]})] += static_cast<float>(h_acc);
        }
      }
    }
    jacobian_cursor += factor_batch->NumFactors() * residual_dim * tangent_dim;
    residual_cursor += factor_batch->NumFactors() * residual_dim;
  }
}

/**
 * @brief Asserts the assembled H and rhs match the CPU oracle.
 *
 * Structure is compared exactly; values to `rel_tol` relative to the largest
 * entry, which is the right scale here because both sides sum the same products
 * in a different order, so the error is bounded by the largest partial sum
 * rather than by each entry's own magnitude.
 */
void ExpectMatchesReference(Problem &problem, float rel_tol = 1e-5f) {
  CudaStream stream;
  SystemBuilder builder;
  builder.Build(stream.GetStream(), problem);
  SystemSnapshot actual = Snapshot(builder, stream.GetStream());

  std::vector<float> jacobians(builder.FactorJacobians().size());
  std::vector<float> residuals(builder.Residuals().size());
  builder.FactorJacobians().CopyToHost(jacobians.data(), jacobians.size());
  builder.Residuals().CopyToHost(residuals.data(), residuals.size());

  ReferenceStructure structure = BuildReferenceStructure(problem);
  std::vector<float> expected_values;
  std::vector<float> expected_rhs;
  ComputeReferenceSystem(problem, jacobians, residuals, structure, expected_values, expected_rhs);

  ASSERT_EQ(structure.row_offsets, actual.row_offsets);
  ASSERT_EQ(structure.col_ids, actual.col_ids);
  ASSERT_EQ(expected_values.size(), actual.values.size());
  ASSERT_EQ(expected_rhs.size(), actual.rhs.size());

  const float h_tol = rel_tol * std::max(MaxAbs(expected_values), 1e-6f);
  for (size_t i = 0; i < expected_values.size(); i++) {
    ASSERT_NEAR(expected_values[i], actual.values[i], h_tol) << "Hessian mismatch at nnz " << i;
  }
  const float b_tol = rel_tol * std::max(MaxAbs(expected_rhs), 1e-6f);
  for (size_t i = 0; i < expected_rhs.size(); i++) {
    ASSERT_NEAR(expected_rhs[i], actual.rhs[i], b_tol) << "RHS mismatch at " << i;
  }
}

/** @brief Asserts the GPU-built pattern matches the CPU oracle exactly. */
void ExpectStructureMatchesReference(Problem &problem) {
  ReferenceStructure expected = BuildReferenceStructure(problem);

  CudaStream stream;
  SystemBuilder builder = SystemBuilder::Scalar();
  builder.Build(stream.GetStream(), problem);
  SystemSnapshot actual = Snapshot(builder, stream.GetStream());

  ASSERT_EQ(expected.row_offsets.size(), actual.row_offsets.size());
  EXPECT_EQ(expected.row_offsets, actual.row_offsets);
  ASSERT_EQ(expected.col_ids.size(), actual.col_ids.size());
  EXPECT_EQ(expected.col_ids, actual.col_ids);
}

TEST(HessianStructureTest, MatchesReferenceOnVectorChain) {
  // Between and prior factors both emit the (i, i) block pair, so this covers
  // the duplicate-key segmentation that an exclusive scan gets wrong.
  auto data = MakeVectorChain(64, {});
  ExpectStructureMatchesReference(data->problem);
}

TEST(HessianStructureTest, MatchesReferenceWithConstantStates) {
  auto data = MakeVectorChain(64, {0, 7, 8, 63});
  ExpectStructureMatchesReference(data->problem);
}

TEST(HessianStructureTest, MatchesReferenceWithMixedTangentDims) {
  auto data = MakeMixedDim(48);
  ExpectStructureMatchesReference(data->problem);
}

TEST(HessianStructureTest, MatchesReferenceOnPoseGraph) {
  auto data = MakePoseGraph(512, /*fix_first_pose=*/true);
  ExpectStructureMatchesReference(data->problem);
}

TEST(HessianStructureTest, MatchesReferenceOnBundleAdjustment) {
  auto data = MakeBundle(8, 200, /*robust_loss=*/false);
  ExpectStructureMatchesReference(data->problem);
}

TEST(HessianStructureTest, MatchesReferenceWithRepeatedStateBlock) {
  auto data = MakeVectorChain(32, {}, /*repeat_block=*/true);
  ExpectStructureMatchesReference(data->problem);
}

// ============================================================================
// Storage-layout tests
// ============================================================================

/**
 * @brief Asserts scalar and block storage assemble the same system.
 *
 * Both go through the same assembler, so this isolates the addressing change:
 * a block pair is exactly tiled by `b x b` tiles, so expanding the block form
 * must reproduce the scalar pattern and values entry for entry.  Accumulation
 * order differs between the two, hence a tolerance on values.
 */
void ExpectStorageLayoutsAgree(Problem &problem, float rel_tol = 1e-5f) {
  CudaStream stream;

  SystemBuilder scalar = SystemBuilder::Scalar();
  scalar.Build(stream.GetStream(), problem);
  ASSERT_FALSE(scalar.UsesBlockStorage());
  SystemSnapshot expected = Snapshot(scalar, stream.GetStream());

  SystemBuilder block;
  block.Build(stream.GetStream(), problem);
  ASSERT_TRUE(block.UsesBlockStorage()) << "problem should qualify for block storage";
  SystemSnapshot actual = Snapshot(block, stream.GetStream());

  EXPECT_EQ(expected.row_offsets, actual.row_offsets);
  EXPECT_EQ(expected.col_ids, actual.col_ids);
  ASSERT_EQ(expected.values.size(), actual.values.size());

  const float h_tol = rel_tol * std::max(MaxAbs(expected.values), 1e-6f);
  for (size_t i = 0; i < expected.values.size(); i++) {
    ASSERT_NEAR(expected.values[i], actual.values[i], h_tol) << "Hessian mismatch at nnz " << i;
  }
  const float b_tol = rel_tol * std::max(MaxAbs(expected.rhs), 1e-6f);
  ASSERT_EQ(expected.rhs.size(), actual.rhs.size());
  for (size_t i = 0; i < expected.rhs.size(); i++) {
    ASSERT_NEAR(expected.rhs[i], actual.rhs[i], b_tol) << "RHS mismatch at " << i;
  }
}

TEST(HessianStorageTest, BlockMatchesScalarOnPoseGraph) {
  auto data = MakePoseGraph(512, /*fix_first_pose=*/true);
  ExpectStorageLayoutsAgree(data->problem);
}

TEST(HessianStorageTest, BlockMatchesScalarOnBundleAdjustment) {
  // Mixed 6-dof poses and 3-dof points: gcd 3, so a pose tile spans a 2x2 grid
  // of blocks while a point tile is a single block.
  auto data = MakeBundle(8, 200, /*robust_loss=*/false);
  ExpectStorageLayoutsAgree(data->problem);
}

TEST(HessianStorageTest, BlockMatchesScalarWithConstantStates) {
  auto data = MakeVectorChain(64, {0, 7, 8, 63});
  ExpectStorageLayoutsAgree(data->problem);
}

TEST(HessianStorageTest, BlockMatchesScalarWithRepeatedStateBlock) {
  auto data = MakeVectorChain(32, {}, /*repeat_block=*/true);
  ExpectStorageLayoutsAgree(data->problem);
}

/**
 * @brief Checks the BSR diagonal operations against their CSR counterparts.
 *
 * These two are what Levenberg-Marquardt applies to the LHS on every
 * iteration, and the assembly-equivalence tests above never damp, so nothing
 * else covers them.
 */
TEST(HessianStorageTest, BlockDiagonalOpsMatchScalar) {
  auto data = MakeBundle(8, 200, /*robust_loss=*/false);
  CudaStream stream;
  cudaStream_t s = stream.GetStream();

  SystemBuilder scalar = SystemBuilder::Scalar();
  scalar.Build(s, data->problem);
  SystemBuilder block;
  block.Build(s, data->problem);
  ASSERT_TRUE(block.UsesBlockStorage());

  // 1. Diagonal extraction.
  dvector<float> csr_diag;
  dvector<float> bsr_diag;
  scalar.Equations().ExtractLhsDiagonal(s, csr_diag);
  block.Equations().ExtractLhsDiagonal(s, bsr_diag);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  std::vector<float> expected_diag(csr_diag.size());
  std::vector<float> actual_diag(bsr_diag.size());
  csr_diag.CopyToHost(expected_diag.data(), expected_diag.size());
  bsr_diag.CopyToHost(actual_diag.data(), actual_diag.size());
  ASSERT_EQ(expected_diag.size(), actual_diag.size());
  const float diag_tol = 1e-5f * std::max(MaxAbs(expected_diag), 1e-6f);
  for (size_t i = 0; i < expected_diag.size(); i++) {
    ASSERT_NEAR(expected_diag[i], actual_diag[i], diag_tol) << "diagonal mismatch at " << i;
  }

  // 2. Damped update, in place, exactly as LM performs it.
  constexpr float kLambda = 0.017f;
  scalar.Equations().AddScaledDiagonalToLhs(s, kLambda, csr_diag);
  block.Equations().AddScaledDiagonalToLhs(s, kLambda, bsr_diag);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  SystemSnapshot expected = Snapshot(scalar, s);
  SystemSnapshot actual = Snapshot(block, s);
  ASSERT_EQ(expected.values.size(), actual.values.size());
  const float tol = 1e-5f * std::max(MaxAbs(expected.values), 1e-6f);
  for (size_t i = 0; i < expected.values.size(); i++) {
    ASSERT_NEAR(expected.values[i], actual.values[i], tol) << "damped LHS mismatch at nnz " << i;
  }
}

/**
 * @brief The PCG solver must behave identically on the two storage layouts.
 *
 * Same matrix, same right-hand side, same preconditioner -- so the solution and
 * the iteration count should match.  A divergence here points at the block
 * SpMV or the block-Jacobi tile gather rather than at assembly.
 */
TEST(HessianStorageTest, PcgAgreesBetweenStorages) {
  auto data = MakePoseGraph(2048, /*fix_first_pose=*/true);
  CudaStream stream;
  cudaStream_t s = stream.GetStream();

  SystemBuilder block;
  block.Build(s, data->problem);
  ASSERT_TRUE(block.UsesBlockStorage());

  CSRSparseMatrix csr;
  dvector<int> expand_scratch;
  test_utils::ExpandBSRToCSR(s, block.Equations().LhsBSR(), csr, expand_scratch);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  const size_t n = block.Rhs().size();
  dvector<float> rhs(n);
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(rhs.data(), block.Rhs().data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, s));

  BlockSparsePCGOptions pcg_options;
  pcg_options.block_size = 6;
  pcg_options.max_iterations = 500;
  pcg_options.relative_tolerance = 1e-6f;

  dvector<float> x_csr(n);
  dvector<float> x_bsr(n);

  BlockSparsePCGSolver csr_solver(pcg_options);
  ASSERT_TRUE(csr_solver.Initialize(s, data->problem, csr, rhs, x_csr));
  ASSERT_TRUE(csr_solver.Solve(s, csr, rhs, x_csr));

  BlockSparsePCGSolver bsr_solver(pcg_options);
  ASSERT_TRUE(bsr_solver.Initialize(s, data->problem, block.Equations().LhsBSR(), rhs, x_bsr));
  ASSERT_TRUE(bsr_solver.Solve(s, block.Equations().LhsBSR(), rhs, x_bsr));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  std::vector<float> a(n);
  std::vector<float> b(n);
  x_csr.CopyToHost(a.data(), n);
  x_bsr.CopyToHost(b.data(), n);

  // Same iteration count means the two are tracking the same recurrence.  The
  // solutions themselves only agree to the level PCG was asked for: it stops on
  // a relative residual, so the reordered block SpMV moves x within that ball.
  EXPECT_EQ(csr_solver.LastIterations(), bsr_solver.LastIterations());
  const float tol = 1e-3f * std::max(MaxAbs(a), 1e-6f);
  for (size_t i = 0; i < n; i++) {
    ASSERT_NEAR(a[i], b[i], tol) << "solution mismatch at " << i;
  }
}

/**
 * @brief One PCG iteration must be bit-comparable across storage layouts.
 *
 * After a single iteration the iterate depends only on the preconditioner and
 * one SpMV, so this separates a bad block-Jacobi tile gather (which would show
 * up here) from ordinary float divergence accumulating over many iterations.
 */
TEST(HessianStorageTest, FirstPcgIterationAgreesBetweenStorages) {
  auto data = MakePoseGraph(1024, /*fix_first_pose=*/true);
  CudaStream stream;
  cudaStream_t s = stream.GetStream();

  SystemBuilder block;
  block.Build(s, data->problem);
  ASSERT_TRUE(block.UsesBlockStorage());

  // Feed both solvers the *same* matrix, so only the reader differs.
  CSRSparseMatrix csr;
  dvector<int> expand_scratch;
  test_utils::ExpandBSRToCSR(s, block.Equations().LhsBSR(), csr, expand_scratch);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  const size_t n = block.Rhs().size();
  dvector<float> rhs(n);
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(rhs.data(), block.Rhs().data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, s));

  BlockSparsePCGOptions pcg_options;
  pcg_options.block_size = 6;
  pcg_options.max_iterations = 1;
  pcg_options.check_period = 1;

  dvector<float> x_csr(n);
  dvector<float> x_bsr(n);
  BlockSparsePCGSolver csr_solver(pcg_options);
  ASSERT_TRUE(csr_solver.Initialize(s, data->problem, csr, rhs, x_csr));
  ASSERT_TRUE(csr_solver.Solve(s, csr, rhs, x_csr));

  BlockSparsePCGSolver bsr_solver(pcg_options);
  ASSERT_TRUE(bsr_solver.Initialize(s, data->problem, block.Equations().LhsBSR(), rhs, x_bsr));
  ASSERT_TRUE(bsr_solver.Solve(s, block.Equations().LhsBSR(), rhs, x_bsr));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(s));

  std::vector<float> a(n);
  std::vector<float> b(n);
  x_csr.CopyToHost(a.data(), n);
  x_bsr.CopyToHost(b.data(), n);

  const float tol = 1e-6f * std::max(MaxAbs(a), 1e-6f);
  for (size_t i = 0; i < n; i++) {
    ASSERT_NEAR(a[i], b[i], tol) << "first iterate mismatch at " << i;
  }
}

/**
 * @brief `step^T H step` must agree across storage layouts.
 *
 * Levenberg-Marquardt divides by this to form rho, so an error here does not
 * corrupt the solution directly -- it corrupts the accept/reject decision, and
 * the minimizer walks off to a worse answer while every matrix-level test still
 * passes.
 */
TEST(HessianStorageTest, WeightedSquaredStepAgreesBetweenStorages) {
  auto data = MakeBundle(8, 300, /*robust_loss=*/false);
  CudaStream stream;
  cudaStream_t s = stream.GetStream();

  SystemBuilder scalar = SystemBuilder::Scalar();
  scalar.Build(s, data->problem);
  SystemBuilder block;
  block.Build(s, data->problem);
  ASSERT_TRUE(block.UsesBlockStorage());

  const size_t n = scalar.Rhs().size();
  std::vector<float> host_step(n);
  std::mt19937 rng(97);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (float &v : host_step) {
    v = dist(rng);
  }
  dvector<float> step(n);
  step.CopyFromHost(host_step.data(), n);

  const float expected = scalar.WeightedSquaredStep(s, step);
  const float actual = block.WeightedSquaredStep(s, step);
  EXPECT_NEAR(expected, actual, 1e-4f * std::max(std::fabs(expected), 1.f));
}

/**
 * @brief One builder reused across problems must not carry structure over.
 *
 * Regression test.  The BSR copy originally skipped the structure whenever the
 * destination's sizes already matched, so a second problem with the same block
 * and tile counts but different connectivity inherited the first one's column
 * indices.  Every matrix-level equivalence test still passed -- the damage only
 * showed up as Levenberg-Marquardt converging to worse costs on a dataset of
 * many similarly-sized problems.
 */
TEST(HessianStorageTest, ReusedBuilderRebuildsStructurePerProblem) {
  // Same block count and same factor count, different connectivity: a chain
  // versus a pairing that skips two.
  auto chain = MakeVectorChain(32, {});
  auto skipped = MakeVectorChain(32, {}, /*repeat_block=*/false, /*stride=*/3);
  ASSERT_TRUE(chain->problem.CheckConsistency());
  ASSERT_TRUE(skipped->problem.CheckConsistency());

  CudaStream stream;
  cudaStream_t s = stream.GetStream();

  SystemBuilder fresh;
  fresh.Build(s, skipped->problem);
  SystemSnapshot expected = Snapshot(fresh, s);

  SystemBuilder reused;
  reused.Build(s, chain->problem);
  reused.Build(s, skipped->problem);
  SystemSnapshot actual = Snapshot(reused, s);

  EXPECT_EQ(expected.row_offsets, actual.row_offsets);
  EXPECT_EQ(expected.col_ids, actual.col_ids);
  ASSERT_EQ(expected.values.size(), actual.values.size());
  const float tol = 1e-5f * std::max(MaxAbs(expected.values), 1e-6f);
  for (size_t i = 0; i < expected.values.size(); i++) {
    ASSERT_NEAR(expected.values[i], actual.values[i], tol) << "stale structure at nnz " << i;
  }
}

TEST(HessianStorageTest, BlockSizeIsTheTangentDimensionGcd) {
  auto pgo = MakePoseGraph(16, /*fix_first_pose=*/false);
  EXPECT_EQ(6, ChooseHessianBlockSize(pgo->problem));

  auto bundle = MakeBundle(4, 20, /*robust_loss=*/false);
  EXPECT_EQ(3, ChooseHessianBlockSize(bundle->problem));

  // 4-dimensional vector states alone tile at 4.
  auto chain = MakeVectorChain(8, {});
  EXPECT_EQ(4, ChooseHessianBlockSize(chain->problem));

  // gcd(3, 6) = 3 stays, but a coprime mix must fall back to scalar storage.
  auto mixed = MakeMixedDim(8);
  EXPECT_EQ(3, ChooseHessianBlockSize(mixed->problem));
}

TEST(HessianStorageTest, FallsBackToScalarWhenSolverNeedsCSR) {
  // cuDSS consumes CSR, so block storage would only have to be expanded again.
  auto data = MakePoseGraph(64, /*fix_first_pose=*/true);
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::cuDSS;

  CudaStream stream;
  SystemBuilder builder(options);
  builder.Build(stream.GetStream(), data->problem);
  EXPECT_FALSE(builder.UsesBlockStorage());
}

// ============================================================================
// Equivalence tests
// ============================================================================

TEST(BlockHessianAssemblerTest, MatchesReferenceOnVectorChain) {
  auto data = MakeVectorChain(64, {});
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceWithConstantStates) {
  auto data = MakeVectorChain(64, {0, 7, 8, 63});
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, AccumulatesRepeatedStateBlock) {
  // Both slots of every between factor point at the same state block, so all
  // four H_f sub-blocks land on the same CSR entries and must accumulate.
  //
  // Checked against the analytic answer directly.  The pre-rewrite path built a
  // triplet Jacobian and resolved duplicate (row, col) entries to a single CSR
  // slot, storing rather than accumulating into it, so it silently dropped one
  // of the two Jacobian blocks and got this case wrong.
  //
  // r_f = x_i - x_i - delta = -delta with J_f = [I | -I], so the four
  // sub-blocks sum to I - I - I + I = 0 and b_f = -(I - I)^T r_f = 0.  Only
  // the prior factor (residual x_i - p_i, Jacobian I) survives.
  constexpr int kBlocks = 32;
  constexpr int kDim = VectorChainProblem::kDim;
  auto data = MakeVectorChain(kBlocks, {}, /*repeat_block=*/true);
  ASSERT_TRUE(data->problem.CheckConsistency());

  CudaStream stream;
  SystemBuilder block;
  block.Build(stream.GetStream(), data->problem);
  SystemSnapshot snapshot = Snapshot(block, stream.GetStream());

  ASSERT_EQ(snapshot.row_offsets.size(), size_t(kBlocks * kDim + 1));
  for (int row = 0; row < kBlocks * kDim; row++) {
    for (int k = snapshot.row_offsets[row]; k < snapshot.row_offsets[row + 1]; k++) {
      float expected = snapshot.col_ids[k] == row ? 1.f : 0.f;
      EXPECT_NEAR(expected, snapshot.values[k], 1e-6f)
          << "row " << row << ", col " << snapshot.col_ids[k];
    }
  }

  ASSERT_EQ(snapshot.rhs.size(), size_t(kBlocks * kDim));
  for (int i = 0; i < kBlocks; i++) {
    for (int k = 0; k < kDim; k++) {
      float expected = data->host_priors[i][k] - data->host_states[i * kDim + k];
      EXPECT_NEAR(expected, snapshot.rhs[i * kDim + k], 1e-5f)
          << "block " << i << ", component " << k;
    }
  }
}

TEST(BlockHessianAssemblerTest, MatchesReferenceWithMixedTangentDims) {
  auto data = MakeMixedDim(48);
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceOnPoseGraph) {
  auto data = MakePoseGraph(512, /*fix_first_pose=*/false);
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceOnPoseGraphWithFixedPose) {
  auto data = MakePoseGraph(512, /*fix_first_pose=*/true);
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceOnBundleAdjustment) {
  auto data = MakeBundle(8, 200, /*robust_loss=*/false);
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceWithRobustLoss) {
  auto data = MakeBundle(8, 200, /*robust_loss=*/true);
  ASSERT_TRUE(data->problem.CheckConsistency());
  ExpectMatchesReference(data->problem);
}

TEST(BlockHessianAssemblerTest, MatchesReferenceWithColumnScaling) {
  auto data = MakePoseGraph(256, /*fix_first_pose=*/true);
  ASSERT_TRUE(data->problem.CheckConsistency());

  CudaStream stream;
  MinimizerOptions options;
  options.column_scaling = ColumnScaling::HessianDiagonal;

  SystemBuilder reference(options);
  reference.Build(stream.GetStream(), data->problem);
  std::vector<float> expected = Snapshot(reference, stream.GetStream()).values;

  SystemBuilder block(options);
  block.Build(stream.GetStream(), data->problem);
  std::vector<float> actual = Snapshot(block, stream.GetStream()).values;

  ASSERT_EQ(expected.size(), actual.size());
  const float tol = 1e-5f * std::max(MaxAbs(expected), 1e-6f);
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_NEAR(expected[i], actual[i], tol) << "mismatch at " << i;
  }
}

TEST(BlockHessianAssemblerTest, EmptyFactorBatchIsSkipped) {
  // A zero-factor batch alongside a populated one must not crash or perturb
  // the assembled system.
  auto data = MakeVectorChain(16, {});
  dvector<Vector<VectorChainProblem::kDim>> empty_priors;
  PriorVectorFactorBatch<VectorChainProblem::kDim> empty_batch(empty_priors.data(), 0);
  std::vector<float *> no_pointers;
  data->problem.AddFactorBatch(&empty_batch, no_pointers);

  CudaStream stream;
  SystemBuilder block;
  ASSERT_NO_THROW(block.Build(stream.GetStream(), data->problem));
  EXPECT_GT(block.HessianAsCSR(stream.GetStream()).NumNonZeros(), 0u);
}

TEST(BlockHessianAssemblerTest, AllConstantStatesProduceZeroSystem) {
  std::vector<int> all_const(16);
  for (int i = 0; i < 16; i++) {
    all_const[i] = i;
  }
  auto data = MakeVectorChain(16, all_const);

  CudaStream stream;
  SystemBuilder block;
  ASSERT_NO_THROW(block.Build(stream.GetStream(), data->problem));

  SystemSnapshot snapshot = Snapshot(block, stream.GetStream());
  for (float v : snapshot.values) {
    EXPECT_EQ(v, 0.f);
  }
  for (float v : snapshot.rhs) {
    EXPECT_EQ(v, 0.f);
  }
}

}  // namespace
}  // namespace cunls
