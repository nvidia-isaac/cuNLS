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

#include <cstddef>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/state/articulation_topology.h"
#include "cunls/state/state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {

/**
 * @brief A batch of instances of one articulation (ArticulationTopology): the
 * generalized coordinates joint_q of each instance, plus a cache of its link
 * poses and motion columns that the kinematic factors read.
 *
 * <b>Storage</b> (owned): Capacity() instances of AmbientSize() floats, laid
 * out as described in ArticulationTopology (joint_q, then the cache). Write
 * coordinates with SetJointQ() and read them with GetJointQ(); a direct write
 * to StateDevicePtr() must be followed by RefreshDerivedData() before an
 * evaluation outside a solve (the minimizers refresh at the start of every
 * solve).
 *
 * <b>Tangent</b> (TangentSize() = topology DOF count), per joint, in the
 * order of joint_qd: PRISMATIC, REVOLUTE and D6 one scalar per axis; BALL a
 * rotation vector δθ in the parent-anchor frame A (quat ← Exp(δθ) ⊗ quat);
 * FREE (δp, δθ) in A (p ← p + δp, quat ← Exp(δθ) ⊗ quat); FIXED none. This is
 * cuNLS's own chart: FREE's δp is not Newton's COM-referenced joint_qd.
 *
 * <b>Plus</b> applies the step per joint, clamps the moved scalar
 * coordinates into the bounds (when there are any) and recomputes the cache
 * (forward kinematics), so factors never run kinematics themselves.
 *
 * <b>Bounds</b>: the topology's joint limits by default (apply_joint_limits),
 * or arrays set with SetBounds(). Like every bounded state, solve with
 * AugmentedLagrangianMinimizer (projected Gauss-Newton keeps the scalar
 * coordinates inside); GN/LM/RANSAC reject bounded problems.
 */
class ArticulationStateBatch : public StateBatch, public BoxBoundedStates, public DerivedStateData {
 public:
  /**
   * @brief Allocates `capacity` instances in the zero configuration (scalar
   * coordinates 0, identity quaternions) with their cache computed. The active
   * count starts at 0: call SetNumActiveStates(n) before solving.
   *
   * @param topology Kinematic tree; must outlive this batch.
   * @param capacity Instances the storage holds.
   * @param apply_joint_limits Use the topology's limits as box bounds (when it
   *        has finite ones). false: unbounded until SetBounds().
   */
  ArticulationStateBatch(const ArticulationTopology &topology, size_t capacity,
                         bool apply_joint_limits = true);

  /**
   * @brief As above, with a user-owned device array of up to `const_capacity`
   * ids of instances held constant (see SetNumActiveStates).
   */
  ArticulationStateBatch(const ArticulationTopology &topology, size_t capacity,
                         const int *device_constant_state_ids, size_t const_capacity,
                         bool apply_joint_limits = true);

  /** @brief The kinematic tree of every instance. */
  const ArticulationTopology &Topology() const { return topology_; }

  /**
   * @brief Copies the coordinates of the first `num_states` instances from
   * `joint_q` (device array, Newton's joint_q layout) and recomputes their
   * cache. Asynchronous on `stream`.
   *
   * @param joint_q Instance i's CoordCount() coordinates start at
   *        joint_q + i * stride.
   * @param stride Floats from one instance to the next; 0: CoordCount()
   *        (packed). A larger stride reads instances embedded in a larger
   *        array in place (e.g. one articulation per world of a scene-wide
   *        joint_q).
   * @throws std::invalid_argument if num_states > Capacity() or 0 < stride <
   *         CoordCount().
   */
  void SetJointQ(const float *joint_q, size_t num_states, cudaStream_t stream, size_t stride = 0);

  /**
   * @brief Copies the coordinates of the first `num_states` instances into
   * `joint_q` (instance i at joint_q + i * stride; stride 0: CoordCount()),
   * leaving the floats between instances untouched. Asynchronous.
   * @throws std::invalid_argument if num_states > Capacity() or 0 < stride <
   *         CoordCount().
   */
  void GetJointQ(float *joint_q, size_t num_states, cudaStream_t stream, size_t stride = 0) const;

  /**
   * @brief Writes the world transform of every link of the first `num_states`
   * instances, from the cache (forward kinematics of the current joint_q): 7
   * floats per link, position then unit quaternion (x, y, z, w) as the factor
   * targets (Newton's body_q layout), links in topology order. Asynchronous.
   *
   * @param transforms Instance i's LinkCount() x 7 floats start at
   *        transforms + i * stride.
   * @param stride Floats from one instance to the next; 0: 7 x LinkCount()
   *        (packed). The floats between instances are left untouched.
   * @throws std::invalid_argument if num_states > Capacity() or 0 < stride <
   *         7 x LinkCount().
   */
  void GetLinkTransforms(float *transforms, size_t num_states, cudaStream_t stream,
                         size_t stride = 0) const;

  /**
   * @brief Box bounds per instance and DOF (device arrays of Capacity() x
   * TangentSize() floats, ±inf: unbounded; entries of BALL and FREE DOFs are
   * ignored). Not owned: they must outlive the solves. Both nullptr removes
   * the bounds.
   * @throws std::invalid_argument if exactly one of them is nullptr.
   */
  void SetBounds(const float *lower, const float *upper);

  /** @brief Restores the topology's joint limits as bounds (none if it has no finite limit). */
  void UseJointLimits();

  // --- StateBatch --------------------------------------------------------
  /** @brief Topology DOF count. */
  size_t TangentSize() const override { return static_cast<size_t>(topology_.DofCount()); }
  /** @brief Topology AmbientSize(): coordinates plus cache. */
  size_t AmbientSize() const override { return static_cast<size_t>(topology_.AmbientSize()); }
  /** @brief Active instances (SetNumActiveStates). */
  size_t NumActiveStates() const override { return num_active_states_; }
  /** @brief Instances the storage holds. */
  size_t Capacity() const override { return capacity_; }
  /** @brief Entries the constant-id buffer holds. */
  size_t ConstCapacity() const override { return const_capacity_; }
  /** @brief Device array of constant instance ids (nullptr without one). */
  const int *ConstStateIds() const override { return const_ids_; }
  /** @brief Active constant ids. */
  size_t NumConstStates() const override { return num_const_states_; }
  /** @brief See StateBatch::SetNumActiveStates. */
  void SetNumActiveStates(size_t num_active_states, size_t num_const_states = 0) override;
  /** @brief Instance `state_idx`'s storage (AmbientSize() floats), nullptr past the capacity. */
  float *StateDevicePtr(size_t state_idx) override;
  /** @brief Instance `state_idx`'s storage (AmbientSize() floats), nullptr past the capacity. */
  const float *StateDevicePtr(size_t state_idx) const override;
  /**
   * @brief x ⊕ δ per joint (see the class comment), clamping moved scalar
   * coordinates into the bounds, then forward kinematics into the cache of
   * the output. Replicas share the bounds.
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override;

  // --- BoxBoundedStates --------------------------------------------------
  /** @brief Device lower bounds (Capacity() x TangentSize()), nullptr without bounds. */
  const float *LowerBounds() const override { return lower_; }
  /** @brief Device upper bounds (Capacity() x TangentSize()), nullptr without bounds. */
  const float *UpperBounds() const override { return upper_; }
  /** @brief The coordinate of each scalar DOF, -1 for BALL and FREE DOFs (device). */
  const int *BoundedComponentOffsets() const override { return topology_.View().scalar_coord; }
  /** @brief Plus without clamping (numeric differentiation). */
  void PlusIgnoringBounds(const float *x, const float *delta, float *x_plus_delta,
                          cudaStream_t stream) override;

  // --- DerivedStateData --------------------------------------------------
  /** @brief Recomputes the cache of the active instances from their joint_q. */
  void RefreshDerivedData(cudaStream_t stream) override;

 private:
  /** @brief Shared constructor body: storage in the zero configuration, default bounds. */
  void Initialize(bool apply_joint_limits);

  const ArticulationTopology &topology_;
  size_t capacity_ = 0;
  size_t num_active_states_ = 0;
  const int *const_ids_ = nullptr;
  size_t const_capacity_ = 0;
  size_t num_const_states_ = 0;
  dvector<float> storage_;
  dvector<float> limit_lower_, limit_upper_;  ///< Topology limits per instance (owned).
  const float *lower_ = nullptr;              ///< Active bounds (owned or user arrays).
  const float *upper_ = nullptr;
};

}  // namespace cunls
