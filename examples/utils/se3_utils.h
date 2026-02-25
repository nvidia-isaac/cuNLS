#pragma once

#include <random>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/lie_math.h"

namespace examples {

using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;
using cunls::dvector;

// Generic 4x4 matrix multiply used for SE(3) composition on the host.
inline SE3Transform ComposeSE3(const SE3Transform& a, const SE3Transform& b) {
  SE3Transform out;
  out.fill(0.0f);
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      for (int k = 0; k < 4; ++k) {
        out[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
      }
    }
  }
  return out;
}

// Host-side inverse of an SE(3) homogeneous transform:
// [R t; 0 1]^{-1} = [R^T  -R^T t; 0  1]
inline SE3Transform InverseSE3(const SE3Transform& pose) {
  SE3Transform inv;
  inv.fill(0.0f);

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      inv[row * 4 + col] = pose[col * 4 + row];
    }
  }
  for (int row = 0; row < 3; ++row) {
    float t = 0.0f;
    for (int col = 0; col < 3; ++col) {
      t += inv[row * 4 + col] * pose[col * 4 + 3];
    }
    inv[row * 4 + 3] = -t;
  }
  inv[15] = 1.0f;
  return inv;
}

// Apply the SE(3) exponential map to a batch of twist vectors on the GPU.
// Low-level building block: callers supply pre-built twists.
inline void TwistsToSE3(const std::vector<Vector<6>>& twists,
                         std::vector<SE3Transform>& poses) {
  const size_t n = twists.size();
  dvector<Vector<6>> twists_device(twists);
  dvector<SE3Transform> poses_device(n);
  cunls::CudaStream stream;
  cunls::ComputeExpSE3(stream.GetStream(),
                       reinterpret_cast<const float*>(twists_device.data()), 6,
                       4, 16, n,
                       reinterpret_cast<float*>(poses_device.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  poses.resize(n);
  poses_device.CopyToHost(poses.data(), n);
}

// Generate random SE(3) transforms by sampling twists from uniform
// distributions and applying the exponential map on the GPU.
inline void GenerateRandomSE3(size_t n, std::mt19937& rng,
                               std::vector<SE3Transform>& poses,
                               float rot_range = 0.4f,
                               float trans_range = 2.0f) {
  std::uniform_real_distribution<float> rot_dist(-rot_range, rot_range);
  std::uniform_real_distribution<float> trans_dist(-trans_range, trans_range);

  std::vector<Vector<6>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    twists[i][0] = rot_dist(rng);
    twists[i][1] = rot_dist(rng);
    twists[i][2] = rot_dist(rng);
    twists[i][3] = trans_dist(rng);
    twists[i][4] = trans_dist(rng);
    twists[i][5] = trans_dist(rng);
  }

  TwistsToSE3(twists, poses);
}

}  // namespace examples
