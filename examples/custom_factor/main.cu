#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/sized_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "utils/validation.h"

using cunls::LogError;
using cunls::Vector;
using cunls::dvector;

namespace {

// ---------------------------------------------------------------------------
// Custom factor kernel
// ---------------------------------------------------------------------------
// This kernel implements a tiny 1D "between" constraint:
//   residual_i = (x_{i+1} - x_i) - measurement_i
//
// Each factor consumes two scalar state blocks:
//   state_pointers[2*i + 0] -> x_i
//   state_pointers[2*i + 1] -> x_{i+1}
//
// Jacobian layout is row-major per factor. Since residual dimension is 1 and
// state sizes are [1, 1], each factor contributes two Jacobian values:
//   [dr/dx_i, dr/dx_{i+1}] = [-1, +1]
__global__ void ScalarDifferenceKernel(const float* measurements,
                                       float const* const* state_pointers,
                                       float* residuals,
                                       float* jacobians,
                                       size_t num_factors) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_factors) {
    return;
  }

  const float* left = state_pointers[idx * 2];
  const float* right = state_pointers[idx * 2 + 1];
  const float residual = (right[0] - left[0]) - measurements[idx];

  if (residuals != nullptr) {
    residuals[idx] = residual;
  }
  if (jacobians != nullptr) {
    jacobians[idx * 2] = -1.0f;
    jacobians[idx * 2 + 1] = 1.0f;
  }
}

// ---------------------------------------------------------------------------
// Custom FactorBatch implementation
// ---------------------------------------------------------------------------
// SizedFactorBatch<1, 1, 1> means:
// - residual size: 1
// - first state block tangent size: 1
// - second state block tangent size: 1
//
// The class only stores pointers to device memory (measurements) and launches
// the kernel in Evaluate(). cuNLS handles assembly and optimization using
// the residuals/Jacobians we provide here.
class ScalarDifferenceFactorBatch : public cunls::SizedFactorBatch<1, 1, 1> {
 public:
  ScalarDifferenceFactorBatch(const float* measurements, size_t num_factors)
      : measurements_(measurements), num_factors_(num_factors) {}

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final {
    constexpr int kBlockSize = 256;
    const int grid_size = static_cast<int>((num_factors_ + kBlockSize - 1) / kBlockSize);
    ScalarDifferenceKernel<<<grid_size, kBlockSize, 0, stream>>>(
        measurements_, state_pointers, residuals, jacobians, num_factors_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
  }

  size_t NumFactors() const final { return num_factors_; }

 private:
  const float* measurements_;
  size_t num_factors_;
};

}  // namespace

int main() {
  try {
    // We model a chain of scalar states:
    //   x_0 -- x_1 -- ... -- x_{N-1}
    //
    // For N states we have N-1 custom "difference" factors.
    const size_t num_states = 256;
    const size_t num_diff_factors = num_states - 1;

    // Ground truth states, noisy initialization, and measured differences.
    std::vector<Vector<1>> gt_states(num_states);
    std::vector<Vector<1>> initial_states(num_states);
    std::vector<float> measurements(num_diff_factors);

    std::mt19937 rng(121314);
    std::uniform_real_distribution<float> step_dist(0.2f, 0.6f);
    std::uniform_real_distribution<float> noise_dist(-0.35f, 0.35f);

    // Create a monotonic synthetic trajectory.
    gt_states[0][0] = 0.5f;
    for (size_t i = 1; i < num_states; ++i) {
      gt_states[i][0] = gt_states[i - 1][0] + step_dist(rng);
    }

    // Disturb all states to create a non-trivial initial estimate.
    for (size_t i = 0; i < num_states; ++i) {
      initial_states[i][0] = gt_states[i][0] + noise_dist(rng);
    }

    // Measurements come from ground truth consecutive differences.
    for (size_t i = 0; i < num_diff_factors; ++i) {
      measurements[i] = gt_states[i + 1][0] - gt_states[i][0];
    }

    // Copy initial data to device.
    dvector<Vector<1>> states_device(initial_states);
    dvector<float> measurements_device(measurements);

    // Anchor x_0 to remove gauge freedom:
    // without this prior, adding a constant offset to all states leaves every
    // difference residual unchanged, so the system is rank-deficient.
    std::vector<Vector<1>> anchor_observation(1);
    anchor_observation[0][0] = gt_states[0][0];
    dvector<Vector<1>> anchor_observation_device(anchor_observation);

    // Build a single state batch containing all scalar states.
    const float* states_ptr = reinterpret_cast<const float*>(states_device.data());
    cunls::VectorStateBatch<1> state_batch(states_ptr, num_states);

    // Build:
    // - custom difference factors over edges (x_i, x_{i+1})
    // - one prior factor anchoring x_0
    ScalarDifferenceFactorBatch difference_factor(measurements_device.data(),
                                                  num_diff_factors);
    cunls::PriorVectorFactorBatch<1> anchor_factor(anchor_observation_device.data(), 1);

    // Create state pointer map for all custom factors.
    std::vector<float*> diff_state_pointers;
    diff_state_pointers.reserve(2 * num_diff_factors);
    for (size_t i = 0; i < num_diff_factors; ++i) {
      diff_state_pointers.push_back(state_batch.StateBlockDevicePtr(i));
      diff_state_pointers.push_back(state_batch.StateBlockDevicePtr(i + 1));
    }

    // State pointer map for the anchor factor: just x_0.
    std::vector<float*> anchor_state_pointers = {state_batch.StateBlockDevicePtr(0)};

    // Assemble the optimization problem graph.
    cunls::Problem problem;
    problem.AddStateBatch(&state_batch);
    problem.AddFactorBatch(&difference_factor, diff_state_pointers);
    problem.AddFactorBatch(&anchor_factor, anchor_state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // Levenberg-Marquardt options: fairly strict tolerances for this small
    // dense-in-logic but sparse-in-structure toy problem.
    cunls::MinimizerOptions options;
    options.max_num_iterations = 50;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;

    cunls::LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(lm_options);

    // Solve on CUDA stream, then synchronize before reading back outputs.
    cunls::CudaStream stream;
    const auto summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // Copy optimized states back to host and evaluate reconstruction quality.
    std::vector<Vector<1>> optimized_states(num_states);
    states_device.CopyToHost(optimized_states.data(), num_states);

    const float initial_mse = examples::ComputeVectorMSE(initial_states, gt_states);
    const float final_mse = examples::ComputeVectorMSE(optimized_states, gt_states);

    std::cout << "Custom Factor Example\n";
    std::cout << "  Initial cost: " << summary.initial_cost << "\n";
    std::cout << "  Final cost:   " << summary.final_cost << "\n";
    std::cout << "  Iterations:   " << summary.num_iterations << "\n";
    std::cout << "  State MSE:    " << initial_mse << " -> " << final_mse << "\n";

    if (summary.final_cost > 1e-5f || final_mse > initial_mse * 0.02f) {
      std::cerr << "Optimization quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
