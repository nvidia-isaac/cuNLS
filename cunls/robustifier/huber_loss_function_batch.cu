
#include <stdint.h>

#include <cuda/std/limits>

#include "cunls/common/helper.h"
#include "cunls/robustifier/huber_loss_function_batch.h"

namespace cunls {

/** @brief CUDA block size for the Huber loss kernel. */
constexpr size_t block_size = 256;

/**
 * @brief CUDA kernel that computes the Huber loss for each squared residual.
 *
 * For each thread index tid < num_losses, computes the Huber loss triplet
 * (rho, rho', rho'') and writes it to out[tid].
 *
 * @param delta      Huber threshold parameter.
 * @param s          Device array of squared residuals (input).
 * @param out        Device array of float3 output triplets (rho, rho', rho'').
 * @param num_losses Total number of residuals.
 *
 * Grid/block: launched with ceil(num_losses / 256) blocks of 256 threads.
 */
__global__ void huber_loss_kernel(float delta, float* s, float3* out,
                                  int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  float delta_squared_ = delta * delta;

  float3& rho = out[tid];
  float sq_error = s[tid];

  if (sq_error > delta_squared_) {
    // Outlier region.
    // 'r' is always positive.
    const float r = sqrtf(sq_error);
    rho.x = 2.0 * delta * r - delta_squared_;
    rho.y = fmaxf(cuda::std::numeric_limits<float>::min(), delta / r);
    rho.z = -rho.y / (2.0 * sq_error);
  } else {
    // Inlier region.
    rho = {sq_error, 1.0, 0};
  }
}

/** @copydoc HuberLossFunctionBatch::HuberLossFunctionBatch */
HuberLossFunctionBatch::HuberLossFunctionBatch(float delta) : delta_(delta) {}

/** @copydoc HuberLossFunctionBatch::Evaluate */
bool HuberLossFunctionBatch::Evaluate(float* s, float3* out, int num_losses,
                                      cudaStream_t stream) const {
  if (num_losses <= 0) {
    return true;
  }
  size_t num_blocks = (num_losses + block_size - 1) / block_size;
  huber_loss_kernel<<<num_blocks, block_size, 0, stream>>>(delta_, s, out,
                                                           num_losses);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}
}  // namespace cunls
