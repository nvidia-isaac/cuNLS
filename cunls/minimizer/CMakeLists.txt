add_library(cunls_minimizer OBJECT
  device_reduction.cu
  gauss_newton_minimizer.cu
  jacobian_ops.cu
  levenberg_marquardt_minimizer.cpp
  minimizer_state.cu
  problem.cpp
  residual_batch.cu
  sparse_matrix.cu
  cusparse_matrix_multiplier.cpp
  sparse_matrix_multiplier.cpp
  fast_matrix_multiplier.cu
)
