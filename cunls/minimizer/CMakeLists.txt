add_library(cunls_minimizer OBJECT
  gauss_newton_minimizer.cu
  jacobian_ops.cu
  levenberg_marquardt_minimizer.cpp
  minimizer_state.cu
  problem.cpp
  residual_batch.cu
  sparse_matrix.cu
  sparse_matrix_multiplication.cpp
)
