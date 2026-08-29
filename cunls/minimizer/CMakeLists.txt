add_library(cunls_minimizer OBJECT
  block_hessian_assembler.cu
  bsr_matrix.cu
  hessian_structure.cu
  device_reduction.cu
  gauss_newton_minimizer.cu
  levenberg_marquardt_minimizer.cpp
  minimizer_state.cu
  normal_equations.cu
  problem.cpp
  residual_batch.cu
  sparse_matrix.cu
)
