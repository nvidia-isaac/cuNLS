add_library(cunls_linear_solver OBJECT
  block_sparse_pcg_solver.cu
  dense_cholesky_solver.cu
  dense_linear_solver.cu
  dense_linear_solver_base.cu
  dense_qr_solver.cu
  sparse_linear_solver.cpp
)

if(CUNLS_ENABLE_CUDSS)
  target_sources(cunls_linear_solver PRIVATE cudss_sparse_linear_solver.cpp)
  target_link_libraries(cunls_linear_solver PRIVATE cudss::headers)
  target_compile_definitions(cunls_linear_solver PRIVATE CUNLS_ENABLE_CUDSS)
endif()
