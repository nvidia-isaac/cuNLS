add_library(cunls_linear_solver OBJECT
  block_sparse_pcg_solver.cu
  cudss_sparse_linear_solver.cpp
  dense_cholesky_solver.cu
  dense_linear_solver.cu
  dense_qr_solver.cu
  sparse_linear_solver.cpp
)

target_link_libraries(cunls_linear_solver PRIVATE cudss)

if(LOCAL_CUDSS_BUILD)
  target_compile_definitions(cunls_linear_solver PRIVATE CUDSS_NEW_API)
endif()
