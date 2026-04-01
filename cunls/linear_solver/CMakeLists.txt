add_library(cunls_linear_solver OBJECT
  cudss_sparse_linear_solver.cpp
  dense_linear_solver.cu
  sparse_linear_solver.cpp
)

target_link_libraries(cunls_linear_solver PRIVATE cudss)
