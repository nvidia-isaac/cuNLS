add_library(cunls_common OBJECT
  cublas_helper.cpp
  cuda_stream.cpp
  cudss_helper.cpp
  cusolver_helper.cpp
  cusparse_helper.cpp
  log.cpp
  profiler.cpp
  utils.cpp
)

target_link_libraries(cunls_common PRIVATE spdlog::spdlog cudss)

if(ENABLE_PROFILING)
  target_link_libraries(cunls_common PRIVATE nvtx3-cpp)
  target_compile_definitions(cunls_common PRIVATE ENABLE_PROFILING)
endif()
