add_library(cunls_common OBJECT
  cublas_helper.cpp
  cuda_stream.cpp
  cusolver_helper.cpp
  cusparse_helper.cpp
  log.cpp
  profiler.cpp
  utils.cpp
)

target_link_libraries(cunls_common PRIVATE spdlog::spdlog)

# cuDSS is loaded at runtime via dlopen() (see cudss_dynamic.cpp), so only
# its headers are needed at compile time; CMAKE_DL_LIBS provides dlopen/dlsym.
if(CUNLS_ENABLE_CUDSS)
  target_sources(cunls_common PRIVATE
    cudss_dynamic.cpp
    cudss_helper.cpp
  )
  target_link_libraries(cunls_common PRIVATE cudss::headers ${CMAKE_DL_LIBS})
endif()

if(ENABLE_PROFILING)
  target_link_libraries(cunls_common PRIVATE nvtx3-cpp)
  target_compile_definitions(cunls_common PRIVATE ENABLE_PROFILING)
endif()
