cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cunls LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_TESTING OFF)
option(BUILD_SHARED_LIBS "Build shared library (OFF for static)" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
include(cmake/AddSpdlog.cmake)
include(cmake/AddCUDSS.cmake)
include(cmake/BundleStaticDeps.cmake)

add_spdlog()
option(LOCAL_CUDSS_BUILD "Build cuDSS locally" ON)
set(CUDSS_BUILD_PATH "${PROJECT_SOURCE_DIR}/libs/cudss")

if(NOT LOCAL_CUDSS_BUILD)
  set(CUDSS_BUILD_PATH "")
endif()
add_cudss(CUDSS_BUILD_PATH "${CUDSS_BUILD_PATH}")

set(CUNLS_LIBS spdlog::spdlog CUDA::cusparse CUDA::cublas CUDA::cusolver cudss)

option(ENABLE_PROFILING "Enable profiling" OFF)
if(ENABLE_PROFILING)
  FetchContent_Declare(
    nvtx
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.3.0.zip
  )
  FetchContent_MakeAvailable(nvtx)

  set(CUNLS_LIBS ${CUNLS_LIBS} nvtx3-cpp)
endif()

include_directories(
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl
  ${PROJECT_SOURCE_DIR})

# Add submodules
add_subdirectory(cunls)

# Build library from object libraries (SHARED or STATIC per BUILD_SHARED_LIBS)
add_library(cunls
  $<TARGET_OBJECTS:cunls_common>
  $<TARGET_OBJECTS:cunls_factor>
  $<TARGET_OBJECTS:cunls_linear_solver>
  $<TARGET_OBJECTS:cunls_math>
  $<TARGET_OBJECTS:cunls_minimizer>
  $<TARGET_OBJECTS:cunls_state>
  $<TARGET_OBJECTS:cunls_robustifier>
)

target_link_libraries(cunls 
  PUBLIC CUDA::cudart
  PRIVATE ${CUNLS_LIBS}
)

# Static archives bundle spdlog and cuDSS (see bundle_static_dependencies below).
# CMake would otherwise propagate those PRIVATE deps on INTERFACE_LINK_LIBRARIES,
# forcing consumers to find spdlog/cudss targets and link them again — duplicate
# symbols and unnecessary dependencies. Only CUDA toolkit libs stay external.
if(NOT BUILD_SHARED_LIBS)
  if(ENABLE_PROFILING)
    set_property(TARGET cunls PROPERTY INTERFACE_LINK_LIBRARIES
      CUDA::cudart
      $<LINK_ONLY:CUDA::cusparse>
      $<LINK_ONLY:CUDA::cublas>
      $<LINK_ONLY:CUDA::cusolver>
      $<LINK_ONLY:nvtx3-cpp>
    )
  else()
    set_property(TARGET cunls PROPERTY INTERFACE_LINK_LIBRARIES
      CUDA::cudart
      $<LINK_ONLY:CUDA::cusparse>
      $<LINK_ONLY:CUDA::cublas>
      $<LINK_ONLY:CUDA::cusolver>
    )
  endif()
endif()

if(BUILD_SHARED_LIBS)
  set_target_properties(cunls PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
  set_target_properties(cunls PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  bundle_static_dependencies(cunls spdlog::spdlog cudss)
endif()

# C++ library install targets (skipped when only building Python wheel)
if(NOT BUILD_PYTHON_BINDINGS)
  install(DIRECTORY cunls/
    DESTINATION include/cunls
    FILES_MATCHING PATTERN "*.h"
  )
  install(TARGETS cunls
    EXPORT cunlsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
  )

  install(EXPORT cunlsTargets
    FILE cunlsTargets.cmake
    NAMESPACE cunls::
    DESTINATION lib/cmake/cunls
  )

  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/cunlsConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/cunlsConfig.cmake"
    @ONLY
  )
  install(FILES "${CMAKE_BINARY_DIR}/cunlsConfig.cmake"
    DESTINATION lib/cmake/cunls
  )
endif()

if(BUILD_TESTING)
  FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(
    nls_tests
    tests/utils.cpp
    tests/factor_batch_test.cpp
    tests/problem_test.cpp
    tests/prior_vector_prior_factor_test.cpp
    tests/residual_batch_test.cpp
    tests/loss_function_test.cpp
    tests/jacobian_ops_test.cpp
    tests/state_batch_ops_test.cpp
    tests/sparse_matrix_test.cpp
    tests/cusparse_matrix_multiplier_test.cpp
    tests/fast_matrix_multiplier_test.cpp
    tests/sparse_linear_solver_test.cpp
    tests/dense_linear_solver_test.cpp
    tests/gauss_newton_test.cpp
    tests/so2_manifold_test.cpp
    tests/se2_manifold_test.cpp
    tests/so3_manifold_test.cpp
    tests/se3_manifold_test.cpp
    tests/sim2_manifold_test.cpp
    tests/sim3_manifold_test.cpp
    tests/sl4_manifold_test.cpp
    tests/vector_manifold_test.cpp
    tests/lie_math_test.cu
    tests/lie_between_factors_smoke_test.cpp
    tests/point_to_plane_factor_batch_test.cpp
    tests/point_to_point_factor_batch_test.cpp
    tests/symmetric_point_to_plane_factor_batch_test.cpp
    tests/reprojection_factor_batch_test.cpp
    tests/pnp_factor_batch_test.cpp
    tests/decreasing_scale_minimizer_test.cpp
    tests/synthetic_pgo_test.cpp
    tests/dense_matrix_ops_test.cpp
    tests/utils_test.cpp
    tests/device_vector_test.cpp
    tests/sba_minimizer_test.cpp
    tests/pgo_minimizer_test.cpp
    tests/weighted_factor_batch_test.cpp
  )

  set_target_properties(nls_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  target_compile_definitions(nls_tests PRIVATE
    "CUNLS_TEST_DATA_DIR=\"${CMAKE_SOURCE_DIR}/tests/data\"")

  set(TEST_LIBS cunls GTest::gtest_main
    CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cudart)
  if(ENABLE_PROFILING)
    set(TEST_LIBS ${TEST_LIBS} nvtx3-cpp)
  endif()
  target_link_libraries(nls_tests PRIVATE ${TEST_LIBS})

  include(GoogleTest)
  gtest_discover_tests(nls_tests)
endif()

if(ENABLE_PROFILING)
  if(BUILD_TESTING)
    target_compile_definitions(nls_tests PRIVATE ENABLE_PROFILING)
  endif()
endif(ENABLE_PROFILING)

option(BUILD_PYTHON_BINDINGS "Build Python bindings via nanobind" OFF)
if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  target_compile_options(cunls PRIVATE -fprofile-arcs -ftest-coverage)
  target_compile_options(nls_tests PRIVATE -fprofile-arcs -ftest-coverage)
  target_link_libraries(nls_tests PRIVATE gcov)
  target_link_libraries(cunls PRIVATE gcov)
  set(REPORT_DIR ${CMAKE_BINARY_DIR}/bin/coverage_report)
  set(COVERAGE_FILE ${CMAKE_BINARY_DIR}/bin/coverage.info)
  add_custom_target(coverage
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${REPORT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${REPORT_DIR}
    COMMAND lcov -c -d . --base-directory ${CMAKE_SOURCE_DIR} --ignore-errors mismatch --ignore-errors source --output-file ${COVERAGE_FILE}
    COMMAND genhtml ${COVERAGE_FILE} -k --include=cunls --exclude=_deps --output-directory ${REPORT_DIR}
    DEPENDS nls_tests
    COMMENT "Generating code coverage report"
  )
endif()
