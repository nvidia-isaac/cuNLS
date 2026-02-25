cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cunls LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_TESTING OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
  spdlog
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.zip
)
FetchContent_MakeAvailable(spdlog)

include(cmake/AddCUDSS.cmake)
option(LOCAL_CUDSS_BUILD "Build cuDSS locally" ON)
set(CUDSS_BUILD_PATH "${PROJECT_SOURCE_DIR}/libs/cudss")

if(NOT LOCAL_CUDSS_BUILD)
  set(CUDSS_BUILD_PATH "")
endif()
add_cudss(CUDSS_BUILD_PATH "${CUDSS_BUILD_PATH}")

set(CUNLS_LIBS spdlog::spdlog cudart cusparse cublas cusolver cudss)

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

# Build shared library from object libraries
add_library(cunls SHARED
  $<TARGET_OBJECTS:cunls_common>
  $<TARGET_OBJECTS:cunls_factor>
  $<TARGET_OBJECTS:cunls_linear_solver>
  $<TARGET_OBJECTS:cunls_math>
  $<TARGET_OBJECTS:cunls_minimizer>
  $<TARGET_OBJECTS:cunls_state>
  $<TARGET_OBJECTS:cunls_robustifier>
)

target_link_libraries(cunls ${CUNLS_LIBS})
set_target_properties(cunls PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Install headers preserving directory structure
install(DIRECTORY cunls/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cunls
  FILES_MATCHING PATTERN "*.h"
)
install(TARGETS cunls DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if(BUILD_TESTING)
  FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(
    nls_tests
    tests/utils.cpp
    tests/factor_batch_test.cpp
    tests/vector_state_batch_test.cpp
    tests/problem_test.cpp
    tests/prior_vector_prior_factor_test.cpp
    tests/residual_batch_test.cpp
    tests/loss_function_test.cpp
    tests/jacobian_ops_test.cpp
    tests/state_batch_ops_test.cpp
    tests/sparse_matrix_test.cpp
    tests/sparse_matrix_multiplication_test.cpp
    tests/sparse_linear_solver_test.cpp
    tests/gauss_newton_test.cpp
    tests/se3_state_test.cpp
    tests/so3_state_test.cpp
    tests/so2_state_test.cpp
    tests/se2_prior_factor_test.cpp
    tests/so2_prior_factor_test.cpp
    tests/so3_prior_factor_test.cpp
    tests/se3_prior_factor_test.cpp
    tests/similarity2_prior_factor_test.cpp
    tests/similarity3_prior_factor_test.cpp
    tests/lie_math_test.cu
    tests/se3_between_factor_batch_test.cpp
    tests/point_to_plane_factor_batch_test.cpp
    tests/point_to_point_factor_batch_test.cpp
    tests/symmetric_point_to_plane_factor_batch_test.cpp
    tests/reprojection_factor_batch_test.cpp
    tests/decreasing_scale_minimizer_test.cpp
    tests/synthetic_pgo_test.cpp
    tests/dense_matrix_ops_test.cpp
    tests/utils_test.cpp
    tests/device_vector_test.cpp
  )

  set_target_properties(nls_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  set(TEST_LIBS cunls GTest::gtest_main)
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
