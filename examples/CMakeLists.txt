cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(cunls_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CUNLS_INSTALL_DIR)
  message(FATAL_ERROR "Set CUNLS_INSTALL_DIR to the cuNLS install prefix.")
endif()

set(CUNLS_INCLUDE_DIR "${CUNLS_INSTALL_DIR}/include")
set(CUNLS_LIBRARY_DIR "${CUNLS_INSTALL_DIR}/lib")

find_package(CUDAToolkit REQUIRED)
find_library(CUNLS_LIBRARY cunls PATHS "${CUNLS_LIBRARY_DIR}" REQUIRED NO_DEFAULT_PATH)

function(add_cunls_example target source_file)
  add_executable(${target} ${source_file})
  target_include_directories(${target} PRIVATE
    "${CUNLS_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(${target} PRIVATE "${CUNLS_LIBRARY}" CUDA::cudart)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${CUNLS_LIBRARY_DIR}"
    INSTALL_RPATH "${CUNLS_LIBRARY_DIR}"
  )
endfunction()

add_cunls_example(
  sparse_bundle_adjustment_example
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse_bundle_adjustment/main.cpp"
)

add_cunls_example(
  pose_graph_optimization_example
  "${CMAKE_CURRENT_SOURCE_DIR}/pose_graph_optimization/main.cpp"
)

add_cunls_example(
  custom_factor_example
  "${CMAKE_CURRENT_SOURCE_DIR}/custom_factor/main.cu"
)
