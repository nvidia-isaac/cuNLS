cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
# Native code for every GPU generation the toolkit supports (no PTX JIT at load time).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaArchitectures.cmake)

project(cunls_install_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cunls REQUIRED)

add_executable(install_test main.cpp)
target_link_libraries(install_test PRIVATE cunls::cunls)
