cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(cunls_install_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cunls REQUIRED)

add_executable(install_test main.cpp)
target_link_libraries(install_test PRIVATE cunls::cunls)
