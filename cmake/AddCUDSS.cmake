# Function to add cuDSS library to the project
#
# Usage:
#   add_cudss()
#
# This function downloads a prebuilt cuDSS release using FetchContent and
# creates an imported target 'cudss' that can be linked against. The cuDSS
# version and per-platform archive are selected automatically based on the
# CUDA compiler version and target architecture (x86_64 or aarch64/sbsa).
function(add_cudss)
  # cuDSS release version to download.
  set(CUDSS_VERSION "0.8.0.10")

  # Create imported target
  add_library(cudss STATIC IMPORTED)

  message(STATUS "Using prebuilt cuDSS ${CUDSS_VERSION}")
  set(CUDSS_URL_PREFIX "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/")
  message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    set(CUDSS_CUDA_TAG "cuda13")
    message(STATUS "Using CUDA 13.0 or newer")
  else()
    set(CUDSS_CUDA_TAG "cuda12")
    message(STATUS "Using CUDA 12.0 or older")
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)")
    set(CUDSS_URL "${CUDSS_URL_PREFIX}linux-sbsa/libcudss-linux-sbsa-${CUDSS_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  else()
    set(CUDSS_URL "${CUDSS_URL_PREFIX}linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  endif()

  include(FetchContent)
  FetchContent_Declare(
    cudss
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL ${CUDSS_URL}
  )
  FetchContent_Populate(cudss)

  set_target_properties(cudss PROPERTIES
    IMPORTED_LOCATION "${cudss_SOURCE_DIR}/lib/libcudss_static.a"
    INTERFACE_INCLUDE_DIRECTORIES "${cudss_SOURCE_DIR}/include"
  )
endfunction()
