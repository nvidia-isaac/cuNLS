# Function to make the cuDSS headers available to the build
#
# Usage:
#   add_cudss(VERSION "0.8.0.10")
#
# This function downloads a prebuilt cuDSS archive (matching the host
# architecture and the CUDA major version) using FetchContent and creates an
# INTERFACE imported target 'cudss::headers' exposing only the cuDSS include
# directory.
#
# cuDSS is an OPTIONAL runtime dependency of cunls: cunls loads libcudss.so
# with dlopen() at runtime (see cunls/common/cudss_dynamic.h) instead of
# linking against it, so cunls binaries carry no link-time or load-time
# dependency on cuDSS. Only the headers are needed at compile time to declare
# the cuDSS types/functions used by cudss_dynamic.cpp and cudss_helper.cpp.
#
# The downloaded archive's lib/ directory (containing libcudss.so and
# libcudss_static.a) is exposed via CUDSS_LIB_DIR in the parent scope so
# callers can point LD_LIBRARY_PATH at it (e.g. for tests) or package it as a
# standalone artifact, separate from cunls's own binaries.
#
# Supported versions: 0.8.0.10 (default) and 0.7.1.4. The cuDSS API differs
# between 0.7.x and 0.8.x; the C++ sources select the right API based on the
# CUDSS_VERSION macro from the cuDSS header, so no extra compile definition is
# needed here.
#
# Parameters:
#   VERSION - cuDSS version to download (optional, defaults to 0.8.0.10)
function(add_cudss)
  # Parse arguments
  set(options "")
  set(oneValueArgs VERSION)
  set(multiValueArgs "")
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_VERSION)
    set(ARG_VERSION "0.8.0.10")
  endif()

  message(STATUS "Using prebuilt cuDSS ${ARG_VERSION} headers (loaded at runtime via dlopen)")
  set(CUDSS_URL_PREFIX "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/")
  message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

  # Archive flavor follows the CUDA major version.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    set(CUDSS_CUDA_TAG "cuda13")
    message(STATUS "Using CUDA 13.0 or newer")
  else()
    set(CUDSS_CUDA_TAG "cuda12")
    message(STATUS "Using CUDA 12.0 or older")
  endif()

  # The aarch64 archive naming changed between releases: 0.7.x ships as
  # "linux-aarch64", while 0.8+ ships as "linux-sbsa".
  if(${ARG_VERSION} VERSION_GREATER_EQUAL 0.8.0)
    set(CUDSS_AARCH64_NAME "linux-sbsa")
  else()
    set(CUDSS_AARCH64_NAME "linux-aarch64")
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)")
    set(CUDSS_URL "${CUDSS_URL_PREFIX}${CUDSS_AARCH64_NAME}/libcudss-${CUDSS_AARCH64_NAME}-${ARG_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  else()
    set(CUDSS_URL "${CUDSS_URL_PREFIX}linux-x86_64/libcudss-linux-x86_64-${ARG_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  endif()

  message(STATUS "cuDSS download URL: ${CUDSS_URL}")

  include(FetchContent)
  FetchContent_Declare(
    cudss
    URL ${CUDSS_URL}
  )
  FetchContent_MakeAvailable(cudss)

  add_library(cudss::headers INTERFACE IMPORTED)
  set_target_properties(cudss::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${cudss_SOURCE_DIR}/include"
  )

  # Exposed so the top-level build can package cuDSS's shared/static
  # libraries as a standalone artifact and so tests can add it to
  # LD_LIBRARY_PATH; cunls itself never links against these files.
  set(CUDSS_LIB_DIR "${cudss_SOURCE_DIR}/lib" PARENT_SCOPE)
  set(CUDSS_INCLUDE_DIR "${cudss_SOURCE_DIR}/include" PARENT_SCOPE)
endfunction()
