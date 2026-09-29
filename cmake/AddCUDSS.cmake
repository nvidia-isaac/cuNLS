# Function to make the cuDSS headers available to the build
#
# Usage:
#   add_cudss(VERSION "0.8.0.10" PLATFORM "auto")
#
# This function downloads a prebuilt cuDSS archive (matching the CUDA Toolkit
# target platform and CUDA major version) using FetchContent and creates an
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
# standalone artifact, separate from cunls's own binaries. The headers are
# platform-independent, but these libraries are not, so the archive must match
# the target platform.
#
# Supported versions: 0.8.0.10 (default) and 0.7.1.4. The cuDSS API differs
# between 0.7.x and 0.8.x; the C++ sources select the right API based on the
# CUDSS_VERSION macro from the cuDSS header, so no extra compile definition is
# needed here.
#
# Parameters:
#   VERSION  - cuDSS version to download (optional, defaults to 0.8.0.10)
#   PLATFORM - cuDSS redistribution platform: auto, linux-x86_64,
#              linux-aarch64 (Jetson), or linux-sbsa (Arm servers and CUDA 13
#              Jetson). auto (the default) uses the CUDA Toolkit target.
function(add_cudss)
  set(options "")
  set(oneValueArgs VERSION PLATFORM)
  set(multiValueArgs "")
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_VERSION)
    set(ARG_VERSION "0.8.0.10")
  endif()
  if(NOT ARG_PLATFORM)
    set(ARG_PLATFORM "auto")
  endif()

  set(CUDSS_CUDA_TAG "cuda${CUDAToolkit_VERSION_MAJOR}")

  # cuDSS archives are published per CUDA Toolkit target, and each toolkit
  # installs its libraries under targets/<target>-linux: x86_64-linux,
  # aarch64-linux (Jetson), or sbsa-linux (Arm servers and CUDA 13 Jetson).
  # Reading the target from the toolkit layout avoids guessing Jetson vs SBSA
  # from the processor name, which is aarch64 for both.
  set(_cudss_platform "${ARG_PLATFORM}")
  if(_cudss_platform STREQUAL "auto")
    foreach(_cudss_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
      file(REAL_PATH "${_cudss_include_dir}" _cudss_include_dir)
      if(_cudss_include_dir MATCHES "/targets/(x86_64|aarch64|sbsa)-linux/include$")
        set(_cudss_platform "linux-${CMAKE_MATCH_1}")
        break()
      endif()
    endforeach()
    # Scattered installations (e.g. distro packages under /usr) have no targets/<target>-linux directory.
    # x86_64 has a single cuDSS platform, but aarch64 could be either Jetson or SBSA.
    if(_cudss_platform STREQUAL "auto")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_cudss_platform "linux-x86_64")
      else()
        message(FATAL_ERROR "Cannot determine the CUDA Toolkit target from '${CUDAToolkit_INCLUDE_DIRS}'. "
                            "Set CUDSS_PLATFORM to linux-aarch64 (Jetson) or linux-sbsa.")
      endif()
    endif()
  endif()

  set(CUDSS_URL "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/${_cudss_platform}/")
  string(APPEND CUDSS_URL "libcudss-${_cudss_platform}-${ARG_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  message(STATUS "Using prebuilt cuDSS ${ARG_VERSION} headers (loaded at runtime via dlopen)")
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
