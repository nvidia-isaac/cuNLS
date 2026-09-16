# Function to add cuDSS library to the project
#
# Usage:
#   add_cudss(VERSION "0.8.0.10" PLATFORM "auto")
#
# This function downloads a prebuilt cuDSS archive (matching the host
# platform and the CUDA major version) using FetchContent and creates an imported
# target 'cudss' that can be linked against.
#
# Supported versions: 0.8.0.10 (default) and 0.7.1.4. The cuDSS API differs
# between 0.7.x and 0.8.x; the C++ sources select the right API based on the
# CUDSS_VERSION macro from the cuDSS header, so no extra compile definition is
# needed here.
#
# Parameters:
#   VERSION - cuDSS version to download (optional, defaults to 0.8.0.10)
#   PLATFORM - Redistribution platform: auto, linux-x86_64, linux-aarch64, or
#              linux-sbsa (optional, defaults to auto)
function(add_cudss)
  # Parse arguments
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

  add_library(cudss STATIC IMPORTED)

  message(STATUS "Using prebuilt cuDSS ${ARG_VERSION}")
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

  string(TOLOWER "${ARG_PLATFORM}" _cudss_platform)
  set(_cudss_supported_platforms auto linux-x86_64 linux-aarch64 linux-sbsa)
  if(NOT _cudss_platform IN_LIST _cudss_supported_platforms)
    message(FATAL_ERROR "Unsupported cuDSS platform '${ARG_PLATFORM}'. "
                        "Expected one of: ${_cudss_supported_platforms}")
  endif()

  if(_cudss_platform STREQUAL "auto")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
      set(_cudss_is_jetson FALSE)
      if(NOT CMAKE_CROSSCOMPILING AND EXISTS "/etc/nv_tegra_release")
        set(_cudss_is_jetson TRUE)
      endif()
      if(CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)(72|87|110)(-real|-virtual)?(;|$)")
        set(_cudss_is_jetson TRUE)
      endif()

      # NVIDIA publishes linux-aarch64 archives for all cuDSS 0.7 variants and
      # for cuDSS 0.8 with CUDA 12. CUDA 13 cuDSS 0.8 only provides SBSA on Arm.
      if(_cudss_is_jetson AND
         (CUDSS_CUDA_TAG STREQUAL "cuda12" OR ARG_VERSION VERSION_LESS 0.8.0))
        set(_cudss_platform "linux-aarch64")
      else()
        set(_cudss_platform "linux-sbsa")
      endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
      set(_cudss_platform "linux-x86_64")
    else()
      message(FATAL_ERROR "Cannot select a cuDSS archive for processor '${CMAKE_SYSTEM_PROCESSOR}'. "
                          "Set CUDSS_PLATFORM explicitly.")
    endif()
  endif()

  if(_cudss_platform STREQUAL "linux-aarch64" AND
     CUDSS_CUDA_TAG STREQUAL "cuda13" AND
     ARG_VERSION VERSION_GREATER_EQUAL 0.8.0)
    message(FATAL_ERROR "cuDSS ${ARG_VERSION} does not publish a linux-aarch64 CUDA 13 archive. "
                        "Use CUDSS_PLATFORM=linux-sbsa if that binary is valid for the target.")
  endif()

  set(CUDSS_URL
      "${CUDSS_URL_PREFIX}${_cudss_platform}/libcudss-${_cudss_platform}-${ARG_VERSION}_${CUDSS_CUDA_TAG}-archive.tar.xz")
  message(STATUS "cuDSS platform: ${_cudss_platform}")
  message(STATUS "cuDSS download URL: ${CUDSS_URL}")

  include(FetchContent)
  FetchContent_Declare(
    cudss
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL ${CUDSS_URL}
  )
  FetchContent_MakeAvailable(cudss)

  set_target_properties(cudss PROPERTIES
    IMPORTED_LOCATION "${cudss_SOURCE_DIR}/lib/libcudss_static.a"
    INTERFACE_INCLUDE_DIRECTORIES "${cudss_SOURCE_DIR}/include"
  )
endfunction()
