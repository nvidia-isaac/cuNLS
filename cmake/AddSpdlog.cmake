# Fetches spdlog and builds it as a static library.
#
# Uses a function scope so that the BUILD_SHARED_LIBS override does not
# leak into the caller.  After the call, the spdlog::spdlog target is
# available globally.
function(add_spdlog)
  include(FetchContent)

  set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF)

  FetchContent_Declare(
    spdlog
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.zip
  )
  FetchContent_MakeAvailable(spdlog)
endfunction()
