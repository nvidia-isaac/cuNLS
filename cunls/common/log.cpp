/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cunls/common/log.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
namespace cunls {

/// Tracks whether the global logger has been initialized.
static std::atomic<bool> is_logger_initialized{false};

namespace {

/**
 * @brief Maps a cunls::Verbosity level to the corresponding spdlog level.
 *
 * @param verbosity The cunls verbosity level to convert.
 * @return The equivalent spdlog::level::level_enum value.
 */
spdlog::level::level_enum GetLogLevel(Verbosity verbosity) {
  spdlog::level::level_enum log_level;
  switch (verbosity) {
    case Verbosity::Error:
      log_level = spdlog::level::err;
      break;
    case Verbosity::Warning:
      log_level = spdlog::level::warn;
      break;
    case Verbosity::Message:
      log_level = spdlog::level::info;
      break;
    case Verbosity::Debug:
      log_level = spdlog::level::debug;
      break;
    case Verbosity::Silent:
      log_level = spdlog::level::off;
      break;
    default:
      log_level = spdlog::level::off;
      break;
  }
  return log_level;
}
}  // namespace

/** @copydoc SetLoggerOptions */
void SetLoggerOptions(Verbosity verbosity, Sink sink, const std::string &path) {
  is_logger_initialized.store(true, std::memory_order_release);
  spdlog::level::level_enum log_level = GetLogLevel(verbosity);
  if (sink == Sink::Console) {
    spdlog::set_level(log_level);
    return;
  }

  if (path.empty()) {
    spdlog::warn("Empty path provided to the logger. Using the default logger.");
    spdlog::set_level(log_level);
    return;
  }

  std::vector<spdlog::sink_ptr> sinks;

  if (sink == Sink::ConsoleAndFile) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path);
    sinks.push_back(file_sink);
  } else if (sink == Sink::File) {
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path);
    sinks.push_back(file_sink);
  }

  auto logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
  logger->set_level(log_level);

  spdlog::set_default_logger(logger);
}

namespace {
std::mutex callback_mutex;
std::shared_ptr<LogCallback> log_callback;  ///< SetLogCallback(); null: spdlog sinks.
}  // namespace

void SetLogCallback(LogCallback callback) {
  std::shared_ptr<LogCallback> replaced =
      callback ? std::make_shared<LogCallback>(std::move(callback)) : nullptr;
  {
    std::lock_guard<std::mutex> lock(callback_mutex);
    log_callback.swap(replaced);
  }
  // The old callback is released here, outside the lock: its destructor may take other locks
  // (e.g. the Python GIL) and must not do so while holding callback_mutex.
}

Verbosity ParseVerbosity(std::string_view name) {
  std::string lower(name);
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "silent" || lower == "off") return Verbosity::Silent;
  if (lower == "error") return Verbosity::Error;
  if (lower == "warning" || lower == "warn") return Verbosity::Warning;
  if (lower == "message" || lower == "info") return Verbosity::Message;
  if (lower == "debug") return Verbosity::Debug;
  throw std::invalid_argument("ParseVerbosity: unknown verbosity '" + std::string(name) +
                              "'; expected one of silent, error, warning, message, debug");
}

/**
 * @brief Logs a message at the specified verbosity via spdlog (or the
 * SetLogCallback() callback).
 *
 * Lazily initializes the logger from CUNLS_LOG_LEVEL (Silent without it) if it
 * hasn't been configured yet via SetLoggerOptions.
 */
void Log(Verbosity verbosity, std::string_view msg) {
  if (!is_logger_initialized.load(std::memory_order_acquire)) {
    Verbosity level = Verbosity::Silent;
    if (const char *env = std::getenv("CUNLS_LOG_LEVEL")) {
      try {
        level = ParseVerbosity(env);
      } catch (const std::invalid_argument &) {
        level = Verbosity::Warning;
        spdlog::warn(
            "CUNLS_LOG_LEVEL='{}' is not a level (silent, error, warning, message, "
            "debug); using warning",
            env);
      }
    }
    SetLoggerOptions(level);
  }

  spdlog::level::level_enum log_level = GetLogLevel(verbosity);
  std::shared_ptr<LogCallback> callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex);
    callback = log_callback;
  }
  if (callback) {
    if (spdlog::should_log(log_level)) (*callback)(verbosity, msg);
    return;
  }
  spdlog::log(log_level, msg);
}

/** @copydoc LogError(std::string_view) */
void LogError(std::string_view msg) { Log(Verbosity::Error, msg); }

/** @copydoc LogWarning(std::string_view) */
void LogWarning(std::string_view msg) { Log(Verbosity::Warning, msg); }

/** @copydoc LogMessage(std::string_view) */
void LogMessage(std::string_view msg) { Log(Verbosity::Message, msg); }

/** @copydoc LogDebug(std::string_view) */
void LogDebug(std::string_view msg) { Log(Verbosity::Debug, msg); }
}  // namespace cunls
