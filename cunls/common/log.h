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

#pragma once
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace cunls {

/**
 * @brief Log verbosity levels, ordered from least to most verbose.
 */
enum class Verbosity { Silent = 0, Error, Warning, Message, Debug };

/**
 * @brief Sets the verbosity of the global logger (console output, or the
 * SetLogCallback() callback).
 *
 * @param verbosity Maximum verbosity level to output (messages above this
 *                  level are suppressed).
 */
void SetLoggerOptions(Verbosity verbosity);

/**
 * @brief Receives every log message that passes the verbosity level
 * (SetLoggerOptions) instead of the console.
 *
 * May be called from any thread that runs cuNLS work; must not throw.
 */
using LogCallback = std::function<void(Verbosity verbosity, std::string_view msg)>;

/**
 * @brief Sends log messages to `callback` instead of the spdlog sinks (e.g.
 * to forward them to Python's logging module). An empty callback restores the
 * sinks. The verbosity level still filters what reaches the callback.
 */
void SetLogCallback(LogCallback callback);

/**
 * @brief Parses a verbosity name ("silent", "error", "warning", "message",
 * "debug"; case-insensitive, also "info" for message and "warn" for warning).
 *
 * @throws std::invalid_argument for any other name, listing the valid ones.
 */
Verbosity ParseVerbosity(std::string_view name);

/**
 * @brief Logs a message at the given verbosity level.
 *
 * Until SetLoggerOptions() is called, the level comes from the environment
 * variable CUNLS_LOG_LEVEL (a ParseVerbosity name), and is Silent without
 * it.
 *
 * @param verbosity The severity level of this message.
 * @param msg       The message string to log.
 */
void Log(Verbosity verbosity, std::string_view msg);

/**
 * @brief Logs a message at Error verbosity.
 * @param msg The error message to log.
 */
void LogError(std::string_view msg);

/**
 * @brief Logs a message at Warning verbosity.
 * @param msg The warning message to log.
 */
void LogWarning(std::string_view msg);

/**
 * @brief Logs a message at Message (informational) verbosity.
 * @param msg The informational message to log.
 */
void LogMessage(std::string_view msg);

/**
 * @brief Logs a message at Debug verbosity.
 * @param msg The debug message to log.
 */
void LogDebug(std::string_view msg);

namespace internal {

/**
 * @brief Writes @p value to @p out, formatted by @p spec, the part of a
 * replacement field after ':' ("" for none).
 *
 * Floating-point values take `.N` (precision) optionally followed by `f`
 * (fixed), `e` (scientific) or `g` (general), as std::format does; other specs
 * and other types print the value as `out << value`.
 */
template <typename T>
void WriteFormatted(std::ostringstream &out, const T &value, std::string_view spec) {
  if constexpr (std::is_floating_point_v<T>) {
    if (spec.size() >= 2 && spec[0] == '.') {
      size_t end = 1;
      int precision = 0;
      while (end < spec.size() && spec[end] >= '0' && spec[end] <= '9') {
        precision = 10 * precision + (spec[end++] - '0');
      }
      const char type = end < spec.size() ? spec[end] : 'g';
      if (end > 1 && end + (end < spec.size() ? 1 : 0) == spec.size() &&
          (type == 'f' || type == 'e' || type == 'g')) {
        const std::ios_base::fmtflags flags = out.flags();
        const std::streamsize old_precision = out.precision(precision);
        if (type == 'f') out << std::fixed;
        if (type == 'e') out << std::scientific;
        out << value;
        out.flags(flags);
        out.precision(old_precision);
        return;
      }
    }
  }
  out << value;
}

/** @brief Writes argument @p index of @p args to @p out (WriteFormatted). */
template <size_t... I, typename... Args>
void WriteArgument(std::ostringstream &out, size_t index, std::string_view spec,
                   std::index_sequence<I...>, const Args &...args) {
  ((I == index ? WriteFormatted(out, args, spec) : void()), ...);
}

/**
 * @brief Formats @p fmt with @p args, std::format style: `{}` takes the next
 * argument, `{N}` argument N, and either may carry a spec after ':' (see
 * WriteFormatted, e.g. `{:.2f}`). A field naming no argument is copied as is.
 */
template <typename... Args>
std::string Format(std::string_view fmt, const Args &...args) {
  std::ostringstream out;
  size_t next = 0;  // the argument of the next `{}`
  size_t pos = 0;
  while (pos < fmt.size()) {
    const size_t open = fmt.find('{', pos);
    const size_t close = open == std::string_view::npos ? open : fmt.find('}', open);
    if (close == std::string_view::npos) {
      out << fmt.substr(pos);
      break;
    }
    out << fmt.substr(pos, open - pos);
    const std::string_view field = fmt.substr(open + 1, close - open - 1);
    const size_t colon = field.find(':');
    const std::string_view id = field.substr(0, colon);
    const std::string_view spec =
        colon == std::string_view::npos ? std::string_view() : field.substr(colon + 1);
    size_t index = next;
    bool valid = true;
    if (id.empty()) {
      ++next;
    } else {
      index = 0;
      for (char c : id) {
        valid = valid && c >= '0' && c <= '9';
        index = 10 * index + static_cast<size_t>(c - '0');
      }
    }
    if (valid && index < sizeof...(Args)) {
      WriteArgument(out, index, spec, std::index_sequence_for<Args...>{}, args...);
    } else {
      out << '{' << field << '}';
    }
    pos = close + 1;
  }
  return out.str();
}

}  // namespace internal

/**
 * @brief Logs a formatted message at the given verbosity level.
 *
 * @tparam Args Argument types (anything with an `operator<<`).
 * @param verbosity The severity level of this message.
 * @param fmt       Format string with std::format-style fields: `{}`, `{N}`,
 *                  and a precision spec for floating-point values (`{:.2f}`).
 * @param args      Values to substitute into the format string.
 */
template <typename... Args>
inline void Log(Verbosity verbosity, std::string_view fmt, Args &&...args) {
  Log(verbosity, internal::Format(fmt, args...));
}

/**
 * @brief Logs a formatted message at Error verbosity.
 *
 * @tparam Args Variadic argument types.
 * @param fmt  Format string with {} placeholders.
 * @param args Values to substitute into the format string.
 */
template <typename... Args>
inline void LogError(std::string_view fmt, Args &&...args) {
  Log(Verbosity::Error, fmt, std::forward<Args>(args)...);
}

/**
 * @brief Logs a formatted message at Warning verbosity.
 *
 * @tparam Args Variadic argument types.
 * @param fmt  Format string with {} placeholders.
 * @param args Values to substitute into the format string.
 */
template <typename... Args>
inline void LogWarning(std::string_view fmt, Args &&...args) {
  Log(Verbosity::Warning, fmt, std::forward<Args>(args)...);
}

/**
 * @brief Logs a formatted message at Message (informational) verbosity.
 *
 * @tparam Args Variadic argument types.
 * @param fmt  Format string with {} placeholders.
 * @param args Values to substitute into the format string.
 */
template <typename... Args>
inline void LogMessage(std::string_view fmt, Args &&...args) {
  Log(Verbosity::Message, fmt, std::forward<Args>(args)...);
}

/**
 * @brief Logs a formatted message at Debug verbosity.
 *
 * @tparam Args Variadic argument types.
 * @param fmt  Format string with {} placeholders.
 * @param args Values to substitute into the format string.
 */
template <typename... Args>
inline void LogDebug(std::string_view fmt, Args &&...args) {
  Log(Verbosity::Debug, fmt, std::forward<Args>(args)...);
}
}  // namespace cunls
