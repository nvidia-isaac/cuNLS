/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

// Check for C++20 support
#if defined(__cpp_lib_format) && __cpp_lib_format >= 201907L
#include <format>
#define CPP_20_OR_GREATER
#endif

namespace cunls {

/**
 * @brief Log verbosity levels, ordered from least to most verbose.
 */
enum class Verbosity { Silent = 0, Error, Warning, Message, Debug };

/**
 * @brief Log output destinations.
 */
enum class Sink { Console = 0, File, ConsoleAndFile };

/**
 * @brief Configures the global logger settings.
 *
 * @param verbosity Maximum verbosity level to output (messages above this
 *                  level are suppressed).
 * @param sink      Where to send log output (console, file, or both).
 * @param file_path Path to the log file (required when sink includes File).
 */
void SetLoggerOptions(Verbosity verbosity, Sink sink = Sink::Console,
                      const std::string &file_path = "");

/**
 * @brief Logs a message at the given verbosity level.
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

#ifdef CPP_20_OR_GREATER
/**
 * @brief Logs a formatted message at the given verbosity level (C++20 path).
 *
 * Uses std::format-style placeholders (e.g. "{}", "{0}") for argument
 * substitution.
 *
 * @tparam Args Variadic argument types forwarded to std::vformat_to.
 * @param verbosity The severity level of this message.
 * @param fmt       Format string with {} placeholders.
 * @param args      Values to substitute into the format string.
 */
template <typename... Args>
inline void Log(Verbosity verbosity, std::string_view fmt, Args &&...args) {
  std::string buf;
  std::vformat_to(std::back_inserter(buf), fmt, std::make_format_args(args...));
  Log(verbosity, buf);
}
#else
namespace {
/**
 * @brief Writes the argument at the given tuple index to the stream.
 *
 * This is a helper for the C++17 formatting fallback. It selects the
 * argument at compile-time index @p Index if it matches @p target_index.
 *
 * @tparam Index  Compile-time tuple index to check.
 * @tparam Tuple  The tuple type holding all arguments.
 * @param oss          Output string stream to write into.
 * @param args         Tuple of all formatting arguments.
 * @param target_index Runtime index of the argument to output.
 */
template <size_t Index, typename Tuple>
void format_arg_at_index(std::ostringstream &oss, const Tuple &args,
                         size_t target_index) {
  if (Index == target_index && Index < std::tuple_size_v<Tuple>) {
    oss << std::get<Index>(args);
  }
}

/**
 * @brief Recursively dispatches to format_arg_at_index for all tuple indices.
 *
 * Uses fold expression to try each tuple index until the target is found.
 *
 * @tparam Indices  Index sequence matching tuple size.
 * @tparam Tuple    The tuple type holding all arguments.
 * @param oss          Output string stream to write into.
 * @param args         Tuple of all formatting arguments.
 * @param target_index Runtime index of the argument to output.
 */
template <size_t... Indices, typename Tuple>
void format_arg_recursive(std::ostringstream &oss, const Tuple &args,
                          size_t target_index,
                          std::index_sequence<Indices...>) {
  (format_arg_at_index<Indices>(oss, args, target_index), ...);
}

/**
 * @brief Simple format string parser for C++17 (fallback for pre-C++20).
 *
 * Replaces {} or {N} placeholders in the format string with the
 * corresponding argument values, similar to std::format.
 *
 * @tparam Args Variadic argument types.
 * @param fmt_str Format string with {} or {N} placeholders.
 * @param args    Values to substitute into the format string.
 * @return Formatted string with all placeholders replaced.
 */
template <typename... Args>
std::string vformat_to(std::string_view fmt_str, Args &&...args) {
  std::ostringstream result;
  std::string fmt(fmt_str);
  auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t num_args = sizeof...(Args);

  size_t pos = 0;
  size_t arg_index = 0;

  // Simple implementation: replace {} or {N} with arguments
  while (pos < fmt.length()) {
    size_t open_brace = fmt.find('{', pos);
    if (open_brace == std::string::npos) {
      result << fmt.substr(pos);
      break;
    }

    result << fmt.substr(pos, open_brace - pos);

    size_t close_brace = fmt.find('}', open_brace);
    if (close_brace == std::string::npos) {
      result << fmt.substr(open_brace);
      break;
    }

    // Extract placeholder content
    std::string placeholder =
        fmt.substr(open_brace + 1, close_brace - open_brace - 1);

    if (placeholder.empty()) {
      // Empty placeholder {} - use next argument in order
      if (arg_index < num_args) {
        format_arg_recursive(result, args_tuple, arg_index,
                             std::make_index_sequence<num_args>{});
        arg_index++;
      } else {
        result << "{}";
      }
    } else {
      // Try to parse as number
      size_t requested_idx = 0;
      bool is_valid_number = true;
      try {
        requested_idx = std::stoull(placeholder);
      } catch (const std::invalid_argument &) {
        is_valid_number = false;
      } catch (const std::out_of_range &) {
        is_valid_number = false;
      }

      if (is_valid_number && requested_idx < num_args) {
        format_arg_recursive(result, args_tuple, requested_idx,
                             std::make_index_sequence<num_args>{});
      } else {
        result << "{" << placeholder << "}";
      }
    }

    pos = close_brace + 1;
  }

  return result.str();
}
}  // namespace

/**
 * @brief Logs a formatted message at the given verbosity level (C++17 path).
 *
 * Uses the vformat_to fallback for {} placeholder substitution.
 *
 * @tparam Args Variadic argument types.
 * @param verbosity The severity level of this message.
 * @param fmt       Format string with {} placeholders.
 * @param args      Values to substitute into the format string.
 */
template <typename... Args>
inline void Log(Verbosity verbosity, std::string_view fmt, Args &&...args) {
  std::string buf = vformat_to(fmt, std::forward<Args>(args)...);
  Log(verbosity, buf);
}
#endif

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
