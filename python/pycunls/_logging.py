# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forwarding of the native log to Python's :mod:`logging`."""

import atexit
import logging

from pycunls._pycunls_core import Verbosity, _set_log_callback, set_log_verbosity

# Verbosity value -> logging level.
_LEVELS = {
    int(Verbosity.Error): logging.ERROR,
    int(Verbosity.Warning): logging.WARNING,
    int(Verbosity.Message): logging.INFO,
    int(Verbosity.Debug): logging.DEBUG,
}


def log_to_python(enable=True, logger_name="pycunls", verbosity=None):
    """Send cuNLS log messages to the Python logger ``logger_name`` instead of the console.

    Every message then goes through :mod:`logging` (handlers, filters, formatting, capture by
    pytest's ``caplog``), at ERROR, WARNING, INFO (per-iteration lines and the final termination
    message) or DEBUG.

    Args:
        enable: ``False`` restores the native console output.
        logger_name: Name of the :class:`logging.Logger` that receives the messages.
        verbosity: Optional :class:`Verbosity` to set at the same time (the native level still
            filters what is forwarded; the default level is ``Silent`` or ``CUNLS_LOG_LEVEL``).

    Example::

        import logging, pycunls
        logging.basicConfig(level=logging.INFO)
        pycunls.log_to_python(verbosity=pycunls.Verbosity.Message)
    """
    if verbosity is not None:
        set_log_verbosity(verbosity)
    if not enable:
        _set_log_callback(None)
        return
    logger = logging.getLogger(logger_name)

    def forward(level, message):
        logger.log(_LEVELS.get(level, logging.INFO), message)

    _set_log_callback(forward)


# Drop the callback before the interpreter tears down.
atexit.register(_set_log_callback, None)
