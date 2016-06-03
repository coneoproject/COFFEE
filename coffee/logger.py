# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2016, Imperial College London.
# Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""The COFFEE logger."""

import logging
import sys


logger = logging.getLogger('COFFEE')
_ch = logging.StreamHandler()
logger.addHandler(_ch)

# Add extra levels between INFO (value=20) and WARNING (value=30)
DEBUG = logging.DEBUG
INFO = logging.INFO
COST_MODEL = 21
PERF_OK = 28
PERF_WARN = 29
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(COST_MODEL, "COST_MODEL")
logging.addLevelName(PERF_OK, "PERF_OK")
logging.addLevelName(PERF_WARN, "PERF_WARN")

LOG_DEFAULT = PERF_OK
logger.setLevel(LOG_DEFAULT)

RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'

COLORS = {
    DEBUG: RED,
    INFO: GREEN,
    COST_MODEL: BLUE,
    PERF_OK: GREEN,
    PERF_WARN: BLUE,
    WARNING: BLUE,
    ERROR: RED,
    CRITICAL: RED
}


def set_log_level(level):
    """Set the log level of the COFFEE logger.

    :arg level: accepted values are: DEBUG, INFO, COST_MODEL, PERF_OK, PERF_WARN, WARNING,
        ERROR, CRITICAL
    """
    logger.setLevel(level)


def set_log_noperf():
    """Do not print performance-related messages."""
    logger.setLevel(WARNING)


def log(msg, level=INFO, *args, **kwargs):
    """Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    :arg msg: the message to be printed.
    :arg level: accepted values are: DEBUG, INFO, COST_MODEL, PERF_OK, PERF_WARN, WARNING,
        ERROR, CRITICAL
    """
    assert level in [DEBUG, INFO, COST_MODEL, PERF_OK, PERF_WARN, WARNING, ERROR, CRITICAL]

    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    log(msg, WARNING, *args, **kwargs)
