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

"""Simple logging infrastructure."""

from collections import OrderedDict
import sys


class Logger(object):
    """Basic logging infrastructure."""

    LEVELS = OrderedDict([
        ('func_warning', '\033[1;37;31m%s\033[0m'),  # red
        ('info', '\033[1;37;32m%s\033[0m'),  # green
        ('perf_warning', '\033[1;37;34m%s\033[0m'),  # blue
        ('verbose', '\033[1;37;34m%s\033[0m'),  # blue
    ])

    DEFAULT = 'info'
    CURRENT = 'info'

    @staticmethod
    def out(output, level='info'):
        """Print ``output`` if ``level`` is below the system verbosity threshold."""
        assert level in Logger.LEVELS

        msg_value = Logger.LEVELS.keys().index(level)
        system_value = Logger.LEVELS.keys().index(Logger.CURRENT)

        # Colors if the terminal supports it (disabled e.g. when piped to file)
        if sys.stdout.isatty() and sys.stderr.isatty():
            color = Logger.LEVELS[level]
        else:
            color = "%s"

        if msg_value <= system_value:
            print (color % output)

    @staticmethod
    def current():
        """Get the current log level."""
        return Logger.CURRENT

    @staticmethod
    def default():
        """Get the default log level."""
        return Logger.DEFAULT

    @staticmethod
    def reset_level():
        """Set to the default log level."""
        Logger.CURRENT = Logger.DEFAULT

    @staticmethod
    def set_level(level):
        """Set a different log level."""
        assert level in Logger.LEVELS
        Logger.CURRENT = level

    @staticmethod
    def set_min_level():
        """Set to minimum log level."""
        Logger.CURRENT = 'func_warning'

    @staticmethod
    def set_max_level():
        """Set to maximum log level."""
        Logger.CURRENT = 'verbose'
