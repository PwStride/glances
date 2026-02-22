#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""GPU stats global registry.

This module provides a global registry for GPU stats that can be shared
between the GPU plugin and GPU monitor plugin.
"""


class GpuList:
    """Global GPU stats list."""

    def __init__(self):
        """Init the GPU list."""
        self._gpu_stats = []

    def set(self, gpu_stats):
        """Set the GPU stats list."""
        self._gpu_stats = gpu_stats if gpu_stats else []

    def get(self):
        """Get the GPU stats list."""
        return self._gpu_stats


# Global GPU stats registry
glances_gpu = GpuList()
