#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""Pipeline Recommender for the AI Pipeline plugin.

Telling the user the pipeline is imbalanced is only half the value;
identifying *which* hardware resource is the choke point turns an observation
into an action.  The same GPU starvation pattern can arise from three distinct
root causes — storage that cannot deliver data fast enough, CPU preprocessing
that cannot keep up with GPU consumption, or insufficient memory preventing
effective dataset caching — and each demands a different remedy.

Classification checks the most constrained resource first.  Memory pressure
takes priority because it compounds the other bottlenecks: a memory-bound
system will show elevated CPU and disk activity as a secondary consequence
of thrashing, not as independent root causes.
"""

from typing import Optional, Tuple

# Thresholds for bottleneck classification
_STARVATION_BALANCED_THRESHOLD = 10.0   # starvation_pct below this → balanced
_GPU_MEM_HIGH_THRESHOLD = 85.0          # gpu_mem_pct above this → memory_bound
_CPU_HIGH_THRESHOLD = 80.0              # cpu_percent above this → cpu_bound
_DISK_SAT_HIGH_THRESHOLD = 70.0         # disk_saturation_pct above this → disk_bound


class PipelineRecommender:
    """Translates imbalance metrics into an actionable bottleneck classification.

    Stateless by design so that each update tick produces an independent
    assessment without carrying forward stale state from prior ticks.
    """

    # Map bottleneck type → short recommendation (fits in a TUI line)
    _RECOMMENDATIONS = {
        'balanced':     'Pipeline balanced - no action needed',
        'disk_bound':   'Increase DataLoader workers & prefetch_factor',
        'cpu_bound':    'Add num_workers, enable pin_memory=True',
        'memory_bound': 'Reduce batch size or cache dataset to RAM',
    }

    def get_recommendation(
        self,
        starvation_pct: float,
        disk_saturation_pct: float,
        gpu_mem_pct: Optional[float] = None,
        cpu_percent: Optional[float] = None,
    ) -> Tuple[str, str]:
        """Identify the most probable root cause of the observed starvation.

        Args:
            starvation_pct: GPU starvation percentage (0–100) from
                GpuStarvationDetector.compute_starvation_pct().
            disk_saturation_pct: Disk I/O saturation (0–100) from
                GpuStarvationDetector.compute_disk_saturation_pct().
            gpu_mem_pct: GPU memory utilization (0–100) or None.
            cpu_percent: Overall CPU utilization (0–100) or None.

        Returns:
            Tuple of (bottleneck_type, recommendation_string).
        """
        bottleneck = self._classify(
            starvation_pct=starvation_pct,
            disk_saturation_pct=disk_saturation_pct,
            gpu_mem_pct=gpu_mem_pct,
            cpu_percent=cpu_percent,
        )
        return bottleneck, self._RECOMMENDATIONS[bottleneck]

    @staticmethod
    def _classify(
        starvation_pct: float,
        disk_saturation_pct: float,
        gpu_mem_pct: Optional[float],
        cpu_percent: Optional[float],
    ) -> str:
        """Map the current hardware readings to the most probable bottleneck type.

        Memory pressure is checked first because it degrades every other
        subsystem: a thrashing system will show high CPU and high disk activity
        as knock-on effects, not independent problems.  Attributing those
        symptoms to CPU or disk would lead to the wrong remedy.

        Disk-bound is the default when starvation is present but no other
        resource is clearly saturated, because storage is the most common
        cause of data pipeline imbalance in practice.
        """
        if starvation_pct < _STARVATION_BALANCED_THRESHOLD:
            return 'balanced'

        if gpu_mem_pct is not None and gpu_mem_pct > _GPU_MEM_HIGH_THRESHOLD:
            return 'memory_bound'

        if cpu_percent is not None and cpu_percent > _CPU_HIGH_THRESHOLD:
            return 'cpu_bound'

        if disk_saturation_pct > _DISK_SAT_HIGH_THRESHOLD:
            return 'disk_bound'

        # Default: disk pipeline is the most common culprit
        return 'disk_bound'
