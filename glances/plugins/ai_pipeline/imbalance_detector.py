#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""GPU Starvation Detector for the AI Pipeline plugin.

GPU data starvation is the defining symptom of a pipeline imbalance: the GPU
finishes processing a batch and then sits idle waiting for the next one because
storage cannot keep pace.  A fixed-size rolling window is used so that metrics
always reflect recent behaviour rather than lifetime averages, making the
detector responsive to workload changes without retaining unbounded history.

A separate, longer peak window tracks the highest observed disk throughput so
that saturation can be expressed relative to what the hardware has demonstrated
it can deliver, avoiding false alarms on systems with inherently modest I/O.
"""

import math
from collections import deque
from time import time
from typing import Tuple


class GpuStarvationDetector:
    """Measures how often the GPU is being kept waiting by the data pipeline.

    A short recent window keeps the metrics actionable; stale samples from
    before a workload change would dilute the signal and delay user awareness.
    """

    def __init__(
        self,
        window_size: int = 30,
        gpu_idle_threshold: float = 15.0,
        disk_saturation_threshold: float = 80.0,
        peak_window_size: int = 300,
    ):
        """
        Args:
            window_size: How far back to look when scoring the current pipeline
                health.  Shorter windows react faster; longer windows smooth noise.
            gpu_idle_threshold: Below this utilisation the GPU is contributing
                nothing to the workload, making every idle tick a wasted cycle.
            disk_saturation_threshold: How close to the disk's demonstrated peak
                throughput counts as "saturated" — the point at which storage
                cannot serve data any faster without a hardware change.
            peak_window_size: Retains enough history to capture the disk's best
                sustained throughput without growing memory unboundedly.
        """
        self.window_size = window_size
        self.gpu_idle_threshold = gpu_idle_threshold
        self.disk_saturation_threshold = disk_saturation_threshold

        # Sliding window: (timestamp, gpu_proc, disk_bytes_per_sec)
        self._window: deque = deque(maxlen=window_size)

        # Longer window for estimating peak (rolling max) disk bandwidth
        self._peak_window: deque = deque(maxlen=peak_window_size)

    def add_sample(self, gpu_proc: float, disk_bytes_per_sec: float) -> None:
        """Accept the latest hardware readings for this tick.

        Args:
            gpu_proc: Current GPU processor utilization (0–100 %).
            disk_bytes_per_sec: Total disk throughput in bytes/sec (all disks combined).
        """
        now = time()
        self._window.append((now, gpu_proc, disk_bytes_per_sec))
        self._peak_window.append(disk_bytes_per_sec)

    def _peak_disk_bytes(self) -> float:
        """Normalisation anchor for saturation scores.

        Using an observed peak rather than a configured constant means the
        detector self-calibrates to the actual hardware without manual tuning.
        """
        if not self._peak_window:
            return 1.0  # Avoid divide-by-zero; caller treats 0 as 0 %
        return max(self._peak_window)

    def compute_starvation_pct(self) -> float:
        """Proportion of recent time the GPU was being actively starved.

        Both conditions must be true simultaneously: an idle GPU alone could
        mean the workload is simply paused, and a saturated disk alone means
        heavy I/O without a corresponding GPU workload.  Only the co-occurrence
        reveals a pipeline that is failing to keep the GPU fed.

        Returns:
            Fraction of window samples that are starvation events, 0–100 %.
        """
        if not self._window:
            return 0.0

        peak = self._peak_disk_bytes()
        sat_bytes = peak * (self.disk_saturation_threshold / 100.0)

        starvation_count = sum(
            1
            for _, gpu_proc, disk_bytes in self._window
            if gpu_proc < self.gpu_idle_threshold and disk_bytes >= sat_bytes
        )

        return starvation_count / len(self._window) * 100.0

    def compute_disk_saturation_pct(self) -> float:
        """How hard the storage subsystem is being driven right now.

        Expressed relative to the observed peak so that the number is
        meaningful across different storage tiers without configuration.

        Returns:
            Mean disk throughput in the window expressed as % of rolling peak.
        """
        if not self._window:
            return 0.0

        peak = self._peak_disk_bytes()
        if peak <= 0:
            return 0.0

        mean_disk = sum(d for _, _, d in self._window) / len(self._window)
        return min(100.0, mean_disk / peak * 100.0)

    def compute_gpu_cv(self) -> float:
        """How erratic the GPU utilisation has been over the recent window.

        A GPU that is being starved of data produces a distinctive sawtooth
        pattern — high utilisation during batch processing, then near-zero
        while waiting for the next batch.  High variance relative to the mean
        is the statistical signature of that pattern, distinguishing it from
        a GPU that is simply lightly loaded at a stable low level.

        Returns:
            CV as a percentage, or 0 if insufficient data.
        """
        if len(self._window) < 2:
            return 0.0

        gpu_samples = [g for _, g, _ in self._window]
        mean = sum(gpu_samples) / len(gpu_samples)

        if mean <= 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in gpu_samples) / len(gpu_samples)
        stdev = math.sqrt(variance)

        return (stdev / mean) * 100.0

    def compute_pipeline_score(self) -> float:
        """A single number the user can act on: how healthy is the pipeline right now.

        Inverting starvation into a positive score makes it intuitive — higher
        is always better — and consistent with how other Glances health metrics
        are displayed.

        Returns:
            Score in [0, 100].
        """
        return max(0.0, 100.0 - self.compute_starvation_pct())

    def get_window_stats(self) -> Tuple[float, float, float, float]:
        """Fetch all four pipeline health metrics in a single call.

        Callers that need the full picture can avoid four separate calls and
        the minor inconsistency that would arise if the window advanced between them.
        """
        return (
            self.compute_pipeline_score(),
            self.compute_starvation_pct(),
            self.compute_disk_saturation_pct(),
            self.compute_gpu_cv(),
        )
