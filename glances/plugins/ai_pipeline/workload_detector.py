#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""AI Training Workload Detector for the AI Pipeline plugin.

Relying on hardware signals rather than process names means the detector
works for any training framework, custom launchers, or containerised jobs
where the visible process name gives no information about what the workload
is doing.  The co-occurrence of high GPU utilisation with high disk I/O is
the observable fingerprint of a data-loading pipeline actively feeding a
model, regardless of how that pipeline was started.

Hysteresis prevents the confirmed state from flickering in response to
short-lived dips (e.g. between epochs or during checkpoint saves) which
would produce misleading transitions in the TUI and confuse downstream logic.
"""

from typing import Optional


class AiWorkloadDetector:
    """Decides whether the system is currently running a heavy GPU+I/O workload.

    Hysteresis guards against noisy transitions: confirming too eagerly would
    activate the pipeline analysis on momentary spikes; clearing too eagerly
    would drop it during the inter-epoch pauses that are normal in training.
    """

    def __init__(
        self,
        detection_history: int = 3,
        gpu_active_threshold: float = 20.0,
        disk_active_bytes: float = 1_000_000.0,
    ):
        """
        Args:
            detection_history: How many consecutive agreeing ticks are needed
                before the state changes.  Longer values trade responsiveness
                for stability; shorter values react faster but may flip on noise.
            gpu_active_threshold: The minimum GPU utilisation that indicates
                the GPU is genuinely doing work rather than sitting in a driver
                or idle loop.
            disk_active_bytes: A noise floor for disk I/O below which the disk
                is considered quiescent.  Filesystem metadata activity and
                background sync would otherwise produce false positives.
        """
        self.detection_history = detection_history
        self.gpu_active_threshold = gpu_active_threshold
        self.disk_active_bytes = disk_active_bytes

        # Count of consecutive positive / negative detections
        self._positive_streak: int = 0
        self._negative_streak: int = 0

        # Whether a workload is currently confirmed
        self._confirmed: bool = False

    def detect(
        self,
        gpu_proc: Optional[float],
        disk_bytes_per_sec: float,
    ) -> bool:
        """Advance the detector by one monitoring tick and return the current state.

        Args:
            gpu_proc: Current GPU processor utilization (0–100 %), or None
                if no GPU is available.  If None, detection always fails.
            disk_bytes_per_sec: Total disk throughput in bytes/sec across
                all disks.

        Returns:
            True if a training-style workload is currently confirmed,
            False otherwise.
        """
        candidate = self._is_candidate(gpu_proc, disk_bytes_per_sec)

        if candidate:
            self._positive_streak += 1
            self._negative_streak = 0
        else:
            self._negative_streak += 1
            self._positive_streak = 0

        # Confirm after enough consecutive positive ticks
        if self._positive_streak >= self.detection_history:
            self._confirmed = True

        # Clear after enough consecutive negative ticks
        if self._negative_streak >= self.detection_history:
            self._confirmed = False

        return self._confirmed

    def _is_candidate(
        self,
        gpu_proc: Optional[float],
        disk_bytes_per_sec: float,
    ) -> bool:
        """Evaluate a single sample against the workload signature.

        Neither signal alone is sufficient: heavy disk I/O without GPU activity
        could be a backup job, and GPU activity without disk I/O could be an
        inference workload that streams data from memory.  Only their combination
        implies a training-style pipeline that is actively loading batches.
        """
        if gpu_proc is None:
            return False

        gpu_active = gpu_proc >= self.gpu_active_threshold
        disk_active = disk_bytes_per_sec >= self.disk_active_bytes

        return gpu_active and disk_active

    @property
    def is_confirmed(self) -> bool:
        """Return True if a training workload is currently confirmed."""
        return self._confirmed
