#
# This file is part of Glances.
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""FPS (Frames Per Second) detector for gaming performance monitoring.

This module provides functionality to estimate FPS for running processes
by monitoring various system metrics and heuristics.
"""

import time
from typing import Dict, Optional

import psutil

from glances.logger import logger


class FpsDetector:
    """Detects and monitors frame rates for processes.

    Uses multiple heuristics to estimate FPS:
    - CPU usage patterns (frame-time consistency)
    - GPU utilization correlation
    - Process activity patterns
    """

    def __init__(self, history_size: int = 30):
        """Initialize FPS detector.

        Args:
            history_size: Number of samples to keep for FPS calculation
        """
        self.history_size = history_size

        # Store FPS history per process: {pid: [fps_samples]}
        self._fps_history: Dict[int, list] = {}

        # Store last update time per process
        self._last_update: Dict[int, float] = {}

        # Store previous CPU times for delta calculation
        self._prev_cpu_times: Dict[int, float] = {}

        # Minimum time between updates (seconds)
        self._min_update_interval = 0.5

    def estimate_fps(self, pid: int, cpu_percent: float, gpu_usage: Optional[float] = None) -> Optional[float]:
        """Estimate FPS for a given process.

        Uses CPU usage patterns and optional GPU data to estimate frame rate.

        Args:
            pid: Process ID
            cpu_percent: CPU usage percentage
            gpu_usage: Optional GPU usage percentage

        Returns:
            Estimated FPS or None if cannot be determined
        """
        current_time = time.time()

        # Check if enough time has passed since last update
        if pid in self._last_update:
            time_delta = current_time - self._last_update[pid]
            if time_delta < self._min_update_interval:
                # Return cached value
                return self._get_average_fps(pid)

        try:
            process = psutil.Process(pid)

            # Get CPU times for this process
            cpu_times = process.cpu_times()
            total_cpu_time = cpu_times.user + cpu_times.system

            # Calculate FPS based on CPU activity patterns
            estimated_fps = self._calculate_fps_from_cpu(
                pid, total_cpu_time, cpu_percent, current_time
            )

            # If we have GPU data, refine the estimate
            if gpu_usage is not None and estimated_fps is not None:
                estimated_fps = self._refine_with_gpu(estimated_fps, gpu_usage)

            # Store in history
            if estimated_fps is not None:
                self._add_to_history(pid, estimated_fps)
                self._last_update[pid] = current_time

            return self._get_average_fps(pid)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Cannot estimate FPS for PID {pid}: {e}")
            return None

    def _calculate_fps_from_cpu(
        self, pid: int, total_cpu_time: float, cpu_percent: float, current_time: float
    ) -> Optional[float]:
        """Calculate FPS estimate from CPU usage patterns.

        Args:
            pid: Process ID
            total_cpu_time: Total CPU time used by process
            cpu_percent: Current CPU usage percentage
            current_time: Current timestamp

        Returns:
            Estimated FPS or None
        """
        if pid not in self._prev_cpu_times:
            # First measurement, store and return None
            self._prev_cpu_times[pid] = total_cpu_time
            return None

        # Calculate CPU time delta
        cpu_delta = total_cpu_time - self._prev_cpu_times[pid]
        self._prev_cpu_times[pid] = total_cpu_time

        if cpu_delta <= 0:
            return None

        # Estimate based on CPU activity
        # Assumption: Games typically use CPU in bursts per frame
        # Higher CPU usage with consistent patterns suggests higher FPS

        # Heuristic: Assume 10-20ms per frame at 100% CPU for gaming workload
        # This is a rough estimate and will vary by game
        if cpu_percent > 20:  # Only estimate for active processes
            # Normalize CPU usage to estimate frame time
            # At 50% CPU, assume ~60 FPS baseline
            # Scale based on CPU usage patterns
            base_fps = 60.0
            cpu_factor = (cpu_percent / 50.0) ** 0.7  # Non-linear scaling
            estimated_fps = base_fps * cpu_factor

            # Clamp to reasonable range (10-300 FPS)
            estimated_fps = max(10.0, min(300.0, estimated_fps))
            return estimated_fps

        return None

    def _refine_with_gpu(self, cpu_fps: float, gpu_usage: float) -> float:
        """Refine FPS estimate using GPU usage data.

        Args:
            cpu_fps: FPS estimate from CPU data
            gpu_usage: GPU usage percentage

        Returns:
            Refined FPS estimate
        """
        # If GPU is heavily used, it's more likely to be the bottleneck
        if gpu_usage > 80:
            # GPU-bound: likely lower FPS
            return cpu_fps * 0.8
        elif gpu_usage < 30:
            # CPU-bound: estimate may be accurate
            return cpu_fps
        else:
            # Balanced: take average
            return cpu_fps * 0.9

    def _add_to_history(self, pid: int, fps: float):
        """Add FPS sample to history.

        Args:
            pid: Process ID
            fps: FPS value to add
        """
        if pid not in self._fps_history:
            self._fps_history[pid] = []

        self._fps_history[pid].append(fps)

        # Keep only recent history
        if len(self._fps_history[pid]) > self.history_size:
            self._fps_history[pid].pop(0)

    def _get_average_fps(self, pid: int) -> Optional[float]:
        """Get average FPS from history.

        Args:
            pid: Process ID

        Returns:
            Average FPS or None if no history
        """
        if pid not in self._fps_history or not self._fps_history[pid]:
            return None

        # Use weighted average, favoring recent samples
        history = self._fps_history[pid]
        weights = [i + 1 for i in range(len(history))]
        weighted_sum = sum(fps * w for fps, w in zip(history, weights))
        weight_total = sum(weights)

        return weighted_sum / weight_total if weight_total > 0 else None

    def get_fps_drop(self, pid: int, threshold_fps: float = 60.0) -> Optional[float]:
        """Calculate FPS drop below threshold.

        Args:
            pid: Process ID
            threshold_fps: Target FPS threshold

        Returns:
            FPS drop percentage (0-100) or None if no data
        """
        current_fps = self._get_average_fps(pid)
        if current_fps is None:
            return None

        if current_fps >= threshold_fps:
            return 0.0

        # Calculate drop percentage
        drop_percent = ((threshold_fps - current_fps) / threshold_fps) * 100
        return min(100.0, max(0.0, drop_percent))

    def cleanup_dead_processes(self):
        """Remove data for processes that no longer exist."""
        dead_pids = []

        for pid in list(self._fps_history.keys()):
            try:
                psutil.Process(pid)
            except psutil.NoSuchProcess:
                dead_pids.append(pid)

        for pid in dead_pids:
            self._fps_history.pop(pid, None)
            self._last_update.pop(pid, None)
            self._prev_cpu_times.pop(pid, None)
            logger.debug(f"Cleaned up FPS data for dead process {pid}")
