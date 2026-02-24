#
# This file is part of Glances.
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""Game process detector for gaming performance monitoring.

This module provides functionality to automatically detect which process
is likely a game based on resource usage patterns.
"""

from typing import Dict, List, Optional, Set

import psutil

from glances.logger import logger


class GameDetector:
    """Detects game processes based on resource usage patterns.

    Uses heuristics to identify processes that are likely games:
    - High CPU usage
    - High GPU usage (if available)
    - Process name patterns (optional)
    - Combined resource metrics
    """

    def __init__(
        self,
        cpu_threshold: float = 40.0,
        gpu_threshold: float = 30.0,
        detection_history: int = 5,
    ):
        """Initialize game detector.

        Args:
            cpu_threshold: Minimum CPU usage to consider as game
            gpu_threshold: Minimum GPU usage to consider as game
            detection_history: Number of consecutive detections before confirming
        """
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.detection_history = detection_history

        # Track detection count per process
        self._detection_count: Dict[int, int] = {}

        # Currently confirmed game process
        self._current_game_pid: Optional[int] = None

        # Processes to exclude from detection (system processes)
        self._excluded_names: Set[str] = {
            'systemd',
            'kernel',
            'kworker',
            'xorg',
            'gnome-shell',
            'kwin',
            'plasmashell',
            'compiz',
            'dwm',
            'i3',
            'glances',
            'python',
            'python3',
        }

    def detect_game_process(
        self, process_list: List[Dict], gpu_stats: Optional[Dict] = None
    ) -> Optional[int]:
        """Detect which process is most likely a game.

        Args:
            process_list: List of process dictionaries with stats
            gpu_stats: Optional GPU statistics

        Returns:
            PID of detected game process or None
        """
        # Filter out excluded processes
        candidates = self._filter_candidates(process_list)

        if not candidates:
            self._current_game_pid = None
            return None

        # Score each candidate
        scored_candidates = []
        for proc in candidates:
            score = self._calculate_game_score(proc, gpu_stats)
            if score > 0:
                scored_candidates.append((proc['pid'], score))

        if not scored_candidates:
            self._current_game_pid = None
            return None

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidate_pid = scored_candidates[0][0]

        # Update detection count
        self._update_detection_count(top_candidate_pid)

        # Check if we have enough consecutive detections
        if self._detection_count.get(top_candidate_pid, 0) >= self.detection_history:
            if self._current_game_pid != top_candidate_pid:
                logger.info(f"Game process detected: PID {top_candidate_pid}")
                self._current_game_pid = top_candidate_pid
            return top_candidate_pid

        return self._current_game_pid

    def _filter_candidates(self, process_list: List[Dict]) -> List[Dict]:
        """Filter process list to potential game candidates.

        Args:
            process_list: List of process dictionaries

        Returns:
            Filtered list of candidate processes
        """
        candidates = []

        for proc in process_list:
            # Skip system processes
            name = proc.get('name', '').lower()
            if any(excluded in name for excluded in self._excluded_names):
                continue

            # Must have minimum CPU usage
            cpu_percent = proc.get('cpu_percent', 0)
            if cpu_percent < self.cpu_threshold:
                continue

            candidates.append(proc)

        return candidates

    def _calculate_game_score(self, process: Dict, gpu_stats: Optional[Dict] = None) -> float:
        """Calculate likelihood score that a process is a game.

        Higher score = more likely to be a game.

        Args:
            process: Process dictionary with stats
            gpu_stats: Optional GPU statistics

        Returns:
            Score value (0-100)
        """
        score = 0.0

        # CPU usage factor (0-40 points)
        cpu_percent = process.get('cpu_percent', 0)
        score += min(40.0, cpu_percent * 0.4)

        # Memory usage factor (0-20 points)
        # Games typically use significant memory
        mem_percent = process.get('memory_percent', 0)
        if mem_percent > 5:
            score += min(20.0, mem_percent * 0.5)

        # GPU usage factor (0-40 points)
        if gpu_stats:
            # Try to correlate with GPU usage
            # This is simplified - in reality, would need per-process GPU tracking
            gpu_proc = gpu_stats.get('proc', 0)
            if gpu_proc > self.gpu_threshold:
                score += min(40.0, gpu_proc * 0.4)

        return score

    def _update_detection_count(self, pid: int):
        """Update detection count for a process.

        Args:
            pid: Process ID
        """
        # Increment count for this PID
        self._detection_count[pid] = self._detection_count.get(pid, 0) + 1

        # Decay counts for other PIDs
        for other_pid in list(self._detection_count.keys()):
            if other_pid != pid:
                self._detection_count[other_pid] = max(
                    0, self._detection_count[other_pid] - 1
                )
                if self._detection_count[other_pid] == 0:
                    del self._detection_count[other_pid]

    def get_current_game_pid(self) -> Optional[int]:
        """Get currently detected game PID.

        Returns:
            PID of current game or None
        """
        # Verify the process still exists
        if self._current_game_pid is not None:
            try:
                psutil.Process(self._current_game_pid)
                return self._current_game_pid
            except psutil.NoSuchProcess:
                logger.info(f"Game process {self._current_game_pid} no longer exists")
                self._current_game_pid = None
                self._detection_count.pop(self._current_game_pid, None)

        return None

    def reset(self):
        """Reset detection state."""
        self._detection_count.clear()
        self._current_game_pid = None
        logger.debug("Game detector reset")

    def add_excluded_name(self, name: str):
        """Add process name to exclusion list.

        Args:
            name: Process name to exclude
        """
        self._excluded_names.add(name.lower())

    def remove_excluded_name(self, name: str):
        """Remove process name from exclusion list.

        Args:
            name: Process name to remove from exclusion
        """
        self._excluded_names.discard(name.lower())
