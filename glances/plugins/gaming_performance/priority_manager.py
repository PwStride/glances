#
# This file is part of Glances.
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""CPU priority manager for gaming performance optimization.

This module provides functionality to dynamically adjust process priorities
to optimize gaming performance.
"""

from typing import Dict, List, Optional, Set

import psutil

from glances.logger import logger


class PriorityManager:
    """Manages CPU priority adjustments for gaming performance.

    Increases game process priority and decreases background process priority
    when gaming performance thresholds are met.
    """

    def __init__(
        self,
        game_nice_level: int = -5,
        background_nice_increase: int = 5,
        min_nice_level: int = -20,
        max_nice_level: int = 19,
    ):
        """Initialize priority manager.

        Args:
            game_nice_level: Target nice level for game process (lower = higher priority)
            background_nice_increase: How much to increase nice for background processes
            min_nice_level: Minimum allowed nice level (highest priority)
            max_nice_level: Maximum allowed nice level (lowest priority)
        """
        self.game_nice_level = max(min_nice_level, min(max_nice_level, game_nice_level))
        self.background_nice_increase = background_nice_increase
        self.min_nice_level = min_nice_level
        self.max_nice_level = max_nice_level

        # Track original nice levels for restoration
        self._original_nice: Dict[int, int] = {}

        # Currently boosted game PID
        self._boosted_game_pid: Optional[int] = None

        # PIDs of deprioritized processes
        self._deprioritized_pids: Set[int] = set()

        # System processes that should never be modified
        self._protected_names: Set[str] = {
            'systemd',
            'init',
            'kernel',
            'kworker',
            'ksoftirqd',
            'migration',
            'watchdog',
            'cpuhp',
            'kthreadd',
            'rcu_sched',
            'rcu_bh',
        }

        # Track if currently in boost mode
        self._boost_active = False

    def boost_game_priority(
        self, game_pid: int, process_list: List[Dict], dry_run: bool = False
    ) -> Dict[str, any]:
        """Boost game process priority and lower background processes.

        Args:
            game_pid: PID of game process to boost
            process_list: List of all process dictionaries
            dry_run: If True, don't actually change priorities (for testing)

        Returns:
            Dictionary with operation results
        """
        result = {
            'game_boosted': False,
            'game_pid': game_pid,
            'background_lowered': 0,
            'errors': [],
        }

        # Check if already boosted
        if self._boost_active and self._boosted_game_pid == game_pid:
            return result

        try:
            # Boost game process
            game_boosted = self._set_game_priority(game_pid, dry_run)
            result['game_boosted'] = game_boosted

            if game_boosted:
                self._boosted_game_pid = game_pid
                self._boost_active = True

                # Lower priority of background processes
                lowered_count = self._lower_background_priorities(
                    game_pid, process_list, dry_run
                )
                result['background_lowered'] = lowered_count

                logger.info(
                    f"Gaming boost activated: game PID {game_pid}, "
                    f"{lowered_count} background processes deprioritized"
                )

        except Exception as e:
            error_msg = f"Error boosting game priority: {e}"
            logger.error(error_msg)
            result['errors'].append(error_msg)

        return result

    def restore_priorities(self, dry_run: bool = False) -> Dict[str, any]:
        """Restore all processes to their original priorities.

        Args:
            dry_run: If True, don't actually change priorities (for testing)

        Returns:
            Dictionary with operation results
        """
        result = {
            'game_restored': False,
            'background_restored': 0,
            'errors': [],
        }

        if not self._boost_active:
            return result

        try:
            # Restore game process
            if self._boosted_game_pid and self._boosted_game_pid in self._original_nice:
                game_restored = self._restore_process_priority(
                    self._boosted_game_pid, dry_run
                )
                result['game_restored'] = game_restored

            # Restore background processes
            restored_count = 0
            for pid in list(self._deprioritized_pids):
                if self._restore_process_priority(pid, dry_run):
                    restored_count += 1

            result['background_restored'] = restored_count

            # Clear state
            self._boost_active = False
            self._boosted_game_pid = None
            self._deprioritized_pids.clear()

            logger.info(
                f"Gaming boost deactivated: "
                f"{restored_count} background processes restored"
            )

        except Exception as e:
            error_msg = f"Error restoring priorities: {e}"
            logger.error(error_msg)
            result['errors'].append(error_msg)

        return result

    def _set_game_priority(self, pid: int, dry_run: bool = False) -> bool:
        """Set game process to high priority.

        Args:
            pid: Process ID
            dry_run: If True, don't actually change priority

        Returns:
            True if successful, False otherwise
        """
        try:
            process = psutil.Process(pid)
            current_nice = process.nice()

            # Store original nice level
            if pid not in self._original_nice:
                self._original_nice[pid] = current_nice

            # Only change if not already at target
            if current_nice != self.game_nice_level:
                if not dry_run:
                    process.nice(self.game_nice_level)
                logger.info(
                    f"Set game process {pid} priority: "
                    f"nice {current_nice} -> {self.game_nice_level}"
                )
                return True
            return True

        except psutil.AccessDenied:
            logger.warning(
                f"Access denied setting priority for PID {pid}. "
                f"Run with elevated privileges for priority adjustment."
            )
            return False
        except psutil.NoSuchProcess:
            logger.debug(f"Process {pid} no longer exists")
            return False
        except Exception as e:
            logger.error(f"Error setting game priority for PID {pid}: {e}")
            return False

    def _lower_background_priorities(
        self, game_pid: int, process_list: List[Dict], dry_run: bool = False
    ) -> int:
        """Lower priority of background processes.

        Args:
            game_pid: PID of game (to exclude)
            process_list: List of all processes
            dry_run: If True, don't actually change priorities

        Returns:
            Number of processes lowered
        """
        lowered_count = 0

        for proc_dict in process_list:
            pid = proc_dict.get('pid')
            if not pid or pid == game_pid:
                continue

            # Skip if already deprioritized
            if pid in self._deprioritized_pids:
                continue

            # Check if process should be protected
            if self._is_protected_process(proc_dict):
                continue

            # Lower this process priority
            if self._lower_process_priority(pid, dry_run):
                self._deprioritized_pids.add(pid)
                lowered_count += 1

        return lowered_count

    def _lower_process_priority(self, pid: int, dry_run: bool = False) -> bool:
        """Lower a single process priority.

        Args:
            pid: Process ID
            dry_run: If True, don't actually change priority

        Returns:
            True if successful, False otherwise
        """
        try:
            process = psutil.Process(pid)
            current_nice = process.nice()

            # Store original nice level
            if pid not in self._original_nice:
                self._original_nice[pid] = current_nice

            # Calculate new nice level
            new_nice = min(
                self.max_nice_level, current_nice + self.background_nice_increase
            )

            # Only change if different
            if new_nice != current_nice:
                if not dry_run:
                    process.nice(new_nice)
                logger.debug(
                    f"Lowered background process {pid} priority: "
                    f"nice {current_nice} -> {new_nice}"
                )
            return True

        except psutil.AccessDenied:
            # Expected for many system processes
            return False
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            logger.debug(f"Error lowering priority for PID {pid}: {e}")
            return False

    def _restore_process_priority(self, pid: int, dry_run: bool = False) -> bool:
        """Restore a process to its original priority.

        Args:
            pid: Process ID
            dry_run: If True, don't actually change priority

        Returns:
            True if successful, False otherwise
        """
        if pid not in self._original_nice:
            return False

        try:
            process = psutil.Process(pid)
            original_nice = self._original_nice[pid]
            current_nice = process.nice()

            if current_nice != original_nice:
                if not dry_run:
                    process.nice(original_nice)
                logger.debug(
                    f"Restored process {pid} priority: "
                    f"nice {current_nice} -> {original_nice}"
                )

            # Clean up tracking
            del self._original_nice[pid]
            self._deprioritized_pids.discard(pid)
            return True

        except psutil.AccessDenied:
            return False
        except psutil.NoSuchProcess:
            # Process is gone, clean up
            del self._original_nice[pid]
            self._deprioritized_pids.discard(pid)
            return False
        except Exception as e:
            logger.debug(f"Error restoring priority for PID {pid}: {e}")
            return False

    def _is_protected_process(self, process: Dict) -> bool:
        """Check if a process should be protected from priority changes.

        Args:
            process: Process dictionary

        Returns:
            True if process should be protected
        """
        # Check name
        name = process.get('name', '').lower()
        if any(protected in name for protected in self._protected_names):
            return True

        # Protect very low nice processes (already high priority)
        nice = process.get('nice', 0)
        if nice < -10:
            return True

        # Protect critical system processes (optional: check by other criteria)
        # Could add more sophisticated checks here

        return False

    def is_boost_active(self) -> bool:
        """Check if gaming boost is currently active.

        Returns:
            True if boost is active
        """
        return self._boost_active

    def get_boosted_game_pid(self) -> Optional[int]:
        """Get PID of currently boosted game.

        Returns:
            Game PID or None
        """
        return self._boosted_game_pid if self._boost_active else None

    def cleanup_dead_processes(self):
        """Remove tracking data for dead processes."""
        dead_pids = []

        for pid in list(self._original_nice.keys()):
            try:
                psutil.Process(pid)
            except psutil.NoSuchProcess:
                dead_pids.append(pid)

        for pid in dead_pids:
            self._original_nice.pop(pid, None)
            self._deprioritized_pids.discard(pid)
            logger.debug(f"Cleaned up priority data for dead process {pid}")
