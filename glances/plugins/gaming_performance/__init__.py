#
# This file is part of Glances.
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""Gaming Performance plugin for Glances.

Features:
- Automatic game process detection based on CPU/GPU usage
- Frame rate monitoring with FPS estimation
- Dynamic CPU priority adjustment to optimize gaming performance
- Automatic restoration when gaming session ends

This plugin monitors running processes to identify games and automatically
adjusts CPU priorities to enhance gaming performance when frame rates drop.
"""

from typing import Dict, List, Optional

from glances.logger import logger
from glances.plugins.gaming_performance.fps_detector import FpsDetector
from glances.plugins.gaming_performance.game_detector import GameDetector
from glances.plugins.gaming_performance.priority_manager import PriorityManager
from glances.plugins.plugin.model import GlancesPluginModel

# Fields description
fields_description = {
    'game_detected': {
        'description': 'Whether a game process is currently detected',
    },
    'game_pid': {
        'description': 'PID of detected game process',
    },
    'game_name': {
        'description': 'Name of detected game process',
    },
    'estimated_fps': {
        'description': 'Estimated frames per second',
        'unit': 'number',
    },
    'fps_drop_percent': {
        'description': 'FPS drop below target',
        'unit': 'percent',
    },
    'priority_boost_active': {
        'description': 'Whether priority boost is currently active',
    },
    'background_lowered': {
        'description': 'Number of background processes with lowered priority',
        'unit': 'number',
    },
}


class GamingPerformancePlugin(GlancesPluginModel):
    """Glances Gaming Performance plugin.

    Monitors gaming performance and dynamically adjusts CPU priorities
    to optimize frame rates.
    """

    def __init__(self, args=None, config=None):
        """Init the plugin."""
        super().__init__(
            args=args,
            config=config,
            items_history_list=None,
            stats_init_value={},
            fields_description=fields_description,
        )

        # Load configuration
        self._load_config(config)

        # Initialize components
        self.fps_detector = FpsDetector(history_size=self.fps_history_size)
        self.game_detector = GameDetector(
            cpu_threshold=self.game_cpu_threshold,
            gpu_threshold=self.game_gpu_threshold,
            detection_history=self.detection_history,
        )
        self.priority_manager = PriorityManager(
            game_nice_level=self.game_nice_level,
            background_nice_increase=self.background_nice_increase,
        )

        # State tracking
        self._last_game_pid: Optional[int] = None
        self._consecutive_low_fps: int = 0

    def _load_config(self, config):
        """Load configuration from glances.conf.

        Args:
            config: Configuration object
        """
        # Feature enable/disable
        self.enable_feature = self._get_config(
            config, 'enable_gaming_performance', default=True, as_bool=True
        )

        # Game detection settings
        self.game_cpu_threshold = self._get_config(
            config, 'game_cpu_threshold', default=40.0
        )
        self.game_gpu_threshold = self._get_config(
            config, 'game_gpu_threshold', default=30.0
        )
        self.detection_history = self._get_config(
            config, 'detection_history', default=5, as_int=True
        )

        # FPS monitoring settings
        self.fps_history_size = self._get_config(
            config, 'fps_history_size', default=30, as_int=True
        )
        self.target_fps = self._get_config(config, 'target_fps', default=60.0)
        self.fps_drop_threshold = self._get_config(
            config, 'fps_drop_threshold', default=20.0
        )
        self.consecutive_drops_required = self._get_config(
            config, 'consecutive_drops_required', default=3, as_int=True
        )

        # Priority adjustment settings
        self.game_nice_level = self._get_config(
            config, 'game_nice_level', default=-5, as_int=True
        )
        self.background_nice_increase = self._get_config(
            config, 'background_nice_increase', default=5, as_int=True
        )

        # Auto-restoration settings
        self.auto_restore = self._get_config(
            config, 'auto_restore', default=True, as_bool=True
        )
        self.restore_delay_seconds = self._get_config(
            config, 'restore_delay_seconds', default=10.0
        )

    def _get_config(
        self, config, key: str, default, as_bool: bool = False, as_int: bool = False
    ):
        """Get configuration value.

        Args:
            config: Configuration object
            key: Configuration key
            default: Default value
            as_bool: Parse as boolean
            as_int: Parse as integer

        Returns:
            Configuration value
        """
        if config is None:
            return default

        section = 'gaming_performance'

        if as_bool:
            return config.get_bool_value(section, key, default=default)
        elif as_int:
            return config.get_int_value(section, key, default=default)
        else:
            return config.get_float_value(section, key, default=default)

    def update(self):
        """Update gaming performance stats."""
        # Reset stats
        self.stats = {}

        # Check if feature is enabled
        if not self.enable_feature:
            return self.stats

        # Get dependencies
        try:
            # Get process list from processlist plugin
            processlist_plugin = self.get_plugin('processlist')
            if not processlist_plugin or not processlist_plugin.stats:
                return self.stats
            process_list = processlist_plugin.stats

            # Get GPU stats if available
            gpu_plugin = self.get_plugin('gpu')
            gpu_stats = None
            if gpu_plugin and gpu_plugin.stats:
                # Use first GPU if multiple
                gpu_stats = gpu_plugin.stats[0] if isinstance(gpu_plugin.stats, list) else gpu_plugin.stats

        except Exception as e:
            logger.debug(f"Error getting plugin dependencies: {e}")
            return self.stats

        # Detect game process
        game_pid = self.game_detector.detect_game_process(process_list, gpu_stats)

        # Update stats
        self.stats['game_detected'] = game_pid is not None
        self.stats['game_pid'] = game_pid
        self.stats['priority_boost_active'] = self.priority_manager.is_boost_active()

        # If no game detected, restore priorities if active
        if game_pid is None:
            if self.auto_restore and self.priority_manager.is_boost_active():
                self.priority_manager.restore_priorities()
                self._consecutive_low_fps = 0
            self._last_game_pid = None
            return self.stats

        # Get game process details
        game_process = next((p for p in process_list if p.get('pid') == game_pid), None)
        if game_process:
            self.stats['game_name'] = game_process.get('name', 'Unknown')

            # Estimate FPS
            cpu_percent = game_process.get('cpu_percent', 0)
            gpu_usage = gpu_stats.get('proc', None) if gpu_stats else None
            estimated_fps = self.fps_detector.estimate_fps(game_pid, cpu_percent, gpu_usage)

            self.stats['estimated_fps'] = estimated_fps

            # Calculate FPS drop
            if estimated_fps is not None:
                fps_drop = self.fps_detector.get_fps_drop(game_pid, self.target_fps)
                self.stats['fps_drop_percent'] = fps_drop

                # Check if we should boost priority
                if fps_drop and fps_drop >= self.fps_drop_threshold:
                    self._consecutive_low_fps += 1
                else:
                    self._consecutive_low_fps = 0

                # Activate boost if threshold met
                if (
                    self._consecutive_low_fps >= self.consecutive_drops_required
                    and not self.priority_manager.is_boost_active()
                ):
                    result = self.priority_manager.boost_game_priority(
                        game_pid, process_list
                    )
                    if result['game_boosted']:
                        self.stats['priority_boost_active'] = True
                        self.stats['background_lowered'] = result['background_lowered']
                        logger.info(
                            f"Gaming performance boost activated: "
                            f"FPS drop {fps_drop:.1f}% detected"
                        )

        # Update tracking
        self._last_game_pid = game_pid

        # Cleanup dead processes
        self.fps_detector.cleanup_dead_processes()
        self.priority_manager.cleanup_dead_processes()

        return self.stats

    def get_key(self):
        """Return the key of the list."""
        return 'game_pid'

    def reset(self):
        """Reset the stats."""
        super().reset()
        self.stats = {}

    def exit(self):
        """Called when plugin is exiting."""
        # Restore all priorities on exit
        if self.priority_manager.is_boost_active():
            logger.info("Restoring process priorities before exit")
            self.priority_manager.restore_priorities()

    def get_export(self):
        """Return the export stats."""
        return self.stats

    def msg_curse(self, args=None, max_width=None):
        """Return the dict to display in the curse interface."""
        ret = []

        # Check if feature is enabled
        if not self.enable_feature:
            return ret

        # Title
        msg = 'GAMING PERFORMANCE'
        ret.append(self.curse_add_line(msg, "TITLE"))
        ret.append(self.curse_new_line())

        # Game detection status
        if self.stats.get('game_detected'):
            game_name = self.stats.get('game_name', 'Unknown')
            game_pid = self.stats.get('game_pid', 'N/A')
            msg = f'Game: {game_name} (PID: {game_pid})'
            ret.append(self.curse_add_line(msg))
            ret.append(self.curse_new_line())

            # FPS info
            estimated_fps = self.stats.get('estimated_fps')
            if estimated_fps is not None:
                msg = f'FPS: {estimated_fps:.1f}'
                ret.append(self.curse_add_line(msg))

                # FPS drop indicator
                fps_drop = self.stats.get('fps_drop_percent', 0)
                if fps_drop > 0:
                    msg = f' (↓{fps_drop:.1f}%)'
                    if fps_drop >= self.fps_drop_threshold:
                        ret.append(self.curse_add_line(msg, 'WARNING'))
                    else:
                        ret.append(self.curse_add_line(msg))

                ret.append(self.curse_new_line())

            # Priority boost status
            if self.stats.get('priority_boost_active'):
                msg = 'Priority Boost: ACTIVE'
                ret.append(self.curse_add_line(msg, 'OK'))

                background_lowered = self.stats.get('background_lowered', 0)
                if background_lowered > 0:
                    msg = f' ({background_lowered} processes)'
                    ret.append(self.curse_add_line(msg))

                ret.append(self.curse_new_line())
        else:
            msg = 'No game detected'
            ret.append(self.curse_add_line(msg))
            ret.append(self.curse_new_line())

        return ret

    def get_plugin(self, plugin_name: str):
        """Get another plugin instance.

        Args:
            plugin_name: Name of plugin to get

        Returns:
            Plugin instance or None
        """
        # Access plugins through the args object
        if self.args is None:
            return None

        # Try to get from the global plugins dict
        try:
            from glances import plugins as glances_plugins

            return glances_plugins.get(plugin_name)
        except Exception:
            pass

        return None
