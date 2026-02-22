#
# This file is part of Glances.
#
# Copyright (C) 2020 Kirby Banman <kirby.banman@gmail.com>
# Copyright (C) 2024 Nicolas Hennion <nicolashennion@gmail.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""GPU plugin for Glances.

Currently supported:
- NVIDIA GPU (need pynvml lib)
- AMD GPU (no lib needed)
- Intel GPU (no lib needed)

Features:
- Real-time GPU monitoring (utilization, memory, temperature)
- Performance monitoring and failure prediction
- Desktop notifications for critical GPU states

Quick test:
- Start Glances
- In a terminal: vblank_mode=0 glxgears
"""

from time import time

from glances.globals import to_fahrenheit
from glances.gpu_list import glances_gpu
from glances.logger import logger
from glances.plugins.gpu.analyzer import GpuPerformanceAnalyzer
from glances.plugins.gpu.cards.amd import AmdGPU
from glances.plugins.gpu.cards.intel import IntelGPU
from glances.plugins.gpu.cards.nvidia import NvidiaGPU
from glances.plugins.gpu.notifier import GpuNotifier
from glances.plugins.plugin.model import GlancesPluginModel

# Fields description
# description: human readable description
# short_name: shortname to use un UI
# unit: unit type
# rate: is it a rate ? If yes, // by time_since_update when displayed,
# min_symbol: Auto unit should be used if value > than 1 'X' (K, M, G)...
fields_description = {
    'gpu_id': {
        'description': 'GPU identification',
    },
    'name': {
        'description': 'GPU name',
    },
    'mem': {
        'description': 'Memory consumption',
        'unit': 'percent',
    },
    'proc': {
        'description': 'GPU processor consumption',
        'unit': 'percent',
    },
    'temperature': {
        'description': 'GPU temperature',
        'unit': 'celsius',
    },
    'fan_speed': {
        'description': 'GPU fan speed',
        'unit': 'roundperminute',
    },
}

# Define the history items list
# All items in this list will be historised if the --enable-history tag is set
items_history_list = [
    {'name': 'proc', 'description': 'GPU processor', 'y_unit': '%'},
    {'name': 'mem', 'description': 'Memory consumption', 'y_unit': '%'},
]


class GpuPlugin(GlancesPluginModel):
    """Glances GPU plugin.

    stats is a list of dictionaries with one entry per GPU
    """

    def __init__(self, args=None, config=None):
        """Init the plugin."""
        super().__init__(
            args=args,
            config=config,
            items_history_list=items_history_list,
            stats_init_value=[],
            fields_description=fields_description,
        )
        # Init the Nvidia GPU API
        try:
            self.nvidia = NvidiaGPU()
        except Exception as e:
            logger.debug(f'Nvidia GPU initialization error: {e}')
            self.nvidia = None

        # Init the AMD GPU API
        try:
            self.amd = AmdGPU()
        except Exception as e:
            logger.debug(f'AMD GPU initialization error: {e}')
            self.amd = None
        # Just for test purpose (uncomment to test on computer without AMD GPU)
        # self.amd = AmdGPU(drm_root_folder='./tests-data/plugins/gpu/amd/sys/class/drm')

        # Init the Intel GPU API
        try:
            self.intel = IntelGPU()
        except Exception as e:
            logger.debug(f'Intel GPU initialization error: {e}')
            self.intel = None
        # Just for test purpose (uncomment to test on computer without Intel GPU)
        # self.intel = IntelGPU(drm_root_folder='./tests-data/plugins/gpu/intel/sys/class/drm')

        # Init performance monitoring (failure prediction)
        self._init_performance_monitoring(config)

        # We want to display the stat in the curse interface
        self.display_curse = True

    def _init_performance_monitoring(self, config):
        """Initialize GPU performance monitoring and failure prediction."""
        # Load performance monitoring configuration
        self.enable_performance_monitoring = self._get_perf_config(
            config, 'enable_performance_monitoring', default=True, as_bool=True
        )

        if not self.enable_performance_monitoring:
            self.analyzer = None
            self.notifier = None
            self.performance_stats = []
            return

        # Load analyzer configuration
        temp_critical = self._get_perf_config(config, 'temp_critical', default=85.0)
        temp_warning = self._get_perf_config(config, 'temp_warning', default=80.0)
        temp_history_size = int(self._get_perf_config(config, 'temp_history_size', default=20))
        prediction_window = self._get_perf_config(config, 'prediction_window', default=180)
        high_load_temp = self._get_perf_config(config, 'high_load_temp', default=75.0)
        high_load_proc = self._get_perf_config(config, 'high_load_proc', default=85.0)
        high_load_duration = self._get_perf_config(config, 'high_load_duration', default=120)

        # Initialize analyzer
        try:
            self.analyzer = GpuPerformanceAnalyzer(
                temp_critical_threshold=temp_critical,
                temp_warning_threshold=temp_warning,
                temp_history_size=temp_history_size,
                prediction_window=prediction_window,
                high_load_temp_threshold=high_load_temp,
                high_load_proc_threshold=high_load_proc,
                high_load_warning_duration=high_load_duration,
            )
        except Exception as e:
            logger.error(f"Failed to initialize GPU performance analyzer: {e}")
            self.analyzer = None

        # Load notifier configuration
        enable_notifications = self._get_perf_config(config, 'enable_notifications', default=True, as_bool=True)
        notification_cooldown = self._get_perf_config(config, 'notification_cooldown', default=300)

        # Initialize notifier
        if enable_notifications:
            try:
                self.notifier = GpuNotifier(cooldown_period=notification_cooldown)
            except Exception as e:
                logger.error(f"Failed to initialize GPU notifier: {e}")
                self.notifier = None
        else:
            self.notifier = None

        # Performance stats storage
        self.performance_stats = []

    def _get_perf_config(self, config, key, default, as_bool=False):
        """Get performance monitoring configuration value."""
        if config is None:
            return default

        if as_bool:
            return config.get_bool_value('gpu', key, default=default)

        value = config.get_float_value('gpu', key, default=default)
        return value if value is not None else default

    def exit(self):
        """Overwrite the exit method to close the GPU API."""
        self.nvidia.exit()
        self.amd.exit()
        self.intel.exit()
        # Call the father exit method
        super().exit()

    def get_key(self):
        """Return the key of the list."""
        return 'gpu_id'

    @GlancesPluginModel._check_decorator
    @GlancesPluginModel._log_result_decorator
    def update(self):
        """Update the GPU stats."""
        # Init new stats
        stats = self.get_init_value()

        # Get the stats
        if self.nvidia:
            stats.extend(self.nvidia.get_device_stats())
        if self.amd:
            stats.extend(self.amd.get_device_stats())
        if self.intel:
            stats.extend(self.intel.get_device_stats())

        # !!!
        # Uncomment to test on computer without Nvidia GPU
        # One GPU sample:
        # stats = [
        #     {
        #         "key": "gpu_id",
        #         "gpu_id": "nvidia0",
        #         "name": "Fake GeForce GTX",
        #         "mem": 5.792331695556641,
        #         "proc": 4,
        #         "temperature": 26,
        #         "fan_speed": 30,
        #     }
        # ]
        # Two GPU sample:
        # stats = [
        #     {
        #         "key": "gpu_id",
        #         "gpu_id": "nvidia0",
        #         "name": "Fake GeForce GTX1",
        #         "mem": 5.792331695556641,
        #         "proc": 4,
        #         "temperature": 26,
        #         "fan_speed": 30,
        #     },
        #     {
        #         "key": "gpu_id",
        #         "gpu_id": "nvidia1",
        #         "name": "Fake GeForce GTX1",
        #         "mem": 15,
        #         "proc": 8,
        #         "temperature": 65,
        #         "fan_speed": 75,
        #     },
        # ]

        # Update the stats
        self.stats = stats

        # Publish to global GPU registry
        glances_gpu.set(stats)

        # Run performance monitoring and failure prediction
        self._update_performance_monitoring()

        return self.stats

    def _update_performance_monitoring(self):
        """Update performance monitoring and failure prediction."""
        if not self.enable_performance_monitoring or self.analyzer is None:
            self.performance_stats = []
            return

        # Analyze each GPU for potential failures
        current_time = time()
        self.performance_stats = []

        for gpu in self.stats:
            try:
                analysis = self.analyzer.analyze_gpu(gpu, current_time)
                if analysis:
                    self.performance_stats.append(analysis)

                    # Send notification if needed
                    if self.notifier and analysis['status'] in ('WARNING', 'CRITICAL', 'ERROR'):
                        try:
                            self.notifier.send_notification(
                                analysis['gpu_id'], analysis['status'], analysis['prediction'], current_time
                            )
                        except Exception as e:
                            logger.debug(f"Failed to send notification for GPU {analysis['gpu_id']}: {e}")
            except Exception as e:
                gpu_id = gpu.get('gpu_id', 'unknown')
                logger.error(f"Failed to analyze GPU {gpu_id}: {e}")
                error_stat = {
                    'gpu_id': gpu_id,
                    'name': gpu.get('name', 'Unknown'),
                    'status': 'ERROR',
                    'prediction': f'Analysis failed: {str(e)}',
                    'temp_trend': 0.0,
                    'time_to_critical': None,
                    'high_load_duration': 0,
                }
                self.performance_stats.append(error_stat)

    def update_views(self):
        """Update stats views."""
        # Call the father's method
        super().update_views()

        # Add specifics information
        # Alert
        for i in self.stats:
            # Init the views for the current GPU
            self.views[i[self.get_key()]] = {'proc': {}, 'mem': {}, 'temperature': {}}
            # Processor alert
            if 'proc' in i:
                alert = self.get_alert(i['proc'], header='proc')
                self.views[i[self.get_key()]]['proc']['decoration'] = alert
            # Memory alert
            if 'mem' in i:
                alert = self.get_alert(i['mem'], header='mem')
                self.views[i[self.get_key()]]['mem']['decoration'] = alert
            # Temperature alert
            if 'temperature' in i:
                alert = self.get_alert(i['temperature'], header='temperature')
                self.views[i[self.get_key()]]['temperature']['decoration'] = alert

        return True

    def _get_mean(self, key):
        """Calculate mean value for a given key across all GPU stats.

        Returns None if calculation fails (e.g., missing data).
        """
        try:
            return sum(s[key] for s in self.stats if s is not None) / len(self.stats)
        except (TypeError, ZeroDivisionError):
            return None

    def _format_value(self, value, unit='%'):
        """Format a value with unit, or return N/A if value is None."""
        if value is None:
            return '{:>4}'.format('N/A')
        return f'{value:>3.0f}{unit}'

    def _build_header(self):
        """Build the header string based on GPU count and names."""
        same_name = all(s['name'] == self.stats[0]['name'] for s in self.stats)
        gpu_name = self.stats[0]['name']
        gpu_count = len(self.stats)

        if gpu_count > 1:
            if same_name:
                header = f'{gpu_count} {gpu_name}'
            else:
                header = f'{gpu_count} GPUs'
        elif same_name:
            header = gpu_name
        else:
            header = 'GPU'
        return header[:17]

    def _add_metric_line(self, ret, key, label, label_mean, unit='%'):
        """Add a metric line (label + value) to the curse output.

        Args:
            ret: The return list to append to
            key: The metric key (e.g., 'proc', 'mem', 'temperature')
            label: Label for single GPU mode
            label_mean: Label for multi-GPU mean mode
            unit: Unit suffix for the value (default: '%')
        """
        gpu_stats = self.stats[0]
        is_multi = len(self.stats) > 1

        ret.append(self.curse_new_line())
        ret.append(self.curse_add_line(f'{label_mean if is_multi else label:13}'))
        mean_value = self._get_mean(key)
        ret.append(
            self.curse_add_line(
                self._format_value(mean_value, unit),
                self.get_views(item=gpu_stats[self.get_key()], key=key, option='decoration'),
            )
        )

    def _msg_curse_summary(self, ret, args):
        """Build curse output for summary view (single GPU or mean mode)."""
        self._add_metric_line(ret, 'proc', 'proc:', 'proc mean:')
        self._add_metric_line(ret, 'mem', 'mem:', 'mem mean:')

        # Temperature needs special handling for Fahrenheit conversion
        gpu_stats = self.stats[0]
        is_multi = len(self.stats) > 1
        ret.append(self.curse_new_line())
        ret.append(self.curse_add_line('{:13}'.format('temp mean:' if is_multi else 'temperature:')))
        mean_temp = self._get_mean('temperature')
        if mean_temp is not None and args.fahrenheit:
            mean_temp = to_fahrenheit(mean_temp)
        unit = 'F' if args.fahrenheit else 'C'
        ret.append(
            self.curse_add_line(
                self._format_value(mean_temp, unit),
                self.get_views(item=gpu_stats[self.get_key()], key='temperature', option='decoration'),
            )
        )

    def _msg_curse_multi(self, ret):
        """Build curse output for multi-GPU detailed view."""
        for gpu_stats in self.stats:
            ret.append(self.curse_new_line())
            # id_msg = '{:7}'.format(gpu_stats['gpu_id'])
            id_msg = '{:7}'.format(gpu_stats['name'][0:9])
            msg = f'{id_msg}'
            ret.append(self.curse_add_line(msg))
            if gpu_stats.get('proc') is not None:
                proc_msg = self._format_value(gpu_stats.get('proc'))
                msg = f' {proc_msg}'
                ret.append(
                    self.curse_add_line(
                        msg,
                        self.get_views(item=gpu_stats[self.get_key()], key='proc', option='decoration'),
                    )
                )
            if gpu_stats.get('mem') is not None:
                mem_msg = self._format_value(gpu_stats.get('mem'))
                msg += f' mem {mem_msg}'
                ret.append(
                    self.curse_add_line(
                        msg,
                        self.get_views(item=gpu_stats[self.get_key()], key='mem', option='decoration'),
                    )
                )

    def msg_curse(self, args=None, max_width=None):
        """Return the dict to display in the curse interface."""
        ret = []

        # Only process if stats exist, not empty (issue #871) and plugin not disabled
        if not self.stats or self.is_disabled():
            return ret

        # Header
        ret.append(self.curse_add_line(self._build_header(), "TITLE"))

        # Build the string message
        if len(self.stats) == 1 or args.meangpu:
            self._msg_curse_summary(ret, args)
        else:
            self._msg_curse_multi(ret)

        # Add performance monitoring warnings if enabled
        self._msg_curse_performance_monitoring(ret)

        return ret

    def _msg_curse_performance_monitoring(self, ret):
        """Add performance monitoring warnings to the curse output."""
        if not self.enable_performance_monitoring or not self.performance_stats:
            return

        # Filter for GPUs with warnings, critical states, or errors
        alerts = [stat for stat in self.performance_stats if stat['status'] in ('WARNING', 'CRITICAL', 'ERROR')]

        if not alerts:
            return

        # Add separator
        ret.append(self.curse_new_line())

        # Display each alert
        for alert in alerts:
            ret.append(self.curse_new_line())

            # Status indicator with color
            status = alert['status']
            if status == 'CRITICAL':
                decoration = 'CRITICAL'
                indicator = '⚠️ '
            elif status == 'WARNING':
                decoration = 'WARNING'
                indicator = '⚡'
            else:  # ERROR
                decoration = 'CRITICAL'
                indicator = '❌'

            # GPU identifier
            ret.append(self.curse_add_line(f'{indicator} {alert["gpu_id"]}: ', decoration))

            # Prediction message (truncated if too long)
            message = alert['prediction']
            if len(message) > 50:
                message = message[:47] + '...'
            ret.append(self.curse_add_line(message))

            # Show time to critical if available and not in error state
            if alert['status'] != 'ERROR' and alert.get('time_to_critical') and alert['time_to_critical'] > 0:
                ret.append(self.curse_new_line())
                ret.append(self.curse_add_line(f"   Critical in ~{int(alert['time_to_critical'])}s"))
