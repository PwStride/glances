#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""AI Training Pipeline plugin for Glances.

Modern GPU hardware is often far faster than the storage and CPU pipeline that
feeds it.  During AI training this shows up as GPU data starvation: the GPU
finishes processing a batch and goes idle while the data loader races to
prepare the next one.  Left unaddressed, this wastes a significant fraction
of available GPU capacity and extends training time proportionally.

This plugin surfaces that imbalance in real time so that operators can tune
DataLoader configuration (worker count, prefetch factor, pin_memory) rather
than waiting for training to complete before discovering the inefficiency
through post-hoc profiling.

Hardware signals drive all detection so the plugin works regardless of which
framework, launcher, or container runtime the training job uses.
"""

from typing import Dict, List, Optional

from glances.logger import logger
from glances.plugins.ai_pipeline.imbalance_detector import GpuStarvationDetector
from glances.plugins.ai_pipeline.recommender import PipelineRecommender
from glances.plugins.ai_pipeline.workload_detector import AiWorkloadDetector
from glances.plugins.plugin.model import GlancesPluginModel

# Fields description
fields_description = {
    'training_detected': {
        'description': 'Whether a heavy GPU+I/O workload (AI-training-style) is currently detected',
    },
    'pipeline_score': {
        'description': 'Pipeline balance score (100=fully balanced, 0=GPU starved)',
        'unit': 'percent',
    },
    'gpu_starvation_pct': {
        'description': 'Fraction of detection window where GPU was idle while disk was busy',
        'unit': 'percent',
    },
    'disk_saturation_pct': {
        'description': 'Disk I/O saturation level relative to observed peak',
        'unit': 'percent',
    },
    'gpu_cv': {
        'description': 'GPU utilization coefficient of variation (spikiness indicator, high=starved)',
        'unit': 'percent',
    },
    'bottleneck_type': {
        'description': 'Classified bottleneck: disk_bound | cpu_bound | memory_bound | balanced',
    },
    'recommendation': {
        'description': 'Actionable recommendation to improve pipeline throughput',
    },
    'severity': {
        'description': 'Severity level derived from pipeline_score: OK | WARNING | CRITICAL',
    },
}

# History items tracked as time-series (for sparklines / graphs)
items_history_list = [
    {'name': 'pipeline_score', 'description': 'Pipeline balance score', 'y_unit': '%'},
    {'name': 'gpu_starvation_pct', 'description': 'GPU starvation %', 'y_unit': '%'},
]


class Ai_pipelinePlugin(GlancesPluginModel):
    """Glances AI Training Pipeline plugin.

    Orchestrates the three sub-components — workload detection, starvation
    measurement, and bottleneck classification — and exposes their combined
    output through the standard Glances plugin interface.
    """

    def __init__(self, args=None, config=None):
        """Init the plugin."""
        super().__init__(
            args=args,
            config=config,
            items_history_list=items_history_list,
            stats_init_value={},
            fields_description=fields_description,
        )

        # Load configuration
        self._load_config(config)

        # Initialize sub-components
        self.starvation_detector = GpuStarvationDetector(
            window_size=self.detection_window,
            gpu_idle_threshold=self.gpu_idle_threshold,
            disk_saturation_threshold=self.disk_saturation_threshold,
        )
        self.workload_detector = AiWorkloadDetector(
            detection_history=self.detection_history,
            gpu_active_threshold=self.gpu_active_threshold,
            disk_active_bytes=self.disk_active_bytes,
        )
        self.recommender = PipelineRecommender()

    def _load_config(self, config):
        """Apply user-defined thresholds, falling back to safe defaults when absent."""
        section = 'ai_pipeline'

        if config is None:
            self.enable_feature = True
            self.gpu_idle_threshold = 15.0
            self.disk_saturation_threshold = 80.0
            self.detection_window = 30
            self.training_detection_enabled = True
            self.detection_history = 3
            self.gpu_active_threshold = 20.0
            self.disk_active_bytes = 1_000_000.0
            # Severity thresholds (pipeline_score bands)
            self.severity_warning = 80.0
            self.severity_critical = 50.0
            return

        self.enable_feature = not config.get_bool_value(section, 'disable', default=False)
        self.gpu_idle_threshold = config.get_float_value(section, 'gpu_idle_threshold', default=15.0)
        self.disk_saturation_threshold = config.get_float_value(
            section, 'disk_saturation_threshold', default=80.0
        )
        self.detection_window = config.get_int_value(section, 'detection_window', default=30)
        self.training_detection_enabled = config.get_bool_value(
            section, 'training_detection_enabled', default=True
        )
        self.detection_history = config.get_int_value(section, 'detection_history', default=3)
        self.gpu_active_threshold = config.get_float_value(
            section, 'gpu_active_threshold', default=20.0
        )
        self.disk_active_bytes = config.get_float_value(
            section, 'disk_active_bytes', default=1_000_000.0
        )
        # Severity thresholds — pipeline_score below severity_warning → WARNING,
        # below severity_critical → CRITICAL
        self.severity_warning = config.get_float_value(
            section, 'severity_warning', default=80.0
        )
        self.severity_critical = config.get_float_value(
            section, 'severity_critical', default=50.0
        )

    def _classify_severity(self, pipeline_score: float) -> str:
        """Translate the pipeline score into a standard Glances severity level.

        Decoupling the severity thresholds from the score calculation means
        operators can adjust what constitutes an acceptable imbalance for their
        hardware without touching the measurement logic.

        Args:
            pipeline_score: Pipeline balance score (0–100).

        Returns:
            One of 'OK', 'WARNING', 'CRITICAL'.
        """
        if pipeline_score < self.severity_critical:
            return 'CRITICAL'
        if pipeline_score < self.severity_warning:
            return 'WARNING'
        return 'OK'

    def update(self):
        """Update AI pipeline stats."""
        self.stats = {}

        if not self.enable_feature:
            return self.stats

        # Gather GPU stats
        gpu_proc = None
        gpu_mem = None
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            gpu_proc = gpu_stats.get('proc')
            gpu_mem = gpu_stats.get('mem')

        # Gather Disk I/O stats
        disk_bytes_per_sec = 0.0
        diskio_stats = self._get_diskio_stats()
        if diskio_stats:
            disk_bytes_per_sec = sum(
                (d.get('read_bytes_rate_per_sec') or 0) + (d.get('write_bytes_rate_per_sec') or 0)
                for d in diskio_stats
                if isinstance(d, dict)
            )

        # Signal-driven training workload detection
        training_detected = False
        if self.training_detection_enabled:
            training_detected = self.workload_detector.detect(gpu_proc, disk_bytes_per_sec)
        self.stats['training_detected'] = training_detected

        # If GPU data is unavailable, skip starvation analysis
        if gpu_proc is None:
            logger.debug('ai_pipeline: no GPU data available, skipping starvation detection')
            return self.stats

        # Feed sample into the starvation detector
        self.starvation_detector.add_sample(gpu_proc, disk_bytes_per_sec)

        # Compute imbalance metrics
        starvation_pct = self.starvation_detector.compute_starvation_pct()
        disk_saturation_pct = self.starvation_detector.compute_disk_saturation_pct()
        gpu_cv = self.starvation_detector.compute_gpu_cv()
        pipeline_score = self.starvation_detector.compute_pipeline_score()

        self.stats['pipeline_score'] = round(pipeline_score, 1)
        self.stats['gpu_starvation_pct'] = round(starvation_pct, 1)
        self.stats['disk_saturation_pct'] = round(disk_saturation_pct, 1)
        self.stats['gpu_cv'] = round(gpu_cv, 1)

        # Derive severity label for this tick
        self.stats['severity'] = self._classify_severity(pipeline_score)

        # Classify bottleneck and generate recommendation
        cpu_percent = self._get_cpu_percent()
        bottleneck_type, recommendation = self.recommender.get_recommendation(
            starvation_pct=starvation_pct,
            disk_saturation_pct=disk_saturation_pct,
            gpu_mem_pct=gpu_mem,
            cpu_percent=cpu_percent,
        )
        self.stats['bottleneck_type'] = bottleneck_type
        self.stats['recommendation'] = recommendation

        return self.stats

    def _get_gpu_stats(self) -> Optional[Dict]:
        """Retrieve GPU metrics from the GPU plugin.

        Using the first GPU keeps the common single-GPU case simple; multi-GPU
        training imbalance is a workload scheduling concern beyond this plugin's scope.
        """
        try:
            plugin = self._get_plugin('gpu')
            if plugin and plugin.stats:
                stats = plugin.stats
                return stats[0] if isinstance(stats, list) else stats
        except Exception as e:
            logger.debug(f'ai_pipeline: error fetching GPU stats: {e}')
        return None

    def _get_diskio_stats(self) -> Optional[List]:
        """Retrieve per-disk I/O metrics from the diskio plugin.

        All disks are summed by the caller because training data may be
        striped or distributed across multiple devices.
        """
        try:
            plugin = self._get_plugin('diskio')
            if plugin and plugin.stats:
                return plugin.stats
        except Exception as e:
            logger.debug(f'ai_pipeline: error fetching diskio stats: {e}')
        return None

    def _get_cpu_percent(self) -> Optional[float]:
        """Retrieve overall CPU utilisation to help distinguish cpu_bound starvation."""
        try:
            plugin = self._get_plugin('cpu')
            if plugin and plugin.stats:
                return plugin.stats.get('total')
        except Exception as e:
            logger.debug(f'ai_pipeline: error fetching CPU stats: {e}')
        return None

    def _get_plugin(self, plugin_name: str):
        """Access a sibling plugin's live stats without creating a hard import dependency.

        Args:
            plugin_name: Plugin name to retrieve

        Returns:
            Plugin instance or None
        """
        if self.args is None:
            return None
        try:
            from glances import plugins as glances_plugins
            return glances_plugins.get(plugin_name)
        except Exception:
            pass
        return None

    def get_key(self):
        """Return the key of the list."""
        return 'bottleneck_type'

    def reset(self):
        """Reset the stats."""
        super().reset()
        self.stats = {}

    def get_export(self):
        """Return the export stats."""
        return self.stats

    def msg_curse(self, args=None, max_width=None):
        """Return the list of strings to display in the curse interface."""
        ret = []

        if not self.enable_feature:
            return ret

        # Title
        msg = 'AI PIPELINE'
        ret.append(self.curse_add_line(msg, 'TITLE'))
        ret.append(self.curse_new_line())

        if not self.stats:
            ret.append(self.curse_add_line('No data'))
            ret.append(self.curse_new_line())
            return ret

        # Training detection status
        training_detected = self.stats.get('training_detected', False)
        if training_detected:
            msg = 'Training: ACTIVE'
        else:
            msg = 'Training: not detected'
        ret.append(self.curse_add_line(msg))
        ret.append(self.curse_new_line())

        # Pipeline score (only show if we have GPU data)
        pipeline_score = self.stats.get('pipeline_score')
        if pipeline_score is not None:
            score = int(pipeline_score)
            severity = self.stats.get('severity', 'OK')

            # Decoration follows the severity label directly
            decoration = severity

            # Score bar: 10 chars wide
            bar_filled = score // 10
            bar_empty = 10 - bar_filled
            bar = '\u2588' * bar_filled + '\u2591' * bar_empty

            msg = f'Score: {score:3d}% '
            ret.append(self.curse_add_line(msg))
            ret.append(self.curse_add_line(bar, decoration))
            ret.append(self.curse_new_line())

            # Starvation and saturation detail
            starvation = self.stats.get('gpu_starvation_pct', 0)
            disk_sat = self.stats.get('disk_saturation_pct', 0)
            msg = f'GPU idle: {starvation:4.1f}%  Disk: {disk_sat:4.1f}%'
            ret.append(self.curse_add_line(msg))
            ret.append(self.curse_new_line())

            # Bottleneck type
            bottleneck = self.stats.get('bottleneck_type', '')
            if bottleneck and bottleneck != 'balanced':
                msg = f'Bottleneck: {bottleneck}'
                ret.append(self.curse_add_line(msg, decoration))
                ret.append(self.curse_new_line())

                # Recommendation
                recommendation = self.stats.get('recommendation', '')
                if recommendation:
                    # Truncate to fit TUI width
                    max_rec_len = (max_width or 40) - 2
                    if len(recommendation) > max_rec_len:
                        recommendation = recommendation[: max_rec_len - 1] + '\u2026'
                    msg = f'\u2192 {recommendation}'
                    ret.append(self.curse_add_line(msg))
                    ret.append(self.curse_new_line())
            else:
                msg = 'Pipeline: balanced'
                ret.append(self.curse_add_line(msg, 'OK'))
                ret.append(self.curse_new_line())
        else:
            msg = 'No GPU data available'
            ret.append(self.curse_add_line(msg))
            ret.append(self.curse_new_line())

        return ret
