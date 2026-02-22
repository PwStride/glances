#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""GPU Performance Analyzer.

This module provides the core analysis logic for GPU performance monitoring,
including temperature trend calculation and failure prediction.
"""

from collections import defaultdict, deque


class GpuPerformanceAnalyzer:
    """Analyzes GPU metrics to predict potential failures."""

    def __init__(
        self,
        temp_critical_threshold=85.0,
        temp_warning_threshold=80.0,
        temp_history_size=20,
        prediction_window=180,
        high_load_temp_threshold=75.0,
        high_load_proc_threshold=85.0,
        high_load_warning_duration=120,
    ):
        """Initialize the GPU performance analyzer.

        Args:
            temp_critical_threshold: Critical temperature threshold in Celsius
            temp_warning_threshold: Warning temperature threshold in Celsius
            temp_history_size: Number of temperature readings to track
            prediction_window: Time window in seconds for predictions
            high_load_temp_threshold: Temperature threshold for high load detection
            high_load_proc_threshold: Processor utilization threshold for high load
            high_load_warning_duration: Duration in seconds before warning
        """
        self.temp_critical_threshold = temp_critical_threshold
        self.temp_warning_threshold = temp_warning_threshold
        self.temp_history_size = temp_history_size
        self.prediction_window = prediction_window
        self.high_load_temp_threshold = high_load_temp_threshold
        self.high_load_proc_threshold = high_load_proc_threshold
        self.high_load_warning_duration = high_load_warning_duration

        # History tracking per GPU
        self.gpu_history = defaultdict(
            lambda: {
                'temp_history': deque(maxlen=self.temp_history_size),
                'high_load_start': None,
            }
        )

    def analyze_gpu(self, gpu, current_time):
        """Analyze a single GPU for potential failures.

        Args:
            gpu: Dictionary containing GPU metrics (gpu_id, temperature, proc, name)
            current_time: Current timestamp in seconds

        Returns:
            Dictionary with analysis results or None if essential metrics are missing.
            Returns error information if data collection fails.
        """
        gpu_id = gpu.get('gpu_id')
        if not gpu_id:
            return {
                'gpu_id': 'unknown',
                'name': 'Unknown',
                'status': 'ERROR',
                'prediction': 'GPU ID missing from metrics',
                'temp_trend': 0.0,
                'time_to_critical': None,
                'high_load_duration': 0,
                'error': 'missing_gpu_id',
            }

        temperature = gpu.get('temperature')
        proc_load = gpu.get('proc')

        # Check if essential metrics are missing
        if temperature is None:
            return {
                'gpu_id': gpu_id,
                'name': gpu.get('name', 'Unknown'),
                'status': 'ERROR',
                'prediction': f'Temperature data unavailable for {gpu_id}',
                'temp_trend': 0.0,
                'time_to_critical': None,
                'high_load_duration': 0,
                'error': 'missing_temperature',
            }

        history = self.gpu_history[gpu_id]

        # Initialize result
        result = {
            'gpu_id': gpu_id,
            'name': gpu.get('name', 'Unknown'),
            'status': 'OK',
            'prediction': '',
            'temp_trend': 0.0,
            'time_to_critical': None,
            'high_load_duration': 0,
        }

        # Add temperature to history with timestamp
        try:
            history['temp_history'].append((current_time, temperature))
        except Exception as e:
            result['status'] = 'ERROR'
            result['prediction'] = f'Failed to track temperature history: {str(e)}'
            result['error'] = 'history_tracking_failure'
            return result

        # Calculate temperature trend
        try:
            temp_trend = self._calculate_temp_trend(history['temp_history'])
            result['temp_trend'] = temp_trend
        except Exception as e:
            result['status'] = 'ERROR'
            result['prediction'] = f'Failed to calculate temperature trend: {str(e)}'
            result['error'] = 'trend_calculation_failure'
            return result

        # Predict time to critical temperature
        if temp_trend > 0:
            try:
                time_to_critical = (self.temp_critical_threshold - temperature) / (temp_trend / 60)
                result['time_to_critical'] = time_to_critical

                # Check if we'll reach critical temp within prediction window
                if 0 < time_to_critical <= self.prediction_window:
                    result['status'] = 'CRITICAL'
                    result['prediction'] = (
                        f"Temperature rising rapidly ({temp_trend:.1f}°C/min). "
                        f"Critical threshold ({self.temp_critical_threshold}°C) will be reached "
                        f"in ~{int(time_to_critical)}s. Reduce GPU load now!"
                    )
                elif temperature >= self.temp_warning_threshold:
                    result['status'] = 'WARNING'
                    result['prediction'] = (
                        f"Temperature at {temperature}°C and trending upward. "
                        f"Monitor closely or reduce workload."
                    )
            except Exception as e:
                result['status'] = 'ERROR'
                result['prediction'] = f'Failed to predict time to critical: {str(e)}'
                result['error'] = 'prediction_failure'
                return result

        # Check for sustained high load conditions
        if proc_load is not None:
            try:
                if proc_load >= self.high_load_proc_threshold and temperature >= self.high_load_temp_threshold:
                    if history['high_load_start'] is None:
                        history['high_load_start'] = current_time

                    high_load_duration = current_time - history['high_load_start']
                    result['high_load_duration'] = high_load_duration

                    if high_load_duration >= self.high_load_warning_duration:
                        if result['status'] == 'OK':
                            result['status'] = 'WARNING'
                        result['prediction'] = (
                            f"Sustained high load ({proc_load:.0f}%) and temperature ({temperature}°C) "
                            f"for {int(high_load_duration)}s. Risk of thermal throttling or failure."
                        )
                else:
                    # Reset high load tracking
                    history['high_load_start'] = None
            except Exception as e:
                result['status'] = 'ERROR'
                result['prediction'] = f'Failed to track high load conditions: {str(e)}'
                result['error'] = 'high_load_tracking_failure'
                return result

        # Check if currently at dangerous temperature
        try:
            if temperature >= self.temp_critical_threshold:
                result['status'] = 'CRITICAL'
                result['prediction'] = (
                    f"CRITICAL: Temperature at {temperature}°C! "
                    f"Reduce GPU load immediately to prevent damage or system crash."
                )
            elif temperature >= self.temp_warning_threshold and not result['prediction']:
                result['status'] = 'WARNING'
                result['prediction'] = f"Temperature at {temperature}°C. Close to warning threshold."
        except Exception as e:
            result['status'] = 'ERROR'
            result['prediction'] = f'Failed to check temperature thresholds: {str(e)}'
            result['error'] = 'threshold_check_failure'
            return result

        return result

    def _calculate_temp_trend(self, temp_history):
        """Calculate temperature trend in degrees per minute using linear regression.

        Args:
            temp_history: Deque of (timestamp, temperature) tuples

        Returns:
            Temperature trend in degrees per minute (float)

        Raises:
            ValueError: If temperature history is insufficient or invalid
        """
        if len(temp_history) < 2:
            return 0.0

        # Use linear regression on recent temperature history
        times = [t for t, _ in temp_history]
        temps = [temp for _, temp in temp_history]

        n = len(times)
        sum_x = sum(times)
        sum_y = sum(temps)
        sum_xy = sum(t * temp for t, temp in zip(times, temps))
        sum_x2 = sum(t * t for t in times)

        # Calculate slope (degrees per second)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Convert to degrees per minute
        return slope * 60

    def reset_history(self, gpu_id=None):
        """Reset history for a specific GPU or all GPUs.

        Args:
            gpu_id: GPU ID to reset, or None to reset all
        """
        if gpu_id is None:
            self.gpu_history.clear()
        elif gpu_id in self.gpu_history:
            del self.gpu_history[gpu_id]
