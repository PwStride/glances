# GPU Performance Monitor Plugin

## Overview

The GPU Performance Monitor plugin analyzes GPU metrics in real-time to predict potential failures during high-performance workloads such as gaming. It helps users monitor gaming performance and limit system GPU crashes by providing ample warning time to make adjustments before a failure occurs.

## Features

### Temperature Trend Analysis
- Monitors GPU temperature changes over time
- Calculates temperature trends (degrees per minute)
- Predicts when critical temperature thresholds will be reached
- Provides early warnings when temperature is rising rapidly

### Sustained High Load Detection
- Tracks periods of sustained high GPU utilization and temperature
- Warns when GPU maintains dangerous conditions for extended periods
- Helps identify potential thermal throttling situations

### Desktop Notifications
- Sends platform-specific desktop notifications on warnings and critical alerts
- Supports:
  - Linux: `notify-send`
  - macOS: `osascript`
  - Windows: PowerShell toast notifications
- Configurable cooldown period to avoid notification spam

## Configuration

Add the following section to your `glances.conf` file:

```ini
[gpu_monitor]
# GPU Performance Monitor - Predicts potential GPU failures during gaming/high-performance workloads
disable=False

# Temperature thresholds for failure prediction (in degrees Celsius)
# Critical: absolute maximum safe temperature
temp_critical=85
# Warning: temperature approaching critical levels
temp_warning=80

# Temperature history size: number of readings to track for trend analysis
temp_history_size=20

# Prediction window: warn if critical temperature will be reached within this time (seconds)
prediction_window=180

# High load detection thresholds
# Temperature threshold for high load detection (Celsius)
high_load_temp=75
# Processor utilization threshold for high load detection (%)
high_load_proc=85
# Duration before warning about sustained high load (seconds)
high_load_duration=120

# Notification settings
# Enable desktop notifications for GPU warnings
enable_notifications=True
# Cooldown period between notifications for the same GPU (seconds)
notification_cooldown=300
```

## Alert Levels

### OK
- GPU is operating within normal parameters
- No action required

### WARNING
- Temperature is approaching critical levels (≥ temp_warning)
- Sustained high load detected for extended period
- Temperature is trending upward
- **Action**: Monitor closely, consider reducing GPU load

### CRITICAL
- Temperature at or above critical threshold (≥ temp_critical)
- Temperature will reach critical level within prediction window
- **Action**: Reduce GPU load immediately to prevent damage or system crash

## Usage

The plugin runs automatically when Glances starts and the GPU plugin is enabled. It will:

1. Collect temperature and utilization data from all detected GPUs
2. Analyze trends and sustained load conditions
3. Calculate time to critical temperature if trending upward
4. Display warnings in the Glances interface
5. Send desktop notifications for WARNING and CRITICAL states

## Display

The plugin displays in the Glances TUI showing:
- GPU ID and name
- Current status (OK, WARNING, CRITICAL) with color coding
- Prediction message explaining the issue
- Temperature trend (degrees per minute)
- Estimated time until critical temperature

Example output:
```
GPU PERFORMANCE MONITOR
 nvidia0 (NVIDIA GeForce RTX 3080): WARNING
   Temperature at 82°C and trending upward. Monitor closely or reduce workload.
   Temp trend: ↑ 2.3°C/min
   Time to critical: ~78s
```

## Requirements

- GPU plugin must be enabled
- For notifications:
  - Linux: `notify-send` (usually pre-installed with GNOME/KDE)
  - macOS: Built-in (uses AppleScript)
  - Windows: PowerShell (built-in on Windows 10+)

## Technical Details

### Temperature Trend Calculation
The plugin uses linear regression on recent temperature history to calculate the rate of temperature change. This allows for accurate prediction of when critical thresholds will be reached.

### High Load Detection
The plugin tracks when GPU maintains both high temperature (≥ high_load_temp) and high utilization (≥ high_load_proc) simultaneously. This condition indicates thermal stress that may lead to throttling or failure.

### Notification Strategy
- Notifications are sent when status changes to WARNING or CRITICAL
- CRITICAL notifications can repeat (subject to cooldown)
- WARNING notifications only sent once per state change
- Cooldown period prevents notification spam

## Dependencies

This plugin depends on the GPU plugin and uses the global GPU stats registry (`glances_gpu`) to access real-time GPU metrics.

## See Also

- [GPU Plugin Documentation](../gpu/)
- [Glances Configuration](../../../../conf/glances.conf)
