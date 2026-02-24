# Dynamic CPU Priority Thresholding for Game Optimization

## Overview

The Gaming Performance plugin automatically detects game processes and dynamically adjusts CPU priorities to optimize gaming performance when frame rates drop below target thresholds.

## Features

- **Automatic Game Detection**: Identifies game processes based on CPU and GPU usage patterns
- **Frame Rate Monitoring**: Estimates FPS and detects performance drops
- **Dynamic Priority Adjustment**: Automatically boosts game process priority and lowers background processes
- **Conservative Approach**: Uses safe priority adjustments (nice +5 to +10 for background processes)
- **Auto-Restoration**: Automatically restores original priorities when gaming session ends

## How It Works

### 1. Game Detection

The plugin monitors all running processes and identifies games using these criteria:

- **CPU Usage**: Process must use >=40% CPU (configurable)
- **GPU Usage**: Process should use >=30% GPU if available (configurable)
- **Exclusion List**: Filters out known system processes (systemd, Xorg, etc.)
- **Confirmation**: Requires 5 consecutive detections to confirm a game (prevents false positives)

### 2. FPS Monitoring

Once a game is detected, the plugin estimates frame rates using:

- **CPU Activity Patterns**: Analyzes CPU time deltas and usage patterns
- **GPU Correlation**: Refines estimates using GPU utilization data
- **Weighted Averaging**: Recent FPS samples are weighted more heavily (30 samples by default)

### 3. Dynamic Priority Adjustment

When FPS drops below the target threshold:

1. **Detection**: Monitors for FPS drop >=20% below target (default: 60 FPS)
2. **Confirmation**: Requires 3 consecutive drops to avoid false triggers
3. **Game Boost**: Sets game process nice level to -5 (higher priority)
4. **Background Lowering**: Increases nice level by +5 for background processes
5. **Protection**: Never modifies system-critical processes

### 4. Auto-Restoration

The plugin automatically restores priorities when:

- Game process is no longer detected
- Game exits or terminates
- Glances exits

## Configuration

All settings are configured in `glances.conf` under the `[gaming_performance]` section:

```ini
[gaming_performance]
# Enable/disable the feature
disable=False
enable_gaming_performance=True

# Game detection thresholds
game_cpu_threshold=40.0          # Min CPU % to consider as game
game_gpu_threshold=30.0          # Min GPU % to help identify games
detection_history=5              # Consecutive detections to confirm

# FPS monitoring settings
fps_history_size=30              # Number of FPS samples to average
target_fps=60.0                  # Target FPS threshold
fps_drop_threshold=20.0          # FPS drop % that triggers boost
consecutive_drops_required=3      # Consecutive drops before boosting

# Priority adjustment settings
game_nice_level=-5               # Nice level for game (-20 to 19)
background_nice_increase=5       # How much to increase nice for background

# Auto-restoration settings
auto_restore=True                # Restore priorities when game ends
restore_delay_seconds=10.0       # Delay before restoring
```

## Usage

### Running with Elevated Privileges

To set negative nice values (higher priority), run Glances with elevated privileges:

```bash
sudo glances
```

Without elevated privileges, the plugin can still:
- Detect games
- Monitor FPS
- Lower background process priorities (increase nice)
- But cannot boost game priority below 0

### Viewing in TUI

The gaming performance stats appear in the Glances terminal interface:

```
GAMING PERFORMANCE
Game: game.exe (PID: 12345)
FPS: 45.3 (drop 24.5%)
Priority Boost: ACTIVE (15 processes)
```

### API Access

Stats are available via the REST API at:

```
http://localhost:61208/api/4/gaming_performance
```

Example response:
```json
{
  "game_detected": true,
  "game_pid": 12345,
  "game_name": "game.exe",
  "estimated_fps": 45.3,
  "fps_drop_percent": 24.5,
  "priority_boost_active": true,
  "background_lowered": 15
}
```

## Technical Details

### FPS Estimation Algorithm

FPS estimation uses heuristics based on CPU usage patterns:

1. **CPU Time Delta**: Measures CPU time used between updates
2. **Usage Normalization**: Scales based on CPU percentage
3. **GPU Refinement**: Adjusts estimate using GPU utilization
4. **Weighted Average**: Recent samples weighted more heavily

**Note**: FPS estimation is approximate and works best with:
- Games that show consistent CPU usage patterns
- Systems with available GPU monitoring
- Frame rates in the 10-300 FPS range

### Priority Adjustment Safety

The plugin includes multiple safety mechanisms:

- **Protected Processes**: Never modifies kernel threads, init, systemd, etc.
- **Conservative Nice Values**: Default +5 increase is gentle
- **Access Control**: Respects system permissions
- **Graceful Degradation**: Continues if some priorities can't be changed
- **Clean Exit**: Always restores on shutdown

### Process Exclusions

These processes are excluded from game detection and priority changes:

- System: `systemd`, `init`, `kernel`, `kworker`
- Display: `Xorg`, `gnome-shell`, `kwin`, `plasmashell`
- Monitoring: `glances`, `python`, `python3`

## Troubleshooting

### Game Not Detected

**Issue**: Plugin doesn't detect your game

**Solutions**:
1. Lower `game_cpu_threshold` in config (try 30.0)
2. Check if game is in exclusion list
3. Ensure game uses sufficient CPU/GPU
4. Increase logging: `glances --debug`

### Priority Changes Not Working

**Issue**: "Access denied" messages in logs

**Solution**: Run with sudo:
```bash
sudo glances
```

### False Positives

**Issue**: Non-game processes detected as games

**Solutions**:
1. Increase `game_cpu_threshold`
2. Increase `detection_history` (requires more consecutive detections)
3. Add process name to exclusion list in code

### FPS Estimates Inaccurate

**Issue**: FPS numbers don't match actual game FPS

**Note**: FPS estimation is approximate and heuristic-based. It's used for *trend detection* (identifying drops) rather than precise measurement. The absolute values may not match game's internal FPS counter.

## Architecture

The plugin consists of four main components:

1. **FpsDetector** (`fps_detector.py`): Estimates frame rates using CPU/GPU patterns
2. **GameDetector** (`game_detector.py`): Identifies game processes with scoring algorithm
3. **PriorityManager** (`priority_manager.py`): Handles CPU priority adjustments
4. **GamingPerformancePlugin** (`__init__.py`): Main plugin integrating all components

## License

SPDX-License-Identifier: LGPL-3.0-only
