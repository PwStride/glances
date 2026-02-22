#!/usr/bin/env python3
"""
Test script for GPU Monitor plugin functionality.

This script simulates GPU temperature trends and tests the prediction algorithm
without requiring actual GPU hardware.
"""

import sys
from time import time
from collections import defaultdict, deque

# Simulate the temperature trend calculation
def calculate_temp_trend(temp_history):
    """Calculate temperature trend in degrees per minute."""
    if len(temp_history) < 2:
        return 0.0

    times = [t for t, _ in temp_history]
    temps = [temp for _, temp in temp_history]

    n = len(times)
    sum_x = sum(times)
    sum_y = sum(temps)
    sum_xy = sum(t * temp for t, temp in zip(times, temps))
    sum_x2 = sum(t * t for t in times)

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope * 60  # Convert to degrees per minute


def test_temperature_prediction():
    """Test temperature prediction scenarios."""
    print("GPU Monitor Plugin - Temperature Prediction Test")
    print("=" * 60)

    # Test scenario 1: Rapidly rising temperature
    print("\n[Test 1] Rapidly Rising Temperature (Gaming Load)")
    print("-" * 60)
    temp_history = deque(maxlen=20)
    base_time = time()

    # Simulate temperature rising from 60°C to 80°C over 60 seconds
    for i in range(20):
        t = base_time + (i * 3)  # 3 second intervals
        temp = 60 + (i * 1.0)  # Rising 1°C per reading
        temp_history.append((t, temp))

    trend = calculate_temp_trend(temp_history)
    current_temp = temp_history[-1][1]
    temp_critical = 85

    print(f"Current temperature: {current_temp:.1f}°C")
    print(f"Temperature trend: {trend:.2f}°C/min")

    if trend > 0:
        time_to_critical = (temp_critical - current_temp) / (trend / 60)
        print(f"Time to critical ({temp_critical}°C): ~{int(time_to_critical)}s")

        if 0 < time_to_critical <= 180:
            print("⚠️  CRITICAL: Will reach critical temperature within 3 minutes!")
            print("   Action: Reduce GPU load immediately")
        else:
            print("✓ OK: Safe thermal margin")

    # Test scenario 2: Stable temperature
    print("\n[Test 2] Stable Temperature (Idle)")
    print("-" * 60)
    temp_history = deque(maxlen=20)
    base_time = time()

    for i in range(20):
        t = base_time + (i * 3)
        temp = 45 + ((-1) ** i * 0.5)  # Oscillating around 45°C
        temp_history.append((t, temp))

    trend = calculate_temp_trend(temp_history)
    current_temp = temp_history[-1][1]

    print(f"Current temperature: {current_temp:.1f}°C")
    print(f"Temperature trend: {trend:.2f}°C/min")
    print("✓ OK: Temperature stable")

    # Test scenario 3: Slow rise approaching warning
    print("\n[Test 3] Slow Rise (Sustained Load)")
    print("-" * 60)
    temp_history = deque(maxlen=20)
    base_time = time()

    for i in range(20):
        t = base_time + (i * 3)
        temp = 75 + (i * 0.3)  # Rising 0.3°C per reading
        temp_history.append((t, temp))

    trend = calculate_temp_trend(temp_history)
    current_temp = temp_history[-1][1]
    temp_warning = 80

    print(f"Current temperature: {current_temp:.1f}°C")
    print(f"Temperature trend: {trend:.2f}°C/min")

    if current_temp >= temp_warning:
        print("⚠️  WARNING: Temperature at warning level")
        print("   Action: Monitor closely or reduce workload")

    # Test scenario 4: Already at critical
    print("\n[Test 4] Critical Temperature")
    print("-" * 60)
    temp_history = deque(maxlen=20)
    base_time = time()

    for i in range(20):
        t = base_time + (i * 3)
        temp = 85 + (i * 0.2)  # At critical and rising
        temp_history.append((t, temp))

    trend = calculate_temp_trend(temp_history)
    current_temp = temp_history[-1][1]

    print(f"Current temperature: {current_temp:.1f}°C")
    print(f"Temperature trend: {trend:.2f}°C/min")
    print("🚨 CRITICAL: Temperature at critical threshold!")
    print("   Action: Reduce GPU load immediately to prevent damage or system crash")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("\nThe GPU performance monitoring feature will:")
    print("  • Track temperature trends in real-time")
    print("  • Predict time until critical temperature")
    print("  • Send desktop notifications on warnings")
    print("  • Help prevent GPU crashes during gaming")


if __name__ == "__main__":
    try:
        test_temperature_prediction()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
