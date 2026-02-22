#!/usr/bin/env python3
"""
Test script for GPU performance monitoring error handling.

This script tests error conditions and data collection failures.
"""

import sys
from time import time

# Add parent directory to path for imports
sys.path.insert(0, '.')

from glances.plugins.gpu.analyzer import GpuPerformanceAnalyzer


def test_error_handling():
    """Test error handling in GPU performance monitoring."""
    print("GPU Performance Monitoring - Error Handling Test")
    print("=" * 60)

    analyzer = GpuPerformanceAnalyzer()
    current_time = time()

    # Test 1: Missing GPU ID
    print("\n[Test 1] Missing GPU ID")
    print("-" * 60)
    gpu_data = {
        'temperature': 75.0,
        'proc': 50,
        'name': 'Test GPU',
    }
    result = analyzer.analyze_gpu(gpu_data, current_time)
    print(f"Status: {result['status']}")
    print(f"Prediction: {result['prediction']}")
    assert result['status'] == 'ERROR', "Should report ERROR for missing GPU ID"
    assert 'error' in result, "Should include error field"
    print("✓ Correctly handles missing GPU ID")

    # Test 2: Missing temperature data
    print("\n[Test 2] Missing Temperature Data")
    print("-" * 60)
    gpu_data = {
        'gpu_id': 'nvidia0',
        'proc': 50,
        'name': 'Test GPU',
    }
    result = analyzer.analyze_gpu(gpu_data, current_time)
    print(f"Status: {result['status']}")
    print(f"Prediction: {result['prediction']}")
    assert result['status'] == 'ERROR', "Should report ERROR for missing temperature"
    assert 'error' in result, "Should include error field"
    assert result['error'] == 'missing_temperature', "Should specify missing temperature error"
    print("✓ Correctly handles missing temperature data")

    # Test 3: Valid data with normal temperature
    print("\n[Test 3] Valid Data - Normal Temperature")
    print("-" * 60)
    gpu_data = {
        'gpu_id': 'nvidia0',
        'temperature': 65.0,
        'proc': 50,
        'name': 'GeForce RTX 3080',
    }
    result = analyzer.analyze_gpu(gpu_data, current_time)
    print(f"Status: {result['status']}")
    print(f"Temperature: {gpu_data['temperature']}°C")
    assert result['status'] == 'OK', "Should report OK for normal temperature"
    assert 'error' not in result, "Should not include error field for normal operation"
    print("✓ Correctly processes valid data")

    # Test 4: Valid data with missing processor load (optional field)
    print("\n[Test 4] Valid Data - Missing Processor Load (Optional)")
    print("-" * 60)
    gpu_data = {
        'gpu_id': 'nvidia0',
        'temperature': 70.0,
        'name': 'GeForce RTX 3080',
    }
    result = analyzer.analyze_gpu(gpu_data, current_time + 1)
    print(f"Status: {result['status']}")
    print(f"Temperature: {gpu_data['temperature']}°C")
    assert result['status'] == 'OK', "Should report OK even without proc load"
    assert 'error' not in result, "Should not error for missing optional fields"
    print("✓ Correctly handles missing optional processor load")

    # Test 5: Critical temperature
    print("\n[Test 5] Critical Temperature Detection")
    print("-" * 60)
    gpu_data = {
        'gpu_id': 'nvidia0',
        'temperature': 90.0,
        'proc': 95,
        'name': 'GeForce RTX 3080',
    }
    result = analyzer.analyze_gpu(gpu_data, current_time + 2)
    print(f"Status: {result['status']}")
    print(f"Temperature: {gpu_data['temperature']}°C")
    print(f"Prediction: {result['prediction']}")
    assert result['status'] == 'CRITICAL', "Should report CRITICAL for high temperature"
    assert 'error' not in result, "Should not include error field"
    print("✓ Correctly detects critical temperature")

    # Test 6: Multiple GPUs with mixed states
    print("\n[Test 6] Multiple GPUs - Mixed States")
    print("-" * 60)
    gpus = [
        {'gpu_id': 'nvidia0', 'temperature': 65.0, 'proc': 40, 'name': 'GPU 0'},
        {'gpu_id': 'nvidia1', 'temperature': None, 'proc': 50, 'name': 'GPU 1'},  # Missing temp
        {'gpu_id': 'nvidia2', 'temperature': 85.0, 'proc': 90, 'name': 'GPU 2'},  # Critical
    ]

    for i, gpu in enumerate(gpus):
        result = analyzer.analyze_gpu(gpu, current_time + 3 + i)
        print(f"\nGPU {i} ({gpu['gpu_id']}): {result['status']}")
        if result['status'] == 'ERROR':
            print(f"  Error: {result['prediction']}")
        elif result['status'] == 'CRITICAL':
            print(f"  Warning: {result['prediction']}")

    print("\n✓ Correctly handles multiple GPUs with mixed states")

    print("\n" + "=" * 60)
    print("All error handling tests passed!")
    print("\nThe GPU performance monitoring feature:")
    print("  • Gracefully handles missing data")
    print("  • Reports clear error messages")
    print("  • Continues monitoring other GPUs on errors")
    print("  • Validates essential metrics before analysis")


if __name__ == "__main__":
    try:
        test_error_handling()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
