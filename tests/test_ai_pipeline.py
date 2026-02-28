#!/usr/bin/env python3
"""
Tests for the AI Training Pipeline plugin.

Tests cover:
- GpuStarvationDetector: sliding-window imbalance metrics
- AiWorkloadDetector: signal-only (GPU+disk) workload detection with hysteresis
- PipelineRecommender: bottleneck classification and recommendation text
"""

import sys
import unittest

sys.path.insert(0, '.')

from glances.plugins.ai_pipeline.imbalance_detector import GpuStarvationDetector
from glances.plugins.ai_pipeline.recommender import PipelineRecommender
from glances.plugins.ai_pipeline.workload_detector import AiWorkloadDetector


# ---------------------------------------------------------------------------
# GpuStarvationDetector Tests
# ---------------------------------------------------------------------------

class TestGpuStarvationDetector(unittest.TestCase):

    def _make_detector(self, window=10, idle=15.0, sat=80.0):
        return GpuStarvationDetector(
            window_size=window,
            gpu_idle_threshold=idle,
            disk_saturation_threshold=sat,
        )

    # -- Empty state --

    def test_empty_window_returns_zero_metrics(self):
        d = self._make_detector()
        self.assertEqual(d.compute_starvation_pct(), 0.0)
        self.assertEqual(d.compute_disk_saturation_pct(), 0.0)
        self.assertEqual(d.compute_gpu_cv(), 0.0)
        self.assertEqual(d.compute_pipeline_score(), 100.0)

    # -- Starvation detection --

    def test_all_idle_samples_while_disk_saturated_yields_full_starvation(self):
        d = self._make_detector(window=10)
        # GPU always idle (5% < 15% threshold), disk always at 1 GB/s
        for _ in range(10):
            d.add_sample(gpu_proc=5.0, disk_bytes_per_sec=1_000_000_000)

        self.assertEqual(d.compute_starvation_pct(), 100.0)
        self.assertEqual(d.compute_pipeline_score(), 0.0)

    def test_all_busy_samples_yields_zero_starvation(self):
        d = self._make_detector(window=10)
        # GPU always busy (90% > 15% threshold)
        for _ in range(10):
            d.add_sample(gpu_proc=90.0, disk_bytes_per_sec=500_000_000)

        self.assertEqual(d.compute_starvation_pct(), 0.0)
        self.assertEqual(d.compute_pipeline_score(), 100.0)

    def test_half_starved_samples_yields_fifty_percent(self):
        d = self._make_detector(window=10)
        # Disk constant at 1 GB/s (so always at peak / saturated);
        # 5 samples GPU idle, 5 samples GPU busy
        for _ in range(5):
            d.add_sample(gpu_proc=5.0, disk_bytes_per_sec=1_000_000_000)
        for _ in range(5):
            d.add_sample(gpu_proc=90.0, disk_bytes_per_sec=1_000_000_000)

        self.assertEqual(d.compute_starvation_pct(), 50.0)
        self.assertEqual(d.compute_pipeline_score(), 50.0)

    def test_gpu_idle_but_disk_quiet_is_not_starvation(self):
        d = self._make_detector(window=10, sat=80.0)
        # Disk peak established at 1 GB/s, then current samples at 0 bytes/s
        d.add_sample(gpu_proc=5.0, disk_bytes_per_sec=1_000_000_000)
        for _ in range(9):
            d.add_sample(gpu_proc=5.0, disk_bytes_per_sec=0.0)

        # Only the first sample (disk at peak) counts as starvation
        # 1 / 10 = 10 %
        self.assertEqual(d.compute_starvation_pct(), 10.0)

    def test_window_respects_max_size(self):
        d = self._make_detector(window=5)
        for i in range(10):
            d.add_sample(gpu_proc=float(i * 10), disk_bytes_per_sec=100.0)
        self.assertEqual(len(d._window), 5)

    # -- Disk saturation --

    def test_disk_saturation_constant_throughput_is_100_pct(self):
        d = self._make_detector()
        for _ in range(10):
            d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=500_000_000)
        # Constant throughput → mean = peak → 100 %
        self.assertEqual(d.compute_disk_saturation_pct(), 100.0)

    def test_disk_saturation_half_peak(self):
        d = self._make_detector()
        # One high-peak sample then the rest at half
        d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=1_000_000_000)
        for _ in range(9):
            d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=500_000_000)
        sat = d.compute_disk_saturation_pct()
        # mean ≈ 550 MB/s, peak = 1000 MB/s → ≈55 %
        self.assertGreater(sat, 50.0)
        self.assertLess(sat, 60.0)

    def test_disk_saturation_zero_peak_returns_zero(self):
        d = self._make_detector()
        for _ in range(5):
            d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=0.0)
        self.assertEqual(d.compute_disk_saturation_pct(), 0.0)

    # -- GPU CV --

    def test_gpu_cv_constant_utilization_is_zero(self):
        d = self._make_detector()
        for _ in range(10):
            d.add_sample(gpu_proc=80.0, disk_bytes_per_sec=100.0)
        self.assertEqual(d.compute_gpu_cv(), 0.0)

    def test_gpu_cv_spiky_utilization_is_high(self):
        d = self._make_detector(window=10)
        # Alternating 100% / 0% → very spiky
        for i in range(10):
            d.add_sample(
                gpu_proc=100.0 if i % 2 == 0 else 0.0,
                disk_bytes_per_sec=100.0,
            )
        cv = d.compute_gpu_cv()
        # mean=50, stdev=50 → CV=100 %
        self.assertGreater(cv, 90.0)

    def test_gpu_cv_single_sample_returns_zero(self):
        d = self._make_detector()
        d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=100.0)
        self.assertEqual(d.compute_gpu_cv(), 0.0)

    def test_gpu_cv_zero_mean_returns_zero(self):
        d = self._make_detector()
        for _ in range(5):
            d.add_sample(gpu_proc=0.0, disk_bytes_per_sec=100.0)
        self.assertEqual(d.compute_gpu_cv(), 0.0)

    # -- Pipeline score --

    def test_pipeline_score_never_negative(self):
        d = self._make_detector(window=5)
        for _ in range(5):
            d.add_sample(gpu_proc=1.0, disk_bytes_per_sec=1_000_000_000)
        self.assertGreaterEqual(d.compute_pipeline_score(), 0.0)

    def test_get_window_stats_returns_four_metrics(self):
        d = self._make_detector()
        for _ in range(5):
            d.add_sample(gpu_proc=50.0, disk_bytes_per_sec=100.0)
        score, starv, disk_sat, cv = d.get_window_stats()
        self.assertIsInstance(score, float)
        self.assertIsInstance(starv, float)
        self.assertIsInstance(disk_sat, float)
        self.assertIsInstance(cv, float)


# ---------------------------------------------------------------------------
# AiWorkloadDetector Tests
# ---------------------------------------------------------------------------

class TestAiWorkloadDetector(unittest.TestCase):

    def _make_detector(self, history=3, gpu_thresh=20.0, disk_thresh=1_000_000.0):
        return AiWorkloadDetector(
            detection_history=history,
            gpu_active_threshold=gpu_thresh,
            disk_active_bytes=disk_thresh,
        )

    def test_not_confirmed_before_history_count(self):
        d = self._make_detector(history=3)
        self.assertFalse(d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0))
        self.assertFalse(d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0))
        # Not yet at 3 consecutive
        self.assertFalse(d.is_confirmed)

    def test_confirmed_after_history_count(self):
        d = self._make_detector(history=3)
        for _ in range(3):
            result = d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        self.assertTrue(result)
        self.assertTrue(d.is_confirmed)

    def test_cleared_after_history_count_negative(self):
        d = self._make_detector(history=3)
        # Confirm
        for _ in range(3):
            d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        self.assertTrue(d.is_confirmed)
        # Clear
        for _ in range(3):
            result = d.detect(gpu_proc=0.0, disk_bytes_per_sec=0.0)
        self.assertFalse(result)
        self.assertFalse(d.is_confirmed)

    def test_stays_confirmed_during_brief_lull(self):
        d = self._make_detector(history=3)
        # Confirm
        for _ in range(3):
            d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        self.assertTrue(d.is_confirmed)
        # One negative tick — still confirmed
        result = d.detect(gpu_proc=0.0, disk_bytes_per_sec=0.0)
        self.assertTrue(result)

    def test_gpu_none_never_confirms(self):
        d = self._make_detector(history=2)
        for _ in range(5):
            result = d.detect(gpu_proc=None, disk_bytes_per_sec=100_000_000.0)
        self.assertFalse(result)
        self.assertFalse(d.is_confirmed)

    def test_gpu_below_threshold_does_not_confirm(self):
        d = self._make_detector(history=2, gpu_thresh=20.0)
        for _ in range(5):
            result = d.detect(gpu_proc=10.0, disk_bytes_per_sec=100_000_000.0)
        self.assertFalse(result)

    def test_disk_below_threshold_does_not_confirm(self):
        d = self._make_detector(history=2, disk_thresh=1_000_000.0)
        for _ in range(5):
            result = d.detect(gpu_proc=80.0, disk_bytes_per_sec=100.0)
        self.assertFalse(result)

    def test_both_signals_required(self):
        d = self._make_detector(history=2)
        # GPU only
        for _ in range(5):
            d.detect(gpu_proc=90.0, disk_bytes_per_sec=0.0)
        self.assertFalse(d.is_confirmed)
        # Disk only
        for _ in range(5):
            d.detect(gpu_proc=0.0, disk_bytes_per_sec=100_000_000.0)
        self.assertFalse(d.is_confirmed)

    def test_positive_streak_reset_by_negative(self):
        d = self._make_detector(history=3)
        # 2 positive, 1 negative, 2 positive → never hits 3 consecutive
        d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        d.detect(gpu_proc=0.0, disk_bytes_per_sec=0.0)
        d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        result = d.detect(gpu_proc=80.0, disk_bytes_per_sec=10_000_000.0)
        # Only 2 consecutive positives after the break
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# PipelineRecommender Tests
# ---------------------------------------------------------------------------

class TestPipelineRecommender(unittest.TestCase):

    def setUp(self):
        self.rec = PipelineRecommender()

    def test_balanced_when_starvation_low(self):
        btype, msg = self.rec.get_recommendation(
            starvation_pct=5.0,
            disk_saturation_pct=90.0,
            gpu_mem_pct=90.0,
            cpu_percent=90.0,
        )
        self.assertEqual(btype, 'balanced')
        self.assertIn('balanced', msg.lower())

    def test_disk_bound_when_disk_saturated(self):
        btype, msg = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=85.0,
            gpu_mem_pct=40.0,
            cpu_percent=30.0,
        )
        self.assertEqual(btype, 'disk_bound')
        self.assertIn('DataLoader', msg)

    def test_cpu_bound_when_cpu_saturated(self):
        btype, msg = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=50.0,
            gpu_mem_pct=40.0,
            cpu_percent=90.0,
        )
        self.assertEqual(btype, 'cpu_bound')
        self.assertIn('num_workers', msg)

    def test_memory_bound_when_gpu_mem_high(self):
        btype, msg = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=50.0,
            gpu_mem_pct=92.0,
            cpu_percent=30.0,
        )
        self.assertEqual(btype, 'memory_bound')
        self.assertIn('batch', msg.lower())

    def test_memory_bound_priority_over_cpu_bound(self):
        # Both CPU and GPU mem high → memory_bound wins
        btype, _ = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=50.0,
            gpu_mem_pct=92.0,
            cpu_percent=95.0,
        )
        self.assertEqual(btype, 'memory_bound')

    def test_cpu_bound_priority_over_disk_bound(self):
        # Both CPU and disk high → cpu_bound wins
        btype, _ = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=90.0,
            gpu_mem_pct=40.0,
            cpu_percent=95.0,
        )
        self.assertEqual(btype, 'cpu_bound')

    def test_defaults_to_disk_bound_when_no_clear_signal(self):
        # Starvation present but neither CPU nor memory high, disk moderate
        btype, _ = self.rec.get_recommendation(
            starvation_pct=30.0,
            disk_saturation_pct=30.0,
            gpu_mem_pct=40.0,
            cpu_percent=30.0,
        )
        self.assertEqual(btype, 'disk_bound')

    def test_handles_none_cpu_percent(self):
        btype, _ = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=90.0,
            gpu_mem_pct=40.0,
            cpu_percent=None,
        )
        self.assertEqual(btype, 'disk_bound')

    def test_handles_none_gpu_mem(self):
        btype, _ = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=90.0,
            gpu_mem_pct=None,
            cpu_percent=95.0,
        )
        self.assertEqual(btype, 'cpu_bound')

    def test_handles_all_none_optional_args(self):
        btype, _ = self.rec.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=90.0,
        )
        self.assertEqual(btype, 'disk_bound')

    def test_starvation_boundary_balanced(self):
        # Exactly at threshold (10.0) → not balanced
        btype, _ = self.rec.get_recommendation(
            starvation_pct=10.0,
            disk_saturation_pct=90.0,
        )
        self.assertNotEqual(btype, 'balanced')
        # Just below threshold → balanced
        btype, _ = self.rec.get_recommendation(
            starvation_pct=9.9,
            disk_saturation_pct=90.0,
        )
        self.assertEqual(btype, 'balanced')


# ---------------------------------------------------------------------------
# Integration: end-to-end starvation → bottleneck → recommendation flow
# ---------------------------------------------------------------------------

class TestEndToEndFlow(unittest.TestCase):

    def test_full_starvation_scenario_yields_disk_bound(self):
        """Simulate the classic GPU starvation pattern and confirm
        all components agree on the diagnosis."""
        starv_det = GpuStarvationDetector(window_size=10)
        workload = AiWorkloadDetector(detection_history=3)
        recommender = PipelineRecommender()

        # Simulate 10 samples: GPU sawtooth (5% then 90%), disk always at 1 GB/s
        disk_rate = 1_000_000_000.0
        for i in range(10):
            gpu = 5.0 if i % 2 == 0 else 90.0
            starv_det.add_sample(gpu_proc=gpu, disk_bytes_per_sec=disk_rate)
            workload.detect(gpu_proc=gpu, disk_bytes_per_sec=disk_rate)

        # Half the samples are starvation events → 50 %
        self.assertEqual(starv_det.compute_starvation_pct(), 50.0)
        self.assertEqual(starv_det.compute_pipeline_score(), 50.0)

        # GPU CV should be high (spiky pattern)
        self.assertGreater(starv_det.compute_gpu_cv(), 50.0)

        # Recommender should classify as disk_bound
        btype, msg = recommender.get_recommendation(
            starvation_pct=50.0,
            disk_saturation_pct=100.0,
            gpu_mem_pct=40.0,
            cpu_percent=30.0,
        )
        self.assertEqual(btype, 'disk_bound')
        self.assertIn('DataLoader', msg)


if __name__ == '__main__':
    unittest.main()
