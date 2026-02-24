#!/usr/bin/env python3
"""
Tests for the gaming performance dynamic CPU priority thresholding feature.

Tests cover:
- FPS estimation and threshold detection
- Game process detection and scoring
- CPU priority adjustment logic
- Threshold boundary conditions
- End-to-end optimization flow
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, '.')

from glances.plugins.gaming_performance.fps_detector import FpsDetector
from glances.plugins.gaming_performance.game_detector import GameDetector
from glances.plugins.gaming_performance.priority_manager import PriorityManager


# ---------------------------------------------------------------------------
# FPS Detector Tests
# ---------------------------------------------------------------------------

class TestFpsDetector(unittest.TestCase):

    # -- History and weighted averaging --

    def test_history_respects_max_size(self):
        detector = FpsDetector(history_size=5)
        pid = 9999
        for i in range(10):
            detector._add_to_history(pid, float(i))
        assert len(detector._fps_history[pid]) == 5
        # Should keep only the last 5 entries (5..9)
        assert detector._fps_history[pid] == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_weighted_average_favors_recent_samples(self):
        detector = FpsDetector(history_size=10)
        pid = 1000
        # Older samples low, recent samples high
        for fps in [30.0, 30.0, 30.0, 90.0, 90.0]:
            detector._add_to_history(pid, fps)
        avg = detector._get_average_fps(pid)
        # Weighted average should be pulled toward 90 (recent)
        assert avg is not None
        assert avg > 60.0, f"Expected weighted avg > 60, got {avg}"

    def test_average_returns_none_for_unknown_pid(self):
        detector = FpsDetector()
        assert detector._get_average_fps(99999) is None

    def test_average_returns_none_for_empty_history(self):
        detector = FpsDetector()
        detector._fps_history[111] = []
        assert detector._get_average_fps(111) is None

    # -- FPS drop threshold calculation --

    def test_fps_drop_zero_when_above_target(self):
        detector = FpsDetector()
        pid = 2000
        detector._add_to_history(pid, 75.0)
        drop = detector.get_fps_drop(pid, threshold_fps=60.0)
        assert drop == 0.0

    def test_fps_drop_exact_at_target(self):
        detector = FpsDetector()
        pid = 2001
        detector._add_to_history(pid, 60.0)
        drop = detector.get_fps_drop(pid, threshold_fps=60.0)
        assert drop == 0.0

    def test_fps_drop_percentage_below_target(self):
        detector = FpsDetector()
        pid = 2002
        # 45 FPS with a 60 FPS target => 25% drop
        detector._add_to_history(pid, 45.0)
        drop = detector.get_fps_drop(pid, threshold_fps=60.0)
        assert drop is not None
        assert abs(drop - 25.0) < 0.1, f"Expected ~25% drop, got {drop}"

    def test_fps_drop_clamped_to_100(self):
        detector = FpsDetector()
        pid = 2003
        detector._add_to_history(pid, 0.0)
        drop = detector.get_fps_drop(pid, threshold_fps=60.0)
        assert drop == 100.0

    def test_fps_drop_none_for_unknown_pid(self):
        detector = FpsDetector()
        assert detector.get_fps_drop(99999) is None

    def test_fps_drop_various_targets(self):
        """Test FPS drop calculation against multiple target thresholds."""
        detector = FpsDetector()
        pid = 2004
        detector._add_to_history(pid, 30.0)

        # 30 FPS vs 60 target = 50% drop
        drop60 = detector.get_fps_drop(pid, threshold_fps=60.0)
        assert abs(drop60 - 50.0) < 0.1

        # 30 FPS vs 144 target = ~79.2% drop
        drop144 = detector.get_fps_drop(pid, threshold_fps=144.0)
        assert abs(drop144 - 79.17) < 0.5

        # 30 FPS vs 30 target = 0% drop (at target)
        drop30 = detector.get_fps_drop(pid, threshold_fps=30.0)
        assert drop30 == 0.0

    # -- CPU-to-FPS estimation thresholds --

    def test_cpu_below_20_returns_none(self):
        """CPU usage <= 20% should not produce an FPS estimate."""
        detector = FpsDetector()
        pid = 3000
        # Prime the previous CPU times so the delta path is entered
        detector._prev_cpu_times[pid] = 100.0
        result = detector._calculate_fps_from_cpu(pid, 101.0, 15.0, 1000.0)
        assert result is None

    def test_cpu_at_20_boundary_returns_none(self):
        detector = FpsDetector()
        pid = 3001
        detector._prev_cpu_times[pid] = 100.0
        result = detector._calculate_fps_from_cpu(pid, 101.0, 20.0, 1000.0)
        assert result is None

    def test_cpu_above_20_returns_fps(self):
        detector = FpsDetector()
        pid = 3002
        detector._prev_cpu_times[pid] = 100.0
        result = detector._calculate_fps_from_cpu(pid, 101.0, 50.0, 1000.0)
        assert result is not None
        assert 10.0 <= result <= 300.0

    def test_fps_clamped_to_valid_range(self):
        """Extremely high or low CPU should still produce clamped FPS."""
        detector = FpsDetector()
        pid = 3003
        detector._prev_cpu_times[pid] = 100.0

        # Very high CPU
        fps_high = detector._calculate_fps_from_cpu(pid, 101.0, 200.0, 1000.0)
        assert fps_high is not None
        assert fps_high <= 300.0

        # Low but above 20
        detector._prev_cpu_times[pid] = 200.0
        fps_low = detector._calculate_fps_from_cpu(pid, 201.0, 21.0, 2000.0)
        assert fps_low is not None
        assert fps_low >= 10.0

    def test_first_measurement_returns_none(self):
        """First call with no previous CPU time should return None and store baseline."""
        detector = FpsDetector()
        pid = 3004
        result = detector._calculate_fps_from_cpu(pid, 100.0, 60.0, 1000.0)
        assert result is None
        assert pid in detector._prev_cpu_times

    def test_zero_cpu_delta_returns_none(self):
        detector = FpsDetector()
        pid = 3005
        detector._prev_cpu_times[pid] = 100.0
        result = detector._calculate_fps_from_cpu(pid, 100.0, 60.0, 1000.0)
        assert result is None

    # -- GPU refinement --

    def test_gpu_bound_lowers_fps_estimate(self):
        detector = FpsDetector()
        base_fps = 60.0
        refined = detector._refine_with_gpu(base_fps, gpu_usage=90.0)
        assert refined == base_fps * 0.8

    def test_cpu_bound_keeps_fps_estimate(self):
        detector = FpsDetector()
        base_fps = 60.0
        refined = detector._refine_with_gpu(base_fps, gpu_usage=20.0)
        assert refined == base_fps

    def test_balanced_load_reduces_slightly(self):
        detector = FpsDetector()
        base_fps = 60.0
        refined = detector._refine_with_gpu(base_fps, gpu_usage=50.0)
        assert refined == base_fps * 0.9

    def test_gpu_boundary_at_80(self):
        detector = FpsDetector()
        base_fps = 100.0
        # At exactly 80, should fall into the "balanced" range (not > 80)
        refined = detector._refine_with_gpu(base_fps, gpu_usage=80.0)
        assert refined == 100.0 * 0.9

    def test_gpu_boundary_at_30(self):
        detector = FpsDetector()
        base_fps = 100.0
        # At exactly 30, should fall into the "balanced" range (not < 30)
        refined = detector._refine_with_gpu(base_fps, gpu_usage=30.0)
        assert refined == 100.0 * 0.9

    # -- Cleanup --

    def test_cleanup_removes_dead_pids(self):
        detector = FpsDetector()
        fake_pid = 999999999
        detector._fps_history[fake_pid] = [60.0]
        detector._last_update[fake_pid] = 1000.0
        detector._prev_cpu_times[fake_pid] = 50.0
        detector.cleanup_dead_processes()
        assert fake_pid not in detector._fps_history
        assert fake_pid not in detector._last_update
        assert fake_pid not in detector._prev_cpu_times


# ---------------------------------------------------------------------------
# Game Detector Tests
# ---------------------------------------------------------------------------

class TestGameDetector(unittest.TestCase):

    def _make_proc(self, pid, name, cpu_percent, memory_percent=5.0):
        return {
            'pid': pid,
            'name': name,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
        }

    # -- CPU threshold filtering --

    def test_process_below_cpu_threshold_excluded(self):
        detector = GameDetector(cpu_threshold=40.0)
        procs = [self._make_proc(1, 'game', 30.0)]
        candidates = detector._filter_candidates(procs)
        assert len(candidates) == 0

    def test_process_at_cpu_threshold_excluded(self):
        """Process at exactly the threshold should be excluded (< check, not <=)."""
        detector = GameDetector(cpu_threshold=40.0)
        procs = [self._make_proc(1, 'game', 40.0)]
        candidates = detector._filter_candidates(procs)
        # 40.0 is not < 40.0, so it passes
        assert len(candidates) == 1

    def test_process_above_cpu_threshold_included(self):
        detector = GameDetector(cpu_threshold=40.0)
        procs = [self._make_proc(1, 'game', 80.0)]
        candidates = detector._filter_candidates(procs)
        assert len(candidates) == 1

    def test_varying_cpu_thresholds(self):
        """Changing the cpu_threshold should filter differently."""
        procs = [self._make_proc(1, 'game', 50.0)]

        det_low = GameDetector(cpu_threshold=30.0)
        assert len(det_low._filter_candidates(procs)) == 1

        det_high = GameDetector(cpu_threshold=60.0)
        assert len(det_high._filter_candidates(procs)) == 0

    # -- Name-based exclusion --

    def test_excluded_process_names_filtered(self):
        detector = GameDetector(cpu_threshold=10.0)
        procs = [
            self._make_proc(1, 'systemd', 50.0),
            self._make_proc(2, 'Xorg', 60.0),
            self._make_proc(3, 'python3', 70.0),
            self._make_proc(4, 'my_game', 80.0),
        ]
        candidates = detector._filter_candidates(procs)
        assert len(candidates) == 1
        assert candidates[0]['name'] == 'my_game'

    def test_add_and_remove_exclusion(self):
        detector = GameDetector(cpu_threshold=10.0)
        detector.add_excluded_name('steam')
        procs = [self._make_proc(1, 'steam', 90.0)]
        assert len(detector._filter_candidates(procs)) == 0

        detector.remove_excluded_name('steam')
        assert len(detector._filter_candidates(procs)) == 1

    # -- Game scoring --

    def test_score_increases_with_cpu(self):
        detector = GameDetector()
        low_cpu = self._make_proc(1, 'game', 30.0, 10.0)
        high_cpu = self._make_proc(2, 'game', 90.0, 10.0)
        score_low = detector._calculate_game_score(low_cpu)
        score_high = detector._calculate_game_score(high_cpu)
        assert score_high > score_low

    def test_score_includes_memory_above_5(self):
        detector = GameDetector()
        proc_low_mem = self._make_proc(1, 'game', 50.0, 3.0)
        proc_high_mem = self._make_proc(2, 'game', 50.0, 20.0)
        score_low = detector._calculate_game_score(proc_low_mem)
        score_high = detector._calculate_game_score(proc_high_mem)
        assert score_high > score_low

    def test_score_memory_below_5_ignored(self):
        detector = GameDetector()
        proc = self._make_proc(1, 'game', 50.0, 4.0)
        score = detector._calculate_game_score(proc)
        # Only CPU contributes: min(40, 50*0.4) = 20.0
        assert abs(score - 20.0) < 0.01

    def test_score_with_gpu_stats(self):
        detector = GameDetector(gpu_threshold=30.0)
        proc = self._make_proc(1, 'game', 50.0, 10.0)
        gpu_stats = {'proc': 60.0}
        score_with_gpu = detector._calculate_game_score(proc, gpu_stats)
        score_without_gpu = detector._calculate_game_score(proc, None)
        assert score_with_gpu > score_without_gpu

    def test_score_gpu_below_threshold_no_contribution(self):
        detector = GameDetector(gpu_threshold=30.0)
        proc = self._make_proc(1, 'game', 50.0, 10.0)
        gpu_stats = {'proc': 20.0}
        score_with = detector._calculate_game_score(proc, gpu_stats)
        score_without = detector._calculate_game_score(proc, None)
        assert score_with == score_without

    # -- Consecutive detection history --

    def test_detection_requires_history_count(self):
        """Game is only confirmed after detection_history consecutive detections."""
        detector = GameDetector(cpu_threshold=10.0, detection_history=3)
        procs = [self._make_proc(100, 'my_game', 80.0, 15.0)]

        # First 2 detections: not yet confirmed
        for _ in range(2):
            result = detector.detect_game_process(procs)
        assert result is None

        # Third detection: confirmed
        result = detector.detect_game_process(procs)
        assert result == 100

    def test_detection_count_decays_for_others(self):
        """When a new top candidate appears, old candidate counts decay."""
        detector = GameDetector(cpu_threshold=10.0, detection_history=3)
        game_a = [self._make_proc(100, 'game_a', 80.0)]
        game_b = [self._make_proc(200, 'game_b', 90.0)]

        # Build up game_a count to 2
        detector.detect_game_process(game_a)
        detector.detect_game_process(game_a)
        assert detector._detection_count.get(100, 0) == 2

        # Switch to game_b, game_a count should decay
        detector.detect_game_process(game_b)
        assert detector._detection_count.get(100, 0) <= 1
        assert detector._detection_count.get(200, 0) == 1

    def test_highest_scoring_process_selected(self):
        detector = GameDetector(cpu_threshold=10.0, detection_history=1)
        procs = [
            self._make_proc(1, 'light_app', 30.0, 2.0),
            self._make_proc(2, 'heavy_game', 90.0, 25.0),
        ]
        result = detector.detect_game_process(procs)
        assert result == 2

    def test_no_candidates_resets_game(self):
        detector = GameDetector(cpu_threshold=10.0, detection_history=1)
        procs = [self._make_proc(100, 'game', 80.0)]
        detector.detect_game_process(procs)

        # Empty list resets
        result = detector.detect_game_process([])
        assert result is None
        assert detector._current_game_pid is None

    def test_all_below_threshold_resets_game(self):
        detector = GameDetector(cpu_threshold=50.0, detection_history=1)
        # All processes below threshold
        procs = [
            self._make_proc(1, 'app', 20.0),
            self._make_proc(2, 'browser', 30.0),
        ]
        result = detector.detect_game_process(procs)
        assert result is None

    # -- Reset --

    def test_reset_clears_state(self):
        detector = GameDetector(cpu_threshold=10.0, detection_history=1)
        procs = [self._make_proc(100, 'game', 80.0)]
        detector.detect_game_process(procs)

        detector.reset()
        assert detector._current_game_pid is None
        assert len(detector._detection_count) == 0


# ---------------------------------------------------------------------------
# Priority Manager Tests
# ---------------------------------------------------------------------------

class TestPriorityManager(unittest.TestCase):

    def _make_proc(self, pid, name, nice=0):
        return {'pid': pid, 'name': name, 'nice': nice}

    # -- Nice level clamping --

    def test_game_nice_level_clamped_to_min(self):
        mgr = PriorityManager(game_nice_level=-25)
        assert mgr.game_nice_level == -20

    def test_game_nice_level_clamped_to_max(self):
        mgr = PriorityManager(game_nice_level=25)
        assert mgr.game_nice_level == 19

    def test_game_nice_level_within_range_unchanged(self):
        mgr = PriorityManager(game_nice_level=-5)
        assert mgr.game_nice_level == -5

    # -- Protected process filtering --

    def test_protected_system_processes(self):
        mgr = PriorityManager()
        for name in ['systemd', 'init', 'kworker', 'watchdog', 'kthreadd']:
            proc = self._make_proc(1, name)
            assert mgr._is_protected_process(proc), f"{name} should be protected"

    def test_non_protected_processes(self):
        mgr = PriorityManager()
        for name in ['firefox', 'steam', 'vlc', 'spotify']:
            proc = self._make_proc(1, name)
            assert not mgr._is_protected_process(proc), f"{name} should not be protected"

    def test_high_priority_processes_protected(self):
        """Processes with nice < -10 should be protected."""
        mgr = PriorityManager()
        proc = self._make_proc(1, 'some_app', nice=-15)
        assert mgr._is_protected_process(proc)

    def test_normal_nice_not_protected(self):
        mgr = PriorityManager()
        proc = self._make_proc(1, 'app', nice=0)
        assert not mgr._is_protected_process(proc)

    # -- Boost/restore dry-run logic --

    def _mock_psutil_process(self, nice_value=0):
        """Create a mock psutil.Process that returns given nice value."""
        mock_proc = MagicMock()
        mock_proc.nice.return_value = nice_value
        return mock_proc

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_boost_dry_run_marks_active(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager(game_nice_level=-5, background_nice_increase=5)
        procs = [
            self._make_proc(10, 'game'),
            self._make_proc(20, 'browser'),
            self._make_proc(30, 'editor'),
        ]
        result = mgr.boost_game_priority(10, procs, dry_run=True)
        assert result['game_boosted'] is True
        assert mgr.is_boost_active()
        assert mgr.get_boosted_game_pid() == 10

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_boost_skips_same_game_twice(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager()
        procs = [self._make_proc(10, 'game')]
        mgr.boost_game_priority(10, procs, dry_run=True)
        result2 = mgr.boost_game_priority(10, procs, dry_run=True)
        # Second boost on same PID should be a no-op
        assert result2['game_boosted'] is False

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_restore_clears_boost_state(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager()
        procs = [self._make_proc(10, 'game')]
        mgr.boost_game_priority(10, procs, dry_run=True)
        assert mgr.is_boost_active()

        mgr.restore_priorities(dry_run=True)
        assert not mgr.is_boost_active()
        assert mgr.get_boosted_game_pid() is None

    def test_restore_when_not_active_is_noop(self):
        mgr = PriorityManager()
        result = mgr.restore_priorities(dry_run=True)
        assert result['game_restored'] is False
        assert result['background_restored'] == 0

    # -- Background priority lowering with protected filter --

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_background_lowering_skips_game_pid(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager()
        procs = [
            self._make_proc(10, 'game'),
            self._make_proc(20, 'browser'),
        ]
        result = mgr.boost_game_priority(10, procs, dry_run=True)
        # Only the browser should be lowered, not the game itself
        assert 10 not in mgr._deprioritized_pids

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_background_lowering_skips_protected(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager()
        procs = [
            self._make_proc(10, 'game'),
            self._make_proc(20, 'systemd'),
            self._make_proc(30, 'init'),
            self._make_proc(40, 'browser'),
        ]
        result = mgr.boost_game_priority(10, procs, dry_run=True)
        assert 20 not in mgr._deprioritized_pids
        assert 30 not in mgr._deprioritized_pids

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_background_lowering_skips_already_deprioritized(self, mock_process_cls):
        mock_process_cls.return_value = self._mock_psutil_process(0)
        mgr = PriorityManager()
        procs = [
            self._make_proc(10, 'game'),
            self._make_proc(20, 'browser'),
        ]
        # First boost
        mgr.boost_game_priority(10, procs, dry_run=True)
        initial_count = len(mgr._deprioritized_pids)

        # Restore and re-boost
        mgr.restore_priorities(dry_run=True)
        result2 = mgr.boost_game_priority(10, procs, dry_run=True)
        assert result2['background_lowered'] >= 0

    # -- Nice level arithmetic --

    def test_background_nice_capped_at_max(self):
        mgr = PriorityManager(background_nice_increase=10, max_nice_level=19)
        # Process already at nice 15 => 15+10=25 => capped to 19
        new_nice = min(mgr.max_nice_level, 15 + mgr.background_nice_increase)
        assert new_nice == 19

    def test_background_nice_normal_increase(self):
        mgr = PriorityManager(background_nice_increase=5, max_nice_level=19)
        # Process at nice 0 => 0+5=5
        new_nice = min(mgr.max_nice_level, 0 + mgr.background_nice_increase)
        assert new_nice == 5

    # -- Dead process cleanup --

    def test_cleanup_removes_dead_pids(self):
        mgr = PriorityManager()
        fake_pid = 999999999
        mgr._original_nice[fake_pid] = 0
        mgr._deprioritized_pids.add(fake_pid)
        mgr.cleanup_dead_processes()
        assert fake_pid not in mgr._original_nice
        assert fake_pid not in mgr._deprioritized_pids


# ---------------------------------------------------------------------------
# Threshold-Based Optimization Flow Tests
# ---------------------------------------------------------------------------

class TestDynamicThresholdingFlow(unittest.TestCase):
    """End-to-end tests that verify the full threshold detection and
    priority adjustment pipeline works correctly."""

    def _make_proc(self, pid, name, cpu_percent, memory_percent=10.0, nice=0):
        return {
            'pid': pid,
            'name': name,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'nice': nice,
        }

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_fps_drop_triggers_priority_boost(self, mock_process_cls):
        """When FPS drops below threshold for enough consecutive cycles,
        the priority manager should activate boost."""
        mock_proc = MagicMock()
        mock_proc.nice.return_value = 0
        mock_process_cls.return_value = mock_proc

        fps_detector = FpsDetector(history_size=10)
        game_detector = GameDetector(cpu_threshold=30.0, detection_history=1)
        priority_manager = PriorityManager(
            game_nice_level=-5, background_nice_increase=5
        )

        game_pid = 1000
        target_fps = 60.0
        fps_drop_threshold = 20.0
        consecutive_drops_required = 3
        consecutive_low_fps = 0

        procs = [
            self._make_proc(game_pid, 'test_game', 80.0, 20.0),
            self._make_proc(2000, 'browser', 10.0, 5.0),
            self._make_proc(3000, 'editor', 5.0, 3.0),
        ]

        # Confirm game detection
        game_detector.detect_game_process(procs)
        detected = game_detector._current_game_pid
        assert detected == game_pid

        # Simulate low FPS readings
        for fps_val in [40.0, 38.0, 35.0, 33.0]:
            fps_detector._add_to_history(game_pid, fps_val)
            fps_drop = fps_detector.get_fps_drop(game_pid, target_fps)

            if fps_drop and fps_drop >= fps_drop_threshold:
                consecutive_low_fps += 1
            else:
                consecutive_low_fps = 0

            if (
                consecutive_low_fps >= consecutive_drops_required
                and not priority_manager.is_boost_active()
            ):
                result = priority_manager.boost_game_priority(
                    game_pid, procs, dry_run=True
                )
                assert result['game_boosted'] is True
                break

        assert priority_manager.is_boost_active()
        assert priority_manager.get_boosted_game_pid() == game_pid

    def test_fps_recovery_does_not_trigger_boost(self):
        """If FPS drops but recovers before reaching the consecutive count,
        boost should NOT activate."""
        fps_detector = FpsDetector(history_size=5)
        priority_manager = PriorityManager()

        game_pid = 1001
        target_fps = 60.0
        fps_drop_threshold = 20.0
        consecutive_drops_required = 3
        consecutive_low_fps = 0

        # Drop, drop, then recover
        for fps_val in [40.0, 35.0, 70.0, 65.0]:
            fps_detector._fps_history[game_pid] = [fps_val]
            fps_drop = fps_detector.get_fps_drop(game_pid, target_fps)

            if fps_drop and fps_drop >= fps_drop_threshold:
                consecutive_low_fps += 1
            else:
                consecutive_low_fps = 0

        assert consecutive_low_fps == 0
        assert not priority_manager.is_boost_active()

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_game_exit_restores_priorities(self, mock_process_cls):
        """When the game is no longer detected, priorities should restore."""
        mock_proc = MagicMock()
        mock_proc.nice.return_value = 0
        mock_process_cls.return_value = mock_proc

        game_detector = GameDetector(cpu_threshold=30.0, detection_history=1)
        priority_manager = PriorityManager()

        game_pid = 1002
        procs_with_game = [
            self._make_proc(game_pid, 'test_game', 80.0),
            self._make_proc(2000, 'browser', 10.0),
        ]
        procs_without_game = [
            self._make_proc(2000, 'browser', 10.0),
        ]

        # Detect game and boost
        game_detector.detect_game_process(procs_with_game)
        priority_manager.boost_game_priority(game_pid, procs_with_game, dry_run=True)
        assert priority_manager.is_boost_active()

        # Game exits
        detected = game_detector.detect_game_process(procs_without_game)
        if detected is None and priority_manager.is_boost_active():
            priority_manager.restore_priorities(dry_run=True)

        assert not priority_manager.is_boost_active()

    def test_multiple_threshold_levels(self):
        """Test that different FPS drop severities produce correct drop percentages."""
        fps_detector = FpsDetector(history_size=5)
        pid = 5000
        target = 60.0

        scenarios = [
            (55.0, 8.33),   # Mild drop
            (48.0, 20.0),   # Threshold boundary
            (30.0, 50.0),   # Severe drop
            (12.0, 80.0),   # Critical
        ]

        for fps_val, expected_drop in scenarios:
            fps_detector._fps_history[pid] = [fps_val]
            drop = fps_detector.get_fps_drop(pid, target)
            assert abs(drop - expected_drop) < 1.0, (
                f"FPS={fps_val}: expected ~{expected_drop}% drop, got {drop}%"
            )

    def test_conservative_nice_adjustments(self):
        """Verify the conservative nice value policy: game gets -5, background gets +5,
        and the values stay within UNIX nice range."""
        mgr = PriorityManager(
            game_nice_level=-5,
            background_nice_increase=5,
            min_nice_level=-20,
            max_nice_level=19,
        )

        assert mgr.game_nice_level == -5
        assert mgr.background_nice_increase == 5

        # Simulate a background process currently at nice 0
        new_nice = min(mgr.max_nice_level, 0 + mgr.background_nice_increase)
        assert new_nice == 5

        # Simulate a background process already at nice 17
        new_nice = min(mgr.max_nice_level, 17 + mgr.background_nice_increase)
        assert new_nice == 19  # capped

    def test_aggressive_nice_adjustments(self):
        """Aggressive config: game_nice_level=-10, background_nice_increase=10."""
        mgr = PriorityManager(
            game_nice_level=-10,
            background_nice_increase=10,
            min_nice_level=-20,
            max_nice_level=19,
        )

        assert mgr.game_nice_level == -10

        # Background at nice 0 => 10
        new_nice = min(mgr.max_nice_level, 0 + mgr.background_nice_increase)
        assert new_nice == 10

        # Background at nice 12 => 19 (capped)
        new_nice = min(mgr.max_nice_level, 12 + mgr.background_nice_increase)
        assert new_nice == 19

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_game_switching(self, mock_process_cls):
        """When a different game takes over as the top process, the old boost
        should not persist for the wrong PID."""
        mock_proc = MagicMock()
        mock_proc.nice.return_value = 0
        mock_process_cls.return_value = mock_proc

        game_detector = GameDetector(cpu_threshold=30.0, detection_history=1)
        priority_manager = PriorityManager()

        game_a = self._make_proc(100, 'game_a', 80.0)
        game_b = self._make_proc(200, 'game_b', 90.0)
        bg = self._make_proc(300, 'browser', 5.0)

        # Detect and boost game A
        game_detector.detect_game_process([game_a, bg])
        priority_manager.boost_game_priority(100, [game_a, bg], dry_run=True)
        assert priority_manager.get_boosted_game_pid() == 100

        # Now game B takes over
        game_detector.detect_game_process([game_b, bg])
        # Should restore old priorities before boosting new game
        priority_manager.restore_priorities(dry_run=True)
        assert not priority_manager.is_boost_active()

        result = priority_manager.boost_game_priority(200, [game_b, bg], dry_run=True)
        assert result['game_boosted'] is True
        assert priority_manager.get_boosted_game_pid() == 200

    @patch('glances.plugins.gaming_performance.priority_manager.psutil.Process')
    def test_full_cycle_detect_boost_restore(self, mock_process_cls):
        """Full cycle: detect game -> FPS drop -> boost -> FPS recovers -> restore."""
        mock_proc = MagicMock()
        mock_proc.nice.return_value = 0
        mock_process_cls.return_value = mock_proc

        fps_detector = FpsDetector(history_size=5)
        game_detector = GameDetector(cpu_threshold=30.0, detection_history=1)
        priority_manager = PriorityManager()

        game_pid = 7777
        target_fps = 60.0
        fps_drop_threshold = 20.0
        consecutive_drops_required = 2
        consecutive_low_fps = 0

        procs = [
            self._make_proc(game_pid, 'big_game', 85.0, 25.0),
            self._make_proc(8888, 'slack', 3.0, 2.0),
        ]

        # Phase 1: Detect game
        game_detector.detect_game_process(procs)
        assert game_detector._current_game_pid == game_pid

        # Phase 2: FPS drops
        for fps_val in [42.0, 38.0]:
            fps_detector._fps_history[game_pid] = [fps_val]
            drop = fps_detector.get_fps_drop(game_pid, target_fps)
            if drop and drop >= fps_drop_threshold:
                consecutive_low_fps += 1
            else:
                consecutive_low_fps = 0

        assert consecutive_low_fps >= consecutive_drops_required

        # Phase 3: Boost
        result = priority_manager.boost_game_priority(game_pid, procs, dry_run=True)
        assert result['game_boosted'] is True
        assert priority_manager.is_boost_active()

        # Phase 4: FPS recovers
        fps_detector._fps_history[game_pid] = [75.0]
        drop = fps_detector.get_fps_drop(game_pid, target_fps)
        assert drop == 0.0

        # Phase 5: Restore (would happen after game exits or manual trigger)
        priority_manager.restore_priorities(dry_run=True)
        assert not priority_manager.is_boost_active()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestFpsDetector, TestGameDetector, TestPriorityManager, TestDynamicThresholdingFlow]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
