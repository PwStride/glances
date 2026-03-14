#!/usr/bin/env python3
"""
Tests for the system_comparison plugin.

Organised to mirror the module split:
  - snapshot.py   -> TestSnapshotPersistence, TestCapture
  - comparison.py -> TestComparison, TestFmtBytes

The separation is enforced, not incidental: snapshot tests never call
compute_comparison(), comparison tests never touch psutil or disk.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, '.')

from glances.plugins.system_comparison.comparison import (
    ComparisonRow,
    _fmt_bytes,
    compute_comparison,
)
from glances.plugins.system_comparison.snapshot import (
    SystemSnapshot,
    capture_current_state,
    load_snapshot,
    save_snapshot,
)


def _make_snapshot(**overrides) -> SystemSnapshot:
    """Build a snapshot with every field populated, override via kwargs.

    Defaults are picked to be distinctive so that if a field accidentally
    bleeds into another, the test failure message points at the culprit.
    """
    base = dict(
        timestamp='2026-01-01T00:00:00+00:00',
        hostname='testhost',
        os_name='Linux',
        os_version='6.6.0',
        platform_bits='64bit',
        cpu_phys_cores=4,
        cpu_log_cores=8,
        mem_total=16 * 1024**3,
        swap_total=4 * 1024**3,
        cpu_percent=50.0,
        cpu_user=30.0,
        cpu_system=15.0,
        cpu_iowait=5.0,
        mem_percent=60.0,
        mem_used=10 * 1024**3,
        swap_percent=10.0,
        load_1=1.5,
        load_5=1.2,
        load_15=1.0,
        proc_total=200,
        proc_running=3,
        fs_mounts={'/': 100 * 1024**3},
    )
    base.update(overrides)
    return SystemSnapshot(**base)


def _row(rows, label):
    """Pull one row by label or fail loudly — nicer than a StopIteration."""
    for r in rows:
        if r.label == label:
            return r
    raise AssertionError(f'no row labelled {label!r} in {[r.label for r in rows]}')


# ===========================================================================
# snapshot.py — capture & persistence only
# ===========================================================================

class TestSnapshotPersistence(unittest.TestCase):

    def test_save_load_roundtrip_preserves_all_fields(self):
        snap = _make_snapshot()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            self.assertTrue(save_snapshot(snap, path))
            loaded = load_snapshot(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.timestamp, snap.timestamp)
            self.assertEqual(loaded.hostname, snap.hostname)
            self.assertEqual(loaded.os_version, snap.os_version)
            self.assertEqual(loaded.cpu_phys_cores, snap.cpu_phys_cores)
            self.assertEqual(loaded.mem_total, snap.mem_total)
            self.assertEqual(loaded.cpu_percent, snap.cpu_percent)
            self.assertEqual(loaded.load_1, snap.load_1)
            self.assertEqual(loaded.fs_mounts, snap.fs_mounts)
        finally:
            os.unlink(path)

    def test_load_missing_file_returns_none(self):
        self.assertIsNone(load_snapshot('/nonexistent/path/snap.json'))

    def test_load_corrupt_json_returns_none_not_crash(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            f.write('{not valid json')
            path = f.name
        try:
            self.assertIsNone(load_snapshot(path))
        finally:
            os.unlink(path)

    def test_load_ignores_unknown_fields(self):
        """Forward compat — snapshot written by a newer Glances still loads."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='wb') as f:
            path = f.name
        try:
            # Save a valid snapshot, then inject an extra key into the JSON.
            snap = _make_snapshot()
            save_snapshot(snap, path)
            with open(path, 'rb') as f:
                raw = f.read().decode()
            injected = raw[:-1] + ', "future_field": 42}'
            with open(path, 'w') as f:
                f.write(injected)

            loaded = load_snapshot(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.hostname, 'testhost')
        finally:
            os.unlink(path)

    def test_save_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'nested', 'deeper', 'snap.json')
            self.assertTrue(save_snapshot(_make_snapshot(), path))
            self.assertTrue(os.path.exists(path))

    def test_save_overwrites_existing_file(self):
        """Re-saving must replace, not append — baseline is singular."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_snapshot(_make_snapshot(hostname='first'), path)
            save_snapshot(_make_snapshot(hostname='second'), path)
            loaded = load_snapshot(path)
            self.assertEqual(loaded.hostname, 'second')
        finally:
            os.unlink(path)


class TestCapture(unittest.TestCase):

    def test_capture_returns_populated_snapshot(self):
        """Smoke test against the live machine — mocking psutil here would
        test the mock, not the code. Bounds are loose enough to pass on any
        real system without asserting anything about the diff layer."""
        snap = capture_current_state()
        self.assertIsInstance(snap, SystemSnapshot)
        self.assertTrue(snap.hostname)
        self.assertTrue(snap.os_name)
        self.assertGreater(snap.mem_total, 0)
        self.assertGreaterEqual(snap.cpu_percent, 0.0)
        self.assertLessEqual(snap.cpu_percent, 100.0)
        self.assertGreater(snap.proc_total, 0)
        self.assertGreaterEqual(snap.proc_running, 0)
        # Timestamp should parse as ISO-8601
        from datetime import datetime
        datetime.fromisoformat(snap.timestamp)


# ===========================================================================
# comparison.py — diff & formatting only (no psutil, no disk)
# ===========================================================================

class TestComparison(unittest.TestCase):

    # -- Identity --

    def test_identical_snapshots_show_no_changes(self):
        snap = _make_snapshot()
        rows = compute_comparison(snap, snap)
        self.assertTrue(all(not r.changed for r in rows))
        for r in rows:
            self.assertEqual(r.delta, '=')

    # -- Config changes --

    def test_ram_doubled_flagged_as_config_change(self):
        saved = _make_snapshot(mem_total=16 * 1024**3)
        current = _make_snapshot(mem_total=32 * 1024**3)
        ram = _row(compute_comparison(saved, current), 'RAM')
        self.assertTrue(ram.changed)
        self.assertTrue(ram.is_config)
        self.assertTrue(ram.delta.startswith('+'))
        self.assertIn('16G', ram.delta)

    def test_swap_removed_shows_negative_byte_delta(self):
        saved = _make_snapshot(swap_total=4 * 1024**3)
        current = _make_snapshot(swap_total=0)
        swap = _row(compute_comparison(saved, current), 'Swap')
        self.assertTrue(swap.changed)
        self.assertTrue(swap.delta.startswith('-'))

    def test_kernel_upgrade_shows_arrow_delta(self):
        saved = _make_snapshot(os_version='6.6.0')
        current = _make_snapshot(os_version='6.8.0')
        kernel = _row(compute_comparison(saved, current), 'Kernel')
        self.assertTrue(kernel.changed)
        self.assertEqual(kernel.delta, '->')

    def test_core_count_change_has_signed_integer_delta(self):
        saved = _make_snapshot(cpu_log_cores=8)
        current = _make_snapshot(cpu_log_cores=16)
        cores = _row(compute_comparison(saved, current), 'Cores L')
        self.assertEqual(cores.delta, '+8')

    # -- Perf changes --

    def test_cpu_percent_decrease_has_negative_delta(self):
        saved = _make_snapshot(cpu_percent=80.0)
        current = _make_snapshot(cpu_percent=30.0)
        cpu = _row(compute_comparison(saved, current), 'CPU%')
        self.assertTrue(cpu.changed)
        self.assertFalse(cpu.is_config)
        self.assertEqual(cpu.delta, '-50.0')

    def test_load_average_increase_has_positive_delta(self):
        saved = _make_snapshot(load_1=0.5)
        current = _make_snapshot(load_1=2.75)
        load = _row(compute_comparison(saved, current), 'Load1')
        self.assertEqual(load.delta, '+2.25')

    def test_process_count_delta_is_integer(self):
        saved = _make_snapshot(proc_total=200)
        current = _make_snapshot(proc_total=186)
        procs = _row(compute_comparison(saved, current), 'Procs')
        self.assertEqual(procs.delta, '-14')

    # -- Mount-point set changes --

    def test_new_mount_point_shows_dash_for_saved(self):
        saved = _make_snapshot(fs_mounts={'/': 100 * 1024**3})
        current = _make_snapshot(fs_mounts={'/': 100 * 1024**3, '/data': 500 * 1024**3})
        rows = compute_comparison(saved, current)
        data = next(r for r in rows if r.label.startswith('/data'))
        self.assertEqual(data.saved, '-')
        self.assertTrue(data.changed)
        self.assertTrue(data.is_config)

    def test_removed_mount_point_shows_dash_for_current(self):
        saved = _make_snapshot(fs_mounts={'/': 100 * 1024**3, '/old': 50 * 1024**3})
        current = _make_snapshot(fs_mounts={'/': 100 * 1024**3})
        rows = compute_comparison(saved, current)
        old = next(r for r in rows if r.label.startswith('/old'))
        self.assertEqual(old.current, '-')
        self.assertTrue(old.changed)

    def test_mount_resized_shows_byte_delta(self):
        saved = _make_snapshot(fs_mounts={'/': 100 * 1024**3})
        current = _make_snapshot(fs_mounts={'/': 200 * 1024**3})
        root = _row(compute_comparison(saved, current), '/')
        self.assertTrue(root.changed)
        self.assertTrue(root.delta.startswith('+'))

    # -- None handling --

    def test_none_load_on_both_sides_is_not_a_change(self):
        """Windows has no getloadavg — both sides None must not flag changed."""
        saved = _make_snapshot(load_1=None, load_5=None, load_15=None)
        current = _make_snapshot(load_1=None, load_5=None, load_15=None)
        rows = compute_comparison(saved, current)
        for label in ('Load1', 'Load5', 'Load15'):
            r = _row(rows, label)
            self.assertEqual(r.saved, '-')
            self.assertEqual(r.current, '-')
            self.assertFalse(r.changed)

    def test_iowait_appearing_is_a_change(self):
        """Saved on a platform without iowait, current on one with it."""
        saved = _make_snapshot(cpu_iowait=None)
        current = _make_snapshot(cpu_iowait=3.2)
        iow = _row(compute_comparison(saved, current), ' iowait')
        self.assertEqual(iow.saved, '-')
        self.assertEqual(iow.current, '3.2')
        self.assertTrue(iow.changed)

    def test_phys_cores_none_handled(self):
        """psutil.cpu_count(logical=False) can return None on some platforms."""
        saved = _make_snapshot(cpu_phys_cores=None)
        current = _make_snapshot(cpu_phys_cores=None)
        cores = _row(compute_comparison(saved, current), 'Cores P')
        self.assertFalse(cores.changed)

    # -- Ordering --

    def test_config_rows_strictly_precede_perf_rows(self):
        rows = compute_comparison(_make_snapshot(), _make_snapshot())
        first_perf = next(i for i, r in enumerate(rows) if not r.is_config)
        self.assertTrue(all(r.is_config for r in rows[:first_perf]))
        self.assertTrue(all(not r.is_config for r in rows[first_perf:]))

    def test_comparison_returns_comparison_row_instances(self):
        rows = compute_comparison(_make_snapshot(), _make_snapshot())
        self.assertTrue(len(rows) > 0)
        self.assertTrue(all(isinstance(r, ComparisonRow) for r in rows))


class TestFmtBytes(unittest.TestCase):

    def test_whole_gigabytes(self):
        self.assertEqual(_fmt_bytes(16 * 1024**3), '16G')

    def test_fractional_gigabytes(self):
        self.assertEqual(_fmt_bytes(int(1.5 * 1024**3)), '1.5G')

    def test_megabytes(self):
        self.assertEqual(_fmt_bytes(512 * 1024**2), '512M')

    def test_zero(self):
        self.assertEqual(_fmt_bytes(0), '0B')

    def test_fits_in_seven_chars(self):
        """Column width constraint — anything reasonable must fit."""
        for n in (0, 1, 1023, 1024, 1536, 16 * 1024**3, 500 * 1024**4):
            self.assertLessEqual(len(_fmt_bytes(n)), 7, f'{n} -> {_fmt_bytes(n)!r}')


# ===========================================================================
# Layering
# ===========================================================================

class TestModuleBoundaries(unittest.TestCase):
    """Guard the one-way dependency: comparison may import snapshot, but
    never the reverse. A circular import here would mean changing display
    formatting forces users to re-save baselines — exactly the coupling
    the split exists to prevent.

    Checked via the AST, not raw text, so docstrings that *describe* the
    boundary ("this module has no psutil") don't trip their own alarm.
    """

    @staticmethod
    def _imports_of(module):
        import ast
        tree = ast.parse(open(module.__file__).read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_snapshot_does_not_import_comparison(self):
        import glances.plugins.system_comparison.snapshot as snap_mod
        imports = self._imports_of(snap_mod)
        self.assertFalse(
            any('comparison' in name for name in imports),
            f'snapshot.py imports: {sorted(imports)}',
        )

    def test_comparison_has_no_system_probe_imports(self):
        """comparison.py must stay pure — psutil/platform belong in snapshot.py."""
        import glances.plugins.system_comparison.comparison as cmp_mod
        imports = self._imports_of(cmp_mod)
        self.assertNotIn('psutil', imports)
        self.assertNotIn('platform', imports)
        self.assertNotIn('os', imports)


if __name__ == '__main__':
    unittest.main()
