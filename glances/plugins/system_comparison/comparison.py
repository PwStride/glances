#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""Snapshot diffing and display-row formatting.

Given two SystemSnapshot values, produce a flat list of pre-formatted
comparison rows ready for the curses panel, the REST API, or anything
else that wants to show "before → after → delta".

This module is pure: no psutil, no filesystem, no clock. It consumes the
schema from snapshot.py but snapshot.py does not import from here, so
the on-disk format stays stable while labels, column widths, ordering,
and delta semantics are free to change release-to-release.
"""

from dataclasses import dataclass

from glances.plugins.system_comparison.snapshot import SystemSnapshot


@dataclass
class ComparisonRow:
    """One line of the comparison table, pre-formatted for display.

    Formatting happens here rather than in msg_curse() so the same rows
    can feed the curses panel, the REST API, and tests without each
    consumer reimplementing byte-humanisation and delta-sign rules.
    """

    label: str
    saved: str
    current: str
    delta: str
    changed: bool
    is_config: bool


def compute_comparison(saved: SystemSnapshot, current: SystemSnapshot) -> list:
    """Build the ordered list of comparison rows.

    Configuration rows always precede performance rows regardless of how
    many mounts or optional metrics are present, because the whole point
    of the panel is to show "you changed X (config) and got Y (perf)".
    The config/perf split is decided *here*, not in the snapshot schema —
    a field can move between sections without touching saved baselines.
    """
    rows = []

    # --- Configuration ---
    _str_row(rows, 'Host', saved.hostname, current.hostname, True)
    _str_row(rows, 'OS', saved.os_name, current.os_name, True)
    _str_row(rows, 'Kernel', saved.os_version, current.os_version, True)
    _int_row(rows, 'Cores P', saved.cpu_phys_cores, current.cpu_phys_cores, True)
    _int_row(rows, 'Cores L', saved.cpu_log_cores, current.cpu_log_cores, True)
    _byte_row(rows, 'RAM', saved.mem_total, current.mem_total, True)
    _byte_row(rows, 'Swap', saved.swap_total, current.swap_total, True)

    # Union of mount points so added/removed filesystems are visible.
    # Label truncation keeps deep mount paths from blowing the column.
    all_mnts = sorted(set(saved.fs_mounts) | set(current.fs_mounts))
    for mnt in all_mnts:
        _byte_row(rows, mnt[:8], saved.fs_mounts.get(mnt), current.fs_mounts.get(mnt), True)

    # --- Performance ---
    _pct_row(rows, 'CPU%', saved.cpu_percent, current.cpu_percent, False)
    _pct_row(rows, ' user', saved.cpu_user, current.cpu_user, False)
    _pct_row(rows, ' sys', saved.cpu_system, current.cpu_system, False)
    _pct_row(rows, ' iowait', saved.cpu_iowait, current.cpu_iowait, False)
    _pct_row(rows, 'MEM%', saved.mem_percent, current.mem_percent, False)
    _pct_row(rows, 'SWAP%', saved.swap_percent, current.swap_percent, False)
    _float_row(rows, 'Load1', saved.load_1, current.load_1, False)
    _float_row(rows, 'Load5', saved.load_5, current.load_5, False)
    _float_row(rows, 'Load15', saved.load_15, current.load_15, False)
    _int_row(rows, 'Procs', saved.proc_total, current.proc_total, False)
    _int_row(rows, ' run', saved.proc_running, current.proc_running, False)

    return rows


# ---------------------------------------------------------------------------
# Row builders
#
# Each helper handles None on either side uniformly: '-' in the value column,
# no delta, and changed=True iff exactly one side is None (something appeared
# or disappeared). When both sides are None the metric simply isn't available
# on this platform, which is not a "change".
# ---------------------------------------------------------------------------

def _str_row(rows, label, s, c, is_config):
    if s is None and c is None:
        rows.append(ComparisonRow(label, '-', '-', '=', False, is_config))
        return
    if s is None or c is None:
        rows.append(ComparisonRow(label, s or '-', c or '-', '->', True, is_config))
        return
    # Long strings (kernel versions, hostnames) are truncated for the
    # narrow sidebar; the full value is still in the JSON export.
    s_disp, c_disp = s[:7], c[:7]
    if s == c:
        rows.append(ComparisonRow(label, s_disp, c_disp, '=', False, is_config))
    else:
        rows.append(ComparisonRow(label, s_disp, c_disp, '->', True, is_config))


def _int_row(rows, label, s, c, is_config):
    if s is None and c is None:
        rows.append(ComparisonRow(label, '-', '-', '=', False, is_config))
        return
    if s is None:
        rows.append(ComparisonRow(label, '-', str(c), '+', True, is_config))
        return
    if c is None:
        rows.append(ComparisonRow(label, str(s), '-', '-', True, is_config))
        return
    diff = c - s
    if diff == 0:
        rows.append(ComparisonRow(label, str(s), str(c), '=', False, is_config))
    else:
        rows.append(ComparisonRow(label, str(s), str(c), f'{diff:+d}', True, is_config))


def _pct_row(rows, label, s, c, is_config):
    if s is None and c is None:
        rows.append(ComparisonRow(label, '-', '-', '=', False, is_config))
        return
    if s is None:
        rows.append(ComparisonRow(label, '-', f'{c:.1f}', '+', True, is_config))
        return
    if c is None:
        rows.append(ComparisonRow(label, f'{s:.1f}', '-', '-', True, is_config))
        return
    diff = round(c - s, 1)
    s_disp, c_disp = f'{s:.1f}', f'{c:.1f}'
    if diff == 0:
        rows.append(ComparisonRow(label, s_disp, c_disp, '=', False, is_config))
    else:
        rows.append(ComparisonRow(label, s_disp, c_disp, f'{diff:+.1f}', True, is_config))


def _float_row(rows, label, s, c, is_config):
    if s is None and c is None:
        rows.append(ComparisonRow(label, '-', '-', '=', False, is_config))
        return
    if s is None:
        rows.append(ComparisonRow(label, '-', f'{c:.2f}', '+', True, is_config))
        return
    if c is None:
        rows.append(ComparisonRow(label, f'{s:.2f}', '-', '-', True, is_config))
        return
    diff = round(c - s, 2)
    s_disp, c_disp = f'{s:.2f}', f'{c:.2f}'
    if diff == 0:
        rows.append(ComparisonRow(label, s_disp, c_disp, '=', False, is_config))
    else:
        rows.append(ComparisonRow(label, s_disp, c_disp, f'{diff:+.2f}', True, is_config))


def _byte_row(rows, label, s, c, is_config):
    if s is None and c is None:
        rows.append(ComparisonRow(label, '-', '-', '=', False, is_config))
        return
    if s is None:
        rows.append(ComparisonRow(label, '-', _fmt_bytes(c), '+', True, is_config))
        return
    if c is None:
        rows.append(ComparisonRow(label, _fmt_bytes(s), '-', '-', True, is_config))
        return
    diff = c - s
    s_disp, c_disp = _fmt_bytes(s), _fmt_bytes(c)
    if diff == 0:
        rows.append(ComparisonRow(label, s_disp, c_disp, '=', False, is_config))
    else:
        sign = '+' if diff > 0 else '-'
        rows.append(ComparisonRow(label, s_disp, c_disp, f'{sign}{_fmt_bytes(abs(diff))}', True, is_config))


def _fmt_bytes(n: int) -> str:
    """Human-readable bytes, trimmed to fit a 7-char column.

    Integer format above 10 in each unit ("16G") and one decimal below
    ("1.5G") keeps precision where it matters while staying compact.
    """
    for unit in ('B', 'K', 'M', 'G', 'T', 'P'):
        if abs(n) < 1024:
            if abs(n) >= 10 or n == int(n):
                return f'{int(n)}{unit}'
            return f'{n:.1f}{unit}'
        n /= 1024.0
    return f'{int(n)}E'
