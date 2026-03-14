#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""System State Comparison plugin.

Orchestrates the two sub-modules — snapshot (capture + persistence) and
comparison (diff + formatting) — and exposes their combined result
through the standard Glances plugin surface.

This file owns no domain logic of its own. It wires:
  - the 'y' hotkey flag to snapshot.save_snapshot()
  - each update tick to snapshot.capture_current_state()
  - the saved and current snapshots to comparison.compute_comparison()
  - the comparison rows to curses colour/column layout

Workflow from the operator's seat:
  1. Press 'y' to capture the current state as the baseline.
  2. The COMPARISON panel in the left sidebar continuously shows how
     live metrics differ from that baseline.
  3. Press 'y' again after making a config change to start a new
     comparison window.

The snapshot persists across Glances restarts so a comparison survives
the reboot that a kernel upgrade or hardware swap typically requires.
"""

from glances.plugins.plugin.model import GlancesPluginModel
from glances.plugins.system_comparison.comparison import compute_comparison
from glances.plugins.system_comparison.snapshot import (
    DEFAULT_SNAPSHOT_PATH,
    capture_current_state,
    load_snapshot,
    save_snapshot,
)

fields_description = {
    'snapshot_timestamp': {
        'description': 'ISO-8601 UTC timestamp of the saved baseline snapshot',
    },
    'snapshot_path': {
        'description': 'Filesystem path to the baseline JSON file',
    },
    'config_changes': {
        'description': 'Number of configuration fields that differ from the baseline',
        'unit': 'number',
    },
    'perf_deltas': {
        'description': 'Number of performance metrics that differ from the baseline',
        'unit': 'number',
    },
}


class System_comparisonPlugin(GlancesPluginModel):
    """Glances system-state comparison plugin."""

    def __init__(self, args=None, config=None):
        """Init the plugin."""
        super().__init__(
            args=args,
            config=config,
            stats_init_value={},
            fields_description=fields_description,
        )

        # We want to display the stat in the curse interface
        self.display_curse = True

        # Resolve snapshot path from config with a sensible platform default.
        section = 'system_comparison'
        if config is not None and config.has_section(section):
            self.snapshot_path = config.get_value(section, 'snapshot_path', default=DEFAULT_SNAPSHOT_PATH)
        else:
            self.snapshot_path = DEFAULT_SNAPSHOT_PATH

        # Load any previously saved baseline once at startup. Re-reading the
        # file every tick would be wasteful and would also make it impossible
        # to tell "just saved" apart from "loaded from disk" for UI feedback.
        self._saved = load_snapshot(self.snapshot_path)

        # One-shot flag for the "Baseline saved" confirmation line; cleared
        # by msg_curse() after it has been rendered once.
        self._just_saved = False

    def update(self):
        """Refresh comparison stats.

        The save action rides on the same args-flag mechanism as graph
        generation: the curses 'y' hotkey flips args.save_snapshot to
        True, we perform the save here on the very next update tick,
        then clear the flag so the hotkey behaves as a one-shot trigger
        rather than a toggle.
        """
        stats = self.get_init_value()

        if self.args is not None and getattr(self.args, 'save_snapshot', False):
            fresh = capture_current_state()
            if save_snapshot(fresh, self.snapshot_path):
                self._saved = fresh
                self._just_saved = True
            # Reset regardless of success so a failed write doesn't retry
            # forever — the user will see "No baseline" and can press 'y'
            # again after fixing permissions.
            self.args.save_snapshot = False

        current = capture_current_state()

        stats['snapshot_path'] = self.snapshot_path

        if self._saved is None:
            stats['snapshot_timestamp'] = None
            stats['comparison'] = []
            stats['config_changes'] = 0
            stats['perf_deltas'] = 0
        else:
            rows = compute_comparison(self._saved, current)
            stats['snapshot_timestamp'] = self._saved.timestamp
            # Dataclass → dict so the REST API / JSON export serialises
            # cleanly without needing to know about ComparisonRow.
            stats['comparison'] = [r.__dict__ for r in rows]
            stats['config_changes'] = sum(1 for r in rows if r.is_config and r.changed)
            stats['perf_deltas'] = sum(1 for r in rows if not r.is_config and r.changed)

        self.stats = stats
        return self.stats

    def msg_curse(self, args=None, max_width=None):
        """Render the comparison panel for the left sidebar.

        Layout targets the 23–34 char left-sidebar width. Columns are
        fixed rather than proportional because perf numbers rarely exceed
        5 chars and a jittering layout is worse than a slightly cramped
        one when the terminal is narrow.
        """
        ret = []

        if not self.stats or self.is_disabled():
            return ret

        width = max_width or 34

        ret.append(self.curse_add_line('COMPARISON', 'TITLE'))
        ret.append(self.curse_new_line())

        if self._just_saved:
            ret.append(self.curse_add_line('Baseline saved', 'OK'))
            ret.append(self.curse_new_line())
            self._just_saved = False

        ts = self.stats.get('snapshot_timestamp')
        if not ts:
            ret.append(self.curse_add_line('No baseline saved'))
            ret.append(self.curse_new_line())
            ret.append(self.curse_add_line("Press 'y' to save"))
            ret.append(self.curse_new_line())
            return ret

        # ISO-8601 → "YYYY-MM-DD HH:MM". Seconds and tz offset are noise
        # for a baseline that's typically hours or days old.
        ts_short = ts.replace('T', ' ')[:16]
        ret.append(self.curse_add_line(f'Saved: {ts_short}'))
        ret.append(self.curse_new_line())

        lbl_w = 8
        col_w = 7
        delta_w = max(4, width - lbl_w - col_w * 2)

        last_section = None
        for row in self.stats['comparison']:
            section = 'CONFIG' if row['is_config'] else 'PERFORMANCE'
            if section != last_section:
                hdr = f'-- {section} '.ljust(width, '-')[:width]
                ret.append(self.curse_add_line(hdr))
                ret.append(self.curse_new_line())
                last_section = section

            # Colour policy:
            #   - Config change → OK (green). It's not good or bad, it's
            #     the thing the operator did on purpose; highlight it so
            #     it's easy to correlate with the perf rows below.
            #   - Perf increase → CAREFUL (yellow). Higher CPU/mem/load is
            #     usually a regression, though not always — yellow says
            #     "look here" without screaming "alert".
            #   - Perf decrease → OK (green). Lower is usually better.
            #   - Unchanged → DEFAULT. Fade into the background.
            if not row['changed']:
                deco = 'DEFAULT'
            elif row['is_config']:
                deco = 'OK'
            elif row['delta'].startswith('+'):
                deco = 'CAREFUL'
            else:
                deco = 'OK'

            line = (
                f"{row['label']:<{lbl_w}}"
                f"{row['saved']:>{col_w}}"
                f"{row['current']:>{col_w}}"
                f"{row['delta']:>{delta_w}}"
            )[:width]
            ret.append(self.curse_add_line(line, deco))
            ret.append(self.curse_new_line())

        return ret
