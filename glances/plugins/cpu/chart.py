#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2024 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""CPU Chart display helper.

Contains the CpuChart class, which generates the time-series chart display
lines for the CpuPlugin.  It is intentionally a plain helper — not a plugin —
so that it can live inside the existing cpu plugin folder without requiring a
separate plugin registration or directory.
"""

from glances.outputs.glances_bars import Bar
from glances.outputs.glances_sparklines import Sparkline


class CpuChart:
    """Renders the CPU chart section of the CPU plugin display.

    Usage (inside CpuPlugin.msg_curse):

        chart = CpuChart(plugin)
        ret.extend(chart.msg_curse(max_width))
    """

    def __init__(self, plugin):
        """Init with a reference to the owning CpuPlugin instance."""
        self._p = plugin

    def msg_curse(self, max_width):
        """Return the list of display items for the chart section.

        :param max_width: Maximum column width available (int).
        :returns: list of curse display dicts (same format as
                  GlancesPluginModel.curse_add_line).
        """
        ret = []
        p = self._p

        if not p.stats:
            return ret

        # ── Title + time-series chart ─────────────────────────────────────
        title = 'CPU CHART'
        msg = '{:{width}}'.format(title, width=len(title))
        ret.append(p.curse_add_line(msg, 'TITLE'))

        bar_size = max(max_width - len(title), 10)
        if p.args.sparkline and p.history_enable() and not p.args.client:
            chart = Sparkline(bar_size)
            history = p.get_raw_history(item='total', nb=chart.size)
            chart.percents = [i[1] for i in history]
            # Pad left so the most recent value is always right-aligned
            chart.percents += [None] * (chart.size - len(chart.percents))
        else:
            bar_char = p.get_conf_value('chart_bar_char', default=['|'])[0]
            chart = Bar(bar_size, bar_char=bar_char)
            chart.percent = p.stats.get('total', 0)

        ret.append(p.curse_add_line('[', decoration='BOLD'))
        ret.append(p.curse_add_line(
            chart.get(),
            p.get_views(key='total', option='decoration'),
        ))
        ret.append(p.curse_add_line(']', decoration='BOLD'))

        # ── Speed / core count line ───────────────────────────────────────
        ret.append(p.curse_new_line())

        hz_cur = p.stats.get('cpu_hz_current')
        hz_max = p.stats.get('cpu_hz')
        if hz_cur is not None and hz_max is not None:
            msg_freq = 'Speed:{:.2f}/{:.2f}GHz'.format(
                _hz_to_ghz(hz_cur), _hz_to_ghz(hz_max)
            )
        elif hz_cur is not None:
            msg_freq = 'Speed:{:.2f}GHz'.format(_hz_to_ghz(hz_cur))
        else:
            msg_freq = 'Speed:N/A'
        ret.append(p.curse_add_line(f'  {msg_freq}'))

        phys = p.stats.get('cpu_phys_core')
        log_ = p.stats.get('cpu_log_core')
        if phys is not None and log_ is not None:
            msg_cores = f'  Cores:{phys}p/{log_}l'
        elif log_ is not None:
            msg_cores = f'  Cores:{log_}l'
        else:
            msg_cores = ''
        ret.append(p.curse_add_line(msg_cores))

        # ── Process / thread summary line ─────────────────────────────────
        ret.append(p.curse_new_line())
        procs = p.stats.get('processes')
        threads = p.stats.get('threads')
        running = p.stats.get('running')
        sleeping = p.stats.get('sleeping')

        if procs is not None:
            msg_tasks = f'  Tasks:{procs}'
            if threads is not None:
                msg_tasks += f'  thr:{threads}'
            if running is not None:
                msg_tasks += f'  run:{running}'
            if sleeping is not None:
                msg_tasks += f'  slp:{sleeping}'
            ret.append(p.curse_add_line(msg_tasks))

        # ── Per-core table ────────────────────────────────────────────────
        ret.append(p.curse_new_line())
        header = '  {:4}  {:>6}  {:>6}  {:>6}'.format('Core', 'Total', 'User', 'Sys')
        ret.append(p.curse_add_line(header, decoration='BOLD'))

        max_cpu_display = p.stats.get('chart_max_cpu_display', 8)
        percpu = p.stats.get('percpu', [])
        percpu_sorted = sorted(percpu, key=lambda c: c.get('cpu_number', 0))
        display_list = percpu_sorted[:max_cpu_display]

        for core in display_list:
            ret.append(p.curse_new_line())
            cpu_id = core.get('cpu_number', '?')
            total = core.get('total')
            user = core.get('user')
            system = core.get('system')

            label = f'CPU{cpu_id}' if isinstance(cpu_id, int) and cpu_id < 10 else f'{cpu_id}'
            ret.append(p.curse_add_line('  {:4}  '.format(label)))

            if total is not None:
                ret.append(p.curse_add_line(
                    '{:5.1f}%  '.format(total),
                    p.get_alert(total, header='total'),
                ))
            else:
                ret.append(p.curse_add_line('    ?%  '))

            if user is not None:
                ret.append(p.curse_add_line('{:5.1f}%  '.format(user)))
            else:
                ret.append(p.curse_add_line('    ?%  '))

            if system is not None:
                ret.append(p.curse_add_line('{:5.1f}%'.format(system)))
            else:
                ret.append(p.curse_add_line('    ?%'))

        # Overflow summary row
        extra = percpu_sorted[max_cpu_display:]
        if extra:
            ret.append(p.curse_new_line())
            avg_total = sum(c.get('total', 0) for c in extra) / len(extra)
            avg_user = sum(c.get('user', 0) for c in extra) / len(extra)
            avg_sys = sum(c.get('system', 0) for c in extra) / len(extra)
            msg_extra = '  CPU*  {:5.1f}%  {:5.1f}%  {:5.1f}%'.format(avg_total, avg_user, avg_sys)
            ret.append(p.curse_add_line(msg_extra, optional=True))

        return ret


# ── Module-level helpers ──────────────────────────────────────────────────────

def _hz_to_ghz(hz: float) -> float:
    return hz / 1_000_000_000.0


def mhz_to_hz(mhz: float) -> float:
    """Convert MHz to Hz (used by CpuPlugin.update_local)."""
    return mhz * 1_000_000.0
