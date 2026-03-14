#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""System-state snapshot: capture and persistence.

Answers exactly one question — "what does the machine look like right
now, and how do I get that on/off disk?" — and nothing else. The schema
(SystemSnapshot) and the JSON it produces are the stable contract; how
two snapshots are *diffed* or *rendered* lives in comparison.py so that
presentation can evolve without forcing users to re-save baselines.

All system probing goes through psutil/platform directly (the same
primitives the cpu/mem/system plugins rely on) rather than reaching into
sibling plugins, keeping this module importable and testable on its own.
"""

import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import psutil

from glances.config import user_cache_dir
from glances.globals import json_dumps, json_loads
from glances.logger import logger

DEFAULT_SNAPSHOT_PATH = os.path.join(user_cache_dir()[0], 'snapshot.json')


@dataclass
class SystemSnapshot:
    """Point-in-time capture of system configuration + performance.

    The config/perf grouping below is semantic documentation only — this
    module does not interpret it. comparison.py owns the decision about
    which fields land in which display section; keeping that policy out
    of the schema means a field can be reclassified without a migration.

    Note on dataclass field order: non-default fields must precede any
    field with a default, so fs_mounts sits at the end even though it is
    logically part of the configuration group.
    """

    # --- Metadata ---
    timestamp: str

    # --- Configuration (static hardware/OS identity) ---
    hostname: str
    os_name: str
    os_version: str
    platform_bits: str
    cpu_phys_cores: Optional[int]
    cpu_log_cores: Optional[int]
    mem_total: int
    swap_total: int

    # --- Performance (dynamic, point-in-time) ---
    cpu_percent: float
    cpu_user: float
    cpu_system: float
    cpu_iowait: Optional[float]
    mem_percent: float
    mem_used: int
    swap_percent: float
    load_1: Optional[float]
    load_5: Optional[float]
    load_15: Optional[float]
    proc_total: int
    proc_running: int

    # --- Configuration (defaulted — must come last per dataclass rules) ---
    fs_mounts: dict = field(default_factory=dict)


def capture_current_state() -> SystemSnapshot:
    """Sample the live system and return a populated SystemSnapshot.

    interval=0.0 on cpu_times_percent gives a non-blocking delta since the
    last call — the same technique the cpu plugin uses so we never stall
    the update loop. The very first call in a process returns meaningless
    zeros; that is acceptable because the plugin calls this every tick and
    the user saves a baseline interactively, never on the first frame.
    """
    uname = platform.uname()
    cpu_times = psutil.cpu_times_percent(interval=0.0)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    try:
        load = os.getloadavg()
    except (AttributeError, OSError):
        # Windows has no getloadavg; some containers block it too.
        load = (None, None, None)

    fs = {}
    for part in psutil.disk_partitions(all=False):
        try:
            fs[part.mountpoint] = psutil.disk_usage(part.mountpoint).total
        except (PermissionError, OSError):
            # Removable media with no disc, network mounts mid-disconnect,
            # etc. Skip rather than abort the whole snapshot.
            continue

    procs = list(psutil.process_iter(['status']))
    running = sum(1 for p in procs if p.info.get('status') == psutil.STATUS_RUNNING)

    return SystemSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hostname=uname.node,
        os_name=uname.system,
        os_version=uname.release,
        platform_bits=platform.architecture()[0],
        cpu_phys_cores=psutil.cpu_count(logical=False),
        cpu_log_cores=psutil.cpu_count(logical=True),
        mem_total=vm.total,
        swap_total=sw.total,
        fs_mounts=fs,
        cpu_percent=round(100.0 - cpu_times.idle, 1),
        cpu_user=round(cpu_times.user, 1),
        cpu_system=round(cpu_times.system, 1),
        cpu_iowait=round(cpu_times.iowait, 1) if hasattr(cpu_times, 'iowait') else None,
        mem_percent=round(vm.percent, 1),
        mem_used=vm.used,
        swap_percent=round(sw.percent, 1),
        load_1=round(load[0], 2) if load[0] is not None else None,
        load_5=round(load[1], 2) if load[1] is not None else None,
        load_15=round(load[2], 2) if load[2] is not None else None,
        proc_total=len(procs),
        proc_running=running,
    )


def save_snapshot(snapshot: SystemSnapshot, path: str = DEFAULT_SNAPSHOT_PATH) -> bool:
    """Persist a snapshot as JSON, creating parent directories as needed.

    Returns False rather than raising on I/O failure because a failed save
    should surface as a missing baseline in the UI, not crash the monitor.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(json_dumps(asdict(snapshot)))
        logger.info(f"system_comparison: snapshot saved to {path}")
        return True
    except OSError as e:
        logger.error(f"system_comparison: failed to save snapshot: {e}")
        return False


def load_snapshot(path: str = DEFAULT_SNAPSHOT_PATH) -> Optional[SystemSnapshot]:
    """Load a snapshot from disk, or None if absent / unreadable.

    Unknown keys in the JSON are silently dropped so a snapshot written by
    a newer Glances with extra fields still loads in an older one. Missing
    keys (snapshot from an older Glances) will raise TypeError from the
    dataclass constructor and fall through to the warning path — the user
    simply re-saves a fresh baseline.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = json_loads(f.read())
        known = {f.name for f in SystemSnapshot.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return SystemSnapshot(**filtered)
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"system_comparison: failed to load snapshot ({e}); save a new baseline with 'y'")
        return None
