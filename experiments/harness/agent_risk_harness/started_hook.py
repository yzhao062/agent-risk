"""Record which samples have begun, for the D58 resume rule.

Register PLAN-RECONCILE-2026-10-04.md, Section 0.4: an interrupted block
resumes by starting only manifest episodes that have not yet begun, and a
started episode is never replaced by another execution. Inspect's log holds
only finished samples, so this hook appends one JSON line per sample init,
start and end to the file named by $ARH_STARTED_LOG. It is enabled only when
that variable is set (run_pilot.py sets it for --manifest runs), reads nothing
from the sample, and changes neither the task nor its solvers.
"""

from __future__ import annotations

import json
import os
import time

from inspect_ai.hooks import Hooks, SampleEnd, SampleInit, SampleStart, hooks

STARTED_LOG_ENV = "ARH_STARTED_LOG"


def _write(event: str, sample_id) -> None:
    path = os.environ.get(STARTED_LOG_ENV)
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps({"event": event, "sample_id": str(sample_id), "time": time.time()}) + "\n")
        f.flush()
        os.fsync(f.fileno())


@hooks(name="arh_started_log", description="Append each sample init, start and end to $ARH_STARTED_LOG")
class StartedLog(Hooks):
    def enabled(self) -> bool:
        return bool(os.environ.get(STARTED_LOG_ENV))

    async def on_sample_init(self, data: SampleInit) -> None:
        _write("init", data.summary.id)

    async def on_sample_start(self, data: SampleStart) -> None:
        _write("start", data.summary.id)

    async def on_sample_end(self, data: SampleEnd) -> None:
        _write("end", data.sample.id)
