"""D65 disk incident on <HOST-1> (descriptive; not part of the registered analysis).

Lists the episodes of VM2's block-03-B that were running while one sandbox's
write filled the disk (watchdog readings: 14,462 MB free at 19:03:36 UTC,
5,199 at 19:03:46, 25 at 19:03:56, the watchdog stop at 19:03:57, 14,480 at
19:04:07), and how each was recorded. Running = the sample's started.jsonl
init time is not after the window end and its end time is not before the
window start.

Usage: python describe_d65_incident.py --logroot logs/d65 --analysis logs/d65/analysis_d65.json --out logs/d65/incident_vm2.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

from log_paths import resolve_log_path


def t(h: int, m: int, s: int) -> float:
    return dt.datetime(2026, 10, 4, h, m, s, tzinfo=dt.timezone.utc).timestamp()


def utc(x: float | None) -> str | None:
    return None if x is None else dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", default=None, help="Root directory for d65 logs (defaults to AGENT_RISK_LOGS/d65)")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logroot = resolve_log_path(a.logroot, "d65")
    analysis = resolve_log_path(a.analysis) if a.analysis else os.path.join(logroot, "analysis_d65.json")
    att = os.path.join(logroot, "vm2", "block-03-B", "attempt-1")
    ev: dict = {}
    for line in open(os.path.join(att, "started.jsonl")):
        e = json.loads(line)
        ev.setdefault(e["sample_id"], {})[e["event"]] = e["time"]
    recs = {json.loads(l)["sample_id"]: json.loads(l) for l in open(os.path.join(att, "episodes.jsonl"))}
    views = {e["episode_id"]: e for e in json.load(open(analysis))["episodes"]}
    windows = {"readings_bracket_19:03:36_19:04:07": (t(19, 3, 36), t(19, 4, 7)),
               "low_reading_to_recovery_19:03:46_19:04:07": (t(19, 3, 46), t(19, 4, 7)),
               "instant_19:03:56_25MB_free": (t(19, 3, 56), t(19, 3, 56))}
    out: dict = {"windows": {}}
    for name, (w0, w1) in windows.items():
        run = sorted(s for s, d in ev.items() if d["init"] <= w1 and d.get("end", 1e18) >= w0)
        out["windows"][name] = {"n_running": len(run), "sample_ids": run}
    bracket = out["windows"]["readings_bracket_19:03:36_19:04:07"]["sample_ids"]
    rows = []
    for sid in bracket:
        r = recs.get(sid, {})
        eid = sid.rsplit("-t", 1)[0]
        steps_txt = json.dumps(r.get("steps") or [])
        failed = [c.get("step") for c in (r.get("ledger_calls") or []) if c.get("state_check") == "failed"]
        rows.append({
            "episode_id": eid, "sample_id": sid, "init": utc(ev[sid]["init"]), "end": utc(ev[sid].get("end")),
            "scored": bool(r.get("scored")), "error": (r.get("error") or "")[:200] or None,
            "check_errors": r.get("check_errors"), "failed_check_steps": failed,
            "enospc_in_tool_output": "No space left on device" in steps_txt,
            "dd_from_dev_zero_in_calls": ("/dev/zero" in steps_txt),
            "analysis_end": (views.get(eid) or {}).get("end"), "analysis_missing": (views.get(eid) or {}).get("missing"),
            "harm_recorded": (r.get("loss") or {}).get("harm"),
        })
    out["episodes"] = rows
    out["counts"] = {
        "running_in_bracket": len(rows),
        "unscored_sample_errors": sum(1 for x in rows if not x["scored"]),
        "scored_with_failed_state_check": sum(1 for x in rows if x["scored"] and x["check_errors"]),
        "scored_with_enospc_tool_output": sum(1 for x in rows if x["scored"] and x["enospc_in_tool_output"]),
        "scored_no_recorded_effect": sum(1 for x in rows if x["scored"] and not x["check_errors"] and not x["enospc_in_tool_output"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
